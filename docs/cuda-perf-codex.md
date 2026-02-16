# CUDA Performance Codex — Qwen3-TTS on Luminal

## Current State (post Wave B.0, 2026-02-16)

- **344.6ms/frame** on A100-80GB (decode 114ms + predictor 225ms + overhead 5ms)
- **Target**: ~28ms/frame (PyTorch reference via CUDA graph replay)
- **Gap**: 12.3x — entirely due to host-dispatch overhead (1677 ops/frame, each a separate cuLaunchKernel or cuGraphLaunch)
- **Branch**: qwen3-tts
- **Completed**: Wave 0, A1, A2 (see `docs/wave-a-results.md`) + B.0 spike (see `docs/wave-b0-results.md`)

## What's Been Done

| Wave | What | Result | Commits |
|------|------|--------|---------|
| 0 | cuBLASLt upstream + output_bytes() | Faster GEMMs, dtype-correct alloc | 767ce087 |
| A1 | Per-op sync removal | -27ms/frame | 767ce087 |
| A2 | Selective buffer zeroing | -55ms/frame (zero_buffers → 0ms) | 767ce087 |
| — | Cache compat fix | Both cuBLAS+cuBLASLt registered | 65ea3dae |
| B.0 | Stream capture feasibility spike | cuBLASLt capturable, cuGraphLaunch NOT capturable | c559aa28 |

Safety flags: `LUMINAL_SYNC_DEBUG=1` (restore sync), `LUMINAL_FORCE_ZERO_ALL=1` (restore zero-all)

## B.0 Key Finding: cuGraphLaunch Not Capturable

**Stream capture cannot nest `cuGraphLaunch`**. Attempting `cuGraphLaunch` during `cuStreamBeginCapture` returns `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`. This is a fundamental CUDA limitation — not a driver version or configuration issue.

**What IS capturable**: cuBLASLt matmul calls, custom kernel launches, memset/memcpy. Just not child graph launches.

**Implication**: The original Wave B.1 plan (stream-capture the entire `execute()` call) won't work because `execute()` dispatches CudaGraphOps which do `cuGraphLaunch` internally.

## What's Next

### Wave B.1: Explicit Graph Construction with Child Nodes (1-2 weeks)

**Strategy change**: Instead of stream capture, build the parent graph **explicitly** using `cuGraphCreate` + `cuGraphAddChildGraphNode` + mini stream captures for cuBLAS ops.

**How it works**:

```
For each runtime.execute() call:
  1. Create empty parent CUgraph
  2. For each HostOp in topo order:
     - CudaGraphOp → cuGraphAddChildGraphNode(parent, deps, child.cu_graph)
     - CuBlasLt    → mini stream capture of the matmul → cuGraphAddChildGraphNode
     - CuBlasSgemm → mini stream capture of the sgemm → cuGraphAddChildGraphNode
  3. Instantiate parent graph → CUgraphExec
  4. Launch with single cuGraphLaunch
  5. Cache (CUgraphExec, DynMapSignature) for subsequent frames
  6. On frame 2+: update buffer pointers via cuGraphExecChildGraphNodeSetParams,
     then replay
```

**Key CUDA APIs**:
- `cuGraphCreate` — create empty graph
- `cuGraphAddChildGraphNode` — add CudaGraphOp as child node (available in cudarc 0.18.2 sys)
- `cuStreamBeginCapture/EndCapture` — mini-capture for cuBLAS ops
- `cuGraphInstantiateWithFlags` — instantiate parent graph
- `cuGraphExecChildGraphNodeSetParams` — update child graph in instantiated exec (if buffer pointers change)

**Data flow dependencies**: The `execute_host_ops_loop` already topologically sorts ops. Each op's dependency chain defines the graph node edges.

**Access to child CUgraph**: `CudaGraphOpState.cuda_graph` holds a `CudaGraphHandle` with `cu_graph: CUgraph`. Need to expose this via a method on `CudaGraphOp` (currently behind `RefCell`).

**Implementation plan**:

1. **B.1a**: Add `fn cu_graph(&self) -> CUgraph` accessor to CudaGraphOp
2. **B.1b**: Add `cuGraphAddChildGraphNode` wrapper to `cuda_graph.rs`
3. **B.1c**: In `CudaRuntime`, implement `build_runtime_graph()`:
   - Iterate exec ops in topo order
   - For CudaGraphOp: ensure child graph is built, add as child node
   - For CuBlasLt/CuBlasSgemm: mini-capture + add as child node
   - Track data-flow edges as graph dependencies
4. **B.1d**: Wire into `execute()` with the existing `RuntimeReplayState` machine:
   - WarmupPending → normal execute (builds child graphs, primes cuBLAS)
   - CapturePending → `build_runtime_graph()` + instantiate
   - Ready → single `cuGraphLaunch` + pointer updates
5. **B.1e**: Correctness test: compare output between dispatch and replay paths
6. **B.1f**: Modal benchmark: measure frame time with replay enabled

**Known challenges**:
- CuBlasLt ops lazily initialize cuBLASLt handles and workspaces (OnceLock) — must be initialized during warmup
- Buffer pointer updates between frames: `set_data` changes input pointers, need to propagate to child graph nodes
- Zeroing: some buffers need zeroing before execution — can be memset nodes in the graph
- CudaGraphOp already manages its own graph rebuild logic — during parent graph construction, child graphs must be in their final state (not pending rebuild)

**Key files**:
- `crates/luminal_cuda/src/runtime.rs` — `execute()`, RuntimeReplayState, build_runtime_graph (new)
- `crates/luminal_cuda/src/kernel/to_host.rs` — CudaGraphOp, cu_graph() accessor (new)
- `crates/luminal_cuda/src/kernel/cuda_graph.rs` — cuGraphAddChildGraphNode wrapper (new)
- `crates/luminal_cuda/src/host/cublaslt/mod.rs` — cuBLASLt warmup
- `crates/luminal_cuda/src/host/cublas/mod.rs` — cuBLAS warmup

**What "done" looks like**:
- Frame 1: normal dispatch (warmup) — ~345ms
- Frame 2: explicit graph build + instantiate — ~345ms + build overhead
- Frame 3+: graph replay — target <80ms (conservative), aspirational <40ms
- Correctness: output waveform matches non-replay path bit-for-bit

### Wave C: Graph Size Investigation (conditional)

**Trigger**: Post-B.1 steady-state >56ms (2x PyTorch's 28ms).

If runtime graph replay gets us to <56ms, Wave C is unnecessary. If not:
- Dump op-type histogram for both runtimes
- Compare kernel count vs PyTorch (~50-100 kernels)
- Investigate predictor graph fusion opportunities (1182 ops is suspicious)
- Consider custom attention kernel, fused SnakeBeta BlockOp

### Wave D: TTFA Pipeline Work

**Prereq**: Per-frame speed is competitive (<56ms).

- Progressive chunk emission (1→2→4→8→16 frames, matching eiri-voice-stack)
- Stream first audio chunk before all frames complete
- Warmup/compilation overlap with prefill
- decode_speech OOM fix (lazy per-chunk compilation instead of all 16 upfront)

## Architecture Reference

### How execution works today (344ms/frame)
```
pipeline.rs: generate_frames loop
  ├── set_data (embed, pos, mask)           → 0.04ms
  ├── decode_rt.execute()                   → 114ms
  │   └── runtime.rs: for each of 495 HostOps:
  │       ├── CudaGraphOp.execute()         → cuGraphLaunch (child graph)
  │       └── CuBlasLt.execute()            → cublasLtMatmul
  ├── get_f32 (3 outputs)                   → 0.3ms
  ├── pred set_data + pred_rt.execute()     → 225ms
  │   └── runtime.rs: for each of 1182 HostOps:
  │       ├── CudaGraphOp.execute()
  │       └── CuBlasLt.execute()
  └── kv_scatter (CPU + partial GPU update) → 4.8ms
```

### HostOp types in execute() dispatch
```
CudaGraphOp    → wraps MegakernelOp subgraph → cuGraphLaunch (child graph)
                 Has internal CUgraph handle, builds/updates graph on demand
                 Buffer pointers updated via cuGraphExecKernelNodeSetParams

CuBlasLt       → cuBLASLt matmul (workspace OnceLock, handle OnceLock)
                 Capturable via stream capture (B.0 confirmed)

CuBlasSgemmV2  → cuBLAS sgemm (legacy, still registered for cache compat)
                 Likely capturable (same GPU-only pattern as cuBLASLt)
```

### How execution should work after Wave B.1 (~40-80ms/frame)
```
pipeline.rs: generate_frames loop
  ├── set_data (embed, pos, mask)           → 0.04ms
  ├── decode_rt.execute()                   → ~15-25ms
  │   └── runtime.rs: cuGraphLaunch(cached_parent_graph)  ← 1 call
  │       Parent graph contains:
  │       ├── child_node[0]: CudaGraphOp #0 (attention block)
  │       ├── child_node[1]: CuBlasLt matmul (captured)
  │       ├── child_node[2]: CudaGraphOp #1 (MLP block)
  │       └── ... (495 total, all as graph nodes)
  ├── get_f32 (3 outputs)                   → 0.3ms
  ├── pred set_data + pred_rt.execute()     → ~20-50ms
  │   └── runtime.rs: cuGraphLaunch(cached_parent_graph)  ← 1 call
  └── kv_scatter (CPU + partial GPU update) → 4.8ms
```

### PyTorch reference (28ms/frame)
```
eiri-voice-stack: inference loop
  ├── Talker forward (80 pre-captured CUDA graphs)  → 6.7ms
  ├── Predictor forward (CUDA graphs)                → 19.1ms
  └── CPU work (scatter, bookkeeping)                → 2.1ms
```

## Key Files

| File | Role |
|------|------|
| `crates/luminal_cuda/src/runtime.rs` | CudaRuntime — buffer mgmt, execute(), replay state machine |
| `crates/luminal_cuda/src/kernel/to_host.rs` | CudaGraphOp — child graph build/launch |
| `crates/luminal_cuda/src/host/cublaslt/mod.rs` | cuBLASLt host op |
| `crates/luminal_cuda/src/host/cublas/mod.rs` | Legacy cuBLAS host op |
| `crates/luminal_cuda/src/kernel/cuda_graph.rs` | CUDA graph wrappers, B.0 spike test |
| `crates/luminal_cuda/src/block/mod.rs` | MegakernelOp (KernelOp, used inside CudaGraphOp) |
| `crates/luminal_cuda/src/lib.rs` | CUDA_STREAM_CAPTURING flag |
| `examples/qwen3_tts/src/pipeline.rs` | TTS pipeline — prefill, generate_frames, decode_speech |
| `examples/qwen3_tts/src/backend.rs` | Backend abstraction — compile(), set_data, get_f32 |
| `scripts/modal_test_tts.py` | Modal A100 test harness |
| `docs/wave-a-results.md` | Wave A measurement data |
| `docs/wave-b0-results.md` | Wave B.0 capture feasibility results |

## Validation Checklist

For every wave:
- [ ] `cargo check -p luminal_cuda`
- [ ] `cargo test -p luminal_cuda --no-run`
- [ ] `cargo check -p qwen3_tts --features cuda`
- [ ] Modal run produces correct WAV output (192k samples, 8.00s at 24kHz)
- [ ] Profiling numbers recorded in `docs/`
- [ ] Fallback flags tested (`LUMINAL_SYNC_DEBUG=1`, `LUMINAL_FORCE_ZERO_ALL=1`)

For Wave B.1 specifically:
- [ ] B.0 spike test passes on A100 (DONE: c559aa28)
- [ ] Parent graph builds without error (child nodes + cuBLAS nodes)
- [ ] Output correctness: codes match between replay and dispatch paths
- [ ] dyn_map change triggers cache invalidation and re-build
- [ ] No memory leak from graph instantiation (check with `nvidia-smi` over 100 frames)
- [ ] Frame 2+ steady-state time measured and recorded

## Housekeeping

- [ ] Remove one-time cache clear from `scripts/modal_test_tts.py` (lines 129-133) — cache is already refreshed
- [ ] decode_speech OOM is pre-existing, not related to perf work — track separately
