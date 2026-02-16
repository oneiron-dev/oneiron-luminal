# CUDA Performance Codex — Qwen3-TTS on Luminal

## Current State (post Wave A, 2026-02-16)

- **344.6ms/frame** on A100-80GB (decode 114ms + predictor 225ms + overhead 5ms)
- **Target**: ~28ms/frame (PyTorch reference via CUDA graph replay)
- **Gap**: 12.3x — entirely due to host-dispatch overhead (1677 ops/frame, each a separate cuLaunchKernel or cuGraphLaunch)
- **Branch**: qwen3-tts
- **Completed**: Wave 0 (cuBLASLt), A1 (sync removal), A2 (selective zeroing) — see `docs/wave-a-results.md`

## What's Been Done

| Wave | What | Result | Commits |
|------|------|--------|---------|
| 0 | cuBLASLt upstream + output_bytes() | Faster GEMMs, dtype-correct alloc | 767ce087 |
| A1 | Per-op sync removal | -27ms/frame | 767ce087 |
| A2 | Selective buffer zeroing | -55ms/frame (zero_buffers → 0ms) | 767ce087 |
| — | Cache compat fix | Both cuBLAS+cuBLASLt registered | 65ea3dae |

Safety flags: `LUMINAL_SYNC_DEBUG=1` (restore sync), `LUMINAL_FORCE_ZERO_ALL=1` (restore zero-all)

## What's Next

### Wave B.0: Capture Feasibility Spike (2-4 days)

**Goal**: Determine if CUDA stream capture can wrap the entire `execute()` call — including both CudaGraphOp launches (child graph replay) and cuBLASLt matmuls — into a single parent CUDA graph.

**Test already exists** (ignored): `crates/luminal_cuda/src/kernel/cuda_graph.rs:491`
```
#[test]
#[ignore = "Wave B.0 feasibility spike; run manually on a CUDA host"]
fn wave_b0_stream_capture_with_graph_launch_and_cublaslt()
```

**What to validate**:
1. Can `cuStreamBeginCapture` capture a child `cuGraphLaunch` (CudaGraphOp's replay)?
2. Can it capture a cuBLASLt matmul in the same stream?
3. Can the resulting parent graph be instantiated and replayed?
4. Can buffer pointers be updated between replays (via `cudaGraphExecKernelNodeSetParams` or full re-instantiate)?

**Known risks**:
- cuBLAS/cuBLASLt may use internal workspace allocation during capture → capture failure
- Child graph launch during capture requires CUDA 12.0+ (A100 supports this)
- If capture fails: fallback is segmented capture (capture CudaGraphOp regions, dispatch cuBLASLt individually)

**Run on Modal**: `modal run scripts/modal_test_tts.py` won't run this test automatically. Need either:
- A Modal function that runs `cargo test ... wave_b0 -- --ignored --nocapture`, or
- Add a standalone binary/integration test

### Wave B.1: Runtime-Level Replay (1-2 weeks)

**Prereq**: B.0 proves capture is feasible.

**Design**: Add a replay cache to `CudaRuntime::execute()`:
1. First call with given `dyn_map` values → normal dispatch + stream capture → store graph
2. Subsequent calls with same `dyn_map` → replay cached graph
3. If `dyn_map` changes → invalidate cache, fall back to dispatch + re-capture
4. Buffer pointer updates between replays (inputs change every frame via `set_data`)

**Key files**:
- `crates/luminal_cuda/src/runtime.rs` — `execute()` method (~line 719)
- `crates/luminal_cuda/src/kernel/to_host.rs` — CudaGraphOp (already does child graph launch)
- `crates/luminal_cuda/src/host/cublaslt/mod.rs` — cuBLASLt execution

**What "done" looks like**:
- Frame 1: normal dispatch (captures graph) — ~345ms
- Frame 2+: graph replay — target <80ms (conservative), aspirational <40ms
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

### How execution should work after Wave B.1 (~40-80ms/frame)
```
pipeline.rs: generate_frames loop
  ├── set_data (embed, pos, mask)           → 0.04ms
  ├── decode_rt.execute()                   → ~15-25ms
  │   └── runtime.rs: cuGraphLaunch(cached_decode_graph)  ← 1 call
  ├── get_f32 (3 outputs)                   → 0.3ms
  ├── pred set_data + pred_rt.execute()     → ~20-50ms
  │   └── runtime.rs: cuGraphLaunch(cached_pred_graph)    ← 1 call
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
| `crates/luminal_cuda/src/runtime.rs` | CudaRuntime — buffer mgmt, execute(), profiling |
| `crates/luminal_cuda/src/kernel/to_host.rs` | CudaGraphOp — child graph build/launch |
| `crates/luminal_cuda/src/host/cublaslt/mod.rs` | cuBLASLt host op |
| `crates/luminal_cuda/src/host/cublas/mod.rs` | Legacy cuBLAS host op (still registered for cache compat) |
| `crates/luminal_cuda/src/kernel/cuda_graph.rs` | CUDA graph tests + B.0 spike |
| `crates/luminal_cuda/src/block/mod.rs` | BlockOp trait (MegakernelOp, interpreter) |
| `examples/qwen3_tts/src/pipeline.rs` | TTS pipeline — prefill, generate_frames, decode_speech |
| `examples/qwen3_tts/src/backend.rs` | Backend abstraction — compile(), set_data, get_f32 |
| `scripts/modal_test_tts.py` | Modal A100 test harness |
| `docs/wave-a-results.md` | Wave A measurement data |

## Validation Checklist

For every wave:
- [ ] `cargo check -p luminal_cuda`
- [ ] `cargo test -p luminal_cuda --no-run`
- [ ] `cargo check -p qwen3_tts --features cuda`
- [ ] Modal run produces correct WAV output (192k samples, 8.00s at 24kHz)
- [ ] Profiling numbers recorded in `docs/`
- [ ] Fallback flags tested (`LUMINAL_SYNC_DEBUG=1`, `LUMINAL_FORCE_ZERO_ALL=1`)

For Wave B specifically:
- [ ] B.0 spike test passes on A100
- [ ] Output correctness: codes match between replay and dispatch paths
- [ ] dyn_map change triggers cache invalidation and re-capture
- [ ] No memory leak from graph instantiation (check with `nvidia-smi` over 100 frames)

## Housekeeping

- [ ] Remove one-time cache clear from `scripts/modal_test_tts.py` (lines 129-133) — cache is already refreshed
- [ ] decode_speech OOM is pre-existing, not related to perf work — track separately
