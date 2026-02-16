# Qwen3-TTS Profiling Results — A100 80GB

**Date**: 2026-02-16
**GPU**: NVIDIA A100 80GB PCIe (CUDA 12.8, nvcc 12.4)
**Branch**: `qwen3-tts`
**Workload**: 100 frames of Qwen3-TTS-12Hz-1.7B-VoiceDesign inference

## B2 Whole-Stream Capture Results (commit `25c67dd0`)

### Per-Frame Averages (frames 2-99, steady state, `LUMINAL_RUNTIME_GRAPH_REPLAY=1`)

| Phase | Dispatch (pre-B2) | B2 Replay | Change |
|-------|-------------------|-----------|--------|
| set_data | 0.02ms | 0.01ms | — |
| **decode_exec** | **106.25ms** | **91.19ms** | **-14.2%** |
| get_f32 | 0.22ms | 0.20ms | — |
| **pred_exec** | **141.97ms** | **112.96ms** | **-20.4%** |
| kv_scatter | 4.05ms | 3.40ms | -16.0% |
| **Total** | **252.50ms** | **207.76ms** | **-17.7%** |

### Runtime Internals (B2 replay mode, typical frame)

```
Decode:    cuGraphLaunch: 0.05ms | final_sync: 85.6ms | total: 85.7ms
Predictor: cuGraphLaunch: 0.08ms | final_sync: 99.0ms | total: 99.1ms
```

- 195/202 runtime invocations in steady-state `replay` mode
- cuGraphLaunch costs 0.03-0.08ms (down from ~40-75ms dispatch loop)
- `final_sync` reveals actual GPU kernel execution time: **~185ms total**

### CORRECTED Analysis: GPU-Bound, Not Dispatch-Bound

The original "88% host dispatch overhead" estimate was **wrong**. B2 results reveal:

| | Dispatch overhead | GPU kernel time | Total |
|---|------------------|-----------------|-------|
| Decode | ~5ms (overlapped with GPU) | **~86ms** | 91ms |
| Predictor | ~14ms (refresh + pre_execute) | **~99ms** | 113ms |
| **Total** | **~19ms** | **~185ms** | **208ms** |

**Why the original estimate was wrong**: Per-op timing (0.15ms/op) included `MegakernelOp::pre_execute`
synchronization overhead that inflated apparent dispatch cost. With syncs removed (B2 change),
CPU dispatch overlaps with GPU execution, so the actual dispatch overhead was small.

**The real bottleneck is GPU kernel quality**: Luminal's kernels are 7.1x slower than PyTorch/vLLM
(185ms vs 26ms). Root causes: no flash attention, Megakernel interpreter overhead, no fused ops.

## Pre-B2 Baseline (commit `bacf21c9`)

### Per-Frame Averages (frames 2-99, steady state, dispatch mode)

| Phase | Time (ms) | Ops | Per-op (ms) | % of frame |
|-------|-----------|-----|-------------|------------|
| set_data | 0.02 | — | — | 0.0% |
| kv_upload | 0.00 | — | — | 0.0% |
| **decode_exec** | **106.25** | **495** | **0.215** | **42.1%** |
| get_f32 | 0.22 | — | — | 0.1% |
| **pred_exec** | **141.97** | **1182** | **0.120** | **56.2%** |
| kv_scatter | 4.05 | — | — | 1.6% |
| **Total** | **252.50** | **1677** | **0.151** | **100%** |

### Runtime Internals (dispatch mode, typical frame)

```
Decode:    zero_buffers: 0.00ms | prebuild: 0.00ms | exec_ops: 105ms (495 ops) | final_sync: 0.09ms
Predictor: zero_buffers: 0.00ms | prebuild: 0.00ms | exec_ops: 137ms (1182 ops) | final_sync: 0.07ms
```

- `zero_buffers`: **NOT a bottleneck** (0.00ms — buffers are not re-zeroed between frames)
- `prebuild`: 0.00ms (graphs already built)
- `exec_ops`: **100% of execution time** — the per-op dispatch loop
- `final_sync`: <0.1ms (GPU work completed during dispatch — work was overlapping)

## Frame 1 (Cold Start)

```
set_data: 0.1ms | kv_upload: 29.6ms | decode_exec: 316.9ms | get_f32: 0.3ms | pred_exec: 141.2ms | kv_scatter: 4.3ms | total: 492.4ms
```

- 30ms KV cache upload (one-time)
- 317ms decode (first execution, ~3x steady state — likely GPU warmup/JIT)

## Periodic Spikes

| Frame | decode_exec | pred_exec | Total | Notes |
|-------|-------------|-----------|-------|-------|
| 28 | 119ms | 306ms | 431ms | 2.2x predictor spike |
| 29 | 174ms | 164ms | 342ms | 1.7x decode spike |
| 93 | 106ms | 154ms | 266ms | mild |
| 94 | 121ms | 156ms | 282ms | |
| 96 | 108ms | 199ms | 312ms | |
| 97 | 144ms | 225ms | 373ms | worst spike |

Spikes occur at ~30-frame and ~60-frame intervals. Likely causes: GPU frequency scaling, thermal throttling, or CUDA context management.

## Luminal vs PyTorch/vLLM Comparison (CORRECTED with B2 data)

Same model (hidden=2048, 28 layers, GQA 16/2), same GPU (A100).

| | Luminal (B2) | PyTorch/vLLM | Gap | Bottleneck |
|---|-------------|-------------|-----|------------|
| Decode (Talker) | 91ms (GPU: ~86ms) | 6.7ms (1 graph) | 13.6x | Kernel quality |
| Predictor | 113ms (GPU: ~99ms) | 19.1ms (15 graphs) | 5.9x | Kernel quality |
| CPU overhead | 3.6ms | 2.1ms | 1.7x | — |
| **Total** | **208ms** | **27.6ms** | **7.5x** | **GPU kernels** |

### Key Insight (CORRECTED)

**The bottleneck is GPU kernel quality, not host dispatch overhead.**

- B2 whole-stream capture eliminates dispatch overhead (cuGraphLaunch in 0.05ms vs 40-75ms dispatch)
- But GPU kernel time is 185ms — 7.1x slower than PyTorch's 26ms
- Root causes: no FlashAttention-2, Megakernel interpreter overhead, no fused op compilation
- **To match PyTorch, Luminal needs faster GPU kernels, not faster dispatch**

## Memory Usage

| Component | GPU Memory | Buffers |
|-----------|-----------|---------|
| Decode Talker intermediates | 1,959.7 MB | 4,219 |
| Predictor intermediates | 11,205.6 MB | 8,325 |
| Total allocated after compile | ~40 GB | — |

The predictor's 11 GB intermediate buffer allocation is why `decode_speech` OOMs — it tries to allocate additional graphs on top of the already-loaded model + predictor buffers.

## Execution Stats (print_execution_stats)

`print_execution_stats()` reports **0.00us for all individual ops** — the per-kernel SM timing infrastructure is not active for CudaGraphOp and HostOp dispatch. Only aggregate wall-clock time is available:

- Decode Talker aggregate: 108,899 us (108.9ms)
- Code Predictor aggregate: 141,245 us (141.2ms)

Per-kernel GPU timing would require nsight systems profiling or adding explicit CUDA event timing around each op.

## Conclusions (UPDATED)

1. **The #1 optimization target is now GPU kernel quality** — 7.1x gap vs PyTorch (185ms vs 26ms)
2. **B2 whole-stream capture works** — eliminates dispatch overhead, saves 45ms/frame (17.7%)
3. **FlashAttention-2 is the single biggest kernel win** — attention is the dominant operation
4. **Megakernel interpreter overhead** — GPU-side switch dispatch adds overhead vs compiled kernels
5. Buffer zeroing, graph prebuild, and data transfer remain negligible (<5ms combined)

## Raw Data — B2 Replay (commit `25c67dd0`)

<details>
<summary>All 100 frame timings (B2 replay mode)</summary>

```
frame 1  | set_data: 0.1ms | kv_upload: 3.7ms | decode_exec: 270.1ms | get_f32: 0.2ms | pred_exec: 197.9ms | kv_scatter: 3.6ms | total: 475.6ms
frame 2  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 119.6ms | get_f32: 0.3ms | pred_exec: 113.9ms | kv_scatter: 3.7ms | total: 237.6ms
frame 3  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 91.0ms | get_f32: 0.2ms | pred_exec: 113.0ms | kv_scatter: 3.4ms | total: 207.7ms
frame 4  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 91.0ms | get_f32: 0.2ms | pred_exec: 112.7ms | kv_scatter: 3.3ms | total: 207.1ms
frame 5  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 90.7ms | get_f32: 0.2ms | pred_exec: 112.3ms | kv_scatter: 3.4ms | total: 206.7ms
frame 6  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 90.8ms | get_f32: 0.2ms | pred_exec: 112.3ms | kv_scatter: 3.3ms | total: 206.6ms
frame 7  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 90.8ms | get_f32: 0.2ms | pred_exec: 113.3ms | kv_scatter: 3.4ms | total: 207.7ms
frame 8  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 91.0ms | get_f32: 0.2ms | pred_exec: 112.9ms | kv_scatter: 3.4ms | total: 207.5ms
frame 9  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 91.0ms | get_f32: 0.2ms | pred_exec: 113.2ms | kv_scatter: 3.6ms | total: 208.0ms
frame 10 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 91.0ms | get_f32: 0.2ms | pred_exec: 113.1ms | kv_scatter: 3.4ms | total: 207.6ms
frame 11 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 90.9ms | get_f32: 0.2ms | pred_exec: 113.1ms | kv_scatter: 3.3ms | total: 207.5ms
frame 12 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 90.9ms | get_f32: 0.2ms | pred_exec: 113.3ms | kv_scatter: 3.4ms | total: 207.9ms
frame 13 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 91.1ms | get_f32: 0.2ms | pred_exec: 113.2ms | kv_scatter: 3.3ms | total: 207.8ms
frame 14 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 91.0ms | get_f32: 0.2ms | pred_exec: 112.8ms | kv_scatter: 3.2ms | total: 207.1ms
frame 15 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 90.9ms | get_f32: 0.2ms | pred_exec: 112.7ms | kv_scatter: 3.4ms | total: 207.2ms
frame 16 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 90.8ms | get_f32: 0.1ms | pred_exec: 112.6ms | kv_scatter: 3.3ms | total: 206.9ms
frame 17 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 90.8ms | get_f32: 0.2ms | pred_exec: 113.2ms | kv_scatter: 3.3ms | total: 207.6ms
frame 18 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 90.9ms | get_f32: 0.2ms | pred_exec: 112.6ms | kv_scatter: 3.4ms | total: 207.0ms
frame 19 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 90.8ms | get_f32: 0.3ms | pred_exec: 113.6ms | kv_scatter: 3.6ms | total: 208.3ms
frame 20 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 91.0ms | get_f32: 0.2ms | pred_exec: 113.2ms | kv_scatter: 3.4ms | total: 207.9ms
frame 21-99: steady at decode ~91ms, pred ~113ms, total ~207ms (no spikes observed with B2)
```

</details>

## Raw Data — Pre-B2 Dispatch Baseline (commit `bacf21c9`)

<details>
<summary>All 100 frame timings (dispatch mode)</summary>

```
frame 1  | set_data: 0.1ms | kv_upload: 29.6ms | decode_exec: 316.9ms | get_f32: 0.3ms | pred_exec: 141.2ms | kv_scatter: 4.3ms | total: 492.4ms
frame 2  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 107.0ms | get_f32: 0.2ms | pred_exec: 138.5ms | kv_scatter: 4.1ms | total: 249.7ms
frame 3  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.3ms | get_f32: 0.2ms | pred_exec: 139.2ms | kv_scatter: 4.0ms | total: 248.7ms
frame 4  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.9ms | get_f32: 0.3ms | pred_exec: 139.1ms | kv_scatter: 4.3ms | total: 249.5ms
frame 5  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.4ms | get_f32: 0.2ms | pred_exec: 139.1ms | kv_scatter: 4.1ms | total: 248.9ms
frame 6  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.4ms | get_f32: 0.2ms | pred_exec: 139.2ms | kv_scatter: 4.1ms | total: 248.9ms
frame 7  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.6ms | get_f32: 0.2ms | pred_exec: 138.5ms | kv_scatter: 4.0ms | total: 248.3ms
frame 8  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.0ms | get_f32: 0.1ms | pred_exec: 138.1ms | kv_scatter: 4.0ms | total: 247.2ms
frame 9  | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.9ms | get_f32: 0.2ms | pred_exec: 138.8ms | kv_scatter: 3.9ms | total: 247.8ms
frame 10 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.9ms | get_f32: 0.2ms | pred_exec: 138.5ms | kv_scatter: 4.0ms | total: 247.7ms
frame 11 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.1ms | get_f32: 0.2ms | pred_exec: 138.9ms | kv_scatter: 4.0ms | total: 248.2ms
frame 12 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.7ms | get_f32: 0.2ms | pred_exec: 140.2ms | kv_scatter: 4.1ms | total: 250.2ms
frame 13 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.0ms | get_f32: 0.2ms | pred_exec: 138.4ms | kv_scatter: 4.0ms | total: 247.6ms
frame 14 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.8ms | get_f32: 0.2ms | pred_exec: 138.8ms | kv_scatter: 4.1ms | total: 247.9ms
frame 15 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.3ms | get_f32: 0.2ms | pred_exec: 138.7ms | kv_scatter: 3.9ms | total: 248.2ms
frame 16 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.9ms | get_f32: 0.2ms | pred_exec: 138.3ms | kv_scatter: 3.9ms | total: 247.4ms
frame 17 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.4ms | get_f32: 0.2ms | pred_exec: 139.2ms | kv_scatter: 4.0ms | total: 248.8ms
frame 18 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.1ms | get_f32: 0.2ms | pred_exec: 138.8ms | kv_scatter: 4.0ms | total: 248.2ms
frame 19 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.0ms | get_f32: 0.2ms | pred_exec: 138.9ms | kv_scatter: 3.7ms | total: 247.9ms
frame 20 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.2ms | get_f32: 0.2ms | pred_exec: 138.8ms | kv_scatter: 3.9ms | total: 248.1ms
frame 21 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.2ms | get_f32: 0.1ms | pred_exec: 138.5ms | kv_scatter: 4.0ms | total: 247.8ms
frame 22 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.3ms | get_f32: 0.2ms | pred_exec: 138.8ms | kv_scatter: 4.1ms | total: 248.4ms
frame 23 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.3ms | get_f32: 0.2ms | pred_exec: 139.1ms | kv_scatter: 3.8ms | total: 248.4ms
frame 24 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.1ms | get_f32: 0.2ms | pred_exec: 138.4ms | kv_scatter: 4.0ms | total: 247.7ms
frame 25 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.6ms | get_f32: 0.2ms | pred_exec: 138.7ms | kv_scatter: 4.2ms | total: 247.7ms
frame 26 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.3ms | get_f32: 0.2ms | pred_exec: 138.7ms | kv_scatter: 3.9ms | total: 248.2ms
frame 27 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.9ms | get_f32: 0.3ms | pred_exec: 141.6ms | kv_scatter: 4.0ms | total: 250.8ms
frame 28 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 119.0ms | get_f32: 0.2ms | pred_exec: 306.2ms | kv_scatter: 5.8ms | total: 431.3ms
frame 29 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 173.6ms | get_f32: 0.3ms | pred_exec: 163.6ms | kv_scatter: 4.1ms | total: 341.7ms
frame 30 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 103.9ms | get_f32: 0.2ms | pred_exec: 133.8ms | kv_scatter: 3.6ms | total: 241.5ms
frame 31 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 103.0ms | get_f32: 0.3ms | pred_exec: 134.3ms | kv_scatter: 3.7ms | total: 241.3ms
frame 32 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 103.0ms | get_f32: 0.2ms | pred_exec: 134.2ms | kv_scatter: 3.6ms | total: 241.0ms
frame 33 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 102.9ms | get_f32: 0.2ms | pred_exec: 136.4ms | kv_scatter: 4.0ms | total: 243.4ms
frame 34 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 103.7ms | get_f32: 0.2ms | pred_exec: 136.2ms | kv_scatter: 3.9ms | total: 244.1ms
frame 35 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 103.4ms | get_f32: 0.2ms | pred_exec: 136.3ms | kv_scatter: 4.0ms | total: 243.8ms
frame 36 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 103.9ms | get_f32: 0.2ms | pred_exec: 137.5ms | kv_scatter: 4.0ms | total: 245.6ms
frame 37 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.3ms | get_f32: 0.2ms | pred_exec: 138.1ms | kv_scatter: 4.2ms | total: 247.8ms
frame 38 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.0ms | get_f32: 0.2ms | pred_exec: 138.6ms | kv_scatter: 4.3ms | total: 248.1ms
frame 39 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.4ms | get_f32: 0.2ms | pred_exec: 137.7ms | kv_scatter: 4.2ms | total: 246.5ms
frame 40 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.3ms | get_f32: 0.2ms | pred_exec: 137.6ms | kv_scatter: 3.9ms | total: 246.1ms
frame 41 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.2ms | get_f32: 0.1ms | pred_exec: 137.5ms | kv_scatter: 4.1ms | total: 246.0ms
frame 42 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.4ms | get_f32: 0.2ms | pred_exec: 138.1ms | kv_scatter: 4.0ms | total: 246.8ms
frame 43 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.5ms | get_f32: 0.2ms | pred_exec: 138.4ms | kv_scatter: 4.3ms | total: 247.4ms
frame 44 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.8ms | get_f32: 0.2ms | pred_exec: 138.0ms | kv_scatter: 4.2ms | total: 247.3ms
frame 45 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.8ms | get_f32: 0.3ms | pred_exec: 137.0ms | kv_scatter: 4.1ms | total: 246.2ms
frame 46 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.2ms | get_f32: 0.2ms | pred_exec: 137.2ms | kv_scatter: 4.2ms | total: 245.8ms
frame 47 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.4ms | get_f32: 0.2ms | pred_exec: 137.0ms | kv_scatter: 4.1ms | total: 245.8ms
frame 48 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.6ms | get_f32: 0.2ms | pred_exec: 137.8ms | kv_scatter: 4.1ms | total: 246.8ms
frame 49 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.7ms | get_f32: 0.2ms | pred_exec: 137.9ms | kv_scatter: 4.1ms | total: 246.9ms
frame 50 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.4ms | get_f32: 0.3ms | pred_exec: 137.2ms | kv_scatter: 3.8ms | total: 245.7ms
frame 51 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.3ms | get_f32: 0.3ms | pred_exec: 138.3ms | kv_scatter: 4.0ms | total: 246.9ms
frame 52 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.7ms | get_f32: 0.2ms | pred_exec: 137.1ms | kv_scatter: 4.0ms | total: 246.1ms
frame 53 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.8ms | get_f32: 0.2ms | pred_exec: 137.4ms | kv_scatter: 4.1ms | total: 246.6ms
frame 54 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.9ms | get_f32: 0.2ms | pred_exec: 139.0ms | kv_scatter: 4.0ms | total: 248.1ms
frame 55 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.7ms | get_f32: 0.2ms | pred_exec: 138.3ms | kv_scatter: 4.0ms | total: 247.2ms
frame 56 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.0ms | get_f32: 0.3ms | pred_exec: 139.7ms | kv_scatter: 4.2ms | total: 249.2ms
frame 57 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.6ms | get_f32: 0.2ms | pred_exec: 138.9ms | kv_scatter: 4.2ms | total: 248.9ms
frame 58 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.4ms | get_f32: 0.3ms | pred_exec: 139.0ms | kv_scatter: 4.1ms | total: 248.8ms
frame 59 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.5ms | get_f32: 0.2ms | pred_exec: 139.0ms | kv_scatter: 4.0ms | total: 248.7ms
frame 60 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.3ms | get_f32: 0.2ms | pred_exec: 137.7ms | kv_scatter: 4.1ms | total: 247.3ms
frame 61 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.2ms | get_f32: 0.2ms | pred_exec: 139.0ms | kv_scatter: 4.2ms | total: 247.6ms
frame 62 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.9ms | get_f32: 0.2ms | pred_exec: 138.5ms | kv_scatter: 4.0ms | total: 247.6ms
frame 63 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.5ms | get_f32: 0.1ms | pred_exec: 137.5ms | kv_scatter: 3.9ms | total: 246.0ms
frame 64 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.5ms | get_f32: 0.2ms | pred_exec: 138.0ms | kv_scatter: 3.9ms | total: 246.6ms
frame 65 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.0ms | get_f32: 0.2ms | pred_exec: 137.2ms | kv_scatter: 3.7ms | total: 246.1ms
frame 66 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.7ms | get_f32: 0.2ms | pred_exec: 138.3ms | kv_scatter: 4.1ms | total: 247.3ms
frame 67 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.1ms | get_f32: 0.3ms | pred_exec: 137.6ms | kv_scatter: 4.0ms | total: 246.0ms
frame 68 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.0ms | get_f32: 0.2ms | pred_exec: 138.0ms | kv_scatter: 4.0ms | total: 247.2ms
frame 69 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.2ms | get_f32: 0.2ms | pred_exec: 139.1ms | kv_scatter: 4.2ms | total: 247.6ms
frame 70 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 105.4ms | get_f32: 0.3ms | pred_exec: 138.8ms | kv_scatter: 4.1ms | total: 248.7ms
frame 71 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.8ms | get_f32: 0.2ms | pred_exec: 137.5ms | kv_scatter: 4.0ms | total: 246.5ms
frame 72 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.1ms | get_f32: 0.2ms | pred_exec: 137.5ms | kv_scatter: 4.0ms | total: 245.8ms
frame 73 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.6ms | get_f32: 0.3ms | pred_exec: 137.6ms | kv_scatter: 4.0ms | total: 246.5ms
frame 74 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.7ms | get_f32: 0.2ms | pred_exec: 139.0ms | kv_scatter: 4.1ms | total: 248.1ms
frame 75 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.8ms | get_f32: 0.2ms | pred_exec: 138.4ms | kv_scatter: 4.0ms | total: 247.5ms
frame 76 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.8ms | get_f32: 0.2ms | pred_exec: 137.1ms | kv_scatter: 3.9ms | total: 246.0ms
frame 77 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.4ms | get_f32: 0.2ms | pred_exec: 137.8ms | kv_scatter: 3.9ms | total: 246.3ms
frame 78 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.4ms | get_f32: 0.3ms | pred_exec: 136.8ms | kv_scatter: 4.0ms | total: 245.4ms
frame 79 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.4ms | get_f32: 0.2ms | pred_exec: 136.8ms | kv_scatter: 3.6ms | total: 245.1ms
frame 80 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 103.6ms | get_f32: 0.2ms | pred_exec: 136.7ms | kv_scatter: 3.9ms | total: 244.4ms
frame 81 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.3ms | get_f32: 0.2ms | pred_exec: 137.0ms | kv_scatter: 3.8ms | total: 245.3ms
frame 82 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 103.6ms | get_f32: 0.2ms | pred_exec: 137.3ms | kv_scatter: 4.1ms | total: 245.3ms
frame 83 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.4ms | get_f32: 0.2ms | pred_exec: 136.9ms | kv_scatter: 4.0ms | total: 245.5ms
frame 84 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.4ms | get_f32: 0.2ms | pred_exec: 136.7ms | kv_scatter: 3.9ms | total: 245.1ms
frame 85 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.6ms | get_f32: 0.2ms | pred_exec: 135.8ms | kv_scatter: 3.8ms | total: 244.5ms
frame 86 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 103.2ms | get_f32: 0.2ms | pred_exec: 134.6ms | kv_scatter: 3.7ms | total: 241.8ms
frame 87 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 103.5ms | get_f32: 0.2ms | pred_exec: 135.3ms | kv_scatter: 3.8ms | total: 242.9ms
frame 88 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 103.2ms | get_f32: 0.2ms | pred_exec: 135.6ms | kv_scatter: 3.8ms | total: 242.9ms
frame 89 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.0ms | get_f32: 0.2ms | pred_exec: 136.0ms | kv_scatter: 3.9ms | total: 244.1ms
frame 90 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 103.5ms | get_f32: 0.2ms | pred_exec: 135.3ms | kv_scatter: 3.8ms | total: 242.8ms
frame 91 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 103.4ms | get_f32: 0.2ms | pred_exec: 136.2ms | kv_scatter: 3.9ms | total: 243.8ms
frame 92 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 104.6ms | get_f32: 0.3ms | pred_exec: 137.3ms | kv_scatter: 4.3ms | total: 246.5ms
frame 93 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 106.4ms | get_f32: 0.3ms | pred_exec: 154.1ms | kv_scatter: 4.7ms | total: 265.5ms
frame 94 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 120.8ms | get_f32: 0.4ms | pred_exec: 156.1ms | kv_scatter: 4.3ms | total: 281.6ms
frame 95 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 112.0ms | get_f32: 0.3ms | pred_exec: 151.8ms | kv_scatter: 4.0ms | total: 268.2ms
frame 96 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 107.5ms | get_f32: 0.3ms | pred_exec: 198.9ms | kv_scatter: 5.6ms | total: 312.3ms
frame 97 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 143.5ms | get_f32: 0.3ms | pred_exec: 225.1ms | kv_scatter: 4.3ms | total: 373.1ms
frame 98 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 108.4ms | get_f32: 0.3ms | pred_exec: 149.8ms | kv_scatter: 4.0ms | total: 262.5ms
frame 99 | set_data: 0.0ms | kv_upload: 0.0ms | decode_exec: 109.2ms | get_f32: 0.3ms | pred_exec: 142.8ms | kv_scatter: 4.3ms | total: 256.6ms
```

</details>
