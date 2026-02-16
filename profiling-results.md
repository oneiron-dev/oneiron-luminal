# Qwen3-TTS Profiling Results — A100 80GB

**Date**: 2026-02-16
**GPU**: NVIDIA A100 80GB PCIe (CUDA 12.8, nvcc 12.4)
**Branch**: `qwen3-tts` (commit `bacf21c9`)
**Workload**: 100 frames of Qwen3-TTS-12Hz-1.7B-VoiceDesign inference

## Per-Frame Averages (frames 2-99, steady state)

| Phase | Time (ms) | Ops | Per-op (ms) | % of frame |
|-------|-----------|-----|-------------|------------|
| set_data | 0.02 | — | — | 0.0% |
| kv_upload | 0.00 | — | — | 0.0% |
| **decode_exec** | **106.25** | **495** | **0.215** | **42.1%** |
| get_f32 | 0.22 | — | — | 0.1% |
| **pred_exec** | **141.97** | **1182** | **0.120** | **56.2%** |
| kv_scatter | 4.05 | — | — | 1.6% |
| **Total** | **252.50** | **1677** | **0.151** | **100%** |

## Runtime Internals (typical frame)

```
Decode:    zero_buffers: 0.00ms | prebuild: 0.00ms | exec_ops: 105ms (495 ops) | final_sync: 0.09ms
Predictor: zero_buffers: 0.00ms | prebuild: 0.00ms | exec_ops: 137ms (1182 ops) | final_sync: 0.07ms
```

- `zero_buffers`: **NOT a bottleneck** (0.00ms — buffers are not re-zeroed between frames)
- `prebuild`: 0.00ms (graphs already built)
- `exec_ops`: **100% of execution time** — the per-op dispatch loop
- `final_sync`: <0.1ms

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

## Luminal vs PyTorch/vLLM Comparison

Same model (hidden=2048, 28 layers, GQA 16/2), same GPU (A100).

| | Luminal | PyTorch/vLLM | Gap | Est. GPU time | **Host overhead** |
|---|---------|-------------|-----|---------------|-------------------|
| Decode (Talker) | 106ms (495 ops) | 6.7ms (1 graph) | 15.8x | ~7ms | **~99ms (93%)** |
| Predictor | 142ms (1182 ops) | 19.1ms (15 graphs) | 7.4x | ~19ms | **~123ms (87%)** |
| CPU overhead | 4.3ms | 2.1ms | 2.0x | — | — |
| **Total** | **252ms** | **27.6ms** | **9.1x** | **~26ms** | **~222ms (88%)** |

### Key Insight

**88% of per-frame execution time is host dispatch overhead**, not GPU kernel time.

- Luminal dispatches 1,677 individual operations per frame (cuGraphLaunch + cuBLAS/cuBLASLt calls)
- Each op costs ~0.12-0.22ms of host-side overhead (parameter setup, buffer resolution, launch)
- PyTorch captures the entire forward pass into 1-15 CUDA graphs → 1-15 `cuGraphLaunch` calls per frame
- Eliminating per-op dispatch overhead would bring Luminal from ~252ms to ~30ms (competitive with PyTorch)

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

## Conclusions

1. **The #1 optimization target is host dispatch overhead elimination** (Wave B.1: whole-graph CUDA capture)
2. Buffer zeroing, graph prebuild, and data transfer are negligible (<5ms combined)
3. kv_scatter (CPU-side) is 4ms — acceptable
4. The theoretical floor is ~30ms/frame if all ops are captured into a single CUDA graph
5. Per-kernel timing data is not yet available — need nsight or CUDA event instrumentation to identify slow kernels

## Raw Data

<details>
<summary>All 100 frame timings</summary>

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
