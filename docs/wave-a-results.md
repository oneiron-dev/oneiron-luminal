# Wave 0 + A1 + A2 Results (2026-02-16)

## Summary

Implemented cuBLASLt upstream integration (Wave 0), per-op sync removal (A1), and selective buffer zeroing (A2) on the Luminal CUDA backend for Qwen3-TTS inference. Measured on A100-80GB via Modal.

**Result: 344.6ms/frame avg (frames 2-99), down from 407.9ms baseline — 63.3ms / 15.5% improvement.**

Still 12.3x slower than PyTorch reference (28ms/frame via CUDA graph replay).

## Per-Frame Breakdown (avg frames 2-99)

| Phase | Before | After | Delta | % of frame |
|-------|--------|-------|-------|------------|
| set_data | 0.05ms | 0.04ms | -0.01ms | 0.0% |
| kv_upload | 0.00ms | 0.00ms | — | 0.0% |
| decode_exec | 144.0ms | 114.3ms | **-29.7ms** (20.6%) | 33.2% |
| get_f32 | 0.5ms | 0.3ms | -0.2ms | 0.1% |
| pred_exec | 258.5ms | 225.1ms | **-33.4ms** (12.9%) | 65.3% |
| kv_scatter | 5.0ms | 4.8ms | -0.2ms | 1.4% |
| **total** | **407.9ms** | **344.6ms** | **-63.3ms** | **100%** |

## Runtime-Level Breakdown

| Runtime | Before total | After total | Savings | zero_buffers saved | sync saved | Ops |
|---------|-------------|------------|---------|-------------------|------------|-----|
| Decode talker | 150ms | 113ms | 37ms | 18ms | 19ms | 495 |
| Code predictor | 263ms | 218ms | 45ms | 37ms | 8ms | 1182 |
| **Combined** | **413ms** | **331ms** | **82ms** | **55ms** | **27ms** | **1677** |

Note: Runtime-level totals don't exactly match frame totals due to host overhead (set_data, get_f32, kv_scatter).

## What Each Wave Contributed

### Wave 0: cuBLASLt + output_bytes()
- Switched matmul from cuBLAS Sgemm to cuBLASLt (10-30% faster GEMMs per NVIDIA benchmarks)
- dtype-aware buffer allocation (correctness fix for future F16/BF16 paths)
- Op counts changed due to different egglog rewrites (273→495 decode, 1009→1182 pred)

### Wave A1: Sync Removal (~27ms saved)
- Removed per-op `synchronize()` from runtime exec loop
- Gated cuBLAS/cuBLASLt post-op sync behind `LUMINAL_SYNC_DEBUG=1`
- Made CudaGraphOp pre-launch sync conditional (only on rebuild/update)
- Added single-stream topology assertion as safety guard

### Wave A2: Selective Zeroing (~55ms saved)
- Classified kernels: atomicAdd → zero, Megakernel → conservative zero, others → skip
- `zero_buffers: 0.00ms (0 bufs)` across all steady-state frames
- Fallback: `LUMINAL_FORCE_ZERO_ALL=1` restores old behavior

## Pipeline Totals (100 frames)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| generate_frames compile | 174.6s | 169.3s | -5.3s |
| generate_frames execute | 43.8s | 35.8s | **-8.0s (18.3%)** |
| generate_frames total | 218.4s | 205.1s | -13.3s |
| decode_speech | OOM | OOM | pre-existing |

## Frame Variability

Steady-state frames (2-99) ranged from 329.3ms to 425.9ms. Periodic spikes (~every 10-20 frames) suggest GPU frequency scaling or memory subsystem contention:

- P50: ~336ms
- P90: ~371ms
- P99: ~404ms
- Max: 425.9ms (frame 16)

Frame 1 was 580.6ms due to one-time KV cache upload (30.1ms) and cold decode (304ms).

## Remaining Bottleneck

The code predictor is still the dominant bottleneck at 65% of frame time (225ms). It has 1182 host-dispatched ops vs the decode talker's 495. Both runtimes dispatch ops individually from the host — **the fundamental architectural gap vs PyTorch (which uses CUDA graph replay: 0 host dispatches per frame)**.

## Environment

- Hardware: NVIDIA A100-80GB PCIe
- CUDA: 12.4 (toolkit) / 12.8 (driver)
- Branch: qwen3-tts
- Commits: 767ce087 (Wave 0+A1+A2), 65ea3dae (cache compat fix)
- Env: `LUMINAL_PROFILE=1`, `LUMINAL_CACHE_DIR=/cache/luminal_cache`

## Commits

- `767ce087` — Wave 0 (cuBLASLt) + A1 (sync removal) + A2 (selective zeroing) + B.0 spike harness
- `65ea3dae` — Cache compatibility fix (register both cuBLASLt + cuBLAS for stale e-graph cache extraction)

## Files Modified

| File | Changes |
|------|---------|
| `crates/luminal_cuda/src/host/mod.rs` | cuBLASLt wired in, `output_bytes()`, `zero_output_nodes()` trait methods |
| `crates/luminal_cuda/src/host/cublaslt/mod.rs` | New cuBLASLt host op (363 lines) + 4 egglog rewrite files |
| `crates/luminal_cuda/src/host/cublas/mod.rs` | Post-op sync gated, `output_bytes()`, `stats_name()` |
| `crates/luminal_cuda/src/runtime.rs` | Selective zeroing, sync removal, output_bytes() allocation, stream guard |
| `crates/luminal_cuda/src/kernel/to_host.rs` | atomicAdd/Megakernel classification, conditional sync, output_bytes() |
| `crates/luminal_cuda/src/kernel/mod.rs` | Default `output_bytes()` |
| `crates/luminal_cuda/src/block/mod.rs` | Default `output_bytes()` |
| `crates/luminal_cuda/src/kernel/cuda_graph.rs` | Wave B.0 spike test (ignored) |
| `crates/luminal_cuda/src/kernel/other_ops.rs` | dtype-safe output_bytes for mean-reduce |
| `crates/luminal_cuda/src/lib.rs` | BF16 CUDA type mapping fix |
| `scripts/modal_test_tts.py` | `LUMINAL_PROFILE=1`, one-time cache clear (remove after first run) |
| `examples/qwen3_tts/src/pipeline.rs` | Per-frame profiling instrumentation |
