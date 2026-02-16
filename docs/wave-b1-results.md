# Wave B.1 Stream Capture Attempt — Results

## Status: FAILED (2026-02-16)

Stream capture approach does not work due to nested `cuGraphLaunch` limitation found in B0.

## Environment

- GPU: NVIDIA A100-80GB (Modal)
- Branch: qwen3-tts @ 753d39b6
- Env: `LUMINAL_RUNTIME_GRAPH_REPLAY=1`, `LUMINAL_PROFILE=1`
- Command: `modal run scripts/modal_test_tts.py --target replay`

## What Happened

1. **Frame 0 (warmup)**: Normal dispatch — all ops execute successfully
2. **Frame 1 (capture)**: `cuStreamBeginCapture` starts, dispatch loop runs, hits `cuGraphLaunch` inside CudaGraphOp which returns `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`, the capture stream enters invalidated state
3. **Cascading panic**: Next op (`MegakernelOp::pre_execute`) attempts `memcpy_htod` on the invalidated stream → `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` → panic

### Stack trace
```
panicked at crates/luminal_cuda/src/block/mod.rs:1178:14:
Failed to re-upload tasks: DriverError(CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
  "operation failed due to a previous error during capture")

  3: <MegakernelOp as KernelOp>::pre_execute
  4: CudaGraphOp::execute_internal
  5: CudaRuntime::execute_host_ops_loop
  6: <CudaRuntime as Runtime>::execute
  7: generate_frames
```

## Pre-Capture Warmup Numbers (Frame 0)

| Runtime | Exec Time | Ops |
|---------|-----------|-----|
| Prefill | 319.7ms | 422 ops |
| Predictor (frame 0) | 381.0ms | 1182 ops |
| Decode (frame 0) | 215.2ms | 495 ops |

## Root Cause

`cuGraphLaunch` during stream capture returns `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.
This is a fundamental CUDA limitation confirmed in B0 Phase 2 testing.

The B1 `capture_runtime_graph()` path calls `execute_host_ops_loop()` during capture,
which dispatches CudaGraphOps that internally call `cuGraphLaunch` — triggering the error.

## Fix: Revised B1 Approach

Stream capture cannot be used for the whole `execute()` call. Instead, use
**explicit graph construction** with `cuGraphAddChildGraphNode`:

- Build parent graph explicitly via `cuGraphCreate`
- Add CudaGraphOp child graphs as child nodes (`cuGraphAddChildGraphNode`)
- Add cuBLAS ops via mini stream captures
- Instantiate and launch the parent graph

See `docs/cuda-perf-codex.md` for the full revised plan.
