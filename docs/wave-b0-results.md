# Wave B.0 Capture Feasibility Results

## Status: COMPLETE (2026-02-16)

## Environment

- GPU: NVIDIA A100-80GB (Modal)
- CUDA: 12.4
- Test: `cargo test -p luminal_cuda wave_b0_stream_capture_with_graph_launch_and_cublaslt -- --ignored --nocapture`
- Commit: c559aa28

## Results

| Phase | Description | Result |
|-------|-------------|--------|
| Phase 1 | cuBLASLt matmul capture | **PASS** |
| Phase 2 | Child `cuGraphLaunch` capture | **FAIL** (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`) |
| Phase 3 | Instantiate + replay (cuBLASLt only) | **PASS** |
| Correctness | cuBLASLt replay output | **PASS** (16.0 = 2*K, expected) |

### Checklist

- [x] cuBLASLt matmul captured successfully
- [ ] ~~Child `cuGraphLaunch` captured successfully~~ — **NOT SUPPORTED** by CUDA
- [x] Parent graph instantiated successfully
- [x] Parent graph replay launched successfully
- [x] Captured path output matched expected value

## Raw Output

```
[B0] Phase 1: capturing cuBLASLt matmul only...
[B0] Phase 1 PASSED: cuBLASLt matmul is capturable
[B0] Phase 2: capturing child graph launch only...
[B0] Phase 2 FAILED: cuGraphLaunch during capture: DriverError(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "operation not permitted when stream is capturing")
[B0] Phase 3: capturing cuBLASLt only (child graph launch not capturable)...
[B0] Phase 3 graph launched successfully
[B0] Child graph output (not captured): 0 (expected 0.0)
[B0] cuBLASLt output verified: 16 (expected 16)

=== B0 Spike Results ===
  Phase 1 (cuBLASLt capture):      PASS
  Phase 2 (child graph capture):    FAIL (expected — cuGraphLaunch not capturable)
  Phase 3 (instantiate + replay):   PASS
  cuBLASLt replay correctness:      PASS
========================

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 34 filtered out; finished in 0.89s
```

## Key Finding

**`cuGraphLaunch` is fundamentally not supported during CUDA stream capture.** This means the runtime-level capture strategy must change:

Instead of capturing `execute()` which calls `CudaGraphOp.launch()` (which internally does `cuGraphLaunch`), we must either:

1. **Inline child graph kernels**: During capture, submit each kernel from the CudaGraphOp individually to the stream instead of launching the child graph as a unit
2. **Use `cuGraphAddChildGraphNode`**: Build the parent graph explicitly using CUDA Graph API, adding child graphs as child nodes (not via stream capture)
3. **Hybrid approach**: Capture non-graph ops via stream, add child graphs via explicit graph API, merge

### Approach Analysis

| Approach | Pros | Cons |
|----------|------|------|
| Inline kernels | Works with stream capture | Need to extract kernels from CudaGraphOps; lose graph-internal optimizations |
| `cuGraphAddChildGraphNode` | Preserves child graph structure | Requires explicit graph construction, not stream capture; complex |
| Hybrid | Best of both worlds | Most complex to implement |

### Recommended: Approach 2 or 3

`cuGraphAddChildGraphNode` is designed exactly for this — nesting graphs. The parent graph is built explicitly (not via stream capture), and child CudaGraphOps are added as child nodes. MegakernelOp kernels and cuBLASLt matmuls are added as kernel nodes or via mini stream captures.

## Fixes Applied During B0

| Commit | Fix |
|--------|-----|
| 85454ba4 | Use non-default stream for capture, hoist allocations |
| 4c65f271 | Suppress synchronize during capture via `CUDA_STREAM_CAPTURING` flag, fix graph leak |
| 024ffe2f | Bypass `bind_to_thread` in cuGraphLaunch path during capture |
| 8ed5ab29 | 3-phase test split with proper validation |
| c559aa28 | Restore required trait imports |
