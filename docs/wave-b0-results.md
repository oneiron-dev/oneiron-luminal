# Wave B.0 Capture Feasibility Results

## Status

- Pending execution on Modal A100.
- Run command:
  - `modal run scripts/modal_test_tts.py --target wave_b0`

## Environment

- GPU: A100-80GB
- Script: `scripts/modal_test_tts.py::test_wave_b0_capture`
- Test:
  - `cargo test -p luminal_cuda wave_b0_stream_capture_with_graph_launch_and_cublaslt -- --ignored --nocapture`

## Checklist

- [ ] Child `cuGraphLaunch` captured successfully
- [ ] cuBLASLt matmul captured successfully
- [ ] Parent graph instantiated successfully
- [ ] Parent graph replay launched successfully
- [ ] Captured path output matched expected value

## Notes

- Populate this file with command output, pass/fail status, and any CUDA error codes.
