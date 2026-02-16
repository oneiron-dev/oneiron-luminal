FlashAttention kernel artifacts are expected in this directory.

Phase 1 expected filenames:

- `flash_attn_masked_h64_sm80.cubin`
- `flash_attn_masked_h128_sm80.cubin`
- `flash_attn_causal_h64_sm80.cubin` (optional for prefill)
- `flash_attn_causal_h128_sm80.cubin` (optional for prefill)

Each `.cubin` also has a sibling `.entry` file written by the compiler script,
for example `flash_attn_masked_h128_sm80.cubin.entry`.

Generate with:

```bash
python3 scripts/compile_flash_attn.py --modes masked causal --head-dims 64 128 --sms 80
```

Runtime enablement:

- By default, qwen3_tts keeps the decomposed attention path.
- Set `LUMINAL_USE_FLASH_ATTN=1` to route CUDA qwen3_tts attention to `FlashAttentionOp`.
- Override kernel directory with `LUMINAL_FLASH_ATTN_KERNEL_DIR=/path/to/kernels`.
