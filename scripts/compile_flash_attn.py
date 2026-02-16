#!/usr/bin/env python3
"""
Compile FlashAttention Triton kernels to CUBIN artifacts.

Requirements:
- Python 3.9+
- triton >= 2.1 (build-time only dependency)

Outputs are written under crates/luminal_cuda/kernels/.
Each generated `.cubin` has a sibling `.entry` file containing the
kernel entry-point symbol name used by the Rust HostOp loader.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, Optional


def _require_triton():
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception as exc:  # pragma: no cover - host tool
        raise SystemExit(
            "Triton is required for FlashAttention AOT compilation.\n"
            "Install with: pip install 'triton>=2.1'"
        ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile FlashAttention CUBINs")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["masked", "causal"],
        choices=["masked", "causal"],
        help="Kernel modes to compile",
    )
    parser.add_argument(
        "--head-dims",
        nargs="+",
        type=int,
        default=[64, 128],
        choices=[64, 128],
        help="Head dimensions to compile",
    )
    parser.add_argument(
        "--sms",
        nargs="+",
        type=int,
        default=[80],
        help="SM architectures (e.g. 80 for A100, 89 for 4090)",
    )
    parser.add_argument(
        "--out-dir",
        default="crates/luminal_cuda/kernels",
        help="Output directory",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=3,
        help="Triton pipeline stages",
    )
    return parser.parse_args()


def _iter_targets(modes: Iterable[str], head_dims: Iterable[int], sms: Iterable[int]):
    for mode in modes:
        for head_dim in head_dims:
            for sm in sms:
                yield mode, head_dim, sm


def _extract_entry_name(compiled) -> str:
    metadata = getattr(compiled, "metadata", None)
    if metadata is not None:
        if isinstance(metadata, dict):
            name = metadata.get("name")
        else:
            name = getattr(metadata, "name", None)
        if name:
            return str(name)

    ptx: Optional[str | bytes] = compiled.asm.get("ptx")
    if isinstance(ptx, bytes):
        ptx = ptx.decode("utf-8", errors="ignore")
    if isinstance(ptx, str):
        match = re.search(r"\.visible\s+\.entry\s+([A-Za-z0-9_]+)", ptx)
        if match:
            return match.group(1)

    # Fallback expected by Rust loader if no metadata is available.
    return "flash_attn_fwd"


def _compile_with_new_api(
    triton,
    kernel_fn,
    *,
    head_dim: int,
    sm: int,
    is_causal: bool,
    has_mask: bool,
    num_warps: int,
    num_stages: int,
):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, make_backend

    signature = {
        "q_ptr": "*fp32",
        "k_ptr": "*fp32",
        "v_ptr": "*fp32",
        "mask_ptr": "*fp32",
        "out_ptr": "*fp32",
        "batch": "i32",
        "heads": "i32",
        "seq_q": "i32",
        "seq_k": "i32",
        "head_dim": "i32",
        "softmax_scale": "fp32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_DMODEL": "constexpr",
        "IS_CAUSAL": "constexpr",
        "HAS_MASK": "constexpr",
    }
    constants = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_DMODEL": head_dim,
        "IS_CAUSAL": is_causal,
        "HAS_MASK": has_mask,
    }
    src = ASTSource(fn=kernel_fn, signature=signature, constexprs=constants)
    target = GPUTarget("cuda", sm, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": num_warps, "num_stages": num_stages})
    return triton.compile(src=src, target=target, options=options.__dict__)


def _compile_with_legacy_api(
    triton,
    kernel_fn,
    *,
    head_dim: int,
    sm: int,
    is_causal: bool,
    has_mask: bool,
    num_warps: int,
    num_stages: int,
):
    signature = "*fp32,*fp32,*fp32,*fp32,*fp32,i32,i32,i32,i32,i32,fp32"
    return triton.compile(
        kernel_fn,
        signature=signature,
        constants={
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_DMODEL": head_dim,
            "IS_CAUSAL": is_causal,
            "HAS_MASK": has_mask,
        },
        num_warps=num_warps,
        num_stages=num_stages,
        device_type="cuda",
        cc=sm,
    )


def _compile_kernel(
    triton,
    kernel_fn,
    *,
    head_dim: int,
    sm: int,
    is_causal: bool,
    has_mask: bool,
    num_warps: int,
    num_stages: int,
):
    try:
        return _compile_with_new_api(
            triton,
            kernel_fn,
            head_dim=head_dim,
            sm=sm,
            is_causal=is_causal,
            has_mask=has_mask,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    except Exception:
        return _compile_with_legacy_api(
            triton,
            kernel_fn,
            head_dim=head_dim,
            sm=sm,
            is_causal=is_causal,
            has_mask=has_mask,
            num_warps=num_warps,
            num_stages=num_stages,
        )


def main() -> None:
    _require_triton()

    import triton
    import triton.language as tl

    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    @triton.jit
    def flash_attn_fwd(
        q_ptr,
        k_ptr,
        v_ptr,
        mask_ptr,
        out_ptr,
        batch,
        heads,
        seq_q,
        seq_k,
        head_dim,
        softmax_scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        # Dense contiguous tensors:
        # Q, K, V: [B, H, S, D]
        # Mask (masked mode): [B, H, S_q, S_k] additive mask
        # Out: [B, H, S_q, D]
        tl.static_assert(BLOCK_DMODEL == 64 or BLOCK_DMODEL == 128)

        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        row_mask = offs_m < seq_q
        d_mask = offs_d < head_dim

        q_head_offset = ((pid_b * heads + pid_h) * seq_q) * head_dim
        kv_head_offset = ((pid_b * heads + pid_h) * seq_k) * head_dim
        mask_head_offset = ((pid_b * heads + pid_h) * seq_q) * seq_k

        q_ptrs = q_ptr + q_head_offset + offs_m[:, None] * head_dim + offs_d[None, :]
        q = tl.load(q_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, seq_k, BLOCK_N):
            k_idx = start_n + offs_n
            key_mask = k_idx < seq_k

            k_ptrs = k_ptr + kv_head_offset + k_idx[:, None] * head_dim + offs_d[None, :]
            k = tl.load(k_ptrs, mask=key_mask[:, None] & d_mask[None, :], other=0.0)

            qk = tl.dot(q, tl.trans(k))
            qk *= softmax_scale
            qk = tl.where(key_mask[None, :], qk, float("-inf"))

            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= k_idx[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))

            if HAS_MASK:
                mask_ptrs = (
                    mask_ptr
                    + mask_head_offset
                    + offs_m[:, None] * seq_k
                    + k_idx[None, :]
                )
                additive = tl.load(
                    mask_ptrs,
                    mask=row_mask[:, None] & key_mask[None, :],
                    other=0.0,
                )
                qk += additive

            m_ij = tl.max(qk, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            l_i_new = alpha * l_i + beta * l_ij

            v_ptrs = v_ptr + kv_head_offset + k_idx[:, None] * head_dim + offs_d[None, :]
            v = tl.load(v_ptrs, mask=key_mask[:, None] & d_mask[None, :], other=0.0)

            p = p * (beta / (l_i_new + 1e-20))[:, None]
            acc = acc * ((l_i / (l_i_new + 1e-20)) * alpha)[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            m_i = m_i_new
            l_i = l_i_new

        # acc is already normalized — inner loop divides by l_i_new each step.

        o_head_offset = ((pid_b * heads + pid_h) * seq_q) * head_dim
        out_ptrs = out_ptr + o_head_offset + offs_m[:, None] * head_dim + offs_d[None, :]
        tl.store(out_ptrs, acc, mask=row_mask[:, None] & d_mask[None, :])

    for mode, head_dim, sm in _iter_targets(args.modes, args.head_dims, args.sms):
        is_causal = mode == "causal"
        has_mask = mode == "masked"
        num_warps = 4 if head_dim == 64 else 8

        compiled = _compile_kernel(
            triton,
            flash_attn_fwd,
            head_dim=head_dim,
            sm=sm,
            is_causal=is_causal,
            has_mask=has_mask,
            num_warps=num_warps,
            num_stages=args.num_stages,
        )

        cubin = compiled.asm.get("cubin")
        if cubin is None:
            raise RuntimeError("Triton compile did not emit CUBIN output")

        out_file = out_dir / f"flash_attn_{mode}_h{head_dim}_sm{sm}.cubin"
        out_file.write_bytes(cubin)

        entry_name = _extract_entry_name(compiled)
        entry_file = out_dir / f"{out_file.name}.entry"
        entry_file.write_text(f"{entry_name}\n", encoding="utf-8")

        print(f"Wrote {out_file} (entry={entry_name})")


if __name__ == "__main__":
    os.environ.setdefault("TRITON_CACHE_DIR", ".triton_cache")
    main()
