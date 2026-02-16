#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use luminal::{
    op::{CustomOp, DType, LLIROp},
    prelude::*,
};
use luminal_cuda::host::{
    flash_attn::{FlashAttentionMode, FlashAttentionOp},
    HostOp,
};

fn flash_attention_enabled_flag() -> bool {
    matches!(
        std::env::var("LUMINAL_USE_FLASH_ATTN")
            .ok()
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

fn kernels_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LUMINAL_FLASH_ATTN_KERNEL_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/luminal_cuda/kernels")
}

pub fn flash_attention_enabled(mode: FlashAttentionMode, head_dim: usize) -> bool {
    if !flash_attention_enabled_flag() {
        return false;
    }
    let stem = match mode {
        FlashAttentionMode::Masked => "flash_attn_masked",
        FlashAttentionMode::Causal => "flash_attn_causal",
    };
    let prefix = format!("{stem}_h{head_dim}_sm");
    std::fs::read_dir(kernels_dir())
        .ok()
        .is_some_and(|entries| {
            entries.flatten().any(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with(&prefix) && name.ends_with(".cubin"))
            })
        })
}

pub fn flash_attention_causal_enabled(head_dim: usize) -> bool {
    flash_attention_enabled(FlashAttentionMode::Causal, head_dim)
}

pub fn flash_attention_masked_enabled(head_dim: usize) -> bool {
    flash_attention_enabled(FlashAttentionMode::Masked, head_dim)
}

#[derive(Debug, Clone)]
struct FlashAttentionCall {
    mode: FlashAttentionMode,
    batch: Expression,
    heads: Expression,
    seq_q: Expression,
    seq_k: Expression,
    head_dim: Expression,
    dtype: DType,
}

impl CustomOp for FlashAttentionCall {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn HostOp>(Box::new(FlashAttentionOp::new(
            self.mode,
            self.batch,
            self.heads,
            self.seq_q,
            self.seq_k,
            self.head_dim,
            self.dtype,
        )) as Box<dyn HostOp>)
    }
}

pub fn flash_attention_causal(
    q: GraphTensor,
    k: GraphTensor,
    v: GraphTensor,
    head_dim: usize,
) -> GraphTensor {
    assert_eq!(
        q.dtype,
        DType::F32,
        "FlashAttention currently supports only F32 tensors"
    );
    let (batch, heads, seq_q, _) = q.dims4();
    let (_, _, seq_k, _) = k.dims4();
    q.graph().custom_op(
        FlashAttentionCall {
            mode: FlashAttentionMode::Causal,
            batch,
            heads,
            seq_q,
            seq_k,
            head_dim: head_dim.into(),
            dtype: q.dtype,
        },
        (q.id, k.id, v.id),
        q.shape,
        q.dtype,
    )
}

pub fn flash_attention_masked(
    q: GraphTensor,
    k: GraphTensor,
    v: GraphTensor,
    mask: GraphTensor,
    head_dim: usize,
) -> GraphTensor {
    assert_eq!(
        q.dtype,
        DType::F32,
        "FlashAttention currently supports only F32 tensors"
    );
    let (batch, heads, seq_q, _) = q.dims4();
    let (_, _, seq_k, _) = k.dims4();
    q.graph().custom_op(
        FlashAttentionCall {
            mode: FlashAttentionMode::Masked,
            batch,
            heads,
            seq_q,
            seq_k,
            head_dim: head_dim.into(),
            dtype: q.dtype,
        },
        (q.id, k.id, v.id, mask.id),
        q.shape,
        q.dtype,
    )
}
