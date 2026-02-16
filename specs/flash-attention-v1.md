# Codex Spec: FlashAttention-2 Integration (Phase 1)

## Goal

Replace Luminal's decomposed attention (Q@K^T → scale → mask → softmax → @V) with FlashAttention-2, implemented as a new HostOp in the CUDA backend. This should reduce per-frame GPU kernel time from ~185ms to ~100-120ms.

## Architecture Overview

Luminal's compilation pipeline:
1. **Model code** creates a computation graph using `GraphTensor` ops (matmul, softmax, etc.)
2. **egglog** optimizes the graph (pattern matching → e.g., Mul+Sum → cuBLASLt)
3. **LLIR extraction** converts egglog output to `LLIROp` nodes
4. **CudaRuntime::load_llir()** compiles `LLIROp` nodes into executable `HostOp`s
5. **Runtime execution** dispatches `HostOp::execute()` calls

FlashAttention follows the **exact same pattern** as cuBLASLt:
- egglog rule matches the attention pattern in the IR
- Creates a `(flash_attention ...)` egglog term
- `extract()` converts it to a `FlashAttentionOp` HostOp
- Runtime calls `FlashAttentionOp::execute()` which launches the FA2 kernel

## Reference Implementation Pattern

Study these files carefully — FlashAttentionOp should follow the cuBLASLt pattern:

| File | What to learn |
|------|---------------|
| `crates/luminal_cuda/src/host/cublaslt/mod.rs` | Full HostOp + EgglogOp implementation |
| `crates/luminal_cuda/src/host/cublaslt/cublaslt_RmRm_rewrite.egg` | egglog rewrite rule syntax |
| `crates/luminal_cuda/src/host/mod.rs` | HostOp trait definition + ops tuple registration |
| `crates/luminal_cuda/src/host/cublas/mod.rs` | Simpler HostOp example |

External references (at `/Users/olety/Desktop/code/refs-voice/`):
| File | What to learn |
|------|---------------|
| `moshi/rust/moshi-core/src/transformer.rs:387-493` | Rust flash-attn FFI pattern (candle) |
| `flashinfer/include/flashinfer/attention/decode.cuh` | FA2 C++ kernel structure |
| `flashinfer/csrc/single_decode.cu` | FA2 CUDA binding structure |
| `triton/` | Triton kernel language reference |

## Current Attention Code (what we're replacing)

### Prefill attention (`model.rs:509-522`)
```rust
// After repeat_kv_heads(k/v, kv_groups) — heads already expanded
let scores = q.matmul(k.transpose(2, 3)) * (1.0 / (config.head_dim as f32).sqrt());
let (batch, heads, seq, _) = scores.dims4();
let causal_mask = scores.graph().tril(seq, 0)
    .expand_dim(0, batch).expand_dim(1, heads);
let masked_scores = scores.cond(
    causal_mask,
    scores.graph().constant_float(-1e9).expand_rhs(scores.shape),
);
let probs = masked_scores.softmax(3);
let context = probs.matmul(v).transpose(1, 2).merge_dims(2, 3);
```

### Decode attention (`model.rs:615-620`)
```rust
// Single-token query, full KV history, with explicit attention mask
let scores = q.matmul(k_exp.transpose(2, 3)) * (1.0 / (config.head_dim as f32).sqrt());
let (_, heads, _, _) = scores.dims4();
let expanded_mask = attn_mask.squeeze(1).expand_dim(1, heads);
let probs = (scores + expanded_mask).softmax(3);
let context = probs.matmul(v_exp).transpose(1, 2).merge_dims(2, 3);
```

### Model dimensions

| Model | Layers | Heads (Q) | KV Heads | Head Dim | kv_groups |
|-------|--------|-----------|----------|----------|-----------|
| Talker | 28 | 16 | 8 | 128 | 2 |
| Code Predictor | 5 | 16 | 8 | 128 | 2 |
| Speech Decoder | 8 | 16 | 16 | 64 | 1 |

### How softmax decomposes in Luminal IR (`src/frontend/unary.rs:132-141`)
```rust
pub fn softmax(self, axes: impl ToAxes) -> GraphTensor {
    let m = self - self.max(axes).expand_to_shape_on_axes(self.shape, axes);
    let exp = m.exp();
    exp / exp.sum(axes).expand_to_shape_on_axes(self.shape, axes)
}
```
This creates: Max → Sub → Exp → Sum → Div (5 IR nodes).

## Implementation Plan

### Step 1: Triton FA2 Kernel + AOT Compilation

Create `scripts/compile_flash_attn.py`:
- Write a Triton FA2 forward kernel supporting:
  - Causal masking (for prefill)
  - Non-causal with additive mask (for decode with explicit mask)
  - Head dims: 64 and 128
  - FP32 inputs (Luminal currently uses FP32 throughout)
  - GQA: NOT handled in the kernel — KV heads are already expanded by `repeat_kv_heads` before attention
- AOT compile to `.cubin` files for sm_80 (A100) and sm_89 (4090)
- Output files to `crates/luminal_cuda/kernels/`:
  - `flash_attn_causal_h64_sm80.cubin`
  - `flash_attn_causal_h128_sm80.cubin`
  - `flash_attn_causal_h64_sm89.cubin`
  - `flash_attn_causal_h128_sm89.cubin`

**Triton kernel signature:**
```python
@triton.jit
def flash_attn_fwd(
    Q, K, V, Out,
    softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    # Optional additive mask (for decode path):
    Mask,  # pointer, can be null
    stride_mask_b, stride_mask_h, stride_mask_m, stride_mask_n,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    N_CTX_Q, N_CTX_K,
    num_heads, num_kv_heads,
):
    ...
```

**AOT compilation:**
```python
import triton

# Compile for each config
for head_dim in [64, 128]:
    for sm in [80, 89]:
        compiled = triton.compile(
            flash_attn_fwd,
            signature=...,
            constants={
                'IS_CAUSAL': True,
                'HAS_MASK': False,
                'BLOCK_M': 128,
                'BLOCK_N': 64,
                'BLOCK_DMODEL': head_dim,
            },
            num_warps=4 if head_dim == 64 else 8,
            num_stages=3,
        )
        with open(f'crates/luminal_cuda/kernels/flash_attn_causal_h{head_dim}_sm{sm}.cubin', 'wb') as f:
            f.write(compiled.asm['cubin'])
```

### Step 2: FlashAttentionOp HostOp

Create `crates/luminal_cuda/src/host/flash_attn/mod.rs`:

```rust
#[derive(Debug)]
pub struct FlashAttentionOp {
    // Dimensions (from egglog extraction, resolved at runtime from dyn_map)
    batch: Expression,
    num_heads: Expression,
    seq_len_q: Expression,
    seq_len_k: Expression,
    head_dim: Expression,
    softmax_scale: f32,
    causal: bool,
    has_mask: bool,  // true for decode path with additive mask
    // CUDA module loaded from .cubin
    module: OnceLock<Arc<CudaModule>>,
    function: OnceLock<CudaFunction>,
}
```

**Key implementation details:**

1. **Module loading**: In `execute()` (or `prepare_for_capture()`), detect GPU SM version, load the appropriate `.cubin` file via `cuModuleLoadData`, get the kernel function.

2. **execute()**: Resolve dimension values from `dyn_map`, compute strides, build kernel params, call `cuLaunchKernel`.

3. **execute_for_capture_raw()**: Same as execute but uses raw `u64` pointers from the `raw_buffers` map instead of `CudaSlice::device_ptr()` (which is not capture-safe — see B2 learnings).

4. **output_size()**: Returns `batch * num_heads * seq_len_q * head_dim` (the output tensor size).

5. **stats_name()**: Return `Some("FlashAttn2")`.

6. **Inputs**: The op takes 3 inputs (Q, K, V) or 4 inputs (Q, K, V, Mask).

**EgglogOp implementation:**
```rust
impl EgglogOp for FlashAttentionOp {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "flash_attention".to_string(),
            vec![
                Input, Input, Input,  // Q, K, V
                Expr, Expr, Expr, Expr, Expr,  // batch, num_heads, seq_q, seq_k, head_dim
                Str,  // "causal" or "masked"
                Dty,  // dtype
            ],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            include_str!["flash_attn_causal_rewrite.egg"].to_string(),
        ]
    }

    fn extract<'a>(&'a self, egraph, children, list_cache, expr_cache) -> (LLIROp, Vec<&'a ENodeId>) {
        // Extract dimensions from egglog children
        // Create FlashAttentionOp with extracted params
        // Return (LLIROp::new::<dyn HostOp>(Box::new(op)), input_node_ids)
    }
}
```

### Step 3: egglog Rewrite Rule

Create `crates/luminal_cuda/src/host/flash_attn/flash_attn_causal_rewrite.egg`:

This is the hardest part. The rule must match the full decomposed attention pattern:

```
Q@K^T:     Mul(q_expanded, k_transposed) → Sum(reduce_k) = scores
Scale:     Mul(scores, scale_const) = scaled
Mask:      Cond(tril_mask, scaled, neg_inf) = masked   [causal path]
  OR:      Sum(scores, additive_mask) = masked          [decode path with mask]
Softmax:   Max(masked) → Sub → Exp → Sum → Div = probs
probs@V:   Mul(probs_expanded, v_expanded) → Sum(reduce_k) = output
```

**Strategy**: Since this pattern is complex (~12 nodes), consider breaking it into stages:

1. First, match the Q@K^T matmul + scale + mask + softmax + @V pattern
2. Verify the stride/shape constraints match attention geometry
3. The rule must have **higher priority than cuBLASLt rules** (otherwise cuBLASLt matches the two matmuls first, fragmenting the pattern)

**Important**: Look at how egglog rule priority works in Luminal. The rules run during equality saturation — all rules fire simultaneously and the extractor picks the best. So the FA rule doesn't need to "beat" cuBLASLt — it just needs to offer an alternative that the extractor prefers (lower cost).

Set the cost of `flash_attention` lower than the sum of costs of `cublaslt + block_ops + cublaslt` to ensure the extractor picks FA.

**Alternative if pattern matching is too complex**: Instead of matching the decomposed pattern, add a `flash_attention_marker` op to the Luminal frontend that the model code uses explicitly. The egglog rule then trivially matches this marker and converts to FlashAttentionOp. This is less general but much simpler for Phase 1. See "Fallback approach" below.

### Step 4: Register in host/mod.rs

```rust
// crates/luminal_cuda/src/host/mod.rs
mod cublas;
mod cublaslt;
mod flash_attn;  // NEW

pub type Ops = (cublaslt::CuBlasLt, cublas::CuBlasSgemmV2, flash_attn::FlashAttentionOp);
```

### Step 5: .cubin files in build

Either:
- **Option A**: Check compiled .cubin files into `crates/luminal_cuda/kernels/` and load via `include_bytes!` at compile time
- **Option B**: Load from filesystem at runtime (need to ship .cubin files alongside binary)

Option A is simpler for deployment. The .cubin files should be small (~10-50KB each).

## Fallback Approach (if egglog pattern matching is too complex)

If matching the full decomposed attention pattern in egglog proves infeasible:

1. Add a `flash_attention` method to `GraphTensor` in `src/frontend/`:
   ```rust
   impl GraphTensor {
       /// Flash attention: Q @ softmax(Q @ K^T / sqrt(d)) @ V
       /// Q: [B, H, S_q, D], K: [B, H, S_k, D], V: [B, H, S_k, D]
       pub fn flash_attention(self, k: GraphTensor, v: GraphTensor, causal: bool) -> GraphTensor {
           // Create a special IR node that the CUDA backend recognizes
           // This node takes Q, K, V as inputs and produces the attention output
           // The exact IR representation depends on Luminal's op system
           todo!("implement based on how custom ops work in Luminal's graph")
       }
   }
   ```

2. Modify model.rs to use it:
   ```rust
   // Replace lines 509-522 with:
   let context = q.flash_attention(k, v, true).transpose(1, 2).merge_dims(2, 3);
   ```

3. The egglog rule trivially matches the custom node and converts to FlashAttentionOp.

**Important**: Investigate how Luminal's graph IR handles custom ops. Look at:
- `src/op.rs` — Op enum/trait definitions
- `src/graph.rs` — how ops are stored in the graph
- How cuBLASLt nodes are created during egglog extraction vs how they could be created at graph construction time

## Testing

1. **Unit test**: Create a test that builds a simple attention graph, compiles with FA, and verifies output matches the decomposed attention (within FP32 tolerance).
   - File: `crates/luminal_cuda/tests/flash_attn_test.rs` or add to existing tests
   - Test shapes: B=1, H=16, S_q=128, S_k=128, D=128 (matches Talker config)

2. **Integration test**: Run `cargo test -p qwen3_tts` — existing tests should still pass.

3. **Modal validation**: `modal run scripts/modal_test_tts.py` — verify generate_frames produces valid output.

4. **Performance**: `modal run scripts/modal_test_tts.py --target replay` with profiling — measure per-frame timing improvement.

## Constraints

- **FP32**: Luminal currently uses FP32 throughout. The FA kernel must support FP32 (not just FP16/BF16). This is unusual — most FA implementations are FP16/BF16 only. If Triton FA2 doesn't support FP32 well, consider:
  - Cast to BF16 before FA, cast back to FP32 after (acceptable precision loss for inference)
  - Use FP32 accumulation with BF16 inputs (common in FA implementations)

- **B2 capture compatibility**: The HostOp MUST work with B2 whole-stream capture. This means:
  - No internal CUDA memory allocations during execute/execute_for_capture_raw
  - All workspace buffers allocated during prepare_for_capture
  - Use raw device pointers (u64), not CudaSlice::device_ptr() during capture

- **head_dim**: Must support both 64 (speech decoder) and 128 (talker, predictor)

- **GQA**: NOT handled in the FA kernel for Phase 1. KV heads are expanded by `repeat_kv_heads` before attention. Phase 2 can optimize this.

## Files to Create

| File | Description |
|------|-------------|
| `scripts/compile_flash_attn.py` | Triton FA2 kernel + AOT compilation |
| `crates/luminal_cuda/kernels/` | Directory for compiled .cubin files |
| `crates/luminal_cuda/src/host/flash_attn/mod.rs` | FlashAttentionOp HostOp + EgglogOp |
| `crates/luminal_cuda/src/host/flash_attn/flash_attn_causal_rewrite.egg` | egglog rewrite rule |

## Files to Modify

| File | Change |
|------|--------|
| `crates/luminal_cuda/src/host/mod.rs` | Add `mod flash_attn` + register in Ops tuple |
| `crates/luminal_cuda/Cargo.toml` | Any new dependencies (if needed) |

## Success Criteria

1. `cargo build -p luminal_cuda` compiles without errors
2. FA2 kernel loads and executes correctly on A100 (sm_80)
3. Attention output matches decomposed attention within 1e-3 tolerance (FP32)
4. B2 whole-stream capture still works with FA in the graph
5. generate_frames produces valid audio on Modal
6. Per-frame decode_exec drops from ~91ms to ~50-60ms
