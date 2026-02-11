use luminal::{
    graph::Graph,
    op::DType,
    prelude::{F32Pow, GraphTensor},
};
use luminal_nn::LayerNorm;

pub const LAYERS: usize = 28;
pub const HIDDEN: usize = 2048;
pub const HEAD_DIM: usize = 128;
pub const N_HEADS: usize = 16;
pub const N_KV_HEADS: usize = 8;
pub const KV_GROUPS: usize = 2;
pub const INTERMEDIATE: usize = 6144;
pub const VOCAB_SIZE: usize = 3072;
pub const RMS_NORM_EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 1_000_000.0;

#[derive(Clone, Debug)]
pub struct TalkerConfig {
    pub layers: usize,
    pub hidden: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub kv_groups: usize,
    pub intermediate: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            layers: LAYERS,
            hidden: HIDDEN,
            head_dim: HEAD_DIM,
            n_heads: N_HEADS,
            n_kv_heads: N_KV_HEADS,
            kv_groups: KV_GROUPS,
            intermediate: INTERMEDIATE,
            vocab_size: VOCAB_SIZE,
            rms_norm_eps: RMS_NORM_EPS,
            rope_theta: ROPE_THETA,
        }
    }
}

pub struct TalkerModel {
    pub embedding: GraphTensor,
    pub layers: Vec<TalkerLayer>,
    pub final_norm: LayerNorm,
    pub config: TalkerConfig,
}

pub struct TalkerLayer {
    pub q_proj: GraphTensor,
    pub k_proj: GraphTensor,
    pub v_proj: GraphTensor,
    pub o_proj: GraphTensor,
    pub gate_proj: GraphTensor,
    pub up_proj: GraphTensor,
    pub down_proj: GraphTensor,
    pub input_norm: LayerNorm,
    pub post_attn_norm: LayerNorm,
}

impl TalkerModel {
    pub fn new(cx: &mut Graph, config: TalkerConfig) -> Self {
        assert_eq!(
            config.hidden,
            config.n_heads * config.head_dim,
            "hidden must equal n_heads * head_dim"
        );
        assert_eq!(
            config.n_heads,
            config.n_kv_heads * config.kv_groups,
            "n_heads must equal n_kv_heads * kv_groups"
        );

        let mut layers = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            layers.push(TalkerLayer {
                q_proj: cx.named_tensor(
                    format!("talker.layers.{i}.self_attn.q_proj.weight"),
                    (config.n_heads * config.head_dim, config.hidden),
                ),
                k_proj: cx.named_tensor(
                    format!("talker.layers.{i}.self_attn.k_proj.weight"),
                    (config.n_kv_heads * config.head_dim, config.hidden),
                ),
                v_proj: cx.named_tensor(
                    format!("talker.layers.{i}.self_attn.v_proj.weight"),
                    (config.n_kv_heads * config.head_dim, config.hidden),
                ),
                o_proj: cx.named_tensor(
                    format!("talker.layers.{i}.self_attn.o_proj.weight"),
                    (config.hidden, config.n_heads * config.head_dim),
                ),
                gate_proj: cx.named_tensor(
                    format!("talker.layers.{i}.mlp.gate_proj.weight"),
                    (config.intermediate, config.hidden),
                ),
                up_proj: cx.named_tensor(
                    format!("talker.layers.{i}.mlp.up_proj.weight"),
                    (config.intermediate, config.hidden),
                ),
                down_proj: cx.named_tensor(
                    format!("talker.layers.{i}.mlp.down_proj.weight"),
                    (config.hidden, config.intermediate),
                ),
                input_norm: LayerNorm::new(
                    config.hidden,
                    Some(&format!("talker.layers.{i}.input_layernorm.weight")),
                    None,
                    false,
                    config.rms_norm_eps,
                    cx,
                ),
                post_attn_norm: LayerNorm::new(
                    config.hidden,
                    Some(&format!(
                        "talker.layers.{i}.post_attention_layernorm.weight"
                    )),
                    None,
                    false,
                    config.rms_norm_eps,
                    cx,
                ),
            });
        }

        let final_norm = LayerNorm::new(
            config.hidden,
            Some("talker.norm.weight"),
            None,
            false,
            config.rms_norm_eps,
            cx,
        );

        let embedding = cx.named_tensor(
            "talker.embed_tokens.weight",
            (config.vocab_size, config.hidden),
        );

        Self {
            embedding,
            layers,
            final_norm,
            config,
        }
    }

    pub fn embed_tokens(&self, token_ids: GraphTensor) -> GraphTensor {
        let (batch, seq) = token_ids.dims2();
        self.embedding.gather(
            (token_ids * self.config.hidden).expand_dim(2, self.config.hidden)
                + token_ids
                    .graph()
                    .arange(self.config.hidden)
                    .expand_lhs([batch, seq]),
        )
    }

    pub fn forward_hidden(&self, token_ids: GraphTensor) -> GraphTensor {
        let mut x = self.embed_tokens(token_ids);
        for layer in &self.layers {
            x = layer.forward(x, &self.config);
        }
        x
    }

    pub fn forward(&self, token_ids: GraphTensor) -> GraphTensor {
        self.final_norm
            .forward(self.forward_hidden(token_ids))
            .matmul(self.embedding.t())
    }
}

fn apply_rope(input: GraphTensor, config: &TalkerConfig) -> GraphTensor {
    let (batch, heads, seq, _) = input.dims4();
    let half_head = config.head_dim / 2;

    let freq_ids = input
        .graph()
        .arange_options(0, config.head_dim as i32, 2)
        .cast(DType::F32);
    let inv_freq = config.rope_theta.pow(-(freq_ids / config.head_dim as f32));
    let pos = input.graph().arange(seq).cast(DType::F32);
    let theta = pos.expand_dim(1, half_head) * inv_freq.expand_dim(0, seq);

    let split = input.split_dims(3, 2);
    let even = split.slice((.., .., .., .., ..1)).squeeze(4);
    let odd = split.slice((.., .., .., .., 1..)).squeeze(4);

    let cos = theta.cos().expand_dim(0, batch).expand_dim(1, heads);
    let sin = theta.sin().expand_dim(0, batch).expand_dim(1, heads);

    let even_out = even * cos - odd * sin;
    let odd_out = even * sin + odd * cos;
    even_out.concat_along(odd_out, 3)
}

fn repeat_kv_heads(kv: GraphTensor, kv_groups: usize) -> GraphTensor {
    if kv_groups <= 1 {
        return kv;
    }
    let mut expanded = kv;
    for _ in 1..kv_groups {
        expanded = expanded.concat_along(kv, 1);
    }
    expanded
}

impl TalkerLayer {
    pub fn forward(&self, mut x: GraphTensor, config: &TalkerConfig) -> GraphTensor {
        let residual = x;
        let x_attn = self.input_norm.forward(x);

        let mut q = x_attn.matmul(self.q_proj.t());
        let mut k = x_attn.matmul(self.k_proj.t());
        let mut v = x_attn.matmul(self.v_proj.t());

        q = q.split_dims(2, config.head_dim).transpose(1, 2);
        k = k.split_dims(2, config.head_dim).transpose(1, 2);
        v = v.split_dims(2, config.head_dim).transpose(1, 2);

        q = apply_rope(q, config);
        k = apply_rope(k, config);

        k = repeat_kv_heads(k, config.kv_groups);
        v = repeat_kv_heads(v, config.kv_groups);

        let scores = q.matmul(k.transpose(2, 3)) * (1.0 / (config.head_dim as f32).sqrt());
        let (batch, heads, seq, _) = scores.dims4();
        let causal_mask = scores
            .graph()
            .tril(seq, 0)
            .expand_dim(0, batch)
            .expand_dim(1, heads);
        let masked_scores = scores.cond(
            causal_mask,
            scores.graph().constant_float(-1e9).expand_rhs(scores.shape),
        );
        let probs = masked_scores.softmax(3);

        let context = probs.matmul(v).transpose(1, 2).merge_dims(2, 3);
        x = residual + context.matmul(self.o_proj.t());

        let ff_in = self.post_attn_norm.forward(x);
        let ff = (ff_in.matmul(self.gate_proj.t()).silu() * ff_in.matmul(self.up_proj.t()))
            .matmul(self.down_proj.t());
        x + ff
    }
}
