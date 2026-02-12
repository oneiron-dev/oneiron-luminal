use luminal::{
    graph::Graph,
    op::DType,
    prelude::{F32Pow, GraphTensor},
};
use luminal_nn::LayerNorm;

pub const CODE_PREDICTOR_HIDDEN: usize = 1024;
pub const CODE_PREDICTOR_HEAD_DIM: usize = 128;
pub const CODE_PREDICTOR_N_HEADS: usize = 16;
pub const CODE_PREDICTOR_N_KV_HEADS: usize = 8;
pub const CODE_PREDICTOR_KV_GROUPS: usize = 2;
pub const CODE_PREDICTOR_INTERMEDIATE: usize = 3072;
pub const CODE_PREDICTOR_LAYERS: usize = 5;
pub const CODE_PREDICTOR_CODEBOOK_VOCAB: usize = 2048;
pub const CODE_PREDICTOR_NUM_CODE_GROUPS: usize = 16;
pub const CODE_PREDICTOR_RMS_NORM_EPS: f32 = 1e-6;
pub const CODE_PREDICTOR_ROPE_THETA: f32 = 1_000_000.0;
pub const CODE_PREDICTOR_TALKER_HIDDEN: usize = 2048;

#[derive(Clone, Debug)]
pub struct CodePredictorConfig {
    pub hidden: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub kv_groups: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub codebook_vocab: usize,
    pub num_code_groups: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub talker_hidden: usize,
}

impl Default for CodePredictorConfig {
    fn default() -> Self {
        Self {
            hidden: CODE_PREDICTOR_HIDDEN,
            head_dim: CODE_PREDICTOR_HEAD_DIM,
            n_heads: CODE_PREDICTOR_N_HEADS,
            n_kv_heads: CODE_PREDICTOR_N_KV_HEADS,
            kv_groups: CODE_PREDICTOR_KV_GROUPS,
            intermediate: CODE_PREDICTOR_INTERMEDIATE,
            layers: CODE_PREDICTOR_LAYERS,
            codebook_vocab: CODE_PREDICTOR_CODEBOOK_VOCAB,
            num_code_groups: CODE_PREDICTOR_NUM_CODE_GROUPS,
            rms_norm_eps: CODE_PREDICTOR_RMS_NORM_EPS,
            rope_theta: CODE_PREDICTOR_ROPE_THETA,
            talker_hidden: CODE_PREDICTOR_TALKER_HIDDEN,
        }
    }
}

pub struct CodePredictorLayer {
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

pub struct CodePredictorModel {
    pub codec_embeddings: Vec<GraphTensor>,
    pub projection_weight: GraphTensor,
    pub projection_bias: GraphTensor,
    pub layers: Vec<CodePredictorLayer>,
    pub final_norm: LayerNorm,
    pub lm_heads: Vec<GraphTensor>,
    pub config: CodePredictorConfig,
}

pub struct PredictorCodeOutput {
    pub embeds: Vec<GraphTensor>,
}

impl CodePredictorModel {
    pub fn new(cx: &mut Graph, config: CodePredictorConfig) -> Self {
        assert_eq!(
            config.n_heads,
            config.n_kv_heads * config.kv_groups,
            "n_heads must equal n_kv_heads * kv_groups"
        );
        assert!(
            config.num_code_groups >= 2,
            "num_code_groups must be at least 2"
        );

        let generated_groups = config.num_code_groups - 1;

        let mut codec_embeddings = Vec::with_capacity(generated_groups);
        for i in 0..generated_groups {
            codec_embeddings.push(cx.named_tensor(
                format!("talker.code_predictor.model.codec_embedding.{i}.weight"),
                (config.codebook_vocab, config.talker_hidden),
            ));
        }

        let projection_weight = cx.named_tensor(
            "talker.code_predictor.small_to_mtp_projection.weight",
            (config.hidden, config.talker_hidden),
        );
        let projection_bias = cx.named_tensor(
            "talker.code_predictor.small_to_mtp_projection.bias",
            (config.hidden,),
        );

        let mut layers = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            layers.push(CodePredictorLayer {
                q_proj: cx.named_tensor(
                    format!("talker.code_predictor.model.layers.{i}.self_attn.q_proj.weight"),
                    (config.n_heads * config.head_dim, config.hidden),
                ),
                k_proj: cx.named_tensor(
                    format!("talker.code_predictor.model.layers.{i}.self_attn.k_proj.weight"),
                    (config.n_kv_heads * config.head_dim, config.hidden),
                ),
                v_proj: cx.named_tensor(
                    format!("talker.code_predictor.model.layers.{i}.self_attn.v_proj.weight"),
                    (config.n_kv_heads * config.head_dim, config.hidden),
                ),
                o_proj: cx.named_tensor(
                    format!("talker.code_predictor.model.layers.{i}.self_attn.o_proj.weight"),
                    (config.hidden, config.n_heads * config.head_dim),
                ),
                gate_proj: cx.named_tensor(
                    format!("talker.code_predictor.model.layers.{i}.mlp.gate_proj.weight"),
                    (config.intermediate, config.hidden),
                ),
                up_proj: cx.named_tensor(
                    format!("talker.code_predictor.model.layers.{i}.mlp.up_proj.weight"),
                    (config.intermediate, config.hidden),
                ),
                down_proj: cx.named_tensor(
                    format!("talker.code_predictor.model.layers.{i}.mlp.down_proj.weight"),
                    (config.hidden, config.intermediate),
                ),
                input_norm: LayerNorm::new(
                    config.hidden,
                    Some(&format!(
                        "talker.code_predictor.model.layers.{i}.input_layernorm.weight"
                    )),
                    None,
                    false,
                    config.rms_norm_eps,
                    cx,
                ),
                post_attn_norm: LayerNorm::new(
                    config.hidden,
                    Some(&format!(
                        "talker.code_predictor.model.layers.{i}.post_attention_layernorm.weight"
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
            Some("talker.code_predictor.model.norm.weight"),
            None,
            false,
            config.rms_norm_eps,
            cx,
        );

        let mut lm_heads = Vec::with_capacity(generated_groups);
        for i in 0..generated_groups {
            lm_heads.push(cx.named_tensor(
                format!("talker.code_predictor.lm_head.{i}.weight"),
                (config.codebook_vocab, config.hidden),
            ));
        }

        Self {
            codec_embeddings,
            projection_weight,
            projection_bias,
            layers,
            final_norm,
            lm_heads,
            config,
        }
    }

    pub fn forward(&self, input_embeds: GraphTensor) -> GraphTensor {
        let input_dims = input_embeds.dims();
        let mut x = input_embeds.matmul(self.projection_weight.t())
            + self
                .projection_bias
                .expand_lhs(&input_dims[..input_dims.len().saturating_sub(1)]);

        for layer in &self.layers {
            x = layer.forward(x, &self.config);
        }

        self.final_norm.forward(x)
    }

    /// Apply the LM head for a specific code group to hidden states.
    /// `group` is 0-indexed (0 = code group 1, ..., 14 = code group 15).
    pub fn logits_for_group(&self, hidden: GraphTensor, group: usize) -> GraphTensor {
        assert!(group < self.lm_heads.len(), "group index out of bounds");
        hidden.matmul(self.lm_heads[group].t())
    }

    /// Embed codec token IDs using the embedding table for a specific code group.
    /// `group` is 0-indexed (0 = code group 1, ..., 14 = code group 15).
    /// Returns embeddings in talker_hidden dimension (NOT predictor_hidden).
    pub fn embed_for_group(&self, code_ids: GraphTensor, group: usize) -> GraphTensor {
        assert!(
            group < self.codec_embeddings.len(),
            "group index out of bounds"
        );
        let (batch, seq) = code_ids.dims2();
        let dim = self.config.talker_hidden;
        self.codec_embeddings[group].gather(
            (code_ids * dim).expand_dim(2, dim)
                + code_ids.graph().arange(dim).expand_lhs([batch, seq]),
        )
    }

    /// Generate all predictor code embeddings by statically unrolling
    /// autoregressive predictor steps (code groups 1..num_code_groups-1).
    pub fn generate_codes(
        &self,
        talker_hidden: GraphTensor,
        code_0_embed: GraphTensor,
    ) -> PredictorCodeOutput {
        let groups = self.config.num_code_groups - 1;
        let mut embeds = Vec::with_capacity(groups);
        let mut seq = talker_hidden.concat_along(code_0_embed, 1);

        for group in 0..groups {
            let hidden = self.forward(seq);
            let logits = self.logits_for_group(hidden, group);
            let last_logits = logits.slice((.., (group + 1).., ..));
            let code = last_logits.argmax(2);
            let embed = self.embed_for_group(code, group);
            embeds.push(embed);
            if group + 1 < groups {
                seq = seq.concat_along(embed, 1);
            }
        }

        PredictorCodeOutput { embeds }
    }
}

fn apply_rope(input: GraphTensor, config: &CodePredictorConfig) -> GraphTensor {
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

impl CodePredictorLayer {
    pub fn forward(&self, mut x: GraphTensor, config: &CodePredictorConfig) -> GraphTensor {
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
