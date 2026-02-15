use crate::maybe_graph_break;
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
pub const TEXT_VOCAB_SIZE: usize = 151_936;
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
    pub text_vocab_size: usize,
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
            text_vocab_size: TEXT_VOCAB_SIZE,
            rms_norm_eps: RMS_NORM_EPS,
            rope_theta: ROPE_THETA,
        }
    }
}

pub struct TextProjection {
    pub fc1_weight: GraphTensor,
    pub fc1_bias: GraphTensor,
    pub fc2_weight: GraphTensor,
    pub fc2_bias: GraphTensor,
}

impl TextProjection {
    pub fn new(hidden: usize, cx: &mut Graph) -> Self {
        Self {
            fc1_weight: cx
                .named_tensor("talker.text_projection.linear_fc1.weight", (hidden, hidden)),
            fc1_bias: cx.named_tensor("talker.text_projection.linear_fc1.bias", hidden),
            fc2_weight: cx
                .named_tensor("talker.text_projection.linear_fc2.weight", (hidden, hidden)),
            fc2_bias: cx.named_tensor("talker.text_projection.linear_fc2.bias", hidden),
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let dims = x.dims();
        let leading = &dims[..dims.len() - 1];
        let h = (x.matmul(self.fc1_weight.t()) + self.fc1_bias.expand_lhs(leading)).silu();
        h.matmul(self.fc2_weight.t()) + self.fc2_bias.expand_lhs(leading)
    }
}

pub struct TalkerModel {
    pub codec_embedding: GraphTensor,
    pub text_embedding: GraphTensor,
    pub text_projection: TextProjection,
    pub codec_head: GraphTensor,
    pub layers: Vec<TalkerLayer>,
    pub final_norm: LayerNorm,
    pub config: TalkerConfig,
}

pub struct TalkerLayer {
    pub q_proj: GraphTensor,
    pub k_proj: GraphTensor,
    pub v_proj: GraphTensor,
    pub q_norm: LayerNorm,
    pub k_norm: LayerNorm,
    pub o_proj: GraphTensor,
    pub gate_proj: GraphTensor,
    pub up_proj: GraphTensor,
    pub down_proj: GraphTensor,
    pub input_norm: LayerNorm,
    pub post_attn_norm: LayerNorm,
}

impl TalkerLayer {
    /// Create a single transformer layer with the correct weight names for layer `i`.
    pub fn new(cx: &mut Graph, config: &TalkerConfig, i: usize) -> Self {
        Self {
            q_proj: cx.named_tensor(
                format!("talker.model.layers.{i}.self_attn.q_proj.weight"),
                (config.n_heads * config.head_dim, config.hidden),
            ),
            k_proj: cx.named_tensor(
                format!("talker.model.layers.{i}.self_attn.k_proj.weight"),
                (config.n_kv_heads * config.head_dim, config.hidden),
            ),
            v_proj: cx.named_tensor(
                format!("talker.model.layers.{i}.self_attn.v_proj.weight"),
                (config.n_kv_heads * config.head_dim, config.hidden),
            ),
            q_norm: LayerNorm::new(
                config.head_dim,
                Some(&format!("talker.model.layers.{i}.self_attn.q_norm.weight")),
                None,
                false,
                config.rms_norm_eps,
                cx,
            ),
            k_norm: LayerNorm::new(
                config.head_dim,
                Some(&format!("talker.model.layers.{i}.self_attn.k_norm.weight")),
                None,
                false,
                config.rms_norm_eps,
                cx,
            ),
            o_proj: cx.named_tensor(
                format!("talker.model.layers.{i}.self_attn.o_proj.weight"),
                (config.hidden, config.n_heads * config.head_dim),
            ),
            gate_proj: cx.named_tensor(
                format!("talker.model.layers.{i}.mlp.gate_proj.weight"),
                (config.intermediate, config.hidden),
            ),
            up_proj: cx.named_tensor(
                format!("talker.model.layers.{i}.mlp.up_proj.weight"),
                (config.intermediate, config.hidden),
            ),
            down_proj: cx.named_tensor(
                format!("talker.model.layers.{i}.mlp.down_proj.weight"),
                (config.hidden, config.intermediate),
            ),
            input_norm: LayerNorm::new(
                config.hidden,
                Some(&format!("talker.model.layers.{i}.input_layernorm.weight")),
                None,
                false,
                config.rms_norm_eps,
                cx,
            ),
            post_attn_norm: LayerNorm::new(
                config.hidden,
                Some(&format!(
                    "talker.model.layers.{i}.post_attention_layernorm.weight"
                )),
                None,
                false,
                config.rms_norm_eps,
                cx,
            ),
        }
    }
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
                    format!("talker.model.layers.{i}.self_attn.q_proj.weight"),
                    (config.n_heads * config.head_dim, config.hidden),
                ),
                k_proj: cx.named_tensor(
                    format!("talker.model.layers.{i}.self_attn.k_proj.weight"),
                    (config.n_kv_heads * config.head_dim, config.hidden),
                ),
                v_proj: cx.named_tensor(
                    format!("talker.model.layers.{i}.self_attn.v_proj.weight"),
                    (config.n_kv_heads * config.head_dim, config.hidden),
                ),
                q_norm: LayerNorm::new(
                    config.head_dim,
                    Some(&format!("talker.model.layers.{i}.self_attn.q_norm.weight")),
                    None,
                    false,
                    config.rms_norm_eps,
                    cx,
                ),
                k_norm: LayerNorm::new(
                    config.head_dim,
                    Some(&format!("talker.model.layers.{i}.self_attn.k_norm.weight")),
                    None,
                    false,
                    config.rms_norm_eps,
                    cx,
                ),
                o_proj: cx.named_tensor(
                    format!("talker.model.layers.{i}.self_attn.o_proj.weight"),
                    (config.hidden, config.n_heads * config.head_dim),
                ),
                gate_proj: cx.named_tensor(
                    format!("talker.model.layers.{i}.mlp.gate_proj.weight"),
                    (config.intermediate, config.hidden),
                ),
                up_proj: cx.named_tensor(
                    format!("talker.model.layers.{i}.mlp.up_proj.weight"),
                    (config.intermediate, config.hidden),
                ),
                down_proj: cx.named_tensor(
                    format!("talker.model.layers.{i}.mlp.down_proj.weight"),
                    (config.hidden, config.intermediate),
                ),
                input_norm: LayerNorm::new(
                    config.hidden,
                    Some(&format!("talker.model.layers.{i}.input_layernorm.weight")),
                    None,
                    false,
                    config.rms_norm_eps,
                    cx,
                ),
                post_attn_norm: LayerNorm::new(
                    config.hidden,
                    Some(&format!(
                        "talker.model.layers.{i}.post_attention_layernorm.weight"
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
            Some("talker.model.norm.weight"),
            None,
            false,
            config.rms_norm_eps,
            cx,
        );

        let codec_embedding = cx.named_tensor(
            "talker.model.codec_embedding.weight",
            (config.vocab_size, config.hidden),
        );
        let text_embedding = cx.named_tensor(
            "talker.model.text_embedding.weight",
            (config.text_vocab_size, config.hidden),
        );
        let text_projection = TextProjection::new(config.hidden, cx);
        let codec_head = cx.named_tensor(
            "talker.codec_head.weight",
            (config.vocab_size, config.hidden),
        );

        Self {
            codec_embedding,
            text_embedding,
            text_projection,
            codec_head,
            layers,
            final_norm,
            config,
        }
    }

    pub fn embed_codec(&self, token_ids: GraphTensor) -> GraphTensor {
        let (batch, seq) = token_ids.dims2();
        self.codec_embedding.gather(
            (token_ids * self.config.hidden).expand_dim(2, self.config.hidden)
                + token_ids
                    .graph()
                    .arange(self.config.hidden)
                    .expand_lhs([batch, seq]),
        )
    }

    pub fn embed_tokens(&self, token_ids: GraphTensor) -> GraphTensor {
        self.embed_codec(token_ids)
    }

    pub fn embed_text(&self, text_token_ids: GraphTensor) -> GraphTensor {
        let (batch, seq) = text_token_ids.dims2();
        let embedded = self.text_embedding.gather(
            (text_token_ids * self.config.hidden).expand_dim(2, self.config.hidden)
                + text_token_ids
                    .graph()
                    .arange(self.config.hidden)
                    .expand_lhs([batch, seq]),
        );
        self.text_projection.forward(embedded)
    }

    pub fn forward_embeds(&self, embeds: GraphTensor) -> GraphTensor {
        let mut x = embeds;
        for layer in &self.layers {
            x = layer.forward(x, &self.config);
            x = maybe_graph_break(x);
        }
        x
    }

    pub fn forward_hidden(&self, token_ids: GraphTensor) -> GraphTensor {
        self.forward_embeds(self.embed_codec(token_ids))
    }

    /// Prefill pass over full prompt embeddings.
    /// Returns (logits, final_norm_hidden, per_layer_kv_cache).
    pub fn prefill(
        &self,
        embeds: GraphTensor,
    ) -> (GraphTensor, GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let mut x = embeds;
        let mut kv_caches = Vec::with_capacity(self.config.layers);
        for layer in &self.layers {
            let (out, k, v) = layer.forward_with_kv(x, &self.config);
            kv_caches.push((k, v));
            x = maybe_graph_break(out);
        }
        let normed = self.final_norm.forward(x);
        let logits = normed.matmul(self.codec_head.t());
        (logits, normed, kv_caches)
    }

    /// Single-token decode with provided KV cache tensors.
    /// Returns (logits, final_norm_hidden, updated_per_layer_kv_cache).
    pub fn decode_cached(
        &self,
        embeds: GraphTensor,
        kv_caches: &[(GraphTensor, GraphTensor)],
        pos_tensor: GraphTensor,
    ) -> (GraphTensor, GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        assert_eq!(
            kv_caches.len(),
            self.layers.len(),
            "kv cache count must match talker layer count"
        );

        let mut x = embeds;
        let mut new_caches = Vec::with_capacity(self.config.layers);
        for (layer, (k_cache, v_cache)) in self.layers.iter().zip(kv_caches.iter()) {
            let (out, k, v) = layer.forward_decode(x, *k_cache, *v_cache, &self.config, pos_tensor);
            new_caches.push((k, v));
            x = maybe_graph_break(out);
        }
        let normed = self.final_norm.forward(x);
        let logits = normed.matmul(self.codec_head.t());
        (logits, normed, new_caches)
    }

    /// Single-token decode with fixed-size KV buffers.
    /// Returns (logits, final_norm_hidden, per_layer_new_kv).
    pub fn decode_fixed(
        &self,
        embeds: GraphTensor,
        kv_bufs: &[(GraphTensor, GraphTensor)],
        attn_mask: GraphTensor,
        pos_tensor: GraphTensor,
    ) -> (GraphTensor, GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        assert_eq!(
            kv_bufs.len(),
            self.layers.len(),
            "kv buffer count must match talker layer count"
        );

        let mut x = embeds;
        let mut new_kvs = Vec::with_capacity(self.config.layers);
        for (layer, (k_buf, v_buf)) in self.layers.iter().zip(kv_bufs.iter()) {
            let (out, k_new, v_new) =
                layer.forward_decode_fixed(x, *k_buf, *v_buf, attn_mask, &self.config, pos_tensor);
            new_kvs.push((k_new, v_new));
            x = maybe_graph_break(out);
        }
        let normed = self.final_norm.forward(x);
        let logits = normed.matmul(self.codec_head.t());
        (logits, normed, new_kvs)
    }

    pub fn decode_step(&self, embeds: GraphTensor) -> (GraphTensor, GraphTensor) {
        let hidden = self.forward_embeds(embeds);
        let normed = self.final_norm.forward(hidden);
        let logits = normed.matmul(self.codec_head.t());
        (logits, normed)
    }

    pub fn forward(&self, token_ids: GraphTensor) -> GraphTensor {
        let hidden = self.final_norm.forward(self.forward_hidden(token_ids));
        hidden.matmul(self.codec_head.t())
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

    let first_half = input.slice((.., .., .., ..half_head));
    let second_half = input.slice((.., .., .., half_head..));

    let cos = theta.cos().expand_dim(0, batch).expand_dim(1, heads);
    let sin = theta.sin().expand_dim(0, batch).expand_dim(1, heads);

    let rotated_first = first_half * cos - second_half * sin;
    let rotated_second = second_half * cos + first_half * sin;
    rotated_first.concat_along(rotated_second, 3)
}

fn apply_rope_with_pos(
    input: GraphTensor,
    config: &TalkerConfig,
    pos_tensor: GraphTensor,
) -> GraphTensor {
    let (batch, heads, _seq, _) = input.dims4();
    let half_head = config.head_dim / 2;

    let freq_ids = input
        .graph()
        .arange_options(0, config.head_dim as i32, 2)
        .cast(DType::F32);
    let inv_freq = config.rope_theta.pow(-(freq_ids / config.head_dim as f32));

    let theta = pos_tensor.expand_dim(1, half_head) * inv_freq.expand_dim(0, 1);

    let first_half = input.slice((.., .., .., ..half_head));
    let second_half = input.slice((.., .., .., half_head..));

    let cos = theta.cos().expand_dim(0, batch).expand_dim(1, heads);
    let sin = theta.sin().expand_dim(0, batch).expand_dim(1, heads);

    let rotated_first = first_half * cos - second_half * sin;
    let rotated_second = second_half * cos + first_half * sin;
    rotated_first.concat_along(rotated_second, 3)
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
    /// Forward pass that also returns layer K/V cache tensors.
    /// k_cache: (1, n_kv_heads, seq, head_dim) after qk-norm + rope
    /// v_cache: (1, n_kv_heads, seq, head_dim) after projection + reshape
    pub fn forward_with_kv(
        &self,
        mut x: GraphTensor,
        config: &TalkerConfig,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let residual = x;
        let x_attn = self.input_norm.forward(x);

        let mut q = x_attn.matmul(self.q_proj.t());
        let mut k = x_attn.matmul(self.k_proj.t());
        let mut v = x_attn.matmul(self.v_proj.t());

        q = q.split_dims(2, config.head_dim).transpose(1, 2);
        k = k.split_dims(2, config.head_dim).transpose(1, 2);
        v = v.split_dims(2, config.head_dim).transpose(1, 2);

        let v_cache = v;

        q = self.q_norm.forward(q);
        k = self.k_norm.forward(k);

        q = apply_rope(q, config);
        k = apply_rope(k, config);

        let k_cache = k;

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
        x += ff;

        (x, k_cache, v_cache)
    }

    /// Single-token decode pass with incoming K/V caches.
    /// Returns (layer_output, k_cache_with_new_token, v_cache_with_new_token).
    pub fn forward_decode(
        &self,
        mut x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        config: &TalkerConfig,
        pos_tensor: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let residual = x;
        let x_attn = self.input_norm.forward(x);

        let mut q = x_attn.matmul(self.q_proj.t());
        let mut k_new = x_attn.matmul(self.k_proj.t());
        let mut v_new = x_attn.matmul(self.v_proj.t());

        q = q.split_dims(2, config.head_dim).transpose(1, 2);
        k_new = k_new.split_dims(2, config.head_dim).transpose(1, 2);
        v_new = v_new.split_dims(2, config.head_dim).transpose(1, 2);

        q = self.q_norm.forward(q);
        k_new = self.k_norm.forward(k_new);

        q = apply_rope_with_pos(q, config, pos_tensor);
        k_new = apply_rope_with_pos(k_new, config, pos_tensor);

        let k_full = k_cache.concat_along(k_new, 2);
        let v_full = v_cache.concat_along(v_new, 2);

        let k_exp = repeat_kv_heads(k_full, config.kv_groups);
        let v_exp = repeat_kv_heads(v_full, config.kv_groups);

        // Single-token query can attend to the full key history, so no causal mask is required.
        let scores = q.matmul(k_exp.transpose(2, 3)) * (1.0 / (config.head_dim as f32).sqrt());
        let probs = scores.softmax(3);

        let context = probs.matmul(v_exp).transpose(1, 2).merge_dims(2, 3);
        x = residual + context.matmul(self.o_proj.t());

        let ff_in = self.post_attn_norm.forward(x);
        let ff = (ff_in.matmul(self.gate_proj.t()).silu() * ff_in.matmul(self.up_proj.t()))
            .matmul(self.down_proj.t());
        x += ff;

        (x, k_full, v_full)
    }

    /// Single-token decode pass with fixed-size KV buffers and attention mask.
    /// Returns (layer_output, new_k, new_v).
    pub fn forward_decode_fixed(
        &self,
        mut x: GraphTensor,
        k_buf: GraphTensor,
        v_buf: GraphTensor,
        attn_mask: GraphTensor,
        config: &TalkerConfig,
        pos_tensor: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let residual = x;
        let x_attn = self.input_norm.forward(x);

        let mut q = x_attn.matmul(self.q_proj.t());
        let mut k_new = x_attn.matmul(self.k_proj.t());
        let mut v_new = x_attn.matmul(self.v_proj.t());

        q = q.split_dims(2, config.head_dim).transpose(1, 2);
        k_new = k_new.split_dims(2, config.head_dim).transpose(1, 2);
        v_new = v_new.split_dims(2, config.head_dim).transpose(1, 2);

        q = self.q_norm.forward(q);
        k_new = self.k_norm.forward(k_new);

        q = apply_rope_with_pos(q, config, pos_tensor);
        k_new = apply_rope_with_pos(k_new, config, pos_tensor);

        let k_full = k_buf.concat_along(k_new, 2);
        let v_full = v_buf.concat_along(v_new, 2);

        let k_exp = repeat_kv_heads(k_full, config.kv_groups);
        let v_exp = repeat_kv_heads(v_full, config.kv_groups);

        let scores = q.matmul(k_exp.transpose(2, 3)) * (1.0 / (config.head_dim as f32).sqrt());
        let (_, heads, _, _) = scores.dims4();
        let expanded_mask = attn_mask.squeeze(1).expand_dim(1, heads);
        let probs = (scores + expanded_mask).softmax(3);

        let context = probs.matmul(v_exp).transpose(1, 2).merge_dims(2, 3);
        x = residual + context.matmul(self.o_proj.t());

        let ff_in = self.post_attn_norm.forward(x);
        let ff = (ff_in.matmul(self.gate_proj.t()).silu() * ff_in.matmul(self.up_proj.t()))
            .matmul(self.down_proj.t());
        x += ff;

        (x, k_new, v_new)
    }

    pub fn forward(&self, mut x: GraphTensor, config: &TalkerConfig) -> GraphTensor {
        let residual = x;
        let x_attn = self.input_norm.forward(x);

        let mut q = x_attn.matmul(self.q_proj.t());
        let mut k = x_attn.matmul(self.k_proj.t());
        let mut v = x_attn.matmul(self.v_proj.t());

        q = q.split_dims(2, config.head_dim).transpose(1, 2);
        k = k.split_dims(2, config.head_dim).transpose(1, 2);
        v = v.split_dims(2, config.head_dim).transpose(1, 2);

        q = self.q_norm.forward(q);
        k = self.k_norm.forward(k);

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
