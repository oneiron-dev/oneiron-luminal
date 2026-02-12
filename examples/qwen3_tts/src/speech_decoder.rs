use luminal::{
    graph::Graph,
    op::DType,
    prelude::{F32Pow, GraphTensor},
};
use luminal_nn::{ConvND, LayerNorm};

pub const SPEECH_DECODER_CODEBOOK_DIM: usize = 512;
pub const SPEECH_DECODER_VQ_DIM: usize = 256;
pub const SPEECH_DECODER_LATENT_DIM: usize = 1024;
pub const SPEECH_DECODER_HIDDEN_SIZE: usize = 512;
pub const SPEECH_DECODER_NUM_ATTENTION_HEADS: usize = 16;
pub const SPEECH_DECODER_NUM_KEY_VALUE_HEADS: usize = 16;
pub const SPEECH_DECODER_INTERMEDIATE_SIZE: usize = 1024;
pub const SPEECH_DECODER_NUM_HIDDEN_LAYERS: usize = 8;
pub const SPEECH_DECODER_RMS_NORM_EPS: f32 = 1e-5;
pub const SPEECH_DECODER_ROPE_THETA: f32 = 10_000.0;
pub const SPEECH_DECODER_SLIDING_WINDOW: usize = 72;
pub const SPEECH_DECODER_HEAD_DIM: usize = 64;
pub const SPEECH_DECODER_SEMANTIC_CODEBOOK_SIZE: usize = 4096;
pub const SPEECH_DECODER_ACOUSTIC_CODEBOOK_SIZE: usize = 2048;
pub const SPEECH_DECODER_NUM_SEMANTIC_QUANTIZERS: usize = 1;
pub const SPEECH_DECODER_NUM_ACOUSTIC_QUANTIZERS: usize = 15;
pub const SPEECH_DECODER_DECODER_DIM: usize = 1536;
pub const SPEECH_DECODER_LAYER_SCALE_INITIAL: f32 = 0.01;

#[derive(Clone, Debug)]
pub struct SpeechDecoderConfig {
    pub codebook_dim: usize,
    pub vq_dim: usize,
    pub latent_dim: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub head_dim: usize,
    pub semantic_codebook_size: usize,
    pub acoustic_codebook_size: usize,
    pub num_semantic_quantizers: usize,
    pub num_acoustic_quantizers: usize,
    pub decoder_dim: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub layer_scale_initial: f32,
}

impl Default for SpeechDecoderConfig {
    fn default() -> Self {
        Self {
            codebook_dim: SPEECH_DECODER_CODEBOOK_DIM,
            vq_dim: SPEECH_DECODER_CODEBOOK_DIM / 2,
            latent_dim: SPEECH_DECODER_LATENT_DIM,
            hidden_size: SPEECH_DECODER_HIDDEN_SIZE,
            num_attention_heads: SPEECH_DECODER_NUM_ATTENTION_HEADS,
            num_key_value_heads: SPEECH_DECODER_NUM_KEY_VALUE_HEADS,
            intermediate_size: SPEECH_DECODER_INTERMEDIATE_SIZE,
            num_hidden_layers: SPEECH_DECODER_NUM_HIDDEN_LAYERS,
            rms_norm_eps: SPEECH_DECODER_RMS_NORM_EPS,
            rope_theta: SPEECH_DECODER_ROPE_THETA,
            sliding_window: SPEECH_DECODER_SLIDING_WINDOW,
            head_dim: SPEECH_DECODER_HEAD_DIM,
            semantic_codebook_size: SPEECH_DECODER_SEMANTIC_CODEBOOK_SIZE,
            acoustic_codebook_size: SPEECH_DECODER_ACOUSTIC_CODEBOOK_SIZE,
            num_semantic_quantizers: SPEECH_DECODER_NUM_SEMANTIC_QUANTIZERS,
            num_acoustic_quantizers: SPEECH_DECODER_NUM_ACOUSTIC_QUANTIZERS,
            decoder_dim: SPEECH_DECODER_DECODER_DIM,
            upsample_rates: vec![8, 5, 4, 3],
            upsampling_ratios: vec![2, 2],
            layer_scale_initial: SPEECH_DECODER_LAYER_SCALE_INITIAL,
        }
    }
}

pub struct EuclideanCodebook {
    pub embedding_sum: GraphTensor,
    pub cluster_usage: GraphTensor,
}

pub struct ResidualVectorQuantizer {
    pub codebooks: Vec<EuclideanCodebook>,
    pub output_proj_weight: GraphTensor,
}

impl ResidualVectorQuantizer {
    fn decode_codebook(code_ids: GraphTensor, codebook: &EuclideanCodebook) -> GraphTensor {
        let (batch, seq) = code_ids.dims2();
        let vq_dim = codebook.embedding_sum.dims()[1];
        let normalized = codebook.embedding_sum
            / codebook
                .cluster_usage
                .maximum_f32(1e-5)
                .expand_dim(1, vq_dim);

        normalized.gather(
            (code_ids * vq_dim).expand_dim(2, vq_dim)
                + code_ids.graph().arange(vq_dim).expand_lhs([batch, seq]),
        )
    }

    pub fn decode(&self, codes: Vec<GraphTensor>) -> GraphTensor {
        assert_eq!(
            codes.len(),
            self.codebooks.len(),
            "Expected {} code tensors, got {}",
            self.codebooks.len(),
            codes.len()
        );
        assert!(
            !self.codebooks.is_empty(),
            "ResidualVectorQuantizer must contain at least one codebook"
        );

        let mut decoded = Self::decode_codebook(codes[0], &self.codebooks[0]);
        for (code_ids, codebook) in codes
            .iter()
            .copied()
            .skip(1)
            .zip(self.codebooks.iter().skip(1))
        {
            decoded += Self::decode_codebook(code_ids, codebook);
        }

        let output_proj = self.output_proj_weight.merge_dims(1, 2);
        decoded.matmul(output_proj.t()).transpose(1, 2)
    }
}

pub struct SplitResidualVectorQuantizer {
    pub rvq_first: ResidualVectorQuantizer,
    pub rvq_rest: ResidualVectorQuantizer,
}

impl SplitResidualVectorQuantizer {
    pub fn new(cx: &mut Graph, config: &SpeechDecoderConfig) -> Self {
        assert_eq!(
            config.num_semantic_quantizers, 1,
            "Only one semantic quantizer is supported in this scaffold"
        );

        let rvq_first = ResidualVectorQuantizer {
            codebooks: vec![EuclideanCodebook {
                embedding_sum: cx.named_tensor(
                    "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
                    (config.semantic_codebook_size, config.vq_dim),
                ),
                cluster_usage: cx.named_tensor(
                    "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage",
                    config.semantic_codebook_size,
                ),
            }],
            output_proj_weight: cx.named_tensor(
                "decoder.quantizer.rvq_first.output_proj.weight",
                (config.codebook_dim, config.vq_dim, 1),
            ),
        };

        let mut rvq_rest_codebooks = Vec::with_capacity(config.num_acoustic_quantizers);
        for i in 0..config.num_acoustic_quantizers {
            rvq_rest_codebooks.push(EuclideanCodebook {
                embedding_sum: cx.named_tensor(
                    format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"),
                    (config.acoustic_codebook_size, config.vq_dim),
                ),
                cluster_usage: cx.named_tensor(
                    format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"),
                    config.acoustic_codebook_size,
                ),
            });
        }

        let rvq_rest = ResidualVectorQuantizer {
            codebooks: rvq_rest_codebooks,
            output_proj_weight: cx.named_tensor(
                "decoder.quantizer.rvq_rest.output_proj.weight",
                (config.codebook_dim, config.vq_dim, 1),
            ),
        };

        Self {
            rvq_first,
            rvq_rest,
        }
    }

    pub fn decode(&self, codes: Vec<GraphTensor>) -> GraphTensor {
        let first_count = self.rvq_first.codebooks.len();
        let expected = first_count + self.rvq_rest.codebooks.len();
        assert_eq!(
            codes.len(),
            expected,
            "Expected {expected} code tensors ({} semantic + {} acoustic), got {}",
            first_count,
            self.rvq_rest.codebooks.len(),
            codes.len()
        );

        let first = self.rvq_first.decode(codes[..first_count].to_vec());
        if self.rvq_rest.codebooks.is_empty() {
            first
        } else {
            first + self.rvq_rest.decode(codes[first_count..].to_vec())
        }
    }
}

pub struct SnakeBeta {
    pub alpha: GraphTensor,
    pub beta: GraphTensor,
}

impl SnakeBeta {
    pub fn new(channels: usize, alpha_name: &str, beta_name: &str, cx: &mut Graph) -> Self {
        Self {
            alpha: cx.named_tensor(alpha_name, channels),
            beta: cx.named_tensor(beta_name, channels),
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let (batch, _, time) = x.dims3();
        let alpha = self.alpha.exp().expand_dim(0, batch).expand_dim(2, time);
        let beta = self.beta.exp().expand_dim(0, batch).expand_dim(2, time);

        let sin_val = (x * alpha).sin();
        x + (sin_val * sin_val) / (beta + 1e-9)
    }
}

pub struct CausalConv1d {
    pub conv: ConvND,
    pub left_pad: usize,
}

impl CausalConv1d {
    pub fn new(
        ch_in: usize,
        ch_out: usize,
        kernel: usize,
        dilation: usize,
        bias: bool,
        cx: &mut Graph,
    ) -> Self {
        let left_pad = dilation * (kernel - 1);
        Self {
            conv: ConvND::new(
                ch_in,
                ch_out,
                vec![kernel],
                vec![1],
                vec![dilation],
                vec![0],
                bias,
                1,
                cx,
            ),
            left_pad,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let padded = x.pad(((0, 0), (0, 0), (self.left_pad, 0)), 0.0);
        self.conv.forward(padded)
    }
}

pub struct LayerScale {
    pub scale: GraphTensor,
}

impl LayerScale {
    pub fn new(name: impl AsRef<str>, hidden_size: usize, cx: &mut Graph) -> Self {
        Self {
            scale: cx.named_tensor(name.as_ref(), hidden_size),
        }
    }

    pub fn forward(&self, hidden: GraphTensor) -> GraphTensor {
        let dims = hidden.dims();
        hidden * self.scale.expand_lhs(&dims[..dims.len() - 1])
    }
}

pub struct PreTransformerLayer {
    pub input_layernorm: LayerNorm,
    pub q_proj: GraphTensor,
    pub k_proj: GraphTensor,
    pub v_proj: GraphTensor,
    pub o_proj: GraphTensor,
    pub self_attn_layer_scale: LayerScale,
    pub post_attention_layernorm: LayerNorm,
    pub gate_proj: GraphTensor,
    pub up_proj: GraphTensor,
    pub down_proj: GraphTensor,
    pub mlp_layer_scale: LayerScale,
}

pub struct PreTransformer {
    pub input_proj_weight: GraphTensor,
    pub input_proj_bias: GraphTensor,
    pub layers: Vec<PreTransformerLayer>,
    pub norm: LayerNorm,
    pub output_proj_weight: GraphTensor,
    pub output_proj_bias: GraphTensor,
}

fn decoder_apply_rope(input: GraphTensor, config: &SpeechDecoderConfig) -> GraphTensor {
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

fn apply_sliding_window_causal_mask(scores: GraphTensor, window_size: usize) -> GraphTensor {
    let (batch, heads, seq, _) = scores.dims4();
    let neg_inf = scores.graph().constant_float(-1e9).expand_rhs(scores.shape);

    let causal_mask = scores
        .graph()
        .tril(seq, 0)
        .expand_dim(0, batch)
        .expand_dim(1, heads);
    let causal_scores = scores.cond(causal_mask, neg_inf);

    let far_past_mask = scores
        .graph()
        .tril(seq, -(window_size as i32))
        .expand_dim(0, batch)
        .expand_dim(1, heads);

    neg_inf.cond(far_past_mask, causal_scores)
}

impl PreTransformerLayer {
    pub fn forward(&self, hidden_states: GraphTensor, config: &SpeechDecoderConfig) -> GraphTensor {
        let residual = hidden_states;
        let hidden = self.input_layernorm.forward(hidden_states);

        let mut q = hidden.matmul(self.q_proj.t());
        let mut k = hidden.matmul(self.k_proj.t());
        let mut v = hidden.matmul(self.v_proj.t());

        q = q.split_dims(2, config.head_dim).transpose(1, 2);
        k = k.split_dims(2, config.head_dim).transpose(1, 2);
        v = v.split_dims(2, config.head_dim).transpose(1, 2);

        q = decoder_apply_rope(q, config);
        k = decoder_apply_rope(k, config);

        let scores = q.matmul(k.transpose(2, 3)) * (1.0 / (config.head_dim as f32).sqrt());
        let masked_scores = apply_sliding_window_causal_mask(scores, config.sliding_window);
        let probs = masked_scores.softmax(3);

        let context = probs.matmul(v).transpose(1, 2).merge_dims(2, 3);
        let hidden = context.matmul(self.o_proj.t());
        let hidden = self.self_attn_layer_scale.forward(hidden);
        let hidden = residual + hidden;

        let residual = hidden;
        let hidden = self.post_attention_layernorm.forward(hidden);
        let gate = hidden.matmul(self.gate_proj.t()).silu();
        let up = hidden.matmul(self.up_proj.t());
        let hidden = (gate * up).matmul(self.down_proj.t());
        let hidden = self.mlp_layer_scale.forward(hidden);

        residual + hidden
    }
}

impl PreTransformer {
    pub fn new(cx: &mut Graph, config: &SpeechDecoderConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let attn_proj_dim = config.num_attention_heads * config.head_dim;
            layers.push(PreTransformerLayer {
                input_layernorm: LayerNorm::new(
                    config.hidden_size,
                    Some(&format!(
                        "decoder.pre_transformer.layers.{i}.input_layernorm.weight"
                    )),
                    None,
                    false,
                    config.rms_norm_eps,
                    cx,
                ),
                q_proj: cx.named_tensor(
                    format!("decoder.pre_transformer.layers.{i}.self_attn.q_proj.weight"),
                    (attn_proj_dim, config.hidden_size),
                ),
                k_proj: cx.named_tensor(
                    format!("decoder.pre_transformer.layers.{i}.self_attn.k_proj.weight"),
                    (attn_proj_dim, config.hidden_size),
                ),
                v_proj: cx.named_tensor(
                    format!("decoder.pre_transformer.layers.{i}.self_attn.v_proj.weight"),
                    (attn_proj_dim, config.hidden_size),
                ),
                o_proj: cx.named_tensor(
                    format!("decoder.pre_transformer.layers.{i}.self_attn.o_proj.weight"),
                    (config.hidden_size, attn_proj_dim),
                ),
                self_attn_layer_scale: LayerScale::new(
                    format!("decoder.pre_transformer.layers.{i}.self_attn_layer_scale.scale"),
                    config.hidden_size,
                    cx,
                ),
                post_attention_layernorm: LayerNorm::new(
                    config.hidden_size,
                    Some(&format!(
                        "decoder.pre_transformer.layers.{i}.post_attention_layernorm.weight"
                    )),
                    None,
                    false,
                    config.rms_norm_eps,
                    cx,
                ),
                gate_proj: cx.named_tensor(
                    format!("decoder.pre_transformer.layers.{i}.mlp.gate_proj.weight"),
                    (config.intermediate_size, config.hidden_size),
                ),
                up_proj: cx.named_tensor(
                    format!("decoder.pre_transformer.layers.{i}.mlp.up_proj.weight"),
                    (config.intermediate_size, config.hidden_size),
                ),
                down_proj: cx.named_tensor(
                    format!("decoder.pre_transformer.layers.{i}.mlp.down_proj.weight"),
                    (config.hidden_size, config.intermediate_size),
                ),
                mlp_layer_scale: LayerScale::new(
                    format!("decoder.pre_transformer.layers.{i}.mlp_layer_scale.scale"),
                    config.hidden_size,
                    cx,
                ),
            });
        }

        Self {
            input_proj_weight: cx.named_tensor(
                "decoder.pre_transformer.input_proj.weight",
                (config.hidden_size, config.latent_dim),
            ),
            input_proj_bias: cx.named_tensor(
                "decoder.pre_transformer.input_proj.bias",
                config.hidden_size,
            ),
            layers,
            norm: LayerNorm::new(
                config.hidden_size,
                Some("decoder.pre_transformer.norm.weight"),
                None,
                false,
                config.rms_norm_eps,
                cx,
            ),
            output_proj_weight: cx.named_tensor(
                "decoder.pre_transformer.output_proj.weight",
                (config.latent_dim, config.hidden_size),
            ),
            output_proj_bias: cx.named_tensor(
                "decoder.pre_transformer.output_proj.bias",
                config.latent_dim,
            ),
        }
    }

    pub fn forward(&self, hidden_states: GraphTensor, config: &SpeechDecoderConfig) -> GraphTensor {
        let input_dims = hidden_states.dims();
        let mut hidden = hidden_states.matmul(self.input_proj_weight.t())
            + self
                .input_proj_bias
                .expand_lhs(&input_dims[..input_dims.len().saturating_sub(1)]);

        for layer in &self.layers {
            hidden = layer.forward(hidden, config);
        }

        hidden = self.norm.forward(hidden);
        let output_dims = hidden.dims();
        hidden.matmul(self.output_proj_weight.t())
            + self
                .output_proj_bias
                .expand_lhs(&output_dims[..output_dims.len().saturating_sub(1)])
    }
}

pub struct SpeechDecoder {
    pub quantizer: SplitResidualVectorQuantizer,
    pub pre_conv: CausalConv1d,
    pub pre_transformer: PreTransformer,
    pub config: SpeechDecoderConfig,
}

impl SpeechDecoder {
    pub fn new(cx: &mut Graph, config: SpeechDecoderConfig) -> Self {
        assert_eq!(
            config.num_key_value_heads, config.num_attention_heads,
            "pre-transformer currently requires num_key_value_heads == num_attention_heads"
        );
        assert_eq!(
            config.head_dim % 2,
            0,
            "head_dim must be even for RoPE rotate-half"
        );
        assert!(config.sliding_window > 0, "sliding_window must be > 0");

        let quantizer = SplitResidualVectorQuantizer::new(cx, &config);
        let pre_conv = CausalConv1d::new(config.codebook_dim, config.latent_dim, 3, 1, true, cx);
        let pre_transformer = PreTransformer::new(cx, &config);

        pre_conv
            .conv
            .weight
            .set_name("decoder.pre_conv.conv.weight");
        if let Some(bias) = pre_conv.conv.bias {
            bias.set_name("decoder.pre_conv.conv.bias");
        }

        Self {
            quantizer,
            pre_conv,
            pre_transformer,
            config,
        }
    }

    pub fn decode_codes(&self, codes: Vec<GraphTensor>) -> GraphTensor {
        let quantized = self.quantizer.decode(codes);
        let latent = self.pre_conv.forward(quantized);
        let hidden = latent.transpose(1, 2);
        let transformed = self.pre_transformer.forward(hidden, &self.config);
        transformed.transpose(1, 2)
    }
}
