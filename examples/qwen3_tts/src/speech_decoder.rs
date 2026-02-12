use luminal::{graph::Graph, prelude::GraphTensor};
use luminal_nn::ConvND;

pub const SPEECH_DECODER_CODEBOOK_DIM: usize = 512;
pub const SPEECH_DECODER_LATENT_DIM: usize = 1024;
pub const SPEECH_DECODER_HIDDEN_SIZE: usize = 512;
pub const SPEECH_DECODER_NUM_ATTENTION_HEADS: usize = 16;
pub const SPEECH_DECODER_NUM_KEY_VALUE_HEADS: usize = 16;
pub const SPEECH_DECODER_INTERMEDIATE_SIZE: usize = 1024;
pub const SPEECH_DECODER_NUM_HIDDEN_LAYERS: usize = 8;
pub const SPEECH_DECODER_RMS_NORM_EPS: f32 = 1e-5;
pub const SPEECH_DECODER_ROPE_THETA: f32 = 10_000.0;
pub const SPEECH_DECODER_SLIDING_WINDOW: usize = 72;
pub const SPEECH_DECODER_HEAD_DIM: usize = 32;
pub const SPEECH_DECODER_SEMANTIC_CODEBOOK_SIZE: usize = 4096;
pub const SPEECH_DECODER_ACOUSTIC_CODEBOOK_SIZE: usize = 2048;
pub const SPEECH_DECODER_NUM_SEMANTIC_QUANTIZERS: usize = 1;
pub const SPEECH_DECODER_NUM_ACOUSTIC_QUANTIZERS: usize = 15;
pub const SPEECH_DECODER_DECODER_DIM: usize = 1536;
pub const SPEECH_DECODER_LAYER_SCALE_INITIAL: f32 = 0.01;

#[derive(Clone, Debug)]
pub struct SpeechDecoderConfig {
    pub codebook_dim: usize,
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

pub struct SplitResidualVectorQuantizer {
    pub semantic_codebook: GraphTensor,
    pub acoustic_codebooks: Vec<GraphTensor>,
}

impl SplitResidualVectorQuantizer {
    pub fn new(cx: &mut Graph, config: &SpeechDecoderConfig) -> Self {
        assert_eq!(
            config.num_semantic_quantizers, 1,
            "Only one semantic quantizer is supported in this scaffold"
        );

        let semantic_codebook = cx.named_tensor(
            "decoder.quantizer.rvq_first.layers.0._codebook.embed",
            (config.semantic_codebook_size, config.codebook_dim),
        );

        let mut acoustic_codebooks = Vec::with_capacity(config.num_acoustic_quantizers);
        for i in 0..config.num_acoustic_quantizers {
            acoustic_codebooks.push(cx.named_tensor(
                format!("decoder.quantizer.rvq_rest.layers.{i}._codebook.embed"),
                (config.acoustic_codebook_size, config.codebook_dim),
            ));
        }

        Self {
            semantic_codebook,
            acoustic_codebooks,
        }
    }

    pub fn decode(&self, codes: Vec<GraphTensor>) -> GraphTensor {
        let expected = 1 + self.acoustic_codebooks.len();
        assert_eq!(
            codes.len(),
            expected,
            "Expected {expected} code tensors (1 semantic + {} acoustic), got {}",
            self.acoustic_codebooks.len(),
            codes.len()
        );

        let semantic_codes = codes[0];
        let (batch, seq) = semantic_codes.dims2();
        let codebook_dim = self.semantic_codebook.dims()[1];

        let mut decoded = self.semantic_codebook.gather(
            (semantic_codes * codebook_dim).expand_dim(2, codebook_dim)
                + semantic_codes
                    .graph()
                    .arange(codebook_dim)
                    .expand_lhs([batch, seq]),
        );

        for (code_ids, codebook) in codes
            .iter()
            .copied()
            .skip(1)
            .zip(self.acoustic_codebooks.iter().copied())
        {
            let embedded = codebook.gather(
                (code_ids * codebook_dim).expand_dim(2, codebook_dim)
                    + code_ids
                        .graph()
                        .arange(codebook_dim)
                        .expand_lhs([batch, seq]),
            );
            decoded = decoded + embedded;
        }

        decoded
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
        let alpha = self.alpha.expand_dim(0, batch).expand_dim(2, time);
        let beta = self.beta.expand_dim(0, batch).expand_dim(2, time);

        let sin_val = (x * alpha).sin();
        x + (sin_val * sin_val) / (beta + 1e-6)
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

pub struct SpeechDecoder {
    pub quantizer: SplitResidualVectorQuantizer,
    pub pre_conv: CausalConv1d,
    pub config: SpeechDecoderConfig,
}

impl SpeechDecoder {
    pub fn new(cx: &mut Graph, config: SpeechDecoderConfig) -> Self {
        assert_eq!(
            config.head_dim,
            config.hidden_size / config.num_attention_heads,
            "head_dim must equal hidden_size / num_attention_heads"
        );

        let quantizer = SplitResidualVectorQuantizer::new(cx, &config);
        let pre_conv = CausalConv1d::new(config.codebook_dim, config.latent_dim, 3, 1, true, cx);

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
            config,
        }
    }

    /// Decode RVQ codes to continuous latent, then apply the pre-convolution stage.
    pub fn decode_codes(&self, codes: Vec<GraphTensor>) -> GraphTensor {
        let latent = self.quantizer.decode(codes);
        let latent_t = latent.transpose(1, 2);
        self.pre_conv.forward(latent_t)
    }
}
