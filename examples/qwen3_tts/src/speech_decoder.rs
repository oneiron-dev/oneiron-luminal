use crate::maybe_graph_break;
use luminal::{
    graph::Graph,
    op::DType,
    prelude::{F32Pow, GraphTensor},
};
use luminal_nn::{ConvND, ConvTranspose1d, LayerNorm};

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
        groups: usize,
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
                groups,
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

/// Efficient depthwise causal conv1d that avoids luminal's grouped conv decomposition.
///
/// Standard ConvND with groups=channels creates N separate 1-channel convolutions,
/// resulting in ~17k kernel ops for channels=1024. This implementation uses a manual
/// slice-multiply-sum pattern (one op per kernel position), reducing to ~20 ops for k=7.
pub struct DepthwiseCausalConv1d {
    pub weight: GraphTensor, // (channels, kernel)
    pub bias: Option<GraphTensor>,
    pub channels: usize,
    pub kernel: usize,
    pub dilation: usize,
    pub left_pad: usize,
}

impl DepthwiseCausalConv1d {
    pub fn new(channels: usize, kernel: usize, dilation: usize, bias: bool, cx: &mut Graph) -> Self {
        let left_pad = dilation * (kernel - 1);
        Self {
            weight: cx.named_tensor("DWConvWeight", (channels, kernel)),
            bias: if bias {
                Some(cx.named_tensor("DWConvBias", channels))
            } else {
                None
            },
            channels,
            kernel,
            dilation,
            left_pad,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        use luminal::shape::Expression;

        let (batch, _channels, seq_len) = x.dims3();
        let padded = x.pad(((0, 0), (0, 0), (self.left_pad, 0)), 0.0);

        // Manual depthwise conv: for each kernel position j, slice the padded input
        // and multiply element-wise with weight[:, j], then accumulate.
        // This produces ~3*kernel ops instead of ~17k for grouped conv.
        let mut result: Option<GraphTensor> = None;
        for j in 0..self.kernel {
            let offset = Expression::from(j * self.dilation);
            // slice: padded[:, :, offset .. offset+seq_len] → [batch, channels, seq_len]
            let slice = padded.slice((.., .., offset.clone()..offset + seq_len));
            // weight[:, j:j+1] → [channels, 1], merge_dims makes it contiguous [channels]
            let w_j = self.weight.slice_along(j..j + 1, 1).merge_dims(0, 1);
            // expand to [batch, channels, seq_len] for broadcasting
            let w_j = w_j.expand_dim(0, batch).expand_dim(2, seq_len);
            let term = slice * w_j;
            result = Some(match result {
                None => term,
                Some(r) => r + term,
            });
        }

        let mut out = result.unwrap();
        if let Some(bias) = self.bias {
            let bias_expanded = bias.expand_dim(0, batch).expand_dim(2, seq_len);
            out = out + bias_expanded;
        }
        out
    }
}

pub struct CausalTransConv1d {
    pub conv: ConvTranspose1d,
    kernel: usize,
    stride: usize,
    trim_right: usize,
}

impl CausalTransConv1d {
    pub fn new(
        ch_in: usize,
        ch_out: usize,
        kernel: usize,
        stride: usize,
        bias: bool,
        cx: &mut Graph,
    ) -> Self {
        assert!(
            kernel >= stride,
            "CausalTransConv1d requires kernel >= stride (got kernel={kernel}, stride={stride})"
        );

        Self {
            conv: ConvTranspose1d::new(ch_in, ch_out, kernel, stride, 0, bias, cx),
            kernel,
            stride,
            trim_right: kernel - stride,
        }
    }

    /// Memory-efficient forward using per-kernel-position matmuls with graph breaks.
    ///
    /// Loops over kernel positions, doing one small matmul per position.
    /// Graph breaks between iterations prevent egglog from fusing all matmuls
    /// into a single 242+ GB kernel operation.
    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        use luminal::shape::Expression;

        let (_batch, _ch_in, input_len) = x.dims3();

        // 1. Upsample: interleave zeros between input samples
        let upsampled_len = (input_len - 1) * self.stride + 1;
        let upsampled = x
            .expand_dim(3, 1)
            .pad_along(0, self.stride - 1, 3, 0.0)
            .merge_dims(2, 3)
            .slice_along(..upsampled_len, 2);

        // 2. Pad for full convolution
        let conv_padding = self.kernel - 1;
        let padded = upsampled.pad(((0, 0), (0, 0), (conv_padding, conv_padding)), 0.0);
        let padded = maybe_graph_break(padded);

        // 3. Compute output length
        let out_len = upsampled_len + conv_padding; // upsampled_len + 2*padding - kernel + 1

        // 4. Loop over kernel positions with graph breaks between iterations.
        //    Weight is (ch_in, ch_out, kernel). We reverse the kernel index for
        //    convolution (cross-correlation with flipped kernel).
        let mut accum: Option<GraphTensor> = None;
        for k in 0..self.kernel {
            let reversed_k = self.kernel - 1 - k;
            let offset = Expression::from(k);
            // slice: padded[:, :, k .. k+out_len] → [batch, ch_in, out_len]
            let slice = padded.slice((.., .., offset.clone()..offset + out_len));
            // transpose for matmul: [batch, out_len, ch_in]
            let slice_t = slice.transpose(1, 2);
            // weight_k: [ch_in, ch_out, 1] → [ch_in, ch_out]
            let weight_k = self
                .conv
                .weight
                .slice_along(reversed_k..reversed_k + 1, 2)
                .merge_dims(1, 2);
            // matmul: [batch, out_len, ch_in] × [ch_in, ch_out] → [batch, out_len, ch_out]
            let partial = slice_t.matmul(weight_k);
            accum = Some(match accum {
                None => partial,
                Some(a) => maybe_graph_break(a + partial),
            });
        }

        // 6. Transpose to [batch, ch_out, out_len]
        let mut out = accum.unwrap().transpose(1, 2);

        // 7. Add bias
        if let Some(bias) = self.conv.bias {
            let out_dims = out.dims();
            out = out + bias.expand_lhs(&out_dims[..1]).expand_dim(2, out_dims[2]);
        }

        // 8. Trim right for causal padding
        if self.trim_right > 0 {
            let target_len = input_len * self.stride;
            out = out.slice((.., .., ..target_len));
        }

        out
    }
}

pub struct ConvNeXtBlock {
    pub dwconv: DepthwiseCausalConv1d,
    pub norm: LayerNorm,
    pub pwconv1_weight: GraphTensor,
    pub pwconv1_bias: GraphTensor,
    pub pwconv2_weight: GraphTensor,
    pub pwconv2_bias: GraphTensor,
    pub gamma: GraphTensor,
}

impl ConvNeXtBlock {
    pub fn new(dim: usize, prefix: impl AsRef<str>, cx: &mut Graph) -> Self {
        let prefix = prefix.as_ref();

        let dwconv = DepthwiseCausalConv1d::new(dim, 7, 1, true, cx);
        dwconv
            .weight
            .set_name(&format!("{prefix}.dwconv.conv.weight"));
        if let Some(bias) = dwconv.bias {
            bias.set_name(&format!("{prefix}.dwconv.conv.bias"));
        }

        let norm_weight = format!("{prefix}.norm.weight");
        let norm_bias = format!("{prefix}.norm.bias");
        let norm = LayerNorm::new(
            dim,
            Some(norm_weight.as_str()),
            Some(norm_bias.as_str()),
            true,
            1e-6,
            cx,
        );

        Self {
            dwconv,
            norm,
            pwconv1_weight: cx.named_tensor(format!("{prefix}.pwconv1.weight"), (4 * dim, dim)),
            pwconv1_bias: cx.named_tensor(format!("{prefix}.pwconv1.bias"), 4 * dim),
            pwconv2_weight: cx.named_tensor(format!("{prefix}.pwconv2.weight"), (dim, 4 * dim)),
            pwconv2_bias: cx.named_tensor(format!("{prefix}.pwconv2.bias"), dim),
            gamma: cx.named_tensor(format!("{prefix}.gamma"), dim),
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let residual = x;
        let h = self.dwconv.forward(x);
        let h = h.transpose(1, 2);
        let h = self.norm.forward(h);
        let dims = h.dims();
        let h = h.matmul(self.pwconv1_weight.t()) + self.pwconv1_bias.expand_lhs(&dims[..2]);
        let h = h.gelu();
        let dims = h.dims();
        let h = h.matmul(self.pwconv2_weight.t()) + self.pwconv2_bias.expand_lhs(&dims[..2]);
        let dims = h.dims();
        let h = self.gamma.expand_lhs(&dims[..2]) * h;
        let h = h.transpose(1, 2);
        residual + h
    }
}

pub struct DecoderResidualUnit {
    pub act1: SnakeBeta,
    pub conv1: CausalConv1d,
    pub act2: SnakeBeta,
    pub conv2: CausalConv1d,
}

impl DecoderResidualUnit {
    pub fn new(channels: usize, dilation: usize, prefix: impl AsRef<str>, cx: &mut Graph) -> Self {
        let prefix = prefix.as_ref();

        let conv1 = CausalConv1d::new(channels, channels, 7, dilation, true, cx, 1);
        conv1
            .conv
            .weight
            .set_name(&format!("{prefix}.conv1.conv.weight"));
        if let Some(bias) = conv1.conv.bias {
            bias.set_name(&format!("{prefix}.conv1.conv.bias"));
        }

        let conv2 = CausalConv1d::new(channels, channels, 1, 1, true, cx, 1);
        conv2
            .conv
            .weight
            .set_name(&format!("{prefix}.conv2.conv.weight"));
        if let Some(bias) = conv2.conv.bias {
            bias.set_name(&format!("{prefix}.conv2.conv.bias"));
        }

        Self {
            act1: SnakeBeta::new(
                channels,
                &format!("{prefix}.act1.alpha"),
                &format!("{prefix}.act1.beta"),
                cx,
            ),
            conv1,
            act2: SnakeBeta::new(
                channels,
                &format!("{prefix}.act2.alpha"),
                &format!("{prefix}.act2.beta"),
                cx,
            ),
            conv2,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let residual = x;
        let hidden = self.act1.forward(x);
        let hidden = self.conv1.forward(hidden);
        let hidden = self.act2.forward(hidden);
        let hidden = self.conv2.forward(hidden);
        residual + hidden
    }
}

pub struct DecoderBlock {
    pub snake: SnakeBeta,
    pub trans_conv: CausalTransConv1d,
    pub residual_units: Vec<DecoderResidualUnit>,
}

impl DecoderBlock {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        rate: usize,
        prefix: impl AsRef<str>,
        cx: &mut Graph,
    ) -> Self {
        let prefix = prefix.as_ref();
        let snake = SnakeBeta::new(
            in_dim,
            &format!("{prefix}.block.0.alpha"),
            &format!("{prefix}.block.0.beta"),
            cx,
        );

        let trans_conv = CausalTransConv1d::new(in_dim, out_dim, 2 * rate, rate, true, cx);
        trans_conv
            .conv
            .weight
            .set_name(&format!("{prefix}.block.1.conv.weight"));
        if let Some(bias) = trans_conv.conv.bias {
            bias.set_name(&format!("{prefix}.block.1.conv.bias"));
        }

        let mut residual_units = Vec::with_capacity(3);
        for (i, dilation) in [1, 3, 9].into_iter().enumerate() {
            residual_units.push(DecoderResidualUnit::new(
                out_dim,
                dilation,
                format!("{prefix}.block.{}", i + 2),
                cx,
            ));
        }

        Self {
            snake,
            trans_conv,
            residual_units,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let mut hidden = self.snake.forward(x);
        hidden = self.trans_conv.forward(hidden);
        hidden = maybe_graph_break(hidden);
        for residual_unit in &self.residual_units {
            hidden = residual_unit.forward(hidden);
            hidden = maybe_graph_break(hidden);
        }
        hidden
    }
}

pub struct WaveformDecoder {
    pub initial_upsample: Vec<(CausalTransConv1d, ConvNeXtBlock)>,
    pub initial_conv: CausalConv1d,
    pub blocks: Vec<DecoderBlock>,
    pub final_snake: SnakeBeta,
    pub final_conv: CausalConv1d,
}

impl WaveformDecoder {
    pub fn new(cx: &mut Graph, config: &SpeechDecoderConfig) -> Self {
        let mut initial_upsample = Vec::with_capacity(config.upsampling_ratios.len());
        for (i, ratio) in config.upsampling_ratios.iter().copied().enumerate() {
            let trans_conv = CausalTransConv1d::new(
                config.latent_dim,
                config.latent_dim,
                ratio,
                ratio,
                true,
                cx,
            );
            trans_conv
                .conv
                .weight
                .set_name(&format!("decoder.upsample.{i}.0.conv.weight"));
            if let Some(bias) = trans_conv.conv.bias {
                bias.set_name(&format!("decoder.upsample.{i}.0.conv.bias"));
            }

            let convnext =
                ConvNeXtBlock::new(config.latent_dim, format!("decoder.upsample.{i}.1"), cx);
            initial_upsample.push((trans_conv, convnext));
        }

        let initial_conv =
            CausalConv1d::new(config.latent_dim, config.decoder_dim, 7, 1, true, cx, 1);
        initial_conv
            .conv
            .weight
            .set_name("decoder.decoder.0.conv.weight");
        if let Some(bias) = initial_conv.conv.bias {
            bias.set_name("decoder.decoder.0.conv.bias");
        }

        let mut blocks = Vec::with_capacity(config.upsample_rates.len());
        let mut in_dim = config.decoder_dim;
        for (i, rate) in config.upsample_rates.iter().copied().enumerate() {
            let out_dim = config.decoder_dim / (1 << (i + 1));
            assert!(
                out_dim > 0,
                "decoder_dim must be large enough for decoder block {i} channel halving"
            );
            blocks.push(DecoderBlock::new(
                in_dim,
                out_dim,
                rate,
                format!("decoder.decoder.{}", i + 1),
                cx,
            ));
            in_dim = out_dim;
        }

        let final_snake_idx = config.upsample_rates.len() + 1;
        let final_conv_idx = final_snake_idx + 1;
        let final_snake = SnakeBeta::new(
            in_dim,
            &format!("decoder.decoder.{final_snake_idx}.alpha"),
            &format!("decoder.decoder.{final_snake_idx}.beta"),
            cx,
        );

        let final_conv = CausalConv1d::new(in_dim, 1, 7, 1, true, cx, 1);
        final_conv
            .conv
            .weight
            .set_name(&format!("decoder.decoder.{final_conv_idx}.conv.weight"));
        if let Some(bias) = final_conv.conv.bias {
            bias.set_name(&format!("decoder.decoder.{final_conv_idx}.conv.bias"));
        }

        Self {
            initial_upsample,
            initial_conv,
            blocks,
            final_snake,
            final_conv,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let mut hidden = x;
        for (trans_conv, convnext) in &self.initial_upsample {
            hidden = trans_conv.forward(hidden);
            hidden = maybe_graph_break(hidden);
            hidden = convnext.forward(hidden);
            hidden = maybe_graph_break(hidden);
        }

        hidden = self.initial_conv.forward(hidden);
        hidden = maybe_graph_break(hidden);
        for block in &self.blocks {
            hidden = block.forward(hidden);
            hidden = maybe_graph_break(hidden);
        }
        hidden = self.final_snake.forward(hidden);
        hidden = self.final_conv.forward(hidden);
        hidden.clip(-1.0, 1.0)
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
            hidden = maybe_graph_break(hidden);
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
    pub waveform_decoder: WaveformDecoder,
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
        let pre_conv = CausalConv1d::new(config.codebook_dim, config.latent_dim, 3, 1, true, cx, 1);
        let pre_transformer = PreTransformer::new(cx, &config);
        let waveform_decoder = WaveformDecoder::new(cx, &config);

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
            waveform_decoder,
            config,
        }
    }

    pub fn decode_codes(&self, codes: Vec<GraphTensor>) -> GraphTensor {
        let quantized = self.quantizer.decode(codes);
        let quantized = maybe_graph_break(quantized);
        let latent = self.pre_conv.forward(quantized);
        let hidden = latent.transpose(1, 2);
        let hidden = maybe_graph_break(hidden);
        let transformed = self.pre_transformer.forward(hidden, &self.config);
        let latent = transformed.transpose(1, 2);
        let latent = maybe_graph_break(latent);
        self.waveform_decoder.forward(latent)
    }
}
