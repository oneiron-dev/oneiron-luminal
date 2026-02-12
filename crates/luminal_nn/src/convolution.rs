use luminal::prelude::*;

fn conv_forward_unfold(
    input: GraphTensor,
    weight: GraphTensor,
    ch_in: usize,
    ch_out: usize,
    kernel: &[usize],
    stride: &[usize],
    dilation: &[usize],
    padding: &[usize],
) -> GraphTensor {
    let input_dims = input.dims();
    let rank = input_dims.len();
    let spatial = kernel.len();

    assert!(
        rank > spatial,
        "Conv forward expects input rank ({rank}) to be greater than kernel dims ({spatial})",
    );

    let batch_len = rank - spatial - 1;
    assert_eq!(
        input_dims[batch_len],
        Expression::from(ch_in),
        "Input channel dimension ({}) must match ch_in ({})",
        input_dims[batch_len],
        ch_in
    );
    assert_eq!(
        weight.dims()[0],
        Expression::from(ch_out),
        "Weight output channels ({}) must match ch_out ({})",
        weight.dims()[0],
        ch_out
    );

    let reshaped_weight = if weight.dims().len() == 2 {
        // Reshape weight from [ch_out, ch_in * kernel_product] to [ch_out, ch_in, kernel...].
        let mut reshaped = weight;
        for k in kernel.iter().rev() {
            reshaped = reshaped.split_dims(1, *k);
        }
        reshaped
    } else {
        assert_eq!(
            weight.dims().len(),
            spatial + 2,
            "Shaped weight rank ({}) must equal 2 + spatial dims ({})",
            weight.dims().len(),
            spatial + 2
        );
        weight
    };
    assert_eq!(
        reshaped_weight.dims()[1],
        Expression::from(ch_in),
        "Weight channel dimension ({}) must match ch_in ({})",
        reshaped_weight.dims()[1],
        ch_in
    );

    // Pad only the spatial dimensions.
    let mut pad_spec = vec![(Expression::from(0), Expression::from(0)); rank];
    for (i, pad) in padding.iter().enumerate() {
        let axis = batch_len + 1 + i;
        pad_spec[axis] = (Expression::from(*pad), Expression::from(*pad));
    }
    let padded = input.pad(pad_spec, 0.0);

    // Build unfold parameters with ones for non-spatial axes.
    let mut kernel_shape = vec![1; rank];
    let mut stride_shape = vec![1; rank];
    let mut dilation_shape = vec![1; rank];
    for i in 0..spatial {
        let axis = batch_len + 1 + i;
        kernel_shape[axis] = kernel[i];
        stride_shape[axis] = stride[i];
        dilation_shape[axis] = dilation[i];
    }

    let unfolded = padded.unfold(kernel_shape, stride_shape, dilation_shape);

    // Reorder to [batch..., out..., channels, kernel_spatial..., kernel_batch..., kernel_channel].
    let mut order2 = Vec::with_capacity(2 * rank);
    // window batch dims
    order2.extend(0..batch_len);
    // window spatial dims (outputs)
    order2.extend(batch_len + 1..batch_len + 1 + spatial);
    // window channel dim
    order2.push(batch_len);
    // kernel spatial dims
    order2.extend(rank + batch_len + 1..rank + batch_len + 1 + spatial);
    // kernel batch dims and kernel channel dim (to be merged away)
    order2.extend(rank..rank + batch_len + 1);
    let mut patches = unfolded.permute(order2);

    // Remove kernel axes for batch + channel (they are all size 1 due kernel_shape construction).
    for _ in 0..=batch_len {
        let last = patches.dims().len() - 1;
        patches = patches.squeeze(last);
    }

    let patch_dims = patches.dims();

    // Broadcasted multiply across [channels, kernel...] then reduce those axes.
    let mut out =
        patches.expand_dim(batch_len + spatial, ch_out) * reshaped_weight.expand_lhs(&patch_dims[..batch_len + spatial]);

    for _ in 0..=spatial {
        let last_axis = out.dims().len() - 1;
        out = out.sum(last_axis);
    }

    // Move channel dimension ahead of the spatial axes: [batch..., ch_out, spatial...]
    let mut final_order: Vec<usize> = (0..batch_len).collect();
    final_order.push(batch_len + spatial);
    final_order.extend(batch_len..batch_len + spatial);
    out.permute(final_order)
}

fn add_channel_bias(mut out: GraphTensor, bias: GraphTensor, spatial_len: usize) -> GraphTensor {
    let out_dims = out.dims();
    let batch_len = out_dims.len() - spatial_len - 1;
    out += bias
        .expand_lhs(&out_dims[..batch_len])
        .expand_rhs(&out_dims[batch_len + 1..]);
    out
}

fn reverse_kernel_1d(weight: GraphTensor, kernel: usize) -> GraphTensor {
    let mut reversed = weight.slice_along(kernel - 1..kernel, 2);
    for k in (0..kernel - 1).rev() {
        reversed = reversed.concat_along(weight.slice_along(k..k + 1, 2), 2);
    }
    reversed
}

/// Generic N-dimensional convolution layer implemented with the GraphTensor `unfold` helper.
///
/// The layer expects inputs shaped like `[batch..., channels, spatial...]` where the number of
/// spatial dimensions is greater than zero. The kernel configuration controls how many spatial
/// axes are convolved (N) and must be shorter than the input rank (K): `K > N` is asserted.
pub struct ConvND {
    pub weight: GraphTensor, // (ch_out, (ch_in / groups) * kernel_product)
    pub bias: Option<GraphTensor>,
    kernel: Vec<usize>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<usize>,
    ch_in: usize,
    ch_out: usize,
    groups: usize,
}

impl ConvND {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ch_in: usize,
        ch_out: usize,
        kernel: Vec<usize>,
        stride: Vec<usize>,
        dilation: Vec<usize>,
        padding: Vec<usize>,
        bias: bool,
        groups: usize,
        cx: &mut Graph,
    ) -> Self {
        assert!(
            !kernel.is_empty(),
            "ConvND requires at least one spatial dimension in the kernel",
        );
        assert!(groups > 0, "ConvND groups must be positive");
        assert_eq!(
            ch_in % groups,
            0,
            "ConvND ch_in ({ch_in}) must be divisible by groups ({groups})",
        );
        assert_eq!(
            ch_out % groups,
            0,
            "ConvND ch_out ({ch_out}) must be divisible by groups ({groups})",
        );
        let k = kernel.len();
        assert_eq!(
            stride.len(),
            k,
            "Stride dimensions ({}) must match kernel dimensions ({k})",
            stride.len()
        );
        assert_eq!(
            dilation.len(),
            k,
            "Dilation dimensions ({}) must match kernel dimensions ({k})",
            dilation.len()
        );
        assert_eq!(
            padding.len(),
            k,
            "Padding dimensions ({}) must match kernel dimensions ({k})",
            padding.len()
        );

        let kernel_product: usize = kernel.iter().product();
        let ch_in_per_group = ch_in / groups;

        Self {
            weight: cx.named_tensor("ConvWeight", (ch_out, ch_in_per_group * kernel_product)),
            bias: if bias {
                Some(cx.named_tensor("ConvBias", ch_out))
            } else {
                None
            },
            kernel,
            stride,
            dilation,
            padding,
            ch_in,
            ch_out,
            groups,
        }
    }

    /// Apply convolution to an input shaped `[batch..., channels, spatial...]`.
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        let input_dims = input.dims();
        let rank = input_dims.len();
        let spatial = self.kernel.len();

        assert!(
            rank > spatial,
            "ConvND expects input rank ({rank}) to be greater than kernel dims ({spatial})",
        );

        let batch_len = rank - spatial - 1;
        assert_eq!(
            input_dims[batch_len],
            Expression::from(self.ch_in),
            "Input channel dimension ({}) must match ch_in ({})",
            input_dims[batch_len],
            self.ch_in
        );
        assert_eq!(
            self.weight.dims()[0],
            Expression::from(self.ch_out),
            "Weight output channels ({}) must match ch_out ({})",
            self.weight.dims()[0],
            self.ch_out
        );
        let kernel_product: usize = self.kernel.iter().product();
        assert_eq!(
            self.weight.dims()[1],
            Expression::from((self.ch_in / self.groups) * kernel_product),
            "Weight input channels * kernel size ({}) must match expected grouped shape ({})",
            self.weight.dims()[1],
            (self.ch_in / self.groups) * kernel_product
        );

        let mut out = if self.groups == 1 {
            conv_forward_unfold(
                input,
                self.weight,
                self.ch_in,
                self.ch_out,
                &self.kernel,
                &self.stride,
                &self.dilation,
                &self.padding,
            )
        } else {
            let ch_in_per_group = self.ch_in / self.groups;
            let ch_out_per_group = self.ch_out / self.groups;

            let mut out = conv_forward_unfold(
                input.slice_along(0..ch_in_per_group, batch_len),
                self.weight.slice_along(0..ch_out_per_group, 0),
                ch_in_per_group,
                ch_out_per_group,
                &self.kernel,
                &self.stride,
                &self.dilation,
                &self.padding,
            );
            for group in 1..self.groups {
                let in_start = group * ch_in_per_group;
                let in_end = in_start + ch_in_per_group;
                let out_start = group * ch_out_per_group;
                let out_end = out_start + ch_out_per_group;

                let group_out = conv_forward_unfold(
                    input.slice_along(in_start..in_end, batch_len),
                    self.weight.slice_along(out_start..out_end, 0),
                    ch_in_per_group,
                    ch_out_per_group,
                    &self.kernel,
                    &self.stride,
                    &self.dilation,
                    &self.padding,
                );
                out = out.concat_along(group_out, batch_len);
            }
            out
        };

        if let Some(bias) = self.bias {
            out = add_channel_bias(out, bias, spatial);
        }

        out
    }

    pub fn infer_output_shape(&self, input: &[usize]) -> Vec<usize> {
        let rank = input.len();
        let spatial = self.kernel.len();

        assert!(rank > spatial, "expected input rank > spatial dims");
        let batch_len = rank - spatial - 1;
        assert_eq!(
            input[batch_len], self.ch_in,
            "input channel dimension does not match ch_in",
        );

        let batch_prefix = &input[..batch_len];
        let spatial_dims = &input[batch_len + 1..];
        let out_spatial: Vec<usize> = spatial_dims
            .iter()
            .zip(
                self.kernel
                    .iter()
                    .zip(self.stride.iter())
                    .zip(self.dilation.iter())
                    .zip(self.padding.iter()),
            )
            .map(|(dim, (((k, s), d), p))| (dim + 2 * p - d * (k - 1) - 1) / s + 1)
            .collect();

        let mut shape = batch_prefix.to_vec();
        shape.push(self.ch_out);
        shape.extend(out_spatial);
        shape
    }
}

pub struct ConvTranspose1d {
    pub weight: GraphTensor, // (ch_in, ch_out, kernel)
    pub bias: Option<GraphTensor>,
    kernel: usize,
    stride: usize,
    padding: usize,
    ch_in: usize,
    ch_out: usize,
}

impl ConvTranspose1d {
    pub fn new(
        ch_in: usize,
        ch_out: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        bias: bool,
        cx: &mut Graph,
    ) -> Self {
        assert!(kernel > 0, "ConvTranspose1d kernel must be positive");
        assert!(stride > 0, "ConvTranspose1d stride must be positive");
        assert!(
            padding < kernel,
            "ConvTranspose1d padding ({padding}) must be less than kernel ({kernel})",
        );

        Self {
            weight: cx.named_tensor("ConvTranspose1dWeight", (ch_in, ch_out, kernel)),
            bias: if bias {
                Some(cx.named_tensor("ConvTranspose1dBias", ch_out))
            } else {
                None
            },
            kernel,
            stride,
            padding,
            ch_in,
            ch_out,
        }
    }

    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        let (_, ch_in, length) = input.dims3();
        assert_eq!(
            ch_in,
            Expression::from(self.ch_in),
            "Input channel dimension ({}) must match ch_in ({})",
            ch_in,
            self.ch_in
        );

        let upsampled_len = (length - 1) * self.stride + 1;
        let upsampled = input
            .expand_dim(3, 1)
            .pad_along(0, self.stride - 1, 3, 0.0)
            .merge_dims(2, 3)
            .slice_along(..upsampled_len, 2);

        let conv_padding = self.kernel - 1 - self.padding;
        let padded = upsampled.pad(((0, 0), (0, 0), (conv_padding, conv_padding)), 0.0);
        let unfolded = padded.unfold([1, 1, self.kernel], [1, 1, 1], [1, 1, 1]);
        let mut patches = unfolded.permute((0, 2, 1, 5, 3, 4));
        patches = patches.squeeze(5).squeeze(4);

        let patch_dims = patches.dims();
        let weight = reverse_kernel_1d(self.weight, self.kernel).permute((1, 0, 2));
        let expanded_weight = weight
            .expand_lhs(&patch_dims[..1])
            .expand_dim(2, patch_dims[1]);

        let mut out = patches.expand_dim(1, self.ch_out) * expanded_weight;
        out = out.sum(4).sum(3);
        if let Some(bias) = self.bias {
            out = add_channel_bias(out, bias, 1);
        }
        out
    }

    pub fn infer_output_shape(&self, input: &[usize]) -> Vec<usize> {
        assert_eq!(
            input.len(),
            3,
            "ConvTranspose1d expects input rank 3 [batch, channels, length]",
        );
        assert_eq!(
            input[1], self.ch_in,
            "input channel dimension does not match ch_in",
        );
        let out_len = (input[2] - 1) * self.stride - (2 * self.padding) + self.kernel;
        vec![input[0], self.ch_out, out_len]
    }
}

#[cfg(test)]
mod tests {
    use super::{ConvND, ConvTranspose1d};
    use candle_core::{Device, Tensor};
    use luminal::prelude::{NativeRuntime, Runtime};

    fn assert_close(a: &[f32], b: &[f32]) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (idx, (lhs, rhs)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            if diff > 1e-4 {
                panic!("values differ at {idx}: {lhs} vs {rhs} (diff {diff})");
            }
        }
    }

    fn candle_conv1d_output(
        conv: &ConvND,
        input: &[f32],
        width: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> candle_core::Result<Vec<f32>> {
        let device = Device::Cpu;
        let input = Tensor::from_vec(input.to_vec(), (1, conv.ch_in, width), &device)?;
        let weight = Tensor::from_vec(
            weight.to_vec(),
            (conv.ch_out, conv.ch_in / conv.groups, conv.kernel[0]),
            &device,
        )?;
        let bias = match bias {
            Some(b) => Some(Tensor::from_vec(b.to_vec(), conv.ch_out, &device)?),
            None => None,
        };

        let output = input.conv1d(
            &weight,
            conv.padding[0],
            conv.stride[0],
            conv.dilation[0],
            conv.groups,
        )?;
        let output = match bias {
            Some(bias) => {
                let bias = bias.reshape((1, conv.ch_out, 1))?;
                output.broadcast_add(&bias)?
            }
            None => output,
        };
        output.flatten_all()?.to_vec1::<f32>()
    }

    fn candle_conv2d_output(
        conv: &ConvND,
        input: &[f32],
        height: usize,
        width: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> candle_core::Result<Vec<f32>> {
        let device = Device::Cpu;
        let input = Tensor::from_vec(input.to_vec(), (1, conv.ch_in, height, width), &device)?;
        let weight = Tensor::from_vec(
            weight.to_vec(),
            (
                conv.ch_out,
                conv.ch_in / conv.groups,
                conv.kernel[0],
                conv.kernel[1],
            ),
            &device,
        )?;
        let bias = match bias {
            Some(b) => Some(Tensor::from_vec(b.to_vec(), conv.ch_out, &device)?),
            None => None,
        };

        assert_eq!(
            conv.padding[0], conv.padding[1],
            "Candle conv2d only supports equal padding"
        );
        assert_eq!(
            conv.stride[0], conv.stride[1],
            "Candle conv2d only supports equal stride"
        );
        assert_eq!(
            conv.dilation[0], conv.dilation[1],
            "Candle conv2d only supports equal dilation"
        );

        let output = input.conv2d(
            &weight,
            conv.padding[0],
            conv.stride[0],
            conv.dilation[0],
            conv.groups,
        )?;
        let output = match bias {
            Some(bias) => {
                let bias = bias.reshape((1, conv.ch_out, 1, 1))?;
                output.broadcast_add(&bias)?
            }
            None => output,
        };
        output.flatten_all()?.to_vec1::<f32>()
    }

    fn candle_conv_transpose1d_output(
        conv: &ConvTranspose1d,
        input: &[f32],
        width: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> candle_core::Result<Vec<f32>> {
        let device = Device::Cpu;
        let input = Tensor::from_vec(input.to_vec(), (1, conv.ch_in, width), &device)?;
        let weight = Tensor::from_vec(
            weight.to_vec(),
            (conv.ch_in, conv.ch_out, conv.kernel),
            &device,
        )?;
        let output = input.conv_transpose1d(&weight, conv.padding, 0, conv.stride, 1, 1)?;
        let output = match bias {
            Some(bias) => {
                let bias = Tensor::from_vec(bias.to_vec(), conv.ch_out, &device)?;
                let bias = bias.reshape((1, conv.ch_out, 1))?;
                output.broadcast_add(&bias)?
            }
            None => output,
        };
        output.flatten_all()?.to_vec1::<f32>()
    }

    #[test]
    fn conv1d_values_match_expected_window_sums() -> candle_core::Result<()> {
        let mut cx = luminal::graph::Graph::new();
        let conv = ConvND::new(1, 1, vec![3], vec![1], vec![1], vec![1], true, 1, &mut cx);

        let input = [1., 2., 3., 4., 5.];
        let weight = [1., 1., 1.];
        let bias = [0.5];

        let out = candle_conv1d_output(&conv, &input, input.len(), &weight, Some(&bias))?;

        assert_close(&out, &[3.5, 6.5, 9.5, 12.5, 9.5]);
        Ok(())
    }

    #[test]
    fn conv2d_values_accumulate_across_channels() -> candle_core::Result<()> {
        let mut cx = luminal::graph::Graph::new();
        let conv = ConvND::new(
            2,
            1,
            vec![2, 2],
            vec![1, 1],
            vec![1, 1],
            vec![0, 0],
            true,
            1,
            &mut cx,
        );

        let input = [
            1., 2., 3., 4., 5., 6., 7., 8., 9., // channel 0
            9., 8., 7., 6., 5., 4., 3., 2., 1., // channel 1
        ];
        let weight = [1., 1., 1., 1., 2., 2., 2., 2.];
        let bias = [0.25];

        let out = candle_conv2d_output(&conv, &input, 3, 3, &weight, Some(&bias))?;

        assert_close(&out, &[68.25, 64.25, 56.25, 52.25]);
        Ok(())
    }

    #[test]
    fn conv1d_shapes_follow_stride_and_padding() {
        let mut cx = luminal::graph::Graph::new();
        let conv = ConvND::new(1, 1, vec![3], vec![2], vec![1], vec![1], false, 1, &mut cx);

        // expected length: floor((padded_len - dilation*(k-1) -1)/stride +1)
        // padded_len = 7 + 2 = 9
        // effective kernel = 3
        // => (9 -3)/2 +1 = 4
        let inferred = conv.infer_output_shape(&[2, 1, 7]);
        assert_eq!(inferred, vec![2, 1, 4]);
    }

    #[test]
    fn conv2d_shapes_follow_stride_and_padding() {
        let mut cx = luminal::graph::Graph::new();
        let conv = ConvND::new(
            3,
            2,
            vec![2, 3],
            vec![1, 2],
            vec![1, 1],
            vec![0, 1],
            true,
            1,
            &mut cx,
        );

        // height: (5 - dilation*(2-1) -1 + 0 +0)/1 +1 = 4
        // width: (6 - dilation*(3-1) -1 + 1 +1)/2 +1 = 3
        let inferred = conv.infer_output_shape(&[1, 3, 5, 6]);
        assert_eq!(inferred, vec![1, 2, 4, 3]);
    }

    #[test]
    fn test_conv1d_with_bias() -> candle_core::Result<()> {
        let mut cx = luminal::graph::Graph::new();
        let conv = ConvND::new(1, 1, vec![3], vec![1], vec![1], vec![1], true, 1, &mut cx);

        let width = 5;
        let input_values = vec![1.0, 2.0, 0.0, -1.0, 3.0];
        let weight_values = vec![1.0, -0.5, 0.25];
        let bias_values = vec![0.1];

        let input_tensor = cx.tensor((1, conv.ch_in, width));
        let output = conv.forward(input_tensor).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        rt.set_data(input_tensor.id, input_values.clone());
        rt.set_data(conv.weight.id, weight_values.clone());
        rt.set_data(conv.bias.unwrap().id, bias_values.clone());
        rt.execute(&cx.dyn_map);

        let expected = candle_conv1d_output(
            &conv,
            &input_values,
            width,
            &weight_values,
            Some(&bias_values),
        )?;

        assert_close(rt.get_f32(output.id), &expected);
        Ok(())
    }

    #[test]
    fn test_grouped_conv1d_depthwise_vs_candle() -> candle_core::Result<()> {
        let mut cx = luminal::graph::Graph::new();
        let conv = ConvND::new(3, 3, vec![7], vec![1], vec![1], vec![3], false, 3, &mut cx);

        let width = 12;
        let input_values: Vec<f32> = (0..(conv.ch_in * width))
            .map(|i| (i as f32 - 6.0) * 0.2)
            .collect();
        let weight_values: Vec<f32> = (0..(conv.ch_out * (conv.ch_in / conv.groups) * conv.kernel[0]))
            .map(|i| (i as f32 - 5.0) * 0.1)
            .collect();

        let input_tensor = cx.tensor((1, conv.ch_in, width));
        let output = conv.forward(input_tensor).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        rt.set_data(input_tensor.id, input_values.clone());
        rt.set_data(conv.weight.id, weight_values.clone());
        rt.execute(&cx.dyn_map);

        let expected = candle_conv1d_output(&conv, &input_values, width, &weight_values, None)?;
        assert_close(rt.get_f32(output.id), &expected);
        Ok(())
    }

    #[test]
    fn test_grouped_conv1d_vs_candle() -> candle_core::Result<()> {
        let mut cx = luminal::graph::Graph::new();
        let conv = ConvND::new(4, 6, vec![3], vec![2], vec![1], vec![1], true, 2, &mut cx);

        let width = 9;
        let input_values: Vec<f32> = (0..(conv.ch_in * width))
            .map(|i| (i as f32 - 7.0) * 0.15)
            .collect();
        let weight_values: Vec<f32> = (0..(conv.ch_out * (conv.ch_in / conv.groups) * conv.kernel[0]))
            .map(|i| (i as f32 - 9.0) * 0.07)
            .collect();
        let bias_values: Vec<f32> = (0..conv.ch_out).map(|i| i as f32 * 0.25 - 0.4).collect();

        let input_tensor = cx.tensor((1, conv.ch_in, width));
        let output = conv.forward(input_tensor).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        rt.set_data(input_tensor.id, input_values.clone());
        rt.set_data(conv.weight.id, weight_values.clone());
        rt.set_data(conv.bias.unwrap().id, bias_values.clone());
        rt.execute(&cx.dyn_map);

        let expected = candle_conv1d_output(
            &conv,
            &input_values,
            width,
            &weight_values,
            Some(&bias_values),
        )?;
        assert_close(rt.get_f32(output.id), &expected);
        Ok(())
    }

    #[test]
    fn test_conv_transpose1d_basic_shape() {
        let mut cx = luminal::graph::Graph::new();
        let conv = ConvTranspose1d::new(2, 3, 4, 2, 1, false, &mut cx);

        let inferred = conv.infer_output_shape(&[1, 2, 5]);
        assert_eq!(inferred, vec![1, 3, 10]);
    }

    #[test]
    fn test_conv_transpose1d_vs_candle() -> candle_core::Result<()> {
        let mut cx = luminal::graph::Graph::new();
        let conv = ConvTranspose1d::new(2, 3, 4, 2, 1, false, &mut cx);

        let width = 6;
        let input_values: Vec<f32> = (0..(conv.ch_in * width))
            .map(|i| (i as f32 - 3.0) * 0.2)
            .collect();
        let weight_values: Vec<f32> = (0..(conv.ch_in * conv.ch_out * conv.kernel))
            .map(|i| (i as f32 - 8.0) * 0.1)
            .collect();

        let input_tensor = cx.tensor((1, conv.ch_in, width));
        let output = conv.forward(input_tensor).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        rt.set_data(input_tensor.id, input_values.clone());
        rt.set_data(conv.weight.id, weight_values.clone());
        rt.execute(&cx.dyn_map);

        let expected =
            candle_conv_transpose1d_output(&conv, &input_values, width, &weight_values, None)?;
        assert_close(rt.get_f32(output.id), &expected);
        Ok(())
    }

    #[test]
    fn test_conv_transpose1d_with_bias() -> candle_core::Result<()> {
        let mut cx = luminal::graph::Graph::new();
        let conv = ConvTranspose1d::new(1, 2, 5, 3, 2, true, &mut cx);

        let width = 4;
        let input_values: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0];
        let weight_values: Vec<f32> = (0..(conv.ch_in * conv.ch_out * conv.kernel))
            .map(|i| (i as f32 - 4.0) * 0.08)
            .collect();
        let bias_values: Vec<f32> = vec![0.3, -0.25];

        let input_tensor = cx.tensor((1, conv.ch_in, width));
        let output = conv.forward(input_tensor).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        rt.set_data(input_tensor.id, input_values.clone());
        rt.set_data(conv.weight.id, weight_values.clone());
        rt.set_data(conv.bias.unwrap().id, bias_values.clone());
        rt.execute(&cx.dyn_map);

        let expected = candle_conv_transpose1d_output(
            &conv,
            &input_values,
            width,
            &weight_values,
            Some(&bias_values),
        )?;
        assert_close(rt.get_f32(output.id), &expected);
        Ok(())
    }
}
