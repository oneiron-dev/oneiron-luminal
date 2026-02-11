use luminal::prelude::*;

/// Generic N-dimensional convolution layer implemented with the GraphTensor `unfold` helper.
///
/// The layer expects inputs shaped like `[batch..., channels, spatial...]` where the number of
/// spatial dimensions is greater than zero. The kernel configuration controls how many spatial
/// axes are convolved (N) and must be shorter than the input rank (K): `K > N` is asserted.
pub struct ConvND {
    pub weight: GraphTensor, // (ch_out, ch_in * kernel_product)
    pub bias: Option<GraphTensor>,
    kernel: Vec<usize>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<usize>,
    ch_in: usize,
    ch_out: usize,
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
        cx: &mut Graph,
    ) -> Self {
        assert!(
            !kernel.is_empty(),
            "ConvND requires at least one spatial dimension in the kernel",
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

        Self {
            weight: cx.named_tensor("ConvWeight", (ch_out, ch_in * kernel_product)),
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

        // Pad only the spatial dimensions.
        let mut padding = vec![(Expression::from(0), Expression::from(0)); rank];
        for (i, pad) in self.padding.iter().enumerate() {
            let axis = batch_len + 1 + i;
            padding[axis] = (Expression::from(*pad), Expression::from(*pad));
        }
        let padded = input.pad(padding, 0.0);

        // Build unfold parameters with ones for non-spatial axes.
        let mut kernel_shape = vec![1; rank];
        let mut stride_shape = vec![1; rank];
        let mut dilation_shape = vec![1; rank];
        for i in 0..spatial {
            let axis = batch_len + 1 + i;
            kernel_shape[axis] = self.kernel[i];
            stride_shape[axis] = self.stride[i];
            dilation_shape[axis] = self.dilation[i];
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

        // Reshape weight from [ch_out, ch_in * kernel_product] to [ch_out, ch_in, kernel...].
        let mut weight = self.weight;
        for k in self.kernel.iter().rev() {
            weight = weight.split_dims(1, *k);
        }

        let patch_dims = patches.dims();

        // Broadcasted multiply across [channels, kernel...] then reduce those axes.
        let mut out = patches.expand_dim(batch_len + spatial, self.ch_out)
            * weight.expand_lhs(&patch_dims[..batch_len + spatial]);

        for _ in 0..=spatial {
            let last_axis = out.dims().len() - 1;
            out = out.sum(last_axis);
        }

        // Move channel dimension ahead of the spatial axes: [batch..., ch_out, spatial...]
        let mut final_order: Vec<usize> = (0..batch_len).collect();
        final_order.push(batch_len + spatial);
        final_order.extend(batch_len..batch_len + spatial);
        out = out.permute(final_order);

        if let Some(b) = self.bias {
            let out_dims = out.dims();
            let spatial_len = self.kernel.len();
            let batch_len = out_dims.len() - spatial_len - 1;
            out += b.expand_lhs(&out_dims[..batch_len])
                    .expand_rhs(&out_dims[batch_len + 1..]);
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

#[cfg(test)]
mod tests {
    use super::ConvND;
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
            (conv.ch_out, conv.ch_in, conv.kernel[0]),
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
            1,
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
            (conv.ch_out, conv.ch_in, conv.kernel[0], conv.kernel[1]),
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
            1,
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

    #[test]
    fn conv1d_values_match_expected_window_sums() -> candle_core::Result<()> {
        let mut cx = luminal::graph::Graph::new();
        let conv = ConvND::new(1, 1, vec![3], vec![1], vec![1], vec![1], true, &mut cx);

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
        let conv = ConvND::new(1, 1, vec![3], vec![2], vec![1], vec![1], false, &mut cx);

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
        let conv = ConvND::new(1, 1, vec![3], vec![1], vec![1], vec![1], true, &mut cx);

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
}
