use crate::{
    code_predictor::{CodePredictorConfig, CodePredictorModel},
    model::{TalkerConfig, TalkerModel, TextProjection, VOCAB_SIZE},
    pipeline::{
        decode_speech, generate_frames, sample_greedy, NonStreamingPromptInputs,
        StreamingPromptInputs, TtsPipeline,
    },
    speech_decoder::{
        CausalConv1d, CausalTransConv1d, ConvNeXtBlock, DecoderBlock, PreTransformer, SnakeBeta,
        SpeechDecoder, SpeechDecoderConfig, SplitResidualVectorQuantizer, WaveformDecoder,
    },
    weight_loader::{expected_element_count, load_safetensors_to_map, load_safetensors_to_native},
};
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::ops::softmax;
use half::bf16;
use luminal::hlir::Input;
use luminal::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use safetensors::{serialize, tensor::TensorView, Dtype, SafeTensors};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

const TEST_EPS: f32 = 1e-3;

fn random_vec(rng: &mut StdRng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| rng.random_range(-0.25f32..0.25f32))
        .collect()
}

fn build_random_weight_map(
    talker_config: &TalkerConfig,
    predictor_config: &CodePredictorConfig,
    rng: &mut StdRng,
) -> HashMap<String, Vec<f32>> {
    use luminal::hlir::Input;

    let mut cx = Graph::new();
    let pipeline = TtsPipeline::new(&mut cx, talker_config.clone(), predictor_config.clone());
    let prompt = cx.tensor((1, 1, talker_config.hidden));
    let (logits, normed) = pipeline.talker.decode_step(prompt);
    let last_logits = logits.slice((.., 0.., ..));
    let code_0 = last_logits.argmax(2);
    let code_0_embed = pipeline.talker.embed_codec(code_0);
    let last_hidden = normed.slice((.., 0.., ..));
    let pred_out = pipeline
        .code_predictor
        .generate_codes(last_hidden, code_0_embed);
    let mut codec_sum = code_0_embed;
    for embed in &pred_out.embeds {
        codec_sum = codec_sum + *embed;
    }
    let _ = codec_sum.output();

    let mut map = HashMap::new();
    for node in cx.graph.node_indices() {
        let Some(input) = cx.graph[node].as_any().downcast_ref::<Input>() else {
            continue;
        };
        if input.label.is_empty() {
            continue;
        }
        if let Some(count) = expected_element_count(&cx, node) {
            map.insert(input.label.clone(), random_vec(rng, count));
        }
    }
    map
}

fn build_random_speech_weight_map(
    speech_config: &SpeechDecoderConfig,
    rng: &mut StdRng,
) -> HashMap<String, Vec<f32>> {
    let mut cx = Graph::new();
    let decoder = SpeechDecoder::new(&mut cx, speech_config.clone());
    let num_codebooks =
        speech_config.num_semantic_quantizers + speech_config.num_acoustic_quantizers;
    let code_tensors: Vec<GraphTensor> = (0..num_codebooks)
        .map(|_| cx.tensor((1, 1)).as_dtype(DType::Int))
        .collect();
    let _ = decoder.decode_codes(code_tensors).output();

    let mut map = HashMap::new();
    for node in cx.graph.node_indices() {
        let Some(input) = cx.graph[node].as_any().downcast_ref::<Input>() else {
            continue;
        };
        if input.label.is_empty() {
            continue;
        }
        if let Some(count) = expected_element_count(&cx, node) {
            map.insert(input.label.clone(), random_vec(rng, count));
        }
    }
    map
}

fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (lhs, rhs)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= tol,
            "mismatch at index {i}: {lhs} vs {rhs} (|diff|={diff}, tol={tol})"
        );
    }
}

struct TempFileCleanup(PathBuf);

impl Drop for TempFileCleanup {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn write_temp_safetensors(tensors: &HashMap<String, TensorView<'_>>, tag: &str) -> PathBuf {
    let serialized = serialize(tensors, None).expect("serialize safetensors");
    let path = std::env::temp_dir().join(format!(
        "qwen3_tts_{tag}_{}_{}.safetensors",
        std::process::id(),
        SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos()
    ));
    fs::write(&path, serialized).expect("write temp safetensors");
    path
}

fn set_random_code_predictor_params(
    rt: &mut NativeRuntime,
    model: &CodePredictorModel,
    config: &CodePredictorConfig,
    rng: &mut StdRng,
) {
    for codec_embedding in &model.codec_embeddings {
        rt.set_data(
            codec_embedding.id,
            random_vec(rng, config.codebook_vocab * config.talker_hidden),
        );
    }
    rt.set_data(
        model.projection_weight.id,
        random_vec(rng, config.hidden * config.talker_hidden),
    );
    rt.set_data(model.projection_bias.id, random_vec(rng, config.hidden));

    for layer in &model.layers {
        rt.set_data(
            layer.q_proj.id,
            random_vec(rng, config.n_heads * config.head_dim * config.hidden),
        );
        rt.set_data(
            layer.k_proj.id,
            random_vec(rng, config.n_kv_heads * config.head_dim * config.hidden),
        );
        rt.set_data(
            layer.v_proj.id,
            random_vec(rng, config.n_kv_heads * config.head_dim * config.hidden),
        );
        rt.set_data(
            layer.q_norm.weight.expect("predictor q norm weight").id,
            random_vec(rng, config.head_dim),
        );
        rt.set_data(
            layer.k_norm.weight.expect("predictor k norm weight").id,
            random_vec(rng, config.head_dim),
        );
        rt.set_data(
            layer.o_proj.id,
            random_vec(rng, config.hidden * config.n_heads * config.head_dim),
        );
        rt.set_data(
            layer.gate_proj.id,
            random_vec(rng, config.intermediate * config.hidden),
        );
        rt.set_data(
            layer.up_proj.id,
            random_vec(rng, config.intermediate * config.hidden),
        );
        rt.set_data(
            layer.down_proj.id,
            random_vec(rng, config.hidden * config.intermediate),
        );
        rt.set_data(
            layer
                .input_norm
                .weight
                .expect("predictor input norm weight")
                .id,
            random_vec(rng, config.hidden),
        );
        rt.set_data(
            layer
                .post_attn_norm
                .weight
                .expect("predictor post attention norm weight")
                .id,
            random_vec(rng, config.hidden),
        );
    }

    rt.set_data(
        model
            .final_norm
            .weight
            .expect("predictor final norm weight")
            .id,
        random_vec(rng, config.hidden),
    );

    for lm_head in &model.lm_heads {
        rt.set_data(
            lm_head.id,
            random_vec(rng, config.codebook_vocab * config.hidden),
        );
    }
}

fn set_random_talker_params(
    rt: &mut NativeRuntime,
    model: &TalkerModel,
    config: &TalkerConfig,
    rng: &mut StdRng,
) {
    rt.set_data(
        model.codec_embedding.id,
        random_vec(rng, config.vocab_size * config.hidden),
    );
    rt.set_data(
        model.text_embedding.id,
        random_vec(rng, config.text_vocab_size * config.hidden),
    );
    rt.set_data(
        model.text_projection.fc1_weight.id,
        random_vec(rng, config.hidden * config.hidden),
    );
    rt.set_data(
        model.text_projection.fc1_bias.id,
        random_vec(rng, config.hidden),
    );
    rt.set_data(
        model.text_projection.fc2_weight.id,
        random_vec(rng, config.hidden * config.hidden),
    );
    rt.set_data(
        model.text_projection.fc2_bias.id,
        random_vec(rng, config.hidden),
    );
    rt.set_data(
        model.codec_head.id,
        random_vec(rng, config.vocab_size * config.hidden),
    );

    for layer in &model.layers {
        rt.set_data(
            layer.q_proj.id,
            random_vec(rng, config.n_heads * config.head_dim * config.hidden),
        );
        rt.set_data(
            layer.k_proj.id,
            random_vec(rng, config.n_kv_heads * config.head_dim * config.hidden),
        );
        rt.set_data(
            layer.v_proj.id,
            random_vec(rng, config.n_kv_heads * config.head_dim * config.hidden),
        );
        rt.set_data(
            layer.q_norm.weight.expect("talker q norm weight").id,
            random_vec(rng, config.head_dim),
        );
        rt.set_data(
            layer.k_norm.weight.expect("talker k norm weight").id,
            random_vec(rng, config.head_dim),
        );
        rt.set_data(
            layer.o_proj.id,
            random_vec(rng, config.hidden * config.n_heads * config.head_dim),
        );
        rt.set_data(
            layer.gate_proj.id,
            random_vec(rng, config.intermediate * config.hidden),
        );
        rt.set_data(
            layer.up_proj.id,
            random_vec(rng, config.intermediate * config.hidden),
        );
        rt.set_data(
            layer.down_proj.id,
            random_vec(rng, config.hidden * config.intermediate),
        );
        rt.set_data(
            layer
                .input_norm
                .weight
                .expect("talker input norm weight")
                .id,
            random_vec(rng, config.hidden),
        );
        rt.set_data(
            layer
                .post_attn_norm
                .weight
                .expect("talker post attention norm weight")
                .id,
            random_vec(rng, config.hidden),
        );
    }

    rt.set_data(
        model
            .final_norm
            .weight
            .expect("talker final norm weight")
            .id,
        random_vec(rng, config.hidden),
    );
}

#[derive(Clone)]
struct OneLayerWeights {
    embed: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    o_proj: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
    input_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
}

#[derive(Clone)]
struct CodePredictorOneLayerWeights {
    projection_weight: Vec<f32>,
    projection_bias: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    o_proj: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
    input_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    final_norm: Vec<f32>,
    lm_head0: Vec<f32>,
}

fn rms_norm_3d(x: &Tensor, weight: &Tensor, eps: f32) -> CandleResult<Tensor> {
    let dims = x.dims();
    let (b, s, h) = (dims[0], dims[1], dims[2]);
    let inv_rms = x
        .powf(2.0)?
        .mean(2)?
        .broadcast_add(&Tensor::new(eps, x.device())?)?
        .sqrt()?
        .recip()?
        .reshape((b, s, 1))?
        .broadcast_as((b, s, h))?;
    let w = weight.reshape((1, 1, h))?.broadcast_as((b, s, h))?;
    x.mul(&inv_rms)?.mul(&w)
}

fn rms_norm_4d_lastdim(x: &Tensor, weight: &Tensor, eps: f32) -> CandleResult<Tensor> {
    let dims = x.dims();
    let (b, h, s, d) = (dims[0], dims[1], dims[2], dims[3]);
    let inv_rms = x
        .powf(2.0)?
        .mean(3)?
        .broadcast_add(&Tensor::new(eps, x.device())?)?
        .sqrt()?
        .recip()?
        .reshape((b, h, s, 1))?
        .broadcast_as((b, h, s, d))?;
    let w = weight.reshape((1, 1, 1, d))?.broadcast_as((b, h, s, d))?;
    x.mul(&inv_rms)?.mul(&w)
}

fn candle_rope_with_theta(input: &Tensor, rope_theta: f32) -> CandleResult<Tensor> {
    let dims = input.dims();
    let (b, h, s, d) = (dims[0], dims[1], dims[2], dims[3]);
    let half = d / 2;

    let mut inv_freq = Vec::with_capacity(half);
    for i in (0..d).step_by(2) {
        inv_freq.push(rope_theta.powf(-(i as f32) / d as f32));
    }

    let mut theta = Vec::with_capacity(s * half);
    for pos in 0..s {
        for &f in &inv_freq {
            theta.push(pos as f32 * f);
        }
    }

    let theta = Tensor::from_vec(theta, (s, half), input.device())?;
    let cos = theta
        .cos()?
        .reshape((1, 1, s, half))?
        .broadcast_as((b, h, s, half))?;
    let sin = theta
        .sin()?
        .reshape((1, 1, s, half))?
        .broadcast_as((b, h, s, half))?;

    let first_half = input.narrow(3, 0, half)?;
    let second_half = input.narrow(3, half, half)?;

    let rotated_first = first_half.mul(&cos)?.sub(&second_half.mul(&sin)?)?;
    let rotated_second = second_half.mul(&cos)?.add(&first_half.mul(&sin)?)?;
    Tensor::cat(&[&rotated_first, &rotated_second], 3)
}

fn candle_rope(input: &Tensor, config: &TalkerConfig) -> CandleResult<Tensor> {
    candle_rope_with_theta(input, config.rope_theta)
}

fn linear_3d(x: &Tensor, weight: &Tensor) -> CandleResult<Tensor> {
    let dims = x.dims();
    let (b, s, h_in) = (dims[0], dims[1], dims[2]);
    let h_out = weight.dims()[0];
    x.reshape((b * s, h_in))?
        .matmul(&weight.transpose(0, 1)?)?
        .reshape((b, s, h_out))
}

fn repeat_kv(kv: &Tensor, kv_groups: usize) -> CandleResult<Tensor> {
    if kv_groups <= 1 {
        return Ok(kv.clone());
    }
    let copies = vec![kv; kv_groups];
    Tensor::cat(&copies, 1)
}

fn embed_lookup(
    token_ids: &[i32],
    batch: usize,
    seq: usize,
    hidden: usize,
    embedding: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0; batch * seq * hidden];
    for b in 0..batch {
        for s in 0..seq {
            let token = token_ids[b * seq + s] as usize;
            let src = token * hidden;
            let dst = (b * seq + s) * hidden;
            out[dst..dst + hidden].copy_from_slice(&embedding[src..src + hidden]);
        }
    }
    out
}

fn candle_single_layer_hidden(
    token_ids: &[i32],
    batch: usize,
    seq: usize,
    config: &TalkerConfig,
    w: &OneLayerWeights,
) -> CandleResult<Tensor> {
    let device = Device::Cpu;
    let hidden = config.hidden;
    let q_dim = config.n_heads * config.head_dim;
    let kv_dim = config.n_kv_heads * config.head_dim;

    let embedded = embed_lookup(token_ids, batch, seq, hidden, &w.embed);
    let mut x = Tensor::from_vec(embedded, (batch, seq, hidden), &device)?;

    let input_norm_w = Tensor::from_vec(w.input_norm.clone(), hidden, &device)?;
    let post_attn_norm_w = Tensor::from_vec(w.post_attn_norm.clone(), hidden, &device)?;
    let q_norm_w = Tensor::from_vec(w.q_norm.clone(), config.head_dim, &device)?;
    let k_norm_w = Tensor::from_vec(w.k_norm.clone(), config.head_dim, &device)?;

    let q_proj = Tensor::from_vec(w.q_proj.clone(), (q_dim, hidden), &device)?;
    let k_proj = Tensor::from_vec(w.k_proj.clone(), (kv_dim, hidden), &device)?;
    let v_proj = Tensor::from_vec(w.v_proj.clone(), (kv_dim, hidden), &device)?;
    let o_proj = Tensor::from_vec(w.o_proj.clone(), (hidden, q_dim), &device)?;
    let gate_proj = Tensor::from_vec(w.gate_proj.clone(), (config.intermediate, hidden), &device)?;
    let up_proj = Tensor::from_vec(w.up_proj.clone(), (config.intermediate, hidden), &device)?;
    let down_proj = Tensor::from_vec(w.down_proj.clone(), (hidden, config.intermediate), &device)?;

    let x_attn = rms_norm_3d(&x, &input_norm_w, config.rms_norm_eps)?;
    let mut q = linear_3d(&x_attn, &q_proj)?
        .reshape((batch, seq, config.n_heads, config.head_dim))?
        .transpose(1, 2)?;
    let mut k = linear_3d(&x_attn, &k_proj)?
        .reshape((batch, seq, config.n_kv_heads, config.head_dim))?
        .transpose(1, 2)?;
    let mut v = linear_3d(&x_attn, &v_proj)?
        .reshape((batch, seq, config.n_kv_heads, config.head_dim))?
        .transpose(1, 2)?;

    q = rms_norm_4d_lastdim(&q, &q_norm_w, config.rms_norm_eps)?;
    k = rms_norm_4d_lastdim(&k, &k_norm_w, config.rms_norm_eps)?;

    q = candle_rope(&q, config)?;
    k = candle_rope(&k, config)?;

    k = repeat_kv(&k, config.kv_groups)?;
    v = repeat_kv(&v, config.kv_groups)?;

    let mut scores = q.matmul(&k.transpose(2, 3)?)?;
    scores = scores.broadcast_mul(&Tensor::new(
        1.0f32 / (config.head_dim as f32).sqrt(),
        &device,
    )?)?;

    let mut mask = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask[i * seq + j] = -1e9;
        }
    }
    let mask = Tensor::from_vec(mask, (1, 1, seq, seq), &device)?.broadcast_as((
        batch,
        config.n_heads,
        seq,
        seq,
    ))?;
    let scores = scores.add(&mask)?;
    let probs = softmax(&scores, 3)?;

    let context = probs
        .matmul(&v)?
        .transpose(1, 2)?
        .reshape((batch, seq, hidden))?;
    let attn_out = linear_3d(&context, &o_proj)?;
    x = x.add(&attn_out)?;

    let x_ff = rms_norm_3d(&x, &post_attn_norm_w, config.rms_norm_eps)?;
    let gate = linear_3d(&x_ff, &gate_proj)?.silu()?;
    let up = linear_3d(&x_ff, &up_proj)?;
    let mlp = linear_3d(&gate.mul(&up)?, &down_proj)?;
    x.add(&mlp)
}

fn candle_code_predictor_one_layer(
    input: &[f32],
    batch: usize,
    seq: usize,
    config: &CodePredictorConfig,
    w: &CodePredictorOneLayerWeights,
) -> CandleResult<Tensor> {
    let device = Device::Cpu;
    let hidden = config.hidden;
    let q_dim = config.n_heads * config.head_dim;
    let kv_dim = config.n_kv_heads * config.head_dim;

    let input = Tensor::from_vec(input.to_vec(), (batch, seq, config.talker_hidden), &device)?;

    let projection_weight = Tensor::from_vec(
        w.projection_weight.clone(),
        (hidden, config.talker_hidden),
        &device,
    )?;
    let projection_bias = Tensor::from_vec(w.projection_bias.clone(), hidden, &device)?
        .reshape((1, 1, hidden))?
        .broadcast_as((batch, seq, hidden))?;
    let mut x = linear_3d(&input, &projection_weight)?.add(&projection_bias)?;

    let input_norm_w = Tensor::from_vec(w.input_norm.clone(), hidden, &device)?;
    let post_attn_norm_w = Tensor::from_vec(w.post_attn_norm.clone(), hidden, &device)?;
    let final_norm_w = Tensor::from_vec(w.final_norm.clone(), hidden, &device)?;
    let q_norm_w = Tensor::from_vec(w.q_norm.clone(), config.head_dim, &device)?;
    let k_norm_w = Tensor::from_vec(w.k_norm.clone(), config.head_dim, &device)?;

    let q_proj = Tensor::from_vec(w.q_proj.clone(), (q_dim, hidden), &device)?;
    let k_proj = Tensor::from_vec(w.k_proj.clone(), (kv_dim, hidden), &device)?;
    let v_proj = Tensor::from_vec(w.v_proj.clone(), (kv_dim, hidden), &device)?;
    let o_proj = Tensor::from_vec(w.o_proj.clone(), (hidden, q_dim), &device)?;
    let gate_proj = Tensor::from_vec(w.gate_proj.clone(), (config.intermediate, hidden), &device)?;
    let up_proj = Tensor::from_vec(w.up_proj.clone(), (config.intermediate, hidden), &device)?;
    let down_proj = Tensor::from_vec(w.down_proj.clone(), (hidden, config.intermediate), &device)?;
    let lm_head0 = Tensor::from_vec(w.lm_head0.clone(), (config.codebook_vocab, hidden), &device)?;

    let x_attn = rms_norm_3d(&x, &input_norm_w, config.rms_norm_eps)?;
    let mut q = linear_3d(&x_attn, &q_proj)?
        .reshape((batch, seq, config.n_heads, config.head_dim))?
        .transpose(1, 2)?;
    let mut k = linear_3d(&x_attn, &k_proj)?
        .reshape((batch, seq, config.n_kv_heads, config.head_dim))?
        .transpose(1, 2)?;
    let mut v = linear_3d(&x_attn, &v_proj)?
        .reshape((batch, seq, config.n_kv_heads, config.head_dim))?
        .transpose(1, 2)?;

    q = rms_norm_4d_lastdim(&q, &q_norm_w, config.rms_norm_eps)?;
    k = rms_norm_4d_lastdim(&k, &k_norm_w, config.rms_norm_eps)?;

    q = candle_rope_with_theta(&q, config.rope_theta)?;
    k = candle_rope_with_theta(&k, config.rope_theta)?;

    k = repeat_kv(&k, config.kv_groups)?;
    v = repeat_kv(&v, config.kv_groups)?;

    let mut scores = q.matmul(&k.transpose(2, 3)?)?;
    scores = scores.broadcast_mul(&Tensor::new(
        1.0f32 / (config.head_dim as f32).sqrt(),
        &device,
    )?)?;

    let mut mask = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask[i * seq + j] = -1e9;
        }
    }
    let mask = Tensor::from_vec(mask, (1, 1, seq, seq), &device)?.broadcast_as((
        batch,
        config.n_heads,
        seq,
        seq,
    ))?;
    let scores = scores.add(&mask)?;
    let probs = softmax(&scores, 3)?;

    let context = probs
        .matmul(&v)?
        .transpose(1, 2)?
        .reshape((batch, seq, q_dim))?;
    let attn_out = linear_3d(&context, &o_proj)?;
    x = x.add(&attn_out)?;

    let x_ff = rms_norm_3d(&x, &post_attn_norm_w, config.rms_norm_eps)?;
    let gate = linear_3d(&x_ff, &gate_proj)?.silu()?;
    let up = linear_3d(&x_ff, &up_proj)?;
    let mlp = linear_3d(&gate.mul(&up)?, &down_proj)?;
    x = x.add(&mlp)?;

    x = rms_norm_3d(&x, &final_norm_w, config.rms_norm_eps)?;
    linear_3d(&x, &lm_head0)
}

#[test]
fn test_talker_graph_builds() {
    let mut cx = Graph::new();
    let model = TalkerModel::new(&mut cx, TalkerConfig::default());
    let token_ids = cx.tensor((1, 's')).as_dtype(DType::Int);
    let _embedded = model.embed_codec(token_ids).output();
    cx.set_dim('s', 8);

    cx.build_search_space::<NativeRuntime>();
    let _rt = cx.search(NativeRuntime::default(), 1);
}

#[test]
fn test_talker_output_shape() {
    let mut cx = Graph::new();
    let model = TalkerModel::new(&mut cx, TalkerConfig::default());
    let token_ids = cx.tensor((1, 's')).as_dtype(DType::Int);
    let logits = model.forward(token_ids);
    cx.set_dim('s', 8);

    let mut shape = logits.shape;
    shape.resolve_dyn_dims(&cx.dyn_map);
    assert_eq!(shape.shape_usize(), vec![1, 8, VOCAB_SIZE]);
}

#[test]
fn test_single_layer_vs_candle() -> CandleResult<()> {
    let config = TalkerConfig {
        layers: 1,
        hidden: 64,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        intermediate: 128,
        vocab_size: 96,
        ..TalkerConfig::default()
    };

    let batch = 1;
    let seq = 6;
    let mut rng = StdRng::seed_from_u64(7);

    let token_ids_data: Vec<i32> = (0..batch * seq)
        .map(|_| rng.random_range(0..config.vocab_size as i32))
        .collect();

    let weights = OneLayerWeights {
        embed: random_vec(&mut rng, config.vocab_size * config.hidden),
        q_proj: random_vec(&mut rng, config.n_heads * config.head_dim * config.hidden),
        k_proj: random_vec(
            &mut rng,
            config.n_kv_heads * config.head_dim * config.hidden,
        ),
        v_proj: random_vec(
            &mut rng,
            config.n_kv_heads * config.head_dim * config.hidden,
        ),
        q_norm: random_vec(&mut rng, config.head_dim),
        k_norm: random_vec(&mut rng, config.head_dim),
        o_proj: random_vec(&mut rng, config.hidden * config.n_heads * config.head_dim),
        gate_proj: random_vec(&mut rng, config.intermediate * config.hidden),
        up_proj: random_vec(&mut rng, config.intermediate * config.hidden),
        down_proj: random_vec(&mut rng, config.hidden * config.intermediate),
        input_norm: random_vec(&mut rng, config.hidden),
        post_attn_norm: random_vec(&mut rng, config.hidden),
    };

    let mut cx = Graph::new();
    let model = TalkerModel::new(&mut cx, config.clone());
    let token_ids = cx.tensor((batch, seq)).as_dtype(DType::Int);
    let hidden = model.forward_hidden(token_ids).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    rt.set_data(token_ids.id, token_ids_data.clone());
    rt.set_data(model.codec_embedding.id, weights.embed.clone());

    let layer = &model.layers[0];
    rt.set_data(layer.q_proj.id, weights.q_proj.clone());
    rt.set_data(layer.k_proj.id, weights.k_proj.clone());
    rt.set_data(layer.v_proj.id, weights.v_proj.clone());
    rt.set_data(
        layer.q_norm.weight.expect("q norm weight").id,
        weights.q_norm.clone(),
    );
    rt.set_data(
        layer.k_norm.weight.expect("k norm weight").id,
        weights.k_norm.clone(),
    );
    rt.set_data(layer.o_proj.id, weights.o_proj.clone());
    rt.set_data(layer.gate_proj.id, weights.gate_proj.clone());
    rt.set_data(layer.up_proj.id, weights.up_proj.clone());
    rt.set_data(layer.down_proj.id, weights.down_proj.clone());
    rt.set_data(
        layer.input_norm.weight.expect("input norm weight").id,
        weights.input_norm.clone(),
    );
    rt.set_data(
        layer
            .post_attn_norm
            .weight
            .expect("post attention norm weight")
            .id,
        weights.post_attn_norm.clone(),
    );

    rt.execute(&cx.dyn_map);
    let luminal_out = rt.get_f32(hidden.id).clone();

    let candle_out = candle_single_layer_hidden(&token_ids_data, batch, seq, &config, &weights)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    assert_close(&luminal_out, &candle_out, TEST_EPS);
    Ok(())
}

#[test]
fn test_qk_norm_changes_output() {
    let config = TalkerConfig {
        layers: 1,
        hidden: 64,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        kv_groups: 2,
        intermediate: 128,
        vocab_size: 96,
        ..TalkerConfig::default()
    };

    let batch = 1;
    let seq = 5;
    let mut rng = StdRng::seed_from_u64(1234);

    let mut cx = Graph::new();
    let model = TalkerModel::new(&mut cx, config.clone());
    let token_ids = cx.tensor((batch, seq)).as_dtype(DType::Int);
    let hidden = model.forward_hidden(token_ids).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    rt.set_data(token_ids.id, vec![0i32, 1, 2, 3, 4]);
    rt.set_data(
        model.codec_embedding.id,
        random_vec(&mut rng, config.vocab_size * config.hidden),
    );

    let layer = &model.layers[0];
    rt.set_data(
        layer.q_proj.id,
        random_vec(&mut rng, config.n_heads * config.head_dim * config.hidden),
    );
    rt.set_data(
        layer.k_proj.id,
        random_vec(
            &mut rng,
            config.n_kv_heads * config.head_dim * config.hidden,
        ),
    );
    rt.set_data(
        layer.v_proj.id,
        random_vec(
            &mut rng,
            config.n_kv_heads * config.head_dim * config.hidden,
        ),
    );
    rt.set_data(
        layer.k_norm.weight.expect("k norm weight").id,
        random_vec(&mut rng, config.head_dim),
    );
    rt.set_data(
        layer.o_proj.id,
        random_vec(&mut rng, config.hidden * config.n_heads * config.head_dim),
    );
    rt.set_data(
        layer.gate_proj.id,
        random_vec(&mut rng, config.intermediate * config.hidden),
    );
    rt.set_data(
        layer.up_proj.id,
        random_vec(&mut rng, config.intermediate * config.hidden),
    );
    rt.set_data(
        layer.down_proj.id,
        random_vec(&mut rng, config.hidden * config.intermediate),
    );
    rt.set_data(
        layer.input_norm.weight.expect("input norm weight").id,
        random_vec(&mut rng, config.hidden),
    );
    rt.set_data(
        layer
            .post_attn_norm
            .weight
            .expect("post attention norm weight")
            .id,
        random_vec(&mut rng, config.hidden),
    );

    let q_norm_a = vec![1.0f32; config.head_dim];
    let q_norm_b: Vec<f32> = (0..config.head_dim)
        .map(|i| if i % 2 == 0 { 0.5 } else { 1.5 })
        .collect();

    rt.set_data(layer.q_norm.weight.expect("q norm weight").id, q_norm_a);
    rt.execute(&cx.dyn_map);
    let out_a = rt.get_f32(hidden.id).clone();

    rt.set_data(layer.q_norm.weight.expect("q norm weight").id, q_norm_b);
    rt.execute(&cx.dyn_map);
    let out_b = rt.get_f32(hidden.id).clone();

    let max_diff = out_a
        .iter()
        .zip(out_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff > 1e-5,
        "changing q_norm should change output, got max diff {max_diff}"
    );
}

#[test]
fn test_code_predictor_single_layer_vs_candle() -> CandleResult<()> {
    let config = CodePredictorConfig {
        layers: 1,
        hidden: 32,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 16,
        kv_groups: 2,
        intermediate: 64,
        talker_hidden: 64,
        codebook_vocab: 48,
        ..CodePredictorConfig::default()
    };

    let batch = 1;
    let seq = 4;
    let mut rng = StdRng::seed_from_u64(17);

    let input_data = random_vec(&mut rng, batch * seq * config.talker_hidden);

    let weights = CodePredictorOneLayerWeights {
        projection_weight: random_vec(&mut rng, config.hidden * config.talker_hidden),
        projection_bias: random_vec(&mut rng, config.hidden),
        q_proj: random_vec(&mut rng, config.n_heads * config.head_dim * config.hidden),
        k_proj: random_vec(
            &mut rng,
            config.n_kv_heads * config.head_dim * config.hidden,
        ),
        v_proj: random_vec(
            &mut rng,
            config.n_kv_heads * config.head_dim * config.hidden,
        ),
        q_norm: random_vec(&mut rng, config.head_dim),
        k_norm: random_vec(&mut rng, config.head_dim),
        o_proj: random_vec(&mut rng, config.hidden * config.n_heads * config.head_dim),
        gate_proj: random_vec(&mut rng, config.intermediate * config.hidden),
        up_proj: random_vec(&mut rng, config.intermediate * config.hidden),
        down_proj: random_vec(&mut rng, config.hidden * config.intermediate),
        input_norm: random_vec(&mut rng, config.hidden),
        post_attn_norm: random_vec(&mut rng, config.hidden),
        final_norm: random_vec(&mut rng, config.hidden),
        lm_head0: random_vec(&mut rng, config.codebook_vocab * config.hidden),
    };

    let mut cx = Graph::new();
    let model = CodePredictorModel::new(&mut cx, config.clone());
    let input = cx.tensor((batch, seq, config.talker_hidden));
    let logits = model.forward(input).matmul(model.lm_heads[0].t()).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    rt.set_data(input.id, input_data.clone());
    rt.set_data(
        model.projection_weight.id,
        weights.projection_weight.clone(),
    );
    rt.set_data(model.projection_bias.id, weights.projection_bias.clone());
    rt.set_data(model.lm_heads[0].id, weights.lm_head0.clone());

    let layer = &model.layers[0];
    rt.set_data(layer.q_proj.id, weights.q_proj.clone());
    rt.set_data(layer.k_proj.id, weights.k_proj.clone());
    rt.set_data(layer.v_proj.id, weights.v_proj.clone());
    rt.set_data(
        layer.q_norm.weight.expect("q norm weight").id,
        weights.q_norm.clone(),
    );
    rt.set_data(
        layer.k_norm.weight.expect("k norm weight").id,
        weights.k_norm.clone(),
    );
    rt.set_data(layer.o_proj.id, weights.o_proj.clone());
    rt.set_data(layer.gate_proj.id, weights.gate_proj.clone());
    rt.set_data(layer.up_proj.id, weights.up_proj.clone());
    rt.set_data(layer.down_proj.id, weights.down_proj.clone());
    rt.set_data(
        layer.input_norm.weight.expect("input norm weight").id,
        weights.input_norm.clone(),
    );
    rt.set_data(
        layer
            .post_attn_norm
            .weight
            .expect("post attention norm weight")
            .id,
        weights.post_attn_norm.clone(),
    );
    rt.set_data(
        model.final_norm.weight.expect("final norm weight").id,
        weights.final_norm.clone(),
    );

    rt.execute(&cx.dyn_map);
    let luminal_out = rt.get_f32(logits.id).clone();

    let candle_out = candle_code_predictor_one_layer(&input_data, batch, seq, &config, &weights)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    assert_close(&luminal_out, &candle_out, TEST_EPS);
    Ok(())
}

#[test]
fn test_text_projection_vs_candle() -> CandleResult<()> {
    let hidden = 32;
    let batch = 1;
    let seq = 4;
    let mut rng = StdRng::seed_from_u64(42);

    let fc1_w = random_vec(&mut rng, hidden * hidden);
    let fc1_b = random_vec(&mut rng, hidden);
    let fc2_w = random_vec(&mut rng, hidden * hidden);
    let fc2_b = random_vec(&mut rng, hidden);
    let input_data = random_vec(&mut rng, batch * seq * hidden);

    let mut cx = Graph::new();
    let tp = TextProjection::new(hidden, &mut cx);
    let input = cx.tensor((batch, seq, hidden));
    let output = tp.forward(input).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    rt.set_data(input.id, input_data.clone());
    rt.set_data(tp.fc1_weight.id, fc1_w.clone());
    rt.set_data(tp.fc1_bias.id, fc1_b.clone());
    rt.set_data(tp.fc2_weight.id, fc2_w.clone());
    rt.set_data(tp.fc2_bias.id, fc2_b.clone());
    rt.execute(&cx.dyn_map);
    let luminal_out = rt.get_f32(output.id).clone();

    let device = Device::Cpu;
    let x = Tensor::from_vec(input_data, (batch, seq, hidden), &device)?;
    let w1 = Tensor::from_vec(fc1_w, (hidden, hidden), &device)?;
    let b1 = Tensor::from_vec(fc1_b, hidden, &device)?.reshape((1, 1, hidden))?;
    let w2 = Tensor::from_vec(fc2_w, (hidden, hidden), &device)?;
    let b2 = Tensor::from_vec(fc2_b, hidden, &device)?.reshape((1, 1, hidden))?;
    let h = linear_3d(&x, &w1)?.broadcast_add(&b1)?.silu()?;
    let candle_out = linear_3d(&h, &w2)?
        .broadcast_add(&b2)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    assert_close(&luminal_out, &candle_out, TEST_EPS);
    Ok(())
}

#[test]
fn test_embedding_sum_executes() {
    let config = TalkerConfig {
        layers: 1,
        hidden: 64,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        kv_groups: 2,
        intermediate: 128,
        vocab_size: 96,
        text_vocab_size: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };

    let mut cx = Graph::new();
    let talker = TalkerModel::new(&mut cx, config.clone());

    let text_ids = cx.tensor((1, 4)).as_dtype(DType::Int);
    let codec_ids = cx.tensor((1, 4)).as_dtype(DType::Int);
    let text_embeds = talker.embed_text(text_ids);
    let codec_embeds = talker.embed_codec(codec_ids);
    let summed = text_embeds + codec_embeds;
    let hidden = talker.forward_embeds(summed);
    let normed = talker.final_norm.forward(hidden);
    let logits = normed.matmul(talker.codec_head.t()).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    let mut rng = StdRng::seed_from_u64(101);
    rt.set_data(text_ids.id, vec![0i32, 1, 2, 3]);
    rt.set_data(codec_ids.id, vec![0i32, 1, 2, 3]);

    rt.set_data(
        talker.text_embedding.id,
        random_vec(&mut rng, config.text_vocab_size * config.hidden),
    );
    rt.set_data(
        talker.codec_embedding.id,
        random_vec(&mut rng, config.vocab_size * config.hidden),
    );
    rt.set_data(
        talker.text_projection.fc1_weight.id,
        random_vec(&mut rng, config.hidden * config.hidden),
    );
    rt.set_data(
        talker.text_projection.fc1_bias.id,
        random_vec(&mut rng, config.hidden),
    );
    rt.set_data(
        talker.text_projection.fc2_weight.id,
        random_vec(&mut rng, config.hidden * config.hidden),
    );
    rt.set_data(
        talker.text_projection.fc2_bias.id,
        random_vec(&mut rng, config.hidden),
    );
    rt.set_data(
        talker.codec_head.id,
        random_vec(&mut rng, config.vocab_size * config.hidden),
    );

    let layer = &talker.layers[0];
    rt.set_data(
        layer.q_proj.id,
        random_vec(&mut rng, config.n_heads * config.head_dim * config.hidden),
    );
    rt.set_data(
        layer.k_proj.id,
        random_vec(
            &mut rng,
            config.n_kv_heads * config.head_dim * config.hidden,
        ),
    );
    rt.set_data(
        layer.v_proj.id,
        random_vec(
            &mut rng,
            config.n_kv_heads * config.head_dim * config.hidden,
        ),
    );
    rt.set_data(
        layer.q_norm.weight.expect("q norm weight").id,
        random_vec(&mut rng, config.head_dim),
    );
    rt.set_data(
        layer.k_norm.weight.expect("k norm weight").id,
        random_vec(&mut rng, config.head_dim),
    );
    rt.set_data(
        layer.o_proj.id,
        random_vec(&mut rng, config.hidden * config.n_heads * config.head_dim),
    );
    rt.set_data(
        layer.gate_proj.id,
        random_vec(&mut rng, config.intermediate * config.hidden),
    );
    rt.set_data(
        layer.up_proj.id,
        random_vec(&mut rng, config.intermediate * config.hidden),
    );
    rt.set_data(
        layer.down_proj.id,
        random_vec(&mut rng, config.hidden * config.intermediate),
    );
    rt.set_data(
        layer.input_norm.weight.expect("input norm weight").id,
        random_vec(&mut rng, config.hidden),
    );
    rt.set_data(
        layer
            .post_attn_norm
            .weight
            .expect("post attention norm weight")
            .id,
        random_vec(&mut rng, config.hidden),
    );
    rt.set_data(
        talker.final_norm.weight.expect("final norm weight").id,
        random_vec(&mut rng, config.hidden),
    );

    rt.execute(&cx.dyn_map);
    let out = rt.get_f32(logits.id);
    assert_eq!(out.len(), 4 * config.vocab_size);
}

#[test]
fn test_decode_step() {
    let config = TalkerConfig {
        layers: 1,
        hidden: 64,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        kv_groups: 2,
        intermediate: 128,
        vocab_size: 96,
        text_vocab_size: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };

    let mut cx = Graph::new();
    let talker = TalkerModel::new(&mut cx, config.clone());
    let embeds = cx.tensor((1, 4, config.hidden));

    let (decode_logits, decode_normed) = talker.decode_step(embeds);
    let decode_logits = decode_logits.output();
    let decode_normed = decode_normed.output();

    let manual_hidden = talker.forward_embeds(embeds);
    let manual_normed = talker.final_norm.forward(manual_hidden);
    let manual_logits = manual_normed.matmul(talker.codec_head.t());
    let manual_normed = manual_normed.output();
    let manual_logits = manual_logits.output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    let mut rng = StdRng::seed_from_u64(2027);
    rt.set_data(embeds.id, random_vec(&mut rng, 1 * 4 * config.hidden));
    set_random_talker_params(&mut rt, &talker, &config, &mut rng);

    rt.execute(&cx.dyn_map);

    let decode_logits_data = rt.get_f32(decode_logits.id).clone();
    let decode_normed_data = rt.get_f32(decode_normed.id).clone();
    let manual_logits_data = rt.get_f32(manual_logits.id).clone();
    let manual_normed_data = rt.get_f32(manual_normed.id).clone();

    assert_eq!(decode_logits_data.len(), 1 * 4 * 96);
    assert_eq!(decode_normed_data.len(), 1 * 4 * 64);

    assert!(
        decode_logits_data.iter().all(|v| v.is_finite()),
        "decode logits contained NaN/Inf"
    );
    assert!(
        decode_normed_data.iter().all(|v| v.is_finite()),
        "decode normed hidden contained NaN/Inf"
    );

    assert_close(&decode_logits_data, &manual_logits_data, TEST_EPS);
    assert_close(&decode_normed_data, &manual_normed_data, TEST_EPS);
}

#[test]
fn test_prefill_kv_matches_full_forward() {
    let config = TalkerConfig {
        layers: 1,
        hidden: 64,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        kv_groups: 2,
        intermediate: 128,
        vocab_size: 96,
        text_vocab_size: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    let prompt_len = 4;

    let mut cx = Graph::new();
    let talker = TalkerModel::new(&mut cx, config.clone());
    let embeds = cx.tensor((1, prompt_len, config.hidden));

    let (prefill_logits, _prefill_normed, prefill_kv) = talker.prefill(embeds);
    let (prefill_k, prefill_v) = prefill_kv[0];
    let prefill_logits_out = prefill_logits.output();
    let prefill_k_out = prefill_k.output();
    let prefill_v_out = prefill_v.output();

    let (manual_hidden, manual_k, manual_v) = talker.layers[0].forward_with_kv(embeds, &config);
    let manual_logits = talker
        .final_norm
        .forward(manual_hidden)
        .matmul(talker.codec_head.t())
        .output();
    let manual_k_out = manual_k.output();
    let manual_v_out = manual_v.output();

    let (decode_logits, _decode_normed) = talker.decode_step(embeds);
    let decode_logits_out = decode_logits.output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    let mut rng = StdRng::seed_from_u64(2048);
    rt.set_data(embeds.id, random_vec(&mut rng, prompt_len * config.hidden));
    set_random_talker_params(&mut rt, &talker, &config, &mut rng);
    rt.execute(&cx.dyn_map);

    let prefill_logits_data = rt.get_f32(prefill_logits_out.id).clone();
    let decode_logits_data = rt.get_f32(decode_logits_out.id).clone();
    let manual_logits_data = rt.get_f32(manual_logits.id).clone();
    let prefill_k_data = rt.get_f32(prefill_k_out.id).clone();
    let prefill_v_data = rt.get_f32(prefill_v_out.id).clone();
    let manual_k_data = rt.get_f32(manual_k_out.id).clone();
    let manual_v_data = rt.get_f32(manual_v_out.id).clone();

    assert_eq!(prefill_logits_data.len(), prompt_len * config.vocab_size);
    assert_eq!(
        prefill_k_data.len(),
        config.n_kv_heads * prompt_len * config.head_dim
    );
    assert_eq!(
        prefill_v_data.len(),
        config.n_kv_heads * prompt_len * config.head_dim
    );

    assert_close(&prefill_logits_data, &decode_logits_data, TEST_EPS);
    assert_close(&prefill_logits_data, &manual_logits_data, TEST_EPS);
    assert_close(&prefill_k_data, &manual_k_data, TEST_EPS);
    assert_close(&prefill_v_data, &manual_v_data, TEST_EPS);
}

#[test]
fn test_decode_cached_single_step() {
    let config = TalkerConfig {
        layers: 1,
        hidden: 64,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        kv_groups: 2,
        intermediate: 128,
        vocab_size: 96,
        text_vocab_size: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    let prompt_len = 4;

    let mut cx = Graph::new();
    let talker = TalkerModel::new(&mut cx, config.clone());
    let prompt = cx.tensor((1, prompt_len, config.hidden));
    let new_embed = cx.tensor((1, 1, config.hidden));
    let pos = cx.tensor(1);

    let full_input = prompt.concat_along(new_embed, 1);
    let (full_logits, _full_normed) = talker.decode_step(full_input);
    let full_last_logits = full_logits.slice((.., prompt_len.., ..)).output();

    let (_prefill_logits, _prefill_normed, kv_caches) = talker.prefill(prompt);
    let (cached_logits, _cached_normed, _updated_kv) =
        talker.decode_cached(new_embed, &kv_caches, pos);
    let cached_logits_out = cached_logits.output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    let mut rng = StdRng::seed_from_u64(4096);
    rt.set_data(prompt.id, random_vec(&mut rng, prompt_len * config.hidden));
    rt.set_data(new_embed.id, random_vec(&mut rng, config.hidden));
    rt.set_data(pos.id, vec![prompt_len as f32]);
    set_random_talker_params(&mut rt, &talker, &config, &mut rng);
    rt.execute(&cx.dyn_map);

    let full_last_logits_data = rt.get_f32(full_last_logits.id).clone();
    let cached_logits_data = rt.get_f32(cached_logits_out.id).clone();

    assert_eq!(full_last_logits_data.len(), config.vocab_size);
    assert_eq!(cached_logits_data.len(), config.vocab_size);
    assert_close(&cached_logits_data, &full_last_logits_data, TEST_EPS);
}

#[test]
fn test_native_runtime_symbolic_concat_executes() {
    // Regression test for NativeRuntime symbolic indexing:
    // this graph used to panic in Gather due to incorrect symbolic Iota evaluation.
    let mut cx = Graph::new();
    let k_cache = cx.tensor((1, 1, 'p', 4));
    let k_new = cx.tensor((1, 1, 1, 4));
    let k_full = k_cache.concat_along(k_new, 2).output();

    cx.set_dim('p', 4);
    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    let cache: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let appended = vec![100.0f32, 101.0, 102.0, 103.0];
    rt.set_data(k_cache.id, cache.clone());
    rt.set_data(k_new.id, appended.clone());
    rt.execute(&cx.dyn_map);

    let mut expected = cache;
    expected.extend_from_slice(&appended);
    assert_eq!(rt.get_f32(k_full.id), &expected);
}

#[test]
fn test_native_runtime_symbolic_dim_reexecution_executes() {
    // Regression test for symbolic dim re-execution:
    // execute once with p=4, then re-execute with p=5 on the same compiled runtime.
    let mut cx = Graph::new();
    let k_cache = cx.tensor((1, 1, 'p', 4));
    let k_new = cx.tensor((1, 1, 1, 4));
    let k_full = k_cache.concat_along(k_new, 2).output();

    cx.set_dim('p', 4);
    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    let cache_first: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let appended_first = vec![100.0f32, 101.0, 102.0, 103.0];
    rt.set_data(k_cache.id, cache_first.clone());
    rt.set_data(k_new.id, appended_first.clone());
    rt.execute(&cx.dyn_map);

    let mut expected_first = cache_first;
    expected_first.extend_from_slice(&appended_first);
    assert_eq!(rt.get_f32(k_full.id), &expected_first);

    cx.set_dim('p', 5);
    let cache_second: Vec<f32> = (0..20).map(|i| 1000.0 + i as f32).collect();
    let appended_second = vec![2000.0f32, 2001.0, 2002.0, 2003.0];
    rt.set_data(k_cache.id, cache_second.clone());
    rt.set_data(k_new.id, appended_second.clone());
    rt.execute(&cx.dyn_map);

    let mut expected_second = cache_second;
    expected_second.extend_from_slice(&appended_second);
    assert_eq!(rt.get_f32(k_full.id), &expected_second);
}

#[test]
fn test_decode_fixed_matches_decode_cached() {
    let config = TalkerConfig {
        layers: 1,
        hidden: 64,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        kv_groups: 2,
        intermediate: 128,
        vocab_size: 96,
        text_vocab_size: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    let prompt_len = 4;
    let num_frames = 3;
    let max_seq = prompt_len + num_frames;

    let mut cx = Graph::new();
    let talker = TalkerModel::new(&mut cx, config.clone());
    let prompt = cx.tensor((1, prompt_len, config.hidden));
    let new_embed = cx.tensor((1, 1, config.hidden));
    let pos = cx.tensor(1);
    let fixed_k_buf = cx.tensor((1, config.n_kv_heads, max_seq, config.head_dim));
    let fixed_v_buf = cx.tensor((1, config.n_kv_heads, max_seq, config.head_dim));
    let fixed_mask = cx.tensor((1, 1, 1, max_seq + 1));

    let (_prefill_logits, _prefill_normed, kv_caches) = talker.prefill(prompt);
    let (prefill_k, prefill_v) = kv_caches[0];
    let prefill_k_out = prefill_k.output();
    let prefill_v_out = prefill_v.output();

    let (cached_logits, _cached_normed, _cached_kv) =
        talker.decode_cached(new_embed, &kv_caches, pos);
    let cached_logits_out = cached_logits.output();

    let fixed_kvs = vec![(fixed_k_buf, fixed_v_buf)];
    let (fixed_logits, _fixed_normed, _fixed_new_kv) =
        talker.decode_fixed(new_embed, &fixed_kvs, fixed_mask, pos);
    let fixed_logits_out = fixed_logits.output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    let mut rng = StdRng::seed_from_u64(8192);
    let prompt_data = random_vec(&mut rng, prompt_len * config.hidden);
    let new_embed_data = random_vec(&mut rng, config.hidden);
    set_random_talker_params(&mut rt, &talker, &config, &mut rng);

    let kv_buf_size = config.n_kv_heads * max_seq * config.head_dim;
    let mut fixed_k_data = vec![0.0f32; kv_buf_size];
    let mut fixed_v_data = vec![0.0f32; kv_buf_size];
    let mut mask_data = vec![-1e9f32; max_seq + 1];
    for v in mask_data.iter_mut().take(prompt_len) {
        *v = 0.0;
    }
    // Last index corresponds to concat'd K/V for the current token.
    mask_data[max_seq] = 0.0;

    rt.set_data(prompt.id, prompt_data.clone());
    rt.set_data(new_embed.id, new_embed_data.clone());
    rt.set_data(pos.id, vec![prompt_len as f32]);
    rt.set_data(fixed_k_buf.id, fixed_k_data.clone());
    rt.set_data(fixed_v_buf.id, fixed_v_data.clone());
    rt.set_data(fixed_mask.id, mask_data.clone());
    rt.execute(&cx.dyn_map);

    let prefill_k_data = rt.get_f32(prefill_k_out.id).clone();
    let prefill_v_data = rt.get_f32(prefill_v_out.id).clone();
    assert_eq!(
        prefill_k_data.len(),
        config.n_kv_heads * prompt_len * config.head_dim
    );
    assert_eq!(
        prefill_v_data.len(),
        config.n_kv_heads * prompt_len * config.head_dim
    );

    for head in 0..config.n_kv_heads {
        let src_offset = head * prompt_len * config.head_dim;
        let src_end = src_offset + prompt_len * config.head_dim;
        let dst_offset = head * max_seq * config.head_dim;
        let dst_end = dst_offset + prompt_len * config.head_dim;
        fixed_k_data[dst_offset..dst_end].copy_from_slice(&prefill_k_data[src_offset..src_end]);
        fixed_v_data[dst_offset..dst_end].copy_from_slice(&prefill_v_data[src_offset..src_end]);
    }

    rt.set_data(prompt.id, prompt_data);
    rt.set_data(new_embed.id, new_embed_data);
    rt.set_data(pos.id, vec![prompt_len as f32]);
    rt.set_data(fixed_k_buf.id, fixed_k_data);
    rt.set_data(fixed_v_buf.id, fixed_v_data);
    rt.set_data(fixed_mask.id, mask_data);
    rt.execute(&cx.dyn_map);

    let cached_logits_data = rt.get_f32(cached_logits_out.id).clone();
    let fixed_logits_data = rt.get_f32(fixed_logits_out.id).clone();

    assert_eq!(cached_logits_data.len(), config.vocab_size);
    assert_eq!(fixed_logits_data.len(), config.vocab_size);
    assert_close(&fixed_logits_data, &cached_logits_data, TEST_EPS);
}

#[test]
fn test_sample_greedy() {
    let logits = vec![
        0.1, 0.5, 0.3, 0.2, // max at index 1
        0.4, 0.1, 0.2, 0.8, // max at index 3
    ];
    let tokens = sample_greedy(&logits, 4);
    assert_eq!(tokens, vec![1, 3]);
}

#[test]
fn test_streaming_prompt_assembly() {
    let talker_config = TalkerConfig {
        layers: 1,
        hidden: 64,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        kv_groups: 2,
        intermediate: 128,
        vocab_size: 96,
        text_vocab_size: 256,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    let predictor_config = CodePredictorConfig {
        layers: 1,
        hidden: 32,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 16,
        kv_groups: 2,
        intermediate: 64,
        codebook_vocab: 48,
        num_code_groups: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        talker_hidden: 64,
    };

    let mut cx = Graph::new();
    let pipeline = TtsPipeline::new(&mut cx, talker_config.clone(), predictor_config);

    let role_text_ids = cx.tensor((1, 3)).as_dtype(DType::Int);
    let overlay_text_ids = cx.tensor((1, 5)).as_dtype(DType::Int);
    let overlay_codec_ids = cx.tensor((1, 5)).as_dtype(DType::Int);
    let transition_text_id = cx.tensor((1, 1)).as_dtype(DType::Int);
    let transition_codec_id = cx.tensor((1, 1)).as_dtype(DType::Int);
    let trailing_text_ids = cx.tensor((1, 3)).as_dtype(DType::Int);
    let tts_pad_text_id = cx.tensor((1, 1)).as_dtype(DType::Int);

    let outputs = pipeline.assemble_streaming_prompt(&StreamingPromptInputs {
        role_text_ids,
        overlay_text_ids,
        overlay_codec_ids,
        transition_text_id,
        transition_codec_id,
        trailing_text_ids,
        tts_pad_text_id,
    });
    let initial_embeds = outputs.initial_embeds.output();
    let trailing_text_hidden = outputs.trailing_text_hidden.output();
    let tts_pad_embed = outputs.tts_pad_embed.output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    rt.set_data(role_text_ids.id, vec![0i32, 1, 2]);
    rt.set_data(overlay_text_ids.id, vec![10i32, 10, 10, 10, 11]);
    rt.set_data(overlay_codec_ids.id, vec![4i32, 5, 6, 7, 8]);
    rt.set_data(transition_text_id.id, vec![20i32]);
    rt.set_data(transition_codec_id.id, vec![9i32]);
    rt.set_data(trailing_text_ids.id, vec![21i32, 22, 12]);
    rt.set_data(tts_pad_text_id.id, vec![10i32]);

    let mut rng = StdRng::seed_from_u64(777);
    rt.set_data(
        pipeline.talker.text_embedding.id,
        random_vec(
            &mut rng,
            talker_config.text_vocab_size * talker_config.hidden,
        ),
    );
    rt.set_data(
        pipeline.talker.codec_embedding.id,
        random_vec(&mut rng, talker_config.vocab_size * talker_config.hidden),
    );
    rt.set_data(
        pipeline.talker.text_projection.fc1_weight.id,
        random_vec(&mut rng, talker_config.hidden * talker_config.hidden),
    );
    rt.set_data(
        pipeline.talker.text_projection.fc1_bias.id,
        random_vec(&mut rng, talker_config.hidden),
    );
    rt.set_data(
        pipeline.talker.text_projection.fc2_weight.id,
        random_vec(&mut rng, talker_config.hidden * talker_config.hidden),
    );
    rt.set_data(
        pipeline.talker.text_projection.fc2_bias.id,
        random_vec(&mut rng, talker_config.hidden),
    );
    rt.set_data(
        pipeline.talker.codec_head.id,
        random_vec(&mut rng, talker_config.vocab_size * talker_config.hidden),
    );

    let layer = &pipeline.talker.layers[0];
    rt.set_data(
        layer.q_proj.id,
        random_vec(
            &mut rng,
            talker_config.n_heads * talker_config.head_dim * talker_config.hidden,
        ),
    );
    rt.set_data(
        layer.k_proj.id,
        random_vec(
            &mut rng,
            talker_config.n_kv_heads * talker_config.head_dim * talker_config.hidden,
        ),
    );
    rt.set_data(
        layer.v_proj.id,
        random_vec(
            &mut rng,
            talker_config.n_kv_heads * talker_config.head_dim * talker_config.hidden,
        ),
    );
    rt.set_data(
        layer.q_norm.weight.expect("q norm weight").id,
        random_vec(&mut rng, talker_config.head_dim),
    );
    rt.set_data(
        layer.k_norm.weight.expect("k norm weight").id,
        random_vec(&mut rng, talker_config.head_dim),
    );
    rt.set_data(
        layer.o_proj.id,
        random_vec(
            &mut rng,
            talker_config.hidden * talker_config.n_heads * talker_config.head_dim,
        ),
    );
    rt.set_data(
        layer.gate_proj.id,
        random_vec(&mut rng, talker_config.intermediate * talker_config.hidden),
    );
    rt.set_data(
        layer.up_proj.id,
        random_vec(&mut rng, talker_config.intermediate * talker_config.hidden),
    );
    rt.set_data(
        layer.down_proj.id,
        random_vec(&mut rng, talker_config.hidden * talker_config.intermediate),
    );
    rt.set_data(
        layer.input_norm.weight.expect("input norm weight").id,
        random_vec(&mut rng, talker_config.hidden),
    );
    rt.set_data(
        layer
            .post_attn_norm
            .weight
            .expect("post attention norm weight")
            .id,
        random_vec(&mut rng, talker_config.hidden),
    );
    rt.set_data(
        pipeline
            .talker
            .final_norm
            .weight
            .expect("final norm weight")
            .id,
        random_vec(&mut rng, talker_config.hidden),
    );

    rt.execute(&cx.dyn_map);

    assert_eq!(
        rt.get_f32(initial_embeds.id).len(),
        9 * talker_config.hidden
    );
    assert_eq!(
        rt.get_f32(trailing_text_hidden.id).len(),
        3 * talker_config.hidden
    );
    assert_eq!(rt.get_f32(tts_pad_embed.id).len(), talker_config.hidden);
}

#[test]
fn test_assemble_nonstreaming_prompt() {
    let talker_config = TalkerConfig {
        layers: 1,
        hidden: 64,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        kv_groups: 2,
        intermediate: 128,
        vocab_size: 96,
        text_vocab_size: 256,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    let predictor_config = CodePredictorConfig {
        layers: 1,
        hidden: 32,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 16,
        kv_groups: 2,
        intermediate: 64,
        codebook_vocab: 48,
        num_code_groups: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        talker_hidden: 64,
    };

    let mut cx = Graph::new();
    let pipeline = TtsPipeline::new(&mut cx, talker_config.clone(), predictor_config);

    let instruct_ids = cx.tensor((1, 2));
    let role_ids = cx.tensor((1, 3));
    let overlay_text_ids = cx.tensor((1, 5));
    let overlay_codec_ids = cx.tensor((1, 5));
    let content_ids = cx.tensor((1, 2));
    let content_codec_ids = cx.tensor((1, 3));
    let tts_eos_id = cx.tensor((1, 1));
    let transition_text_id = cx.tensor((1, 1));
    let transition_codec_id = cx.tensor((1, 1));

    let outputs = pipeline.assemble_nonstreaming_prompt(&NonStreamingPromptInputs {
        instruct_ids: Some(instruct_ids),
        role_ids,
        overlay_text_ids,
        overlay_codec_ids,
        content_ids,
        content_codec_ids,
        tts_eos_id,
        transition_text_id,
        transition_codec_id,
    });
    let initial_embeds = outputs.initial_embeds.output();
    let tts_pad_embed = outputs.tts_pad_embed.output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    rt.set_data(instruct_ids.id, vec![30.0f32, 31.0]);
    rt.set_data(role_ids.id, vec![0.0f32, 1.0, 2.0]);
    rt.set_data(overlay_text_ids.id, vec![10.0f32, 10.0, 10.0, 10.0, 11.0]);
    rt.set_data(overlay_codec_ids.id, vec![4.0f32, 5.0, 6.0, 7.0, 8.0]);
    rt.set_data(content_ids.id, vec![21.0f32, 22.0]);
    rt.set_data(content_codec_ids.id, vec![9.0f32, 9.0, 9.0]);
    rt.set_data(tts_eos_id.id, vec![12.0f32]);
    rt.set_data(transition_text_id.id, vec![10.0f32]);
    rt.set_data(transition_codec_id.id, vec![9.0f32]);

    let mut rng = StdRng::seed_from_u64(1701);
    set_random_talker_params(&mut rt, &pipeline.talker, &talker_config, &mut rng);

    rt.execute(&cx.dyn_map);

    let initial_data = rt.get_f32(initial_embeds.id);
    let tts_pad_data = rt.get_f32(tts_pad_embed.id);
    let expected_prompt_len = 2 + 3 + 5 + 3 + 1;

    assert_eq!(
        initial_data.len(),
        expected_prompt_len * talker_config.hidden
    );
    assert_eq!(tts_pad_data.len(), talker_config.hidden);
    assert!(
        initial_data.iter().all(|v| v.is_finite()),
        "non-streaming initial embeds contained NaN/Inf"
    );
    assert!(
        tts_pad_data.iter().all(|v| v.is_finite()),
        "non-streaming tts pad embed contained NaN/Inf"
    );
    assert!(
        initial_data.iter().any(|v| v.abs() > 1e-8),
        "non-streaming initial embeds should not be all zeros"
    );

    let mut initial_shape = initial_embeds.shape;
    initial_shape.resolve_dyn_dims(&cx.dyn_map);
    assert_eq!(
        initial_shape.shape_usize(),
        vec![1, expected_prompt_len, talker_config.hidden]
    );

    let mut tts_shape = tts_pad_embed.shape;
    tts_shape.resolve_dyn_dims(&cx.dyn_map);
    assert_eq!(tts_shape.shape_usize(), vec![1, 1, talker_config.hidden]);
}

#[test]
fn test_pipeline_end_to_end_shapes() {
    let talker_config = TalkerConfig {
        layers: 1,
        hidden: 64,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        kv_groups: 2,
        intermediate: 128,
        vocab_size: 96,
        text_vocab_size: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    let predictor_config = CodePredictorConfig {
        hidden: 32,
        head_dim: 16,
        n_heads: 4,
        n_kv_heads: 2,
        kv_groups: 2,
        intermediate: 64,
        layers: 1,
        codebook_vocab: 48,
        num_code_groups: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        talker_hidden: 64,
    };

    let mut cx = Graph::new();
    let pipeline = TtsPipeline::new(&mut cx, talker_config.clone(), predictor_config.clone());

    let prompt_embeds = cx.tensor((1, 6, talker_config.hidden));
    let predictor_input = cx.tensor((1, 2, predictor_config.talker_hidden));

    let talker_hidden = pipeline.talker.forward_embeds(prompt_embeds);
    let talker_normed = pipeline.talker.final_norm.forward(talker_hidden);
    let first_logits = talker_normed
        .matmul(pipeline.talker.codec_head.t())
        .output();

    let pred_hidden = pipeline.code_predictor.forward(predictor_input);
    let second_logits = pred_hidden
        .matmul(pipeline.code_predictor.lm_heads[0].t())
        .output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    let mut rng = StdRng::seed_from_u64(2026);
    rt.set_data(
        prompt_embeds.id,
        random_vec(&mut rng, 6 * talker_config.hidden),
    );
    rt.set_data(
        predictor_input.id,
        random_vec(&mut rng, 2 * predictor_config.talker_hidden),
    );

    let talker_layer = &pipeline.talker.layers[0];
    rt.set_data(
        talker_layer.q_proj.id,
        random_vec(
            &mut rng,
            talker_config.n_heads * talker_config.head_dim * talker_config.hidden,
        ),
    );
    rt.set_data(
        talker_layer.k_proj.id,
        random_vec(
            &mut rng,
            talker_config.n_kv_heads * talker_config.head_dim * talker_config.hidden,
        ),
    );
    rt.set_data(
        talker_layer.v_proj.id,
        random_vec(
            &mut rng,
            talker_config.n_kv_heads * talker_config.head_dim * talker_config.hidden,
        ),
    );
    rt.set_data(
        talker_layer.q_norm.weight.expect("talker q norm weight").id,
        random_vec(&mut rng, talker_config.head_dim),
    );
    rt.set_data(
        talker_layer.k_norm.weight.expect("talker k norm weight").id,
        random_vec(&mut rng, talker_config.head_dim),
    );
    rt.set_data(
        talker_layer.o_proj.id,
        random_vec(
            &mut rng,
            talker_config.hidden * talker_config.n_heads * talker_config.head_dim,
        ),
    );
    rt.set_data(
        talker_layer.gate_proj.id,
        random_vec(&mut rng, talker_config.intermediate * talker_config.hidden),
    );
    rt.set_data(
        talker_layer.up_proj.id,
        random_vec(&mut rng, talker_config.intermediate * talker_config.hidden),
    );
    rt.set_data(
        talker_layer.down_proj.id,
        random_vec(&mut rng, talker_config.hidden * talker_config.intermediate),
    );
    rt.set_data(
        talker_layer
            .input_norm
            .weight
            .expect("input norm weight")
            .id,
        random_vec(&mut rng, talker_config.hidden),
    );
    rt.set_data(
        talker_layer
            .post_attn_norm
            .weight
            .expect("post attention norm weight")
            .id,
        random_vec(&mut rng, talker_config.hidden),
    );
    rt.set_data(
        pipeline
            .talker
            .final_norm
            .weight
            .expect("talker final norm weight")
            .id,
        random_vec(&mut rng, talker_config.hidden),
    );
    rt.set_data(
        pipeline.talker.codec_head.id,
        random_vec(&mut rng, talker_config.vocab_size * talker_config.hidden),
    );

    rt.set_data(
        pipeline.code_predictor.projection_weight.id,
        random_vec(
            &mut rng,
            predictor_config.hidden * predictor_config.talker_hidden,
        ),
    );
    rt.set_data(
        pipeline.code_predictor.projection_bias.id,
        random_vec(&mut rng, predictor_config.hidden),
    );
    let predictor_layer = &pipeline.code_predictor.layers[0];
    rt.set_data(
        predictor_layer.q_proj.id,
        random_vec(
            &mut rng,
            predictor_config.n_heads * predictor_config.head_dim * predictor_config.hidden,
        ),
    );
    rt.set_data(
        predictor_layer.k_proj.id,
        random_vec(
            &mut rng,
            predictor_config.n_kv_heads * predictor_config.head_dim * predictor_config.hidden,
        ),
    );
    rt.set_data(
        predictor_layer.v_proj.id,
        random_vec(
            &mut rng,
            predictor_config.n_kv_heads * predictor_config.head_dim * predictor_config.hidden,
        ),
    );
    rt.set_data(
        predictor_layer
            .q_norm
            .weight
            .expect("predictor q norm weight")
            .id,
        random_vec(&mut rng, predictor_config.head_dim),
    );
    rt.set_data(
        predictor_layer
            .k_norm
            .weight
            .expect("predictor k norm weight")
            .id,
        random_vec(&mut rng, predictor_config.head_dim),
    );
    rt.set_data(
        predictor_layer.o_proj.id,
        random_vec(
            &mut rng,
            predictor_config.hidden * predictor_config.n_heads * predictor_config.head_dim,
        ),
    );
    rt.set_data(
        predictor_layer.gate_proj.id,
        random_vec(
            &mut rng,
            predictor_config.intermediate * predictor_config.hidden,
        ),
    );
    rt.set_data(
        predictor_layer.up_proj.id,
        random_vec(
            &mut rng,
            predictor_config.intermediate * predictor_config.hidden,
        ),
    );
    rt.set_data(
        predictor_layer.down_proj.id,
        random_vec(
            &mut rng,
            predictor_config.hidden * predictor_config.intermediate,
        ),
    );
    rt.set_data(
        predictor_layer
            .input_norm
            .weight
            .expect("predictor input norm weight")
            .id,
        random_vec(&mut rng, predictor_config.hidden),
    );
    rt.set_data(
        predictor_layer
            .post_attn_norm
            .weight
            .expect("predictor post attention norm weight")
            .id,
        random_vec(&mut rng, predictor_config.hidden),
    );
    rt.set_data(
        pipeline
            .code_predictor
            .final_norm
            .weight
            .expect("predictor final norm weight")
            .id,
        random_vec(&mut rng, predictor_config.hidden),
    );
    rt.set_data(
        pipeline.code_predictor.lm_heads[0].id,
        random_vec(
            &mut rng,
            predictor_config.codebook_vocab * predictor_config.hidden,
        ),
    );

    rt.execute(&cx.dyn_map);
    assert_eq!(
        rt.get_f32(first_logits.id).len(),
        6 * talker_config.vocab_size
    );
    assert_eq!(
        rt.get_f32(second_logits.id).len(),
        2 * predictor_config.codebook_vocab
    );
}

#[test]
fn test_single_frame_generation_flow() {
    let talker_config = TalkerConfig {
        layers: 1,
        hidden: 64,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        kv_groups: 2,
        intermediate: 128,
        vocab_size: 96,
        text_vocab_size: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    let predictor_config = CodePredictorConfig {
        hidden: 32,
        head_dim: 16,
        n_heads: 4,
        n_kv_heads: 2,
        kv_groups: 2,
        intermediate: 64,
        layers: 1,
        codebook_vocab: 48,
        num_code_groups: 3,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        talker_hidden: 64,
    };

    let mut cx = Graph::new();
    let pipeline = TtsPipeline::new(&mut cx, talker_config.clone(), predictor_config.clone());

    let prompt_embeds = cx.tensor((1, 6, talker_config.hidden));
    let talker_hidden = pipeline.talker.forward_embeds(prompt_embeds);
    let talker_normed = pipeline.talker.final_norm.forward(talker_hidden);

    let talker_logits = talker_normed.matmul(pipeline.talker.codec_head.t());
    let last_logits = talker_logits.slice((.., 5.., ..));
    let code_0 = last_logits.argmax(2);

    let code_0_embed = pipeline.talker.embed_codec(code_0);
    let last_hidden = talker_normed.slice((.., 5.., ..));
    let predictor_input = last_hidden.concat_along(code_0_embed, 1);

    let pred_hidden = pipeline.code_predictor.forward(predictor_input);
    let pred_logits_0 = pipeline.code_predictor.logits_for_group(pred_hidden, 0);
    let last_pred_logits = pred_logits_0.slice((.., 1.., ..));
    let code_1 = last_pred_logits.argmax(2);

    let code_1_embed = pipeline.code_predictor.embed_for_group(code_1, 0);
    let codec_sum = (code_0_embed + code_1_embed).output();
    let talker_logits_out = talker_logits.output();
    let pred_logits_out = pred_logits_0.output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    let mut rng = StdRng::seed_from_u64(111);
    rt.set_data(
        prompt_embeds.id,
        random_vec(&mut rng, 6 * talker_config.hidden),
    );
    set_random_talker_params(&mut rt, &pipeline.talker, &talker_config, &mut rng);
    set_random_code_predictor_params(
        &mut rt,
        &pipeline.code_predictor,
        &predictor_config,
        &mut rng,
    );

    rt.execute(&cx.dyn_map);

    let talker_logits_data = rt.get_f32(talker_logits_out.id).clone();
    let pred_logits_data = rt.get_f32(pred_logits_out.id).clone();
    let codec_sum_data = rt.get_f32(codec_sum.id).clone();

    assert_eq!(talker_logits_data.len(), 1 * 6 * 96);
    assert_eq!(pred_logits_data.len(), 1 * 2 * 48);
    assert_eq!(codec_sum_data.len(), 1 * 1 * 64);

    assert!(
        talker_logits_data.iter().all(|v| v.is_finite()),
        "talker logits contained NaN/Inf"
    );
    assert!(
        pred_logits_data.iter().all(|v| v.is_finite()),
        "predictor logits contained NaN/Inf"
    );
    assert!(
        codec_sum_data.iter().all(|v| v.is_finite()),
        "codec embedding sum contained NaN/Inf"
    );
}

#[test]
fn test_two_frame_generation() {
    let talker_config = TalkerConfig {
        layers: 1,
        hidden: 64,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        kv_groups: 2,
        intermediate: 128,
        vocab_size: 96,
        text_vocab_size: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    let predictor_config = CodePredictorConfig {
        hidden: 32,
        head_dim: 16,
        n_heads: 4,
        n_kv_heads: 2,
        kv_groups: 2,
        intermediate: 64,
        layers: 1,
        codebook_vocab: 48,
        num_code_groups: 3,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        talker_hidden: 64,
    };

    let mut cx0 = Graph::new();
    let pipeline0 = TtsPipeline::new(&mut cx0, talker_config.clone(), predictor_config.clone());
    let prompt0 = cx0.tensor((1, 6, talker_config.hidden));
    let prompt0_out = prompt0.output();

    let (logits0, normed0) = pipeline0.talker.decode_step(prompt0);
    let logits0_out = logits0.output();

    let last_logits0 = logits0.slice((.., 5.., ..));
    let code0 = last_logits0.argmax(2);
    let code0_embed = pipeline0.talker.embed_codec(code0);

    let last_hidden0 = normed0.slice((.., 5.., ..));
    let pred_out0 = pipeline0
        .code_predictor
        .generate_codes(last_hidden0, code0_embed);

    let mut codec_sum0 = code0_embed;
    for embed in &pred_out0.embeds {
        codec_sum0 = codec_sum0 + *embed;
    }
    let codec_sum0_out = codec_sum0.output();

    cx0.build_search_space::<NativeRuntime>();
    let mut rt0 = cx0.search(NativeRuntime::default(), 1);

    let mut rng = StdRng::seed_from_u64(42);
    rt0.set_data(prompt0.id, random_vec(&mut rng, 6 * talker_config.hidden));
    let rng_before_weights = rng.clone();
    set_random_talker_params(&mut rt0, &pipeline0.talker, &talker_config, &mut rng);
    set_random_code_predictor_params(
        &mut rt0,
        &pipeline0.code_predictor,
        &predictor_config,
        &mut rng,
    );
    rt0.execute(&cx0.dyn_map);

    let logits0_data = rt0.get_f32(logits0_out.id).clone();
    let codec_sum0_data = rt0.get_f32(codec_sum0_out.id).clone();

    assert_eq!(logits0_data.len(), 6 * talker_config.vocab_size);
    assert_eq!(codec_sum0_data.len(), talker_config.hidden);
    assert!(
        logits0_data.iter().all(|v| v.is_finite()),
        "frame 0 talker logits contained NaN/Inf"
    );
    assert!(
        codec_sum0_data.iter().all(|v| v.is_finite()),
        "frame 0 codec sum contained NaN/Inf"
    );

    let sampled0 = sample_greedy(
        &logits0_data[5 * talker_config.vocab_size..],
        talker_config.vocab_size,
    );
    assert_eq!(sampled0.len(), 1);
    assert!(
        (sampled0[0] as usize) < talker_config.vocab_size,
        "sampled token {} out of range {}",
        sampled0[0],
        talker_config.vocab_size
    );

    let mut cx1 = Graph::new();
    let pipeline1 = TtsPipeline::new(&mut cx1, talker_config.clone(), predictor_config.clone());
    let prompt1 = cx1.tensor((1, 7, talker_config.hidden));

    let (logits1, normed1) = pipeline1.talker.decode_step(prompt1);
    let logits1_out = logits1.output();

    let last_logits1 = logits1.slice((.., 6.., ..));
    let code1_0 = last_logits1.argmax(2);
    let code1_0_embed = pipeline1.talker.embed_codec(code1_0);

    let last_hidden1 = normed1.slice((.., 6.., ..));
    let pred_out1 = pipeline1
        .code_predictor
        .generate_codes(last_hidden1, code1_0_embed);

    let mut codec_sum1 = code1_0_embed;
    for embed in &pred_out1.embeds {
        codec_sum1 = codec_sum1 + *embed;
    }
    let codec_sum1_out = codec_sum1.output();

    cx1.build_search_space::<NativeRuntime>();
    let mut rt1 = cx1.search(NativeRuntime::default(), 1);

    let mut prompt1_data = rt0.get_f32(prompt0_out.id).clone();
    prompt1_data.extend_from_slice(&codec_sum0_data);
    assert_eq!(prompt1_data.len(), 7 * talker_config.hidden);
    rt1.set_data(prompt1.id, prompt1_data);

    let mut rng1 = rng_before_weights.clone();
    set_random_talker_params(&mut rt1, &pipeline1.talker, &talker_config, &mut rng1);
    set_random_code_predictor_params(
        &mut rt1,
        &pipeline1.code_predictor,
        &predictor_config,
        &mut rng1,
    );
    rt1.execute(&cx1.dyn_map);

    let logits1_data = rt1.get_f32(logits1_out.id).clone();
    let codec_sum1_data = rt1.get_f32(codec_sum1_out.id).clone();

    assert_eq!(logits1_data.len(), 7 * talker_config.vocab_size);
    assert_eq!(codec_sum1_data.len(), talker_config.hidden);
    assert!(
        logits1_data.iter().all(|v| v.is_finite()),
        "frame 1 talker logits contained NaN/Inf"
    );
    assert!(
        codec_sum1_data.iter().all(|v| v.is_finite()),
        "frame 1 codec sum contained NaN/Inf"
    );
    assert_ne!(
        codec_sum0_data, codec_sum1_data,
        "frame 0 and frame 1 codec sums should differ"
    );
}

#[test]
fn test_generate_frames() {
    let talker_config = TalkerConfig {
        layers: 1,
        hidden: 64,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        kv_groups: 2,
        intermediate: 128,
        vocab_size: 96,
        text_vocab_size: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    let predictor_config = CodePredictorConfig {
        hidden: 32,
        head_dim: 16,
        n_heads: 4,
        n_kv_heads: 2,
        kv_groups: 2,
        intermediate: 64,
        layers: 1,
        codebook_vocab: 48,
        num_code_groups: 3,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        talker_hidden: 64,
    };

    let prompt_len = 6;
    let num_frames = 3;

    let mut rng = StdRng::seed_from_u64(99);
    let weights = build_random_weight_map(&talker_config, &predictor_config, &mut rng);
    let initial_embeds = random_vec(&mut rng, prompt_len * talker_config.hidden);
    let tts_pad_embed = random_vec(&mut rng, talker_config.hidden);

    let frames = generate_frames(
        &talker_config,
        &predictor_config,
        &initial_embeds,
        &tts_pad_embed,
        prompt_len,
        num_frames,
        &weights,
    );

    assert_eq!(frames.len(), num_frames);
    for (i, frame) in frames.iter().enumerate() {
        assert_eq!(frame.len(), predictor_config.num_code_groups, "frame {i}");
        assert!((frame[0] as usize) < talker_config.vocab_size);
        for &code in &frame[1..] {
            assert!((code as usize) < predictor_config.codebook_vocab);
        }
    }

    let mut rng2 = StdRng::seed_from_u64(99);
    let weights2 = build_random_weight_map(&talker_config, &predictor_config, &mut rng2);
    let initial_embeds2 = random_vec(&mut rng2, prompt_len * talker_config.hidden);
    let tts_pad_embed2 = random_vec(&mut rng2, talker_config.hidden);
    let frames2 = generate_frames(
        &talker_config,
        &predictor_config,
        &initial_embeds2,
        &tts_pad_embed2,
        prompt_len,
        num_frames,
        &weights2,
    );
    assert_eq!(frames, frames2, "generation should be deterministic");
}

#[test]
fn test_predictor_per_group_logits() {
    let config = CodePredictorConfig {
        hidden: 32,
        head_dim: 16,
        n_heads: 4,
        n_kv_heads: 2,
        kv_groups: 2,
        intermediate: 64,
        layers: 1,
        codebook_vocab: 48,
        num_code_groups: 4,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        talker_hidden: 64,
    };

    let batch = 1;
    let seq = 2;

    let mut cx = Graph::new();
    let model = CodePredictorModel::new(&mut cx, config.clone());
    let input_embeds = cx.tensor((batch, seq, config.talker_hidden));
    let hidden = model.forward(input_embeds);

    let logits0 = model.logits_for_group(hidden, 0).output();
    let logits1 = model.logits_for_group(hidden, 1).output();
    let logits2 = model.logits_for_group(hidden, 2).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    let mut rng = StdRng::seed_from_u64(99);
    rt.set_data(
        input_embeds.id,
        random_vec(&mut rng, batch * seq * config.talker_hidden),
    );
    set_random_code_predictor_params(&mut rt, &model, &config, &mut rng);

    rt.execute(&cx.dyn_map);

    let out0 = rt.get_f32(logits0.id).clone();
    let out1 = rt.get_f32(logits1.id).clone();
    let out2 = rt.get_f32(logits2.id).clone();

    assert_eq!(out0.len(), batch * seq * config.codebook_vocab);
    assert_eq!(out1.len(), batch * seq * config.codebook_vocab);
    assert_eq!(out2.len(), batch * seq * config.codebook_vocab);

    assert_ne!(out0, out1);
    assert_ne!(out0, out2);
    assert_ne!(out1, out2);
}

#[test]
fn test_predictor_per_group_embedding() {
    let config = CodePredictorConfig {
        hidden: 32,
        head_dim: 16,
        n_heads: 4,
        n_kv_heads: 2,
        kv_groups: 2,
        intermediate: 64,
        layers: 1,
        codebook_vocab: 48,
        num_code_groups: 4,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        talker_hidden: 64,
    };

    let batch = 1;
    let seq = 3;

    let mut cx = Graph::new();
    let model = CodePredictorModel::new(&mut cx, config.clone());
    let code_ids = cx.tensor((batch, seq)).as_dtype(DType::Int);

    let emb0 = model.embed_for_group(code_ids, 0).output();
    let emb1 = model.embed_for_group(code_ids, 1).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    let mut rng = StdRng::seed_from_u64(100);
    rt.set_data(code_ids.id, vec![0i32, 1, 2]);
    set_random_code_predictor_params(&mut rt, &model, &config, &mut rng);

    rt.execute(&cx.dyn_map);

    let out0 = rt.get_f32(emb0.id).clone();
    let out1 = rt.get_f32(emb1.id).clone();

    assert_eq!(out0.len(), batch * seq * config.talker_hidden);
    assert_eq!(out1.len(), batch * seq * config.talker_hidden);
    assert_ne!(out0, out1);
}

#[test]
fn test_predictor_generate_codes() {
    let config = CodePredictorConfig {
        hidden: 32,
        head_dim: 16,
        n_heads: 4,
        n_kv_heads: 2,
        kv_groups: 2,
        intermediate: 64,
        layers: 1,
        codebook_vocab: 48,
        num_code_groups: 4,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        talker_hidden: 64,
    };

    let mut cx = Graph::new();
    let model = CodePredictorModel::new(&mut cx, config.clone());

    let talker_hidden = cx.tensor((1, 1, config.talker_hidden));
    let code_0_embed = cx.tensor((1, 1, config.talker_hidden));
    let out = model.generate_codes(talker_hidden, code_0_embed);

    assert_eq!(out.embeds.len(), config.num_code_groups - 1);
    assert_eq!(out.codes.len(), config.num_code_groups - 1);
    let embed0 = out.embeds[0].output();
    let embed1 = out.embeds[1].output();
    let embed2 = out.embeds[2].output();
    let code0 = out.codes[0].cast(DType::F32).output();
    let code1 = out.codes[1].cast(DType::F32).output();
    let code2 = out.codes[2].cast(DType::F32).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    let mut rng = StdRng::seed_from_u64(101);
    rt.set_data(talker_hidden.id, random_vec(&mut rng, config.talker_hidden));
    rt.set_data(code_0_embed.id, random_vec(&mut rng, config.talker_hidden));
    set_random_code_predictor_params(&mut rt, &model, &config, &mut rng);

    rt.execute(&cx.dyn_map);

    let out0 = rt.get_f32(embed0.id).clone();
    let out1 = rt.get_f32(embed1.id).clone();
    let out2 = rt.get_f32(embed2.id).clone();
    let code0_val = rt.get_f32(code0.id)[0] as usize;
    let code1_val = rt.get_f32(code1.id)[0] as usize;
    let code2_val = rt.get_f32(code2.id)[0] as usize;

    assert_eq!(out0.len(), 1 * 1 * config.talker_hidden);
    assert_eq!(out1.len(), 1 * 1 * config.talker_hidden);
    assert_eq!(out2.len(), 1 * 1 * config.talker_hidden);

    assert!(
        out0.iter().all(|v| v.is_finite()),
        "group 0 embedding contained NaN/Inf"
    );
    assert!(
        out1.iter().all(|v| v.is_finite()),
        "group 1 embedding contained NaN/Inf"
    );
    assert!(
        out2.iter().all(|v| v.is_finite()),
        "group 2 embedding contained NaN/Inf"
    );
    assert!(
        !(out0 == out1 && out1 == out2),
        "all generated predictor embeddings were identical"
    );
    assert!(code0_val < config.codebook_vocab);
    assert!(code1_val < config.codebook_vocab);
    assert!(code2_val < config.codebook_vocab);
}

fn random_positive_vec(rng: &mut StdRng, n: usize, min: f32, max: f32) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(min..max)).collect()
}

fn small_speech_decoder_config() -> SpeechDecoderConfig {
    SpeechDecoderConfig {
        codebook_dim: 8,
        vq_dim: 4,
        latent_dim: 16,
        hidden_size: 8,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        intermediate_size: 16,
        num_hidden_layers: 1,
        head_dim: 4,
        semantic_codebook_size: 8,
        acoustic_codebook_size: 8,
        decoder_dim: 32,
        upsample_rates: vec![2, 2],
        upsampling_ratios: vec![2, 2],
        sliding_window: 4,
        ..SpeechDecoderConfig::default()
    }
}

fn set_random_rvq_params(
    rt: &mut NativeRuntime,
    quantizer: &SplitResidualVectorQuantizer,
    config: &SpeechDecoderConfig,
    rng: &mut StdRng,
) {
    for codebook in &quantizer.rvq_first.codebooks {
        rt.set_data(
            codebook.embedding_sum.id,
            random_vec(&mut *rng, config.semantic_codebook_size * config.vq_dim),
        );
        rt.set_data(
            codebook.cluster_usage.id,
            random_positive_vec(&mut *rng, config.semantic_codebook_size, 0.1, 2.0),
        );
    }
    for codebook in &quantizer.rvq_rest.codebooks {
        rt.set_data(
            codebook.embedding_sum.id,
            random_vec(&mut *rng, config.acoustic_codebook_size * config.vq_dim),
        );
        rt.set_data(
            codebook.cluster_usage.id,
            random_positive_vec(&mut *rng, config.acoustic_codebook_size, 0.1, 2.0),
        );
    }
    rt.set_data(
        quantizer.rvq_first.output_proj_weight.id,
        random_vec(&mut *rng, config.codebook_dim * config.vq_dim),
    );
    rt.set_data(
        quantizer.rvq_rest.output_proj_weight.id,
        random_vec(&mut *rng, config.codebook_dim * config.vq_dim),
    );
}

fn set_random_pre_transformer_params(
    rt: &mut NativeRuntime,
    pre_transformer: &PreTransformer,
    config: &SpeechDecoderConfig,
    rng: &mut StdRng,
) {
    let attn_proj_dim = config.num_attention_heads * config.head_dim;

    rt.set_data(
        pre_transformer.input_proj_weight.id,
        random_vec(&mut *rng, config.hidden_size * config.latent_dim),
    );
    rt.set_data(
        pre_transformer.input_proj_bias.id,
        random_vec(&mut *rng, config.hidden_size),
    );

    for layer in &pre_transformer.layers {
        rt.set_data(
            layer
                .input_layernorm
                .weight
                .expect("pre-transformer input layernorm weight")
                .id,
            random_vec(&mut *rng, config.hidden_size),
        );
        rt.set_data(
            layer.q_proj.id,
            random_vec(&mut *rng, attn_proj_dim * config.hidden_size),
        );
        rt.set_data(
            layer.k_proj.id,
            random_vec(&mut *rng, attn_proj_dim * config.hidden_size),
        );
        rt.set_data(
            layer.v_proj.id,
            random_vec(&mut *rng, attn_proj_dim * config.hidden_size),
        );
        rt.set_data(
            layer.o_proj.id,
            random_vec(&mut *rng, config.hidden_size * attn_proj_dim),
        );
        rt.set_data(
            layer.self_attn_layer_scale.scale.id,
            random_positive_vec(&mut *rng, config.hidden_size, 0.01, 0.2),
        );
        rt.set_data(
            layer
                .post_attention_layernorm
                .weight
                .expect("pre-transformer post attention layernorm weight")
                .id,
            random_vec(&mut *rng, config.hidden_size),
        );
        rt.set_data(
            layer.gate_proj.id,
            random_vec(&mut *rng, config.intermediate_size * config.hidden_size),
        );
        rt.set_data(
            layer.up_proj.id,
            random_vec(&mut *rng, config.intermediate_size * config.hidden_size),
        );
        rt.set_data(
            layer.down_proj.id,
            random_vec(&mut *rng, config.hidden_size * config.intermediate_size),
        );
        rt.set_data(
            layer.mlp_layer_scale.scale.id,
            random_positive_vec(&mut *rng, config.hidden_size, 0.01, 0.2),
        );
    }

    rt.set_data(
        pre_transformer
            .norm
            .weight
            .expect("pre-transformer final norm weight")
            .id,
        random_vec(&mut *rng, config.hidden_size),
    );
    rt.set_data(
        pre_transformer.output_proj_weight.id,
        random_vec(&mut *rng, config.latent_dim * config.hidden_size),
    );
    rt.set_data(
        pre_transformer.output_proj_bias.id,
        random_vec(&mut *rng, config.latent_dim),
    );
}

fn set_random_waveform_decoder_params(
    rt: &mut NativeRuntime,
    waveform_decoder: &WaveformDecoder,
    config: &SpeechDecoderConfig,
    rng: &mut StdRng,
) {
    for ((trans_conv, convnext), ratio) in waveform_decoder
        .initial_upsample
        .iter()
        .zip(config.upsampling_ratios.iter().copied())
    {
        rt.set_data(
            trans_conv.conv.weight.id,
            random_vec(&mut *rng, config.latent_dim * config.latent_dim * ratio),
        );
        rt.set_data(
            trans_conv
                .conv
                .bias
                .expect("initial upsample transposed conv bias")
                .id,
            random_vec(&mut *rng, config.latent_dim),
        );

        rt.set_data(
            convnext.dwconv.weight.id,
            random_vec(&mut *rng, config.latent_dim * 7),
        );
        rt.set_data(
            convnext
                .dwconv
                .bias
                .expect("convnext depthwise conv bias")
                .id,
            random_vec(&mut *rng, config.latent_dim),
        );
        rt.set_data(
            convnext.norm.weight.expect("convnext norm weight").id,
            random_vec(&mut *rng, config.latent_dim),
        );
        rt.set_data(
            convnext.norm.bias.expect("convnext norm bias").id,
            random_vec(&mut *rng, config.latent_dim),
        );
        rt.set_data(
            convnext.pwconv1_weight.id,
            random_vec(&mut *rng, 4 * config.latent_dim * config.latent_dim),
        );
        rt.set_data(
            convnext.pwconv1_bias.id,
            random_vec(&mut *rng, 4 * config.latent_dim),
        );
        rt.set_data(
            convnext.pwconv2_weight.id,
            random_vec(&mut *rng, 4 * config.latent_dim * config.latent_dim),
        );
        rt.set_data(
            convnext.pwconv2_bias.id,
            random_vec(&mut *rng, config.latent_dim),
        );
        rt.set_data(
            convnext.gamma.id,
            random_positive_vec(&mut *rng, config.latent_dim, 0.01, 0.2),
        );
    }

    rt.set_data(
        waveform_decoder.initial_conv.conv.weight.id,
        random_vec(&mut *rng, config.decoder_dim * config.latent_dim * 7),
    );
    rt.set_data(
        waveform_decoder
            .initial_conv
            .conv
            .bias
            .expect("waveform initial conv bias")
            .id,
        random_vec(&mut *rng, config.decoder_dim),
    );

    let mut in_dim = config.decoder_dim;
    for (i, (block, rate)) in waveform_decoder
        .blocks
        .iter()
        .zip(config.upsample_rates.iter().copied())
        .enumerate()
    {
        let out_dim = config.decoder_dim / (1 << (i + 1));
        rt.set_data(block.snake.alpha.id, random_vec(&mut *rng, in_dim));
        rt.set_data(block.snake.beta.id, random_vec(&mut *rng, in_dim));
        rt.set_data(
            block.trans_conv.conv.weight.id,
            random_vec(&mut *rng, in_dim * out_dim * (2 * rate)),
        );
        rt.set_data(
            block
                .trans_conv
                .conv
                .bias
                .expect("decoder block transposed conv bias")
                .id,
            random_vec(&mut *rng, out_dim),
        );

        for residual_unit in &block.residual_units {
            rt.set_data(residual_unit.act1.alpha.id, random_vec(&mut *rng, out_dim));
            rt.set_data(residual_unit.act1.beta.id, random_vec(&mut *rng, out_dim));
            rt.set_data(
                residual_unit.conv1.conv.weight.id,
                random_vec(&mut *rng, out_dim * out_dim * 7),
            );
            rt.set_data(
                residual_unit
                    .conv1
                    .conv
                    .bias
                    .expect("decoder residual unit conv1 bias")
                    .id,
                random_vec(&mut *rng, out_dim),
            );
            rt.set_data(residual_unit.act2.alpha.id, random_vec(&mut *rng, out_dim));
            rt.set_data(residual_unit.act2.beta.id, random_vec(&mut *rng, out_dim));
            rt.set_data(
                residual_unit.conv2.conv.weight.id,
                random_vec(&mut *rng, out_dim * out_dim),
            );
            rt.set_data(
                residual_unit
                    .conv2
                    .conv
                    .bias
                    .expect("decoder residual unit conv2 bias")
                    .id,
                random_vec(&mut *rng, out_dim),
            );
        }

        in_dim = out_dim;
    }

    rt.set_data(
        waveform_decoder.final_snake.alpha.id,
        random_vec(&mut *rng, in_dim),
    );
    rt.set_data(
        waveform_decoder.final_snake.beta.id,
        random_vec(&mut *rng, in_dim),
    );
    rt.set_data(
        waveform_decoder.final_conv.conv.weight.id,
        random_vec(&mut *rng, in_dim * 7),
    );
    rt.set_data(
        waveform_decoder
            .final_conv
            .conv
            .bias
            .expect("waveform final conv bias")
            .id,
        random_vec(&mut *rng, 1),
    );
}

#[test]
fn test_rvq_decode_shape() {
    let config = small_speech_decoder_config();
    let batch = 1;
    let seq = 4;

    let mut cx = Graph::new();
    let quantizer = SplitResidualVectorQuantizer::new(&mut cx, &config);
    let code_tensors: Vec<GraphTensor> = (0..(1 + config.num_acoustic_quantizers))
        .map(|_| cx.tensor((batch, seq)).as_dtype(DType::Int))
        .collect();
    let decoded = quantizer.decode(code_tensors.clone()).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    let mut rng = StdRng::seed_from_u64(1601);
    set_random_rvq_params(&mut rt, &quantizer, &config, &mut rng);

    for (i, code_ids) in code_tensors.iter().enumerate() {
        let vocab = if i == 0 {
            config.semantic_codebook_size
        } else {
            config.acoustic_codebook_size
        };
        let ids: Vec<i32> = (0..seq).map(|t| ((i + t) % vocab) as i32).collect();
        rt.set_data(code_ids.id, ids);
    }

    rt.execute(&cx.dyn_map);
    let out = rt.get_f32(decoded.id).clone();
    assert_eq!(out.len(), batch * config.codebook_dim * seq);

    let mut shape = decoded.shape;
    shape.resolve_dyn_dims(&cx.dyn_map);
    assert_eq!(shape.shape_usize(), vec![batch, config.codebook_dim, seq]);
}

#[test]
fn test_rvq_decode_normalized() {
    let config = SpeechDecoderConfig {
        codebook_dim: 4,
        vq_dim: 2,
        latent_dim: 8,
        hidden_size: 8,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        intermediate_size: 16,
        num_hidden_layers: 1,
        head_dim: 4,
        semantic_codebook_size: 4,
        acoustic_codebook_size: 4,
        num_acoustic_quantizers: 1,
        sliding_window: 4,
        ..SpeechDecoderConfig::default()
    };

    let batch = 1;
    let seq = 3;

    let mut cx = Graph::new();
    let quantizer = SplitResidualVectorQuantizer::new(&mut cx, &config);
    let semantic_codes = cx.tensor((batch, seq)).as_dtype(DType::Int);
    let acoustic_codes = cx.tensor((batch, seq)).as_dtype(DType::Int);
    let decoded = quantizer
        .decode(vec![semantic_codes, acoustic_codes])
        .output();

    let semantic_embedding_sum = vec![
        2.0, 4.0, //
        6.0, 8.0, //
        10.0, 12.0, //
        14.0, 16.0,
    ];
    let semantic_cluster_usage = vec![1.0, 2.0, 5.0, 4.0];
    let acoustic_embedding_sum = vec![
        1.0, 3.0, //
        5.0, 7.0, //
        9.0, 11.0, //
        13.0, 15.0,
    ];
    let acoustic_cluster_usage = vec![1.0, 5.0, 3.0, 2.0];

    let first_output_proj = vec![
        1.0, 0.0, //
        0.0, 1.0, //
        1.0, 1.0, //
        2.0, -1.0,
    ];
    let rest_output_proj = vec![
        0.5, 0.0, //
        0.0, 0.5, //
        1.0, -1.0, //
        -0.5, 1.0,
    ];

    let semantic_ids = vec![0i32, 1, 2];
    let acoustic_ids = vec![2i32, 1, 0];

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);

    rt.set_data(
        quantizer.rvq_first.codebooks[0].embedding_sum.id,
        semantic_embedding_sum.clone(),
    );
    rt.set_data(
        quantizer.rvq_first.codebooks[0].cluster_usage.id,
        semantic_cluster_usage.clone(),
    );
    rt.set_data(
        quantizer.rvq_rest.codebooks[0].embedding_sum.id,
        acoustic_embedding_sum.clone(),
    );
    rt.set_data(
        quantizer.rvq_rest.codebooks[0].cluster_usage.id,
        acoustic_cluster_usage.clone(),
    );
    rt.set_data(
        quantizer.rvq_first.output_proj_weight.id,
        first_output_proj.clone(),
    );
    rt.set_data(
        quantizer.rvq_rest.output_proj_weight.id,
        rest_output_proj.clone(),
    );
    rt.set_data(semantic_codes.id, semantic_ids.clone());
    rt.set_data(acoustic_codes.id, acoustic_ids.clone());

    rt.execute(&cx.dyn_map);
    let out = rt.get_f32(decoded.id).clone();

    let lookup = |embedding_sum: &[f32], usage: &[f32], code: usize| -> Vec<f32> {
        let denom = usage[code].max(1e-5);
        vec![
            embedding_sum[code * config.vq_dim] / denom,
            embedding_sum[code * config.vq_dim + 1] / denom,
        ]
    };
    let apply_proj = |proj: &[f32], x: &[f32]| -> Vec<f32> {
        let mut y = vec![0.0f32; config.codebook_dim];
        for out_c in 0..config.codebook_dim {
            for in_c in 0..config.vq_dim {
                y[out_c] += proj[out_c * config.vq_dim + in_c] * x[in_c];
            }
        }
        y
    };

    let mut expected = vec![0.0f32; batch * config.codebook_dim * seq];
    for t in 0..seq {
        let sem = lookup(
            &semantic_embedding_sum,
            &semantic_cluster_usage,
            semantic_ids[t] as usize,
        );
        let ac = lookup(
            &acoustic_embedding_sum,
            &acoustic_cluster_usage,
            acoustic_ids[t] as usize,
        );
        let first = apply_proj(&first_output_proj, &sem);
        let rest = apply_proj(&rest_output_proj, &ac);

        for c in 0..config.codebook_dim {
            expected[c * seq + t] = first[c] + rest[c];
        }
    }

    assert_close(&out, &expected, 1e-5);

    let mut shape = decoded.shape;
    shape.resolve_dyn_dims(&cx.dyn_map);
    assert_eq!(shape.shape_usize(), vec![batch, config.codebook_dim, seq]);
}

#[test]
fn test_snakebeta_exp_transform() {
    let mut cx = Graph::new();
    let snake = SnakeBeta::new(3, "test.snake.alpha", "test.snake.beta", &mut cx);
    let x = cx.tensor((1, 3, 4));
    let out = snake.forward(x).output();

    let input = vec![
        -1.0, -0.5, 0.0, 0.5, //
        -0.8, -0.2, 0.2, 0.9, //
        -1.2, -0.4, 0.3, 1.1,
    ];
    let alpha = vec![-0.7, 0.4, 1.2];
    let beta = vec![-0.2, 0.3, 0.9];

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    rt.set_data(x.id, input.clone());
    rt.set_data(snake.alpha.id, alpha.clone());
    rt.set_data(snake.beta.id, beta.clone());

    rt.execute(&cx.dyn_map);
    let out_data = rt.get_f32(out.id).clone();

    let mut expected = vec![0.0f32; input.len()];
    let time = 4;
    for c in 0..3 {
        let alpha_exp = alpha[c].exp();
        let beta_exp = beta[c].exp();
        for t in 0..time {
            let idx = c * time + t;
            let x_val = input[idx];
            let sin_val = (x_val * alpha_exp).sin();
            expected[idx] = x_val + (sin_val * sin_val) / (beta_exp + 1e-9);
        }
    }

    assert_close(&out_data, &expected, 1e-6);
}

#[test]
fn test_causal_conv_shape() {
    let batch = 2;
    let ch_in = 3;
    let ch_out = 5;
    let time = 7;

    let mut cx = Graph::new();
    let conv = CausalConv1d::new(ch_in, ch_out, 3, 2, true, &mut cx, 1);
    let x = cx.tensor((batch, ch_in, time));
    let out = conv.forward(x).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    let mut rng = StdRng::seed_from_u64(1602);
    rt.set_data(x.id, random_vec(&mut rng, batch * ch_in * time));
    rt.set_data(
        conv.conv.weight.id,
        random_vec(&mut rng, ch_out * ch_in * 3),
    );
    rt.set_data(
        conv.conv.bias.expect("causal conv bias").id,
        random_vec(&mut rng, ch_out),
    );

    rt.execute(&cx.dyn_map);
    let out_data = rt.get_f32(out.id).clone();
    assert_eq!(out_data.len(), batch * ch_out * time);
}

#[test]
fn test_causal_trans_conv_shape() {
    let mut cx = Graph::new();
    let trans_conv = CausalTransConv1d::new(8, 4, 6, 3, true, &mut cx);
    let x = cx.tensor((1, 8, 5));
    let out = trans_conv.forward(x).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    let mut rng = StdRng::seed_from_u64(1603);

    rt.set_data(x.id, random_vec(&mut rng, 8 * 5));
    rt.set_data(trans_conv.conv.weight.id, random_vec(&mut rng, 8 * 4 * 6));
    rt.set_data(
        trans_conv
            .conv
            .bias
            .expect("causal transposed conv bias")
            .id,
        random_vec(&mut rng, 4),
    );

    rt.execute(&cx.dyn_map);
    let mut shape = out.shape;
    shape.resolve_dyn_dims(&cx.dyn_map);
    assert_eq!(shape.shape_usize(), vec![1, 4, 15]);
}

#[test]
fn test_convnext_block_shape() {
    let mut cx = Graph::new();
    let block = ConvNeXtBlock::new(8, "test.convnext", &mut cx);
    let x = cx.tensor((1, 8, 10));
    let out = block.forward(x).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    let mut rng = StdRng::seed_from_u64(1606);

    rt.set_data(x.id, random_vec(&mut rng, 8 * 10));
    rt.set_data(block.dwconv.weight.id, random_vec(&mut rng, 8 * 7));
    rt.set_data(
        block.dwconv.bias.expect("convnext dwconv bias").id,
        random_vec(&mut rng, 8),
    );
    rt.set_data(
        block.norm.weight.expect("convnext norm weight").id,
        random_vec(&mut rng, 8),
    );
    rt.set_data(
        block.norm.bias.expect("convnext norm bias").id,
        random_vec(&mut rng, 8),
    );
    rt.set_data(block.pwconv1_weight.id, random_vec(&mut rng, 4 * 8 * 8));
    rt.set_data(block.pwconv1_bias.id, random_vec(&mut rng, 4 * 8));
    rt.set_data(block.pwconv2_weight.id, random_vec(&mut rng, 4 * 8 * 8));
    rt.set_data(block.pwconv2_bias.id, random_vec(&mut rng, 8));
    rt.set_data(block.gamma.id, random_positive_vec(&mut rng, 8, 0.01, 0.2));

    rt.execute(&cx.dyn_map);
    let out_data = rt.get_f32(out.id).clone();
    assert_eq!(out_data.len(), 8 * 10);

    let mut shape = out.shape;
    shape.resolve_dyn_dims(&cx.dyn_map);
    assert_eq!(shape.shape_usize(), vec![1, 8, 10]);
}

#[test]
fn test_decoder_block_shape() {
    let mut cx = Graph::new();
    let block = DecoderBlock::new(16, 8, 2, "test.decoder.1", &mut cx);
    let x = cx.tensor((1, 16, 6));
    let out = block.forward(x).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    let mut rng = StdRng::seed_from_u64(1607);

    rt.set_data(x.id, random_vec(&mut rng, 16 * 6));
    rt.set_data(block.snake.alpha.id, random_vec(&mut rng, 16));
    rt.set_data(block.snake.beta.id, random_vec(&mut rng, 16));
    rt.set_data(
        block.trans_conv.conv.weight.id,
        random_vec(&mut rng, 16 * 8 * 4),
    );
    rt.set_data(
        block
            .trans_conv
            .conv
            .bias
            .expect("decoder block transposed conv bias")
            .id,
        random_vec(&mut rng, 8),
    );
    for residual_unit in &block.residual_units {
        rt.set_data(residual_unit.act1.alpha.id, random_vec(&mut rng, 8));
        rt.set_data(residual_unit.act1.beta.id, random_vec(&mut rng, 8));
        rt.set_data(
            residual_unit.conv1.conv.weight.id,
            random_vec(&mut rng, 8 * 8 * 7),
        );
        rt.set_data(
            residual_unit
                .conv1
                .conv
                .bias
                .expect("decoder block conv1 bias")
                .id,
            random_vec(&mut rng, 8),
        );
        rt.set_data(residual_unit.act2.alpha.id, random_vec(&mut rng, 8));
        rt.set_data(residual_unit.act2.beta.id, random_vec(&mut rng, 8));
        rt.set_data(
            residual_unit.conv2.conv.weight.id,
            random_vec(&mut rng, 8 * 8),
        );
        rt.set_data(
            residual_unit
                .conv2
                .conv
                .bias
                .expect("decoder block conv2 bias")
                .id,
            random_vec(&mut rng, 8),
        );
    }

    rt.execute(&cx.dyn_map);
    let out_data = rt.get_f32(out.id).clone();
    assert_eq!(out_data.len(), 8 * 12);

    let mut shape = out.shape;
    shape.resolve_dyn_dims(&cx.dyn_map);
    assert_eq!(shape.shape_usize(), vec![1, 8, 12]);
}

#[test]
fn test_pre_transformer_builds() {
    let config = small_speech_decoder_config();
    let mut cx = Graph::new();
    let pre_transformer = PreTransformer::new(&mut cx, &config);
    let hidden = cx.tensor((1, 3, config.latent_dim));
    let _out = pre_transformer.forward(hidden, &config).output();

    cx.build_search_space::<NativeRuntime>();
    let _rt = cx.search(NativeRuntime::default(), 1);
}

#[test]
fn test_pre_transformer_forward_shape() {
    let config = small_speech_decoder_config();
    let batch = 2;
    let seq = 5;

    let mut cx = Graph::new();
    let pre_transformer = PreTransformer::new(&mut cx, &config);
    let hidden = cx.tensor((batch, seq, config.latent_dim));
    let out = pre_transformer.forward(hidden, &config).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    let mut rng = StdRng::seed_from_u64(1604);

    rt.set_data(
        hidden.id,
        random_vec(&mut rng, batch * seq * config.latent_dim),
    );
    set_random_pre_transformer_params(&mut rt, &pre_transformer, &config, &mut rng);

    rt.execute(&cx.dyn_map);
    let out_data = rt.get_f32(out.id).clone();
    assert_eq!(out_data.len(), batch * seq * config.latent_dim);

    let mut shape = out.shape;
    shape.resolve_dyn_dims(&cx.dyn_map);
    assert_eq!(shape.shape_usize(), vec![batch, seq, config.latent_dim]);
}

#[test]
fn test_speech_decoder_with_transformer() {
    let config = small_speech_decoder_config();
    let batch = 1;
    let seq = 4;

    let mut cx = Graph::new();
    let decoder = SpeechDecoder::new(&mut cx, config.clone());
    let code_tensors: Vec<GraphTensor> = (0..(1 + config.num_acoustic_quantizers))
        .map(|_| cx.tensor((batch, seq)).as_dtype(DType::Int))
        .collect();
    let out = decoder.decode_codes(code_tensors.clone()).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    let mut rng = StdRng::seed_from_u64(1605);

    for (i, code_ids) in code_tensors.iter().enumerate() {
        let vocab = if i == 0 {
            config.semantic_codebook_size
        } else {
            config.acoustic_codebook_size
        };
        let ids: Vec<i32> = (0..seq).map(|t| ((i + t) % vocab) as i32).collect();
        rt.set_data(code_ids.id, ids);
    }

    set_random_rvq_params(&mut rt, &decoder.quantizer, &config, &mut rng);
    rt.set_data(
        decoder.pre_conv.conv.weight.id,
        random_vec(&mut rng, config.latent_dim * config.codebook_dim * 3),
    );
    rt.set_data(
        decoder.pre_conv.conv.bias.expect("pre-conv bias").id,
        random_vec(&mut rng, config.latent_dim),
    );
    set_random_pre_transformer_params(&mut rt, &decoder.pre_transformer, &config, &mut rng);
    set_random_waveform_decoder_params(&mut rt, &decoder.waveform_decoder, &config, &mut rng);

    rt.execute(&cx.dyn_map);
    let out_data = rt.get_f32(out.id).clone();
    let total_upsample = config.upsampling_ratios.iter().product::<usize>()
        * config.upsample_rates.iter().product::<usize>();
    assert_eq!(out_data.len(), batch * seq * total_upsample);
    assert!(out_data.iter().all(|v| (-1.0..=1.0).contains(v)));

    let mut shape = out.shape;
    shape.resolve_dyn_dims(&cx.dyn_map);
    assert_eq!(shape.shape_usize(), vec![batch, 1, seq * total_upsample]);
}

#[test]
fn test_full_waveform_decoder() {
    let config = small_speech_decoder_config();
    let batch = 1;
    let seq = 4;

    let mut cx = Graph::new();
    let decoder = SpeechDecoder::new(&mut cx, config.clone());
    let code_tensors: Vec<GraphTensor> = (0..(1 + config.num_acoustic_quantizers))
        .map(|_| cx.tensor((batch, seq)).as_dtype(DType::Int))
        .collect();
    let out = decoder.decode_codes(code_tensors.clone()).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    let mut rng = StdRng::seed_from_u64(1611);

    for (i, code_ids) in code_tensors.iter().enumerate() {
        let vocab = if i == 0 {
            config.semantic_codebook_size
        } else {
            config.acoustic_codebook_size
        };
        let ids: Vec<i32> = (0..seq).map(|t| ((i + t) % vocab) as i32).collect();
        rt.set_data(code_ids.id, ids);
    }

    set_random_rvq_params(&mut rt, &decoder.quantizer, &config, &mut rng);
    rt.set_data(
        decoder.pre_conv.conv.weight.id,
        random_vec(&mut rng, config.latent_dim * config.codebook_dim * 3),
    );
    rt.set_data(
        decoder.pre_conv.conv.bias.expect("pre-conv bias").id,
        random_vec(&mut rng, config.latent_dim),
    );
    set_random_pre_transformer_params(&mut rt, &decoder.pre_transformer, &config, &mut rng);
    set_random_waveform_decoder_params(&mut rt, &decoder.waveform_decoder, &config, &mut rng);

    rt.execute(&cx.dyn_map);
    let out_data = rt.get_f32(out.id).clone();
    assert_eq!(out_data.len(), batch * 64);
    assert!(out_data.iter().all(|v| (-1.0..=1.0).contains(v)));

    let mut shape = out.shape;
    shape.resolve_dyn_dims(&cx.dyn_map);
    assert_eq!(shape.shape_usize(), vec![batch, 1, 64]);
}

#[test]
fn test_decode_speech() {
    let config = SpeechDecoderConfig {
        codebook_dim: 4,
        vq_dim: 2,
        latent_dim: 4,
        hidden_size: 4,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        intermediate_size: 8,
        num_hidden_layers: 0,
        rms_norm_eps: 1e-6,
        rope_theta: 1e4,
        sliding_window: 2,
        head_dim: 4,
        semantic_codebook_size: 8,
        acoustic_codebook_size: 8,
        num_semantic_quantizers: 1,
        num_acoustic_quantizers: 1,
        decoder_dim: 8,
        upsample_rates: vec![],
        upsampling_ratios: vec![],
        layer_scale_initial: 0.01,
    };

    let num_frames = 5;
    let num_codebooks = config.num_semantic_quantizers + config.num_acoustic_quantizers;

    let frames: Vec<Vec<u32>> = (0..num_frames)
        .map(|frame_idx| {
            (0..num_codebooks)
                .map(|codebook_idx| {
                    let vocab = if codebook_idx < config.num_semantic_quantizers {
                        config.semantic_codebook_size
                    } else {
                        config.acoustic_codebook_size
                    };
                    ((frame_idx + codebook_idx) % vocab) as u32
                })
                .collect()
        })
        .collect();

    let mut rng = StdRng::seed_from_u64(1729);
    let weights = build_random_speech_weight_map(&config, &mut rng);
    let audio = decode_speech(&frames, &config, &weights);

    let total_upsample = config.upsampling_ratios.iter().product::<usize>()
        * config.upsample_rates.iter().product::<usize>();
    assert_eq!(audio.len(), num_frames * total_upsample);
    assert!(
        audio.iter().all(|v| v.is_finite()),
        "decoded speech audio contained NaN/Inf"
    );
    assert!(audio.iter().all(|v| (-1.0..=1.0).contains(v)));
}

fn validate_named_inputs_against_safetensors(
    cx: &Graph,
    tensors: &SafeTensors<'_>,
    required_prefix: Option<&str>,
) -> (usize, Vec<String>) {
    let mut matched = 0usize;
    let mut missing = Vec::new();

    for node in cx.graph.node_indices() {
        let Some(input) = cx.graph[node].as_any().downcast_ref::<Input>() else {
            continue;
        };
        if input.label.is_empty() {
            continue;
        }
        if let Some(prefix) = required_prefix {
            if !input.label.starts_with(prefix) {
                continue;
            }
        }

        if tensors.tensor(&input.label).is_ok() {
            matched += 1;
        } else {
            missing.push(input.label.clone());
        }
    }

    (matched, missing)
}

#[test]
#[ignore]
fn test_main_model_weight_names() {
    use crate::code_predictor::*;
    use crate::model::*;

    let talker_config = TalkerConfig {
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
    };
    let predictor_config = CodePredictorConfig {
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
    };

    let mut cx = Graph::new();
    let _pipeline = TtsPipeline::new(&mut cx, talker_config, predictor_config);

    let path = Path::new(
        "/home/ubuntu/projects/luminal-qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/model.safetensors",
    );
    let data =
        std::fs::read(path).unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
    let tensors = SafeTensors::deserialize(&data)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));

    assert_eq!(tensors.names().len(), 404);

    let (matched, missing) =
        validate_named_inputs_against_safetensors(&cx, &tensors, Some("talker."));

    println!("Validated {matched} weights");
    assert!(
        missing.is_empty(),
        "Missing {} safetensors keys:\n{}",
        missing.len(),
        missing.join("\n")
    );
    assert_eq!(matched, 404, "Expected 404 weight matches, got {matched}");
}

#[test]
#[ignore]
fn test_speech_decoder_weight_names() {
    use crate::speech_decoder::*;

    let config = SpeechDecoderConfig {
        codebook_dim: SPEECH_DECODER_CODEBOOK_DIM,
        vq_dim: SPEECH_DECODER_VQ_DIM,
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
    };

    let mut cx = Graph::new();
    let _decoder = SpeechDecoder::new(&mut cx, config);

    let path =
        Path::new("/home/ubuntu/projects/luminal-qwen/Qwen3-TTS-Tokenizer-12Hz/model.safetensors");
    let data =
        std::fs::read(path).unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
    let tensors = SafeTensors::deserialize(&data)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));

    assert_eq!(tensors.names().len(), 496);

    let (matched, missing) =
        validate_named_inputs_against_safetensors(&cx, &tensors, Some("decoder."));

    println!("Validated {matched} weights");
    assert!(
        missing.is_empty(),
        "Missing {} safetensors keys:\n{}",
        missing.len(),
        missing.join("\n")
    );
}

#[test]
fn test_load_safetensors_basic() {
    let expected_a = vec![1.0f32, -2.0, 3.5, 4.25];
    let expected_b = vec![0.5f32, 0.25, -1.0, 2.0];

    let a_bytes: Vec<u8> = expected_a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = expected_b.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "a".to_string(),
        TensorView::new(Dtype::F32, vec![expected_a.len()], &a_bytes).expect("f32 tensor view a"),
    );
    tensors.insert(
        "b".to_string(),
        TensorView::new(Dtype::F32, vec![expected_b.len()], &b_bytes).expect("f32 tensor view b"),
    );

    let path = write_temp_safetensors(&tensors, "basic");
    let _cleanup = TempFileCleanup(path.clone());

    let mut cx = Graph::new();
    let a = cx.named_tensor("a", expected_a.len());
    let b = cx.named_tensor("b", expected_b.len());
    let out_a = (a * 1.0).output();
    let out_b = (b + 0.0).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    let loaded = load_safetensors_to_native(&mut rt, &cx, &path).expect("load safetensors");
    assert_eq!(loaded, 2);

    rt.execute(&cx.dyn_map);
    assert_close(rt.get_f32(out_a.id), &expected_a, TEST_EPS);
    assert_close(rt.get_f32(out_b.id), &expected_b, TEST_EPS);
}

#[test]
fn test_load_safetensors_bf16_conversion() {
    let source = vec![-1.3f32, 0.0, 2.75, 6.5, -0.125];
    let expected: Vec<f32> = source.iter().map(|v| bf16::from_f32(*v).to_f32()).collect();
    let bf16_bytes: Vec<u8> = source
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect();

    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "bf16_input".to_string(),
        TensorView::new(Dtype::BF16, vec![source.len()], &bf16_bytes).expect("bf16 tensor view"),
    );

    let path = write_temp_safetensors(&tensors, "bf16");
    let _cleanup = TempFileCleanup(path.clone());

    let mut cx = Graph::new();
    let input = cx.named_tensor("bf16_input", source.len());
    let out = (input * 1.0).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    let loaded = load_safetensors_to_native(&mut rt, &cx, &path).expect("load bf16 safetensors");
    assert_eq!(loaded, 1);

    rt.execute(&cx.dyn_map);
    assert_close(rt.get_f32(out.id), &expected, 1e-6);
}

#[test]
fn test_load_safetensors_to_map() {
    let expected_f32 = vec![1.0f32, -2.0, 3.5, 4.25];
    let source_bf16 = vec![-1.3f32, 0.0, 2.75, 6.5, -0.125];
    let expected_bf16: Vec<f32> = source_bf16
        .iter()
        .map(|v| bf16::from_f32(*v).to_f32())
        .collect();

    let f32_bytes: Vec<u8> = expected_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bf16_bytes: Vec<u8> = source_bf16
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect();

    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "f32_tensor".to_string(),
        TensorView::new(Dtype::F32, vec![expected_f32.len()], &f32_bytes).expect("f32 tensor view"),
    );
    tensors.insert(
        "bf16_tensor".to_string(),
        TensorView::new(Dtype::BF16, vec![source_bf16.len()], &bf16_bytes)
            .expect("bf16 tensor view"),
    );

    let path = write_temp_safetensors(&tensors, "to_map");
    let _cleanup = TempFileCleanup(path.clone());

    let loaded = load_safetensors_to_map(&path).expect("load safetensors to map");
    assert_eq!(loaded.len(), 2);
    assert_eq!(
        loaded
            .get("f32_tensor")
            .expect("missing f32_tensor in safetensors map"),
        &expected_f32
    );
    assert_close(
        loaded
            .get("bf16_tensor")
            .expect("missing bf16_tensor in safetensors map"),
        &expected_bf16,
        1e-6,
    );
}

#[test]
fn test_load_safetensors_missing_keys() {
    let present_values = vec![3.0f32, -4.0, 5.5];
    let present_bytes: Vec<u8> = present_values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "present".to_string(),
        TensorView::new(Dtype::F32, vec![present_values.len()], &present_bytes)
            .expect("present tensor view"),
    );

    let path = write_temp_safetensors(&tensors, "missing");
    let _cleanup = TempFileCleanup(path.clone());

    let mut cx = Graph::new();
    let present = cx.named_tensor("present", present_values.len());
    let _missing = cx.named_tensor("missing", present_values.len());
    let out = (present * 1.0).output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    let loaded = load_safetensors_to_native(&mut rt, &cx, &path).expect("load with missing keys");
    assert_eq!(loaded, 1);
    assert_eq!(rt.buffers.len(), 1);

    rt.execute(&cx.dyn_map);
    assert_close(rt.get_f32(out.id), &present_values, TEST_EPS);
}
