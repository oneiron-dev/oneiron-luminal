use crate::{
    code_predictor::{CodePredictorConfig, CodePredictorModel},
    model::{TalkerConfig, TalkerModel, VOCAB_SIZE},
};
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::ops::softmax;
use luminal::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

const TEST_EPS: f32 = 1e-3;

fn random_vec(rng: &mut StdRng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| rng.random_range(-0.25f32..0.25f32))
        .collect()
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

#[derive(Clone)]
struct OneLayerWeights {
    embed: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
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

    let split = input.reshape((b, h, s, half, 2))?;
    let even = split.narrow(4, 0, 1)?.squeeze(4)?;
    let odd = split.narrow(4, 1, 1)?.squeeze(4)?;

    let even_out = even.mul(&cos)?.sub(&odd.mul(&sin)?)?;
    let odd_out = even.mul(&sin)?.add(&odd.mul(&cos)?)?;
    Tensor::cat(&[&even_out, &odd_out], 3)
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
    let _embedded = model.embed_tokens(token_ids).output();
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
    rt.set_data(model.embedding.id, weights.embed.clone());

    let layer = &model.layers[0];
    rt.set_data(layer.q_proj.id, weights.q_proj.clone());
    rt.set_data(layer.k_proj.id, weights.k_proj.clone());
    rt.set_data(layer.v_proj.id, weights.v_proj.clone());
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
