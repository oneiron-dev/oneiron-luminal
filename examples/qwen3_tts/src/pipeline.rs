use crate::backend;
use crate::code_predictor::{CodePredictorConfig, CodePredictorModel};
use crate::model::{TalkerConfig, TalkerLayer, TalkerModel};
use crate::speech_decoder::SpeechDecoderConfig;
use luminal::{
    graph::Graph,
    op::{DType, Runtime},
    prelude::GraphTensor,
};
use luminal_nn::LayerNorm;
use std::collections::HashMap;

// Codec token IDs from HuggingFace config.json
pub const CODEC_PAD_ID: i32 = 2148;
pub const CODEC_BOS_ID: i32 = 2149;
pub const CODEC_EOS_ID: i32 = 2150;
pub const CODEC_THINK_ID: i32 = 2154;
pub const CODEC_NOTHINK_ID: i32 = 2155;
pub const CODEC_THINK_BOS_ID: i32 = 2156;
pub const CODEC_THINK_EOS_ID: i32 = 2157;

// TTS text-vocabulary special token IDs from HuggingFace config.json
pub const TTS_BOS_TOKEN_ID: i32 = 151672;
pub const TTS_EOS_TOKEN_ID: i32 = 151673;
pub const TTS_PAD_TOKEN_ID: i32 = 151671;

// Codec language IDs
pub const CODEC_LANG_ENGLISH: i32 = 2050;
pub const CODEC_LANG_CHINESE: i32 = 2055;

pub fn sample_greedy(logits: &[f32], vocab_size: usize) -> Vec<u32> {
    logits
        .chunks_exact(vocab_size)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0 as u32
        })
        .collect()
}

// ===== Helper structs for compiled graph reuse =====

struct MatmulIds {
    slice_input: GraphTensor,
    weight_input: GraphTensor,
    accum_input: GraphTensor,
    output: GraphTensor,
}

struct SnakeBetaIds {
    input: GraphTensor,
    alpha: GraphTensor,
    beta: GraphTensor,
    output: GraphTensor,
}

#[derive(Clone)]
#[allow(dead_code)]
struct BlockShapes {
    in_dim: usize,
    out_dim: usize,
    hidden_len: usize,
    rate: usize,
    kernel_size: usize,
    upsampled_len: usize,
    padded_len: usize,
    out_len: usize,
    target_len: usize,
}

struct BlockGraphs {
    // Snake + upsample sub-stage (named tensors, compiled with weights)
    snake_up_cx: Graph,
    snake_up_rt: backend::Rt,
    snake_up_input: GraphTensor,
    snake_up_output: GraphTensor,

    // Upsample conv matmul (unnamed tensors)
    up_conv_cx: Graph,
    up_conv_rt: backend::Rt,
    up_conv_ids: MatmulIds,

    // Residual SnakeBeta (unnamed tensors, reused for all 6 calls)
    res_snake_cx: Graph,
    res_snake_rt: backend::Rt,
    res_snake_ids: SnakeBetaIds,

    // Residual conv matmul (unnamed tensors, reused for all 6 calls)
    res_conv_cx: Graph,
    res_conv_rt: backend::Rt,
    res_conv_ids: MatmulIds,
}

fn compute_block_shapes(
    num_frames: usize,
    config: &SpeechDecoderConfig,
) -> (usize, Vec<BlockShapes>) {
    // Stage 2a: upsample by upsampling_ratios, then initial_conv (preserves length)
    let mut s2a_len = num_frames;
    for ratio in &config.upsampling_ratios {
        s2a_len *= ratio;
    }

    // Decoder blocks
    let mut hidden_len = s2a_len;
    let mut in_dim = config.decoder_dim;
    let mut block_shapes = Vec::new();

    for (i, &rate) in config.upsample_rates.iter().enumerate() {
        let out_dim = config.decoder_dim / (1 << (i + 1));
        let kernel_size = 2 * rate;
        let upsampled_len = (hidden_len - 1) * rate + 1;
        let conv_padding = kernel_size - 1;
        let padded_len = upsampled_len + 2 * conv_padding;
        let out_len = upsampled_len + conv_padding;
        let target_len = hidden_len * rate;

        block_shapes.push(BlockShapes {
            in_dim,
            out_dim,
            hidden_len,
            rate,
            kernel_size,
            upsampled_len,
            padded_len,
            out_len,
            target_len,
        });

        hidden_len = target_len;
        in_dim = out_dim;
    }

    (s2a_len, block_shapes)
}

/// Build and compile a reusable matmul+accumulate graph with unnamed tensors.
fn compile_matmul_graph(
    seq_len: usize,
    in_ch: usize,
    out_ch: usize,
) -> (Graph, backend::Rt, MatmulIds) {
    let mut cx = Graph::new();
    let slice_input = cx.tensor((1, seq_len, in_ch));
    let weight_input = cx.tensor((in_ch, out_ch));
    let accum_input = cx.tensor((1, seq_len, out_ch));
    let partial = slice_input.matmul(weight_input);
    let result = accum_input + partial;
    let output = result.output();
    let rt = backend::compile(&mut cx, &HashMap::new());
    (
        cx,
        rt,
        MatmulIds {
            slice_input,
            weight_input,
            accum_input,
            output,
        },
    )
}

/// Build and compile a reusable SnakeBeta graph with unnamed tensors.
fn compile_snake_beta_graph(
    channels: usize,
    seq_len: usize,
) -> (Graph, backend::Rt, SnakeBetaIds) {
    let mut cx = Graph::new();
    let input = cx.tensor((1, channels, seq_len));
    let alpha = cx.tensor(channels);
    let beta_t = cx.tensor(channels);

    // Inline SnakeBeta::forward with unnamed tensors
    let (batch, _, time) = input.dims3();
    let alpha_exp = alpha.exp().expand_dim(0, batch).expand_dim(2, time);
    let beta_exp = beta_t.exp().expand_dim(0, batch).expand_dim(2, time);
    let sin_val = (input * alpha_exp).sin();
    let out = input + (sin_val * sin_val) / (beta_exp + 1e-9);
    let output = out.output();

    let rt = backend::compile(&mut cx, &HashMap::new());
    (
        cx,
        rt,
        SnakeBetaIds {
            input,
            alpha,
            beta: beta_t,
            output,
        },
    )
}

/// Execute SnakeBeta using a pre-compiled graph with weights from the map.
fn execute_snake_beta(
    hidden_data: &mut Vec<f32>,
    snake_cx: &Graph,
    snake_rt: &mut backend::Rt,
    snake_ids: &SnakeBetaIds,
    alpha_name: &str,
    beta_name: &str,
    weights: &HashMap<String, Vec<f32>>,
) {
    backend::set_data(snake_rt, snake_ids.input.id, std::mem::take(hidden_data));
    backend::set_data(
        snake_rt,
        snake_ids.alpha.id,
        weights
            .get(alpha_name)
            .unwrap_or_else(|| panic!("missing: {alpha_name}"))
            .clone(),
    );
    backend::set_data(
        snake_rt,
        snake_ids.beta.id,
        weights
            .get(beta_name)
            .unwrap_or_else(|| panic!("missing: {beta_name}"))
            .clone(),
    );
    snake_rt.execute(&snake_cx.dyn_map);
    *hidden_data = backend::get_f32(snake_rt, snake_ids.output.id);
}

/// Execute causal conv1d using a pre-compiled matmul graph.
///
/// Weight layout: `[out_ch, in_ch, kernel]` (ConvND groups=1, row-major).
fn causal_conv1d_with_matmul(
    hidden_data: &mut Vec<f32>,
    matmul_cx: &Graph,
    matmul_rt: &mut backend::Rt,
    matmul_ids: &MatmulIds,
    in_ch: usize,
    out_ch: usize,
    seq_len: usize,
    kernel: usize,
    dilation: usize,
    weight_prefix: &str,
    weights: &HashMap<String, Vec<f32>>,
) {
    // 1. CPU-side causal padding (left only)
    let left_pad = dilation * (kernel - 1);
    let padded_len = seq_len + left_pad;
    let mut padded_data = vec![0.0f32; in_ch * padded_len];
    for c in 0..in_ch {
        for t in 0..seq_len {
            padded_data[c * padded_len + left_pad + t] = hidden_data[c * seq_len + t];
        }
    }

    // 2. Load weight + bias from weights map (CPU)
    let weight_name = format!("{weight_prefix}.weight");
    let weight_data = weights
        .get(&weight_name)
        .unwrap_or_else(|| panic!("missing weight: {weight_name}"));
    let bias_name = format!("{weight_prefix}.bias");
    let bias_data = weights.get(&bias_name);

    // 3. Loop over kernel positions with CPU-side slicing
    let mut accum_data = vec![0.0f32; seq_len * out_ch];
    for k in 0..kernel {
        // CPU: extract padded[:, :, k*d .. k*d + seq_len] and transpose to [1, seq_len, in_ch]
        let mut slice_t_data = vec![0.0f32; seq_len * in_ch];
        for c in 0..in_ch {
            for t in 0..seq_len {
                slice_t_data[t * in_ch + c] = padded_data[c * padded_len + k * dilation + t];
            }
        }

        // CPU: extract weight[:, :, k] transposed to [in_ch, out_ch]
        // weight layout: [out_ch, in_ch, kernel] row-major
        let mut weight_k_data = vec![0.0f32; in_ch * out_ch];
        for oc in 0..out_ch {
            for ic in 0..in_ch {
                weight_k_data[ic * out_ch + oc] =
                    weight_data[oc * in_ch * kernel + ic * kernel + k];
            }
        }

        backend::set_data(matmul_rt, matmul_ids.slice_input.id, slice_t_data);
        backend::set_data(matmul_rt, matmul_ids.weight_input.id, weight_k_data);
        backend::set_data(matmul_rt, matmul_ids.accum_input.id, accum_data);
        matmul_rt.execute(&matmul_cx.dyn_map);
        accum_data = backend::get_f32(matmul_rt, matmul_ids.output.id);
    }

    // 4. Transpose accum [1, seq_len, out_ch] → [1, out_ch, seq_len] and add bias
    let mut result_data = vec![0.0f32; out_ch * seq_len];
    for c in 0..out_ch {
        let bias_val = bias_data.map_or(0.0, |b| b[c]);
        for t in 0..seq_len {
            result_data[c * seq_len + t] = accum_data[t * out_ch + c] + bias_val;
        }
    }

    *hidden_data = result_data;
}

// ===== generate_frames (Steps 1-3: hoisted compilation) =====

pub fn generate_frames(
    talker_config: &TalkerConfig,
    predictor_config: &CodePredictorConfig,
    initial_embeds: &[f32],
    tts_pad_embed: &[f32],
    prompt_len: usize,
    num_frames: usize,
    weights: &HashMap<String, Vec<f32>>,
) -> Vec<Vec<u32>> {
    let hidden = talker_config.hidden;
    let n_layers = talker_config.layers;
    let n_kv_heads = talker_config.n_kv_heads;
    let head_dim = talker_config.head_dim;
    assert_eq!(
        tts_pad_embed.len(),
        hidden,
        "tts_pad_embed length must equal hidden"
    );
    assert_eq!(
        initial_embeds.len(),
        prompt_len * hidden,
        "initial_embeds length must equal prompt_len * hidden"
    );
    if num_frames == 0 {
        return Vec::new();
    }

    let mut all_frames = Vec::with_capacity(num_frames);
    let t0 = std::time::Instant::now();

    // ===== COMPILE PHASE =====
    // All graph constructions and compilations happen upfront.
    // This separates compile time from execution time and enables graph dedup.

    // --- 1. Prefill: single graph with all layers + graph breaks ---
    // luminal's build_grouped_egraphs hashes each chunk after normalizing
    // Input labels — all layer chunks are structurally identical, so egglog
    // runs once and re-extracts LLIR with remapped node IDs.
    let mut prefill_cx = Graph::new();
    let prefill_input = prefill_cx.tensor((1, prompt_len, hidden));
    let mut h = prefill_input;
    let mut prefill_k_outs = Vec::new();
    let mut prefill_v_outs = Vec::new();
    let mut prefill_hidden_outs = Vec::new();
    for layer_idx in 0..n_layers {
        let layer = TalkerLayer::new(&mut prefill_cx, talker_config, layer_idx);
        let (out, k_cache, v_cache) = layer.forward_with_kv(h, talker_config);
        prefill_k_outs.push(k_cache.output());
        prefill_v_outs.push(v_cache.output());
        // Mark hidden output on ALL layers so every chunk has the same number
        // of Output nodes (3: k, v, hidden). This ensures egglog dedup groups
        // them correctly — mismatched Output counts trigger an assertion.
        // We only read the last layer's hidden output, but the others are cheap.
        prefill_hidden_outs.push(out.output());
        if layer_idx < n_layers - 1 {
            h = crate::maybe_graph_break(out);
        } else {
            h = out;
        }
    }
    let prefill_hidden_out = *prefill_hidden_outs.last().unwrap();
    let mut prefill_rt = backend::compile(&mut prefill_cx, weights);
    eprintln!(
        "  [compile] prefill ({} layers, 1 graph) in {:.1}s",
        n_layers,
        t0.elapsed().as_secs_f32()
    );

    // --- 2. Final stage: norm + logits + argmax + embed ---
    let mut final_cx = Graph::new();
    let final_input = final_cx.tensor((1, prompt_len, hidden));
    let final_norm = LayerNorm::new(
        hidden,
        Some("talker.model.norm.weight"),
        None,
        false,
        talker_config.rms_norm_eps,
        &mut final_cx,
    );
    let codec_head = final_cx.named_tensor(
        "talker.codec_head.weight",
        (talker_config.vocab_size, hidden),
    );
    let codec_embedding = final_cx.named_tensor(
        "talker.model.codec_embedding.weight",
        (talker_config.vocab_size, hidden),
    );
    let normed = final_norm.forward(final_input);
    let logits = normed.matmul(codec_head.t());
    let last_logits = logits.slice((.., (prompt_len - 1).., ..));
    let code_0 = last_logits.argmax(2);
    let (batch, seq) = code_0.dims2();
    let code_0_embed = codec_embedding.gather(
        (code_0 * hidden).expand_dim(2, hidden)
            + code_0
                .graph()
                .arange(hidden)
                .expand_lhs([batch, seq]),
    );
    let last_hidden = normed.slice((.., (prompt_len - 1).., ..));
    let code_0_out = code_0.cast(DType::F32).output();
    let code_0_embed_out = code_0_embed.output();
    let last_hidden_out = last_hidden.output();
    let mut final_rt = backend::compile(&mut final_cx, weights);
    eprintln!(
        "  [compile] final stage in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    // --- 3. Code predictor (single instance, reused for prefill + decode) ---
    let mut pred_cx = Graph::new();
    let code_predictor = CodePredictorModel::new(&mut pred_cx, predictor_config.clone());
    let pred_hidden_input = pred_cx.tensor((1, 1, hidden));
    let pred_code0_embed_input = pred_cx.tensor((1, 1, hidden));
    let pred_out =
        code_predictor.generate_codes(pred_hidden_input, pred_code0_embed_input);
    let mut pred_codec_sum = pred_code0_embed_input;
    for embed in &pred_out.embeds {
        pred_codec_sum = pred_codec_sum + *embed;
    }
    let pred_code_outs: Vec<_> = pred_out
        .codes
        .iter()
        .map(|code| code.cast(DType::F32).output())
        .collect();
    let pred_codec_sum_out = pred_codec_sum.output();
    let mut pred_rt = backend::compile(&mut pred_cx, weights);
    eprintln!(
        "  [compile] code predictor in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    // --- 4. Decode talker ---
    let max_seq = prompt_len + num_frames;
    let mut decode_cx = Graph::new();
    let decode_talker = TalkerModel::new(&mut decode_cx, talker_config.clone());
    let new_embed_input = decode_cx.tensor((1, 1, hidden));
    let pos_input = decode_cx.tensor(1);
    let mask_input = decode_cx.tensor((1, 1, 1, max_seq + 1));
    let kv_inputs: Vec<(GraphTensor, GraphTensor)> = (0..n_layers)
        .map(|_| {
            (
                decode_cx.tensor((1, n_kv_heads, max_seq, head_dim)),
                decode_cx.tensor((1, n_kv_heads, max_seq, head_dim)),
            )
        })
        .collect();
    let (decode_logits, decode_normed, decode_new_kv_outs) =
        decode_talker.decode_fixed(new_embed_input, &kv_inputs, mask_input, pos_input);
    let decode_code_0 = decode_logits.argmax(2);
    let decode_code_0_embed = decode_talker.embed_codec(decode_code_0);
    let decode_code_0_out = decode_code_0.cast(DType::F32).output();
    let decode_code_0_embed_out = decode_code_0_embed.output();
    let decode_normed_out = decode_normed.output();
    let decode_new_kv_cache_outs: Vec<_> = decode_new_kv_outs
        .iter()
        .map(|(k, v)| (k.output(), v.output()))
        .collect();
    let mut decode_rt = backend::compile(&mut decode_cx, weights);
    eprintln!(
        "  [compile] decode talker in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    let compile_time = t0.elapsed();
    eprintln!(
        "  [generate_frames] all compiled in {:.1}s",
        compile_time.as_secs_f32()
    );

    // ===== EXECUTE PHASE =====
    let t1 = std::time::Instant::now();

    // --- Execute prefill ---
    backend::set_data(&mut prefill_rt, prefill_input.id, initial_embeds.to_vec());
    prefill_rt.execute(&prefill_cx.dyn_map);
    let hidden_data = backend::get_f32(&prefill_rt, prefill_hidden_out.id);
    let prefill_k: Vec<Vec<f32>> = prefill_k_outs
        .iter()
        .map(|k| backend::get_f32(&prefill_rt, k.id))
        .collect();
    let prefill_v: Vec<Vec<f32>> = prefill_v_outs
        .iter()
        .map(|v| backend::get_f32(&prefill_rt, v.id))
        .collect();
    drop(prefill_rt);
    eprintln!(
        "  [execute] prefill ({} layers) in {:.1}s",
        n_layers,
        t1.elapsed().as_secs_f32()
    );

    // --- Execute final stage ---
    backend::set_data(&mut final_rt, final_input.id, hidden_data);
    final_rt.execute(&final_cx.dyn_map);
    let code_0_val = backend::get_f32(&final_rt, code_0_out.id)[0] as u32;
    if code_0_val == CODEC_EOS_ID as u32 {
        return all_frames;
    }
    let code_0_embed_data = backend::get_f32(&final_rt, code_0_embed_out.id);
    let last_hidden_data = backend::get_f32(&final_rt, last_hidden_out.id);
    drop(final_rt);
    eprintln!(
        "  [execute] final stage in {:.1}s",
        t1.elapsed().as_secs_f32()
    );

    // --- Execute code predictor (prefill frame 0) ---
    backend::set_data(&mut pred_rt, pred_hidden_input.id, last_hidden_data);
    backend::set_data(&mut pred_rt, pred_code0_embed_input.id, code_0_embed_data);
    pred_rt.execute(&pred_cx.dyn_map);

    let pred_codes: Vec<u32> = pred_code_outs
        .iter()
        .map(|code| backend::get_f32(&pred_rt, code.id)[0] as u32)
        .collect();
    let mut frame_codes = vec![code_0_val];
    frame_codes.extend(pred_codes);
    all_frames.push(frame_codes);

    if num_frames <= 1 {
        return all_frames;
    }

    let codec_sum_data = backend::get_f32(&pred_rt, pred_codec_sum_out.id);
    assert_eq!(
        codec_sum_data.len(),
        hidden,
        "codec_sum output length must equal hidden"
    );
    let mut next_embed: Vec<f32> = codec_sum_data
        .iter()
        .zip(tts_pad_embed.iter())
        .map(|(codec, tts)| codec + tts)
        .collect();
    eprintln!(
        "  [execute] predictor (frame 0) in {:.1}s",
        t1.elapsed().as_secs_f32()
    );

    // --- Decode loop ---
    // Set up KV buffers from prefill caches
    let kv_buf_size = n_kv_heads * max_seq * head_dim;
    let prompt_kv_size = n_kv_heads * prompt_len * head_dim;
    let mut k_bufs: Vec<Vec<f32>> = vec![vec![0.0; kv_buf_size]; n_layers];
    let mut v_bufs: Vec<Vec<f32>> = vec![vec![0.0; kv_buf_size]; n_layers];
    for (layer_idx, (k_src, v_src)) in prefill_k.iter().zip(prefill_v.iter()).enumerate() {
        assert_eq!(
            k_src.len(),
            prompt_kv_size,
            "prefill K cache size must equal n_kv_heads * prompt_len * head_dim"
        );
        assert_eq!(
            v_src.len(),
            prompt_kv_size,
            "prefill V cache size must equal n_kv_heads * prompt_len * head_dim"
        );
        for head in 0..n_kv_heads {
            let src_offset = head * prompt_len * head_dim;
            let dst_offset = head * max_seq * head_dim;
            let src_end = src_offset + prompt_len * head_dim;
            let dst_end = dst_offset + prompt_len * head_dim;
            k_bufs[layer_idx][dst_offset..dst_end].copy_from_slice(&k_src[src_offset..src_end]);
            v_bufs[layer_idx][dst_offset..dst_end].copy_from_slice(&v_src[src_offset..src_end]);
        }
    }

    let mut attn_mask = vec![-1e9f32; max_seq + 1];
    for v in attn_mask.iter_mut().take(prompt_len) {
        *v = 0.0;
    }
    attn_mask[max_seq] = 0.0;

    let profiling = std::env::var("LUMINAL_PROFILE").map_or(false, |v| v == "1");
    let mut prof_set_data: Vec<f64> = Vec::new();
    let mut prof_kv_upload: Vec<f64> = Vec::new();
    let mut prof_decode_exec: Vec<f64> = Vec::new();
    let mut prof_get_f32: Vec<f64> = Vec::new();
    let mut prof_pred_exec: Vec<f64> = Vec::new();
    let mut prof_kv_scatter: Vec<f64> = Vec::new();
    let mut prof_total: Vec<f64> = Vec::new();

    for frame in 1..num_frames {
        let frame_start = std::time::Instant::now();
        let p = prompt_len + frame - 1;

        // Phase 1: set_data (upload embed/pos/mask)
        let phase_t = std::time::Instant::now();
        backend::set_data(&mut decode_rt, new_embed_input.id, next_embed.clone());
        backend::set_data(&mut decode_rt, pos_input.id, vec![p as f32]);
        backend::set_data(&mut decode_rt, mask_input.id, attn_mask.clone());
        let set_data_us = phase_t.elapsed().as_micros() as f64;

        // Phase 2: kv_upload (full KV cache, frame 1 only on CUDA)
        let phase_t = std::time::Instant::now();
        if frame == 1 || cfg!(not(feature = "cuda")) {
            for (i, (k_in, v_in)) in kv_inputs.iter().enumerate() {
                backend::set_data(&mut decode_rt, k_in.id, k_bufs[i].clone());
                backend::set_data(&mut decode_rt, v_in.id, v_bufs[i].clone());
            }
        }
        let kv_upload_us = phase_t.elapsed().as_micros() as f64;

        // Phase 3: decode_exec
        let phase_t = std::time::Instant::now();
        decode_rt.execute(&decode_cx.dyn_map);
        let decode_exec_us = phase_t.elapsed().as_micros() as f64;

        // Phase 4: get_f32 (download outputs)
        let phase_t = std::time::Instant::now();
        let code_0_val = backend::get_f32(&decode_rt, decode_code_0_out.id)[0] as u32;
        if code_0_val == CODEC_EOS_ID as u32 {
            eprintln!("  [decode] EOS at frame {}", frame);
            break;
        }
        let code_0_embed_data = backend::get_f32(&decode_rt, decode_code_0_embed_out.id);
        let normed_data = backend::get_f32(&decode_rt, decode_normed_out.id);
        let get_f32_us = phase_t.elapsed().as_micros() as f64;

        // Phase 5: pred_exec (code predictor set_data + execute + get outputs)
        let phase_t = std::time::Instant::now();
        backend::set_data(&mut pred_rt, pred_hidden_input.id, normed_data);
        backend::set_data(&mut pred_rt, pred_code0_embed_input.id, code_0_embed_data);
        pred_rt.execute(&pred_cx.dyn_map);
        let pred_codes: Vec<u32> = pred_code_outs
            .iter()
            .map(|code| backend::get_f32(&pred_rt, code.id)[0] as u32)
            .collect();
        let mut frame_codes = vec![code_0_val];
        frame_codes.extend(pred_codes);
        all_frames.push(frame_codes);
        let pred_exec_us = phase_t.elapsed().as_micros() as f64;

        // Phase 6: kv_scatter (CPU scatter + partial GPU updates + codec_sum)
        let phase_t = std::time::Instant::now();
        for (i, (k_out, v_out)) in decode_new_kv_cache_outs.iter().enumerate() {
            let new_k = backend::get_f32(&decode_rt, k_out.id);
            let new_v = backend::get_f32(&decode_rt, v_out.id);
            assert_eq!(
                new_k.len(),
                n_kv_heads * head_dim,
                "new K shape must be n_kv_heads * head_dim"
            );
            assert_eq!(
                new_v.len(),
                n_kv_heads * head_dim,
                "new V shape must be n_kv_heads * head_dim"
            );
            for head in 0..n_kv_heads {
                let src_start = head * head_dim;
                let src_end = src_start + head_dim;
                let dst_start = head * max_seq * head_dim + p * head_dim;
                let dst_end = dst_start + head_dim;

                k_bufs[i][dst_start..dst_end].copy_from_slice(&new_k[src_start..src_end]);
                v_bufs[i][dst_start..dst_end].copy_from_slice(&new_v[src_start..src_end]);

                if cfg!(feature = "cuda") {
                    let byte_offset = dst_start * 4;
                    backend::update_data_slice(
                        &mut decode_rt,
                        kv_inputs[i].0.id,
                        byte_offset,
                        &new_k[src_start..src_end],
                    );
                    backend::update_data_slice(
                        &mut decode_rt,
                        kv_inputs[i].1.id,
                        byte_offset,
                        &new_v[src_start..src_end],
                    );
                }
            }
        }
        attn_mask[p] = 0.0;

        let codec_sum_data = backend::get_f32(&pred_rt, pred_codec_sum_out.id);
        assert_eq!(
            codec_sum_data.len(),
            hidden,
            "codec_sum output length must equal hidden"
        );
        next_embed = codec_sum_data
            .iter()
            .zip(tts_pad_embed.iter())
            .map(|(codec, tts)| codec + tts)
            .collect();
        let kv_scatter_us = phase_t.elapsed().as_micros() as f64;

        let total_us = frame_start.elapsed().as_micros() as f64;

        if profiling {
            eprintln!(
                "  [profile] frame {} | set_data: {:.1}ms | kv_upload: {:.1}ms | decode_exec: {:.1}ms | get_f32: {:.1}ms | pred_exec: {:.1}ms | kv_scatter: {:.1}ms | total: {:.1}ms",
                frame, set_data_us / 1000.0, kv_upload_us / 1000.0, decode_exec_us / 1000.0,
                get_f32_us / 1000.0, pred_exec_us / 1000.0, kv_scatter_us / 1000.0, total_us / 1000.0,
            );
            prof_set_data.push(set_data_us);
            prof_kv_upload.push(kv_upload_us);
            prof_decode_exec.push(decode_exec_us);
            prof_get_f32.push(get_f32_us);
            prof_pred_exec.push(pred_exec_us);
            prof_kv_scatter.push(kv_scatter_us);
            prof_total.push(total_us);
        } else {
            eprintln!(
                "  [decode] frame {} done at {:.1}s",
                frame,
                t1.elapsed().as_secs_f32()
            );
        }
    }

    // Print profiling summary (averages excluding frame 1 which has KV upload overhead)
    if profiling && prof_total.len() > 1 {
        let skip = 1; // skip frame 1 (has full KV upload)
        let n = (prof_total.len() - skip) as f64;
        let avg = |v: &[f64]| v[skip..].iter().sum::<f64>() / n / 1000.0;
        eprintln!("\n  [profile] === Decode Loop Averages (frames 2-{}) ===", prof_total.len());
        eprintln!(
            "  [profile] set_data: {:.2}ms | kv_upload: {:.2}ms | decode_exec: {:.2}ms | get_f32: {:.2}ms | pred_exec: {:.2}ms | kv_scatter: {:.2}ms | total: {:.2}ms",
            avg(&prof_set_data), avg(&prof_kv_upload), avg(&prof_decode_exec),
            avg(&prof_get_f32), avg(&prof_pred_exec), avg(&prof_kv_scatter), avg(&prof_total),
        );

        #[cfg(feature = "cuda")]
        {
            eprintln!("\n  === Decode Talker Execution Stats ===");
            decode_rt.print_execution_stats();
            eprintln!("\n  === Code Predictor Execution Stats ===");
            pred_rt.print_execution_stats();
        }
    }

    let exec_time = t1.elapsed();
    eprintln!(
        "  [generate_frames] compile: {:.1}s, execute: {:.1}s, total: {:.1}s",
        compile_time.as_secs_f32(),
        exec_time.as_secs_f32(),
        t0.elapsed().as_secs_f32()
    );

    all_frames
}

pub struct StreamingPromptInputs {
    pub role_text_ids: GraphTensor,     // [1, 3] - role prefix text tokens
    pub overlay_text_ids: GraphTensor,  // [1, overlay_len] - tts_pad repeated + tts_bos
    pub overlay_codec_ids: GraphTensor, // [1, overlay_len] - codec control tokens
    pub transition_text_id: GraphTensor, // [1, 1] - first real content text token
    pub transition_codec_id: GraphTensor, // [1, 1] - codec BOS token
    pub trailing_text_ids: GraphTensor, // [1, trailing_len] - rest of text + tts_eos
    pub tts_pad_text_id: GraphTensor,   // [1, 1] - tts_pad for generation loop
}

pub struct StreamingPromptOutputs {
    pub initial_embeds: GraphTensor,       // [1, prompt_len, hidden]
    pub trailing_text_hidden: GraphTensor, // [1, trailing_len, hidden]
    pub tts_pad_embed: GraphTensor,        // [1, 1, hidden]
}

pub struct NonStreamingPromptInputs {
    pub instruct_ids: Option<GraphTensor>, // [1, instruct_len]
    pub role_ids: GraphTensor,             // [1, 3]
    pub overlay_text_ids: GraphTensor,     // [1, overlay_len]
    pub overlay_codec_ids: GraphTensor,    // [1, overlay_len]
    pub content_ids: GraphTensor,          // [1, content_len]
    pub content_codec_ids: GraphTensor,    // [1, content_len + 1] (codec_pad repeated)
    pub tts_eos_id: GraphTensor,           // [1, 1]
    pub transition_text_id: GraphTensor,   // [1, 1] (tts_pad)
    pub transition_codec_id: GraphTensor,  // [1, 1] (codec_bos)
}

pub struct NonStreamingPromptOutputs {
    pub initial_embeds: GraphTensor, // [1, prompt_len, hidden]
    pub tts_pad_embed: GraphTensor,  // [1, 1, hidden]
}

pub struct TtsPipeline {
    pub talker: TalkerModel,
    pub code_predictor: CodePredictorModel,
}

impl TtsPipeline {
    pub fn new(
        cx: &mut Graph,
        talker_config: TalkerConfig,
        predictor_config: CodePredictorConfig,
    ) -> Self {
        assert_eq!(
            talker_config.hidden, predictor_config.talker_hidden,
            "talker hidden must match predictor talker_hidden"
        );
        Self {
            talker: TalkerModel::new(cx, talker_config),
            code_predictor: CodePredictorModel::new(cx, predictor_config),
        }
    }

    pub fn assemble_streaming_prompt(
        &self,
        inputs: &StreamingPromptInputs,
    ) -> StreamingPromptOutputs {
        // 1. Text path: embed through text_embedding + text_projection
        let role_embeds = self
            .talker
            .embed_text(inputs.role_text_ids.cast(DType::Int));
        let overlay_text = self
            .talker
            .embed_text(inputs.overlay_text_ids.cast(DType::Int));
        let transition_text = self
            .talker
            .embed_text(inputs.transition_text_id.cast(DType::Int));
        let trailing_text = self
            .talker
            .embed_text(inputs.trailing_text_ids.cast(DType::Int));
        let tts_pad_embed = self
            .talker
            .embed_text(inputs.tts_pad_text_id.cast(DType::Int));

        // 2. Codec path: embed through codec_embedding
        let overlay_codec = self
            .talker
            .embed_codec(inputs.overlay_codec_ids.cast(DType::Int));
        let transition_codec = self
            .talker
            .embed_codec(inputs.transition_codec_id.cast(DType::Int));

        // 3. Element-wise addition at overlay and transition positions
        let overlay = overlay_text + overlay_codec;
        let transition = transition_text + transition_codec;

        // 4. Concatenate: [role | overlay | transition]
        let initial = role_embeds
            .concat_along(overlay, 1)
            .concat_along(transition, 1);

        StreamingPromptOutputs {
            initial_embeds: initial,
            trailing_text_hidden: trailing_text,
            tts_pad_embed,
        }
    }

    pub fn assemble_nonstreaming_prompt(
        &self,
        inputs: &NonStreamingPromptInputs,
    ) -> NonStreamingPromptOutputs {
        let role_embed = self.talker.embed_text(inputs.role_ids.cast(DType::Int));

        let overlay_text = self
            .talker
            .embed_text(inputs.overlay_text_ids.cast(DType::Int));
        let overlay_codec = self
            .talker
            .embed_codec(inputs.overlay_codec_ids.cast(DType::Int));
        let overlay = overlay_text + overlay_codec;

        let content_text = self.talker.embed_text(inputs.content_ids.cast(DType::Int));
        let tts_eos_embed = self.talker.embed_text(inputs.tts_eos_id.cast(DType::Int));
        let text_with_eos = content_text.concat_along(tts_eos_embed, 1);
        let content_codec = self
            .talker
            .embed_codec(inputs.content_codec_ids.cast(DType::Int));
        let (_, text_seq, _) = text_with_eos.dims3();
        let (_, codec_seq, _) = content_codec.dims3();
        assert_eq!(
            text_seq, codec_seq,
            "content text + tts_eos length must match content codec length"
        );
        let content_section = text_with_eos + content_codec;

        let transition_text = self
            .talker
            .embed_text(inputs.transition_text_id.cast(DType::Int));
        let transition_codec = self
            .talker
            .embed_codec(inputs.transition_codec_id.cast(DType::Int));
        let transition = transition_text + transition_codec;

        let mut initial_embeds = role_embed
            .concat_along(overlay, 1)
            .concat_along(content_section, 1)
            .concat_along(transition, 1);

        if let Some(instruct_ids) = inputs.instruct_ids {
            let instruct_embed = self.talker.embed_text(instruct_ids.cast(DType::Int));
            initial_embeds = instruct_embed.concat_along(initial_embeds, 1);
        }

        NonStreamingPromptOutputs {
            initial_embeds,
            tts_pad_embed: transition_text,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn assemble_prompt_embeds(
    talker_config: &TalkerConfig,
    predictor_config: &CodePredictorConfig,
    role_ids: &[f32],
    content_ids: &[f32],
    overlay_text_ids: &[f32],
    overlay_codec_ids: &[f32],
    content_codec_ids: &[f32],
    tts_eos_id: f32,
    transition_text_id: f32,
    transition_codec_id: f32,
    instruct_ids: Option<&[f32]>,
    weights: &HashMap<String, Vec<f32>>,
) -> (Vec<f32>, Vec<f32>) {
    let mut cx = Graph::new();
    let pipeline = TtsPipeline::new(&mut cx, talker_config.clone(), predictor_config.clone());

    let role_ids_tensor = cx.tensor((1, role_ids.len()));
    let content_ids_tensor = cx.tensor((1, content_ids.len()));
    let overlay_text_ids_tensor = cx.tensor((1, overlay_text_ids.len()));
    let overlay_codec_ids_tensor = cx.tensor((1, overlay_codec_ids.len()));
    let content_codec_ids_tensor = cx.tensor((1, content_codec_ids.len()));
    let tts_eos_id_tensor = cx.tensor((1, 1));
    let transition_text_id_tensor = cx.tensor((1, 1));
    let transition_codec_id_tensor = cx.tensor((1, 1));
    let instruct_ids_tensor = instruct_ids.map(|ids| cx.tensor((1, ids.len())));

    let outputs = pipeline.assemble_nonstreaming_prompt(&NonStreamingPromptInputs {
        instruct_ids: instruct_ids_tensor,
        role_ids: role_ids_tensor,
        overlay_text_ids: overlay_text_ids_tensor,
        overlay_codec_ids: overlay_codec_ids_tensor,
        content_ids: content_ids_tensor,
        content_codec_ids: content_codec_ids_tensor,
        tts_eos_id: tts_eos_id_tensor,
        transition_text_id: transition_text_id_tensor,
        transition_codec_id: transition_codec_id_tensor,
    });

    let initial_embeds_out = outputs.initial_embeds.output();
    let tts_pad_embed_out = outputs.tts_pad_embed.output();

    let mut rt = backend::compile(&mut cx, weights);

    backend::set_data(&mut rt, role_ids_tensor.id, role_ids.to_vec());
    backend::set_data(&mut rt, content_ids_tensor.id, content_ids.to_vec());
    backend::set_data(
        &mut rt,
        overlay_text_ids_tensor.id,
        overlay_text_ids.to_vec(),
    );
    backend::set_data(
        &mut rt,
        overlay_codec_ids_tensor.id,
        overlay_codec_ids.to_vec(),
    );
    backend::set_data(
        &mut rt,
        content_codec_ids_tensor.id,
        content_codec_ids.to_vec(),
    );
    backend::set_data(&mut rt, tts_eos_id_tensor.id, vec![tts_eos_id]);
    backend::set_data(
        &mut rt,
        transition_text_id_tensor.id,
        vec![transition_text_id],
    );
    backend::set_data(
        &mut rt,
        transition_codec_id_tensor.id,
        vec![transition_codec_id],
    );
    if let (Some(ids), Some(tensor)) = (instruct_ids, instruct_ids_tensor) {
        backend::set_data(&mut rt, tensor.id, ids.to_vec());
    }

    rt.execute(&cx.dyn_map);
    (
        backend::get_f32(&rt, initial_embeds_out.id),
        backend::get_f32(&rt, tts_pad_embed_out.id),
    )
}

// ===== decode_speech (Step 4: hoisted compilation + SnakeBeta/matmul reuse) =====

pub fn decode_speech(
    frames: &[Vec<u32>],
    speech_config: &SpeechDecoderConfig,
    weights: &HashMap<String, Vec<f32>>,
) -> Vec<f32> {
    use crate::speech_decoder::{
        CausalConv1d, CausalTransConv1d, ConvNeXtBlock, SnakeBeta, SpeechDecoder,
    };

    if frames.is_empty() {
        return Vec::new();
    }

    let t0 = std::time::Instant::now();

    let num_frames = frames.len();
    let num_codebooks =
        speech_config.num_semantic_quantizers + speech_config.num_acoustic_quantizers;

    let mut codes_by_codebook: Vec<Vec<i32>> = vec![Vec::with_capacity(num_frames); num_codebooks];
    for frame in frames {
        assert_eq!(
            frame.len(),
            num_codebooks,
            "frame codebook count must match speech decoder quantizer count"
        );
        for (codebook_idx, &code) in frame.iter().enumerate() {
            codes_by_codebook[codebook_idx].push(code as i32);
        }
    }

    // Pre-calculate all shapes for deterministic compilation
    let (s2a_hidden_len, block_shapes) = compute_block_shapes(num_frames, speech_config);
    let latent_dim = speech_config.latent_dim;
    let num_blocks = speech_config.upsample_rates.len();

    // ===== COMPILE PHASE =====
    // All graphs are built and compiled before any execution.

    // --- Stage 1: quantizer + pre_conv + pre_transformer ---
    let mut cx1 = Graph::new();
    let decoder = SpeechDecoder::new(&mut cx1, speech_config.clone());
    let code_tensors: Vec<GraphTensor> = (0..num_codebooks)
        .map(|_| cx1.tensor((1, num_frames)).as_dtype(DType::Int))
        .collect();
    let quantized = decoder.quantizer.decode(code_tensors.clone());
    let quantized = crate::maybe_graph_break(quantized);
    let latent = decoder.pre_conv.forward(quantized);
    let s1_hidden = latent.transpose(1, 2);
    let s1_hidden = crate::maybe_graph_break(s1_hidden);
    let transformed = decoder.pre_transformer.forward(s1_hidden, speech_config);
    let latent_out_tensor = transformed.transpose(1, 2);
    let latent_out = latent_out_tensor.output();
    let mut rt1 = backend::compile(&mut cx1, weights);
    eprintln!(
        "  [compile] speech stage1 in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    // --- Stage 2a: initial_upsample + initial_conv ---
    let latent_len = num_frames;
    let mut s2a_cx = Graph::new();
    let s2a_input = s2a_cx.tensor((1, latent_dim, latent_len));
    let mut s2a_h = s2a_input;
    for (i, ratio) in speech_config.upsampling_ratios.iter().copied().enumerate() {
        let trans_conv =
            CausalTransConv1d::new(latent_dim, latent_dim, ratio, ratio, true, &mut s2a_cx);
        trans_conv
            .conv
            .weight
            .set_name(&format!("decoder.upsample.{i}.0.conv.weight"));
        if let Some(bias) = trans_conv.conv.bias {
            bias.set_name(&format!("decoder.upsample.{i}.0.conv.bias"));
        }
        let convnext =
            ConvNeXtBlock::new(latent_dim, format!("decoder.upsample.{i}.1"), &mut s2a_cx);
        s2a_h = trans_conv.forward(s2a_h);
        s2a_h = crate::maybe_graph_break(s2a_h);
        s2a_h = convnext.forward(s2a_h);
        s2a_h = crate::maybe_graph_break(s2a_h);
    }
    let initial_conv =
        CausalConv1d::new(latent_dim, speech_config.decoder_dim, 7, 1, true, &mut s2a_cx, 1);
    initial_conv
        .conv
        .weight
        .set_name("decoder.decoder.0.conv.weight");
    if let Some(bias) = initial_conv.conv.bias {
        bias.set_name("decoder.decoder.0.conv.bias");
    }
    s2a_h = initial_conv.forward(s2a_h);
    let s2a_output = s2a_h.output();
    let mut s2a_rt = backend::compile(&mut s2a_cx, weights);
    eprintln!(
        "  [compile] speech stage2a in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    // --- Per-block graphs (4 graphs per block, compiled upfront) ---
    let mut compiled_blocks: Vec<BlockGraphs> = Vec::with_capacity(num_blocks);
    for (i, shapes) in block_shapes.iter().enumerate() {
        let block_prefix = format!("decoder.decoder.{}", i + 1);
        let stage_name = format!("stage2{}", (b'b' + i as u8) as char);

        // 1. Snake + upsample + pad graph (named tensors)
        let mut su_cx = Graph::new();
        let su_input = su_cx.tensor((1, shapes.in_dim, shapes.hidden_len));
        let snake = SnakeBeta::new(
            shapes.in_dim,
            &format!("{block_prefix}.block.0.alpha"),
            &format!("{block_prefix}.block.0.beta"),
            &mut su_cx,
        );
        let su_h = snake.forward(su_input);
        let upsampled = su_h
            .expand_dim(3, 1)
            .pad_along(0, shapes.rate - 1, 3, 0.0)
            .merge_dims(2, 3)
            .slice_along(..shapes.upsampled_len, 2);
        let conv_padding = shapes.kernel_size - 1;
        let padded = upsampled.pad(((0, 0), (0, 0), (conv_padding, conv_padding)), 0.0);
        let su_output = padded.output();
        let su_rt = backend::compile(&mut su_cx, weights);

        // 2. Upsample conv matmul graph (unnamed tensors)
        let (uc_cx, uc_rt, uc_ids) =
            compile_matmul_graph(shapes.out_len, shapes.in_dim, shapes.out_dim);

        // 3. Residual SnakeBeta graph (unnamed tensors, reused for all 6 calls)
        let (rs_cx, rs_rt, rs_ids) =
            compile_snake_beta_graph(shapes.out_dim, shapes.target_len);

        // 4. Residual conv matmul graph (unnamed tensors, reused for all 6 calls)
        let (rc_cx, rc_rt, rc_ids) =
            compile_matmul_graph(shapes.target_len, shapes.out_dim, shapes.out_dim);

        compiled_blocks.push(BlockGraphs {
            snake_up_cx: su_cx,
            snake_up_rt: su_rt,
            snake_up_input: su_input,
            snake_up_output: su_output,
            up_conv_cx: uc_cx,
            up_conv_rt: uc_rt,
            up_conv_ids: uc_ids,
            res_snake_cx: rs_cx,
            res_snake_rt: rs_rt,
            res_snake_ids: rs_ids,
            res_conv_cx: rc_cx,
            res_conv_rt: rc_rt,
            res_conv_ids: rc_ids,
        });

        eprintln!(
            "  [compile] speech {} (4 graphs) in {:.1}s",
            stage_name,
            t0.elapsed().as_secs_f32()
        );
    }

    // --- Final stage: final_snake + final_conv + clip ---
    let final_dim = block_shapes
        .last()
        .map_or(speech_config.decoder_dim, |s| s.out_dim);
    let final_hidden_len = block_shapes
        .last()
        .map_or(s2a_hidden_len, |s| s.target_len);
    let final_snake_idx = num_blocks + 1;
    let final_conv_idx = final_snake_idx + 1;

    let mut fin_cx = Graph::new();
    let fin_input = fin_cx.tensor((1, final_dim, final_hidden_len));
    let final_snake = SnakeBeta::new(
        final_dim,
        &format!("decoder.decoder.{final_snake_idx}.alpha"),
        &format!("decoder.decoder.{final_snake_idx}.beta"),
        &mut fin_cx,
    );
    let final_conv = CausalConv1d::new(final_dim, 1, 7, 1, true, &mut fin_cx, 1);
    final_conv
        .conv
        .weight
        .set_name(&format!("decoder.decoder.{final_conv_idx}.conv.weight"));
    if let Some(bias) = final_conv.conv.bias {
        bias.set_name(&format!("decoder.decoder.{final_conv_idx}.conv.bias"));
    }
    let mut fin_h = final_snake.forward(fin_input);
    fin_h = final_conv.forward(fin_h);
    fin_h = fin_h.clip(-1.0, 1.0);
    let fin_output = fin_h.output();
    let mut fin_rt = backend::compile(&mut fin_cx, weights);
    eprintln!(
        "  [compile] speech final in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    let compile_time = t0.elapsed();
    eprintln!(
        "  [decode_speech] all compiled in {:.1}s",
        compile_time.as_secs_f32()
    );

    // ===== EXECUTE PHASE =====
    let t1 = std::time::Instant::now();

    // --- Execute stage 1 ---
    for (i, tensor) in code_tensors.iter().enumerate() {
        backend::set_data_i32(&mut rt1, tensor.id, codes_by_codebook[i].clone());
    }
    rt1.execute(&cx1.dyn_map);
    let latent_data = backend::get_f32(&rt1, latent_out.id);
    eprintln!(
        "  [execute] speech stage1 in {:.1}s, latent size {}",
        t1.elapsed().as_secs_f32(),
        latent_data.len(),
    );
    drop(rt1);

    // --- Execute stage 2a ---
    backend::set_data(&mut s2a_rt, s2a_input.id, latent_data);
    s2a_rt.execute(&s2a_cx.dyn_map);
    let mut hidden_data = backend::get_f32(&s2a_rt, s2a_output.id);
    let mut hidden_ch = speech_config.decoder_dim;
    let mut hidden_len = hidden_data.len() / hidden_ch;
    eprintln!(
        "  [execute] speech stage2a in {:.1}s, hidden [{}, {}, {}]",
        t1.elapsed().as_secs_f32(),
        1,
        hidden_ch,
        hidden_len,
    );
    drop(s2a_rt);

    // --- Execute decoder blocks ---
    for (i, (block, shapes)) in compiled_blocks
        .iter_mut()
        .zip(block_shapes.iter())
        .enumerate()
    {
        let block_prefix = format!("decoder.decoder.{}", i + 1);
        let stage_name = format!("stage2{}", (b'b' + i as u8) as char);

        // Sub-stage A: snake + upsample + pad
        backend::set_data(
            &mut block.snake_up_rt,
            block.snake_up_input.id,
            hidden_data,
        );
        block.snake_up_rt.execute(&block.snake_up_cx.dyn_map);
        hidden_data = backend::get_f32(&block.snake_up_rt, block.snake_up_output.id);
        let padded_ch = shapes.in_dim;
        let padded_len = hidden_data.len() / padded_ch;
        eprintln!(
            "  [execute] speech {}/snake+upsample in {:.1}s, padded [{}, {}, {}]",
            stage_name,
            t1.elapsed().as_secs_f32(),
            1,
            padded_ch,
            padded_len,
        );

        // Sub-stage B: per-kernel-position matmul (upsample conv)
        {
            let padded_data = &hidden_data;
            let mut accum_data = vec![0.0f32; shapes.out_len * shapes.out_dim];

            // Load weight + bias from weights map (CPU-side)
            let weight_name = format!("{block_prefix}.block.1.conv.weight");
            let weight_data = weights
                .get(&weight_name)
                .unwrap_or_else(|| panic!("missing weight: {weight_name}"));
            // weight_data layout: [ch_in, ch_out, kernel_size] row-major
            let bias_name = format!("{block_prefix}.block.1.conv.bias");
            let bias_data = weights
                .get(&bias_name)
                .unwrap_or_else(|| panic!("missing bias: {bias_name}"));

            for k in 0..shapes.kernel_size {
                let reversed_k = shapes.kernel_size - 1 - k;

                // CPU: slice padded[:, :, k..k+out_len] and transpose to [1, out_len, ch_in]
                let mut slice_t_data = vec![0.0f32; shapes.out_len * shapes.in_dim];
                for c in 0..shapes.in_dim {
                    for t in 0..shapes.out_len {
                        slice_t_data[t * shapes.in_dim + c] =
                            padded_data[c * padded_len + k + t];
                    }
                }

                // CPU: slice weight[:, :, reversed_k] → [ch_in, ch_out]
                let mut weight_k_data = vec![0.0f32; shapes.in_dim * shapes.out_dim];
                for ci in 0..shapes.in_dim {
                    for co in 0..shapes.out_dim {
                        weight_k_data[ci * shapes.out_dim + co] = weight_data[ci
                            * shapes.out_dim
                            * shapes.kernel_size
                            + co * shapes.kernel_size
                            + reversed_k];
                    }
                }

                backend::set_data(
                    &mut block.up_conv_rt,
                    block.up_conv_ids.slice_input.id,
                    slice_t_data,
                );
                backend::set_data(
                    &mut block.up_conv_rt,
                    block.up_conv_ids.weight_input.id,
                    weight_k_data,
                );
                backend::set_data(
                    &mut block.up_conv_rt,
                    block.up_conv_ids.accum_input.id,
                    accum_data,
                );
                block.up_conv_rt.execute(&block.up_conv_cx.dyn_map);
                accum_data = backend::get_f32(&block.up_conv_rt, block.up_conv_ids.output.id);
            }

            // Transpose [1, out_len, ch_out] → [1, ch_out, out_len]
            let mut transposed = vec![0.0f32; shapes.out_len * shapes.out_dim];
            for c in 0..shapes.out_dim {
                for t in 0..shapes.out_len {
                    transposed[c * shapes.out_len + t] = accum_data[t * shapes.out_dim + c];
                }
            }

            // Add bias
            for c in 0..shapes.out_dim {
                let b = bias_data[c];
                for t in 0..shapes.out_len {
                    transposed[c * shapes.out_len + t] += b;
                }
            }

            // Trim for causal: keep first target_len samples
            let mut trimmed = vec![0.0f32; shapes.out_dim * shapes.target_len];
            for c in 0..shapes.out_dim {
                for t in 0..shapes.target_len {
                    trimmed[c * shapes.target_len + t] = transposed[c * shapes.out_len + t];
                }
            }

            hidden_data = trimmed;
            hidden_ch = shapes.out_dim;
            hidden_len = shapes.target_len;
            eprintln!(
                "  [execute] speech {}/conv ({} matmuls) in {:.1}s, hidden [{}, {}, {}]",
                stage_name,
                shapes.kernel_size,
                t1.elapsed().as_secs_f32(),
                1,
                hidden_ch,
                hidden_len,
            );
        }

        // Sub-stages: each DecoderResidualUnit (reusing SnakeBeta + conv matmul graphs)
        for (j, dilation) in [1usize, 3, 9].iter().enumerate() {
            let prefix = format!("{block_prefix}.block.{}", j + 2);
            let residual_data = hidden_data.clone();

            // SnakeBeta 1
            execute_snake_beta(
                &mut hidden_data,
                &block.res_snake_cx,
                &mut block.res_snake_rt,
                &block.res_snake_ids,
                &format!("{prefix}.act1.alpha"),
                &format!("{prefix}.act1.beta"),
                weights,
            );

            // CausalConv1d 1 (kernel=7, dilation=dilation)
            causal_conv1d_with_matmul(
                &mut hidden_data,
                &block.res_conv_cx,
                &mut block.res_conv_rt,
                &block.res_conv_ids,
                shapes.out_dim,
                shapes.out_dim,
                shapes.target_len,
                7,
                *dilation,
                &format!("{prefix}.conv1.conv"),
                weights,
            );

            // SnakeBeta 2
            execute_snake_beta(
                &mut hidden_data,
                &block.res_snake_cx,
                &mut block.res_snake_rt,
                &block.res_snake_ids,
                &format!("{prefix}.act2.alpha"),
                &format!("{prefix}.act2.beta"),
                weights,
            );

            // CausalConv1d 2 (kernel=1, dilation=1)
            causal_conv1d_with_matmul(
                &mut hidden_data,
                &block.res_conv_cx,
                &mut block.res_conv_rt,
                &block.res_conv_ids,
                shapes.out_dim,
                shapes.out_dim,
                shapes.target_len,
                1,
                1,
                &format!("{prefix}.conv2.conv"),
                weights,
            );

            // Residual add (CPU)
            for idx in 0..hidden_data.len() {
                hidden_data[idx] += residual_data[idx];
            }

            eprintln!(
                "  [execute] speech {}/res{} done at {:.1}s",
                stage_name,
                j,
                t1.elapsed().as_secs_f32()
            );
        }
    }

    // --- Execute final stage ---
    backend::set_data(&mut fin_rt, fin_input.id, hidden_data);
    fin_rt.execute(&fin_cx.dyn_map);
    hidden_data = backend::get_f32(&fin_rt, fin_output.id);
    eprintln!(
        "  [execute] speech final in {:.1}s, {} audio samples",
        t1.elapsed().as_secs_f32(),
        hidden_data.len(),
    );
    drop(fin_rt);

    let exec_time = t1.elapsed();
    eprintln!(
        "  [decode_speech] compile: {:.1}s, execute: {:.1}s, total: {:.1}s",
        compile_time.as_secs_f32(),
        exec_time.as_secs_f32(),
        t0.elapsed().as_secs_f32()
    );

    hidden_data
}
