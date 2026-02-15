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

    // ===== PREFILL PHASE (Talker) =====
    // Process per-layer to keep peak GPU memory low — each layer gets its own
    // Graph/compile/execute/drop cycle so intermediate buffers are freed between layers.
    let t0 = std::time::Instant::now();

    // --- Per-layer transformer pass ---
    let mut hidden_data = initial_embeds.to_vec();
    let mut prefill_k: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
    let mut prefill_v: Vec<Vec<f32>> = Vec::with_capacity(n_layers);

    for layer_idx in 0..n_layers {
        let mut cx = Graph::new();
        let input = cx.tensor((1, prompt_len, hidden));
        let layer = TalkerLayer::new(&mut cx, talker_config, layer_idx);
        let (out, k_cache, v_cache) = layer.forward_with_kv(input, talker_config);
        let out_t = out.output();
        let k_t = k_cache.output();
        let v_t = v_cache.output();

        let mut rt = backend::compile(&mut cx, weights);
        backend::set_data(&mut rt, input.id, hidden_data);
        rt.execute(&cx.dyn_map);

        hidden_data = backend::get_f32(&rt, out_t.id);
        prefill_k.push(backend::get_f32(&rt, k_t.id));
        prefill_v.push(backend::get_f32(&rt, v_t.id));
        // rt dropped here, freeing GPU memory
    }
    eprintln!(
        "  [prefill/talker] {} layers done in {:.1}s",
        n_layers,
        t0.elapsed().as_secs_f32()
    );

    // --- Final stage: norm + logits + argmax + embed ---
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
    // embed_codec: gather from codec_embedding using code_0
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
        "  [prefill/talker] complete in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    // --- Graph 2: Code predictor (15 autoregressive codebook steps) ---
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

    eprintln!(
        "  [prefill/predictor] graph built in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    let mut pred_rt = backend::compile(&mut pred_cx, weights);
    eprintln!(
        "  [prefill/predictor] compiled in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    backend::set_data(&mut pred_rt, pred_hidden_input.id, last_hidden_data);
    backend::set_data(&mut pred_rt, pred_code0_embed_input.id, code_0_embed_data);
    pred_rt.execute(&pred_cx.dyn_map);
    eprintln!(
        "  [prefill/predictor] executed in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

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

    drop(pred_rt);
    eprintln!("  [prefill/predictor] runtime dropped, freeing GPU memory");

    // ===== DECODE PHASE =====
    // Split into separate talker decode and code predictor graphs (same
    // rationale as prefill). The code predictor graph is reused from prefill
    // since its inputs (hidden [1,1,H] + code_0_embed [1,1,H]) are identical.
    let max_seq = prompt_len + num_frames;
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
    // Last position is always the concat slot for the current token's K/V.
    attn_mask[max_seq] = 0.0;

    // --- Decode talker graph (28 layers + argmax + embed_codec) ---
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

    eprintln!(
        "  [decode/talker] graph built in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    let mut decode_rt = backend::compile(&mut decode_cx, weights);
    eprintln!(
        "  [decode/talker] compiled in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    // Reuse code predictor from prefill (already compiled as pred_rt)
    // Reconstruct it since it was dropped — same graph structure, same weights
    let mut pred_cx2 = Graph::new();
    let code_predictor2 = CodePredictorModel::new(&mut pred_cx2, predictor_config.clone());
    let pred_hidden_input2 = pred_cx2.tensor((1, 1, hidden));
    let pred_code0_embed_input2 = pred_cx2.tensor((1, 1, hidden));
    let pred_out2 =
        code_predictor2.generate_codes(pred_hidden_input2, pred_code0_embed_input2);
    let mut pred_codec_sum2 = pred_code0_embed_input2;
    for embed in &pred_out2.embeds {
        pred_codec_sum2 = pred_codec_sum2 + *embed;
    }
    let pred_code_outs2: Vec<_> = pred_out2
        .codes
        .iter()
        .map(|code| code.cast(DType::F32).output())
        .collect();
    let pred_codec_sum_out2 = pred_codec_sum2.output();
    let mut pred_rt2 = backend::compile(&mut pred_cx2, weights);
    eprintln!(
        "  [decode/predictor] compiled in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    for frame in 1..num_frames {
        let p = prompt_len + frame - 1;

        // Small inputs: always full-upload (tiny data ~10 KB)
        backend::set_data(&mut decode_rt, new_embed_input.id, next_embed.clone());
        backend::set_data(&mut decode_rt, pos_input.id, vec![p as f32]);
        backend::set_data(&mut decode_rt, mask_input.id, attn_mask.clone());

        // CUDA: full KV upload only on frame 1 to establish GPU buffers;
        // subsequent frames use partial updates applied after previous frame.
        // Non-CUDA: always full-upload (no GPU transfer cost).
        if frame == 1 || cfg!(not(feature = "cuda")) {
            for (i, (k_in, v_in)) in kv_inputs.iter().enumerate() {
                backend::set_data(&mut decode_rt, k_in.id, k_bufs[i].clone());
                backend::set_data(&mut decode_rt, v_in.id, v_bufs[i].clone());
            }
        }

        decode_rt.execute(&decode_cx.dyn_map);

        let code_0_val = backend::get_f32(&decode_rt, decode_code_0_out.id)[0] as u32;
        if code_0_val == CODEC_EOS_ID as u32 {
            eprintln!("  [decode] EOS at frame {}", frame);
            break;
        }

        let code_0_embed_data = backend::get_f32(&decode_rt, decode_code_0_embed_out.id);
        let normed_data = backend::get_f32(&decode_rt, decode_normed_out.id);

        // Run code predictor
        backend::set_data(&mut pred_rt2, pred_hidden_input2.id, normed_data);
        backend::set_data(&mut pred_rt2, pred_code0_embed_input2.id, code_0_embed_data);
        pred_rt2.execute(&pred_cx2.dyn_map);

        let pred_codes: Vec<u32> = pred_code_outs2
            .iter()
            .map(|code| backend::get_f32(&pred_rt2, code.id)[0] as u32)
            .collect();
        let mut frame_codes = vec![code_0_val];
        frame_codes.extend(pred_codes);
        all_frames.push(frame_codes);

        eprintln!(
            "  [decode] frame {} done at {:.1}s",
            frame,
            t0.elapsed().as_secs_f32()
        );

        // Scatter new K/V into CPU buffers AND do partial GPU updates for next frame
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

                // Update CPU buffer (keep for correctness)
                k_bufs[i][dst_start..dst_end].copy_from_slice(&new_k[src_start..src_end]);
                v_bufs[i][dst_start..dst_end].copy_from_slice(&new_v[src_start..src_end]);

                // CUDA: partial GPU update - write only head_dim floats at byte offset
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

        let codec_sum_data = backend::get_f32(&pred_rt2, pred_codec_sum_out2.id);
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
    }

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

pub fn decode_speech(
    frames: &[Vec<u32>],
    speech_config: &SpeechDecoderConfig,
    weights: &HashMap<String, Vec<f32>>,
) -> Vec<f32> {
    use crate::speech_decoder::{
        CausalConv1d, CausalTransConv1d, ConvNeXtBlock, DecoderResidualUnit, SnakeBeta,
        SpeechDecoder,
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

    // ===== Stage 1: Quantizer + pre_conv + pre_transformer → latent =====
    let mut cx1 = Graph::new();
    let decoder = SpeechDecoder::new(&mut cx1, speech_config.clone());
    let code_tensors: Vec<GraphTensor> = (0..num_codebooks)
        .map(|_| cx1.tensor((1, num_frames)).as_dtype(DType::Int))
        .collect();
    let quantized = decoder.quantizer.decode(code_tensors.clone());
    let quantized = crate::maybe_graph_break(quantized);
    let latent = decoder.pre_conv.forward(quantized);
    let hidden = latent.transpose(1, 2);
    let hidden = crate::maybe_graph_break(hidden);
    let transformed = decoder.pre_transformer.forward(hidden, speech_config);
    let latent_out_tensor = transformed.transpose(1, 2);
    let latent_out = latent_out_tensor.output();

    eprintln!(
        "  [speech_decode/stage1] graph built in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    let mut rt1 = backend::compile(&mut cx1, weights);
    eprintln!(
        "  [speech_decode/stage1] compiled in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    for (i, tensor) in code_tensors.iter().enumerate() {
        backend::set_data_i32(&mut rt1, tensor.id, codes_by_codebook[i].clone());
    }
    rt1.execute(&cx1.dyn_map);
    let latent_data = backend::get_f32(&rt1, latent_out.id);
    eprintln!(
        "  [speech_decode/stage1] executed in {:.1}s, latent size {}",
        t0.elapsed().as_secs_f32(),
        latent_data.len(),
    );
    drop(rt1);

    // ===== Stage 2a: initial_upsample + initial_conv =====
    let latent_dim = speech_config.latent_dim;
    let latent_len = latent_data.len() / latent_dim;
    assert_eq!(
        latent_data.len(),
        latent_dim * latent_len,
        "latent data size must be divisible by latent_dim"
    );

    let mut hidden_data = latent_data;
    let mut hidden_ch = latent_dim;
    let mut hidden_len = latent_len;

    {
        let mut cx = Graph::new();
        let input = cx.tensor((1, hidden_ch, hidden_len));
        let mut hidden = input;

        for (i, ratio) in speech_config.upsampling_ratios.iter().copied().enumerate() {
            let trans_conv =
                CausalTransConv1d::new(latent_dim, latent_dim, ratio, ratio, true, &mut cx);
            trans_conv
                .conv
                .weight
                .set_name(&format!("decoder.upsample.{i}.0.conv.weight"));
            if let Some(bias) = trans_conv.conv.bias {
                bias.set_name(&format!("decoder.upsample.{i}.0.conv.bias"));
            }
            let convnext =
                ConvNeXtBlock::new(latent_dim, format!("decoder.upsample.{i}.1"), &mut cx);
            hidden = trans_conv.forward(hidden);
            hidden = crate::maybe_graph_break(hidden);
            hidden = convnext.forward(hidden);
            hidden = crate::maybe_graph_break(hidden);
        }

        let initial_conv =
            CausalConv1d::new(latent_dim, speech_config.decoder_dim, 7, 1, true, &mut cx, 1);
        initial_conv
            .conv
            .weight
            .set_name("decoder.decoder.0.conv.weight");
        if let Some(bias) = initial_conv.conv.bias {
            bias.set_name("decoder.decoder.0.conv.bias");
        }
        hidden = initial_conv.forward(hidden);
        let out = hidden.output();

        eprintln!(
            "  [speech_decode/stage2a] graph built in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        let mut rt = backend::compile(&mut cx, weights);
        eprintln!(
            "  [speech_decode/stage2a] compiled in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        backend::set_data(&mut rt, input.id, hidden_data);
        rt.execute(&cx.dyn_map);
        hidden_data = backend::get_f32(&rt, out.id);
        hidden_ch = speech_config.decoder_dim;
        hidden_len = hidden_data.len() / hidden_ch;
        eprintln!(
            "  [speech_decode/stage2a] executed in {:.1}s, hidden [{}, {}, {}]",
            t0.elapsed().as_secs_f32(),
            1,
            hidden_ch,
            hidden_len,
        );
    }

    // ===== Stages 2b+: DecoderBlock sub-stages =====
    // Each DecoderBlock is split into separate Graph/compile/execute/drop cycles:
    //   1. SnakeBeta + CausalTransConv1d (upsample, changes shape)
    //   2-4. Each DecoderResidualUnit (preserves shape)
    // This prevents the 98+ GB combined intermediate allocation that OOMs on A100-80GB.
    let mut in_dim = speech_config.decoder_dim;
    let num_blocks = speech_config.upsample_rates.len();
    for (i, rate) in speech_config.upsample_rates.iter().copied().enumerate() {
        let out_dim = speech_config.decoder_dim / (1 << (i + 1));
        let block_prefix = format!("decoder.decoder.{}", i + 1);
        let stage_name = format!("stage2{}", (b'b' + i as u8) as char);

        // --- Sub-stage A: SnakeBeta + upsample + pad ---
        // Separate from matmuls so egglog doesn't fuse them into 79+ GB kernels
        let kernel_size = 2 * rate;
        let padded_ch = in_dim;
        let padded_len;
        {
            let mut cx = Graph::new();
            let input = cx.tensor((1, in_dim, hidden_len));
            let snake = SnakeBeta::new(
                in_dim,
                &format!("{block_prefix}.block.0.alpha"),
                &format!("{block_prefix}.block.0.beta"),
                &mut cx,
            );
            let h = snake.forward(input);
            // Upsample: zero-interleave by stride
            let upsampled_len = (hidden_len - 1) * rate + 1;
            let upsampled = h
                .expand_dim(3, 1)
                .pad_along(0, rate - 1, 3, 0.0)
                .merge_dims(2, 3)
                .slice_along(..upsampled_len, 2);
            // Pad for full convolution
            let conv_padding = kernel_size - 1;
            let padded = upsampled.pad(((0, 0), (0, 0), (conv_padding, conv_padding)), 0.0);
            let out = padded.output();

            eprintln!(
                "  [speech_decode/{}/snake+upsample] graph built in {:.1}s",
                stage_name,
                t0.elapsed().as_secs_f32()
            );
            let mut rt = backend::compile(&mut cx, weights);
            eprintln!(
                "  [speech_decode/{}/snake+upsample] compiled in {:.1}s",
                stage_name,
                t0.elapsed().as_secs_f32()
            );
            backend::set_data(&mut rt, input.id, hidden_data);
            rt.execute(&cx.dyn_map);
            hidden_data = backend::get_f32(&rt, out.id);
            padded_len = hidden_data.len() / padded_ch;
            eprintln!(
                "  [speech_decode/{}/snake+upsample] executed in {:.1}s, padded [{}, {}, {}]",
                stage_name,
                t0.elapsed().as_secs_f32(),
                1,
                padded_ch,
                padded_len,
            );
        }

        // --- Sub-stage B: Per-kernel-position matmul (single compiled graph, reused) ---
        // Compiles ONE matmul+accum graph and reuses it for each kernel position.
        // CPU handles slicing the padded tensor and weight per position, GPU does matmul.
        // This avoids egglog fusing all positions into a single 242+ GB kernel.
        {
            let upsampled_len = (hidden_len - 1) * rate + 1;
            let conv_padding = kernel_size - 1;
            let out_len = upsampled_len + conv_padding;

            // Build a reusable matmul+accum graph (unnamed tensors, no weights)
            let mut cx = Graph::new();
            let slice_input = cx.tensor((1, out_len, in_dim)); // [1, out_len, ch_in]
            let weight_input = cx.tensor((in_dim, out_dim)); // [ch_in, ch_out]
            let accum_input = cx.tensor((1, out_len, out_dim)); // [1, out_len, ch_out]
            let partial = slice_input.matmul(weight_input);
            let result = accum_input + partial;
            let out = result.output();

            eprintln!(
                "  [speech_decode/{}/conv] matmul graph built ({} kernel positions) in {:.1}s",
                stage_name,
                kernel_size,
                t0.elapsed().as_secs_f32()
            );
            let mut rt = backend::compile(&mut cx, &HashMap::new());
            eprintln!(
                "  [speech_decode/{}/conv] compiled in {:.1}s",
                stage_name,
                t0.elapsed().as_secs_f32()
            );

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

            // hidden_data from Sub-stage A is [1, ch_in, padded_len] row-major
            let padded_data = &hidden_data;
            let mut accum_data = vec![0.0f32; out_len * out_dim];

            for k in 0..kernel_size {
                let reversed_k = kernel_size - 1 - k;

                // CPU: slice padded[:, :, k..k+out_len] and transpose to [1, out_len, ch_in]
                let mut slice_t_data = vec![0.0f32; out_len * in_dim];
                for c in 0..in_dim {
                    for t in 0..out_len {
                        slice_t_data[t * in_dim + c] = padded_data[c * padded_len + k + t];
                    }
                }

                // CPU: slice weight[:, :, reversed_k] → [ch_in, ch_out]
                let mut weight_k_data = vec![0.0f32; in_dim * out_dim];
                for ci in 0..in_dim {
                    for co in 0..out_dim {
                        weight_k_data[ci * out_dim + co] = weight_data
                            [ci * out_dim * kernel_size + co * kernel_size + reversed_k];
                    }
                }

                backend::set_data(&mut rt, slice_input.id, slice_t_data);
                backend::set_data(&mut rt, weight_input.id, weight_k_data);
                backend::set_data(&mut rt, accum_input.id, accum_data);
                rt.execute(&cx.dyn_map);
                accum_data = backend::get_f32(&rt, out.id);
            }

            // accum_data is [1, out_len, ch_out] row-major → transpose to [1, ch_out, out_len]
            let mut transposed = vec![0.0f32; out_len * out_dim];
            for c in 0..out_dim {
                for t in 0..out_len {
                    transposed[c * out_len + t] = accum_data[t * out_dim + c];
                }
            }

            // Add bias: each channel gets bias[c] added to all timesteps
            for c in 0..out_dim {
                let b = bias_data[c];
                for t in 0..out_len {
                    transposed[c * out_len + t] += b;
                }
            }

            // Trim for causal: keep first hidden_len * stride samples
            let target_len = hidden_len * rate;
            let mut trimmed = vec![0.0f32; out_dim * target_len];
            for c in 0..out_dim {
                for t in 0..target_len {
                    trimmed[c * target_len + t] = transposed[c * out_len + t];
                }
            }

            hidden_data = trimmed;
            hidden_ch = out_dim;
            hidden_len = target_len;
            eprintln!(
                "  [speech_decode/{}/conv] executed ({} matmuls) in {:.1}s, hidden [{}, {}, {}]",
                stage_name,
                kernel_size,
                t0.elapsed().as_secs_f32(),
                1,
                hidden_ch,
                hidden_len,
            );
        }

        // --- Sub-stages: each DecoderResidualUnit ---
        for (j, dilation) in [1usize, 3, 9].iter().enumerate() {
            let mut cx = Graph::new();
            let input = cx.tensor((1, out_dim, hidden_len));
            let unit = DecoderResidualUnit::new(
                out_dim,
                *dilation,
                format!("{block_prefix}.block.{}", j + 2),
                &mut cx,
            );
            let h = unit.forward(input);
            let out = h.output();

            eprintln!(
                "  [speech_decode/{}/res{}] graph built in {:.1}s",
                stage_name,
                j,
                t0.elapsed().as_secs_f32()
            );
            let mut rt = backend::compile(&mut cx, weights);
            eprintln!(
                "  [speech_decode/{}/res{}] compiled in {:.1}s",
                stage_name,
                j,
                t0.elapsed().as_secs_f32()
            );
            backend::set_data(&mut rt, input.id, hidden_data);
            rt.execute(&cx.dyn_map);
            hidden_data = backend::get_f32(&rt, out.id);
            eprintln!(
                "  [speech_decode/{}/res{}] executed in {:.1}s",
                stage_name,
                j,
                t0.elapsed().as_secs_f32()
            );
        }

        in_dim = out_dim;
    }

    // ===== Final stage: final_snake + final_conv + clip =====
    {
        let final_dim = in_dim; // channel dim after last block (or decoder_dim if no blocks)
        let final_snake_idx = num_blocks + 1;
        let final_conv_idx = final_snake_idx + 1;

        let mut cx = Graph::new();
        let input = cx.tensor((1, final_dim, hidden_len));
        let final_snake = SnakeBeta::new(
            final_dim,
            &format!("decoder.decoder.{final_snake_idx}.alpha"),
            &format!("decoder.decoder.{final_snake_idx}.beta"),
            &mut cx,
        );
        let final_conv = CausalConv1d::new(final_dim, 1, 7, 1, true, &mut cx, 1);
        final_conv
            .conv
            .weight
            .set_name(&format!("decoder.decoder.{final_conv_idx}.conv.weight"));
        if let Some(bias) = final_conv.conv.bias {
            bias.set_name(&format!("decoder.decoder.{final_conv_idx}.conv.bias"));
        }
        let mut hidden = final_snake.forward(input);
        hidden = final_conv.forward(hidden);
        hidden = hidden.clip(-1.0, 1.0);
        let out = hidden.output();

        eprintln!(
            "  [speech_decode/final] graph built in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        let mut rt = backend::compile(&mut cx, weights);
        eprintln!(
            "  [speech_decode/final] compiled in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        backend::set_data(&mut rt, input.id, hidden_data);
        rt.execute(&cx.dyn_map);
        hidden_data = backend::get_f32(&rt, out.id);
        eprintln!(
            "  [speech_decode/final] executed in {:.1}s, {} audio samples",
            t0.elapsed().as_secs_f32(),
            hidden_data.len(),
        );
    }

    hidden_data
}
