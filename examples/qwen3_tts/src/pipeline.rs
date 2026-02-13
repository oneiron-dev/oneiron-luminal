use crate::code_predictor::{CodePredictorConfig, CodePredictorModel};
use crate::model::{TalkerConfig, TalkerModel};
use crate::speech_decoder::{SpeechDecoder, SpeechDecoderConfig};
use crate::weight_loader::load_weights_from_map;
use luminal::{
    graph::Graph,
    op::{DType, Runtime},
    prelude::{GraphTensor, NativeRuntime},
};
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

    let mut embeddings = initial_embeds.to_vec();
    let mut all_frames = Vec::with_capacity(num_frames);

    for frame in 0..num_frames {
        let seq_len = prompt_len + frame;

        let mut cx = Graph::new();
        let pipeline = TtsPipeline::new(&mut cx, talker_config.clone(), predictor_config.clone());
        let prompt = cx.tensor((1, seq_len, hidden));

        let (logits, normed) = pipeline.talker.decode_step(prompt);
        let last_logits = logits.slice((.., (seq_len - 1).., ..));
        let code_0 = last_logits.argmax(2);
        let code_0_embed = pipeline.talker.embed_codec(code_0);
        let last_hidden = normed.slice((.., (seq_len - 1).., ..));
        let pred_out = pipeline
            .code_predictor
            .generate_codes(last_hidden, code_0_embed);

        let mut codec_sum = code_0_embed;
        for embed in &pred_out.embeds {
            codec_sum = codec_sum + *embed;
        }

        let code_0_out = code_0.cast(DType::F32).output();
        let code_outs: Vec<_> = pred_out
            .codes
            .iter()
            .map(|code| code.cast(DType::F32).output())
            .collect();
        let codec_sum_out = codec_sum.output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);
        load_weights_from_map(&mut rt, &cx, weights);
        rt.set_data(prompt.id, embeddings.clone());
        rt.execute(&cx.dyn_map);

        let code_0_val = rt.get_f32(code_0_out.id)[0] as u32;
        if code_0_val == CODEC_EOS_ID as u32 {
            break;
        }

        let pred_codes: Vec<u32> = code_outs
            .iter()
            .map(|code| rt.get_f32(code.id)[0] as u32)
            .collect();

        let mut frame_codes = vec![code_0_val];
        frame_codes.extend(pred_codes);
        all_frames.push(frame_codes);

        let codec_sum_data = rt.get_f32(codec_sum_out.id);
        assert_eq!(
            codec_sum_data.len(),
            hidden,
            "codec_sum output length must equal hidden"
        );
        let combined: Vec<f32> = codec_sum_data
            .iter()
            .zip(tts_pad_embed.iter())
            .map(|(codec, tts)| codec + tts)
            .collect();
        embeddings.extend_from_slice(&combined);
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

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    load_weights_from_map(&mut rt, &cx, weights);

    rt.set_data(role_ids_tensor.id, role_ids.to_vec());
    rt.set_data(content_ids_tensor.id, content_ids.to_vec());
    rt.set_data(overlay_text_ids_tensor.id, overlay_text_ids.to_vec());
    rt.set_data(overlay_codec_ids_tensor.id, overlay_codec_ids.to_vec());
    rt.set_data(content_codec_ids_tensor.id, content_codec_ids.to_vec());
    rt.set_data(tts_eos_id_tensor.id, vec![tts_eos_id]);
    rt.set_data(transition_text_id_tensor.id, vec![transition_text_id]);
    rt.set_data(transition_codec_id_tensor.id, vec![transition_codec_id]);
    if let (Some(ids), Some(tensor)) = (instruct_ids, instruct_ids_tensor) {
        rt.set_data(tensor.id, ids.to_vec());
    }

    rt.execute(&cx.dyn_map);
    (
        rt.get_f32(initial_embeds_out.id).to_vec(),
        rt.get_f32(tts_pad_embed_out.id).to_vec(),
    )
}

pub fn decode_speech(
    frames: &[Vec<u32>],
    speech_config: &SpeechDecoderConfig,
    weights: &HashMap<String, Vec<f32>>,
) -> Vec<f32> {
    if frames.is_empty() {
        return Vec::new();
    }

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

    let mut cx = Graph::new();
    let decoder = SpeechDecoder::new(&mut cx, speech_config.clone());
    let code_tensors: Vec<GraphTensor> = (0..num_codebooks)
        .map(|_| cx.tensor((1, num_frames)).as_dtype(DType::Int))
        .collect();
    let audio = decoder.decode_codes(code_tensors.clone());
    let audio_out = audio.output();

    cx.build_search_space::<NativeRuntime>();
    let mut rt = cx.search(NativeRuntime::default(), 1);
    load_weights_from_map(&mut rt, &cx, weights);

    for (i, tensor) in code_tensors.iter().enumerate() {
        rt.set_data(tensor.id, codes_by_codebook[i].clone());
    }

    rt.execute(&cx.dyn_map);
    rt.get_f32(audio_out.id).to_vec()
}
