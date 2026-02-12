use crate::code_predictor::{CodePredictorConfig, CodePredictorModel};
use crate::model::{TalkerConfig, TalkerModel};
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
    prompt_len: usize,
    num_frames: usize,
    weights: &HashMap<String, Vec<f32>>,
) -> Vec<Vec<u32>> {
    let hidden = talker_config.hidden;
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
        let pred_codes: Vec<u32> = code_outs
            .iter()
            .map(|code| rt.get_f32(code.id)[0] as u32)
            .collect();

        let mut frame_codes = vec![code_0_val];
        frame_codes.extend(pred_codes);
        all_frames.push(frame_codes);

        let codec_sum_data = rt.get_f32(codec_sum_out.id);
        embeddings.extend_from_slice(codec_sum_data);
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
        let role_embeds = self.talker.embed_text(inputs.role_text_ids);
        let overlay_text = self.talker.embed_text(inputs.overlay_text_ids);
        let transition_text = self.talker.embed_text(inputs.transition_text_id);
        let trailing_text = self.talker.embed_text(inputs.trailing_text_ids);
        let tts_pad_embed = self.talker.embed_text(inputs.tts_pad_text_id);

        // 2. Codec path: embed through codec_embedding
        let overlay_codec = self.talker.embed_codec(inputs.overlay_codec_ids);
        let transition_codec = self.talker.embed_codec(inputs.transition_codec_id);

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
}
