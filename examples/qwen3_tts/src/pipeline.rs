use crate::code_predictor::{CodePredictorConfig, CodePredictorModel};
use crate::model::{TalkerConfig, TalkerModel};
use luminal::{graph::Graph, prelude::GraphTensor};

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

pub struct StreamingPromptInputs {
    pub role_text_ids: GraphTensor,       // [1, 3] - role prefix text tokens
    pub overlay_text_ids: GraphTensor,    // [1, overlay_len] - tts_pad repeated + tts_bos
    pub overlay_codec_ids: GraphTensor,   // [1, overlay_len] - codec control tokens
    pub transition_text_id: GraphTensor,  // [1, 1] - first real content text token
    pub transition_codec_id: GraphTensor, // [1, 1] - codec BOS token
    pub trailing_text_ids: GraphTensor,   // [1, trailing_len] - rest of text + tts_eos
    pub tts_pad_text_id: GraphTensor,     // [1, 1] - tts_pad for generation loop
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
        let initial = role_embeds.concat_along(overlay, 1).concat_along(transition, 1);

        StreamingPromptOutputs {
            initial_embeds: initial,
            trailing_text_hidden: trailing_text,
            tts_pad_embed,
        }
    }
}
