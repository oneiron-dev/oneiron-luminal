use crate::code_predictor::{CodePredictorConfig, CodePredictorModel};
use crate::model::{TalkerConfig, TalkerModel};
use luminal::graph::Graph;

// Codec token IDs from HuggingFace config.json
pub const CODEC_PAD_ID: i32 = 2148;
pub const CODEC_BOS_ID: i32 = 2149;
pub const CODEC_EOS_ID: i32 = 2150;
pub const CODEC_THINK_ID: i32 = 2154;
pub const CODEC_NOTHINK_ID: i32 = 2155;
pub const CODEC_THINK_BOS_ID: i32 = 2156;
pub const CODEC_THINK_EOS_ID: i32 = 2157;

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
}
