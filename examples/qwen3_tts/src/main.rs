use qwen3_tts::code_predictor::CodePredictorConfig;
use qwen3_tts::model::TalkerConfig;
use qwen3_tts::pipeline::{
    assemble_prompt_embeds, decode_speech, generate_frames, CODEC_BOS_ID, CODEC_LANG_ENGLISH,
    CODEC_PAD_ID, CODEC_THINK_BOS_ID, CODEC_THINK_EOS_ID, CODEC_THINK_ID, TTS_BOS_TOKEN_ID,
    TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID,
};
use qwen3_tts::speech_decoder::SpeechDecoderConfig;
use qwen3_tts::weight_loader::load_safetensors_to_map;
use std::path::Path;

const SAMPLE_RATE: u32 = 24_000;
const DEFAULT_MAX_FRAMES: usize = 500;

// Pre-tokenized: "<|im_start|>assistant\nHello world.<|im_end|>\n<|im_start|>assistant\n"
const ASSISTANT_IDS: &[i32] = &[
    151644, 77091, 198, 9707, 1879, 13, 151645, 198, 151644, 77091, 198,
];

// Pre-tokenized: "<|im_start|>user\nA young female voice, bright and cheerful.<|im_end|>\n"
const INSTRUCT_IDS: &[i32] = &[
    151644, 872, 198, 32, 3908, 8778, 7743, 11, 9906, 323, 70314, 13, 151645, 198,
];

fn main() {
    let model_dir = std::env::var("QWEN3_TTS_MODEL_DIR")
        .unwrap_or_else(|_| "./Qwen3-TTS-12Hz-1.7B-VoiceDesign".to_string());
    let tokenizer_dir = std::env::var("QWEN3_TTS_TOKENIZER_DIR")
        .unwrap_or_else(|_| "./Qwen3-TTS-Tokenizer-12Hz".to_string());
    let main_model_path = Path::new(&model_dir).join("model.safetensors");
    let decoder_model_path = Path::new(&tokenizer_dir).join("model.safetensors");
    let output_path = Path::new("output.wav");

    eprintln!("Loading main model weights...");
    let main_weights =
        load_safetensors_to_map(&main_model_path).expect("Failed to load main model");
    eprintln!("  Loaded {} tensors", main_weights.len());

    eprintln!("Loading speech decoder weights...");
    let decoder_weights =
        load_safetensors_to_map(&decoder_model_path).expect("Failed to load decoder");
    eprintln!("  Loaded {} tensors", decoder_weights.len());

    let talker_config = TalkerConfig::default();
    let predictor_config = CodePredictorConfig::default();
    let speech_config = SpeechDecoderConfig::default();

    let role_ids: Vec<f32> = ASSISTANT_IDS[..3].iter().map(|&x| x as f32).collect();
    let content_ids: Vec<f32> = ASSISTANT_IDS[3..ASSISTANT_IDS.len() - 5]
        .iter()
        .map(|&x| x as f32)
        .collect();
    let instruct_ids: Vec<f32> = INSTRUCT_IDS.iter().map(|&x| x as f32).collect();

    let overlay_text_ids: Vec<f32> = vec![
        TTS_PAD_TOKEN_ID as f32,
        TTS_PAD_TOKEN_ID as f32,
        TTS_PAD_TOKEN_ID as f32,
        TTS_PAD_TOKEN_ID as f32,
        TTS_BOS_TOKEN_ID as f32,
    ];
    let overlay_codec_ids: Vec<f32> = vec![
        CODEC_THINK_ID as f32,
        CODEC_THINK_BOS_ID as f32,
        CODEC_LANG_ENGLISH as f32,
        CODEC_THINK_EOS_ID as f32,
        CODEC_PAD_ID as f32,
    ];

    let content_len = content_ids.len();
    let content_codec_ids: Vec<f32> = vec![CODEC_PAD_ID as f32; content_len + 1];

    eprintln!("Assembling prompt...");
    let (initial_embeds, tts_pad_embed) = assemble_prompt_embeds(
        &talker_config,
        &predictor_config,
        &role_ids,
        &content_ids,
        &overlay_text_ids,
        &overlay_codec_ids,
        &content_codec_ids,
        TTS_EOS_TOKEN_ID as f32,
        TTS_PAD_TOKEN_ID as f32,
        CODEC_BOS_ID as f32,
        Some(&instruct_ids),
        &main_weights,
    );
    let prompt_len = initial_embeds.len() / talker_config.hidden;
    eprintln!("  Prompt length: {} positions", prompt_len);

    let max_frames: usize = std::env::var("QWEN3_TTS_MAX_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_FRAMES);
    eprintln!("Generating frames (max {max_frames})...");
    let frames = generate_frames(
        &talker_config,
        &predictor_config,
        &initial_embeds,
        &tts_pad_embed,
        prompt_len,
        max_frames,
        &main_weights,
    );
    eprintln!("  Generated {} frames", frames.len());

    if frames.is_empty() {
        eprintln!("No frames generated (immediate EOS). Exiting.");
        return;
    }

    eprintln!("Decoding speech...");
    let audio = decode_speech(&frames, &speech_config, &decoder_weights);
    eprintln!(
        "  Audio samples: {} ({:.2}s at {}Hz)",
        audio.len(),
        audio.len() as f64 / SAMPLE_RATE as f64,
        SAMPLE_RATE
    );

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer =
        hound::WavWriter::create(output_path, spec).expect("Failed to create WAV file");
    for &sample in &audio {
        writer
            .write_sample(sample.clamp(-1.0, 1.0))
            .expect("Failed to write sample");
    }
    writer.finalize().expect("Failed to finalize WAV");

    eprintln!("Written to {}", output_path.display());
}
