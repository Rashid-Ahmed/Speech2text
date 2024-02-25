from speech2text.config import Config
from transformers import WhisperProcessor, WhisperConfig, WhisperForConditionalGeneration


def initialize_processor(config: Config):
    auto_config = WhisperConfig()
    processor = WhisperProcessor.from_pretrained(
        pretrained_model_name_or_path=config.model.processor_name, language=config.data.transcription_language, task="transcribe"
    )

    return auto_config, processor


def initialize_model(
        config: Config,
):
    model = WhisperForConditionalGeneration.from_pretrained(
        config.model.model_name,
    )
    return model
