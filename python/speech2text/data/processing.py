from speech2text.config import Config
from datasets import load_dataset, Audio


def load_data(config: Config):
    dataset = load_dataset(config.data.dataset_name, config.data.dataset_language)
    train_dataset = dataset["train"].select_columns(["audio", "sentence"])
    validation_dataset = dataset["validation"].select_columns(["audio", "sentence"])

    return train_dataset, validation_dataset


def prepare_dataset(example, processor):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["sentence"],
    )

    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example


def is_audio_in_length_range(length, config):
    return length < config.data.max_audio_length


def process_dataset(dataset, processor, config):
    dataset = dataset.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

    dataset = dataset.map(
        prepare_dataset, num_proc=1,
        fn_kwargs={"processor": processor})

    dataset = dataset.filter(
        is_audio_in_length_range,
        input_columns=["input_length"], fn_kwargs={"config": config},
    )
    return dataset

