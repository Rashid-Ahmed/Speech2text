import torch
from transformers import Seq2SeqTrainingArguments
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from pathlib import Path
from speech2text.config import Config


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def get_training_args(output_path: Path, config: Config):
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_path),  # name on the HF Hub
        per_device_train_batch_size=config.training.batch_size_per_device,
        gradient_accumulation_steps=1,
        learning_rate=config.training.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=config.training.warmup_steps,
        max_steps=4000,
        fp16=False,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True
    )

    return training_args
