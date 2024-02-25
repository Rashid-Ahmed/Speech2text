import logging
from pathlib import Path
from transformers import Seq2SeqTrainer
from speech2text.config import Config
from speech2text.data.processing import load_data, process_dataset
from speech2text.training.initializers import initialize_model, initialize_processor
from speech2text.data.utils import DataCollatorSpeechSeq2SeqWithPadding, get_training_args
from speech2text.training.metrics import compute_metrics
from huggingface_hub import login

logger = logging.getLogger(__name__)



def train(output_path: Path, token:str, config: Config):
    login(token = token)
    train_dataset, validation_dataset = load_data(config)
    auto_config, processor = initialize_processor(config)
    train_dataset = process_dataset(train_dataset, processor, config)
    validation_dataset = process_dataset(validation_dataset, processor, config)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    model = initialize_model(config)
    training_args = get_training_args(output_path, config)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor
    )
    trainer.train()
