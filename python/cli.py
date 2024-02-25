#!/usr/bin/ven python
import typer
from pathlib import Path

import speech2text
from speech2text.config import Config

app = typer.Typer()


@app.command()
def train(output_path: Path = typer.Argument(None, help="Directory to save model in "),
          token: str = typer.Argument(None, help="Your Huggingface account token ")):
    config = Config()
    speech2text.train(output_path, token, config)


@app.command()
def evaluate(model_path: Path = typer.Argument(None, help="Path of trained model"),
             output_path: Path = typer.Argument(None, help="Output directory of evaluation results")):
    pass


if __name__ == "__main__":
    app()
