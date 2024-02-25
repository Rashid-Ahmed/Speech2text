# Speech2text

This is a Github repository to finetune the Whisper Speech2text model on different datasets. 

## Usage

Poetry is needed to run this project. 
To change the dataset and model among other things, go to python/ner/config.py 
Steps to follow
1. Clone the project
2. Go to the python folder and do poetry lock -> poetry install
3. Run cli.py train <output_directory> <hugging_face_token> (You need to sign up on huggingface go to settings -> Access Tokens -> copy token to get the token needed to access the dataset used here)


## Model

Whisper model from OpenAI is used in this project. 

## Dataset

The Speech2text model is finetuned on the Mozilla Common Voice dataset.

