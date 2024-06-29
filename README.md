# Improved-README-Summarization

This is the implementation of the paper: "SALA: Summarization of GitHub artifacts with Pre-trained and Large Language Models"

## Introduction
- An approach to the summarization of textual artifacts in GitHub
- An empirical evaluation of real-world datasets, and comparison with SOTA baselines
- The SALA tool together with the data curated in this work is published online to enable future research

## Environment: 
- Require: python=3.9\
Other packages you can find at requirements.txt

## Folder Walkthrough
### Dataset
`dataset/` contains all data for collected data
- Dataset is available at this [link](https://drive.google.com/file/d/1hNiaype-4XqsKq38pZ-qE7iUkJaorrI7/view?usp=sharing)\
Or you can find it in `dataset/` folder
### Models
- `models/` contains all the implementation of data processing, training, and testing pre-trained models.\
- Include some language models: T5, BART, PEGASUS and their variances
- Include two large language models: Llama2 and Llama2-chat
- All models are available at my repository in HuggingFace
### Results
- `results/` contains the results of our test set based on ROUGE scores, including ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-LSUM and the predictions
### Summarization
- `summarization/` contains all the implementation of data processing, training, and testing pre-trained models.\
- To fine-tune from a pre-trained LM model:
  - Open script file `train.sh`, uncomment the pre-trained model which you want to do
  - Modify some arguments if you want
  - Run command: `sh train.sh`
- To test a fine-tuned LM model:
  - Open script file `test.sh`, and modify some arguments if you want
  - Run command: `sh test.sh <CHECKPOINT_NAME> <DEVICE>`. Where:
    `<CHECKPOINT_NAME>` is one of the following: 
      - `bart-base`
      - `bart-large`
      - `bart-large-xsum`
      - `t5-small`
      - `t5-base`
      - `t5-large`
      - `pegasus-large`
      - `pegasus-x-base`
      - `pegasus-x-large`
      - `pegasus-xsum`
  
    `<DEVICE>` is the ID number of GPU (i.e. 0, 1, 2,...)
    
    For example: `sh test.sh bart-base 0`
- To test a fine-tuned LLM model (Llama2-7b):  
    
