# Improved-README-Summarization

This is the implementation of the paper: "SALA: Summarization of GitHub artifacts with Pre-trained and Large Language Models"

## Introduction
- An approach to the summarization of textual artifacts in GitHub
- An empirical evaluation of real-world datasets, and comparison with SOTA baselines
- The SALA tool together with the data curated in this work is published online to enable future research

## Environment: 
- Require: python=3.9\
Other packages you can find at requirements.txt

## Large language models:
- Llama2-7b 
- Llama2-7b-chat

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
- `results/` contains the results of our test set based on ROUGE scores, includes ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-LSUM and the predictions
### Summarization
- `summarization/` contains all the implementation of data processing, training, and testing pre-trained models.\
