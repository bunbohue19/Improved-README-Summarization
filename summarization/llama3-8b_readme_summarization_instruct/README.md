---
license: other
library_name: peft
tags:
- trl
- sft
- generated_from_trainer
base_model: meta-llama/Meta-Llama-3-8B
model-index:
- name: llama3-8b_readme_summarization
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama3-8b_readme_summarization

This model is a fine-tuned version of [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 1.6496

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 2
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 4
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step  | Validation Loss |
|:-------------:|:------:|:-----:|:---------------:|
| 1.7397        | 0.9998 | 2915  | 1.7288          |
| 1.3617        | 2.0    | 5831  | 1.5983          |
| 0.8781        | 2.9998 | 8746  | 1.5681          |
| 0.6176        | 3.9993 | 11660 | 1.6496          |


### Framework versions

- PEFT 0.10.0
- Transformers 4.40.1
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1