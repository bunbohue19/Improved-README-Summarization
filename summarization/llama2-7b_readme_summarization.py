import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import evaluate
import re
from datasets import Dataset, DatasetDict
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, set_peft_model_state_dict, get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer

def preprocessing_readme(readme):
    readme = re.sub(r"http\S+", "", readme)
    readme = re.sub(r"@[^\s]+", "", readme)
    readme = re.sub(r"\s+", " ", readme)
    readme = re.sub(r"#+", " ", readme)
    readme = re.sub(r"\^[^ ]+", "", readme)
    return readme.strip()

def preprocessing_description(description):
    if description.endswith('.'):
        description = description[:-1]
    description = re.sub(r"\. ", ", ", description)
    description = description + '.'
    return description.strip()

def prompts(df):
    prompts = []
    for readme, description in zip(df['readme'], df['description']):
        prompts.append(f"""### Instruction:
            Summarize the following README contents with LESS THAN 50 words. Your answer should be based on the provided README contents only.
            ### README contents:
            {readme}
            ### Summary:
            {description}
            """)
    return prompts

def formatting_func(sample):        
    return {
        "readme": sample["readme"],
        "description": sample["description"],
        "prompt": sample["prompt"]
    }

if __name__ == '__main__':
    device = torch.device("cuda:0")
    train_df = pd.read_csv('../dataset/train.csv', usecols=['readme', 'description'])
    val_df = pd.read_csv('../dataset/validation.csv', usecols=['readme', 'description'])
    
    for i, sample in enumerate(train_df['readme']):
        train_df.at[i, 'readme'] = preprocessing_readme(sample)
    
    for i, sample in enumerate(val_df['readme']):
        val_df.at[i, 'readme'] = preprocessing_readme(sample)
    
    for i, sample in enumerate(train_df['description']):
        train_df.at[i, 'description'] = preprocessing_description(sample)
    
    for i, sample in enumerate(val_df['description']):
        val_df.at[i, 'description'] = preprocessing_description(sample)
    
    train_prompts_df = pd.DataFrame(data=prompts(train_df), columns=['prompt'])
    val_prompts_df = pd.DataFrame(data=prompts(val_df), columns=['prompt'])

    for prompt in train_prompts_df:
        train_prompts_df.loc[-1] = [prompt]
        train_prompts_df.index += 1
    train_prompts_df.index -= 1

    for prompt in val_prompts_df:
        val_prompts_df.loc[-1] = [prompt]
        val_prompts_df.index += 1
    val_prompts_df.index -= 1
    
    new_train_df = pd.concat([train_df, train_prompts_df], axis=1).dropna()
    new_val_df = pd.concat([val_df, val_prompts_df], axis=1).dropna()
    
    readme_dataset = DatasetDict({
        'train' : Dataset.from_pandas(new_train_df),
        'val'  : Dataset.from_pandas(new_val_df)
    })
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    checkpoint = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = LlamaForCausalLM.from_pretrained(
        checkpoint, 
        quantization_config=bnb_config,
        use_cache=False,
        device_map={"":0}
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "q_proj",
            "up_proj",
            "o_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
            "v_proj"
        ]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    tokenized_readme = readme_dataset.map(function=formatting_func, batched=True)
    rouge = evaluate.load("rouge")
    
    training_args = TrainingArguments(
        output_dir="llama2-7b_readme_summarization",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        group_by_length=True,
        learning_rate=1e-4,
        optim="paged_adamw_32bit",
        gradient_accumulation_steps=4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_total_limit=3,
        num_train_epochs=4,
        load_best_model_at_end=True,
        fp16=True,
        report_to="wandb",
        push_to_hub=True
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_readme["train"],
        eval_dataset=tokenized_readme["val"],
        peft_config=peft_config,
        dataset_text_field="prompt",
        max_seq_length=2048,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    trainer.push_to_hub()
