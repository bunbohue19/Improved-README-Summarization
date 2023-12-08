import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import evaluate
import re
from datasets import Dataset, DatasetDict
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
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

def formatting_func(sample):
    return f"""### Instruction:
        Summarize the following README contents with LESS THAN 50 words. Your answer should be based on the provided README contents only.
        ### README contents:
        {sample["readme"]}
        ### Summary:
        """
    for word in prompt:      
        if word is None:      
            continue            
        inputs = [word for word in prompt] 
        model_inputs = tokenizer(inputs, max_length=4096, truncation=True)    
        labels = tokenizer(text_target=sample["description"], max_length=128, truncation=True)                
        model_inputs["labels"] = labels["input_ids"]                                                                       
    return model_inputs 

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
    
    readme_dataset = DatasetDict({
        'train' : Dataset.from_pandas(train_df),
        'val'  : Dataset.from_pandas(val_df)
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
        device_map=device
    )
    
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
        max_seq_length=4096,
        tokenizer=tokenizer,
        formatting_func=formatting_func
    )
    
    trainer.train()
    
    trainer.push_to_hub()
