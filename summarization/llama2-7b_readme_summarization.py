import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import evaluate
from datasets import Dataset, DatasetDict
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, set_peft_model_state_dict, get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer

def formatting_func(example):
    return f"""<s>### Instruction:
        Summarize the following README contents.\
        Your answer should be based on the provided README contents only.
        
        ### README contents:
        {example["readme"]}
        
        ### Summary:
        {example["description"]}</s>"""

if __name__ == '__main__':
    device = torch.device("cuda:0")
    train_df = pd.read_csv('../dataset/train.csv', usecols=['readme', 'description'])
    val_df = pd.read_csv('../dataset/validation.csv', usecols=['readme', 'description'])
    
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
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    rouge = evaluate.load("rouge")
    
    training_args = TrainingArguments(
        output_dir="llama2-7b-chat_readme_summarization",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        optim="paged_adamw_32bit",
        gradient_accumulation_steps=4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_total_limit=3,
        num_train_epochs=4,
        load_best_model_at_end=True,
        fp16=False,
        report_to="wandb",
        push_to_hub=True
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=readme_dataset["train"],
        eval_dataset=readme_dataset["val"],
        peft_config=peft_config,
        max_seq_length=1024,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=formatting_func
    )
    
    trainer.train()
    
    trainer.push_to_hub()
