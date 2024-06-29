import re
import pandas as pd
import torch
from markdown import markdown
from bs4 import BeautifulSoup
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

def generate_training_prompt(readme: str, summary: str) -> str:
    return f"""### Instruction: Summarize the following README contents with LESS THAN 30 words. Your answer should be based on the provided README contents only.

    ### README contents:
    {readme.strip()}

    ### Summary:
    {summary}
    """.strip()

# Function to remove tags
def format_entry(md_data) :
    html = markdown(md_data)
    # parse html content
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.findAll('a', href=True):
        a.decompose()
    for data in soup(['style', 'script', 'img', 'pre', 'code']):
        # Remove tags
        data.decompose()
    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)

def process_description(s: str) -> str:
    if s.endswith('.'):
        s = s[:-1]
        s = re.sub(r"\. ", ", ", s)
    return s + '.'

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"#+", " ", text)
    return re.sub(r"\^[^ ]+", "", text)

def generate_sample_with_prompt(entry):
    readme = entry['readme']
    readme = clean_text(readme)
    description = process_description(entry['description'])
    return {
        "formatted_readme": readme,
        "summary": description,
        "prompt_text": generate_training_prompt(readme, description),
    }

def process_dataset(data: Dataset):
    return data.shuffle(seed=42).map(generate_sample_with_prompt).remove_columns(
        [
            "readme",
            "description",
        ]
    )

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    # MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

    # You need to change this parameter according to your real path.
    OUTPUT_DIR = "./llama2-7b-chat_readme_summarization"
    train_csv_file = '../dataset/train.csv'
    val_csv_file = '../dataset/validation.csv'

    # For access LLama2 pre-trained model in HuggingFace
    AUTH_TOKEN='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'
    
    # Read data
    train_df = pd.read_csv(train_csv_file, usecols=['readme', 'description'])
    val_df = pd.read_csv(val_csv_file, usecols=['readme', 'description'])

    for i, readme in enumerate(train_df['readme']):
        train_df.at[i, 'readme'] = format_entry(readme)

    for i, readme in enumerate(val_df['readme']):
        val_df.at[i, 'readme'] = format_entry(readme)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    processed_train_dataset = process_dataset(train_dataset)
    processed_val_dataset = process_dataset(val_dataset)
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        use_auth_token=AUTH_TOKEN, 
        truncation=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        use_auth_token=AUTH_TOKEN
    )
    
    lora_r = 16
    lora_alpha = 64
    lora_dropout = 0.1
    lora_target_modules = [
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "down_proj",
        "gate_proj",
        "v_proj",
    ]


    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_arguments = TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        logging_steps=100,
        learning_rate=1e-4,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=4,
        warmup_ratio=0.05,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        group_by_length=True,
        output_dir=OUTPUT_DIR,
        report_to="wandb",
        save_safetensors=True,
        lr_scheduler_type="cosine",
        seed=42,
        push_to_hub=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_val_dataset,
        peft_config=peft_config,
        dataset_text_field="prompt_text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
    )
    
    trainer.train()
    trainer.save_model()
    trainer.push_to_hub()
