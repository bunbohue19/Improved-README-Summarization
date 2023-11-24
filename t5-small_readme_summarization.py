import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import evaluate
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

def preprocess_function(examples):
    for doc in examples["readme"]:
        if doc is None:
            continue
        inputs = [prefix + doc for doc in examples["readme"]]

    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["description"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_df = pd.read_csv('../dataset/train.csv', usecols=['readme', 'description'])
    val_df = pd.read_csv('../dataset/validation.csv', usecols=['readme', 'description'])
    readme_dataset = DatasetDict({
        'train' : Dataset.from_pandas(train_df),
        'val'  : Dataset.from_pandas(val_df)
    })
    
    checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    prefix = "summarize: "
    tokenized_readme = readme_dataset.map(function=preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
    rouge = evaluate.load("rouge")
    
    training_args = Seq2SeqTrainingArguments(
    output_dir="t5-small_readme_summarization",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True,
    # push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_readme["train"],
    eval_dataset=tokenized_readme["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
    
trainer.train()

# trainer.push_to_hub()
