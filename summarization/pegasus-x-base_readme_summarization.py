import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import evaluate
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, PegasusXForConditionalGeneration, PegasusXConfig, PegasusXModel, Seq2SeqTrainingArguments, Seq2SeqTrainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, set_peft_model_state_dict, get_peft_model, LoraConfig, TaskType

def preprocess_function(examples):
    for doc in examples["readme"]:
        if doc is None:
            continue
        inputs = [prefix + doc for doc in examples["readme"]]
        # inputs = [doc for doc in examples["readme"]]  
    # inputs = examples["readme"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding='max_length', return_tensors="pt")
    # model_inputs = model_inputs.to(device)
    
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
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_df = pd.read_csv('../dataset/train.csv', usecols=['readme', 'description'])
    val_df = pd.read_csv('../dataset/validation.csv', usecols=['readme', 'description'])
    readme_dataset = DatasetDict({
        'train' : Dataset.from_pandas(train_df),
        'val'  : Dataset.from_pandas(val_df)
    })

    # bnb_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_use_double_quant=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_compute_dtype=torch.bfloat16
    # )
    
    checkpoint = "google/pegasus-x-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, truncation=True)

    # config = PegasusXConfig(max_position_embeddings=1024,
    #                              encoder_layers=12,
    #                              encoder_ffn_dim=3072,
    #                              encoder_attention_heads=12,
    #                              decoder_layers=12,
    #                              decoder_ffn_dim=3072,
    #                              decoder_attention_heads=12,
    #                              d_model=768)

    # model = PegasusXModel(config)
    model = PegasusXForConditionalGeneration.from_pretrained(checkpoint)
    config = model.config
    config.update({"max_length":1024})
    config.update({"max_position_embeddings":1024})
    model.config = config
    # model = PegasusXForConditionalGeneration.from_pretrained(checkpoint, quantization_config=bnb_config)
    # model = model.to(device)
    
    # peft_config = LoraConfig(
    #    task_type=TaskType.SEQ_2_SEQ_LM,
    #    inference_mode=False,
    #    r=8,
    #    lora_alpha=32,
    #    lora_dropout=0.1,
    #    target_modules=[
    #        "q_proj",
    #        "k_proj",
    #        "v_proj",
    #        "o_proj"
    #    ]
    # )
    
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    prefix = "summarize: "
    tokenized_readme = readme_dataset.map(function=preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="pt", padding='max_length')
    rouge = evaluate.load("rouge")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="pegasus-x-base_readme_summarization",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        load_best_model_at_end=True,
        fp16=False,
        report_to="wandb",
        # push_to_hub=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_readme["train"],
        eval_dataset=tokenized_readme["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # trainer.push_to_hub()
