import argparse
import torch
import pandas as pd
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test(args):
    checkpoint = str(args.checkpoint)
    device = f"cuda:{args.device}"
    
    # Load test set
    test_df = pd.read_csv('../dataset/test.csv', usecols=['readme', 'description'])
    
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    if "pegasus" not in checkpoint:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, attention_type="original_full")
    
    model = model.to(device)
        
    # Load metric
    rouge = evaluate.load("rouge")

    # Evaluate in test set
    predictions, references = [], []

    prefix = "summarize: "
    
    print("Testing...")
    for sample in test_df['readme']:
        inputs = tokenizer(prefix + sample, return_tensors="pt", truncation=True).input_ids.to(device)
        outputs = model.generate(inputs, max_new_tokens=128, do_sample=False)
        predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    references = [sample for sample in test_df['description']]
    print('Calculating...')
    results = rouge.compute(predictions=predictions, references=references)
    
    print('ROUGE-1 : ', round(results['rouge1'] * 100, 2),
          '\nROUGE-2 : ', round(results['rouge2'] * 100, 2),
          '\nROUGE-L : ', round(results['rougeL'] * 100, 2),
          '\nROUGE-LSUM : ', round(results['rougeLsum'] * 100, 2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Specify model checkpoint name:\n\
                        1.bart-base\n\
                        2.bart-large\n\
                        3.bart-large-xsum\n\
                        4.t5-small\n\
                        5.t5-base\n\
                        6.t5-large\n\
                        7.pegasus-large\n\
                        8.pegasus-x-base\n\
                        9.pegasus-x-large\n\
                        10.pegasus-xsum")
    
    parser.add_argument("--device", type=str, help="Specify the device number")
    args = parser.parse_args()
    test(args)
