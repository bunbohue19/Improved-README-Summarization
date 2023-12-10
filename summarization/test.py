import argparse
import torch
import pandas as pd
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test(args):
    checkpoint = f"{args.checkpoint}"
    device = f"cuda:{args.device}"
    
    print(f"You are using checkpoint: {checkpoint}")
    print(f"And device: {device}")
    
    # Load test set
    test_df = pd.read_csv('../dataset/test.csv', usecols=['readme', 'description'])
    
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model = model.to(device)
        
    # Load metric
    rouge = evaluate.load("rouge")

    # Evaluate in test set
    # predictions, references = [], []

    prefix = "summarize: "
    
    print("Testing...")
    
    ### Get the average score:
    # for sample in test_df['readme']:
    #     inputs = tokenizer(prefix + sample, return_tensors="pt", truncation=True).input_ids.to(device)
    #     outputs = model.generate(inputs, max_new_tokens=128, do_sample=False)
    #     predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # references = [sample for sample in test_df['description']]
    # print('Calculating...')
    # results = rouge.compute(predictions=predictions, references=references)
    
    # print('ROUGE-1 : ', round(results['rouge1'] * 100, 2),
    #       '\nROUGE-2 : ', round(results['rouge2'] * 100, 2),
    #       '\nROUGE-L : ', round(results['rougeL'] * 100, 2),
    #       '\nROUGE-LSUM : ', round(results['rougeLsum'] * 100, 2))

    ### Get the score per sample
    idx = 1
    results, predictions = [], []
    for readme, description in zip(test_df['readme'], test_df['description']):
        inputs = tokenizer(prefix + readme, return_tensors="pt", truncation=True).input_ids.to(device)
        outputs = model.generate(inputs, max_new_tokens=128, do_sample=False)
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reference = description
        result = rouge.compute(predictions=[prediction], references=[reference])
        result['rouge1'] = round(result['rouge1'] * 100, 2)
        result['rouge2'] = round(result['rouge2'] * 100, 2)
        result['rougeL'] = round(result['rougeL'] * 100, 2)
        result['rougeLsum'] = round(result['rougeLsum'] * 100, 2)
        
        print('Sample: ', idx)
        print('ROUGE-1 : ', result['rouge1'],
              '\nROUGE-2 : ', result['rouge2'],
              '\nROUGE-L : ', result['rougeL'],
              '\nROUGE-LSUM : ', result['rougeLsum'])
        print('\n')
        idx += 1
        results.append(result)
        predictions.append(prediction)

    r1s, r2s, rls, rlsums = [], [], [], []
    for result in results:
        r1s.append(result['rouge1'])
        r2s.append(result['rouge2'])
        rls.append(result['rougeL'])
        rlsums.append(result['rougeLsum'])

    r1_df = pd.DataFrame(data=r1s, columns=['ROUGE-1'])
    r2_df = pd.DataFrame(data=r2s, columns=['ROUGE-2'])
    rl_df = pd.DataFrame(data=rls, columns=['ROUGE-L'])
    rlsum_df = pd.DataFrame(data=rlsums, columns=['ROUGE-LSUM'])

    predictions_df = pd.DataFrame(data=predictions, columns=['prediction'])
    
    for r1 in r1_df:
        r1_df.loc[-1] = [r1]
        r1_df.index += 1
    r1_df.index -= 1

    for r2 in r2_df:
        r2_df.loc[-1] = [r1]
        r2_df.index += 1
    r2_df.index -= 1    
    
    for rl in rl_df:
        rl_df.loc[-1] = [r1]
        rl_df.index += 1
    rl_df.index -= 1

    for rlsum in rlsum_df:
        rlsum_df.loc[-1] = [r1]
        rlsum_df.index += 1
    rlsum_df.index -= 1

    for prediction in predictions_df:
        predictions_df.loc[-1] = [prediction]
        predictions_df.index += 1
    predictions_df.index -= 1
    
    full_results_df = pd.concat([test_df, predictions_df, r1_df, r2_df, rl_df, rlsum_df], axis=1)
    full_results_df = full_results_df.dropna()
    full_results_df.to_csv(f'../results/result_{checkpoint.replace("bunbohue/", "")}.csv')
    
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
                        10.pegasus-xsum\n\
                        11.llama2-7b")
    
    parser.add_argument("--device", type=str, help="Specify the device number")
    args = parser.parse_args()
    test(args)
