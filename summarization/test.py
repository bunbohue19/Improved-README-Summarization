import argparse
import pandas as pd
import evaluate
import torch
import sys
import os
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models, InputExample, losses, util
from sentence_transformers import evaluation
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def test(args):
    checkpoint = f"{args.checkpoint}"
    device = f"cuda:{args.device}"
    
    print(f"You are using checkpoint: {checkpoint}")
    print(f"And device: {device}")
    
    # Load test set
    test_df = pd.read_csv('../dataset/updated_test.csv', usecols=['readme', 'description'])
    
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

    ### ROUGE scores
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
    
    ### SIDE_baseline scores
    ### Get the score per sample
    checkPointFolder = "/media/aiotlab3/27934be5-a11a-44ba-8b28-750d135bc3b3/RISE/Loc/code-summarization-metric/Models/baseline/103080"
    tokenizer = AutoTokenizer.from_pretrained(checkPointFolder)
    model = AutoModel.from_pretrained(checkPointFolder).to(device)
    
    idx = 1
    SIDE_baseline_results = []
    for reference_summary, prediction_summary in zip(test_df['readme'], test_df['description']):
        pair = [str(reference_summary), str(prediction_summary)]
        encoded_input = tokenizer(pair, padding=True, truncation=True, return_tensors='pt').to(device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        sim_result = round(util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1]).item() * 100, 2)
    
        print('Sample: ', idx)
        print('SIDE_baseline: ', sim_result)
        print('\n')
        idx += 1
        SIDE_baseline_results.append(sim_result)

    SIDE_baseline_results_df = pd.DataFrame(data=SIDE_baseline_results, columns=['SIDE_baseline'])

    for res in SIDE_baseline_results:
        SIDE_baseline_results_df.loc[-1] = [res]
        SIDE_baseline_results_df.index += 1
    SIDE_baseline_results_df.index -= 1
    
    ### SIDE_hard_negatives scores
    ### Get the score per sample
    checkPointFolder = "/media/aiotlab3/27934be5-a11a-44ba-8b28-750d135bc3b3/RISE/Loc/code-summarization-metric/Models/hard-negatives/141205"
    tokenizer = AutoTokenizer.from_pretrained(checkPointFolder)
    model = AutoModel.from_pretrained(checkPointFolder).to(device)
    
    idx = 1
    SIDE_hard_negatives_results = []
    for reference_summary, prediction_summary in zip(test_df['readme'], test_df['description']):
        pair = [str(reference_summary), str(prediction_summary)]
        encoded_input = tokenizer(pair, padding=True, truncation=True, return_tensors='pt').to(device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        sim_result = round(util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1]).item() * 100, 2)
    
        print('Sample: ', idx)
        print('SIDE_hard_negatives: ', sim_result)
        print('\n')
        idx += 1
        SIDE_hard_negatives_results.append(sim_result)

    SIDE_hard_negatives_results_df = pd.DataFrame(data=SIDE_hard_negatives_results, columns=['SIDE_hard_negatives'])

    for res in SIDE_hard_negatives_results:
        SIDE_hard_negatives_results_df.loc[-1] = [res]
        SIDE_hard_negatives_results_df.index += 1
    SIDE_hard_negatives_results_df.index -= 1
    
    full_results_df = pd.concat([test_df, predictions_df, r1_df, r2_df, rl_df, rlsum_df, SIDE_baseline_results_df, SIDE_hard_negatives_results_df], axis=1)
    full_results_df = full_results_df.dropna()
    
    file_name = checkpoint.replace("bunbohue/", "")
    file_name = file_name.replace("readme_summarization", "")
    full_results_df.to_csv(f'../results/updated_result_{file_name}.csv')
    
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
                        10.pegasus-xsum\n")
    
    parser.add_argument("--device", type=str, help="Specify the device number")
    args = parser.parse_args()
    test(args)
