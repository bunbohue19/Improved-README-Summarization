import argparse
import re
import torch
import numpy as np
import pandas as pd
import evaluate
import sys
import os
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models, InputExample, losses, util
from sentence_transformers import evaluation
from tqdm import tqdm
from markdown import markdown
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel

"""
    Return item and drop from frame. Raise KeyError if not found.
"""
def pop(df : pd.DataFrame, idx : int):
    readme = df['readme'][idx]
    description = df['description'][idx]
    result = {'readme' : readme, 'description' : description}
    df.at[idx, 'readme'] = np.nan
    df.at[idx, 'description'] = np.nan
    return result

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Few-shots prompting
def generate_testing_prompt(readme, shots):
    if len(shots) == 0:
        return f"""### Instruction: You are a helpful assistant. You need to summarize the following README contents. A good answer should be based on the provided README contents only and LESS THAN 20 words.

        ### README contents:
        {readme.strip()}

        ### Summary:
        """.strip()
    else:
        prompt = """### Instruction: You are a helpful assistant. You need to summarize the following README contents. A good answer should be based on the provided README contents only and LESS THAN 20 words.
        ### For examples:
        """
        
        for i in range(len(shots)):
            prompt += f""" 
            ### README contents: 
            {shots[i]['readme'].strip()}
            
            ### Summary:
            {shots[i]['description'].strip()}            
            """

        prompt += f"""
        ### README contents:
        {readme.strip()}

        ### Summary:
        """.strip()
        return prompt
        
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

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"#+", " ", text)
    return re.sub(r"\^[^ ]+", "", text)

def process_description(s: str) -> str:
    if s.endswith('.'):
        s = s[:-1]
        s = re.sub(r"\. ", ", ", s)
    return s + '.'

def test(args):
    
    num_of_shots = int(f"{args.shots}")
    
    print(num_of_shots)  
    
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    OUTPUT_DIR = "./zero-shot-prompting-llama-2-7b_readsum_29-6-2024"
    
    # Read data
    test_df = pd.read_csv('../dataset/updated_test.csv', usecols=['readme', 'description'])
    
    for i, readme in enumerate(test_df['readme']):
        test_df.at[i, 'readme'] = format_entry(readme)
    
    shots = []
    if num_of_shots == 0:
        pass
    elif num_of_shots == 1:
        shots.append(pop(test_df, 8))
    elif num_of_shots == 2:
        shots.append(pop(test_df, 8))
        shots.append(pop(test_df, 10))
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        truncation=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    
    samples = []
    for readme, description in zip(test_df['readme'], test_df['description']):
        readme = clean_text(str(readme))
        description = process_description(str(description))

        sample = {
            "readme": readme,
            "description": description,
            "prompt": generate_testing_prompt(readme, shots),
        }
        samples.append(sample)
    results_df = pd.DataFrame(samples)
    
    # Load metric
    rouge = evaluate.load("rouge")
    
    print("Testing...")
    ### ROUGE scores
    ### Get the score per sample
    idx = 1
    results, predictions = [], []
    for prompt, description in zip(results_df['prompt'], results_df['description']):
        inputs = tokenizer(prompt, max_length=4096, truncation=True, return_tensors="pt").to(DEVICE)
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.0001)
            
        prediction = tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)
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
    
    ### SIDE scores
    ### Get the score per sample
    checkPointFolder = "/mnt/4TData/vuquang/code-summarization-metric/Models/baseline/103080" 
    tokenizer = AutoTokenizer.from_pretrained(checkPointFolder)
    model = AutoModel.from_pretrained(checkPointFolder).to(DEVICE)
    
    idx = 1
    SIDE_results = []
    for reference_summary, prediction_summary in zip(test_df['readme'], test_df['description']):
        pair = [str(reference_summary), str(prediction_summary)]
        encoded_input = tokenizer(pair, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        sim_result = round(util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1]).item() * 100, 2)
    
        print('Sample: ', idx)
        print('SIDE: ', sim_result)
        print('\n')
        idx += 1
        SIDE_results.append(sim_result)

    SIDE_results_df = pd.DataFrame(data=SIDE_results, columns=['SIDE'])

    for res in SIDE_results_df:
        SIDE_results_df.loc[-1] = [res]
        SIDE_results_df.index += 1
    SIDE_results_df.index -= 1
    full_results_df = pd.concat([test_df, predictions_df, r1_df, r2_df, rl_df, rlsum_df, SIDE_results_df], axis=1)
    full_results_df = full_results_df.dropna()
    full_results_df.to_csv(f'../results/updated_result-{num_of_shots}-shot_llama2-7b_29-06.csv', escapechar=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=str, help="Enter the number of shots!\n")
    args = parser.parse_args()
    test(args)