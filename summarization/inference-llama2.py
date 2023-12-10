import re
import torch
import pandas as pd
import evaluate
from markdown import markdown
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def generate_testing_prompt(readme: str) -> str:
    return f"""### Instruction: Summarize the following README contents with LESS THAN 50 words. Your answer should be based on the provided README contents only.

    ### README contents:
    {readme.strip()}

    ### Summary:
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

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    AUTH_TOKEN='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'
    OUTPUT_DIR = "./llama2-7b_readme_summarization"
    test_csv_file = '../dataset/test.csv'
    
    # Read data
    test_df = pd.read_csv(test_csv_file, usecols=['readme', 'description'])

    for i, readme in enumerate(test_df['readme']):
        test_df.at[i, 'readme'] = format_entry(readme)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        use_auth_token=AUTH_TOKEN, 
        truncation=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        use_auth_token=AUTH_TOKEN
    )
    
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    
    samples = []
    for readme, description in zip(test_df['readme'], test_df['description']):
        readme = clean_text(readme)
        description = process_description(description)

        sample = {
            "readme": readme,
            "description": description,
            "prompt": generate_testing_prompt(readme),
        }
        samples.append(sample)
    results_df = pd.DataFrame(samples)
    
    # Load metric
    rouge = evaluate.load("rouge")
    
    print("Testing...")
    
    ### Get the score per sample
    idx = 1
    results, predictions = [], []
    for prompt, description in zip(results_df['prompt'], results_df['description']):
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
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
    
    full_results_df = pd.concat([results_df, predictions_df, r1_df, r2_df, rl_df, rlsum_df], axis=1)
    full_results_df = full_results_df.dropna()
    full_results_df.to_csv('../results/result_llama2-7b_readme_summarization.csv')