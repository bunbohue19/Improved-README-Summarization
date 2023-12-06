from transformers import BitsAndBytesConfig, LlamaTokenizer, LlamaForCausalLM
import torch
import pandas as pd
import evaluate

if __name__ == '__main__':
    test_df = pd.read_csv('../dataset/test.csv', usecols=['readme', 'description'])

    # Load metric
    rouge = evaluate.load("rouge")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    checkpoint = "meta-llama/Llama-2-7b-chat-hf"
    device = torch.device("cuda:0")
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    model = LlamaForCausalLM.from_pretrained(checkpoint, quantization_config=bnb_config, device_map={"": 0})
    
    print("Inferencing...")
    
    # Inference in test set
    predictions, references = [], []
    
    ### Get the score per sample
    idx = 1
    results = []
    for readme, description in zip(test_df['readme'], test_df['description']): 
        prompt = f"""### Instruction:
        Summarize the following README contents with LESS THAN OR EQUAL {len(description)} words\
        Your answer should be based on the provided README contents only.
        
        ### README contents:
        {readme}
        
        ### Summary:
        """
        
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # Generate
        generate_ids = model.generate(inputs)
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        prediction = outputs
        reference = description
        
        result = rouge.compute(predictions=[prediction], references=[reference])
        r1 = round(result['rouge1'] * 100, 2)
        r2 = round(result['rouge2'] * 100, 2)
        rl = round(result['rougeL'] * 100, 2)
        rlsum = round(result['rougeLsum'] * 100, 2)
        
        print('Sample: ', idx)
        print('ROUGE-1 : ', r1,
              '\nROUGE-2 : ', r2,
              '\nROUGE-L : ', rl,
              '\nROUGE-LSUM : ', rlsum)
        print('\n')
        idx += 1
        results.append(result)
        predictions.append(prediction)
        
    results_df = pd.DataFrame(columns=['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-LSUM'])
    predictions_df = pd.DataFrame(columns=['prediction'])

    for result in results:
        results_df.loc[-1] = [result['rouge1'], result['rouge2'], result['rougeL'], result['rougeLsum']]
        results_df.index += 1

    for prediction in predictions:
        predictions_df.loc[-1] = [prediction]
        predictions_df.index += 1

    full_results_df = pd.concat([test_df, predictions_df, results_df], axis=1)
    full_results_df.dropna()
    full_results_df.to_csv(f'../results/result_inference_Llama-2-7b-chat.csv')
