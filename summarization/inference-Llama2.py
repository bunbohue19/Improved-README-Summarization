from transformers import BitsAndBytesConfig, LlamaTokenizer, LlamaForCausalLM
import torch

def post_processing(outputs : str):
    outputs = outputs.splitlines()
    return outputs[1:]

if __name__ == '__main__':
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    device = "cuda:0"
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", quantization_config=bnb_config, device_map=device)
    
    while True: 
        prompt = input("Enter your prompt: ")
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # Generate
        generate_ids = model.generate(inputs)
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        for line in post_processing(outputs):
            print(line)
