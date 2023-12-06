# Specify the cache directory   
# export TRANSFORMERS_CACHE=~/READMESum/hf-pretrained-checkpoints/      
# export TRANSFORMERS_CACHE=/media/aiotlab3/27934be5-a11a-44ba-8b28-750d135bc3b3/RISE/Loc/Improved-README-Summarization/hf-pretrained-checkpoints/
# export TRANSFORMERS_CACHE=../../hf-pretrained-checkpoints/

# Set access tokens                                                                                                  
huggingface-cli login --token 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

## Disable tokenizers parallelism                                                                                    
export TOKENIZERS_PARALLELISM=false

# Inference
python inference-Llama2-7b-chat.py
