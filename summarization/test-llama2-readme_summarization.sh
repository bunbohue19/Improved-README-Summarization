# Specify the cache directory
# export TRANSFORMERS_CACHE=~/READMESum/hf-pretrained-checkpoints/
# export TRANSFORMERS_CACHE=/media/aiotlab3/27934be5-a11a-44ba-8b28-750d135bc3b3/RISE/Loc/hf-pretrained-checkpoints/
export TRANSFORMERS_CACHE=../../hf-pretrained-checkpoints/

# Specify the device
export CUDA_VISIBLE_DEVICES="0"

# Set access tokens
huggingface-cli login --token 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

# Disable tokenizers parallelism
# export TOKENIZERS_PARALLELISM=false

# Test
python test-llama2-readme_summarization.py --is_chat=${1} --shots=${2}