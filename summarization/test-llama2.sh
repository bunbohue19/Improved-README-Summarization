# Specify the cache directory
# export TRANSFORMERS_CACHE=../../hf-pretrained-checkpoints/

# Specify the device
export CUDA_VISIBLE_DEVICES="0"

# Set access tokens
huggingface-cli login --token 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

# Disable tokenizers parallelism
# export TOKENIZERS_PARALLELISM=false

# Test
python test-llama2.py --shots=${1}