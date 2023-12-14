# Specify the cache directory
export TRANSFORMERS_CACHE=../../hf-pretrained-checkpoints/

# Specify the device
export CUDA_VISIBLE_DEVICES="0"

# Disable tokenizers parallelism
# export TOKENIZERS_PARALLELISM=false

# Test
python five-shots-test-llama2.py