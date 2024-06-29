# Set access tokens
huggingface-cli login --token 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
wandb login 1183ae2e25d9d913eb2e8c1dc43b7cdba6c18910

# Specify the cache directory
export HF_HOME=../../hf-pretrained-checkpoints/

# Specify the device
export CUDA_VISIBLE_DEVICES="0"

# Disable tokenizers parallelism
export TOKENIZERS_PARALLELISM=false