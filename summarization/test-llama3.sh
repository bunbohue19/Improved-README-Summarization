# Set access tokens
huggingface-cli login --token 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

# Disable tokenizers parallelism
# export TOKENIZERS_PARALLELISM=false

# Test
python test-llama3.py --shots=${1}