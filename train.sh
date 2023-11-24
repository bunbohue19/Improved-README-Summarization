 # Set access tokens
huggingface-cli login --token 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

# Specify the cache directory
export TRANSFORMERS_CACHE=/media/aiotlab3/27934be5-a11a-44ba-8b28-750d135bc3b3/RISE/Loc/Improved-README-Summarization/hf-pretrained-checkpoints/

# Specify the device
export CUDA_VISIBLE_DEVICES="0,1"

## Fine-tune bigbird-pegasus-large-arxiv
## Disable tokenizers parallelism
export TOKENIZERS_PARALLELISM=false
python bigbird-pegasus-large-arxiv_readme_summarization.py

## Fine-tune t5-large
# python t5-large_readme_summarization.py
