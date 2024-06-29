# Set access tokens
huggingface-cli login --token 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

# Specify the cache directory
# export TRANSFORMERS_CACHE=~/READMESum/hf-pretrained-checkpoints/
# export TRANSFORMERS_CACHE=/media/aiotlab3/27934be5-a11a-44ba-8b28-750d135bc3b3/RISE/Loc/hf-pretrained-checkpoints/
export TRANSFORMERS_CACHE=../../hf-pretrained-checkpoints/

# Specify the device
export CUDA_VISIBLE_DEVICES="0,1"

# Disable tokenizers parallelism
export TOKENIZERS_PARALLELISM=false

## Fine-tune bart-base
# python bart-base_readme_summarization.py

## Fine-tune bart-large
# python bart-large_readme_summarization.py

## Fine-tune bart-large-xsum
# python bart-large-xsum_readme_summarization.py

## Fine-tune t5-small
# python t5-small_readme_summarization.py

## Fine-tune t5-base
# python t5-base_readme_summarization.py

## Fine-tune t5-large
# python t5-large_readme_summarization.py

## Fine-tune pegasus-large
# python pegasus-large_readme_summarization.py

## Fine-tune pegasus-x-base
# python pegasus-x-base_readme_summarization.py

## Fine-tune pegasus-x-large
# python pegasus-x-large_readme_summarization.py

## Fine-tune pegasus-xsum
# python pegasus-xsum_readme_summarization.py

## Fine-tune pegasus-xsum
# python pegasus-xsum_readme_summarization.py

## Fine-tune Llama2
python llama2-7b_readme_summarization.py
