# Script to test
# sh test.sh <CHECKPOINT_NAME> <DEVICE>

CHECKPOINT="bunbohue/"${1}"_readme_summarization"

# Specify the device
# export CUDA_VISIBLE_DEVICES=${2}

# Specify the cache directory
# export TRANSFORMERS_CACHE=~/READMESum/hf-pretrained-checkpoints/
export TRANSFORMERS_CACHE=/media/aiotlab3/27934be5-a11a-44ba-8b28-750d135bc3b3/RISE/Loc/hf-pretrained-checkpoints/
# export TRANSFORMERS_CACHE=../../hf-pretrained-checkpoints/

# Set access tokens
huggingface-cli login --token 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

## Disable tokenizers parallelism
export TOKENIZERS_PARALLELISM=false

# Test 
python test.py --checkpoint=$CHECKPOINT --device=${2}
