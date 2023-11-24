# Script to load 
# sh test.sh <CHECKPOINT_NAME>                                                                                                                                                                      
echo "Enter the name of the checkpoint:" 
echo "1.bart-base" 
echo "2.bart-large"
echo "3.t5-small"                                                                                                   
echo "4.t5-base"                                                                                                     
echo "5.t5-large"                                                                                                    
echo "6.bigbird-pegasus-large-arxiv"                                                                                 
echo "7.bigbird-pegasus-large-bigpatent"                                                                             
echo "8.bigbird-pegasus-large-pubmed"                                                                                
echo "9.pegasus-large"                                                                                               
echo "10.pegasus-xsum"                                                                                                 
read -p "Enter the name of the checkpoint: " checkpoint_name            
CHECKPOINT="bunbohue/"${checkpoint_name}"_readme_summarization"

# Specify the cache directory                                                                                        
export TRANSFORMERS_CACHE=~/READMESum/hf-pretrained-checkpoints/
export TRANSFORMERS_CACHE=/media/aiotlab3/27934be5-a11a-44ba-8b28-750d135bc3b3/RISE/Loc/Improved-README-Summarization/hf-pretrained-checkpoints/

# Set access tokens                                                                                                  
huggingface-cli login --token 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                                                     
# Load                                                                                                               
python load-checkpoint.py --checkpoint ${CHECKPOINT}