import argparse 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if __name__ == '__main__': 
    parser.add_argument("--checkpoint", type=str, help="Specify model checkpoint name:")         
    args = parser.parse_args()      
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint) 
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)