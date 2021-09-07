# Train the Tokenizer for my German BERT
# Christoph Windheuser
# September 7 2021
#

from datasets import load_dataset
from pathlib import Path

# from tqdm.auto import tqdm
# import os
# import pickle

# import torch
from tokenizers import ByteLevelBPETokenizer
# from transformers import RobertaTokenizer
# from transformers import RobertaConfig
# from transformers import RobertaForMaskedLM
# from transformers import AdamW
# from transformers import pipeline

print ("Start loading dataset")
dataset = load_dataset('oscar', 'unshuffled_deduplicated_de')
print ("Loading dataset done")

print ("Start loading file names")
paths = [str(x) for x in Path('data/train/oscar_de').glob('**/*.txt')]
print (paths[:3])
print ("Number of files: %d" % len(paths))

print ("Start training the Tokenizer")
tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, vocab_size=30_522, min_frequency=2,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])
print ("Training Tokenizer done")

print ("Saving Tokenizer")
tokenizer.save_model('GermanBERT')
print ("Saving Tokenizer done")

