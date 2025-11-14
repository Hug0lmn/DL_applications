#This code will generate the fasttext embedding and the subword tokenizer from the corpus and 
# then prepare the context windows corresponding to the specific corpus

from pathlib import Path

script_dir = Path(__file__).resolve()
file_path = script_dir / "Corpora"

corpus = file_path.read_text(encoding='utf-8', errors="replace")

# -- Generate subword tokenizer -- #

import numpy as np
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

tokenizer = ByteLevelBPETokenizer()

# Train it
tokenizer.train([f"{file_path}/final_corpus.txt"], vocab_size=4000, min_frequency=2, 
    special_tokens=["<PAD>","α","β","γ","ε","ζ","η","θ","/β","/γ","/ε","/ζ","/η"])

tokenizer.save(f"{file_path}/Subword/tokenizer.json")

# Encode corpus
tokenizer = PreTrainedTokenizerFast( #Can't use directly the tokenizer, first need to save it and then load with specific function
    tokenizer_file=f"{file_path}/Subword/tokenizer.json", 
    pad_token = "<PAD>"
)

np.save(f"{file_path}/Subword/encoded_corpus.npy",tokenizer.encode(corpus))

print(f"Corpus went from {len(corpus)} characters to {len(tokenizer.tokenize(corpus))} subwords")

# -- Generate fasttext embedding -- #

import fasttext
import re

#dim 100 because if more I get memory problem during training due to the fact that I store all the possible cut in order to have faster training
#but this can be changed (and will be changed I think) to remove limitation

# Regarding the choice of cbow, I have tested the two and after further examination of the results with get_nearest_neighbors 
# I opted for it (because it captures not only semantic neighbors but also deeper meaning : example police ~= porcs ~= politique) 
word2vec = fasttext.train_unsupervised(f"{file_path}/final_corpus.txt", model = "cbow", epoch = 30, dim = 100, lr=0.05, ws=8)
word2vec.save_model(f"{file_path}/fasttext_embedding")

# -- Context window -- #

# This part will create and store the context window obtained from the word2vec embedding
# Idea : characters contains only one meaning that is the order in a word, so we can add other meaning by using the embedding of the previous words
# Problem : using a char lvl embedding make this harder because the model can see half a word, just one charac of a word...
# How : Store, for each word in the corpus, it's embedding and also the nb indicating the beginning and end of the word in the corpus
# and for each word in the corpus, construct a vector containing the 3 previous word embedding but depending on the seq length adapt : 
# the first word in a specific sequence doesn't have previous words so it results in a vector of 0 and so on 

# There also can be a choice, to reduce memory usage or maybe to have another type of info, to average the three previous words

#fast_emb = fasttext.load_model(f"{file_path}/fasttext_corpus")

not_words = list(re.finditer("[ |\n|.|,|!|:|;|?]", corpus))
start = [0] + [val.end() for val in not_words[:-1]]
end = np.array([val.start() for val in not_words])

all_ = np.array([word2vec.get_word_vector(corpus[start[i]:end[i]]) for i in range(len(end))])

np.save(f"{file_path}/Char_level/starts.npy", start)
np.save(f"{file_path}/Char_level/ends.npy", end)
np.save(f"{file_path}/Char_level/word_matrix.npy", all_) #I can't store directly the 3 previous embed for each words because it's too heavy and it's crashing python