import pickle
import numpy as np
from pathlib import Path
import fasttext
import re
import argparse
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast


parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, help="Indicate type of corpus needed", required=True)

args = parser.parse_args()
type = args.type

script_dir = Path(__file__).resolve().parent
path = script_dir / "Corpora"
corpus_path = path / "final_corpus.txt"

corpus = corpus_path.read_text(encoding='utf-8', errors="replace")

### ---- Character level processing ----- ###

if type == "char" :

    #dim 100 because if more I get memory problem during training due to the fact that I store all the possible cut in order to have faster training
    #but this can be changed (and will be changed I think) to remove limitation

    # Regarding the choice of cbow, I have tested the two and after further examination of the results with get_nearest_neighbors 
    # I opted for it (because it captures not only semantic neighbors but also deeper meaning : example police ~= porcs ~= politique) 
    word2vec = fasttext.train_unsupervised(f"{path}/final_corpus.txt", model = "cbow", epoch = 30, dim = 100, lr=0.05, ws=8)
    word2vec.save_model(f"{path}/Char_level/fasttext_embedding")

    # -- Context window -- #

    # This part will create and store the context window obtained from the word2vec embedding
    # Idea : characters contains only one meaning that is the order in a word, so we can add other meaning by using the embedding of the previous words
    # Problem : using a char lvl embedding make this harder because the model can see half a word, just one charac of a word...
    # How : Store, for each word in the corpus, it's embedding and also the nb indicating the beginning and end of the word in the corpus
    # and for each word in the corpus, construct a vector containing the 3 previous word embedding but depending on the seq length adapt : 
    # the first word in a specific sequence doesn't have previous words so it results in a vector of 0 and so on 

    # There also can be a choice, to reduce memory usage or maybe to have another type of info, to average the three previous words

    not_words = list(re.finditer("[ |\n|.|,|!|:|;|?]", corpus))
    start = [0] + [val.end() for val in not_words[:-1]]
    end = np.array([val.start() for val in not_words])

    all_ = np.array([word2vec.get_word_vector(corpus[start[i]:end[i]]) for i in range(len(end))])

    np.save(f"{path}/Char_level/starts.npy", start)
    np.save(f"{path}/Char_level/ends.npy", end)
    np.save(f"{path}/Char_level/word_matrix.npy", all_) #I can't store directly the 3 previous embed for each words because it's too heavy and it's crashing python

    ## Encode corpus

    characs = sorted(set(corpus))
    char2int = {ch: i for i, ch in enumerate(characs)}
    print("Here the entire char encoding : \n",char2int)

    encode_corpora = [char2int[i] for i in corpus]

    np.save(f"{path}/Char_level/corpora_encoded.npy",encode_corpora)
    with open(f"{path}/Char_level/encoding_map.pkl", "wb") as f:
        pickle.dump(char2int, f)

### ---- Subword level processing ----- ###

elif type == "subword" :

    tokenizer = ByteLevelBPETokenizer()

    # Train it
    tokenizer.train([f"{path}/final_corpus.txt"], vocab_size=4000, min_frequency=2, 
        special_tokens=["<PAD>","α","β","γ","ε","ζ","η","θ","/β","/γ","/ε","/ζ","/η","/n"])

    tokenizer.save(f"{path}/Subword/tokenizer.json")

    # Encode corpus
    tokenizer = PreTrainedTokenizerFast( #Can't use directly the tokenizer, first need to save it and then load with specific function
        tokenizer_file=f"{path}/Subword/tokenizer.json", 
        pad_token = "<PAD>"
    )

    np.save(f"{path}/Subword/encoded_corpus.npy",tokenizer.encode(corpus))

    print(f"Corpus went from {len(corpus)} characters to {len(tokenizer.tokenize(corpus))} subwords")