import pickle
import numpy as np
import os

script_dir = os.path.dirname(__file__)
path = os.path.join(script_dir, "Cleaning")
list_files = os.listdir(path)

new_corp = []
for i in list_files :
    if "clean_corpus" in i  :
        with open(f"{path}\{i}", "r", encoding="utf-8", errors="replace") as f:
            corpus = f.read()
            new_corp.extend(corpus)
corpora = "".join(new_corp)

characs = sorted(set(corpora))
char2int = {ch: i for i, ch in enumerate(characs)}

encode_corpora = [char2int[i] for i in corpora]

new_path = os.path.join(script_dir, "Encoding_RNN_LSTM")

np.save(f"{new_path}/corpora_encoded.npy",encode_corpora)
with open(f"{new_path}/encoding_map.pkl", "wb") as f:
    pickle.dump(char2int, f)