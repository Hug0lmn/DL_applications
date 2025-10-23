import pickle
import numpy as np
import os

script_dir = os.path.dirname(__file__)
path = os.path.join(script_dir, "Cleaning")
list_files = os.listdir(path)

for i in list_files :
    if "regroup_clean_corpus" in i  :
        with open(os.path.join(path,i), "r", encoding="utf-8", errors="replace") as f:
            corpora = f.read()
        break

characs = sorted(set(corpora))
print("Length corpus : ", len(corpora))
char2int = {ch: i for i, ch in enumerate(characs)}
print("Here the entire char encoding : \n",char2int)

encode_corpora = [char2int[i] for i in corpora]

new_path = os.path.join(script_dir, "Encoding_RNN_LSTM")

np.save(f"{new_path}/Char_level/corpora_encoded.npy",encode_corpora)
with open(f"{new_path}/Char_level/encoding_map.pkl", "wb") as f:
    pickle.dump(char2int, f)