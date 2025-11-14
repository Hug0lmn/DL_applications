import pickle
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
path = script_dir / "Corpora"
list_files = [file for file in path.iterdir()]

for i in list_files :
    if "final_corpus" in i  :
        file = path / i
        corpora = file.read_text(encoding='utf-8', errors="replace")
        break

characs = sorted(set(corpora))
print("Length corpus : ", len(corpora))
char2int = {ch: i for i, ch in enumerate(characs)}
print("Here the entire char encoding : \n",char2int)

encode_corpora = [char2int[i] for i in corpora]

np.save(f"{path}/Char_level/corpora_encoded.npy",encode_corpora)
with open(f"{path}/Char_level/encoding_map.pkl", "wb") as f:
    pickle.dump(char2int, f)