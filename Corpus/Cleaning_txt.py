import re
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="", help="Name of the artist")
args = parser.parse_args()
name = args.name

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, f"corpus_RNN_{name}.txt")

with open(file_path, "r", encoding="utf-8", errors="replace") as f:
    corpus = f.read()

#First we remove the indication of featuring
corpus = re.sub(r">(True|False)\n", ">\n", corpus)

#Then we remove the unecessary spaces
corpus = corpus.strip()
corpus_splitted = re.split(r"\n", corpus, flags=re.DOTALL)
new_list = []
for i in range(len(corpus_splitted)) :
    stripped = corpus_splitted[i].strip()
    if stripped != "" :
#        if stripped == "<END>" :
#            new_list.extend([stripped+"<>"])
#        else :
        new_list.extend([stripped+"\n"])
corpus = "".join(new_list)

#Beginning of the corpus cleaning
print("Nb of individual char",len(set(corpus)))

dict_change = {}
substitutions = [
    (r"<.*>\n<END_SECTION>\n", "", 0), #Some parts are empty of lyrics because they were used just for training on the structure
    (r"<BEGINNING>\n<END>\n", "", 0), #No songs
    (r"<END_SECTION>", "§", 0), #End_Section can be replaced by a specific character that isn't in the corpus such as §
#    (r"<BEGINNING>\n", "", 0),
#    (r"<END>\n", "", 0),
    (r"\n([a-z]\w+)\n", r"\1\n", 0), #If lowercase, we assume that the sentence wasn't finished
    (r"\n([A-Z]\w+)\n", r"\n\1", 0), #If uppercase, we assume we are at the beginning of the sentence
    (r"\(.*?\)", "", re.DOTALL), # Characters : ()
    (r"\*", "", re.DOTALL), # Character : *
    (r"«.*?»", "", 0), # Characters : « »
    ("”", '"', 0), # Characters : “ ”
    ("“", '"', 0),
    ("…", "...", 0), # Character : …
    ("‘", "'", 0), # Characters : ‘ ’
    ("’", "'", 0),
    ("е", "e", 0),   # Cyrillic e → Latin e
    ("#", "", 0), # Character : \#
    ('"', "", 0), # Character : "
    ("\(", "", 0), # Character : (
    ("»|«", "", 0), # Character : »|«
    ("\x80", "", 0),
    ("\x99", "", 0),
    ("\ufeff", "", 0),
    ("ʿ", "", 0), # Character : ʿ
    ("·", "", 0), # Character : ·
    (r"\n(\n)", "", 0), #Sometime the text is in « and the previous will just replace it with only a "" but the line are kept and on some songs there is huge part of this
    (r"><", ">\n<", 0), #If two parts are fuzed together, unfuze them
    (r"(>)(\w+)", r"\1\n\2",0),
#    (r"\w+(§)", "\n§", 0) #I can't explain it exactly but some § can be found near a word and not after a \n
    (r"(?:(?<!\n)§)", "\n§", 0)
]

# Apply them
for pattern, repl, flags in substitutions:
    corpus = re.sub(pattern, repl, corpus, flags=flags)

# Special case: remove apostrophes around spaces
def remove_special(match):
    return match.group(0).replace("'", "")

corpus = re.sub(r"'\s|\s+'", remove_special, corpus, flags=re.DOTALL)

print("New nb of char:", len(set(corpus)))

save_path = os.path.join(script_dir, f"clean_corpus_{name}.txt")
with open(save_path, "w", encoding="utf-8") as f :
    f.write(corpus)