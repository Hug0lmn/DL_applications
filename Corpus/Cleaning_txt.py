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
#corpus = re.sub(r">(True|False)\n", ">\n", corpus)

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
#print("Nb of individual char",len(set(corpus)))

dict_change = {}
substitutions = [
#Data cleaning
    (r"\n([a-z]\w+)\n", r"\1\n", 0), #If lowercase, we assume that the sentence wasn't finished
    (r"\n([A-Z]\w+)\n", r"\n\1", 0), #If uppercase, we assume we are at the beginning of the sentence
#Specific characters
    (r"\(.*?\)", "", re.DOTALL), # Characters : ()
    (r"\*", "", re.DOTALL), # Character : *
    (r"«.*?»", "", 0), # Characters : « »
    ('–', '-', 0),
    ('ğ', 'g', 0),
    ('ć', 'c', 0),
    ('¿', '', 0),
    ('Ä', 'A', 0),
    ("Î", "I", 0),
    ("Ô", "O", 0),
    ("Ö", "O", 0),
    ("Ü", "U", 0),
    ("á", "a", 0),
    ("ã", "a", 0),
    ("ä", "a", 0),
    ("æ", "ae", 0),
    ("í", "i", 0),
    ("ñ", "n", 0),
    ("ü", "u", 0),
    ("ā", "a", 0),
    ("ō", "o", 0),
    ("Œ","OE",0),
    ("œ", "oe", 0),
    ("Ş","S",0),
    ("ş","s",0),
    ("—","-",0),
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
    ("\u205f", "", 0),
    ("\xa0", "", 0),
    ("ʿ", "", 0), # Character : ʿ
    ("·", "", 0), # Character : ·
    (r"\n(\n)", "", 0), #Sometime the text is in « and the previous will just replace it with only a "" but the line are kept and on some songs there is huge part of this
    (r"><", ">\n<", 0), #If two parts are fuzed together, unfuze them
    (r"(>)(\w+)", r"\1\n\2",0),
    (r"(?:(?<!\n)<END_\w+>)", "\n\g<0>", 0), #If the end of a part is preceded by a thing that is not \n then add a \n

#Replace indication part by specific charac (greek)
#η θ	ι	κ	λ	μ	ν	ξ	Ο	π	ρ	Σ	τ	υ	φ	χ ψ	Ω	
    (r"<BEGINNING_SONG>", "<α>", 0),
 
    (r"<INTRO>", "<β>", 0),
    (r"<END_INTRO>", "</β>", 0),
 
    (r"<COUPLET>", "<γ>", 0),
    (r"<END_COUPLET>", "</γ>", 0),
 
    (r"<REFRAIN>", "<ε>", 0),
    (r"<END_REFRAIN>", "</ε>", 0),

    (r"<PONT>", "<ζ>", 0),
    (r"<END_PONT>", "</ζ>", 0),
    
    (r"<OUTRO>", "<η>", 0),
    (r"<END_OUTRO>", "</η>", 0),

    (r"<END_SONG>", "<θ>", 0),
    (r"<α>\n<θ>\n", "", 0), #No lyrics inside a song

    (r"<.*>\n<END_\w+>\n", "", 0), #Some parts are empty of lyrics because they were used just for training on the structure
    (r"\n<END_>\n", "\n", 0), 
]

# Apply them
for pattern, repl, flags in substitutions:
    corpus = re.sub(pattern, repl, corpus, flags=flags)

# Special case: remove apostrophes around spaces
def remove_special(match):
    return match.group(0).replace("'", "")

corpus = re.sub(r"'\s|\s+'", remove_special, corpus, flags=re.DOTALL)

save_path = os.path.join(script_dir, f"Cleaning\clean_corpus_{name}.txt")
os.remove(file_path)
with open(save_path, "w", encoding="utf-8") as f :
    f.write(corpus)