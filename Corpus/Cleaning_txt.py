import re
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="", help="Name of the artist")
args = parser.parse_args()
name = args.name

script_dir = Path(__file__).resolve().parent
file_path = script_dir / f"corpus_{name}.txt"
corpus = file_path.read_text(encoding='utf-8', errors="replace")
#script_dir = os.path.dirname(__file__)
#file_path = os.path.join(script_dir, f"corpus_{name}.txt")

#with open(file_path, "r", encoding="utf-8", errors="replace") as f:
#    corpus = f.read()

#First we remove the unecessary spaces
corpus = corpus.strip()
corpus_splitted = re.split(r"\n", corpus, flags=re.DOTALL)
new_list = []
for i in range(len(corpus_splitted)) :
    stripped = corpus_splitted[i].strip()
    if stripped != "" :
        new_list.extend([stripped+"\n"])
corpus = "".join(new_list)

#Beginning of the corpus cleaning

dict_change = {}
substitutions = [
#Specific characters
    (r"\(.*?\)", "", re.DOTALL), # Characters : ()
    (r"\*", "", re.DOTALL), # Character : *
    (r"«.*?»", "", 0), # Characters : « »
    ('–', '-', 0),
    ('ğ', 'g', 0),
    ('ć', 'c', 0),
    ('¿', '', 0),
    ("á", "a", 0),
    ("ã", "a", 0),
    ("ż", "z", 0),
    ("ä", "a", 0),
    ("æ", "ae", 0),
    ("í", "i", 0),
    ("ñ", "n", 0),
    ("ü", "u", 0),
    ("ā", "a", 0),
    ("ō", "o", 0),
    ("ó", "o", 0),
    ("ò", "o", 0),
    ("ö", "o", 0),
    ("ú", "u", 0),
    ("ū", "u", 0),
    ("č", "c", 0),
    ("&", "et", 0),
    ("°","",0),
    ("œ", "oe", 0),
    ("$","s",0),
    ("ş","s",0),
    ("$","s", 0),
    ("”", '"', 0), # Characters : “ ”
    ("“", '"', 0),
    ("…", "", 0), # Character : …
    ("‘", "'", 0), # Characters : ‘ ’
    ("’", "'", 0),
    ("´","'",0),
    ("′","'",0),
    ("×","x",0),
    ("{.*}", "", 0),
    ("е", "e", 0),   # Cyrillic e → Latin e
    ("#", "", 0), # Character : #
    ('"', "", 0), # Character : "
    (r"\(", "", 0), # Character : (
    (r"\)", "", 0),
    ("»|«", "", 0), # Character : »|«
    ("\x80", "", 0),
    ("\x90", "", 0),
    ("\x99", "", 0),
    ("\ufeff", "", 0),
    ("\u200a", "", 0),
    ("\u200b", "", 0),
    ("\u200c", "", 0),
    ("\u2028", "", 0),
    ("\u205f", "", 0),
    ("\xa0", "", 0),
    ("ʿ", "", 0), # Character : ʿ
    ("·", "", 0), # Character : ·
    ("s\ns", "", 0), #I don't know exactly why and I don't know exactly where to check but each individual corpus ends by s\ns
    (r"\n(\n)", "", 0), #Sometime the text is in « and the previous will just replace it with only a "" but the line are kept and on some songs there is huge part of this
    (r"><", ">\n<", 0), #If two parts are fuzed together, unfuze them
    (r"(>)(\w+)", r"\1\n\2",0),
    (r"(?:(?<!\n)<end_\w+>)", r"\n\g<0>", 0), #If the end of a part is preceded by a thing that is not \n then add a \n

#Replace indication part by specific charac (greek)
#η θ	ι	κ	λ	μ	ν	ξ	Ο	π	ρ	Σ	τ	υ	φ	χ ψ	Ω	
    (r"<beginning_song>", "α", 0),
 
    (r"<intro>", "β", 0),
    (r"<end_intro>", "/β", 0),
 
    (r"<couplet>", "γ", 0),
    (r"<end_couplet>", "/γ", 0),
 
    (r"<refrain>", "ε", 0),
    (r"<end_refrain>", "/ε", 0),

    (r"<pont>", "ζ", 0),
    (r"<end_pont>", "/ζ", 0),
    
    (r"<outro>", "η", 0),
    (r"<end_outro>", "/η", 0),

    (r"<end_song>", "θ", 0),
    (r"α\nθ\n", "", 0), #No lyrics inside a song

    (r"\n<end_>", "\n", 0), 
    (r"\n\n", "\n", 0), 
    ("—","-",0),
]

pre_substitu = [
    #Data cleaning
    (r"\n([a-z]\w+)\n", r"\1\n", 0), #If lowercase, we assume that the sentence wasn't finished
    (r"\n([A-Z]\w+)\n", r"\n\1", 0), #If uppercase, we assume we are at the beginning of the sentence]
]

for pattern, repl, flags in pre_substitu:
    corpus = re.sub(pattern, repl, corpus, flags=flags)

corpus = corpus.lower()

# Apply them
for pattern, repl, flags in substitutions:
    corpus = re.sub(pattern, repl, corpus, flags=flags)

# Special case: remove apostrophes around spaces
def remove_special(match):
    return match.group(0).replace("'", "")

corpus = re.sub(r"'\s|\s+'", remove_special, corpus, flags=re.DOTALL)

save_path = script_dir / "Corpora" / f"clean_corpus_{name}.txt"
file_path.unlink()
save_path.write_text(corpus, encoding='utf-8')
#save_path = os.path.join(script_dir, "Corpora", f"clean_corpus_{name}.txt")
#os.remove(file_path)
#with open(save_path, "w", encoding="utf-8") as f :
#    f.write(corpus)