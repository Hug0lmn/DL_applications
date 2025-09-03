import re
import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="", help="Name of the artist")
args = parser.parse_args()
name = args.name

def count_each_interact(splitted) :
    #This function count what part appeared after another one and generate the structure that will be used to calculate the transition matrix
    
    for i in range(len(splitted)) :
        splitted[i] = splitted[i]+" END"

    good_parts = (" ".join(splitted)).split(" ")

    dict_glob = {}
    for i in range(len(good_parts)-1) :
        try : 
            dict_glob[f"{good_parts[i]} {good_parts[i+1]}"] += 1
        except :
            dict_glob[f"{good_parts[i]} {good_parts[i+1]}"] = 1

    beginn = {}
    intro = {}
    couplet = {}
    pont = {}
    refrain = {}
    outro = {}

    order = ["BEGINNING","INTRO", "COUPLET", "PONT", "REFRAIN", "OUTRO", "END"]
    list_dicti = [beginn, intro, couplet, pont, refrain, outro]
    for part in list_dicti :
        for j in order :
            part[j] = 0    
    
    for i in dict_glob.keys() :
        actual = i.split(" ")[0]
        next = i.split(" ")[1]

        if actual == "BEGINNING" :  
            beginn[next] = dict_glob[i]

        elif actual == "INTRO" :
            intro[next] = dict_glob[i]

        elif actual == "COUPLET" :
            couplet[next] = dict_glob[i]
        elif actual == "PONT" :
            pont[next] = dict_glob[i]
        elif actual == "REFRAIN" :
            refrain[next] = dict_glob[i]
        elif actual == "OUTRO" :
            outro[next] = dict_glob[i]

    return list_dicti

def generate_a_song_structure(matrix) :
    #This function use a transition matrix and generate a song structure out of it
    
    song_struct = [0]

    index = [i for i, p in enumerate(matrix[0]) if p>0]
    proba = [p for p in matrix[0] if p > 0]

    cumsum = np.cumsum(proba)
    r = np.random.rand()

    idx = np.searchsorted(cumsum, r)
    selected_value = index[idx] 
    song_struct.append(selected_value)

    end = False

    while end == False :

        index = [i for i, p in enumerate(matrix[selected_value]) if p>0]
        proba = [p for p in matrix[selected_value] if p > 0]

        cumsum = np.cumsum(proba)
        r = np.random.rand()

        idx = np.searchsorted(cumsum, r)
        selected_value = index[idx]
        song_struct.append(selected_value)

        if selected_value == 6 :
            end = True

    print([order[i] for i in song_struct])
    return



script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "..", "Corpus", f"corpus_RNN_{name}.txt")
###This code will create the matrix of transition of the Markov Chain

with open(file_path, "r", encoding="utf-8", errors="replace") as f:
    corpus = f.read()

## Removing featuring 
corpus_solo = re.sub(r"<BEGINNING>True.*?<END>\n\n", " ", corpus, flags=re.DOTALL)

## Removing the songs with only one part identified
all_parts = [(m.group(1), m.start(), m.end()) for m in re.finditer(r"<(\w+)>", corpus_solo)]

for i in range(len(all_parts)-3) :
    nn = len(all_parts)-3 -i
    if all_parts[nn-2][0] == "BEGINNING" :
        if all_parts[nn-1][0] == "END_SECTION" :
            if all_parts[nn][0] == "END" :
                part_1_corp = corpus_solo[:all_parts[nn-2][1]]
                part_2_corp = corpus_solo[all_parts[nn][2]:]
                corpus_solo = part_1_corp+part_2_corp

##Count the number of parts in songs
#If a song has more than 10 parts indicates the beginning and end of the song in the corpus
all_parts = [(m.group(1), m.start(), m.end()) for m in re.finditer(r"<(\w+)>", corpus_solo)]

count = 0
beg = 0
endi =0

for i in all_parts :
    if i[0] == "BEGINNING" :
        count = 0
        beg = i[1]
    elif i[0] == "END_SECTION" :
        continue
    else : 
        if i[0] == "END" :
            count+=1
            endi = i[2]
            if count >=10 :
                print("Nb parts :",count, beg, endi)
        else : 
            count+=1

##Prep for Markov Chain

parts = re.findall(r"<(\w+)>", corpus_solo) #Identify each part
regrouped = " ".join(parts) #re.findall gave back a list, we need a full text to perform next modif
regrouped = re.sub("END_SECTION ", "", regrouped) 
splitted = regrouped.split("END")

splitted = [i.strip() for i in splitted[:-1]]

all_count = count_each_interact(splitted)

## Creation of matrix of transition
matrix = []
for i in all_count :
    inter_matrix = []
    for j in i :
        inter_matrix.append(i[j]/sum(i.values()))
    matrix.append(inter_matrix)

order = ["<BEGINNING>","<INTRO>", "<COUPLET>", "<PONT>", "<REFRAIN>", "<OUTRO>", "<END>"]

print("\nGeneration of 3 structures :")
for run in range(3) :
    generate_a_song_structure(matrix)

print(f"\nTransition_matrix {name}:\n",pd.concat((pd.DataFrame(matrix),pd.DataFrame(order).transpose())))

save_path = os.path.join(script_dir, f"transition_matrix_{name}.csv")
pd.concat((pd.DataFrame(matrix),pd.DataFrame(order).transpose())).to_csv(save_path, index=False)

print(f"\nTransition_matrix saved as transition_matrix_{name}.csv")