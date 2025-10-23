import re
import numpy as np
import pandas as pd
import os

def count_each_interact(splitted) :
    #This function count what part appeared after another one and generate the structure that will be used to calculate the transition matrix
    
    dict_translation = {"α" : "BEGINNING",
                        "β" : "INTRO",
                        "γ" : "COUPLET",
                        "ε" : "REFRAIN",
                        "ζ" : "PONT",
                        "η" : "OUTRO",
                        "θ" : "END"
                    }
    
    for i in range(len(splitted)) :
#        print(splitted[i])
        splitted[i] = splitted[i]+" θ"
    
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
        next = dict_translation[i.split(" ")[1]]

        if actual == "α" :  
            beginn[next] = dict_glob[i]

        elif actual == beginning_sections[0] :
            intro[next] = dict_glob[i]
        elif actual == beginning_sections[1] :
            couplet[next] = dict_glob[i]
        elif actual == beginning_sections[3] :
            pont[next] = dict_glob[i]
        elif actual == beginning_sections[2] :
            refrain[next] = dict_glob[i]
        elif actual == beginning_sections[-1] :
            outro[next] = dict_glob[i]

    return list_dicti

def generate_a_song_structure(matrix) :
    #This function use a transition matrix and generate a song structure out of it
    
    song_struct = [0]

    index = [i for i, p in enumerate(matrix[0]) if p>0] #Get the index of part
    proba = [p for p in matrix[0] if p > 0] #Get the proba of transition

    #Select the first part of the song
    cumsum = np.cumsum(proba)
    r = np.random.rand()
    idx = np.searchsorted(cumsum, r)
    selected_value = index[idx] 
    song_struct.append(selected_value)

    end = False

    while end == False : #Select the next parts

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
path = os.path.join(script_dir, "..", "Corpus", "Cleaning")
list_files = os.listdir(path)

#Regroup the songs, necessary when multiple artists

global_corpus = []
for i in list_files :      
    if "clean_corpus_" in i  :
        file_path = os.path.join(path, i)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            corpus = f.read()
            global_corpus.extend([corpus])
        
        good_corpus = re.sub(r"(true|false)\n", "\n", corpus)
        os.remove(file_path)
        with open(file_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(good_corpus)

one_corpus = "".join(global_corpus)

beginning_sections =["β", "γ", "ε", "ζ", "η"]
end_sections =["/β", "/γ", "/ε", "/ζ", "/η"]

#Remove potential duplicates
one_corpus = "θ\n".join(set(one_corpus.split("θ\n")))

#Final clean before usable corpus for training
regroup_corpus = re.sub("(true|false)\n", "", one_corpus)
regroup_corpus = re.sub("α\n\n", "α\n", regroup_corpus)
regroup_corpus = regroup_corpus.lower()

#When a part is empty, I will delete it because I generate the structure from the Markov and not the model
#If I let it, it introduce noise into the model 

for i in beginning_sections :
    regroup_corpus = re.sub(f"{i}\n/{i}\n", f"", regroup_corpus) 
regroup_corpus = re.sub(r"α\nθ\n",r"", regroup_corpus, flags=re.DOTALL) #Delete the songs where now there is only the token indicating the beginning and the end

save_path = os.path.join(path, f"regroup_clean_corpus.txt")
with open(save_path, "w", encoding="utf-8") as f :
    f.write(regroup_corpus)

## Removing featuring 
solo_corpus = re.sub(r"αTrue.*?θ\n\n", " ", one_corpus, flags=re.DOTALL)

## Removing the songs with only one part identified
all_parts = [(m.group(1), m.start(), m.end()) for m in re.finditer(r"(/β|/γ|/ε|/ζ|/η|α|θ|β|γ|ε|ζ|η)", solo_corpus)]

for i in range(len(all_parts)-3) :
    nn = len(all_parts)-3 -i
    if all_parts[nn-2][0] == "α" :
        if all_parts[nn-1][0] in end_sections :
            if all_parts[nn][0] == "θ" :
                part_1_corp = solo_corpus[:all_parts[nn-2][1]]
                part_2_corp = solo_corpus[all_parts[nn][2]:]
                solo_corpus = part_1_corp+part_2_corp

###This code will create the matrix of transition of the Markov Chain
##Prep for Markov Chain

parts = re.findall(r"(/β|/γ|/ε|/ζ|/η|α|θ|β|γ|ε|ζ|η)", solo_corpus) #Identify each part
regrouped = " ".join(parts) #re.findall gave back a list, we need a full text to perform next modif

#Delete end_sections
for i in end_sections : 
    regrouped = regrouped.replace(" "+i,"")

splitted = [i.strip() for i in regrouped.split("θ") if i.strip()]
all_count = count_each_interact(splitted)

#Remove impossible transitions, in case the pre-processing doesn't work properly or has let passed an error
all_count[0]["OUTRO"] = 0
all_count[0]["END"] = 0
all_count[1]["END"] = 0
all_count[4]["INTRO"] = 0
all_count[5]["INTRO"] = 0

## Creation of matrix of transition
matrix = []
for i in all_count :
    inter_matrix = []
    for j in i :
        inter_matrix.append(i[j]/sum(i.values()))
    matrix.append(inter_matrix)

order = ["<BEGINNING>","<INTRO>", "<COUPLET>", "<PONT>", "<REFRAIN>", "<OUTRO>", "<END>"]

#print("\nGeneration of 3 structures :")
#for run in range(3) :
#    generate_a_song_structure(matrix)

save_path = os.path.join(script_dir, f"transition_matrix.csv")
pd.concat((pd.DataFrame(matrix),pd.DataFrame(order).transpose())).to_csv(save_path, index=False)

#print(f"\nTransition_matrix saved as transition_matrix.csv")