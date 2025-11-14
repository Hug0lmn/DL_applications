import numpy as np
import torch

# --- FUNCTIONS PART --- #
# --- Streamlit PART --- #

def obtain_context_from_seed (seed, mapping, fasttext_emb) :

    whole_context = list([np.zeros((3,100),dtype=np.float32)])
    #Identify the words

    for i in range(1,len(seed)) :
        current_seed = seed[:i]
        word_limits = [j+1 for j,n in enumerate(current_seed) if n in ["\n"," ","!","."]]
        word_limits.insert(0,0)

        words_ = []
        for j in range(len(word_limits)-1) :
            segment = seed[word_limits[j]:word_limits[j+1]]
            words_.append(fasttext_emb.get_word_vector(segment))

        len_word = len(words_)
        if len_word == 0 :
            context = np.vstack([np.zeros((3,100),dtype=np.float32)])
        elif len_word < 3 :
            context = np.vstack([np.zeros((3-len_word,100),dtype=np.float32),words_[-len_word:]])
        else : 
            context = np.vstack(words_[-3:])

        whole_context.append(context)

    encoded = torch.tensor([mapping[i] for i in seed])
    whole_context = torch.from_numpy(np.array(whole_context))

    return encoded, whole_context

def generate_from_text(model, seed, length, temp, top, mapping, fasttext_emb, forbidden, hidden = False) :

    encoded, context = obtain_context_from_seed(seed, mapping, fasttext_emb)

    int2char = {i: ch for ch, i in mapping.items()}

    #Priming
    model.eval()
    with torch.no_grad():
        gen_chars = encoded.tolist()

        if hidden :
            hid = model.init_hidden(1)
            pred, hid = model(encoded.unsqueeze(0), context.unsqueeze(0), hid)
        
        else : 
            pred = model(encoded.unsqueeze(0), context.unsqueeze(0))
        
        gen_chars.append(sample_with_temp_topk(pred[:,-1][0],temperature = temp, top_k = top, forbidden_char=forbidden).item())

    #Generation 
    for _ in range(length) :

        #Identify the words in the encoded text
        word_limits = [j+1 for j,n in enumerate(gen_chars) if n in [0,1,2,9]]
        word_limits.insert(0,0)
    
        words_ = []
        for j in range(len(word_limits)-1) :
            segment = [i for i in gen_chars[word_limits[j]:word_limits[j+1]]]
            words_.append(fasttext_emb.get_word_vector("".join([int2char[j] for j in segment])))

        len_word = len(words_)
        if len_word < 3 :
            context = np.vstack([np.zeros((3-len_word,100),dtype = np.float32),words_[-len_word:]])
        else : 
            context = np.vstack(words_[-3:])

        model.eval()
        last=torch.tensor(gen_chars[-1],dtype = torch.long).view(1,1)
        context = torch.from_numpy(context).float()

        if hidden : 
            pred,hid = model(last,context.unsqueeze(0).unsqueeze(0),hid)
        else : 
            pred = model(last,context.unsqueeze(0).unsqueeze(0))
    
        gen_chars.append(sample_with_temp_topk(pred[:,-1][0],temperature=temp, top_k=top, forbidden_char=forbidden).item())

    text_generated = "".join([int2char[i] for i in gen_chars])

    return text_generated

def load_and_clean(path) :
    
    ckpt = torch.load(path, map_location="cpu")

    new_state_dict = {}
    for key, value in ckpt["model_state_dict"].items():
        new_key = key.replace("_orig_mod.", "model.")
        new_state_dict[new_key] = value

    ckpt["model_state_dict"] = new_state_dict

#    print("Performance_metric (ppl) : ", ckpt["val_ppl"])
    
    return ckpt

def generate_a_song_structure(matrix,states) :
    
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

    print([states[i] for i in song_struct])
    return([states[i] for i in song_struct])

def sample_with_temp_topk(logits, temperature=1.0, top_k=50, forbidden_char = None):
    logits = logits / temperature

    if forbidden_char : #Non authorized logit to 0
        for i in forbidden_char :       
            logits[i] = 0
    
    # keep top k char
    if top_k is not None:
        values, indices = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float('-inf'))
        mask[indices] = logits[indices]
        logits = mask
    
    probs = torch.softmax(logits, dim=-1)
    
    return torch.multinomial(probs, 1)