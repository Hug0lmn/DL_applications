import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def load_and_clean(path) :
    
    ckpt = torch.load(path, map_location="cpu")

    new_state_dict = {}
    for key, value in ckpt["model_state_dict"].items():
        new_key = key.replace("_orig_mod.", "model.")
        new_state_dict[new_key] = value

    ckpt["model_state_dict"] = new_state_dict

    print("Performance_metric (ppl) : ", ckpt["val_ppl"])
    
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

def sample_with_temp_topk(logits, temperature=1.0, top_k=50):
    # appliquer température
    logits = logits / temperature
    
    # garder seulement les top_k logits
    if top_k is not None:
        values, indices = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float('-inf'))
        mask[indices] = logits[indices]
        logits = mask
    
    # convertir en proba
    probs = torch.softmax(logits, dim=-1)
    
    # tirage
    return torch.multinomial(probs, 1)

## --- Models --- ##
## --- Character-level --- ##

class Char_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 128, hidden_size = 512, num_layers=3):
        super(Char_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            batch_first=True, dropout = 0.15)
        self.drop = nn.Dropout(p=0.15)
        self.ln = nn.LayerNorm(hidden_size)

#        self.proj = nn.Linear(hidden_size, embedding_dim, bias=False)
        self.fc = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x, hidden):
        x = self.drop(self.embedding(x))              # (batch, seq, hidden_size)
        out, hidden = self.lstm(x, hidden)
        out = self.drop(self.ln(out))
        logits = self.fc(out)                  
        return logits, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h0,c0)
    
class Char_RNN(nn.Module):
    def __init__(self, vocab_size, emb_size = 128, hidden_size = 512, num_layers = 3, dropout = 0):
        super(Char_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
         
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers, batch_first=True, dropout = dropout, nonlinearity ="relu")
        self.drop = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.drop(self.embedding(x))
        out, hidden = self.rnn(x, hidden)
        out = self.drop(out)
        out = self.fc(out)                  
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
## Multihead attention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, nheads, dropout, bias=True):
        super().__init__()

        self.nheads = nheads
        assert embed_dim % nheads == 0, "Embedding dim is not divisible by nheads"
        self.head_dim = embed_dim // nheads
        self.dropout = dropout
        
        self.packed_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor, attn_mask=None, is_causal=False) -> torch.Tensor:
        
        # Step 1
        result = self.packed_proj(x)
        query, key, value = torch.chunk(result, 3, dim=-1)

        # Step 2
        # (N, L_t, head_dim) -> (N, L_t, nheads, head_dim) -> (N, nheads, L_t, head_dim)
        query = query.unflatten(-1, [self.nheads, self.head_dim]).transpose(1, 2)
        key = key.unflatten(-1, [self.nheads, self.head_dim]).transpose(1, 2)
        value = value.unflatten(-1, [self.nheads, self.head_dim]).transpose(1, 2)

        # Step 3
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal)
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        return self.out_proj(attn_output)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=384*4, dropout=0.2):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # masked multi-head self-attention
        x = x + self.attn(self.norm1(x), is_causal=True) #Is causal == for token t mask on every tokens after (cannot see what's coming after)
        # feed-forward
        x = x + self.ff(self.norm2(x))
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_heads=6, n_layers=8, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.emb(x) + self.pos(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.fc(x)


## --- Subword-level --- ##

class Subword_Models(nn.Module) : 
    def __init__(self, model_type):
        super().__init__()
        
        if model_type == "LSTM" :
            self.model = Subword_LSTM()
        elif model_type == "RNN" :
            self.model = Subword_RNN()
        elif model_type == "MHA":
            self.model = Subword_MHA()
        else : 
            raise ValueError("model_type must be either 'LSTM','RNN' or 'MHA'")
    
    def forward(self, x, hidden = None) :
        if hidden is None :
            return self.model(x)
        else :
            return self.model(x, hidden)
    
    def init_hidden(self, batch_size) :
        if isinstance(self.model, Subword_LSTM) or isinstance(self.model, Subword_RNN):
            return self.model.init_hidden(batch_size)
        
# --- LSTM --- #
class Subword_LSTM(nn.Module) :
    def __init__(self, vocab_size = 4000, embed_dim = 384, hidden_size = 512, num_layers = 3, proba = 0.2):
        super().__init__()
    
        self.hidden_size = hidden_size
        self.num_layers = num_layers
 
        self.lstm_layers = nn.ModuleList([nn.LSTMCell(embed_dim if i == 0 else hidden_size, hidden_size, bias=True) for i in range(self.num_layers)])

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx = 0)
        self.drop = nn.Dropout(p=proba)
        self.ln = nn.LayerNorm(hidden_size)

        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x, hidden) :
        h,c = hidden 

        x_t = self.embed(x)
        for i, lstm_cell in enumerate(self.lstm_layers) :
            h[i], c[i] = lstm_cell(x_t,(h[i],c[i]))
            x_t = self.drop(self.ln(h[i]))

        logits = self.linear(x_t)

        return logits, (h,c)
    
    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h, c)

# --- RNN --- #    
class Subword_RNN(nn.Module):
    def __init__(self, vocab_size = 4000, emb_size = 384, hidden_size = 512, num_layers=3, dropout = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx= 0)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers, batch_first=True, dropout = dropout, nonlinearity ="relu")
        self.drop = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.drop(self.embedding(x))
        out, hidden = self.rnn(x, hidden)
        out = self.drop(out)
        out = self.fc(out)                  
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# --- MHA Transformer --- #
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, nheads, dropout, max_seq, bias=True):
        super().__init__()

        self.nheads = nheads
        assert embed_dim % nheads == 0, "Embedding dim is not divisible by nheads"
        self.head_dim = embed_dim // nheads
        self.drop = nn.Dropout(dropout)
        self.prob_drop = dropout
        
        self.proj_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len = max_seq)

    def forward(self, x: torch.Tensor, attn_mask=None, is_causal=True) -> torch.Tensor:
        
        # Step 1
        B,L,_ = x.shape
        result = self.proj_qkv(x)
        q, k, v = torch.chunk(result, 3, dim=-1)

        # Step 2
        # (N, L_t, head_dim) -> (N, L_t, nheads, head_dim) -> (N, nheads, L_t, head_dim)
        q,k,v = [t.reshape(B, L, self.nheads, self.head_dim) for t in (q,k,v)]

        q = self.rope(q)
        k = self.rope(k)

        #Adapt dim 
        q,k,v = [t.transpose(1,2) for t in (q,k,v)]

        # Step 3
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.prob_drop, attn_mask = attn_mask, is_causal=is_causal)
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        return self.drop(self.out_proj(attn_output))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.15):
        super().__init__()
        self.d_ff = d_model * 4
        self.norm = nn.RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, max_seq=256)
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # masked multi-head self-attention
        x = x + self.attn(self.norm(x), is_causal=True) #Is causal = for token t, mask on every tokens after (cannot see what's coming after)
        # feed-forward
        x = x + self.ff(self.norm(x))
        return x
class Subword_MHA(nn.Module):
    def __init__(self, vocab_size = 4000, d_model=320, n_heads=5, n_layers=6, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        
        self.drop = nn.Dropout(0.1)
        self.norm = nn.RMSNorm(d_model)
        
        self.fc = nn.Linear(d_model, vocab_size, bias = False)
        self.fc.weight = self.emb.weight

    def forward(self, x):
        x = self.drop(self.emb(x))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.fc(x)