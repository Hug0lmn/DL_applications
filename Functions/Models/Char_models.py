from Functions.Models.Transformer import TransformerBlock

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Char_Models(nn.Module) : 
    def __init__(self, model_type):
        super().__init__()
        
        if model_type == "GRU" :
            self.model = Char_GRU()
        elif model_type == "RNN" :
            self.model = Char_RNN()
        elif model_type == "Transformer":
            self.model = Char_transformer()
        else : 
            raise ValueError("model_type must be either 'GRU','RNN' or 'Transformer'")
    
    def forward(self, x, context, hidden = None) :
        if hidden is None :
            return self.model(x, context)
        else :
            return self.model(x, context, hidden)
    
    def init_hidden(self, batch_size) :
        if isinstance(self.model, Char_RNN) or isinstance(self.model, Char_GRU):
            return self.model.init_hidden(batch_size)
        
class Char_RNN(nn.Module):
    def __init__(self, vocab_size = 72, emb_size = 128, hidden_size = 512, num_layers=4, dropout = 0.2):
        super(Char_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(hidden_size)

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=71)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Context dim proj
        self.context_proj = nn.Linear(3 * 100, emb_size)  # flatten context_matrix to emb dim

        self.rnn = nn.RNN(
            input_size=emb_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            nonlinearity="tanh")

        # Attention
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size) #After concat, need to stream back the params to hidden_size
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False) #Learnable attention matrix

    def forward(self, x, context, hidden):
        x_embed = self.embedding(x) 

        # Context projection 
        context_flat = context.view(context.size(0), context.size(1), -1)  # Flatten context
        context_proj = self.context_proj(context_flat)                     

        x_input = torch.cat([x_embed, context_proj], dim=-1)               
        x_drop = self.drop(x_input)
        out, hidden = self.rnn(x_drop, hidden)
        out = self.drop(out)

        #Attention part Q/K/V
        #query = hidden[-1].unsqueeze(1)    
        keys = self.Wa(out)
        values = out        
        attn_scores = torch.bmm(out, keys.transpose(1, 2))

        #Mask attention
        L = out.size(1)
        mask = torch.tril(torch.ones(L, L, device=out.device)).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)                    
        context_vec = torch.bmm(attn_weights, values)

        combined = torch.cat((out, context_vec), dim=-1)                 
        combined = torch.tanh(self.attn_combine(combined)) #Non linearity
        
        out = self.fc(combined)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
class Char_GRU(nn.Module):
    def __init__(self, vocab_size = 72, embedding_dim = 128, hidden_size = 512, dropout=0.2, num_layers=3):
        super(Char_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(hidden_size)

        self.embed_drop = nn.Dropout(p=0.1)    
        self.rnn_drop = nn.Dropout(p=0.2)      
        self.attn_drop = nn.Dropout(p=0.1)     
        self.proj_drop = nn.Dropout(p=0.1)  

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=71)
        self.hidden_to_emb = nn.Linear(hidden_size, embedding_dim)

        self.fc = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.fc.weight = self.embedding.weight
        
        # Context dim proj
        self.context_proj = nn.Linear(3 * 100, embedding_dim)  # 3 prev words × 100-dim vectors

        self.gru = nn.GRU(
            input_size=embedding_dim * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )

        # Attention
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size) 
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False) #Learnable attention matrix

    def forward(self, x, context, h):
        
        x_embed = self.embed_drop(self.embedding(x))

        # --- Context projection ---
        context_flat = context.view(context.size(0), context.size(1), -1)  
        context_proj = self.context_proj(context_flat)                     

        x_input = torch.cat([x_embed, context_proj], dim=-1)               

        out, h = self.gru(x_input, h)
        out = self.rnn_drop(self.ln(out))

        #Attention part Q/K/V
        #query = hidden[-1].unsqueeze(1)    
        keys = self.Wa(out)
        values = out
        
        attn_scores = torch.bmm(out, keys.transpose(1, 2))/ math.sqrt(out.size(-1))

        #Mask attention
        L = out.size(1)
        mask = torch.tril(torch.ones(L, L, device=out.device)).unsqueeze(0)  
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)                    
        context_vec = torch.bmm(attn_weights, values)

        combined = torch.cat((out, context_vec), dim=-1)                 
        combined = torch.tanh(self.attn_combine(combined)) #Non linearity

        emb_space = self.hidden_to_emb(combined)

        out = self.fc(emb_space)
        return out, h

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return h0

## Not used because GRU perfs are better
class Char_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 128, hidden_size = 384, dropout=0.2, num_layers=4):
        super(Char_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(hidden_size)

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=71)
        self.context_proj = nn.Linear(3 * 100, embedding_dim)

        # LSTM with projection
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Attention layers now use proj_size
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

        self.fc = nn.Linear(hidden_size, vocab_size)
#        self.fc.weight = self.embedding.weight  # weight tying

    def forward(self, x, context, h, c):
        
        x_embed = self.embedding(x)

        context_flat = context.view(context.size(0), context.size(1), -1)
        context_proj = self.context_proj(context_flat)

        x_input = torch.cat([x_embed, context_proj], dim=-1)
        x_drop = self.drop(x_input)

        # --- LSTM forward ---
        out, (h, c) = self.lstm(x_drop, (h, c))
        out = self.drop(out)

        # --- Attention ---
        keys = self.Wa(out)
        values = out
        attn_scores = torch.bmm(out, keys.transpose(1, 2)) / math.sqrt(self.proj_size) #rescaling

        # causal mask
        L = out.size(1)
        mask = torch.tril(torch.ones(L, L, device=out.device)).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        context_vec = torch.bmm(attn_weights, values)

        # combine attention and output
        combined = torch.cat((out, context_vec), dim=-1)
        combined = torch.tanh(self.attn_combine(combined))

        # --- Output ---
        out = self.fc(combined)
        return out, h, c

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.proj_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

## Multihead attention
    
class Char_transformer(nn.Module):
    def __init__(self, vocab_size = 72, d_model=256, n_heads=4, n_layers=4, dropout = 0.25, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        
        # --- Context ---
        self.context_proj = nn.Linear(3*100, d_model)
        self.merge_proj = nn.Linear(d_model * 2, d_model)  
        
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        
        self.drop = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(d_model)
        
        self.fc = nn.Linear(d_model, vocab_size, bias = False)
        self.fc.weight = self.emb.weight

    def forward(self, x, context):
        
        context_flat = context.view(context.size(0), context.size(1), -1)
        context_emb = self.context_proj(context_flat)
        x_emb = self.emb(x)
        
        x_concat = torch.cat([x_emb, context_emb], dim=-1)
        x = self.drop(self.merge_proj(x_concat))

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.fc(x)