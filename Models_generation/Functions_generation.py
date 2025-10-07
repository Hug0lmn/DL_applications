import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

#LSTM model
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            batch_first=True, dropout = 0.15)
        self.drop = nn.Dropout(p=0.15)
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.drop(self.embedding(x))              # (batch, seq, hidden_size)
        out, hidden = self.lstm(x, hidden)
#        out = self.ln(out)
        out = self.drop(out)
        logits = self.fc(out)                  
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device = device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device = device)
        return (h0,c0)
    
#RNN model
class CharRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers=1, dropout = 0):
        super(CharRNN, self).__init__()
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
#        out = self.ln(out)
        out = self.drop(out)
        out = self.fc(out)                  
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
#Seq2seq Attention
class Encodeur_Atten(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(Encodeur_Atten, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            dropout = 0.2,
                            batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, length_batch):

        emb = self.embed(x)
        emb_drop = self.dropout(emb)

        packed_x = pack_padded_sequence(emb_drop, length_batch, batch_first=True, enforce_sorted=False)
        packed_out, (h,c) = self.lstm(packed_x)
        output, length_out = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        output = self.dropout(output)

        return output, length_out, h, c
    
class Decoder_Atten(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, masked_mapping, num_layers=1):
        super(Decoder_Atten, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mask = masked_mapping

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim + hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=0.2,
                            batch_first=True)

        self.final = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward_step(self, x, h, c, encod_out, mask_att):
        """
        x : (B,)          token ids
        h, c : (num_layers, B, H)
        encod_out : (B, L_enc, H)
        mask_att : (B, L_enc) attention mask
        """
        B, L_enc, H = encod_out.shape
        embedded = self.dropout(self.embed(x))  # (B,E)

        # ---- Luong dot-product attention ----
        # query = dernier état caché de la dernière couche
        query = h[-1]  # (B,H)

        # scores = produit scalaire (B,L_enc)
        scores = torch.bmm(encod_out, query.unsqueeze(2)).squeeze(2)

        # masque sur les PAD
        if mask_att is not None:
            scores = scores.masked_fill(mask_att == 0, float('-inf'))

        # softmax pour obtenir les poids
        weights = F.softmax(scores, dim=-1)  # (B,L_enc)

        # contexte pondéré
        h_prime = torch.bmm(weights.unsqueeze(1), encod_out).squeeze(1)  # (B,H)

        # ---- LSTM step ----
        lstm_in = torch.cat([embedded, h_prime], dim=-1).unsqueeze(1)  # (B,1,E+H)
        out, (h, c) = self.lstm(lstm_in, (h, c))

        logit = self.final(out.squeeze(1))  # (B, vocab_size)
        masked_logit = logit.masked_fill(self.mask, float("-inf"))

        return masked_logit, (h, c)

    def forward(self, x, encod_out, h, c, targets, teacher_forcing_ratio, mask_att, loss_fn=None):
      batch_size, max_len = targets.size()
      input_x = x[:, 0]  # <SOS>

      all_logits = []

      for t in range(max_len):
        out, (h, c) = self.forward_step(input_x, h, c, encod_out, mask_att)
        all_logits.append(out.unsqueeze(1))   # (B,1,V)

        # Teacher forcing
        if torch.rand(1).item() < teacher_forcing_ratio:
            input_x = targets[:, t]
        else:
            input_x = out.argmax(dim=-1)

      all_logits = torch.cat(all_logits, dim=1)  # (B, T, V)

      if loss_fn is None:
          return all_logits
      else:
        # reshape for CrossEntropyLoss
          loss = loss_fn(
              all_logits.reshape(-1, all_logits.size(-1)),
              targets.reshape(-1)
          )
          return loss
      

#Seq2seq no attention      
class Encodeur_no_att(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(Encodeur_no_att, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            dropout = 0.2,
                            batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, length_batch):
        emb = self.embed(x)
        emb_drop = self.dropout(emb)
        packed_x = pack_padded_sequence(emb_drop, length_batch, batch_first=True, enforce_sorted=False)                      
        packed_out, (h,c) = self.lstm(packed_x)             
        _, _ = pad_packed_sequence(packed_out, batch_first=True) #First right now only h,c
        return (h, c)
    
class Decodeur_no_att(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, masked_mapping, num_layers=1):
        super(Decodeur_no_att, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers, 
                            dropout = 0.2,
                            batch_first=True)
        
        self.final = nn.Linear(hidden_size,vocab_size)
        self.mask = masked_mapping

    def forward(self, x, h, c, length_batch, targets=None, teacher_forcing_ratio=1.0):
        batch_size, max_len = targets.size()
        device = x.device

        all_logits = []

        # Premier input : <SOS> pour tout le batch
        input_t = x[:, 0]

        for t in range(0, max_len):
            emb = self.embed(input_t).unsqueeze(1)  # (batch, 1, emb_dim)
            out, (h, c) = self.lstm(emb, (h, c))    # (batch, 1, hidden)
            logit = self.final(out.squeeze(1))      # (batch, vocab_size)
            masked_logit = logit.masked_fill(self.mask, float("-inf"))
            all_logits.append(masked_logit.unsqueeze(1))

            if torch.rand(1).item() < teacher_forcing_ratio:
                input_t = targets[:, t]   
            else:
                input_t = masked_logit.argmax(dim=-1)  

        all_logits = torch.cat(all_logits, dim=1)  
        return all_logits, (h, c)