import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, nheads, dropout, max_seq, bias=False):
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
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        self.d_ff = d_model * 4
        self.norm_1 = nn.RMSNorm(d_model)
        self.norm_2 = nn.RMSNorm(d_model)
        
        self.attn = MultiHeadAttention(d_model, n_heads, dropout = 0.15, max_seq=256)
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(approximate="tanh"),
#            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # masked multi-head self-attention
        x = x + self.attn(self.norm_1(x), is_causal=True) #Is causal = for token t, mask on every tokens after (cannot see what's coming after)
        # feed-forward
        x = x + self.ff(self.norm_2(x))
        return x
