import torch 
from torch import  nn 
from torch import Tensor
from typing import Optional, Any 
class TransformerEncoderLayer(nn.Module):
    def __init__(self,d_model, n_head, max_len, dropout):
        super().__init__()
        self.mhead_attn = nn.MultiheadAttention(
                        embed_dim=d_model,
                        num_heads = n_head,
                        dropout=dropout) 
        self.dropout = nn.Dropout(dropout)
        self.cnn1 = nn.Conv1d(max_len,max_len//2,kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.cnn2 = nn.Conv1d(max_len//2,max_len,kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
    
    def forward(self,src,src_mask=None,src_key_padding_mask=None):
        src2 = self.mhead_attn(src,src,src,attn_mask = src_mask,key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.cnn2(self.dropout(self.activation(self.cnn1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src 


if __name__ == "__main__":
    t_enc = TransformerEncoderLayer(10,2,12,0.2)
    src_input = torch.randn((5,12,10))
    print(t_enc(src_input).shape)