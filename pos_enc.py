import torch 
from torch import nn 
import pdb 
import math 

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=10):
        super().__init__() 
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(d_model,d_model,bias=False)
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)#shape(max_len,1)
        even_pos = torch.arange(0,d_model,2,dtype=torch.float)
        odd_pos = torch.arange(1,d_model,2,dtype=torch.float)
        div_eve = 1/torch.pow(10,4*(even_pos)/d_model)
        div_odd = 1/torch.pow(10,4*(odd_pos-1)/d_model)
        pe[:,0::2] = torch.sin(position/div_eve)
        pe[:,1::2] = torch.sin(position/div_odd)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer("pe",pe)

    
    def forward(self,x):
        x = x + self.linear(self.pe[:x.size(0),:])
        return self.dropout(x)

if __name__ == "__main__":
    pos_enc = PositionalEncoding(d_model=6)
    #TODO:Undestand the input data
    #input_x = torch.randn((8,2,6))
    #print(pos_enc(input_x).shape)
