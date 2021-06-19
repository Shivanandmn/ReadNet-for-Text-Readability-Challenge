import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from pos_enc import PositionalEncoding
from tr_enc_layer import TransformerEncoderLayer

#TODO: WHAT IS "ninp"?
#TODO: BUILD TRANSFORMER ENCODER LAYER
class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, max_len, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout,max_len)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, max_len, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        #TODO: attention aggregation layer 
        return output


class AttnAggregationLayer(nn.Module):
    def __init__(self):
        super().__init__()
        #TODO: adding attention aggregationlayer to the model 
    
    def forward(self):
        pass 


if __name__ == "__main__":
    tf_model = TransformerModel(100,10,2,5,2)
    src_input = torch.randint(0,10,(5,11,10))
    mask_input = tf_model.generate_square_subsequent_mask(11)
    print(mask_input.dtype)
    print(tf_model(src_input).shape)