### Implement of "MLP-Mixer: An all-MLP Architecture for Vision" for MLP in Transformer
### From https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/mlp/mlp_mixer.py

import torch
from torch import nn


class MlpBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self,x):
        #x: (bs,tokens,channels) or (bs,channels,tokens)
        return self.fc2(self.gelu(self.fc1(x)))


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim=16, channels_mlp_dim=1024, tokens_hidden_dim=32, channels_hidden_dim=1024):
        super().__init__()
        self.ln = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp_block = MlpBlock(tokens_mlp_dim, hidden_dim=tokens_hidden_dim)
        self.channels_mlp_block = MlpBlock(channels_mlp_dim, hidden_dim=channels_hidden_dim)

    def forward(self, x):
        """
        x: (bs,tokens,channels)
        """
        ### tokens mixing
        y = self.ln(x)
        y = y.transpose(1,2) #(bs,channels,tokens)
        y = self.tokens_mlp_block(y) #(bs,channels,tokens)
        ### channels mixing
        y = y.transpose(1,2) #(bs,tokens,channels)
        out  = x + y #(bs,tokens,channels)
        y = self.ln(out) #(bs,tokens,channels)
        y = out + self.channels_mlp_block(y) #(bs,tokens,channels)
        return y