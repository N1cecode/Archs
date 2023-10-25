### Channel Attention Block implement and variants
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    """Channel Attention Block used in Hybrid Attention Transformer.

    Args:
        num_feat (int): Channel number of intermediate features.
        compress_ratio (int): Compress ratio used to reduce the complexity of the Convolutional layer. Default: 3.
        squeeze_factor (int): Channel squeeze factor. Default: 30.
    """
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)
    

class ChannelAttention_MaxAvg(nn.Module):
    """Channel attention with maxpooling and avgpooling

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention_MaxAvg, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x) :
        max_x = self.maxpool(x)
        avg_x = self.avgpool(x)
        max_x = self.attention(max_x)
        avg_x = self.attention(avg_x)
        x = self.sigmoid(max_x + avg_x)
        return x


class ChannelAttention_Std(nn.Module):
    """Channel attention with standard deviation as pooling layer. 
    Use torch.std(x, dim(2,3), keepdim=True) to implement.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention_Std, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = torch.std(x, dim=(2,3), keepdim=True)
        y = self.attention(y)
        return x * y