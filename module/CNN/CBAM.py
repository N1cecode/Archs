### Implement of "CBAM: Convolutional Block Attention Module"

import torch
from torch import nn
from torch.nn import init

from .CAB import ChannelAttention_MaxAvg


class SpatialAttention(nn.Module):
    """Spatial Attention used in CBAM.

    Args:
        kernel_size (int): Kernel size used in Spatial Attention.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_x, _= torch.max(x, dim=1, keepdim=True)
        avg_x = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_x, avg_x], 1)
        x = self.conv(result)
        x = self.sigmoid(x)
        return x


class CBAMBlock(nn.Module):
    """Implement of CBAMBlock.

    Args:
        num_feat (int): Channel number of intermediate features in Channel Attention.
        squeeze_factor (int): Channel squeeze factor in Channel Attention. Default: 16.
        kernel_size (int): Kernel size used in Spatial Attention. Default: 49.
    """
    def __init__(self, num_feat, squeeze_factor=16, kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention_MaxAvg(num_feat=num_feat, squeeze_factor=squeeze_factor)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        x_res = x
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x + x_res

