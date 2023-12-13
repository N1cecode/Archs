import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqConv3x3(nn.Module):

    def __init__(self, seq_type, in_channels, out_channels, depth_multiplier=1):
        super(SeqConv3x3, self).__init__()
        self.seq_type = seq_type

        if self.seq_type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_channels * depth_multiplier)
            self.conv0 = nn.Conv2d(in_channels, self.mid_planes, kernel_size=1)
            self.conv1 = nn.Conv2d(self.mid_planes, out_channels, kernel_size=3, padding=1)

        elif self.seq_type == 'conv1x1-sobelx':
            self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
            self.sobel_x_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
            sobel_x_weight = torch.zeros_like(self.sobel_x_conv.weight)
            for i in range(out_channels):
                sobel_x_weight[i, 0, 0, 0] = 1.0
                sobel_x_weight[i, 0, 1, 0] = 2.0
                sobel_x_weight[i, 0, 2, 0] = 1.0
                sobel_x_weight[i, 0, 0, 2] = -1.0
                sobel_x_weight[i, 0, 1, 2] = -2.0
                sobel_x_weight[i, 0, 2, 2] = -1.0
            self.sobel_x_conv.weight.data = sobel_x_weight
            self.sobel_x_conv.weight.requires_grad = False

        elif self.seq_type == 'conv1x1-sobely':
            self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
            self.sobel_y_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
            sobel_y_weight = torch.zeros_like(self.sobel_y_conv.weight)
            for i in range(out_channels):
                sobel_y_weight[i, 0, 0, 0] = 1.0
                sobel_y_weight[i, 0, 0, 1] = 2.0
                sobel_y_weight[i, 0, 0, 2] = 1.0
                sobel_y_weight[i, 0, 2, 0] = -1.0
                sobel_y_weight[i, 0, 2, 1] = -2.0
                sobel_y_weight[i, 0, 2, 2] = -1.0
            self.sobel_y_conv.weight.data = sobel_y_weight
            self.sobel_y_conv.weight.requires_grad = False

        elif self.seq_type == 'conv1x1-laplacian':
            self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
            self.laplacian_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
            laplacian_weight = torch.zeros_like(self.laplacian_conv.weight)
            for i in range(out_channels):
                laplacian_weight[i, 0, 0, 1] = 1.0
                laplacian_weight[i, 0, 1, 0] = 1.0
                laplacian_weight[i, 0, 1, 2] = 1.0
                laplacian_weight[i, 0, 2, 1] = 1.0
                laplacian_weight[i, 0, 1, 1] = -4.0
            self.laplacian_conv.weight.data = laplacian_weight
            self.laplacian_conv.weight.requires_grad = False

        else:
            raise ValueError('The type of seqconv is not supported!')

    def forward(self, x):
        if self.seq_type == 'conv1x1-conv3x3':
            y0 = self.conv0(x)
            y1 = self.conv1(y0)
            
        elif self.seq_type == 'conv1x1-sobelx':
            y0 = self.conv0(x)
            y1 = self.sobel_x_conv(y0)

        elif self.seq_type == 'conv1x1-sobely':
            y0 = self.conv0(x)
            y1 = self.sobel_y_conv(y0)
            
        elif self.seq_type == 'conv1x1-laplacian':
            y0 = self.conv0(x)
            y1 = self.laplacian_conv(y0)

        else:
            raise ValueError('The type of seqconv is not supported!')
        
        return y1

    def rep_params(self):
        # We'll assume that the weight for conv0 remains the same and only calculate for the specific filters (sobelx, sobely, laplacian)

        if self.seq_type == 'conv1x1-conv3x3':
            rep_weight = F.conv2d(input=self.conv1.weight, weight=self.conv0.weight.permute(1, 0, 2, 3))
            rep_bias = torch.ones(1, self.mid_planes, 3, 3) * self.conv0.bias.view(1, -1, 1, 1)
            rep_bias = F.conv2d(input=rep_bias, weight=self.conv1.weight).view(-1) + self.conv1.bias

        elif self.seq_type == 'conv1x1-sobelx':
            input_weight = self.sobel_x_conv.weight.repeat(1, self.conv0.out_channels, 1, 1)
            rep_weight = F.conv2d(input=input_weight, weight=self.conv0.weight.permute(1, 0, 2, 3))
            rep_bias = self.sobel_x_conv.bias

        elif self.seq_type == 'conv1x1-sobely':
            input_weight = self.sobel_y_conv.weight.repeat(1, self.conv0.out_channels, 1, 1)
            rep_weight = F.conv2d(input=input_weight, weight=self.conv0.weight.permute(1, 0, 2, 3))
            rep_bias = self.sobel_y_conv.bias

        elif self.seq_type == 'conv1x1-laplacian':
            input_weight = self.laplacian_conv.weight.repeat(1, self.conv0.out_channels, 1, 1)
            rep_weight = F.conv2d(input=input_weight, weight=self.conv0.weight.permute(1, 0, 2, 3))
            rep_bias = self.laplacian_conv.bias

        else:
            raise ValueError('The type of seqconv is not supported!')

        return rep_weight, rep_bias

class RepConv2d(nn.Module):
    def __init__(self, regular_conv, seqconvs):
        super(RepConv2d, self).__init__()
        self.regular_conv = regular_conv
        self.seqconvs = nn.ModuleList(seqconvs)

    def forward(self, x):
        y = self.regular_conv(x)
        for seqconv in self.seqconvs:
            weight, bias = seqconv.rep_params()
            y += F.conv2d(input=x, weight=weight, bias=bias, stride=1, padding=1)
        return y

class ECB(nn.Module):
    """The ECB block used in the ECBSR architecture.

    Paper: Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices
    Ref git repo: https://github.com/xindongzhang/ECBSR

    Args:
        in_channels (int): Channel number of input.
        out_channels (int): Channel number of output.
        depth_multiplier (int): Width multiplier in the expand-and-squeeze conv. Default: 1.
        act_type (str): Activation type. Option: prelu | relu | rrelu | softplus | linear. Default: prelu.
        with_idt (bool): Whether to use identity connection. Default: False.
    """

    def __init__(self, in_channels, out_channels, depth_multiplier, act_type='prelu', with_idt=False):
        super(ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_type = act_type

        if with_idt and (self.in_channels == self.out_channels):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3 = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.in_channels, self.out_channels, self.depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.in_channels, self.out_channels)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.in_channels, self.out_channels)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.in_channels, self.out_channels)


        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_channels)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')


    def forward(self, x):
        if self.training:
            y = self.conv3x3(x) + self.conv1x1_3x3(x) + self.conv1x1_sbx(x) + self.conv1x1_sby(x) + self.conv1x1_lpl(x)
            if self.with_idt:
                y += x
        else:
            self.rep_conv = RepConv2d(self.conv3x3, [self.conv1x1_3x3, self.conv1x1_sbx, self.conv1x1_sby, self.conv1x1_lpl])
            y = self.rep_conv(x)  # 使用新的重参数化层
            if self.with_idt:
                y += x
        if self.act_type != 'linear':
            y = self.act(y)
        return y


# @ARCH_REGISTRY.register()
class ECBSR(nn.Module):
    """ECBSR architecture.

    Paper: Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices
    Ref git repo: https://github.com/xindongzhang/ECBSR

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_block (int): Block number in the trunk network.
        num_channel (int): Channel number.
        with_idt (bool): Whether use identity in convolution layers.
        act_type (str): Activation type.
        scale (int): Upsampling factor.
    """

    def __init__(self, num_in_ch, num_out_ch, num_block, num_channel, with_idt, act_type, scale):
        super(ECBSR, self).__init__()
        self.num_in_ch = num_in_ch
        self.scale = scale

        backbone = []
        backbone += [ECB(num_in_ch, num_channel, depth_multiplier=2.0, act_type=act_type, with_idt=with_idt)]
        for _ in range(num_block):
            backbone += [ECB(num_channel, num_channel, depth_multiplier=2.0, act_type=act_type, with_idt=with_idt)]
        backbone += [
            ECB(num_channel, num_out_ch * scale * scale, depth_multiplier=2.0, act_type='linear', with_idt=with_idt)
        ]

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x):
        if self.num_in_ch > 1:
            shortcut = torch.repeat_interleave(x, self.scale * self.scale, dim=1)
        else:
            shortcut = x  # will repeat the input in the channel dimension (repeat  scale * scale times)
        y = self.backbone(x) + shortcut
        y = self.upsampler(y)
        return y
