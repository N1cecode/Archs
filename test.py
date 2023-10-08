import torch
from torchsummary import summary

from arch.swinir_arch import SwinIR

hr_size = (1024, 720)
height = (hr_size[0] // 4 // 8 + 1) * 8
width = (hr_size[1] // 4 // 8 + 1) * 8

light_weight = {
    'upscale': 4,
    'window_size': 8,
    'img_range':1.,
    'img_size': (0, 0),
    'depths': [6, 6, 6, 6],
    'embed_dim': 60,
    'num_heads': [6, 6, 6, 6],
    'mlp_ratio': 2,
    'upsampler':'pixelshuffledirect'
}
light_weight['img_size'] = ((hr_size[0] // light_weight['upscale'] // light_weight['window_size'] + 1) * light_weight['window_size'], 
                            (hr_size[1] // light_weight['upscale'] // light_weight['window_size'] + 1) * light_weight['window_size'])

classical_weight = {
    'upscale': 4,
    'window_size': 8,
    'img_range':1.,
    'img_size': (0, 0),
    'depths': [6, 6, 6, 6, 6, 6],
    'embed_dim': 180,
    'num_heads': [6, 6, 6, 6, 6, 6],
    'mlp_ratio': 2,
    'upsampler':'pixelshuffle'
}
classical_weight['img_size'] = ((hr_size[0] // classical_weight['upscale'] // classical_weight['window_size'] + 1) * classical_weight['window_size'], 
                                (hr_size[1] // classical_weight['upscale'] // classical_weight['window_size'] + 1) * classical_weight['window_size'])

params = light_weight

model = SwinIR(
    upscale=params['upscale'],
    img_size=params['img_size'],
    window_size=params['window_size'],
    img_range=params['img_range'],
    depths=params['depths'],
    embed_dim=params['embed_dim'],
    num_heads=params['num_heads'],
    mlp_ratio=params['mlp_ratio'],
    upsampler=params['upsampler'])

summary(model, (3, height, width), device='cpu')
# print(height, width, model.flops() / 1e9)
