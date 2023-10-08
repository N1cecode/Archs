import torch
from torchsummary import summary, summary_name

from arch.ecbsr_conv2d import ECBSR

size_dict = {
    '480p': (720, 480),
    '720p': (1280, 720),
    '1080p':(1920, 1080),
    '2k':   (2560, 1440),
    '4k':   (4096, 2160)}

hr_size = size_dict['720p']

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

model = ECBSR(num_in_ch=3, num_out_ch=3, num_block=4, num_channel=16, with_idt=False, act_type='prelu', scale=4)
model.train()
summary_name(model, (3, params['img_size'][0], params['img_size'][1]), device='cpu')
model.eval()
summary_name(model, (3, params['img_size'][0], params['img_size'][1]), device='cpu')
# print(height, width, model.flops() / 1e9)
