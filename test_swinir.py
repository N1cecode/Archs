import torch
from torchinfo import summary
from ptflops import get_model_complexity_info

# from arch.swinir_arch import SwinIR
from arch.SwinIR.swinir_arch import SwinIR

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

summary(model, (3, params['img_size'][0], params['img_size'][1]), verbose=2, device='cpu', depth=3)
# summary_name(model, (3, params['img_size'][0], params['img_size'][1]), device='cpu')
# print(height, width, model.flops() / 1e9)


# macs, params = get_model_complexity_info(model, (3, 64, 64), verbose=False, print_per_layer_stat=True)

# macs = float(macs[:-4])

# print('MACs:', macs)
# print('Params:', params)