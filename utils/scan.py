import os
import yaml
import importlib

def build_arch(folder_path, opt):
    # Scan the folder for yaml and arch files
    files = os.listdir(folder_path)
    arch_files = [f for f in files if f.endswith('.py')]
    
    # Load arch file
    if len(arch_files) > 0:
        spec = importlib.util.spec_from_file_location("arch", os.path.join(folder_path, arch_files[0]))
        arch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arch_module)
        
        # Get the class name from the yaml config and instantiate the class
        network_opt = opt['network_g'].copy()
        class_name = network_opt.pop('type')
        hyperparameters = network_opt
        if class_name:
            arch_class = getattr(arch_module, class_name)
            arch = arch_class(**hyperparameters)  # instantiate the network class with hyperparameters
        else:
            raise ValueError('Class name not found in the yaml config.')
    else:
        raise FileNotFoundError('No arch file found in the specified folder.')
    
    return arch

if __name__ == "__main__":
    # Usage
    folder_path = 'experiments/Pretrained_SwinIR_Res_DIV2K_P48W8_LR2e-4_B8G1_2080ti_cos'
    with open(os.path.join(folder_path, "train_SwinIR_Pretrained_SRx2_DIV2K_Cos.yml"), 'r') as file:
        opt = yaml.safe_load(file)
    print(opt)
    arch = build_arch(folder_path, opt)
    print(arch)