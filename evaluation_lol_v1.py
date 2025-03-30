import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import time
import os
import argparse
import numpy as np
from utils import PSNR, validation, LossNetwork
from model.IAT_main import IAT
from IQA_pytorch import SSIM, MS_SSIM
from data_loaders.lol_v1_new import lowlight_loader_new
from tqdm import tqdm
from torchprofile import profile_macs
# Import thop for calculating FLOPs and parameters.
from thop import profile
from fvcore.nn import FlopCountAnalysis
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--img_val_path', type=str, default='./LOL_v1/eval15/low/')
config = parser.parse_args()

print(config)
val_dataset = lowlight_loader_new(images_path=config.img_val_path, mode='test')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

model = IAT().cuda()
model.load_state_dict(torch.load("./workdirs/snapshots_folder_lol_v1_whole/best_Epochxaiorong.pth"))
model.eval()

input_tensor = torch.randn(1, 3, 256, 256).cuda()
flops, params = profile(model, inputs=(input_tensor,))

print(f'The calculated FLOPs (MACs): {flops}')
flops = flops * 4 / (1024 ** 3)
num_params = sum(p.numel() for p in model.parameters())
print(f'The calculated FLOPs in GB: {flops:.6f} GB')
print(f'The total number of parameters: {num_params}')


ssim = SSIM()
psnr = PSNR()
ssim_list = []
psnr_list = []

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if config.save:
    result_path = config.img_val_path.replace('low', 'Result')
    mkdir(result_path)

with torch.no_grad():
    for i, imgs in tqdm(enumerate(val_loader)):
        low_img, high_img, name = imgs[0].cuda(), imgs[1].cuda(), str(imgs[2][0])
        print(name)

        mul, add ,enhanced_img = model(low_img)

        if config.save:
            torchvision.utils.save_image(enhanced_img.cpu(), result_path + str(name) + '.png')

        ssim_value = ssim(enhanced_img.cpu(), high_img.cpu(), as_loss=False).item()
        print(ssim_value)
        psnr_value = psnr(enhanced_img.cpu(), high_img.cpu()).item()
        print(psnr_value)

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)

SSIM_mean = np.mean(ssim_list)
PSNR_mean = np.mean(psnr_list)
print('The SSIM Value is:', SSIM_mean)
print('The PSNR Value is:', PSNR_mean)

