## LFT-Net: A Lightweight Frequency-based Transformer for Low light Enhancement and Exposure Correction
Abstract: Low light images frequently suffer from composite degradations including underexposure, overexposure, and diminished visibility. Existing low light enhancement and exposure correction methods demonstrate two critical limitations: 
computational inefficiency and neglect of frequency-specific restoration patterns. Inspired by the concept that high-low frequency features captures different image features, we propose LFT-Net—A Lightweight Frequency-based Transformer for 
Low-light Enhancement and Exposure Correction. The model features a dual-branch architecture, with parallel detail processing and global correction pathways. The detail branch produces the base enhanced image, while the correction branch 
refines it through gamma adjustment and color calibration. We propose a High-Low Frequency Enhancement (HLFE) module, which leverages two branches to independently extract and enhance high-low frequency features, and a Residual Detail 
Enhancement (RDE) module to improve the generation of local image components. 
## Dependencies & Installation
- Python 3.7
- PyTorch 1.13.1
- NVIDIA GPU + [CUDA 11.7](https://developer.nvidia.com/cuda-downloads)
## I. Low-Light Enhancement (LOL-V1 dataset)
1. Download the dataset from the [here](https://daooshee.github.io/BMVC2018website/).
2. Training your model on LOL-V1 dataset

Step 1: crop the LOL-V1 dataset to 256 $\times$ 256 patches:
```
python LOL_patch.py --src_dir Your_Path/our485 --tar_dir Your_Path/our485_patch
```

Step 2: train on LOL-V1 patch images:
```
python train_lol_v1_patch.py --img_path Your_Path/our485_patch/low/ --img_val_path Your_Path/eval15/low/
```

Step 3: tuned the pre-train model (in Step 2) on LOL-V1 patches on the full resolution LOL-V1 image:
```
python train_lol_v1_whole.py --img_path Your_Path/our485/low/ --img_val_path Your_Path/eval15/low/ --pretrain_dir workdirs/snapshots_folder_lol_v1_patch/best_Epoch.pth
```
Step 4: Evaluation pretrain model on LOL-V1 dataset
```
python evaluation_lol_v1.py --img_val_path Your_Path/eval15/low/
```
## II. Low-Light Enhancement (LOL-V2-real dataset)
Step 1: Training your model on LOL-V2-real dataset for LOL-V2-real.
```
python train_lol_v2.py --gpu_id 0 --img_path Your_Path/Train/Low/ --img_val_path Your_Path/Test/Low/ 
```
Step 2: Evaluation pretrain model on LOL-V2-real dataset
```
python evaluation_lol_v2.py --img_val_path Your_Path/Test/Low/
```
## III. Exposure Correction
1. Download the dataset from [Training](https://ln2.sync.com/dl/141f68cf0/mrt3jtm9-ywbdrvtw-avba76t4-w6fw8fzj),
[Validation](https://ln2.sync.com/dl/49a6738c0/3m3imxpe-w6eqiczn-vripaqcf-jpswtcfr), [Testing](https://ln2.sync.com/dl/098a6c5e0/cienw23w-usca2rgh-u5fxikex-q7vydzkp).
Step 1: Training your model on Exposure dataset 
```
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 python -m torch.distributed.launch --nproc_per_node=2 train_exposure.py --img_path Your_Path/train/INPUT_IMAGES --img_val_path Your_Path/validation/INPUT_IMAGES
```
Step 2: Evaluation pretrain model on Exposure dataset
```
python evaluation_exposure.py --gpu_id 0 --img_val_path Your_Path/test/INPUT_IMAGES/ --expert a/b/c/d/e 
```
