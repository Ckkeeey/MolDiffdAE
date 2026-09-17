#!/bin/bash
#SBATCH --partition=v100g32
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=gpu_check

module load cuda/11.8
source ~/.bashrc
conda activate moldiffdae

python -c "
import torch
print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU设备名:', torch.cuda.get_device_name(0))
"
