#!/bin/bash
#SBATCH --job-name=moldiffdAE
#SBATCH --partition=v100g32
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --chdir=/data/home/cktmu/MolDiffdAE
#SBATCH --output=slurm-train-%j.log

export LD_LIBRARY_PATH=$HOME/miniconda3/envs/moldiffdae/lib:$LD_LIBRARY_PATH

exec $HOME/miniconda3/envs/moldiffdae/bin/python -u \
  scripts/train_drug3d_ae.py \
  --config configs/train/train_MolDiffAE_continuous.yml \
  --emb_dim 32 --wass_weight 10.0 --batch_size 32 \
  --logdir output/train_MolDiffAE_continuous
