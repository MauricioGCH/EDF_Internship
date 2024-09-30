#!/bin/bash
#SBATCH -N 1
#SBATCH --time=2:00:00
#SBATCH --wckey=p123z:python
#SBATCH --partition=an
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -o training_session_20240910_094546/valVisualization_JOB.out
##SBATCH --mem=100G

module load Python/3.8.6
source TestUnet2/bin/activate
srun python3 Visualisation_test.py --model training_session_20240910_094546/best_model_epoch58.pth --dataset_size 350x350 --BGS_input #--TempDim #--BGS_input  ## N750x664 N1276x664 N350x350 N500x500#--TempDim

# "/fscronos/home/d42684/Documents/CODE/Pytorch-UNet-masterv2/training_session_20240725_155323/checkpoints/checkpoint_epoch200.pth"

