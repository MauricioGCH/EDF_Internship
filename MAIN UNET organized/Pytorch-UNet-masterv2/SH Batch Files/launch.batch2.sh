#!/bin/bash
#SBATCH -N 1
#SBATCH --time=48:00:00
#SBATCH --wckey=p123z:python
#SBATCH --partition=an
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -o AfterMods_JOB_%A.out
##SBATCH --mem=12G

module load Python/3.8.6
source TestUnet2/bin/activate
srun python3 NewTrain2.py --epochs 60 --batch-size 10 --learning-rate 1e-4 --amp --patience 5 --early_stopping_patience 10 --dataset_size 350 350 --TrainRotation --verticalFlip --BGS_input #--BGS_input #--ModLoss #--TempDim #--contrast --GaussianNoise --SpeckleNoise #--load training_session_20240725_155348/checkpoints/checkpoint_epoch50.pth
##'N500x500', 'N750x664', 'N350x350', 'N1276x664'

