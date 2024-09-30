#!/bin/bash
#SBATCH -N 1
#SBATCH --time=48:00:00
#SBATCH --wckey=p123z:python
#SBATCH --partition=an
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -o AfterMods_JOB_%A.out
##SBATCH --mem=100G

module load Python/3.8.6
source TestUnet2/bin/activate
srun python3 sequential_training.py --epochs 60 --batch-size 10 --learning-rate 1e-4 --amp --patience 5 --early_stopping_patience 10 --TrainRotation --verticalFlip  #--BGS_input #--ModLoss #--TempDim #--contrast --GaussianNoise --SpeckleNoise


