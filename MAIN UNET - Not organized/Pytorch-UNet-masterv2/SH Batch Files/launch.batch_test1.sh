#!/bin/bash
#SBATCH -N 1
#SBATCH --time=2:00:00
#SBATCH --wckey=p123z:python
#SBATCH --partition=an
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -o training_session_20240912_100816/Metrics_testFUllIMG_JOB.out
##SBATCH --mem=100G

module load Python/3.8.6
source TestUnet2/bin/activate
srun python3 test1.py --model training_session_20240912_100816/best_model_epoch65.pth --dataset_size 1276x664 --TempDim --test #--TempDim #--BGS_input  ## N750x664 N1276x664 N350x350 --test N500x500

