#!/bin/bash
#SBATCH -N 1
#SBATCH --time=2:00:00
#SBATCH --wckey=p123z:python
#SBATCH --partition=an
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -o training_session_20240925_133153/Track_TEST1276_JOB.out
##SBATCH --mem=100G

module load Python/3.8.6
source TestUnet2/bin/activate
srun python3 trackPrediction.py --load training_session_20240925_133153/best_model_epoch9.pth --dataset_size "T1276x664" --division data_test --TempDim #--BGS_input  ## N750x664 N1276x664 N350x350 N500x500

