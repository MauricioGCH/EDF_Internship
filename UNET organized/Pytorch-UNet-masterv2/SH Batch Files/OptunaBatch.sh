#!/bin/bash
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --wckey=p123z:python
#SBATCH --partition=an
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -o Optunahyperparam_search_%j.out
##SBATCH --mem=100G

module load Python/3.8.6
source TestUnet2/bin/activate

srun python3 OptunaOptim.py