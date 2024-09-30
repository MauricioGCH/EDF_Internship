#!/bin/bash
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --wckey=p123z:python
#SBATCH --partition=an
#SBATCH -o hyperparam_search_%j.out

module load Python/3.8.6
source TestUnet2/bin/activate

# Run the hyperparameter search
srun python3 hyperparameter_search.py 
