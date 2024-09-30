#!/bin/bash
#SBATCH -N 1
#SBATCH --time=48:00:00
#SBATCH --wckey=p123z:python
#SBATCH --partition=an
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -o AfterMods_JOB_EXPS_MORE_EPOCHS_%A.out
##SBATCH --mem=100G

module load Python/3.8.6
source TestUnet2/bin/activate
srun python3 train_noWandb_uniquefolder.py --epochs 80 --batch-size 16 --learning-rate 7.27373103184475e-05 --weight_decay 6.377507022373094e-08 --amp --patience 20 --early_stopping_patience 20 --dataset_size 1276x664 --TrainRotation --contrast --verticalFlip --SpeckleNoise --BGS_input --load training_session_20240910_094546/best_model_epoch58.pth   #--load training_session_20240918_165613/best_model_epoch44.pth #--verticalFlip --load training_session_20240918_125031/best_model_epoch44.pth  #--BGS_input #--ModLoss #--TempDim #--contrast --GaussianNoise --SpeckleNoise 
##'500x500', '750x664', '350x350', '1276x664' --TrainRotation


## DE tTEMPDIM POR AHORA LA BLACION ESTA CORRIENDO DOS EXPS SIN SPECKLE, Y SIN SPECKLE CONTRAST? Y LA TALLA ORIGINAL SIN CROPS PEQUENOS

#vertical y rot

# luego paciencia y terminacion temprana para mas epocas y poder hacer mas experimentos

# luego contraste

# luego ruido speckle y gaussiano
# ya tengo muchos hyperparametros, usar OPtuna
# info temporal de bgs y 3 frames

# Ver efecto de reentrenar en escalas mayores contra directamente entrenar en en escala mayor
