import itertools
import subprocess
import os
import re
import time
import numpy as np
import pandas as pd

# Define hyperparameter ranges
batch_sizes = [8, 16]
patiences = [10, 20]
speckle_noises = [True, False]
gaussian_noises = [True, False]
bgs_inputs = [True, False]
temp_dims = [True, False]
contrasts = [True, False]
vertical_flips = [True]
train_rotations = [True]
learning_rates = [1e-5, 1e-4]


# Function to check if a combination exists in the CSV
def is_combination_tested(existing_combinations, params):
    batch_size, patience, speckle_noise, gaussian_noise, bgs_input, temp_dim, contrast, vertical_flip, train_rotation, learning_rate = params

    for _, row in existing_combinations.iterrows():
        if (row['LR (1e-)'] == learning_rate and
            row['batch_size'] == batch_size and
            row['Patience'] == patience and
            row['Rotation'] == train_rotation and
            row['V_Flip'] == vertical_flip and
            row['Contrast'] == contrast and
            row['Gaussian Noise'] == gaussian_noise and
            row['Speckle Noise'] == speckle_noise and
            row['Frame Stacking'] == temp_dim and
            row['BG'] == bgs_input):
            
            return True
            
            

    return False
    
# Create combinations
def generate_param_combinations():

    existing_combinations = pd.read_csv('training_log_AllModels.csv')
    
    combinations = []
    for params in itertools.product(
        batch_sizes,
        patiences,
        speckle_noises,
        gaussian_noises,
        bgs_inputs,
        temp_dims,
        contrasts,
        vertical_flips,
        train_rotations,
        learning_rates
    ):
        batch_size, patience, speckle_noise, gaussian_noise, bgs_input, temp_dim, contrast, vertical_flip, train_rotation, learning_rate = params
        if bgs_input and temp_dim:
            continue  # Skip invalid combination
        
        if is_combination_tested(existing_combinations, params):
            print("This combination was already used : ")
            print(params)
            continue  # Skip already tested combination
            
        combinations.append(params)
    return combinations

param_combinations = generate_param_combinations()

# SLURM job script template
slurm_script_template = """#!/bin/bash
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --wckey=p123z:python
#SBATCH --partition=an
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -o AfterMods_JOB_%A.out

module load Python/3.8.6
source TestUnet2/bin/activate
srun python3 train_noWandb_uniquefolder.py --epochs 200 --batch-size {batch_size} --learning-rate {learning_rate} --amp --patience {patience} --early_stopping_patience 30 --dataset_size 350x350 {train_rotation} {vertical_flip} {contrast} {speckle_noise} {gaussian_noise} {bgs_input} {temp_dim}
"""

# Function to create SLURM job script
def create_slurm_script(params, job_id):
    batch_size, patience, speckle_noise, gaussian_noise, bgs_input, temp_dim, contrast, vertical_flip, train_rotation, learning_rate = params
    slurm_script = slurm_script_template.format(
        batch_size=batch_size,
        patience=patience,
        speckle_noise='--SpeckleNoise' if speckle_noise else '',
        gaussian_noise='--GaussianNoise' if gaussian_noise else '',
        bgs_input='--BGS_input' if bgs_input else '',
        temp_dim='--TempDim' if temp_dim else '',
        contrast='--contrast' if contrast else '',
        vertical_flip='--verticalFlip' if vertical_flip else '',
        train_rotation='--TrainRotation' if train_rotation else '',
        learning_rate=learning_rate
    )
    
    script_filename = f"hyperparam_search_{job_id}.sh"
    with open(script_filename, 'w') as f:
        f.write(slurm_script)
    
    return script_filename

# Function to submit job and wait for completion
def submit_and_wait(script_filename):
    # Submit the job
    result = subprocess.run(['sbatch', script_filename], capture_output=True, text=True)
    job_id = re.search(r'Submitted batch job (\d+)', result.stdout).group(1)
    
    print(f"Submitted job {job_id} with script {script_filename}")
    
    # Wait for job to complete
    while True:
        # Check job status
        result = subprocess.run(['squeue', '--job', job_id, '--noheader'], capture_output=True, text=True)
        if not result.stdout.strip():
            break  # Job is not in the queue anymore
        time.sleep(60)  # Check every minute
    
    print(f"Job {job_id} completed")

    # Delete the SLURM script file
    os.remove(script_filename)
    print(f"Deleted script file {script_filename}")

# Function to evaluate and track results
def evaluate_results(job_id):
    # Check the output file for performance metrics
    output_file = f"AfterMods_JOB_{job_id}.out"
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            output = f.read()
            # Extract relevant information from the output
            best_val_score = extract_best_val_score(output)
            return best_val_score
    return None

def get_current_best_score(): # THIS ONE IS FROM THE CSV FILE
    try:
        # Load the CSV file
        df = pd.read_csv('training_log_AllModels.csv')

        # Check if 'Val No BG' column exists and is not empty
        if 'Val No BG' in df.columns and not df['Val No BG'].isnull().all():
            # Return the maximum value in the 'Val No BG' column
            return df['Val No BG'].max()
        else:
            return None
    except FileNotFoundError:
        # If the file doesn't exist, return None
        return None

def extract_best_val_score(output):
    # Extract the best validation score from the output
    match = re.search(r"Best Validation Score: ([\d.]+)", output)
    if match:
        return float(match.group(1))
    return None

# Submit SLURM jobs and evaluate
best_score = get_current_best_score()

if best_score is None:
    best_score = -np.inf  # If no previous score is found, set to a very low value


for i, params in enumerate(param_combinations):
    print(f"Running right now params: {params}")
    script_filename = create_slurm_script(params, i)
    submit_and_wait(script_filename)
    
    score = evaluate_results(i)
    
    if score and score > best_score:
        best_score = score
        
    else:
        print(f"Current model score was {score}, but it wasnt best than current best at {best_score}")

print(f"Best score: {best_score} with params: {best_params}")
