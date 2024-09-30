#!/bin/bash
#SBATCH -N 3
#SBATCH --time=48:00:00
#SBATCH --wckey=p123z:python
#SBATCH --partition=an
#SBATCH --gres=gpu:4
#SBATCH --exclusive
##SBATCH --mem=100G

module load Python/3.8.6
source TestUnet2/bin/activate

# Define arrays of parameter sets
declare -a epochs=("100" "100" "100")
declare -a batch_sizes=("64" "64" "64")
declare -a learning_rates=("1e-4" "5e-5" "1e-5")
declare -a loads=("false" "false" "false")  # Adjusted to use string "false" for boolean False
declare -a amps=("true" "false" "true")     # Assuming amp flag changes per job
declare -a patiences=("30" "60" "60")        # Assuming different patience values per job

# Define output directory
output_dir="job_outputs"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Iterate over the arrays using index
for ((i=0; i<${#epochs[@]}; i++)); do
    epoch=${epochs[i]}
    batch_size=${batch_sizes[i]}
    learning_rate=${learning_rates[i]}
    load=${loads[i]}
    amp=${amps[i]}
    patience=${patiences[i]}

    # Construct the command to execute
    cmd="python3 train_noWandb_uniquefolder.py --epochs ${epoch} --batch-size ${batch_size} --learning-rate ${learning_rate}"
   
    # Add load option if not "false"
    if [ "${load}" != "false" ]; then
        cmd+=" --load '${load}'"
    fi
   
    # Add amp flag if true
    if [ "${amp}" == "true" ]; then
        cmd+=" --amp"
    fi
   
    # Add patience option
    cmd+=" --patience ${patience}"
   
    # Define output file name with unique identifier (job index)
    output_file="${output_dir}/JOB_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${i}.out"
    
    # Run the command using srun and redirect output to unique file
    echo "Running: $cmd"
    srun $cmd > $output_file &
done

wait  # Wait for all background jobs to finish

echo "All jobs submitted."



