#!/bin/bash
#SBATCH -N 1
#SBATCH -J ISN
#SBATCH -A hpc-prf-c+ # Account
#SBATCH -p short # Partition: short, batch, long
#SBATCH -t 00:30:00 # Walltime
#SBATCH --mail-type all #FAIL
#SBATCH --mail-user mohammadreza.mohagheghinejad@rub.de

# Load necessary modules
module restore ISN

# Set NEST paths
source /upb/departments/pc2/users/c/clbbs001/ISN/packages/nest-install/bin/nest_vars.sh

echo "running simulation for task ID ${SLURM_ARRAY_TASK_ID}"
    
./run_multiple_sims.sh 40
