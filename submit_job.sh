#!/bin/bash -l
#SBATCH -J ISN # A single job name for the array
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # All cores on one Node
#SBATCH --mem 10000 # Memory request
#SBATCH -t 1-00:00 # Maximum execution time (D-HH:MM)
#SBATCH -o ./job_files/ISN_%A_%a.out # Standard output
#SBATCH -e ./job_files/ISN_%A_%a.err # Standard error
#SBATCH --mail-type FAIL
#SBATCH --mail-user m.mohagheghi@ini.rub.de

# Load necessary modules
module restore 5HT2A

# Set NEST paths
source /home/mohagmnr/projects/5HT2A/packages/nest-install/bin/nest_vars.sh

echo "running simulation for task ID ${SLURM_ARRAY_TASK_ID}"
    
python simulateNetworks.py ${SLURM_ARRAY_TASK_ID} 10
