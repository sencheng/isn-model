#!/bin/bash
#SBATCH -J ANALIZE_ISN # A single job name for the array
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # All cores on one Node
#SBATCH --mem 5000 # Memory request
#SBATCH -t 1-00:00 # Maximum execution time (D-HH:MM)
#SBATCH -o ./job_files/ANALYZE_ISN_%A_%a.out # Standard output
#SBATCH -e ./job_files/ANALYZE_ISN_%A_%a.err # Standard error
#SBATCH --mail-type FAIL
#SBATCH --mail-user m.mohagheghi@ini.rub.de
#module restore cobel-spike
#conda init
#source /home/mohagmnr/projects/CoBeL-spike/packages/set_vars.sh
module restore 5HT2A

# Set NEST paths
source /home/mohagmnr/projects/5HT2A/packages/nest-install/bin/nest_vars.sh
echo "running simulation for task ID ${SLURM_ARRAY_TASK_ID}"
    
python figureNumInput_hpc.py ${SLURM_ARRAY_TASK_ID} 1
