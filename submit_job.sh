#!/bin/bash
#SBATCH -N 1
#SBATCH -J ISN
#SBATCH -A hpc-prf-clbbs # Account
#SBATCH -t 02:00:00
#SBATCH -p normal
#SBATCH -o ./job_files/ISN_%A.out # Standard output
#SBATCH -e ./job_files/ISN_%A.err # Standard error
#SBATCH --mail-type FAIL
#SBATCH --mail-user m.mohagheghi@ini.rub.de

# Load necessary modules
module restore isn

# Set NEST paths
# source /upb/departments/pc2/groups/hpc-prf-clbbs/ISN/packages/nest-install/bin/nest_vars.sh
source /upb/departments/pc2/users/c/clbbs001/ISN/packages/nest-install/bin/nest_vars.sh

echo "running3 simulation for task ID ${SLURM_ARRAY_TASK_ID}"
    
./run_multiple_sims.sh $(($1*10)) $(($(($1+1))*10)) 70
sleep 7190
#mpirun -np 40 python simulateNetworks.py 19 125
