#!/bin/bash
#SBATCH -N 1
#SBATCH -J ISN
#SBATCH -A hpc-prf-clbbs # Account
#SBATCH -p batch # Partition: short, batch, long
#SBATCH -t 02:00:00
#SBATCH -o ./job_files/ISN_%A.out # Standard output
#SBATCH -e ./job_files/ISN_%A.err # Standard error
#SBATCH --mail-type FAIL
#SBATCH --mail-user mohammadreza.mohagheghinejad@rub.de

# Load necessary modules
module restore ISN

# Set NEST paths
# source /upb/departments/pc2/groups/hpc-prf-clbbs/ISN/packages/nest-install/bin/nest_vars.sh
source /upb/departments/pc2/users/c/clbbs001/ISN/packages/nest-install/bin/nest_vars.sh

echo "running simulation for task ID ${SLURM_ARRAY_TASK_ID}"
    
./run_multiple_sims.sh $(($1*40)) $(($(($1+1))*40)) 400
sleep 1000
#mpirun -np 40 python simulateNetworks.py 19 125
