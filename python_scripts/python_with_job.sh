#!/bin/bash
######################
## JEANZAY    IDRIS ##
######################
#SBATCH --job-name=winds1          # Job Name
#SBATCH --output=job_outputs_errors/out_winds1        # standard output
#SBATCH --error=job_outputs_errors/err_winds1        # error output
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1   # Number of MPI tasks
#SBATCH --hint=nomultithread         # 1 processus MPI par par physical core (no hyperthreading) 
#SBATCH --time=00:10:00                # Wall clock limit (minutes)
#SBATCH --account cuz@cpu
#SBATCH --qos=qos_cpu-dev
##BATCH_NUM_PROC_TOT=$BRIDGE_SBATCH_NPROC
set +x

date

python winds_for_Pedro.py
# python test_display
