#!/bin/bash
######################
## JEANZAY    IDRIS ##
######################
#SBATCH --job-name=merge18          # Job Name
#SBATCH --output=job_outputs_errors/output_merge_18        # standard output
#SBATCH --error=job_outputs_errors/output_merge_18        # error output
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1   # Number of MPI tasks
#SBATCH --hint=nomultithread         # 1 processus MPI par par physical core (no hyperthreading) 
#SBATCH --time=01:00:00                # Wall clock limit (minutes)
#SBATCH --account cuz@cpu
#SBATCH --qos=qos_cpu-dev
##BATCH_NUM_PROC_TOT=$BRIDGE_SBATCH_NPROC
set +x

# months="02 03 04 05 06 07 08 09 10 11 12"
month="01"

echo Doing month $month
r_file=ERA5_R_2011/unpacked/ERA5_R_2011_$month.nc
rest_file=espagne_nf/2011/without_R/espagne_nf_2011$month.nc
output=ERA5_R_2011/unpacked/espagne_nf_2011$month.nc
echo Copying month $month
cp $rest_file $output
echo Appending R
ncks -A -v "r" $r_file $output
echo Moving output file to espagne_nf dir
mv $output espagne_nf/2011/
echo Over.

