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
#SBATCH --time=02:00:00                # Wall clock limit (minutes)
#SBATCH --account cuz@cpu
#SBATCH --qos=qos_cpu-dev
##BATCH_NUM_PROC_TOT=$BRIDGE_SBATCH_NPROC
set +x
# ---- This script concatenates files along the time dimension ---
#  If the files are packed (i.e. have scale_factors + off_sets
# they are unpacked beforehand and repacked after the merge ------

date

# input parameters
year=2011
month_list="01 02 03 04 05 06 07 08 09 10 11 12"
working_dir=$SCRATCH/tmp_unpacking_$year
# save_dir=$STORE/DATA/LAM_DATA/espagne_nf/$year
save_dir=$STORE/DATA/LAM_DATA/ERA5_R_2011/unpacked
# file_out=espagne_nf_${year}_all.nc

echo Going to working directory $working_dir 
rm -rf $working_dir
mkdir $working_dir
cd $working_dir

for month in $month_list; do
      fichier=$save_dir/../packed/ERA5_R_2011_$month.nc	
      echo Unpacking $fichier 
#      ncpdq --unpack $fichier unpacked_$year$month.nc
      ncpdq --unpack $fichier $save_dir/../unpacked/ERA5_R_2011_$month.nc
done

# Concatenate files
# echo Mergetime
# cdo mergetime unpacked_* unpacked_${file_out}

#echo Packing the output file
#ncatted -O -a _FillValue,,o,f,-32767 unpacked_${file_out}
#ncpdq -P all_xst unpacked_${file_out} $file_out

#mv $file_out $save_dir

#echo $file_out moved to $save_dir

date

echo  Youpi it is finished



