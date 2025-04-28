#!/bin/bash
######################
## JEANZAY    IDRIS ##
######################
#SBATCH --job-name=job_merge_1          # Job Name
#SBATCH --output=job_outputs_errors/output_merge_1        # standard output
#SBATCH --error=job_outputs_errors/error_merge_1        # error output
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1   # Number of MPI tasks
#SBATCH --hint=nomultithread         # 1 processus MPI par par physical core (no hyperthreading) 
#SBATCH --time=01:00:00                # Wall clock limit (minutes)
#SBATCH --account cuz@cpu
#SBATCH --qos=qos_cpu-dev
##BATCH_NUM_PROC_TOT=$BRIDGE_SBATCH_NPROC
set +x
# ---- This script concatenates files along the time dimension ---
#  If the files are packed (i.e. have scale_factors + off_sets
# they are unpacked beforehand and repacked after the merge ------

date

module load nco

# input parameters to specify:

working_dir=$SCRATCH/tmp_unpacking
save_dir=$STORE/DATA/LAM_DATA/espagne_nf/2019

#list_date="122019 012020"
file_in_1=espagne_nf_201901.nc
file_in_2=espagne_nf_201902.nc
file_out=espagne_nf_201901-02.nc

echo Going to working directory $working_dir 
rm -rf $working_dir
mkdir $working_dir
cd $working_dir

# for date in $list_date; do
# 	fichier=/gpfsstore/rech/wuu/uql95ey/MOSAIC_files/MOSAIC_nudging_forcing-${date}.nc
# 	echo Unpacking $fichier 
# 	ncpdq --unpack $fichier unpacked-MOSAIC_nudging_forcing-${date}.nc
# done

echo Unpacking $file_in_1
ncpdq --unpack $save_dir/$file_in_1 unpacked_${file_in_1}
echo Unpacking $file_in_2
ncpdq --unpack $save_dir/$file_in_2 unpacked_${file_in_2}

# Concatenate files
echo Mergetime
cdo mergetime unpacked_${file_in_1} unpacked_${file_in_2} unpacked_${file_out}

echo Packing the output file
ncatted -O -a _FillValue,,o,f,-32767 unpacked_${file_out}
ncpdq -P all_xst unpacked_${file_out} $file_out

mv $file_out $save_dir

echo $file_out moved to $save_dir

date

echo  Youpi it is finished
