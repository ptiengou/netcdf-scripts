#!/bin/bash
######################
## JEANZAY    IDRIS ##
######################
#SBATCH --job-name=job_concatenate_201802          # Job Name
#SBATCH --output=output_concatenate_201802        # standard output
#SBATCH --error=error_concatenate_201802        # error output
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1   # Number of MPI tasks
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread         # 1 processus MPI par par physical core (no hyperthreading) 
#SBATCH --time=02:00:00                # Wall clock limit (minutes)
#SBATCH --account cuz@cpu
#SBATCH --qos=qos_cpu-dev
##BATCH_NUM_PROC_TOT=$BRIDGE_SBATCH_NPROC
set +x

date

module load nco

year=2018
month=02

# working directory
working_dir=$SCRATCH/tmp_nf_split_$year$month

# output file name
file_out=espagne_nf_$year$month.nc

# Variables on the global grid
list_var="sp t2m geopt ta q r u v ciwc clwc"
#list_var="ta q r u v ciwc clwc"

echo I go to $working_dir
cd $working_dir

if [ ! -d tmp_concat/ ] ; then
mkdir tmp_concat
else
rm -f tmp_concat/*
fi

for var in $list_var; do
    echo Merging separate files into a single one for variable ${var}
    fichier1=${working_dir}/backup/backup_${var}_part1.nc
    fichier2=${working_dir}/backup/backup_${var}_part2.nc
    echo Changing the longitude range for both files without reordering
    # ncks -O --msa_usr_rdr -d longitude,180.0,359.75 -d longitude,0.0,179.75 $fichier1 tmp180_${var}1.nc
    ncap2 -O -s 'where(longitude>179.8) longitude=longitude-360' $fichier1  tmp180_${var}1.nc
    # ncks -O --msa_usr_rdr -d longitude,180.0,359.75 -d longitude,0.0,179.75 $fichier2 tmp180_${var}2.nc
    ncap2 -O -s 'where(longitude>179.8) longitude=longitude-360' $fichier2  tmp180_${var}2.nc
    echo Changing concatenation dimension to longitude 
    ncpdq -O -a longitude,time tmp180_${var}1.nc tmp_${var}1.nc
    ncpdq -O -a longitude,time tmp180_${var}2.nc tmp_${var}2.nc
    echo Concatenating on longitude
    ncrcat -O tmp_${var}2.nc tmp_${var}1.nc tmp_${var}_all.nc
    echo Putting back time as main dimension
    ncpdq -O -a time,longitude tmp_${var}_all.nc tmp_${var}_all.nc
    if [ ! -f $file_out ] ; then
    	echo creating ${file_out} by copying first variable file
	cp tmp_${var}_all.nc $file_out
    else
    	echo ${file_out} exists, adding ${var} to it
	ncks -A -v $var tmp_${var}_all.nc $file_out
    fi
    echo Moving tmp files to tmp_concat
    mv tmp180_${var}1.nc tmp_concat/
    mv tmp180_${var}2.nc tmp_concat/
    mv tmp_${var}1.nc tmp_concat/
    mv tmp_${var}2.nc tmp_concat/
    mv tmp_${var}_all.nc tmp_concat/
done

echo Renaming some variables 
ncrename -O -v geopt,z $file_out
ncrename -O -v ta,t $file_out

# rm -rf tmp_concat #optional if still debugging

echo  Concatenation complete

