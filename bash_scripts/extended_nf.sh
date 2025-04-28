#!/bin/bash
######################
## JEANZAY    IDRIS ##
######################
#SBATCH --job-name=ext_2017         # Job Name
#SBATCH --output=job_outputs_errors/output_extended        # standard output
#SBATCH --error=job_outputs_errors/error_extended        # error output
#SBATCH --nodes=1
#SBATCH --ntasks=1   # Number of MPI tasks
#SBATCH --hint=nomultithread         # 1 processus MPI par par physical core (no hyperthreading) 
#SBATCH --time=01:00:00                # Wall clock limit (minutes)
#SBATCH --qos=qos_cpu-dev
#SBATCH --account cuz@cpu
##BATCH_NUM_PROC_TOT=$BRIDGE_SBATCH_NPROC
set +x

date

year=2017
months="01 02 03 04 05 06 07 08 09 10 11 12"
# months="01"

storage_dir=$STORE/DATA/LAM_DATA/espagne_nf
workdir=$SCRATCH/tmp_extended_nf_$year

mkdir $workdir
cd $workdir

mkdir -p $storage_dir/$year/extended

for month in $months; do
    echo Current month $year $month
    next_month=$(printf "%02d" $((10#$month + 1)))
    echo Next month $next_month
    # next year
    if [ $next_month -eq 13 ]; then
        echo December, taking January of next year
        next_year=$((year + 1))
        next_month=01
    else
        next_year=$year
    fi

    # current month file
    file_current=$storage_dir/$year/espagne_nf_$year$month.nc
    # next month file
    file_next=$storage_dir/$next_year/espagne_nf_$next_year$next_month.nc
    # intermediate first day file
    file_first_day=$workdir/tmp_firstday_$next_year$next_month.nc
    # output file
    file_extended=$workdir/tmp_extended_nf_$year$month.nc

    # extract one month
    echo extracting first day of $next_year $next_month
    ncks -O -d time,0,0 $file_next $file_first_day
    # adding first day to current month
    echo Adding first day of $next_year $next_month to $year $month
    ncrcat -O $file_current $file_first_day $file_extended
    #move file to extended folder
    cp $file_extended $storage_dir/$year/extended/espagne_nf_extended_$year$month.nc
done

