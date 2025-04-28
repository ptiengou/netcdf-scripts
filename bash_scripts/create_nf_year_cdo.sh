#!/bin/bash
######################
## JEANZAY    IDRIS ##
######################
#SBATCH --job-name=job_create_nf_2017          # Job Name
#SBATCH --output=job_outputs_errors/output_create_nf_era5_2017        # standard output
#SBATCH --error=job_outputs_errors/error_create_nf_era5_2017        # error output
#SBATCH --nodes=1
#SBATCH --ntasks=1   # Number of MPI tasks
#SBATCH --hint=nomultithread         # 1 processus MPI par par physical core (no hyperthreading) 
#SBATCH --time=10:00:00                # Wall clock limit (minutes)
#SBATCH --account cuz@cpu
##BATCH_NUM_PROC_TOT=$BRIDGE_SBATCH_NPROC
set +x

# This script is used to create a nudging forcing file with variables taken from jean-zay (global files with longitudes 
# in [0,360] and variables from files on a different grid (here already on the restricted grid, with longitudes in [-180,180])

date

year=2017
# months="01 02 03 04 05 06 07 08 09 10 11 12"
months="08 09 10 11 12"

lat_min=20.0
lat_max=55.0
lon_min=-25
lon_max=20.0

# Variables taken from ERA5 repo on JZ
list_var_pl="ta q r u v ciwc clwc"
list_var_sf="sp geopt t2m"
# list_var_pl="ta q u v ciwc clwc"
# list_var_sf=""

# freq is the frequence of the era5 input files ( can be either hourly or 4xdaily)
freq=hourly

# Folder with ERA5 files
ref_dir=/gpfsstore/rech/psl/rpsl376/ergon/ERA5/NETCDF/GLOBAL_025                #<-- ERA5 files on JZ

input_dir_pl=${ref_dir}/${freq}/AN_PL/${year}
input_dir_sf=${ref_dir}/${freq}/AN_SF/${year}

# Variables uploaded manually, already on the restricted grid
list_var_pl_bis=""

for month in $months; do
	
	# working directory
	working_dir=$SCRATCH/tmp_nf_${year}${month}
	
	# output file name
	file_out=espagne_nf_$year$month.nc
	
	if [ ! -d $working_dir ] ; then
	echo I create $working_dir
	mkdir $working_dir
	else
	rm -rf $working_dir/*
	fi
	
	echo I go to $working_dir
	cd $working_dir
	
	echo Variables on pressure levels
	for var_pl in $list_var_pl; do
	        echo Starting variable $var_pl
	        fichier=${input_dir_pl}/${var_pl}.${year}${month}.ap1e5.GLOBAL_025.nc
	        if [ ! -f tmp_all.nc ] ; then
	                echo Extracting $var_pl from $fichier
	                cdo sellonlatbox,$lon_min,$lon_max,$lat_min,$lat_max $fichier tmp_$var_pl.nc
	                echo I copy $var_pl to tmp_all.nc
	                cp tmp_$var_pl.nc tmp_all.nc
	        else
	                echo Extracting $var_pl from $fichier
	                cdo sellonlatbox,$lon_min,$lon_max,$lat_min,$lat_max $fichier tmp_$var_pl.nc
	                echo I add $var_pl to tmp_all.nc    
	                ncks -A -v $var_pl tmp_${var_pl}.nc tmp_all.nc
	        fi
	done
	
	echo Variables at the surface
	for var_sf in $list_var_sf; do
	        echo Starting variable $var_sf
	        fichier=${input_dir_sf}/${var_sf}.${year}${month}.as1e5.GLOBAL_025.nc
	        echo Extracting $var_sf from $fichier
	        cdo sellonlatbox,$lon_min,$lon_max,$lat_min,$lat_max $fichier tmp_${var_sf}.nc
	        echo I add $var_sf to tmp_all.nc
	        ncks -A -v $var_sf tmp_${var_sf}.nc tmp_all.nc    
	done
	
	# Do next part only if you have variables already on the restricted domain
	
	# for var_pl in $list_var_pl_bis; do
	#       fichier=${input_dir_pl}/${var_pl}.${year}${month}.ap1e5.GLOBAL_025.nc
	#       echo I add $var_pl to tmp_all.nc    
	#       ncks -A -v $var_pl $fichier tmp_all.nc
	# done
	
	echo Renaming variables geopt and ta
	ncrename -O -v geopt,z tmp_all.nc tmp_all.nc
	ncrename -O -v ta,t tmp_all.nc tmp_all.nc 
		
	echo Moving output file to repo
	cp tmp_all.nc $file_out
	mv $file_out /gpfsstore/rech/cuz/upj17my/DATA/LAM_DATA/espagne_nf/${year}/
	
	echo Moving all tmp files to tmp_backup folder
	mkdir tmp_backup
	mv tmp_*.nc tmp_backup
	rm -f *.tmp
	
	echo  Extraction complete for $month $year
done

