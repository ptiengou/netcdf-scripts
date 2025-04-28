#!/bin/bash
######################
## JEANZAY    IDRIS ##
######################
#SBATCH --job-name=job_create_nf_era5          # Job Name
#SBATCH --output=listing_create_nf_era5        # standard output
#SBATCH --error=listing_create_nf_era5        # error output
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1   # Number of MPI tasks
#SBATCH --hint=nomultithread         # 1 processus MPI par par physical core (no hyperthreading) 
#SBATCH --time=08:00:00                # Wall clock limit (minutes)
#SBATCH --account cuz@cpu
##BATCH_NUM_PROC_TOT=$BRIDGE_SBATCH_NPROC
set +x

# This script is used to create a nudging forcing file with variables taken from jean-zay (global files with longitudes 
# in [0,360] and variables from files on a different grid (here already on the restricted grid, with longitudes in [-180,180])

date

module load nco

# input parameters to specify:

year=2018
month=02

lat_min=20.0
lat_max=55.0
lon_min=335.0
lon_max=20.0
# lat_min=0.0
# lat_max=1.0
# lon_min=0.0
# lon_max=1.0

# freq is the frequence of the era5 input files
# can be either hourly or 4xdaily
freq=hourly

# working directory
working_dir=$SCRATCH/tmp_nf_split_${year}${month}

# output file name
file_out=maroc-espagne-nudging_forcing-$year$month.nc


#=============================

ref_dir=/gpfsstore/rech/psl/rpsl376/ergon/ERA5/NETCDF/GLOBAL_025		#<-- ERA5 files on JZ
#ref_dir=/gpfsstore/rech/cuz/upj17my/DATA/LAM_DATA/GLOBAL_025		      # <-- I have symbolic links pointing on the wanted JZ files

input_dir_pl=${ref_dir}/${freq}/AN_PL/${year}
input_dir_sf=${ref_dir}/${freq}/AN_SF/${year}
#input_dir_pl=${ref_dir}
#input_dir_sf=${ref_dir}

# Variables on the global grid
list_var_pl="ta q r u v ciwc clwc"
list_var_sf="sp t2m geopt"
# list_var_pl="ciwc"
# list_var_sf=""

# Variables on the restricted grid
list_var_pl_bis="ciwc clwc"

if [ ! -d $working_dir ] ; then
echo I create $working_dir
mkdir $working_dir
else
rm -f $working_dir/*
fi

echo I go to $working_dir
cd $working_dir
mkdir backup

# I select my variables:
for var_pl in $list_var_pl; do
	echo starting variable ${var_pl}
	fichier=${input_dir_pl}/${var_pl}.${year}${month}.ap1e5.GLOBAL_025.nc
	if [ ! -f tmp360.nc ] ; then
		echo restricting the domain
		if (( $(echo "$lon_min >= $lon_max" | bc -l) )); then
		    # If lon_min is greater than or equal to lon_max, it means the domain crosses the 180° meridian
		    # In this case, split the domain into two parts: [-180°, lon_max] and [lon_min, 180°]
		    echo spliting the work in two parts before and after Greenwich meridian
		    ncks -d longitude,0.0,$lon_max -d latitude,$lat_min,$lat_max $fichier tmp1.nc
		    cp tmp1.nc backup/backup_${var_pl}_part1.nc
		    ncks -d longitude,$lon_min,360.0 -d latitude,$lat_min,$lat_max $fichier -O tmp2.nc
		    cp tmp2.nc backup/backup_${var_pl}_part2.nc
		    echo merging two parts into single file tmp1.nc
		    ncks -A tmp2.nc tmp1.nc
		    rm tmp2.nc
		else
		    # Normal case where lon_min is less than lon_max
		    ncks -d longitude,$lon_min,$lon_max -d latitude,$lat_min,$lat_max $fichier tmp1.nc
		fi
		# echo I copy $fichier to tmp360.nc
		# cp tmp1.nc tmp360.nc
	else
		echo restricting the domain
		if (( $(echo "$lon_min >= $lon_max" | bc -l) )); then
		    # If lon_min is greater than or equal to lon_max, it means the domain crosses the 180° meridian
		    # In this case, split the domain into two parts: [-180°, lon_max] and [lon_min, 180°]
		    ncks -d longitude,0.0,$lon_max -d latitude,$lat_min,$lat_max $fichier tmp1.nc
		    cp tmp1.nc backup/backup_${var_pl}_part1.nc
		    ncks -d longitude,$lon_min,360.0 -d latitude,$lat_min,$lat_max $fichier -O tmp2.nc
                    cp tmp2.nc backup/backup_${var_pl}_part2.nc
		    echo variable ${var_pl} copied in backup folder
		    #  ncks -A tmp2.nc tmp1.nc
		    rm tmp2.nc
		else
		    # Normal case where lon_min is less than lon_max
		    ncks -d longitude,$lon_min,$lon_max -d latitude,$lat_min,$lat_max $fichier tmp1.nc
		fi
		# echo I add $var_pl to tmp360.nc    
		# ncks -A -v $var_pl tmp1.nc tmp360.nc
	fi
	rm tmp1.nc
done


for var_sf in $list_var_sf; do
	echo starting variable ${var_sf}
	fichier=${input_dir_sf}/${var_sf}.${year}${month}.as1e5.GLOBAL_025.nc
	echo I restrict the domain
		if (( $(echo "$lon_min >= $lon_max" | bc -l) )); then
		    # If lon_min is greater than or equal to lon_max, it means the domain crosses the 180° meridian
		    # In this case, split the domain into two parts: [-180°, lon_max] and [lon_min, 180°]
		    echo spliting the work in two parts before and after Greenwich meridian
		    ncks -d longitude,0.0,$lon_max -d latitude,$lat_min,$lat_max $fichier tmp1.nc
                    cp tmp1.nc backup/backup_${var_sf}_part1.nc
		    ncks -d longitude,$lon_min,360.0 -d latitude,$lat_min,$lat_max $fichier -O tmp2.nc
                    cp tmp2.nc backup/backup_${var_sf}_part2.nc
		    echo variable ${var_sf} copied in backup folder
		    # ncks -A tmp2.nc tmp1.nc
		    rm tmp2.nc
		else
		    # Normal case where lon_min is less than lon_max
		    ncks -d longitude,$lon_min,$lon_max -d latitude,$lat_min,$lat_max $fichier tmp1.nc
		fi
	# echo I add $var_sf to tmp360.nc
	# ncks -A -v $var_sf tmp1.nc tmp360.nc
	rm tmp1.nc
done

# echo I change the longitude range
# ncks -O --msa_usr_rdr -d longitude,180.0,359.75 -d longitude,0.0,179.75 tmp360.nc tmp180.nc
# ncap2 -O -s 'where(longitude>179.8) longitude=longitude-360' tmp180.nc tmp.nc

# rm tmp180.nc # but we keep tmp360, just in case the following part does not go as planned

# Do next part only if you have variables already on the restricted domain

# for var_pl in $list_var_pl_bis; do
# 	fichier=${input_dir_pl}/${var_pl}.${year}${month}.ap1e5.GLOBAL_025.nc
# 	echo I add $var_pl to tmp.nc    
# 	ncks -A -v $var_pl $fichier tmp.nc
# 	rm tmp1.nc
# done


# echo I rename the variables 
# ncrename -O -v geopt,z tmp.nc tmp.nc
# ncrename -O -v ta,t tmp.nc tmp.nc 

# mv tmp.nc $file_out

echo  youpi it is finished

# rm -f tmp.nc
