#!/bin/bash
######################
## JEANZAY    IDRIS ##
######################
#SBATCH --job-name=job_create_nf_era5_201804          # Job Name
#SBATCH --output=listing_create_nf_era5_201804        # standard output
#SBATCH --error=listing_create_nf_era5_201804        # error output
#SBATCH --nodes=64
#SBATCH --exclusive
#SBATCH --ntasks=64   # Number of MPI tasks
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
month=04

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
file_out=espagne_nf_$year$month.nc


#=============================

ref_dir=/gpfsstore/rech/psl/rpsl376/ergon/ERA5/NETCDF/GLOBAL_025		#<-- ERA5 files on JZ
#ref_dir=/gpfsstore/rech/cuz/upj17my/DATA/LAM_DATA/GLOBAL_025		      # <-- I have symbolic links pointing on the wanted JZ files

input_dir_pl=${ref_dir}/${freq}/AN_PL/${year}
input_dir_sf=${ref_dir}/${freq}/AN_SF/${year}
#input_dir_pl=${ref_dir}
#input_dir_sf=${ref_dir}

# Variables on the global grid
list_var_pl="ta q r u v ciwc clwc"
#list_var_sf="sp t2m geopt"
#list_var_pl=""
list_var_sf=""

# Variables on the restricted grid
list_var_pl_bis=""

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

echo Extraction complete
echo Starting concatenation 

#list_var="sp t2m geopt ta q r u v ciwc clwc"
list_var="ta q r u v ciwc clwc"

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
