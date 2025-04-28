date

year=2021
months="01 02 03 04 05 06 07 08 09 10 11 12"
# months="01"

storage_dir=$STORE/DATA/LAM_DATA/espagne_nf
workdir=$SCRATCH/tmp_extended_nf_$year

mkdir $workdir
cd $workdir

mkdir -p $storage_dir/$year/extended

for month in $months; do
    # output file
    file_extended=$workdir/tmp_extended_nf_$year$month.nc

    #move file to extended folder
    cp $file_extended $storage_dir/$year/extended/espagne_nf_extended_$year$month.nc
done

