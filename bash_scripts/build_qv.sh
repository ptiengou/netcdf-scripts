#extract variable from netcdf files
varname=vitv
irr_tag=irr
#irr_tag=noirr

if [ $irr_tag == 'noirr' ]; then
    simnb1='14'
    simnb2='12'
elif [ $irr_tag == 'irr' ]; then
    simnb1='15'
    simnb2='13'
fi

dir1=$STORE/IGCM_OUT/ICOLMDZOR/DEVT/amip_LAM/sim${simnb1}/ATM/Output/HF
dir2=$STORE/IGCM_OUT/ICOLMDZOR/DEVT/amip_LAM/sim${simnb2}/ATM/Output/HF
out_dir=$SCRATCH/SCRIPTS_EXTRACTION

cd $out_dir

months='01 02 03 04 05 06 07 08 09 10 11 12'

output_final=TS_${irr_tag}_q${varname}.nc
echo starting extraction for q * $varname
echo 1st part of simulation sim${simnb1}

years='2010 2011 2012 2013 2014'
#years=''
outputfile1=$out_dir/TS_sim${simnb1}_q$varname.nc

mkdir tmp_${varname}_sim$simnb1

#loop over all files in the directory
for year in $years; do
    for month in $months; do
        in_file=$dir1/sim${simnb1}_$year$month*.nc
	tmp_out_file=tmp_${varname}_sim$simnb1/q${varname}_$year$month.nc
        tmp_mean_file=tmp_${varname}_sim$simnb1/q${varname}_mean_$year$month.nc
        echo Extracting from $in_file
	if [ $varname == 'vitu' ]; then
                cdo expr,'qu=ovap*vitu' $in_file $tmp_out_file
        elif [ $varname == 'vitv' ]; then
                cdo expr,'qv=ovap*vitv' $in_file $tmp_out_file
        fi
	echo Computing monthly mean of file
	cdo monmean $tmp_out_file $tmp_mean_file
        done
done
time cdo mergetime tmp_${varname}_sim${simnb1}/q${varname}_mean_*.nc $outputfile1
rm -rf tmp_${varname}_sim$simnb1
echo First part of sim done

echo 2nd part of simulation sim${simnb2}
years='2015 2016 2017 2018 2019 2020 2021 2022'
months='01 02 03 04 05 06 07 08 09 10 11 12'

outputfile2=$out_dir/TS_sim${simnb2}_q$varname.nc

mkdir tmp_${varname}_sim$simnb2

#loop over all files in the directory
for year in $years; do
    for month in $months; do
        in_file=$dir2/sim${simnb2}_$year$month*.nc
        tmp_out_file=tmp_${varname}_sim$simnb2/q${varname}_$year$month.nc
        tmp_mean_file=tmp_${varname}_sim$simnb2/q${varname}_mean_$year$month.nc
        echo Extracting from $in_file
        if [ $varname == 'vitu' ]; then
		cdo expr,'qu=ovap*vitu' $in_file $tmp_out_file
	elif [ $varname == 'vitv' ]; then
		cdo expr,'qv=ovap*vitv' $in_file $tmp_out_file
	fi 
        echo Computing monthly mean of file
        cdo monmean $tmp_out_file $tmp_mean_file
    done
done
time cdo mergetime tmp_${varname}_sim${simnb2}/q${varname}_mean_*.nc $outputfile2
rm -rf tmp_${varname}_sim$simnb2
echo 2nd part of sim done, merging both parts

cdo mergetime $outputfile1 $outputfile2 $output_final
