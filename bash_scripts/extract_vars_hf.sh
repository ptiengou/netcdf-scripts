#extract variable from netcdf files
vars='fbase'

#irr_tag=irr
irr_tag=noirr

if [ $irr_tag == 'noirr' ]; then
    simnb1='14'
    simnb2='12'
elif [ $irr_tag == 'irr' ]; then
    simnb1='15'
    simnb2='13'
fi

dir1=/gpfsstore/rech/cuz/upj17my/IGCM_OUT/ICOLMDZOR/DEVT/amip_LAM/sim${simnb1}/ATM/Output/HF
dir2=/gpfsstore/rech/cuz/upj17my/IGCM_OUT/ICOLMDZOR/DEVT/amip_LAM/sim${simnb2}/ATM/Output/HF

months='01 02 03 04 05 06 07 08 09 10 11 12'

for varname in $vars; do
    output_final=/gpfswork/rech/cuz/upj17my/HF_outputs_LAM_concat/TS_HF_${irr_tag}_$varname.nc
    rm -f $output_final

    echo starting extraction for var $varname
    echo 1st part of simulation sim${simnb1}

    years='2010 2011 2012 2013 2014'
    outputfile1=sim${simnb1}_$varname.nc

    #remove the output file if it already exists
    rm -f $outputfile1
    mkdir tmp$simnb1

    #loop over all files in the directory
    for year in $years; do
        for month in $months; do
            file=$dir1/sim${simnb1}_$year$month*.nc
            echo Extracting from $file
            ncks -v $varname $file -O tmp${simnb1}/tmp$year$month.nc
        done
    done
    time cdo mergetime tmp${simnb1}/tmp*.nc $outputfile1
    rm -rf tmp$simnb1
    echo First part of sim done


    echo 2nd part of simulation sim${simnb2}
    years='2015 2016 2017 2018 2019 2020 2021 2022'
    months='01 02 03 04 05 06 07 08 09 10 11 12'

    outputfile2=sim${simnb2}_$varname.nc

    #remove the output file if it already exists
    rm -f $outputfile2
    mkdir tmp$simnb2

    #loop over all files in the directory
    for year in $years; do
        for month in $months; do
            file=$dir2/sim${simnb2}_$year$month*.nc
            echo Extracting from $file
            ncks -v $varname $file -O tmp${simnb2}/tmp$year$month.nc
        done
    done
    cdo mergetime tmp${simnb2}/tmp*.nc $outputfile2
    rm -rf tmp$simnb2
    echo 2nd part of sim done, merging both parts

    cdo mergetime $outputfile1 $outputfile2 $output_final
    rm -f $outputfile1 $outputfile2
done
