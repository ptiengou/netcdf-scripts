date

years="2010 2011 2012 2013 2014"
months="01 02 03 04 05 06 07 08 09 10 11 12"
# months="01"


for year in $years; do
	in_dir=$STORE/DATA/LAM_DATA/espagne_nf/$year/extended
	out_dir=$STORE/DATA/LAM_DATA/espagne_nf/$year/6hours

	cd $in_dir

	mkdir -p $out_dir

	for month in $months; do
		in_file=espagne_nf_extended_$year$month.nc
		out_file=$out_dir/espagne_nf_extended_6h_$year$month.nc
   
	time cdo timselmean,6 $in_file $out_file 
	done
done
