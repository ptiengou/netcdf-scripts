#!/bin/bash

# Base directory where the NetCDF files are located
base_dir="."
output_dir="$base_dir/850hPa"
mkdir -p "$output_dir"

# Temporary directory for intermediate files
temp_dir="$output_dir/temp"
mkdir -p "$temp_dir"

# Loop through each year from 2010 to 2022
for year in {2016..2022}; do
    echo "Processing year $year..."
    year_dir="$base_dir/$year"

    # Loop through each month
    for month in {01..12}; do
        input_file="$year_dir/espagne_nf_${year}${month}.nc"
        
        # Check if the input file exists
        if [ -f "$input_file" ]; then
            # Define intermediate and final output file names
            temp_file="$temp_dir/level850_${year}${month}.nc"
            output_file="$temp_dir/TQRUV_850_${year}${month}.nc"
            
            # Step 1: Select the specific level
            cdo sellevidx,31 "$input_file" "$temp_file"
            
            # Step 2: Extract the desired variables
            cdo selname,t,q,r,u,v "$temp_file" "$output_file"
        else
            echo "File $input_file does not exist."
        fi
    done
done

echo extraction over, concatenating

# Concatenate all the processed files into a single file
output_file="$output_dir/TQRUV_850_2010_2022.nc"
cdo cat "$temp_dir/TQRUV_850_*.nc" "$output_file"

# Clean up temporary files
# rm -r "$temp_dir"

echo "Processing complete. Final output is located at $output_file"

