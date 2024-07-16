print("Importing tools")
from tools import *

print("Loading netcdf files")
#no_irr_dir='../../../JZ_simu_outputs/LAM/noirr_2010_2022'
no_irr_dir='/gpfsstore/rech/ngp/unm64zs/IGCM_OUT/LMDZOR/PROD/amip/NoIrr-IPSLcm6Hist*/ATM/Output/MO'

#irr_dir='../../../JZ_simu_outputs/LAM/irr_2010_2022'
irr_dir='/gpfsstore/rech/ngp/unm64zs/IGCM_OUT/LMDZOR/PROD/amip/Irr-IPSLcm6Hist/ATM/Output/MO'

fig_save_dir='figures'

#filename1 = '{}/ATM/TS*.nc'.format(no_irr_dir)
filename1 = '{}/NoIrr-IPSLcm6Hist*_2014*histmth.nc'.format(no_irr_dir)
sim_noirr = xr.open_mfdataset(filename1)
sim_noirr = sim_noirr.rename({'time_counter':'time'})
sim_noirr.attrs['name'] = 'no_irr'

filename2 = '{}/Irr-IPSLcm6Hist_2014*histmth.nc'.format(irr_dir)
sim_irr0 = xr.open_mfdataset(filename2)
sim_irr = sim_irr0.rename({'time_counter':'time'})
sim_irr.attrs['name'] = 'irr'

print("Plotting winds in sim")
#plot winds for both sims
heights=['850']
ds_list=[sim_noirr, sim_irr]
for ds in ds_list:
    for in_height in heights:
        print("Plotting for height {} in sim {}".format(in_height, ds))
        map_wind(ds, height=in_height, in_figsize=(10,8), in_cmap=reds, dist=6, in_scale=100, hex=False, hex_center=False)
        save_path='{}/{}_wind_{}.png'.format(fig_save_dir, ds.attrs['name'], in_height)
        plt.savefig(save_path, dpi=300)

print("Plotting diff")
#plot diff between sims
for in_height in heights:
    map_wind_diff(ds_list[0]-ds_list[1], height=in_height, in_figsize=(10,8), in_cmap=reds, dist=6, in_scale=100, hex=False, hex_center=False)
    save_path='{}/wind_diff_{}.png'.format(fig_save_dir, in_height)
    plt.savefig(save_path, dpi=300)
