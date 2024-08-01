print("Importing tools")
from tools import *

print("Loading netcdf files")
#no_irr_dir='../../../JZ_simu_outputs/LAM/noirr_2010_2022'
# no_irr_dir='/data/ptiengou'
no_irr_dir='/gpfsstore/rech/ngp/unm64zs/IGCM_OUT/LMDZOR/PROD/amip/NoIrr-IPSLcm6Hist*/ATM/Output/MO'
#no_irr_dir='/gpfswork/rech/cuz/upj17my/wind_tests_python/noirr'

#irr_dir='../../../JZ_simu_outputs/LAM/irr_2010_2022'
# irr_dir='/data/ptiengou'
irr_dir='/gpfsstore/rech/ngp/unm64zs/IGCM_OUT/LMDZOR/PROD/amip/Irr-IPSLcm6Hist/ATM/Output/MO'
#irr_dir='/gpfswork/rech/cuz/upj17my/wind_tests_python/irr'

fig_save_dir='figures'

#filename1 = '{}/ATM/TS*.nc'.format(no_irr_dir)
# filename1 = '{}/pedro_test_noirr_2014.nc'.format(no_irr_dir)
# filename1 = '{}/NoIrr-IPSLcm6Hist*_2014*histmth.nc'.format(no_irr_dir)
filename1 = '{}/NoIrr-IPSLcm6Hist*_*histmth.nc'.format(no_irr_dir)
sim_noirr = xr.open_mfdataset(filename1)
sim_noirr = sim_noirr.rename({'time_counter':'time'})
sim_noirr.attrs['name'] = 'no_irr'
sim_noirr=sim_noirr.sel(lon=slice(-20,100),lat=slice(45,-10))

# filename2 = '{}/pedro_test_irr_2014.nc'.format(irr_dir)
# filename2 = '{}/Irr-IPSLcm6Hist_2014*histmth.nc'.format(irr_dir)
filename2 = '{}/Irr-IPSLcm6Hist_*histmth.nc'.format(irr_dir)
sim_irr = xr.open_mfdataset(filename2)
sim_irr = sim_irr.rename({'time_counter':'time'})
sim_irr.attrs['name'] = 'irr'
sim_irr=sim_irr.sel(lon=slice(-20,100),lat=slice(45,-10))

var='u850'
ds=sim_noirr
#print("Plotting u850")
#map_ave(ds, var)
#print("Saving figure")
#save_path='figures/test1.png'
#plt.savefig(save_path,dpi=300)

#time series
var='z850'
ds_list=[sim_noirr, sim_irr]
time_series(ds_list, var, in_figsize=(9, 5.5), year_min=1950, year_max=2014, in_title=None, in_ylabel=None, in_xlabel=None)
save_path='figures/time_series_1.png'
plt.savefig(save_path, dpi=300)


heights=['850']
ds_list=[sim_noirr, sim_irr]
print("Plotting winds")
for ds in ds_list:
    for in_height in heights:
        print("Plotting for height {} in sim {}".format(in_height, ds.attrs['name']))
        map_wind(ds, height=in_height, in_figsize=(15,8), in_cmap=reds, dist=6, in_scale=100, hex=False, hex_center=False)
        print("Now saving figure")
        save_path='{}/{}_wind_{}.png'.format(fig_save_dir, ds.attrs['name'], in_height)
        plt.savefig(save_path, dpi=300)

print("Plotting diff")
for in_height in heights:
    map_wind_diff(ds_list[1],ds_list[0], height=in_height, in_figsize=(15,8), in_cmap=emb, dist=6, in_scale=10, hex=False, hex_center=False)
    print("Now saving figure")
    save_path='{}/wind_diff_{}.png'.format(fig_save_dir, in_height)
    plt.savefig(save_path, dpi=300)
