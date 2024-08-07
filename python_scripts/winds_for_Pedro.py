print("Importing tools")
import sys
sys.path.append('../python_tools/')
from tools import *

print("Loading netcdf files")
# no_irr_dir='../../../JZ_simu_outputs/LAM/noirr_2010_2022'
# no_irr_dir='/data/ptiengou'
no_irr_dir='/gpfsstore/rech/ngp/unm64zs/IGCM_OUT/LMDZOR/PROD/amip/NoIrr-IPSLcm6Hist*/ATM/Output/MO'
# no_irr_dir='/gpfswork/rech/cuz/upj17my/wind_tests_python/noirr'

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
time_series_ave(ds_list, var, figsize=(9, 5.5), year_min=1950, year_max=2014, title=None, ylabel=None, xlabel=None)
save_path='figures/time_series_1.png'
plt.savefig(save_path, dpi=300)


heights=['850']
ds_list=[sim_noirr, sim_irr]

print("Plotting winds")
for ds in ds_list:
    for height in heights:
        print("Plotting for height {} in sim {}".format(height, ds.attrs['name']))
        map_wind(ds, height=height, figsize=(15,8), cmap=reds, dist=6, scale=100, hex=False, hex_center=False)
        print("Now saving figure")
        save_path='{}/{}_wind_{}.png'.format(fig_save_dir, ds.attrs['name'], height)
        plt.savefig(save_path, dpi=300)

print("Plotting diff")
for height in heights:
    map_wind_diff(sim_irr,sim_noirr, height=height, figsize=(15,8), cmap=emb, dist=6, scale=10, hex=False, hex_center=False)
    print("Now saving figure")
    save_path='{}/wind_diff_{}.png'.format(fig_save_dir, height)
    plt.savefig(save_path, dpi=300)

print("Seasonnal plots")
noirr_list = seasonnal_ds_list(sim_noirr)
irr_list = seasonnal_ds_list(sim_irr)
total_list = [noirr_list, irr_list]
print("Plotting winds")
for ds_list in total_list:
    for ds in ds_list:
        for height in heights:
            print("Plotting for height {} in sim {}".format(height, ds.attrs['name']))
            map_wind(ds, height=height, figsize=(15,8), cmap=reds, dist=6, scale=100, hex=False, hex_center=False)
            print("Now saving figure")
            save_path='{}/{}_wind_{}_seasonnal.png'.format(fig_save_dir, ds.attrs['name'], height)
            plt.savefig(save_path, dpi=300)

print("Plotting diff")
for i in range(4):
    for height in heights:
        map_wind_diff(irr_list[i],noirr_list[i], height=height, figsize=(15,8), cmap=emb, dist=6, scale=10, hex=False, hex_center=False)
        print("Now saving figure")
        save_path='{}/wind_diff_{}_seasonnal.png'.format(fig_save_dir, i)
        plt.savefig(save_path, dpi=300)