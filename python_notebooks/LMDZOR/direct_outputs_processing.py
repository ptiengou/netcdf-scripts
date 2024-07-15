#import personnal tools
from tools import *

#open histmth files and preprocess
noirr_dir='../../../JZ_simu_outputs/LAM/noirr_2010_2022'
irr_dir='../../../JZ_simu_outputs/LAM/irr_2010_2022'

filename1 = '{}/ATM/TS*.nc'.format(noirr_dir)
sim_noirr = xr.open_mfdataset(filename1)
sim_noirr = sim_noirr.rename({'time_counter':'time'})
sim_noirr.attrs['name'] = 'no_irr'
sim_noirr = sim_noirr.sel(lon=slice(-13,6),lat=slice(32,49))
sim_noirr['evap']=sim_noirr['evap'] * 3600 * 24
sim_noirr['evap'].attrs['units']='mm/d'
sim_noirr['precip']=sim_noirr['precip'] * 3600 * 24
sim_noirr['precip'].attrs['units']='mm/d'
sim_noirr['sens']=-sim_noirr['sens']
sim_noirr['flat']=-sim_noirr['flat']
sim_noirr['P - E'] = sim_noirr['precip'] - sim_noirr['evap']
sim_noirr['P - E'].attrs['units'] = 'mm/d'

filename2 = '{}/ATM/TS*.nc'.format(irr_dir)
sim_irr0 = xr.open_mfdataset(filename2)
sim_irr = sim_irr0.rename({'time_counter':'time'})
sim_irr.attrs['name'] = 'irr'
sim_irr = sim_irr.sel(lon=slice(-13,6),lat=slice(32,49))
sim_irr['evap']=sim_irr['evap'] * 3600 * 24
sim_irr['evap'].attrs['units']='mm/d'
sim_irr['precip']=sim_irr['precip'] * 3600 * 24
sim_irr['precip'].attrs['units']='mm/d'
sim_irr['sens']=-sim_irr['sens']
sim_irr['flat']=-sim_irr['flat']
sim_irr['P - E'] = sim_irr['precip'] - sim_irr['evap']
sim_irr['P - E'].attrs['units'] = 'mm/d'


#plot var for sim
var='P - E'
ds=sim_noirr
#save in figure
fig_path='figures/{}_{}.png'.format(ds.attrs['name'],var)
map_ave(ds, var, hex=True, in_figsize=(10,7.5))
plt.savefig(fig_path, dpi=300)

#plot diff between sims
var='P - E'