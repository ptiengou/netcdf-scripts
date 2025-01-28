import psyplot.project as psy
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import xarray as xr

import cartopy
import cartopy.util as cu
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


psy.rcParams['plotter.maps.xgrid'] = False
psy.rcParams['plotter.maps.ygrid'] = False
mpl.rcParams['figure.figsize'] = [10., 8.]


# we show the figures after they are drawn or updated. This is useful for the
# visualization in the ipython notebook
psy.rcParams['auto_show'] = True
sud=True
if sud:
   fichier0='start0_MARCUS.nc'
   lon = 144
   lat = -66
   lon = 62.85
   lat = -67.59
else:
   fichier0='start0_THINICE.nc'
   lon = 20
   lat = 76
os.system('ncap2 -O -s "zz=phis/9.8" '+fichier0+ ' tmp.nc')
fichier0 = 'limit.nc'#_L95_2018.nc'

maps=psy.plot.mapplot('tmp.nc', name='zz', datagrid=dict(color='k', linewidth=0.2), cbar='r',clon=lon,clat=lat,tight=True,bounds=np.linspace(0,2000,21,endpoint=True),lsm='50m',projection='ortho',cmap='terrain',xgrid=True,ygrid=True)

#maps=psy.plot.mapplot(fichier0, name='RUG', datagrid=dict(color='k', linewidth=0.2), cbar='r',clon=lon,clat=lat,tight=True,bounds=np.linspace(-1,1,30),lsm='50m',projection='ortho',cmap='cividis',xgrid=True,ygrid=True)

maps=psy.plot.mapplot(fichier0, name='FSIC', datagrid=dict(color='k', linewidth=0.2), cbar='r',clon=lon,clat=lat,tight=True,bounds=np.linspace(0,1,10),lsm='50m',projection='ortho',cmap='pink',xgrid=True,ygrid=True)

maps=psy.plot.mapplot(fichier0, name='FOCE', datagrid=dict(color='k', linewidth=0.2), cbar='r',clon=lon,clat=lat,tight=True,bounds=np.linspace(0,1,10),lsm='50m',projection='ortho',cmap='Blues',xgrid=True,ygrid=True)

maps=psy.plot.mapplot(fichier0, name='FLIC', datagrid=dict(color='k', linewidth=0.2), cbar='r',clon=lon,clat=lat,tight=True,bounds=np.linspace(0,1,10),lsm='50m',projection='ortho',cmap='bone',xgrid=True,ygrid=True)

maps=psy.plot.mapplot(fichier0, name='FTER', datagrid=dict(color='k', linewidth=0.2), cbar='r',clon=lon,clat=lat,tight=True,bounds=np.linspace(0,1,10),lsm='50m',projection='ortho',cmap='YlOrBr',xgrid=True,ygrid=True)

#maps=psy.plot.mapplot(fichier1, name='t2m', datagrid=None, cbar='r',clon=lon,clat=lat,tight=True,bounds=np.linspace(250,280,16),lsm='50m',projection='ortho',cmap='terrain',xgrid=True,ygrid=True)
#plt.savefig('zz_1800km.png')

#fig, ax = plt.subplots(1,1,figsize=(10,8), subplot_kw={'projection': ccrs.Orthographic(central_longitude=0)})
#ax.plot(ds_mo.lon,ds_mo.lat,color='red',transform=ccrs.Orthographic(central_longitude=0.0, central_latitude=67.0))

#map2=psy.plot.mapplot(fichier,name='phis',cmap='terrain')

plt.show()



