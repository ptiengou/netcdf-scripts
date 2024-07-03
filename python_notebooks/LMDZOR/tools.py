import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import cartopy.crs as ccrs
import cartopy
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from scipy.stats import ttest_ind

emb = ListedColormap(mpl.colormaps['RdYlBu_r'](np.linspace(0, 1, 10)))
myvir = ListedColormap(mpl.colormaps['viridis'](np.linspace(0, 1, 10)))
reds = ListedColormap(mpl.colormaps['Reds'](np.linspace(0, 1, 10)))
greens = ListedColormap(mpl.colormaps['Greens'](np.linspace(0, 1, 10)))
wet = ListedColormap(mpl.colormaps['YlGnBu'](np.linspace(0, 1, 10)))

rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',edgecolor=(0, 0, 0, 0.3), facecolor='none')

def map_plotvar(plotvar, in_vmin=None, in_vmax=None, in_cmap=myvir, multiplier=1, in_figsize=(9,6.5), in_title=None):
    fig = plt.figure(figsize=in_figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    _nice_map(plotvar, ax, in_cmap, in_vmin, in_vmax)
    plt.title(in_title)

def map_ave(ds, var, in_vmin=None, in_vmax=None, in_cmap=myvir, multiplier=1, in_figsize=(9,6.5)):
    fig = plt.figure(figsize=in_figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    plotvar = ds[var].mean(dim='time') * multiplier
    _nice_map(plotvar, ax, in_cmap=in_cmap, in_vmin=in_vmin, in_vmax=in_vmax)
    plt.title(var + ' (' + ds[var].attrs['units'] + ')')

def map_diff_ave(ds1, ds2, var, in_vmin=None, in_vmax=None, in_cmap=emb, in_figsize=(9,6.5), sig=False):
    fig = plt.figure(figsize=in_figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    diff = (ds1[var]-ds2[var]).mean(dim='time')

    if sig:
        p_values = xr.apply_ufunc(
            lambda x, y: ttest_ind(x, y, axis=0, nan_policy='omit').pvalue, 
            ds1[var], ds2[var],
            input_core_dims=[['time'], ['time']],
            output_core_dims=[[]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float]
        )
        mask=p_values<0.05
        diff=diff.where(mask)
        
    _nice_map(diff, ax, in_cmap=in_cmap, in_vmin=in_vmin, in_vmax=in_vmax)
    plt.title(var + ' difference (' + ds1.name + ' - ' + ds2.name + ', ' + ds1[var].attrs['units'] + ')')

def map_rel_diff_ave(ds1, ds2, var, in_vmin=None, in_vmax=None, in_cmap=emb, multiplier=1, in_figsize=(9,6.5)):
    fig = plt.figure(figsize=in_figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    rel_diff = ((ds1[var]-ds2[var] + 1E-16) / (ds2[var] + 1E-16)).mean(dim='time') * 100

    _nice_map(rel_diff, ax, in_cmap=in_cmap, in_vmin=in_vmin, in_vmax=in_vmax)
    plt.title(var + ' relative difference (' + ds1.name + ' - ' + ds2.name + ' ; %)')

def map_two_ds(ds1, ds2, var, in_vmin=None, in_vmax=None, in_cmap=reds, in_figsize=(9,6.5)):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(var + ' ({})'.format(ds1[var].units))
    plotvar_1 = ds1[var].mean(dim='time')
    plotvar_2 = ds2[var].mean(dim='time')
    _nice_map(plotvar_1, axs[0], in_cmap, in_vmin, in_vmax)
    _nice_map(plotvar_2, axs[1], in_cmap, in_vmin, in_vmax)

#internal functions
def _nice_map(plotvar, in_ax, in_cmap, in_vmin, in_vmax):
    in_ax.coastlines()
    in_ax.add_feature(rivers)
    gl = in_ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.8)
    gl.right_labels = False
    gl.top_labels = False
    gl.xlocator = plt.MaxNLocator(10)
    gl.ylocator = plt.MaxNLocator(9)
    plotvar.plot(ax=in_ax, transform=ccrs.PlateCarree(), cmap=in_cmap, vmin=in_vmin, vmax=in_vmax)
    plt.tight_layout()
