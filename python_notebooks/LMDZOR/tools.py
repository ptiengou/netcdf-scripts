import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
from cycler import cycler
import cartopy.crs as ccrs
import cartopy
from scipy.stats import ttest_ind

plt.rcParams.update(
        {
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 12,
            'figure.dpi': 72.0,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 5.0,
            'xtick.minor.size': 2.5,
            'ytick.major.size': 5.0,
            'ytick.minor.size': 2.5,
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
            'axes.grid': True,
            'axes.titlesize': 'larger',
            'axes.labelsize': 'larger',
            'grid.color': 'dimgray',
            'grid.linestyle': '-',
            'grid.alpha': 0.3,
            'axes.prop_cycle': cycler(
                color=[
                    '#0C5DA5',
                    '#FF9500',
                    '#00B945',
                    '#FF2C00',
                    '#845B97',
                    '#474747',
                    '#9E9E9E',
                ]
            ) * cycler(alpha=[0.8]),
            'scatter.marker': 'x',
            'lines.linewidth': 1.0,
        })

emb = ListedColormap(mpl.colormaps['RdYlBu_r'](np.linspace(0, 1, 10)))
myvir = ListedColormap(mpl.colormaps['viridis'](np.linspace(0, 1, 10)))
reds = ListedColormap(mpl.colormaps['Reds'](np.linspace(0, 1, 10)))
greens = ListedColormap(mpl.colormaps['Greens'](np.linspace(0, 1, 10)))
wet = ListedColormap(mpl.colormaps['YlGnBu'](np.linspace(0, 1, 10)))
emb_neutral = ListedColormap(mpl.colormaps['BrBG_r'](np.linspace(0, 1, 16)))

rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',edgecolor=(0, 0, 0, 0.3), facecolor='none')

#maps 
def nice_map(plotvar, in_ax, in_cmap=myvir, in_vmin=None, in_vmax=None):
    in_ax.coastlines()
    in_ax.add_feature(rivers)
    gl = in_ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.8)
    gl.right_labels = False
    gl.top_labels = False
    gl.xlocator = plt.MaxNLocator(10)
    gl.ylocator = plt.MaxNLocator(9)
    plot_obj = plotvar.plot(ax=in_ax, transform=ccrs.PlateCarree(), cmap=in_cmap, vmin=in_vmin, vmax=in_vmax)
    #remove legend on the cmap bar
    plot_obj.colorbar.set_label('')
    plt.tight_layout()

def map_plotvar(plotvar, in_vmin=None, in_vmax=None, in_cmap=myvir, in_figsize=(9,6.5), in_title=None, hex=False, hex_center=False):
    fig = plt.figure(figsize=in_figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    nice_map(plotvar, ax, in_cmap, in_vmin, in_vmax)
    if hex:
        plot_hexagon(ax, show_center=hex_center)
    plt.title(in_title)

def map_ave(ds, var, in_vmin=None, in_vmax=None, in_cmap=myvir, multiplier=1, in_figsize=(9,6.5), hex=False, hex_center=False):
    fig = plt.figure(figsize=in_figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    plotvar = ds[var].mean(dim='time') * multiplier
    nice_map(plotvar, ax, in_cmap=in_cmap, in_vmin=in_vmin, in_vmax=in_vmax)
    if hex:
        plot_hexagon(ax, show_center=hex_center)
    plt.title(var + ' (' + ds[var].attrs['units'] + ')')

def map_diff_ave(ds1, ds2, var, in_vmin=None, in_vmax=None, in_cmap=emb, in_figsize=(9,6.5), sig=False, hex=False, hex_center=False):
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
        
    nice_map(diff, ax, in_cmap=in_cmap, in_vmin=in_vmin, in_vmax=in_vmax)
    if hex:
        plot_hexagon(ax, show_center=hex_center)
    plt.title(var + ' difference (' + ds1.name + ' - ' + ds2.name + ', ' + ds1[var].attrs['units'] + ')')

def map_rel_diff_ave(ds1, ds2, var, in_vmin=None, in_vmax=None, in_cmap=emb, multiplier=1, in_figsize=(9,6.5), hex=False, hex_center=False):
    fig = plt.figure(figsize=in_figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    rel_diff = ((ds1[var]-ds2[var] + 1E-16) / (ds2[var] + 1E-16)).mean(dim='time') * 100

    nice_map(rel_diff, ax, in_cmap=in_cmap, in_vmin=in_vmin, in_vmax=in_vmax)
    if hex:
        plot_hexagon(ax, show_center=hex_center)
    plt.title(var + ' relative difference (' + ds1.name + ' - ' + ds2.name + ' ; %)')

def map_two_ds(ds1, ds2, var, in_vmin=None, in_vmax=None, in_cmap=reds, in_figsize=(15,6), hex=False, hex_center=False):
    fig, axs = plt.subplots(1, 2, figsize=in_figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(var + ' ({})'.format(ds1[var].units))
    plotvar_1 = ds1[var].mean(dim='time')
    plotvar_2 = ds2[var].mean(dim='time')
    nice_map(plotvar_1, axs[0], in_cmap, in_vmin, in_vmax)
    axs[0].set_title(ds1.name)
    if hex:
        plot_hexagon(axs[0], show_center=hex_center)
    nice_map(plotvar_2, axs[1], in_cmap, in_vmin, in_vmax)
    axs[1].set_title(ds2.name)
    if hex:
        plot_hexagon(axs[1], show_center=hex_center)

def map_seasons(plotvar, in_vmin=None, in_vmax=None, in_cmap=myvir, in_figsize=(12,9), hex=False, hex_center=False, in_title=None):
    fig, axs = plt.subplots(2, 2, figsize=in_figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(in_title)
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    for i, season in enumerate(seasons):
        plotvar_season = plotvar.where(plotvar['time.season']==season, drop=True)
        nice_map(plotvar_season.mean(dim='time'), axs.flatten()[i], in_cmap, in_vmin, in_vmax)
        axs.flatten()[i].set_title(season)
        if hex:
            plot_hexagon(axs.flatten()[i], show_center=hex_center)

def map_wind(ds, height='10m', in_figsize=(9,6.5), in_cmap=reds, dist=6, in_scale=100, hex=False, hex_center=False):
    fig = plt.figure(figsize=in_figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    windvar_u=ds['u'+height].mean(dim='time')
    windvar_v=ds['v'+height].mean(dim='time')
    #show wind speed in background color
    wind_speed = (windvar_u**2 + windvar_v**2 ) ** (1/2)
    nice_map(wind_speed, ax, in_cmap=in_cmap)
    if hex:
        plot_hexagon(ax, show_center=hex_center)
    #plot wind vectors
    windx = windvar_u[::dist,::dist]
    windy = windvar_v[::dist,::dist]
    longi=ds['lon'][::dist]
    lati=ds['lat'][::dist]
    quiver = ax.quiver(longi, lati, windx, windy, transform=ccrs.PlateCarree(), scale=in_scale)

    quiverkey_scale = in_scale/10
    plt.quiverkey(quiver, X=0.95, Y=0.05, U=quiverkey_scale, label='{} m/s'.format(quiverkey_scale), labelpos='S')

    if (height == '10m'):
        plt.title('10m wind speed (m/s) and direction')
    else :
        plt.title('{} hPa wind speed (m/s) and direction'.format(height))

def map_wind_diff(ds1, ds2, height='10m', in_figsize=(9,6.5), in_cmap=emb, dist=6, in_scale=100, hex=False, hex_center=False):
    fig = plt.figure(figsize=in_figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    windvar_u1=ds1['u'+height].mean(dim='time')
    windvar_v1=ds1['v'+height].mean(dim='time')
    windvar_u2=ds2['u'+height].mean(dim='time')
    windvar_v2=ds2['v'+height].mean(dim='time')
    #show wind speed in background color
    wind_speed1 = (windvar_u1**2 + windvar_v1**2 ) ** (1/2)
    wind_speed2 = (windvar_u2**2 + windvar_v2**2 ) ** (1/2)
    wind_speed_diff = wind_speed1 - wind_speed2
    nice_map(wind_speed_diff, ax, in_cmap=in_cmap)
    if hex:
        plot_hexagon(ax, show_center=hex_center)
    #plot wind vectors
    windx = (windvar_u1-windvar_u2)[::dist,::dist]
    windy = (windvar_v1-windvar_v2)[::dist,::dist]
    longi=ds1['lon'][::dist]
    lati=ds1['lat'][::dist]
    quiver = ax.quiver(longi, lati, windx, windy, transform=ccrs.PlateCarree(), scale=in_scale)

    quiverkey_scale = in_scale/10
    plt.quiverkey(quiver, X=0.95, Y=0.05, U=quiverkey_scale, label='{} m/s'.format(quiverkey_scale), labelpos='S')

    if (height == '10m'):
        plt.title('10m wind speed (m/s) and direction change ({} - {})'.format(ds1.name, ds2.name))
    else :
        plt.title('{} hPa wind speed (m/s) and direction change ({} - {})'.format(height, ds1.name, ds2.name))


#time series
def time_series(ds_list, var, in_figsize=(9, 5.5), year_min=2010, year_max=2022, in_title=None, in_ylabel=None, in_xlabel=None):
    fig = plt.figure(figsize=in_figsize)
    ax = plt.axes()
    for ds in ds_list:
        ds = ds.where(ds['time.year'] >= year_min, drop=True).where(ds['time.year'] <= year_max, drop=True)
        ds[var].mean(dim=['lon', 'lat']).plot(ax=ax, label=ds.name)
    ax.grid()
    ax.set_title(in_title)
    ax.set_ylabel(in_ylabel)
    ax.set_xlabel(in_xlabel)
    ax.legend()

def seasonal_cycle(ds_list, var, in_figsize=(9, 5.5), year_min=2010, year_max=2022, in_title=None, in_ylabel=None, in_xlabel=None):
    fig = plt.figure(figsize=in_figsize)
    ax = plt.axes()
    for ds in ds_list:
        ds = ds.where(ds['time.year'] >= year_min, drop=True).where(ds['time.year'] <= year_max, drop=True)
        mean=ds[var].mean(dim=['lon','lat', 'time']).values
        print(ds.attrs['name'] + ' : %.2f' % mean + ' ({})'.format(ds[var].attrs['units']))
        ds[var].mean(dim=['lon', 'lat']).groupby('time.month').mean(dim='time').plot(ax=ax, label=ds.name)
    ax.grid()
    ax.set_title(in_title)
    ax.set_ylabel(in_ylabel)
    ax.set_xlabel(in_xlabel)
    ax.legend()

    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    ax.set_xticks(np.arange(1,13))
    ax.set_xticklabels(months)

#hexagon
def _destination_point(lon, lat, bearing, distance_km):
    # function to calculate destination point given start point, bearing, and distance
    R = 6371.0  # Earth radius in kilometers
    bearing = np.radians(bearing)
    
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    
    lat2 = np.arcsin(np.sin(lat1) * np.cos(distance_km / R) +
                     np.cos(lat1) * np.sin(distance_km / R) * np.cos(bearing))
    
    lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(distance_km / R) * np.cos(lat1),
                             np.cos(distance_km / R) - np.sin(lat1) * np.sin(lat2))
    
    return np.degrees(lon2), np.degrees(lat2)

def plot_hexagon(ax, center_lon=-3.7, center_lat=40.43, sim_radius_km=825, nbp=40, forced=5, nudged=8, show_center=False):
    no_influence_portion = (nbp - forced - nudged) / (nbp - forced)
    no_influence_radius_km = sim_radius_km * no_influence_portion
    # no_influence_radius_km = 850

    # Calculate the vertices of the hexagon
    angles = np.linspace(0, 360, 6, endpoint=False)
    hexagon_lons = []
    hexagon_lats = []
    for angle in angles:
        lon, lat = _destination_point(center_lon, center_lat, angle, no_influence_radius_km)
        hexagon_lons.append(lon)
        hexagon_lats.append(lat)

    # Add the hexagon to the plot
    hexagon_points = np.column_stack((hexagon_lons, hexagon_lats))
    ax.plot(*hexagon_points.T, marker='.', color='red', markersize=3, transform=ccrs.PlateCarree())
    # Connect the last point to the first
    ax.plot([hexagon_points[-1, 0], hexagon_points[0, 0]], [hexagon_points[-1, 1], hexagon_points[0, 1]], marker='.', color='red', markersize=3, transform=ccrs.PlateCarree())
    #Show center if appropriate
    if show_center:
        ax.plot(center_lon, center_lat, marker='.', color='red', markersize=8)

#polygonal subdomain
iberic_peninsula = {
            1 : {'lon':-10.0    , 'lat':36.0 },
            2 : {'lon':-10.0    , 'lat':44.0 },
            3 : {'lon':-1.5     , 'lat':44.0 },
            4 : {'lon':3.3      , 'lat':43.0 },
            5 : {'lon':3.3      , 'lat':41.5 },
            6 : {'lon':-2.0      , 'lat':36.0 },
}

def polygon_to_mask(ds, dict_polygon):
    polygon = np.array([[point['lon'], point['lat']] for point in dict_polygon.values()])
    polygon = np.vstack([polygon, polygon[0]])  # Close the polygon
    # plt.plot(polygon[:, 0], polygon[:, 1], 'r-', linewidth=2, c='black')

    # Create a Path object from the polygon
    polygon_path = Path(polygon)

    # Get the coordinates from the dataset
    lons = ds['lon'].values
    lats = ds['lat'].values

    # Create a meshgrid of coordinates
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Create a 2D array of coordinates
    lon_lat_pairs = np.vstack((lon_grid.ravel(), lat_grid.ravel())).T

    # Check which points are inside the polygon
    inside_polygon = polygon_path.contains_points(lon_lat_pairs).reshape(lon_grid.shape)

    # Convert the mask to a DataArray with the same coordinates as the dataset
    inside_polygon_da = xr.DataArray(inside_polygon, dims=['lat', 'lon'], coords={'lat': ds['lat'], 'lon': ds['lon']})

    # Create a dataset for the mask
    mask_ds = xr.Dataset({'mask': inside_polygon_da})
    return(mask_ds)


#vertical profiles
#plot vertical profile from reference level variables
def profile_reflevs(ds_list, var, in_figsize=(6,8), in_title=' vertical profile'):
    fig = plt.figure(figsize=in_figsize)
    ax = plt.axes()
    ax.grid()
    plt.gca().invert_yaxis()
    ax.set_title(var + in_title)
    ax.set_ylabel('Pressure (hPa)')
    ax.set_xlabel(var + ' ({})'.format(ds_list[0][var+'500'].attrs['units']))
    pressure=[200,500,700,850,1013]
    for ds in ds_list:
        var0 = ds[var+'2m'].mean(dim=['time', 'lon', 'lat'])
        var200 = ds[var+'200'].mean(dim=['time', 'lon', 'lat'])
        var500 = ds[var+'500'].mean(dim=['time', 'lon', 'lat'])
        var700 = ds[var+'700'].mean(dim=['time', 'lon', 'lat'])
        var850 = ds[var+'850'].mean(dim=['time', 'lon', 'lat'])
        profile=[var200,var500,var700,var850,var0]
        ax.plot(profile, pressure, label=ds.name) 
    ax.legend()

#plot a profile from variable with all pressure levels
def profile_preslevs(ds_list, var, in_figsize=(6,8), preslevelmax=20, in_title=' vertical profile'):
    fig = plt.figure(figsize=in_figsize)
    ax = plt.axes()
    ax.grid()
    plt.gca().invert_yaxis()
    ax.set_title(var + in_title)
    ax.set_ylabel('Pressure (hPa)')
    ax.set_xlabel(var + ' ({})'.format(ds_list[0][var].attrs['units']))
    pressure=ds_list[0]['presnivs'][0:preslevelmax]/100 #select levels and convert from Pa to hPa
    for ds in ds_list:
        plotvar = ds[var].mean(dim=['time', 'lon', 'lat'])[0:preslevelmax]
        ax.plot(plotvar, pressure, label=ds.name) 
    ax.legend()
                     