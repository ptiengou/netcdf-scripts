import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
import matplotlib.ticker as ticker
from cycler import cycler
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ttest_ind
from scipy.stats import linregress
from scipy.interpolate import griddata
from matplotlib.markers import MarkerStyle

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
                    '#0C5DA5', #blue
                    '#FF2C00', #red
                    '#00B945', #green
                    '#845B97', #purple
                    '#474747', #grey
                    '#9E9E9E', #light grey
                    '#FF9500', #orange/yellow
                ]
            ) * cycler(alpha=[0.8]),
            'scatter.marker': 'x',
            'lines.linewidth': 2.0,
        })
rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',edgecolor=(0, 0, 0, 0.3), facecolor='none')

### Manage colormaps ###
def make_cmap_white(cmap, nbins=10):
    colors = plt.get_cmap(cmap, nbins)(np.linspace(0, 1, nbins))
    colors[0] = [1, 1, 1, 1]  # RGBA for white
    cmapW = ListedColormap(colors[:nbins])
    return cmapW
def add_central_white_cmap(cmap, nbins=11):
    colors = plt.get_cmap(cmap, nbins)(np.linspace(0, 1, nbins))
    colors[5] = [1, 1, 1, 1]  # RGBA for white
    cmapW = ListedColormap(colors[:nbins])
    return cmapW

emb = ListedColormap(mpl.colormaps['RdYlBu_r'](np.linspace(0, 1, 10)))
emb_r = ListedColormap(mpl.colormaps['RdYlBu'](np.linspace(0, 1, 10)))

emb2 = ListedColormap(mpl.colormaps['seismic'](np.linspace(0, 1, 11)))
emb_neutral = ListedColormap(mpl.colormaps['BrBG'](np.linspace(0, 1, 10)))
myvir = ListedColormap(mpl.colormaps['viridis'](np.linspace(0, 1, 10)))
reds = ListedColormap(mpl.colormaps['Reds'](np.linspace(0, 1, 10)))
redsW = make_cmap_white('Reds', 10)
greens = ListedColormap(mpl.colormaps['Greens'](np.linspace(0, 1, 10)))
greensW = make_cmap_white('Greens', 10)
blues = ListedColormap(mpl.colormaps['Blues'](np.linspace(0, 1, 10)))
bluesW = make_cmap_white('Blues', 10)
wet = ListedColormap(mpl.colormaps['YlGnBu'](np.linspace(0, 1, 10)))
wetW = make_cmap_white('YlGnBu', 10)
bad_good=ListedColormap(mpl.colormaps['RdYlGn'](np.linspace(0, 1, 10)))
good_bad=ListedColormap(mpl.colormaps['RdYlGn_r'](np.linspace(0, 1, 10)))
good_badW=add_central_white_cmap('RdYlGn_r', 11)
bad_goodW=add_central_white_cmap('RdYlGn', 11)

### Dataset manipulations ###

#from full dataset to seasonnal ones
def seasonnal_ds(ds, season):
    ds_season = ds.where(ds['time.season']==season, drop=True)
    ds_season.attrs['name'] = '{}_{}'.format(ds.attrs['name'], season)
    return ds_season

def seasonnal_ds_list(ds):
    season_list = ['DJF', 'MAM', 'JJA', 'SON']
    ds_list = [seasonnal_ds(ds, season) for season in season_list]
    return ds_list

#restrict to square subdomain
subdomain_spain={'lonmin':-9.5, 'lonmax':3.0, 'latmin':36.0, 'latmax':44.0}
subdomain_ebro={'lonmin':-2.0, 'lonmax':1.0, 'latmin':41.0, 'latmax':43.0}

def restrict_ds(ds, subdomain):
    # ds = ds.sel(lon=slice(subdomain['lonmin'], subdomain['lonmax']), lat=slice(subdomain['latmin'], subdomain['latmax']))
    ds = ds.where(ds['lon'] >= subdomain['lonmin'], drop=True).where(ds['lon'] <= subdomain['lonmax'], drop=True)
    ds = ds.where(ds['lat'] >= subdomain['latmin'], drop=True).where(ds['lat'] <= subdomain['latmax'], drop=True)
    return ds

#regrid obs data for irrigation
def regrid_to_lon_lat(ds, new_lon_res=500, new_lat_res=500):
    """
    Regrid irrigation data from X, Y coordinates to lon, lat coordinates, keeping the time dimension.
    
    Parameters:
        ds (xarray.Dataset): The input dataset containing 'irrigation', 'lon', 'lat' variables.
        new_lon_res (int): Resolution of the new longitude grid.
        new_lat_res (int): Resolution of the new latitude grid.
    
    Returns:
        xarray.Dataset: A new Dataset containing regridded 'irrigation' on lon/lat grid with time.
    """
    # Extract lon, lat, and irrigation
    lon = ds['lon'].values  # Shape (X, Y)
    lat = ds['lat'].values  # Shape (X, Y)
    irrigation = ds['irrigation'].values  # Shape (time, X, Y)
    time = ds['time'].values  # Time dimension

    # Create a new regular lon/lat grid
    lon_new = np.linspace(lon.min(), lon.max(), new_lon_res)
    lat_new = np.linspace(lat.min(), lat.max(), new_lat_res)
    lon_new_grid, lat_new_grid = np.meshgrid(lon_new, lat_new)

    # Prepare an empty array for regridded irrigation data
    irrigation_new = np.empty((len(time), new_lat_res, new_lon_res))

    # Loop through each time step and interpolate
    for t in range(len(time)):
        irrigation_t = irrigation[t, :, :]  # Get the irrigation data at time t

        # Flatten lon, lat, and irrigation_t for griddata
        points = np.array([lon.flatten(), lat.flatten()]).T
        values = irrigation_t.flatten()

        # Interpolate to the new lon/lat grid
        irrigation_new_t = griddata(points, values, (lon_new_grid, lat_new_grid), method='linear')

        # Store the interpolated values in the new array
        irrigation_new[t, :, :] = irrigation_new_t

    # Step 1: Create a DataArray with time, lat, and lon coordinates
    irrigation_da = xr.DataArray(
        data=irrigation_new, 
        dims=['time', 'lat', 'lon'],  # Keep time, lat, and lon dimensions
        coords={'time': time, 'lat': lat_new, 'lon': lon_new},
        name='irrigation'
    )

    # Step 2: Create a Dataset with the DataArray as a single variable
    irrigation_ds = xr.Dataset({'irrigation': irrigation_da})

    return irrigation_ds

#polygonal subdomain
iberian_peninsula = {
            1 : {'lon':-10.0    , 'lat':36.0 },
            2 : {'lon':-10.0    , 'lat':44.0 },
            3 : {'lon':-2.0     , 'lat':43.3 },
            4 : {'lon':3.3      , 'lat':42.8 },
            5 : {'lon':3.3      , 'lat':41.5 },
            6 : {'lon':-2.0      , 'lat':36.0 },
}

pyrenees = {
            1 : {'lon':-2.0    , 'lat':43.5 },
            2 : {'lon':3.0    , 'lat':43.5 },
            3 : {'lon':3.0     , 'lat':42.2 },
            4 : {'lon':-2.0      , 'lat':42.2 },
}

ebro = {
            1 : {'lon':-2.0    , 'lat':42.5 },
            2 : {'lon':1.2    , 'lat':42.5 },
            3 : {'lon':1.2     , 'lat':41 },
            4 : {'lon':-2.0      , 'lat':41 },
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

def plot_polygon(dict_polygon, ax):
    polygon = np.array([[point['lon'], point['lat']] for point in dict_polygon.values()])
    polygon = np.vstack([polygon, polygon[0]])  # Close the polygon
    ax.plot(polygon[:, 0], polygon[:, 1], 'r-', linewidth=2, c='red')


### time plots ###
months_name_list=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def nice_time_plot(plotvar, ax, label=None, title=None, ylabel=None, xlabel=None, color=None):
    plotvar.plot(ax=ax, label=label, c=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend()
    ax.grid()

def time_series_ave(ds_list, var, ds_colors=False, figsize=(8.5, 5), year_min=2010, year_max=2022, title=None, ylabel=None, xlabel=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.grid()
    if not title:
        title = var + (' ({})'.format(ds_list[0][var].attrs['units']))
    for ds in ds_list:
        plotvar=ds[var].where(ds['time.year'] >= year_min, drop=True).where(ds['time.year'] <= year_max, drop=True).mean(dim=['lon','lat'])
        if ds_colors:
            nice_time_plot(plotvar,ax,label=ds.name, color=ds.attrs["plot_color"], title=title, ylabel=ylabel, xlabel=xlabel)
        else:
            nice_time_plot(plotvar,ax,label=ds.name, title=title, ylabel=ylabel, xlabel=xlabel)
def time_series_lonlat(ds_list, var, lon, lat, figsize=(8.5, 5), year_min=2010, year_max=2022, title=None, ylabel=None, xlabel=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.grid()
    if not title:
        title = var + (' at ({}°N,{}°W) ({})'.format(lon,lat,ds_list[0][var].attrs['units']))
    for ds in ds_list:
        plotvar=ds[var].where(ds['time.year'] >= year_min, drop=True).where(ds['time.year'] <= year_max, drop=True)
        plotvar=plotvar.sel(lon=lon, lat=lat, method='nearest')
        nice_time_plot(plotvar,ax,label=ds.name, title=title, ylabel=ylabel, xlabel=xlabel)

def seasonal_cycle_ave(ds_list, var, figsize=(8.5, 5), ds_colors=False, year_min=2010, year_max=2022, title=None, ylabel=None, xlabel=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    if not title:
        title = var + (' ({})'.format(ds_list[0][var].attrs['units']))
    for ds in ds_list:
        ds = ds.where(ds['time.year'] >= year_min, drop=True).where(ds['time.year'] <= year_max, drop=True)
        mean=ds[var].mean(dim=['lon','lat', 'time']).values
        print(ds.attrs['name'] + ' : %.2f' % mean + ' ({})'.format(ds[var].attrs['units']))
        plotvar=ds[var].mean(dim=['lon', 'lat']).groupby('time.month').mean(dim='time')
        if ds_colors:
            nice_time_plot(plotvar,ax,label=ds.name, color=ds.attrs["plot_color"], title=title, ylabel=ylabel, xlabel=xlabel)
        else:
            nice_time_plot(plotvar,ax,label=ds.name, title=title, ylabel=ylabel, xlabel=xlabel)
    ax.grid()
    ax.set_xticks(np.arange(1,13))
    ax.set_xticklabels(months_name_list)

###scatter plots###
def scatter_vars(ds1, ds2, var1, var2, reg=False, plot_one=False, plot_m_one=False, title=None):
    """
    Plots a scatter plot of values of two variables in an xarray dataset,
    with an optional linear regression line and coefficient.

    Parameters:
    - ds: xarray.Dataset
        The dataset containing the variables.
    - var1: str
        The name of the first variable.
    - var2: str
        The name of the second variable.
    - reg: bool, optional
        Whether to add a linear regression line and display the regression coefficient. Default is False.
    """
    
    # Flatten the grid data for scatter plotting
    x = ds1[var1].values.flatten()
    y = ds2[var2].values.flatten()
    
    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel(f'{var1} ({ds1.attrs["name"]})')
    plt.ylabel(f'{var2} ({ds2.attrs["name"]})')
    plt.grid(True)
    if title:
        plt.title(title)
    else:
        plt.title(f'Scatter plot')

    # Optional linear regression
    if reg:
        # Remove NaN values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        # Perform linear regression
        slope, intercept, r_value, _, _ = linregress(x, y)
        # Plot regression line
        plt.plot(x, slope * x + intercept, color='red', label=f'Regression: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value**2:.2f}')
        plt.legend()
    
    #optional line for 1:1 relationship
    if plot_one:
        plt.plot([x.min(), x.max()], [x.min(), x.max()], color='black', linestyle='--', label='1:1')
        plt.legend() 
    #optional line for 1:-1 relationship
    if plot_m_one:
        plt.plot([x.min(), x.max()], [-x.min(), -x.max()], color='black', linestyle='--', label='1:-1')
        plt.legend()
    plt.show()

def scatter_temporal_mean(ds1, ds2, var1, var2, reg=False):
    """
    Plots a scatter plot of the temporal mean values of two variables in an xarray dataset,
    with an optional linear regression line and coefficient.

    Parameters:
    - ds: xarray.Dataset
        The dataset containing the variables.
    - var1: str
        The name of the first variable.
    - var2: str
        The name of the second variable.
    - reg: bool, optional
        Whether to add a linear regression line and display the regression coefficient. Default is False.
    """
    # Compute the temporal mean for each variable across the time dimension
    var1_mean = ds1[var1].mean(dim='time')
    var2_mean = ds2[var2].mean(dim='time')
    
    # Flatten the grid data for scatter plotting
    x = var1_mean.values.flatten()
    y = var2_mean.values.flatten()
    
    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel(f'{var1} ({ds1.attrs["name"]})')
    plt.ylabel(f'{var2} ({ds2.attrs["name"]})')
    plt.title(f'Scatter plot (1 data point = 1 grid cell)')
    plt.grid(True)
    
    # Optional linear regression
    if reg:
        # Remove NaN values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        # Perform linear regression
        slope, intercept, r_value, _, _ = linregress(x, y)
        # Plot regression line
        plt.plot(x, slope * x + intercept, color='red', label=f'Regression: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value**2:.2f}')
        plt.legend()
    
    plt.show()

#version with specifiable function for temporal aggregation, not working (yet)
def scatter_temporal_aggregate(ds, var1, var2, agg_func="mean"):
    """
    Plots a scatter plot of aggregated temporal values of two variables in an xarray dataset.

    Parameters:
    - ds: xarray.Dataset
        The dataset containing the variables.
    - var1: str
        The name of the first variable.
    - var2: str
        The name of the second variable.
    - agg_func: str or callable, optional
        The aggregation function to apply along the time dimension. It can be a string (like 'mean', 'max', 'min', 'sum') or
        a callable function. Default is 'mean'.
    """
    # Select the aggregation function
    if isinstance(agg_func, str):
        agg_func = getattr(ds[var1], agg_func, None)
        if agg_func is None:
            raise ValueError(f"Aggregation function '{agg_func}' is not supported.")
        var1_agg = agg_func(dim='time')
        var2_agg = getattr(ds[var2], agg_func)(dim='time')
    else:
        var1_agg = ds[var1].reduce(agg_func, dim='time')
        var2_agg = ds[var2].reduce(agg_func, dim='time')

    # Flatten the grid data for scatter plotting
    x = var1_agg.values.flatten()
    y = var2_agg.values.flatten()
    
    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel(f'{var1} (temporal {agg_func.__name__ if callable(agg_func) else agg_func})')
    plt.ylabel(f'{var2} (temporal {agg_func.__name__ if callable(agg_func) else agg_func})')
    plt.title(f'Scatter plot of {var1} vs {var2} (temporal {agg_func.__name__ if callable(agg_func) else agg_func})')
    plt.grid(True)
    plt.show()

### vertical profiles ###

#plot vertical profile from reference level variables
def profile_reflevs(ds_list, var, figsize=(6,8), title=' vertical profile'):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.grid()
    plt.gca().invert_yaxis()
    ax.set_title(var + title)
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
def profile_preslevs(ds_list, var, figsize=(6,8), preslevelmax=20, title=' vertical profile'):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.grid()
    plt.gca().invert_yaxis()
    ax.set_title(var + title)
    ax.set_ylabel('Pressure (hPa)')
    ax.set_xlabel(var + ' ({})'.format(ds_list[0][var].attrs['units']))
    pressure=ds_list[0]['presnivs'][0:preslevelmax]/100 #select levels and convert from Pa to hPa
    for ds in ds_list:
        plotvar = ds[var].mean(dim=['time', 'lon', 'lat'])[0:preslevelmax]
        ax.plot(plotvar, pressure, label=ds.name) 
    ax.legend()
                     
