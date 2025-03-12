import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
import matplotlib.ticker as ticker
from cycler import cycler
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ttest_ind
import scipy.stats as stats
from scipy.stats import linregress
from scipy.interpolate import griddata
from matplotlib.markers import MarkerStyle
import json
from pprint import pprint
import matplotlib.gridspec as gridspec
from io import StringIO
import psyplot.project as psy



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
# rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',edgecolor=(0, 0, 0, 0.3), facecolor='none')
letters=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
         
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
emb2 = ListedColormap(mpl.colormaps['seismic'](np.linspace(0, 1, 10)))
emb2_r = ListedColormap(mpl.colormaps['seismic_r'](np.linspace(0, 1, 10)))
emb3   = ListedColormap(mpl.colormaps['Spectral_r'](np.linspace(0, 1, 10)))
emb3_r = ListedColormap(mpl.colormaps['Spectral'](np.linspace(0, 1, 10)))
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
myterrain=ListedColormap(mpl.colormaps['terrain'](np.linspace(0, 1, 10)))

### Dataset manipulations ###

#from full dataset to seasonnal ones
def seasonnal_ds(ds, season):
    ds_season = ds.where(ds['time.season']==season, drop=True)
    ds_season.attrs['name'] = '{}_{}'.format(ds.attrs['name'], season)
    return ds_season

def seasonnal_ds_list(ds):
    seasons_list = ['DJF', 'MAM', 'JJA', 'SON']
    ds_list = [seasonnal_ds(ds, season) for season in seasons_list]
    return ds_list

#temporal average
def mean_dataset(ds):
    ds_mean = ds.mean(dim='time')
    ds_mean.attrs = ds.attrs
    # ds_mean.attrs['name'] = '{}'.format(ds.attrs['name'])
    # ds_mean.attrs['plot_color'] = ds.attrs['plot_color']
    return ds_mean

#spatial average
def aggr_dataset(ds):
    ds_aggr = ds.mean(dim='lon').mean(dim='lat')
    ds_aggr.attrs = ds.attrs
    # ds_aggr.attrs['name'] = '{}'.format(ds.attrs['name'])
    # ds_aggr.attrs['plot_color'] = ds.attrs['plot_color']
    return ds_aggr

#diff between two ds
def diff_dataset(ds1, ds2):
    ds_diff = ds1 - ds2
    for var in ds1.var():
        ds_diff[var].attrs = ds1[var].attrs
    ds_diff.attrs['name'] = 'Diff ({} - {})'.format(ds1.attrs['name'], ds2.attrs['name'])
    diff_mean = mean_dataset(ds_diff)
    return (ds_diff, diff_mean)

#filtering
def filter_xarray_by_day(ds: xr.Dataset, day: str) -> xr.Dataset:
    """
    Filters an xarray Dataset to keep only data from the specified day.
    
    Parameters:
        ds (xr.Dataset): Input dataset containing a time coordinate.
        day (str): Date string in the format 'YYYY-MM-DD'.
    
    Returns:
        xr.Dataset: Filtered dataset containing only data from the specified day,
                    with all original attributes preserved.
    """
    day = str(day)  # Ensure it's a string
    filtered_ds = ds.sel(time=slice(day, day))  # Select data for the specific day
    
    # Preserve attributes
    filtered_ds.attrs = ds.attrs  # Preserve dataset attributes
    for var in filtered_ds.data_vars:
        filtered_ds[var].attrs = ds[var].attrs  # Preserve variable attributes
    
    return filtered_ds

def filter_xarray_by_timestamps(ds: xr.Dataset, start_time: str, end_time: str) -> xr.Dataset:
    """
    Filters an xarray Dataset to keep only data between the specified timestamps.

    Parameters:
        ds (xr.Dataset): Input dataset containing a time coordinate.
        start_time (str): Start timestamp string in the format 'YYYY-MM-DD HH:MM:SS'.
        end_time (str): End timestamp string in the format 'YYYY-MM-DD HH:MM:SS'.

    Returns:
        xr.Dataset: Filtered dataset containing only data between the specified timestamps,
                    with all original attributes preserved.
    """
    # Ensure timestamps are strings
    start_time = str(start_time)
    end_time = str(end_time)

    # Select data between the two timestamps
    filtered_ds = ds.sel(time=slice(start_time, end_time))

    # Preserve attributes
    filtered_ds.attrs = ds.attrs  # Preserve dataset attributes
    for var in filtered_ds.data_vars:
        filtered_ds[var].attrs = ds[var].attrs  # Preserve variable attributes

    return filtered_ds

def convert_mm_per_month_to_mm_per_day(data):
    """
    Convert xarray DataArray from mm/month to mm/day using the number of days in each month.
    
    Parameters:
    - data: xarray.DataArray with time dimension (monthly data in mm/month)
    
    Returns:
    - xarray.DataArray with data in mm/day
    """
    # Ensure the time dimension exists
    if "time" not in data.dims:
        raise ValueError("The data must have a 'time' dimension.")
    
    # Get the number of days in each month
    days_in_month = data["time"].dt.days_in_month
    
    # Convert mm/month to mm/day
    data_mm_per_day = data / days_in_month
    
    return data_mm_per_day

def build_stats_df(datasets, variables):
    """
    Build a Pandas DataFrame where rows are datasets, columns are variables, 
    and subcolumns for each variable are mean, std, min, and max.

    Parameters:
        datasets (list): A list of xarray datasets.
        variables (list): A list of variable names to extract statistics for.

    Returns:
        pd.DataFrame: A DataFrame with hierarchical columns for each variable 
                      and subcolumns for mean, std, min, and max.
    """
    # Prepare an empty dictionary to store statistics for each dataset and variable
    data = []
    dataset_names = [ds.attrs['name'] for ds in datasets]
    
    # Loop through each dataset
    for idx, dataset in enumerate(datasets):
        row = {}
        # Loop through each variable and calculate statistics
        for variable in variables:
            if variable in dataset:
                var_data = dataset[variable]
                row[(variable, 'mean')] = var_data.mean(dim=['lon','lat']).values
                row[(variable, 'std')] = var_data.std(dim=['lon','lat']).values
                row[(variable, 'min')] = var_data.min(dim=['lon','lat']).values
                row[(variable, 'max')] = var_data.max(dim=['lon','lat']).values
            else:
                # Add NaN if the variable is missing
                row[(variable, 'mean')] = float('nan')
                row[(variable, 'std')] = float('nan')
                row[(variable, 'min')] = float('nan')
                row[(variable, 'max')] = float('nan')
        data.append(row)
    
    # Convert list of rows into a DataFrame
    df = pd.DataFrame(data, index=dataset_names)
    
    # Set hierarchical columns
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    
    return df

#add wind variables from individual components
def add_wind_speed(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the wind speed from zonal and meridional wind components
    and add it to the dataset.

    Parameters:
        ds (xr.Dataset): Input dataset containing 'vitu', 'vitv' variables.

    Returns:
        xr.Dataset: Updated dataset with the wind speed added as a new variable.
    """
    # Extract the wind components
    vitu = ds['vitu']
    vitv = ds['vitv']

    # Calculate the wind speed as the Euclidean norm
    wind_speed = np.sqrt(vitu**2 + vitv**2)

    # Set attributes for the wind speed DataArray
    wind_speed.attrs['long_name'] = 'Wind Speed'
    wind_speed.attrs['units'] = 'm/s'

    # Add the wind speed to the dataset
    ds['wind_speed'] = wind_speed

    return ds

def add_wind_10m(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the wind speed from zonal and meridional wind components
    and add it to the dataset as well as direction

    Parameters:
        ds (xr.Dataset): Input dataset 

    Returns:
        xr.Dataset: Updated dataset with the wind speed added as a new variable.
    """
    # Extract the wind components
    u = ds['u10m']
    v = ds['v10m']

    # Calculate the wind speed as the Euclidean norm
    wind_speed_10m = np.sqrt(u**2 + v**2)

    # Set attributes for the wind speed DataArray
    wind_speed_10m.attrs['long_name'] = 'Wind speed at 10m'
    wind_speed_10m.attrs['units'] = 'm/s'

    # Add the wind speed to the dataset
    ds['wind_speed_10m'] = wind_speed_10m

    # Calculate the wind direction
    wind_direction_10m = np.arctan2(u, v)  
    # Calculate angle in radians
    wind_direction_10m = np.degrees(wind_direction_10m)  # Convert to degrees
    wind_direction_10m = (wind_direction_10m + 180) % 360  # Convert to meteorological convention (0° is north)

    # Set attributes for the wind direction DataArray
    wind_direction_10m.attrs['long_name'] = 'Wind direction at 10m'
    wind_direction_10m.attrs['units'] = 'degrees'

    # Add the wind direction to the dataset
    ds['wind_direction_10m'] = wind_direction_10m

    return ds

def add_wind_direction(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the wind direction from zonal and meridional wind components
    and add it to the dataset.

    Parameters:
        ds (xr.Dataset): Input dataset containing 'vitu' and 'vitv' variables.

    Returns:
        xr.Dataset: Updated dataset with the wind direction added as a new variable.
    """
    # Extract the wind components
    vitu = ds['vitu']
    vitv = ds['vitv']

    # Calculate the wind direction
    wind_direction = np.arctan2(vitu, vitv)  # Calculate angle in radians
    wind_direction = np.degrees(wind_direction)  # Convert to degrees
    wind_direction = (wind_direction + 180) % 360  # Convert to meteorological convention (0° is north)

    # Set attributes for the wind direction DataArray
    wind_direction.attrs['long_name'] = 'Wind Direction'
    wind_direction.attrs['units'] = 'degrees'

    # Add the wind direction to the dataset
    ds['wind_direction'] = wind_direction

    return ds

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

def apply_2Dmask_to_dataset(dataset: xr.Dataset, mask: xr.DataArray, dsname=None) -> xr.Dataset:
    """
    Apply a 2D spatial mask (lat, lon) to an xarray.Dataset, preserving the structure
    of each variable in the dataset.

    Parameters:
        dataset (xr.Dataset): The dataset to mask.
        mask (xr.DataArray): A 2D mask with dimensions ('lat', 'lon').
                             Values should be 1 for valid data and 0 for masked data.

    Returns:
        xr.Dataset: The masked dataset, retaining the structure of all variables.
    """
    # Ensure the mask has 'lat' and 'lon' dimensions
    if not {'lat', 'lon'}.issubset(mask.dims):
        raise ValueError("The mask must have 'lat' and 'lon' dimensions.")
    
    # Create a copy of the dataset to ensure the structure is preserved
    masked_dataset = dataset.copy()

    # Apply the mask to each variable individually
    for var_name, var_data in dataset.data_vars.items():
        # Check if the variable has lat and lon dimensions
        if {'lat', 'lon'}.issubset(var_data.dims):
            # Align the mask to the variable's dimensions
            aligned_mask = mask.broadcast_like(var_data)
            # Apply the mask without affecting other dimensions
            masked_dataset[var_name] = var_data.where(aligned_mask)

    if not dsname:
        input_name=dataset.attrs['name']
        mask_name=mask.attrs['name']
        masked_dataset.attrs['name'] = '{}_{}'.format(mask_name,input_name)
    else:
        masked_dataset.attrs['name'] = dsname
    return masked_dataset

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
    #choose name of variable
    inside_polygon_da.name = 'mask'
    
    return(inside_polygon_da)

def plot_polygon(dict_polygon, ax):
    polygon = np.array([[point['lon'], point['lat']] for point in dict_polygon.values()])
    polygon = np.vstack([polygon, polygon[0]])  # Close the polygon
    ax.plot(polygon[:, 0], polygon[:, 1], 'r-', linewidth=2, c='red')

def polygon_area(ds, poly):
    """
    Calculate the area of a polygon defined by a dictionary of coordinates
    """
    mask=polygon_to_mask(ds, poly)
    mask.attrs['name'] = 'mask'
    polygon_ds=apply_2Dmask_to_dataset(ds, mask)
    area = (polygon_ds['cell_height'] * polygon_ds['cell_width']).sum()
    area.attrs['units'] = 'm²'
    print('Area : ',area.values, area.attrs['units'])
    return area.values

def compute_mask_area(ds, mask):
    """
    Calculate the area of a mask
    """
    mask.attrs['name'] = 'mask'
    masked_ds=apply_2Dmask_to_dataset(ds, mask).mean(dim='time')
    area = (masked_ds['cell_height'] * masked_ds['cell_width']).sum()
    area.attrs['units'] = 'm²'
    print('Area : ',area.values, area.attrs['units'])
    return area.values

def compute_grid_cell_width(dataset, lon_name="lon", lat_name="lat"):
    """
    Compute the width (east-west distance) of each grid cell in meters and add it as a variable to the dataset.

    Parameters:
        dataset (xarray.Dataset): The input dataset with longitude and latitude dimensions.
        lon_name (str): Name of the longitude dimension in the dataset. Default is "lon".
        lat_name (str): Name of the latitude dimension in the dataset. Default is "lat".
    
    Returns:
        xarray.Dataset: The input dataset with a new field "cell_width" added.
    """
    # Extract latitudes and longitudes
    lon = dataset[lon_name]
    lat = dataset[lat_name]

    # Calculate differences between adjacent grid points in longitude (in radians)
    dlon = lon.diff(dim=lon_name)
    dlat = lat.diff(dim=lat_name)

    #add on the last value to have the same height
    dlon = xr.concat([dlon, dlon.isel({lon_name:-1})], dim=lon_name)
    dlat = xr.concat([dlat, dlat.isel({lat_name:-1})], dim=lat_name)

    width = 111.32 * 1000 * np.cos(np.radians(lat)) * np.abs(dlon)  # in meters
    height = 111.32 * 1000 * np.abs(dlat) * (0.0 * lon +1.0)  # in meters
    # Add the width as a new variable to the dataset
    width_expanded = xr.DataArray(
        width,
        dims=[lat_name,lon_name],  # The dimensions should match the dataset's dimensions
        coords={ lon_name: lon, lat_name: lat}  # Correctly align the coordinates
    )
    height_expanded = xr.DataArray(
        height,
        dims=[lat_name,lon_name],  # The dimensions should match the dataset's dimensions
        coords={ lon_name: lon, lat_name: lat}  # Correctly align the coordinates
    )

    # Assign the calculated width as a new variable in the dataset
    dataset = dataset.assign(cell_width=width_expanded)
    dataset['cell_width'].attrs['units'] = 'm'
    dataset['cell_width'].attrs['name'] = 'grid cell width'

    dataset = dataset.assign(cell_height=height_expanded)
    dataset['cell_height'].attrs['units'] = 'm'
    dataset['cell_height'].attrs['name'] = 'grid cell height'

    dataset["manual_area"] = dataset["cell_width"] * dataset["cell_height"]
    dataset["manual_area"].attrs["units"] = "m²"
    dataset["manual_area"].attrs["name"] = "Grid cell area"


    dataset['cell_width']  = dataset['cell_width'].expand_dims(time=dataset.time)
    dataset['cell_height'] = dataset['cell_height'].expand_dims(time=dataset.time)
    dataset["manual_area"] = dataset["manual_area"].expand_dims(time=dataset.time)

    return dataset

def polygon_edge(ds, polygon):
    """
    Create a mask for the edges of a polygon on the grid of a dataset.

    Parameters:
    - ds (xr.Dataset): The dataset with `lon` and `lat` coordinates.
    - polygon (dict): Dictionary with vertices of the polygon, defined as:
                      {1: {'lon': lon1, 'lat': lat1}, 2: {'lon': lon2, 'lat': lat2}, ...}

    Returns:
    - xr.DataArray: A boolean mask with the same shape as the dataset grid, where
                    True indicates the points that lie on the polygon edges.
    """
    lons = [polygon[key]['lon'] for key in sorted(polygon)]
    lats = [polygon[key]['lat'] for key in sorted(polygon)]
    if (lons[0], lats[0]) != (lons[-1], lats[-1]):
        lons.append(lons[0])
        lats.append(lats[0])
    mask = xr.DataArray(np.zeros((len(ds['lat']), len(ds['lon'])), dtype=bool), coords=[ds['lat'], ds['lon']], dims=['lat', 'lon'])
    edges = zip(lons[:-1], lats[:-1], lons[1:], lats[1:])
    for lon1, lat1, lon2, lat2 in edges:
        num_samples = max(int(np.hypot(lon2 - lon1, lat2 - lat1) * 100), 2)
        lons_edge = np.linspace(lon1, lon2, num_samples)
        lats_edge = np.linspace(lat1, lat2, num_samples)
        lon_indices = ds['lon'].to_numpy()
        lat_indices = ds['lat'].to_numpy()
        for lon, lat in zip(lons_edge, lats_edge):
            lon_idx = np.abs(lon_indices - lon).argmin()
            lat_idx = np.abs(lat_indices - lat).argmin()
            mask[lat_idx, lon_idx] = True
    return mask

def mask_edge(input_mask):
    """
    Create a mask for the edges of an input mask.

    Parameters:
    - input_mask (xr.DataArray): A boolean mask where True indicates the region of interest.

    Returns:
    - xr.DataArray: A boolean mask with the same shape as the input mask, where
                    True indicates the edge of the original mask.
    """
    padded_mask = xr.concat([input_mask.shift(lat=1, fill_value=False),
                              input_mask.shift(lat=-1, fill_value=False),
                              input_mask.shift(lon=1, fill_value=False),
                              input_mask.shift(lon=-1, fill_value=False)],
                             dim='direction').any(dim='direction')
    edge_mask = padded_mask & ~input_mask
    return edge_mask

def mask_edge_right(input_mask):
    return input_mask.shift(lon=1, fill_value=False) & ~input_mask
def mask_edge_left(input_mask):
    return input_mask.shift(lon=-1, fill_value=False) & ~input_mask
def mask_edge_top(input_mask):
    return input_mask.shift(lat=1, fill_value=False) & ~input_mask
def mask_edge_bottom(input_mask):
    return input_mask.shift(lat=-1, fill_value=False) & ~input_mask

### time plots ###
months_name_list=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def nice_time_plot(plotvar, ax, label=None, title=None, ylabel=None, xlabel=None, color=None, vmin=None, vmax=None, linestyle='-'):
    plotvar.plot(ax=ax, label=label, color=color, linestyle=linestyle)
    if not (title=='off'):
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if vmin is not None:
        if vmax is not None:
            ax.set_ylim(vmin, vmax)
    ax.legend()
    # ax.grid()

def time_series_ave(ds_list, var, ds_colors=False, ds_linestyle=False, figsize=(8.5, 5), year_min=2010, year_max=2022, title=None, ylabel=None, xlabel=None, vmin=None, vmax=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    # ax.grid()
    if not title:
        title = f"{var} ({ds_list[0][var].attrs['units']})"

    for ds in ds_list:
        # Filter the dataset by the specified year range
        plotvar = ds[var].where(ds['time.year'] >= year_min, drop=True).where(ds['time.year'] <= year_max, drop=True)

        # Check if 'lon' and 'lat' dimensions exist before calculating the mean
        if 'lon' in plotvar.dims and 'lat' in plotvar.dims:
            plotvar = plotvar.mean(dim=['lon', 'lat'])
        
        color=ds.attrs["plot_color"] if ds_colors else None
        linestyle=ds.attrs["linestyle"] if ds_linestyle else '-'

        nice_time_plot(plotvar, ax, label=ds.name, color=color, linestyle=linestyle, title=title, ylabel=ylabel, xlabel=xlabel, vmin=vmin, vmax=vmax)

def time_series_lonlat(ds_list, var, lon, lat, figsize=(8.5, 5), year_min=2010, year_max=2022, title=None, ylabel=None, xlabel=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
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

def seasonal_cycle_lonlat(ds_list, var, lon, lat, figsize=(8.5, 5), ds_colors=False, year_min=2010, year_max=2022, title=None, ylabel=None, xlabel=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    if not title:
        title = ds_list[0][var].attrs['long_name'] + (' at ({}°N,{}°W) ({})'.format(lon,lat,ds_list[0][var].attrs['units']))
    for ds in ds_list:
        ds = ds.where(ds['time.year'] >= year_min, drop=True).where(ds['time.year'] <= year_max, drop=True)
        plotvar=ds[var].sel(lon=lon, lat=lat, method='nearest')
        #print annual mean
        mean=plotvar.mean(dim='time').values
        print(ds.attrs['name'] + ' : %.2f' % mean + ' ({})'.format(ds[var].attrs['units']))
        
        monthly_plotvar=plotvar.groupby('time.month').mean(dim='time')
        if ds_colors:
            nice_time_plot(monthly_plotvar,ax,label=ds.name, color=ds.attrs["plot_color"], title=title, ylabel=ylabel, xlabel=xlabel)
        else:
            nice_time_plot(monthly_plotvar,ax,label=ds.name, title=title, ylabel=ylabel, xlabel=xlabel)
    # ax.grid()
    ax.set_xticks(np.arange(1,13))
    ax.set_xticklabels(months_name_list)

def time_series(plotvars, labels, colors=None, figsize=(8.5, 5), year_min=None, year_max=None, title=None, ylabel=None, xlabel=None, vmin=None, vmax=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    i=j=0
    for plotvar in plotvars:
        label=labels[j]
        j+=1
        if year_min:
            plotvar=plotvar.where(plotvar['time.year'] >= year_min, drop=True)
        if year_max:
            plotvar=plotvar.where(plotvar['time.year'] <= year_max, drop=True)
        if colors:
            color=colors[i]
            i+=1
            nice_time_plot(plotvar,ax,label=label, color=color, title=title, ylabel=ylabel, xlabel=xlabel, vmin=vmin, vmax=vmax)
        else:
            nice_time_plot(plotvar,ax,label=label, title=title, ylabel=ylabel, xlabel=xlabel,  vmin=vmin, vmax=vmax)

def seasonal_cycle(plotvars, labels, colors=None, figsize=(8.5, 5), year_min=2010, year_max=2022, title=None, ylabel=None, xlabel=None, vmin=None, vmax=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    i=j=0
    for plotvar in plotvars:
        label=labels[j]
        j+=1
        plotvar=plotvar.where(plotvar['time.year'] >= year_min, drop=True).where(plotvar['time.year'] <= year_max, drop=True)
        mean=plotvar.mean(dim=['time']).values
        print('{} : %.3f'.format(label) % mean) 
        plotvar=plotvar.groupby('time.month').mean(dim='time')
        if colors:
            color=colors[i]
            i+=1
            nice_time_plot(plotvar,ax,label=label, color=color, title=title, ylabel=ylabel, xlabel=xlabel,  vmin=vmin, vmax=vmax)
        else:
            nice_time_plot(plotvar,ax,label=label, title=title, ylabel=ylabel, xlabel=xlabel, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(1,13))
    ax.set_xticklabels(months_name_list)
    ax.grid()

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

def scatter_vars_density(ds1, ds2, var1, var2, reg=False, plot_one=False, plot_m_one=False, title=None, gridsize=30, cmap='Blues'):
    """
    Plots a scatter plot of values of two variables in an xarray dataset,
    with an optional linear regression line and coefficient.
    Optionally displays the density of points using a hexbin plot.

    Parameters:
    - ds1: xarray.Dataset
        The dataset containing the first variable.
    - ds2: xarray.Dataset
        The dataset containing the second variable.
    - var1: str
        The name of the first variable.
    - var2: str
        The name of the second variable.
    - reg: bool, optional
        Whether to add a linear regression line and display the regression coefficient. Default is False.
    - gridsize: int, optional
        The size of the hexagons in the hexbin plot. Default is 30.
    """
    
    # Flatten the grid data for scatter plotting
    x = ds1[var1].values.flatten()
    y = ds2[var2].values.flatten()
    
    # Remove NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    # Plot the hexbin plot for density
    plt.figure(figsize=(8, 6))
    plt.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=1)
    plt.colorbar(label='Density')  # Show colorbar to indicate density
    plt.xlabel(f'{var1} ({ds1.attrs["name"]})')
    plt.ylabel(f'{var2} ({ds2.attrs["name"]})')
    plt.grid(True)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Scatter plot with Density')

    # Optional linear regression
    if reg:
        # Perform linear regression
        slope, intercept, r_value, _, _ = linregress(x, y)
        # Plot regression line
        plt.plot(x, slope * x + intercept, color='red', label=f'Regression: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value**2:.2f}')
        plt.legend()

    # Optional line for 1:1 relationship
    if plot_one:
        plt.plot([x.min(), x.max()], [x.min(), x.max()], color='black', linestyle='--', label='1:1')
        plt.legend() 

    # Optional line for 1:-1 relationship
    if plot_m_one:
        plt.plot([x.min(), x.max()], [-x.min(), -x.max()], color='black', linestyle='--', label='1:-1')
        plt.legend()
    
    plt.show()

def scatter_vars_months(ds1, ds2, var1, var2, reg=False, plot_one=False, plot_m_one=False, title=None, coloring=False, months_list=None, is_1D=False):
    """
    Plots a scatter plot of values of two variables in xarray datasets,
    with optional linear regression and month-based coloring.

    Parameters:
    - ds1: xarray.Dataset
        The first dataset containing the variable var1.
    - ds2: xarray.Dataset
        The second dataset containing the variable var2.
    - var1: str
        The name of the first variable.
    - var2: str
        The name of the second variable.
    - reg: bool, optional
        Whether to add a linear regression line and display the regression coefficient. Default is False.
    - plot_one: bool, optional
        Whether to plot a 1:1 line. Default is False.
    - plot_m_one: bool, optional
        Whether to plot a 1:-1 line. Default is False.
    - title: str, optional
        The title of the plot. Default is None.
    - coloring: bool, optional
        Whether to color points by their corresponding month. Default is False.
    - months_list: list of int, optional
        A list of months to include in the plot (e.g., [1, 2, 12] for Jan, Feb, Dec). Default is None (all months included).
    """
    # Flatten the grid data for scatter plotting
    x = ds1[var1].values.flatten()
    y = ds2[var2].values.flatten()
    time = ds1.time.values  # Assumes `time` is a dimension

    if coloring:
        # Expand time to match the grid dimensions, then flatten
        if is_1D:
            time_expanded = np.broadcast_to(ds1.time.values[:], ds1[var1].shape).flatten()
        else:
            time_expanded = np.broadcast_to(ds1.time.values[:, None, None], ds1[var1].shape).flatten()
        months = pd.to_datetime(time_expanded).month

        # Filter to include only selected months
        if months_list is not None:
            months_to_plot = months_list
        else:
            months_to_plot = range(1, 13)  # Default: all months

        # Assign fixed colors for all months (1 = January, ..., 12 = December)
        month_colors = {
            1: "blue", 2: "cyan", 3: "green", 4: "lime",
            5: "yellow", 6: "orange", 7: "red", 8: "magenta",
            9: "purple", 10: "brown", 11: "pink", 12: "gray"
        }

        # Map month numbers to names
        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }

        plt.figure(figsize=(10, 7))
        for month in months_to_plot:
            mask = months == month
            plt.scatter(x[mask], y[mask], alpha=0.5, label=month_names[month], color=month_colors[month])
        plt.legend(title='Month')
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.5)

    plt.xlabel(f'{var1} ({ds1.attrs.get("name", "")})')
    plt.ylabel(f'{var2} ({ds2.attrs.get("name", "")})')
    plt.grid(True)
    if title:
        plt.title(title)
    else:
        plt.title('Monthly mean (1pt=1grid cell)')

    # Optional linear regression
    if reg:
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        slope, intercept, r_value, _, _ = linregress(x, y)
        plt.plot(x, slope * x + intercept, color='red', label=f'Regression: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value**2:.2f}')
        plt.legend()

    # Optional 1:1 and 1:-1 lines
    if plot_one:
        plt.plot([x.min(), x.max()], [x.min(), x.max()], color='black', linestyle='--', label='1:1')
    if plot_m_one:
        plt.plot([x.min(), x.max()], [-x.min(), -x.max()], color='black', linestyle='--', label='1:-1')

    if plot_one or plot_m_one:
        plt.legend()

    plt.show()

def scatter_annual_mean(ds1, ds2, var1, var2, reg=False, plot_one=False, title=None):
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
    if title:
        plt.title(title)
    else:
        plt.title(f'Annual mean (1 data point = 1 grid cell)')
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
    if plot_one:
        plt.plot([x.min(), x.max()], [x.min(), x.max()], color='black', linestyle='--', label='1:1')
        plt.legend()
    
    # plt.show()

def scatter_vars_seasons(ds1, ds2, var1, var2, reg=False, plot_one=False, plot_m_one=False, title=None, coloring=False, is_1D=False, seasons_list=None, xlabel=None, ylabel=None):
    """
    Plots a scatter plot of values of two variables in xarray datasets,
    with optional linear regression and season-based coloring.

    Parameters:
    - ds1: xarray.Dataset
        The first dataset containing the variable var1.
    - ds2: xarray.Dataset
        The second dataset containing the variable var2.
    - var1: str
        The name of the first variable.
    - var2: str
        The name of the second variable.
    - reg: bool, optional
        Whether to add a linear regression line and display the regression coefficient. Default is False.
    - plot_one: bool, optional
        Whether to plot a 1:1 line. Default is False.
    - plot_m_one: bool, optional
        Whether to plot a 1:-1 line. Default is False.
    - title: str, optional
        The title of the plot. Default is None.
    - coloring: bool, optional
        Whether to color points by their corresponding season. Default is False.
    - seasons_list: list of str, optional
        A list of seasons to include in the plot (e.g., ['DJF', 'MAM']). Default is None (all seasons included).
    """
    # Filter ds depending on the seasons
    # Define seasons
    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11],
    }
    months_list=[]
    for season in seasons_list:
        months_list+=seasons[season]
    print('Months : {}'.format(months_list))
    ds1=ds1.where(ds1['time.month'].isin(months_list), drop=True)
    ds2=ds2.where(ds2['time.month'].isin(months_list), drop=True)

    # Flatten the grid data for scatter plotting
    x = ds1[var1].values.flatten()
    y = ds2[var2].values.flatten()
    time = ds1.time.values  # Assumes `time` is a dimension

    plt.figure(figsize=(8, 6))
    
    #plotting data
    if coloring:
        # Expand time to match the grid dimensions, then flatten
        if is_1D:
            time_expanded = np.broadcast_to(ds1.time.values[:], ds1[var1].shape).flatten()
        else:
            time_expanded = np.broadcast_to(ds1.time.values[:, None, None], ds1[var1].shape).flatten()
        months = pd.to_datetime(time_expanded).month

        # Filter to include only selected seasons
        if seasons_list is not None:
            seasons_to_plot = seasons_list
        else:
            seasons_to_plot = seasons.keys()  # Default: all seasons

        # Assign fixed colors for all seasons
        season_colors = {
            'DJF': "blue",  # Winter
            'MAM': "green",  # Spring
            'JJA': "red",    # Summer
            'SON': "orange"  # Autumn
        }

        for season in seasons_to_plot:
            if season not in seasons:
                raise ValueError(f"Invalid season name: {season}. Must be one of {list(seasons.keys())}.")
            months_in_season = seasons[season]
            mask = np.isin(months, months_in_season)
            plt.scatter(x[mask], y[mask], alpha=0.5, label=season, color=season_colors[season])
        plt.legend(title='Season')
    else:
        plt.scatter(x, y, alpha=0.5)

    #labels on axes and title
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(f'{var1} ({ds1.attrs.get("name", "")})')
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(f'{var2} ({ds2.attrs.get("name", "")})')
    plt.grid(True)
    if title:
        plt.title(title)
    else:
        plt.title('Monthly mean')

    # Optional linear regression
    if reg:
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        slope, intercept, r_value, _, _ = linregress(x, y)
        plt.plot(x, slope * x + intercept, color='grey', label=f'y = {slope:.2f}x + {intercept:.2f} ($R^2$ = {r_value**2:.2f})')
        plt.legend()

    # Optional 1:1 and 1:-1 lines
    if plot_one:
        plt.plot([x.min(), x.max()], [x.min(), x.max()], color='grey', linestyle='--', label='1:1')
    if plot_m_one:
        plt.plot([x.min(), x.max()], [-x.min(), -x.max()], color='grey', linestyle='--', label='1:-1')
    if plot_one or plot_m_one:
        plt.legend()

    plt.show()

def scatter_vars_seasons_ax(ax, ds1, ds2, var1, var2, reg=False, plot_one=False, plot_m_one=False, title=None, coloring=False, is_1D=False, seasons_list=None, xlabel=None, ylabel=None):
    # Filter ds depending on the seasons
    # Define seasons
    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11],
    }
    months_list=[]
    for season in seasons_list:
        months_list+=seasons[season]
    print('Months : {}'.format(months_list))
    ds1=ds1.where(ds1['time.month'].isin(months_list), drop=True)
    ds2=ds2.where(ds2['time.month'].isin(months_list), drop=True)

    # Flatten the grid data for scatter plotting
    x = ds1[var1].values.flatten()
    y = ds2[var2].values.flatten()
    time = ds1.time.values  # Assumes `time` is a dimension
    
    #plotting data
    if coloring:
        # Expand time to match the grid dimensions, then flatten
        if is_1D:
            time_expanded = np.broadcast_to(ds1.time.values[:], ds1[var1].shape).flatten()
        else:
            time_expanded = np.broadcast_to(ds1.time.values[:, None, None], ds1[var1].shape).flatten()
        months = pd.to_datetime(time_expanded).month

        # Filter to include only selected seasons
        if seasons_list is not None:
            seasons_to_plot = seasons_list
        else:
            seasons_to_plot = seasons.keys()  # Default: all seasons

        # Assign fixed colors for all seasons
        season_colors = {
            'DJF': "blue",  # Winter
            'MAM': "green",  # Spring
            'JJA': "red",    # Summer
            'SON': "orange"  # Autumn
        }

        for season in seasons_to_plot:
            if season not in seasons:
                raise ValueError(f"Invalid season name: {season}. Must be one of {list(seasons.keys())}.")
            months_in_season = seasons[season]
            mask = np.isin(months, months_in_season)
            ax.scatter(x[mask], y[mask], alpha=0.5, label=season, color=season_colors[season])
        # ax.legend(title='Season')
    else:
        ax.scatter(x, y, alpha=0.5)

    #labels on axes and title
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(f'{var1} ({ds1.attrs.get("name", "")})')
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(f'{var2} ({ds2.attrs.get("name", "")})')
    plt.grid(True)
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Monthly mean')

    # Optional linear regression
    if reg:
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        slope, intercept, r_value, _, _ = linregress(x, y)
        ax.plot(x, slope * x + intercept, color='grey', label=f'y = {slope:.2f}x + {intercept:.2f} ($R^2$ = {r_value**2:.2f})')
        ax.legend()

    # Optional 1:1 and 1:-1 lines
    if plot_one:
        ax.plot([x.min(), x.max()], [x.min(), x.max()], color='grey', linestyle='--', label='1:1')
    if plot_m_one:
        ax.plot([x.min(), x.max()], [-x.min(), -x.max()], color='grey', linestyle='--', label='1:-1')
    if plot_one or plot_m_one:
        ax.legend()

def scatter_seasonal_mean(ds1, ds2, var1, var2, seasons_list=['DJF','MAM','JJA','SON'], plot_one=False, plot_m_one=False, reg=False):
    """
    Plots a scatter plot of the seasonal mean values of two variables in an xarray dataset,
    with an optional linear regression line and coefficient for each season.

    Parameters:
    - ds1: xarray.Dataset
        The dataset containing the first variable.
    - ds2: xarray.Dataset
        The dataset containing the second variable.
    - var1: str
        The name of the first variable.
    - var2: str
        The name of the second variable.
    - seasons_list: list of str
        The list of seasons to include (e.g., ['DJF', 'MAM', 'JJA', 'SON']).
    - reg: bool, optional
        Whether to add a linear regression line and display the regression coefficient. Default is False.
    """
    
    # Define season months mapping
    season_months = {
        'DJF': [12, 1, 2],  # December, January, February
        'MAM': [3, 4, 5],   # March, April, May
        'JJA': [6, 7, 8],   # June, July, August
        'SON': [9, 10, 11]  # September, October, November
    }
    
    # Create a color map for the seasons
    season_colors = {
        'DJF': 'blue',
        'MAM': 'green',
        'JJA': 'orange',
        'SON': 'red'
    }

    # Flatten the grid data for plotting
    x = ds1[var1].values.flatten()
    y = ds2[var2].values.flatten()
    
    # Plot the scatter plot
    plt.figure(figsize=(8, 6))

    for season in seasons_list:
        # Filter the time dimension based on the season months
        months = season_months[season]
        season_mask = np.isin(ds1['time.month'].values, months)

        # Compute the seasonal mean for each variable by season
        var1_season_mean = ds1[var1].isel(time=season_mask).mean(dim='time')
        var2_season_mean = ds2[var2].isel(time=season_mask).mean(dim='time')

        # Flatten the seasonal means for plotting
        x_season = var1_season_mean.values.flatten()
        y_season = var2_season_mean.values.flatten()

        # Plot each season with a unique color
        plt.scatter(x_season, y_season, alpha=0.5, color=season_colors[season], label=season)
        
        # Optional linear regression for each season
        if reg:
            # Remove NaN values for regression
            mask = ~np.isnan(x_season) & ~np.isnan(y_season)
            x_season = x_season[mask]
            y_season = y_season[mask]
            
            # Perform linear regression
            slope, intercept, r_value, _, _ = linregress(x_season, y_season)
            
            # Plot regression line for each season
            plt.plot(x_season, slope * x_season + intercept, color=season_colors[season], 
                     label=f'{season} regression: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value**2:.2f}')

    plt.xlabel(f'{var1} ({ds1.attrs["name"]})')
    plt.ylabel(f'{var2} ({ds2.attrs["name"]})')
    plt.title(f'Scatter plot of seasonal means')
    plt.grid(True)

    # Optional 1:1 and 1:-1 lines
    if plot_one:
        plt.plot([x.min(), x.max()], [x.min(), x.max()], color='black', linestyle='--', label='1:1')
    if plot_m_one:
        plt.plot([x.min(), x.max()], [-x.min(), -x.max()], color='black', linestyle='--', label='1:-1')
    # if plot_one or plot_m_one:
    #     plt.legend()
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
                     
#moisture flux functions
def moisture_flux_rectangle(ds, polygon):
    """
    Used for development
    Function that takes a dataset and a polygon
    Calculates the moisture input in kg/m2/s using uq and vq variables
    Summing along vertical segment for uq and horizontal segment for vq
    """
    var1='uq'
    var2='vq'
    # Extract polygon coordinates
    lons = [polygon[key]['lon'] for key in polygon]
    lats = [polygon[key]['lat'] for key in polygon]

    # Calculate bounds of the polygon
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    uq_integ = ds[var1].mean(dim='time') * ds['cell_height']
    vq_integ = ds[var2].mean(dim='time') * ds['cell_width']

    # Select data along the polygon's edges (bounding box)
    uq_left     = uq_integ.sel(lon=lon_min, method='nearest').sel(lat=slice(lat_min, lat_max))  # Left edge
    uq_right    = uq_integ.sel(lon=lon_max, method='nearest').sel(lat=slice(lat_min, lat_max))  # Right edge
    vq_bottom   = vq_integ.sel(lat=lat_min, method='nearest').sel(lon=slice(lon_min, lon_max))  # Bottom edge
    vq_top      = vq_integ.sel(lat=lat_max, method='nearest').sel(lon=slice(lon_min, lon_max))  # Top edge

    # Sum moisture fluxes along respective edges
    u_input     = uq_left.sum(dim='lat')  # Moisture entering through the left
    u_output    = uq_right.sum(dim='lat')  # Moisture exiting through the right
    v_input     = vq_bottom.sum(dim='lon')  # Moisture entering through the bottom
    v_output    = vq_top.sum(dim='lon')  # Moisture exiting through the top

    # Calculate total moisture flux
    total = u_input - u_output + v_input - v_output

    #make a dict with 4 components and total
    output_dict = {'u_input': u_input.values,
                   'u_output': u_output.values,
                   'v_input': v_input.values,
                   'v_output': v_output.values,
                   'total': total.values
                   }
    return(output_dict)

def moisture_flux_polygon(ds, polygon):
    """
    Used for develoment
    """
    #create variables for integrated flux over edges
    ds['uq_integ'] = ds['uq'].mean(dim='time') * ds['cell_height']
    ds['vq_integ'] = ds['vq'].mean(dim='time') * ds['cell_width']

    #creat masks for edges
    mask0=polygon_to_mask(ds, polygon)
    mask_b=mask_edge_bottom(mask0)
    mask_b.attrs['name']='mask_bottom'
    mask_t=mask_edge_top(mask0)
    mask_t.attrs['name']='mask_top'
    mask_r=mask_edge_right(mask0)
    mask_r.attrs['name']='mask_right'
    mask_l=mask_edge_left(mask0)
    mask_l.attrs['name']='mask_left'

    #apply masks to dataset
    masked_b = apply_2Dmask_to_dataset(ds, mask_b)
    masked_t = apply_2Dmask_to_dataset(ds, mask_t)
    masked_l = apply_2Dmask_to_dataset(ds, mask_l)
    masked_r = apply_2Dmask_to_dataset(ds, mask_r)

    #identify relevant masked variable for each edge
    masked_vq_b = masked_b['vq_integ']
    masked_vq_t = masked_t['vq_integ']
    masked_uq_l = masked_l['uq_integ']
    masked_uq_r = masked_r['uq_integ']

    #calculate moisture fluxes on each edge
    u_input     = masked_uq_l.sum(dim=['lon','lat'])  # Moisture entering through the left
    u_output    = masked_uq_r.sum(dim=['lon','lat'])  # Moisture exiting through the right
    v_input     = masked_vq_b.sum(dim=['lon','lat'])  # Moisture entering through the bottom
    v_output    = masked_vq_t.sum(dim=['lon','lat'])  # Moisture exiting through the top

    # Calculate total moisture flux
    total = u_input - u_output + v_input - v_output
    #make a dict with 4 components and total
    output_dict = {'u_input': u_input.values,
                   'u_output': u_output.values,
                   'v_input': v_input.values,
                   'v_output': v_output.values,
                   'total': total.values
                   }
    return(output_dict)

def moisture_flux_mask(ds, mask):
    """
    Compute the moisture fluxes across the edges of a given mask.
    The ds must have cell width and height, and one time dimension only
    ds and mask must have a name attribute.
    """
    #create variables for integrated flux over edges
    # ds['uq_integ'] = ds['uq'].mean(dim='time') * ds['cell_height']
    # ds['vq_integ'] = ds['vq'].mean(dim='time') * ds['cell_width']
    ds['uq_integ'] = ds['uq'] * ds['cell_height']
    ds['vq_integ'] = ds['vq'] * ds['cell_width']

    #creat masks for edges
    mask0=mask
    mask_b=mask_edge_bottom(mask0)
    mask_b.attrs['name']='mask_bottom'
    mask_t=mask_edge_top(mask0)
    mask_t.attrs['name']='mask_top'
    mask_r=mask_edge_right(mask0)
    mask_r.attrs['name']='mask_right'
    mask_l=mask_edge_left(mask0)
    mask_l.attrs['name']='mask_left'

    #apply masks to dataset
    masked_b = apply_2Dmask_to_dataset(ds, mask_b)
    masked_t = apply_2Dmask_to_dataset(ds, mask_t)
    masked_l = apply_2Dmask_to_dataset(ds, mask_l)
    masked_r = apply_2Dmask_to_dataset(ds, mask_r)

    #identify relevant masked variable for each edge
    masked_vq_b = masked_b['vq_integ']
    masked_vq_t = masked_t['vq_integ']
    masked_uq_l = masked_l['uq_integ']
    masked_uq_r = masked_r['uq_integ']

    #calculate moisture fluxes on each edge
    u_input     = masked_uq_l.sum(dim=['lon','lat'])  # Moisture entering through the left
    u_output    = masked_uq_r.sum(dim=['lon','lat'])  # Moisture exiting through the right
    v_input     = masked_vq_b.sum(dim=['lon','lat'])  # Moisture entering through the bottom
    v_output    = masked_vq_t.sum(dim=['lon','lat'])  # Moisture exiting through the top

    #make a dict with 4 components and total
    output_dict = {'u_input': u_input.values,
                   'u_output': u_output.values,
                   'v_input': v_input.values,
                   'v_output': v_output.values
                   }
    return(output_dict)

def moisture_budget_mean_table(ds, mask, mask_area=None, recalc_width=False):
    if recalc_width:
        ds=compute_grid_cell_width(ds)
    ds_mean = ds.mean(dim='time')
    ds_mean.attrs['name'] = 'ds mean'
    #add name to mask if it doesn't have one
    if 'name' not in mask.attrs:
        mask.attrs['name'] = 'mask'
    #calculate mask area
    if not mask_area:
        print('Calculating mask area')
        mask_area = compute_mask_area(ds, mask)
    #calculate moisture fluxes
    # print('Calculating moisture fluxes')
    fluxes_dict = moisture_flux_mask(ds_mean, mask)
    pprint(fluxes_dict)
    #turn into pd df
    # Transform the dictionary into a suitable format
    fluxes_data = {key: [float(value)] for key, value in fluxes_dict.items()}
    # Create a DataFrame
    fluxes_df = pd.DataFrame(fluxes_data)
    fluxes_df.index = [ds.attrs['name'] + ' (kg/s)']

    # add row for mm/d
    new_row = {key: (value / mask_area * 86400) for key, value in fluxes_dict.items()}
    fluxes_df.loc[ds.attrs['name'] + ' (mm/d)'] = new_row

    # compute inputs outputs and total
    fluxes_df['inputs'] = fluxes_df['u_input'] + fluxes_df['v_input']
    fluxes_df['outputs'] = fluxes_df['u_output'] + fluxes_df['v_output']
    fluxes_df['total'] = fluxes_df['inputs'] - fluxes_df['outputs']
    
    # format numbers
    fluxes_df = fluxes_df.applymap(lambda x: float(f"{x:.3g}"))

    return(fluxes_df)

def add_moisture_flux_to_ds(ds, mask, mask_area=None, recalc_width=False):
    """
    Compute the moisture fluxes across the edges of a given mask, on a dataset
    Output is an xarray datsaet of moiture input, output and total over time
    """
    if recalc_width:
        ds=compute_grid_cell_width(ds)
    # compute mask area
    if not mask_area:
        print('Calculating mask area')
        mask_area = compute_mask_area(ds, mask)

    #compute moisture fluxes
    ds['uq_integ'] = ds['uq'] * ds['cell_height']
    ds['vq_integ'] = ds['vq'] * ds['cell_width']

    #creat masks for edges
    mask0=mask
    mask_b=mask_edge_bottom(mask0)
    mask_b.attrs['name']='mask_bottom'
    mask_t=mask_edge_top(mask0)
    mask_t.attrs['name']='mask_top'
    mask_r=mask_edge_right(mask0)
    mask_r.attrs['name']='mask_right'
    mask_l=mask_edge_left(mask0)
    mask_l.attrs['name']='mask_left'

    #apply masks to dataset
    masked_b = apply_2Dmask_to_dataset(ds, mask_b)
    masked_t = apply_2Dmask_to_dataset(ds, mask_t)
    masked_l = apply_2Dmask_to_dataset(ds, mask_l)
    masked_r = apply_2Dmask_to_dataset(ds, mask_r)

    #identify relevant masked variable for each edge
    masked_vq_b = masked_b['vq_integ']
    masked_vq_t = masked_t['vq_integ']
    masked_uq_l = masked_l['uq_integ']
    masked_uq_r = masked_r['uq_integ']

    #calculate moisture fluxes on each edge
    u_input     = masked_uq_l.sum(dim=['lon','lat'])  # Moisture entering through the left
    u_output    = masked_uq_r.sum(dim=['lon','lat'])  # Moisture exiting through the right
    v_input     = masked_vq_b.sum(dim=['lon','lat'])  # Moisture entering through the bottom
    v_output    = masked_vq_t.sum(dim=['lon','lat'])  # Moisture exiting through the top

    ds['q_input'] = (u_input + v_input) * 86400 / mask_area
    ds['q_output'] = (u_output + v_output) * 86400 / mask_area
    ds['q_total'] = ds['q_input'] - ds['q_output']
    #add attribute units to 2 vars
    ds['q_input'].attrs['units']  = 'mm/d'
    ds['q_output'].attrs['units'] = 'mm/d'
    ds['q_total'].attrs['units']  = 'mm/d'
    return(ds)

def add_moisture_divergence(ds, uq_var='uq', vq_var='vq', cell_width_var='cell_width', cell_height_var='cell_height'):  
    """
    Compute and add moisture divergence to the dataset.

    Parameters:
    ds (xr.Dataset): Input dataset with variables `uq`, `vq`, and `cell_width`.
    uq_var (str): Name of the variable for zonal moisture flux. Default is 'uq'.
    vq_var (str): Name of the variable for meridional moisture flux. Default is 'vq'.
    cell_width_var (str): Name of the variable for grid cell width. Default is 'cell_width'.

    Returns:
    xr.Dataset: Dataset with added moisture divergence variable.
    """
    # Get variables
    uq = ds[uq_var]
    vq = ds[vq_var]
    cell_width = ds[cell_width_var]
    cell_height = ds[cell_height_var]

    # Compute gradients
    # dqdx = uq.differentiate('lon')# / cell_width 
    # dqdy = vq.differentiate('lat')# / cell_height
    forward_dx = uq.diff('lon', label='upper')
    backward_dx = uq.diff('lon', label='lower')
    dqdx = ( backward_dx + forward_dx.shift(lon=-1) ) / 2 / cell_width
    dqdx = dqdx * 86400

    forward_dy = vq.diff('lat', label='upper')
    backward_dy = vq.diff('lat', label='lower')
    dqdy = ( backward_dy + forward_dy.shift(lat=-1) ) / 2 / cell_height
    dqdy = dqdy * 86400

    # Compute divergence    

    divergence = dqdx + dqdy
    convergence = -divergence

    # Add divergence to the dataset
    ds['moisture_divergence'] = divergence
    ds['moisture_divergence'].attrs = {
        'units': 'mm/d',
        'description': 'Moisture divergence computed from uq and vq',
        'long_name': 'Moisture divergence'
    }
    ds['moisture_convergence'] = convergence
    ds['moisture_convergence'].attrs = {
        'units': 'mm/d',
        'description': 'Moisture convergence computed from uq and vq',
        'long_name': 'Moisture convergence'
    }

    return ds

def plot_side_by_side_barplots(df):
    """
    Creates a figure with 3 side-by-side bar plots, one for each numeric column in the DataFrame.
    Each plot has 4 bars, one for each row in the DataFrame.
    """
    categories = df.iloc[:, 0]  # First column (categorical labels)
    values = df.iloc[:, 1:]     # Numeric columns

    # Setting up the figure
    fig, axes = plt.subplots(1, len(values.columns), figsize=(10, 5), sharey=True)
    
    for i, col in enumerate(values.columns):
        axes[i].bar(categories, values[col], color=['lightgrey', 'green', 'blue','black' ])
        axes[i].set_title(f'{col}', fontsize=14)
        axes[i].set_xlabel('', fontsize=12)
        axes[i].set_ylabel('mm y⁻¹' if i == 0 else '', fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()