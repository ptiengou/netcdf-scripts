from tools import *
from tools_mapping import *
from tools_LIAISE import *
from tools_hf import *


##DATASETS##
def subset_dataset_by_lon_lat_box(ds_input, lon_min, lon_max, lat_min, lat_max):
    """
    Returns a new xarray Dataset containing only data points within the
    specified longitude-latitude bounding box. Points outside the box will be
    set to NaN. This is for datasets where 'lon' and 'lat' are 2D coordinates
    with (nj, ni) dimensions.

    Args:
        ds_input (xr.Dataset): The input xarray Dataset.
        lon_min (float): Minimum longitude for the bounding box.
        lon_max (float): Maximum longitude for the bounding box.
        lat_min (float): Minimum latitude for the bounding box.
        lat_max (float): Maximum latitude for the bounding box.

    Returns:
        xr.Dataset: A new dataset with data outside the bounding box set to NaN.
    """
    spatial_mask = (ds_input['lon'] >= lon_min) & \
                   (ds_input['lon'] <= lon_max) & \
                   (ds_input['lat'] >= lat_min) & \
                   (ds_input['lat'] <= lat_max)

    ds_subsetted = ds_input.where(spatial_mask)

    return ds_subsetted

def subset_dataset_by_polygon(ds_input, polygon_dict, name=None, plot_color=None):
    """
    Returns a new xarray Dataset containing only data points inside the
    specified polygon. Points outside the polygon will be set to NaN.
    This is for datasets where 'lon' and 'lat' are 2D coordinates
    with (nj, ni) dimensions.

    Args:
        ds_input (xr.Dataset): The input xarray Dataset.
        polygon_dict (dict): A dictionary defining the polygon vertices.
                             Example: {1: {'lon': 0.83035, 'lat': 41.72748}, ...}
                             The keys are used to sort the points into order.

    Returns:
        xr.Dataset: A new dataset with data outside the polygon set to NaN.
    """
    # 1. Extract polygon vertices in order
    polygon_vertices = np.array([[point['lon'], point['lat']] for key, point in sorted(polygon_dict.items())])

    # 2. Create a matplotlib Path object from the polygon vertices
    polygon_path = Path(polygon_vertices)

    # 3. Prepare the dataset's 2D lon/lat coordinates for checking
    lon_coords_flat = ds_input['lon'].values.flatten()
    lat_coords_flat = ds_input['lat'].values.flatten()
    grid_points_flat = np.vstack((lon_coords_flat, lat_coords_flat)).T # Shape (N, 2)

    # 4. Check which grid points are inside the polygon
    mask_flat = polygon_path.contains_points(grid_points_flat)

    # 5. Reshape the 1D mask back to the original (nj, ni) dimensions
    original_shape = ds_input['lon'].shape # Should be (nj_dim, ni_dim)
    mask_2d = mask_flat.reshape(original_shape)

    # Convert the numpy boolean mask to an xarray DataArray for proper alignment
    xr_mask = xr.DataArray(mask_2d, coords={'nj': ds_input['nj'], 'ni': ds_input['ni']})

    # 6. Apply the mask to the entire dataset
    # This will set values outside the polygon to NaN for all variables
    ds_subsetted = ds_input.where(xr_mask)
    # Optionally, set the name of the dataset if provided
    if name:
        ds_subsetted.attrs['name'] = name

    #add plot_color
    if plot_color:
        ds_subsetted.attrs['plot_color'] = plot_color


    return ds_subsetted

def select_dataset_lon_lat(ds, lon, lat, name=None, plot_color=None):
    """
    Returns a new xarray Dataset containing only data points at the specified
    longitude and latitude. This is for datasets where 'lon' and 'lat' are 2D
    coordinates with (nj, ni) dimensions.

    Args:
        ds_input (xr.Dataset): The input xarray Dataset.
        lon (float): Longitude to select.
        lat (float): Latitude to select.

    Returns:
        xr.Dataset: A new dataset with data only at the specified lon/lat.
    """
    ni_idx, nj_idx = get_lonlat_index(ds, lon, lat)
    ds_selected = ds.copy().isel(ni=ni_idx, nj=nj_idx)
    # Optionally, set the name of the dataset if provided
    if name:
        ds_selected.attrs['name'] = name
    #add plot_color
    if plot_color:
        ds_selected.attrs['plot_color'] = plot_color
    return ds_selected
    
## MAPS ##

def nice_map_mesoNH(data_to_plot, vmin=None, vmax=None, cmap='viridis', 
                         add_liaise=False, title=None, label=None,
                         poly=None):
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # pcolormesh can directly use the 2D 'lon' and 'lat' coordinates
    data_to_plot.plot.pcolormesh(
        ax=ax,
        x="lon",
        y="lat",
        vmin=vmin, 
        vmax=vmax, 
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        cbar_kwargs={"label": label}
    )

    ax.coastlines()
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)

    gl=ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, alpha=0.8)
    gl.right_labels = False
    # gl.left_labels = left_labels
    gl.top_labels = False
    gl.xlocator = plt.MaxNLocator(8)
    gl.ylocator = plt.MaxNLocator(9)
    if title:
        ax.set_title(title)
    if add_liaise:
        add_liaise_site_loc(ax=ax)

    # Set the map extent to the *valid data's* lon/lat bounds after masking
    # This ensures the plot zooms into the visible data.
    min_lon_data = data_to_plot['lon'].where(~np.isnan(data_to_plot)).min()
    max_lon_data = data_to_plot['lon'].where(~np.isnan(data_to_plot)).max()
    min_lat_data = data_to_plot['lat'].where(~np.isnan(data_to_plot)).min()
    max_lat_data = data_to_plot['lat'].where(~np.isnan(data_to_plot)).max()

    ax.set_extent([min_lon_data, max_lon_data, min_lat_data, max_lat_data], crs=ccrs.PlateCarree())

    # Optional: plot polygon
    if poly:
        plot_polygon(poly, ax)

    plt.tight_layout()
    plt.show()

def map_mesoNH_timestamp(ds, var, vmin=None, vmax=None, cmap='viridis', 
                         add_liaise=False,
                         record_index=0, timestamp='2021-07-14T01:00:00', poly=None):
    # data_to_plot = ds[var].isel(record=record_index).squeeze()
    data_to_plot = ds[var].sel(time=timestamp, method='nearest')
    title=f'{var} on {timestamp}'
    label = f'{var} ({ds[var].attrs.get("units", "")})'
    nice_map_mesoNH(data_to_plot, vmin=vmin, vmax=vmax, cmap=cmap, 
                         add_liaise=add_liaise, title=title, label=label, poly=poly)

def map_mesoNH_timestamp_restrict(ds, var, vmin=None, vmax=None, cmap='viridis',
                         add_liaise=False,
                         timestamp='2021-07-14T01:00:00',
                         lon_min=None, lon_max=None, lat_min=None, lat_max=None,
                         poly=None):
    # Select data for the given timestamp
    data_to_plot_time_selected = ds[var].sel(time=timestamp, method='nearest')

    # Get the 2D lon/lat coordinates from the dataset (they have nj, ni dimensions)
    lons_coord = ds['lon']
    lats_coord = ds['lat']

    # Apply spatial subsetting if bounds are provided
    if lon_min is not None and lon_max is not None and \
       lat_min is not None and lat_max is not None:

        # Create a boolean mask based on the 2D lon/lat coordinates
        # This mask will have dimensions (nj, ni)
        spatial_mask = (lons_coord >= lon_min) & (lons_coord <= lon_max) & \
                       (lats_coord >= lat_min) & (lats_coord <= lat_max)

        # Apply the mask to the data. Values outside the region will become NaN.
        data_to_plot = data_to_plot_time_selected.where(spatial_mask)
    else:
        # If no subsetting is requested, use the time-selected data directly
        data_to_plot = data_to_plot_time_selected
        
    title = f'{var} on {pd.to_datetime(timestamp).strftime("%Y-%m-%d %H:%M UTC")}'
    label = f'{var} ({ds[var].attrs.get("units", "")})'

    nice_map_mesoNH(data_to_plot, vmin=vmin, vmax=vmax, cmap=cmap,
                         add_liaise=add_liaise, title=title, label=label, poly=poly)

def map_mesoNH_mean(ds, var, cmap='viridis', vmin=None, vmax=None, title=None, add_liaise=False, poly=None):
    """
    Maps the mean of a variable from a MesoNH dataset.
    """
    mean_ds = ds[var].mean(dim='time')
    title = title or f'{var} mean'
    label = f'{var} ({ds[var].attrs.get("units", "")})'
    nice_map_mesoNH(mean_ds, cmap=cmap, vmin=vmin, vmax=vmax, title=title, label=label, add_liaise=add_liaise, poly=poly)

def map_mesoNH_mean_restrict(ds, var, cmap='viridis', vmin=None, vmax=None, title=None, add_liaise=False, poly=None,
                             lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    """
    Maps the mean of a variable from a MesoNH dataset, with limitations on the spatial extent.
    """
    data_to_plot_mean_selected = ds[var].mean(dim='time')

    # Get the 2D lon/lat coordinates from the dataset (they have nj, ni dimensions)
    lons_coord = ds['lon']
    lats_coord = ds['lat']

    # Apply spatial subsetting if bounds are provided
    if lon_min is not None and lon_max is not None and \
       lat_min is not None and lat_max is not None:

        # Create a boolean mask based on the 2D lon/lat coordinates
        # This mask will have dimensions (nj, ni)
        spatial_mask = (lons_coord >= lon_min) & (lons_coord <= lon_max) & \
                       (lats_coord >= lat_min) & (lats_coord <= lat_max)

        # Apply the mask to the data. Values outside the region will become NaN.
        data_to_plot = data_to_plot_mean_selected.where(spatial_mask)
    else:
        # If no subsetting is requested, use the time-selected data directly
        data_to_plot = data_to_plot_mean_selected
     
    title = title or f'{var} mean'
    label = f'{var} ({ds[var].attrs.get("units", "")})'
    nice_map_mesoNH(data_to_plot, cmap=cmap, vmin=vmin, vmax=vmax, title=title, label=label, add_liaise=add_liaise, poly=poly)



## Time Series ##
def time_series_ave_mesoNH(ds, var, figsize=(7.5, 4)):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    
    # Select the variable and the point
    ts = ds[var].mean(dim=['ni', 'nj'])

    nice_time_plot(ts, ax)

def time_series_ninj_mesoNH(ds, var, ni, nj, figsize=(7.5, 4)):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    # Select the variable and the point
    ts = ds[var].sel(ni=ni, nj=nj, method='nearest')
    # Plot the time series
    nice_time_plot(ts, ax)

def time_series_ninj_index_mesoNH(ds, var, ni_index, nj_index, figsize=(7.5, 4), title=None, label=None, vmin=None, vmax=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    # Select the variable and the point
    ts = ds[var][:, nj_index, ni_index]
    # Plot the time series
    nice_time_plot(ts, ax, title=title, label=label, vmin=vmin, vmax=vmax)

def get_lonlat_index(ds, lon_point, lat_point):
    """
    Get the ni and njindices of the closest grid point in the dataset to the given lon/lat.
    """
    # Calculate squared distance from target point to all points in the 2D grid
    distance_sq = (ds['lon'] - lon_point)**2 + (ds['lat'] - lat_point)**2

    # Find the indices of the minimum distance
    # idxmin() returns a dictionary of the coordinate labels at the minimum
    min_indices = distance_sq.argmin(dim=('ni','nj'))

    ni_idx = min_indices['ni'].values.item()
    nj_idx = min_indices['nj'].values.item()
    return ni_idx, nj_idx

def time_series_lonlat_mesoNH(ds, var, lon, lat, figsize=(7.5, 4), vmin=None, vmax=None):
    # Select the variable and the point
    ni_idx, nj_idx = get_lonlat_index(ds, lon, lat)
    # Extract the time series for the specified lon/lat
    time_series_ninj_index_mesoNH(ds, var, ni_idx, nj_idx,
                                  figsize=figsize,
                                  title=f'{var} at lon={lon}, lat={lat}',
                                  vmin=vmin, vmax=vmax,
                                  )
    

def bins_timestamp(ds, var, timestamp, nbins=10,
                   xmin=None, xmax=None):
    """
    Make a histogram of the values of a variable at a given timestamp.
    """
    ds = ds.sel(time=timestamp)
    values = ds[var].values.flatten()
    values = values[~np.isnan(values)]


    # Create bins
    #take into account xmin and xmax
    if xmin is None:
        xmin = np.min(values)
    if xmax is None:
        xmax = np.max(values)
    bins = np.linspace(xmin, xmax, nbins + 1)
    
    # Create histogram
    hist, edges = np.histogram(values, bins=bins)
    

    # Plot histogram
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor='black', align='edge')
    plt.xlabel(var)
    plt.ylabel('Frequency (%)')
    plt.title(f'Distribution of {var} at {timestamp}')
    plt.grid()
    plt.show()