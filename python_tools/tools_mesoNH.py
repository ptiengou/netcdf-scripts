from tools import *
from tools_mapping import *
from tools_LIAISE import *
from tools_hf import *
from matplotlib.ticker import MultipleLocator, FuncFormatter



lon_min_mesoNH=-3.4
lon_max_mesoNH=4
lat_min_mesoNH=40.5
lat_max_mesoNH=43.5

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

    if 'level' in ds_subsetted.dims:
        ds_subsetted['altitude_agl'] = ds_subsetted['level']

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
    
    if 'level' in ds_selected.dims:
        ds_selected['altitude_agl'] = ds_selected['level']
    
    return ds_selected
    
def extract_pressure_surface(mesoNH_4D, mesoNH):
    # Extract the pressure at the surface (level 0)
    psol = mesoNH_4D['PABST'].isel(level=0)
    psol.attrs['long_name'] = 'Pressure at Surface'
    psol.attrs['units'] = 'Pa'
    
    # Add the pressure to the mesoNH dataset
    mesoNH['psol'] = psol
    
    return mesoNH

def extract_w10m(mesoNH_4D, mesoNH):
    # Extract the wind at 10m (level 3)
    w10m = mesoNH_4D['vitw'].isel(level=3)
    w10m.attrs['long_name'] = 'Wind at 10m'
    w10m.attrs['units'] = 'm s⁻¹'

    # Add the wind to the mesoNH dataset
    mesoNH['w10m'] = w10m

    return mesoNH

def extract_wind_pressure(mesoNH_4D, mesoNH, pressure_level=850):
    # Extract wind at specified pressure level
    #find level closest to pressure_level
    pres_levels = mesoNH_4D['PABST'].isel(time=0, nj=0, ni=0).values
    pres_idx = np.abs(pres_levels - pressure_level *100).argmin()
    u_name = f'u{pressure_level}'
    v_name = f'v{pressure_level}'
    w_name = f'w{pressure_level}'
    u_level = mesoNH_4D['vitu'].isel(level=pres_idx)
    v_level = mesoNH_4D['vitv'].isel(level=pres_idx)
    w_level = mesoNH_4D['vitw'].isel(level=pres_idx)
    mesoNH[u_name] = u_level
    mesoNH[v_name] = v_level
    mesoNH[w_name] = w_level
    return mesoNH

def flux_pt_to_mass_pt(ds, only_basic_vars=True, 
                       basic_vars = [
                           'UT', 'VT', 'WT',
                           'UMME', 'VMME', 'WMME',
                           'latitude_f', 'longitude_f',], 
                       verbose=True):
    """
    Change flux point coordinates of dataset into mass point coordinates.
    Done through interpolation between flux points.
    Generalization of center_uvw() function.
    
    parameter:
        ds: xarray.dataset
    Taken from Tanguy Lunel's repo https://github.com/tylunel/postproc_python
    """
    
    for var in list(ds.keys()):
        if only_basic_vars and var not in basic_vars:
            continue
        
        # NOT WORKING
#        for coord in ['nj_v', 'nj_u', 'ni_v', 'ni_u', 'level_w']:
#            new_coord = coord[:-2]
#            ds[var] = ds[var].interp(coords={coord:ds[new_coord].values}).rename(
#                {coord: new_coord})
#            if verbose:
#                print(f"{var}: coordinates '{coord}' changed to '{new_coord}'")
            
        # change coordinates for 'nj_v' flux points
        if 'nj_v' in list(ds[var].coords):
            ds[var] = ds[var].interp(nj_v=ds.nj.values).rename(
                {'nj_v': 'nj'})
            if verbose:
                print(f"{var}: coordinates 'nj_v' changed to 'nj'")
        
        # change coordinates for 'ni_v' flux points
        if 'ni_v' in list(ds[var].coords):
            ds[var] = ds[var].interp(ni_v=ds.ni.values).rename(
                {'ni_v': 'ni'})
            if verbose:
                print(f"{var}: coordinates 'ni_v' changed to 'ni'")
        
        # change coordinates for 'ni_u' flux points
        if 'ni_u' in list(ds[var].coords):
            ds[var] = ds[var].interp(ni_u=ds.ni.values).rename(
                {'ni_u': 'ni'})
            if verbose:
                print(f"{var}: coordinates 'ni_u' changed to 'ni'")
        
        # change coordinates for 'nj_u' flux points
        if 'nj_u' in list(ds[var].coords):
            ds[var] = ds[var].interp(nj_u=ds.nj.values).rename(
                {'nj_u': 'nj'})
            if verbose:
                print(f"{var}: coordinates 'nj_u' changed to 'nj'")
        
        # change coordinates for '_w' flux points
        if 'level_w' in list(ds[var].coords):
            ds[var] = ds[var].interp(level_w=ds.level.values).rename(
                {'level_w': 'level'})
            if verbose:
                print(f"{var}: coordinate 'level_w' changed to 'level'")
    
        # ALT coordinate badly encoded ('level_w not defnied as coords)
        if var == 'ALT':
            ds[var] = ds[var].interp(level_w=ds.level.values).rename(
                {'level_w': 'level'})
            if verbose:
                print(f"{var}: coordinate 'level_w' changed to 'level'")
    
    # remove flux coordinates
    try:
        ds_new = ds.drop(['latitude_u', 'longitude_u', 'ni_u', 'nj_u',
                      'latitude_v', 'longitude_v', 'ni_v', 'nj_v',
                      'level_w',
                      'latitude_f', 'longitude_f',
                      ])
    except ValueError:
        try:
            ds_new = ds.drop(['latitude_u', 'longitude_u', 'ni_u', 'nj_u',
                          'latitude_v', 'longitude_v', 'ni_v', 'nj_v',
                          'level_w',
                          ])
        except:
            if verbose:
                print("no old coordinate dropped - error during ds.drop()")
            ds_new = ds
    except:
        if verbose:
            print("no old coordinate dropped - error during ds.drop()")
        ds_new = ds

    return ds_new

## MAPS ##

def nice_map_mesoNH(data_to_plot, ax=None, vmin=None, vmax=None, cmap='viridis', 
                         add_liaise=False, title=None, label=None,
                         poly=None):
    if ax is None:
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
        cbar_kwargs={"label": label},
    )

    ax.coastlines()
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)

    gl=ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, alpha=0.9)
    gl.right_labels = False
    gl.left_labels = True
    gl.top_labels = False
    gl.bottom_labels = True
    gl.xlocator = plt.MaxNLocator(5)
    gl.ylocator = plt.MaxNLocator(5)
    if title:
        if title == 'off':
            pass
        else:
            ax.set_title(title)
    if add_liaise:
        add_liaise_site_loc(ax=ax)
    #remove gridlines

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
    # plt.show()

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
                         poly=None, title=None, figsize=None, label=None):
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
    
    if not title:
        title = f'{var} on {pd.to_datetime(timestamp).strftime("%Y-%m-%d %H:%M UTC")}'
    else:
        title = title  # use the provided title
    if not label:
        label = f'{var} ({ds[var].attrs.get("units", "")})'
    else:
        label = label  # use the provided label

    if figsize:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        ax=None
    nice_map_mesoNH(data_to_plot,ax=ax,  vmin=vmin, vmax=vmax, cmap=cmap,
                         add_liaise=add_liaise, title=title, label=label, poly=poly)

def map_mesoNH_mean(ds, var, cmap='viridis', vmin=None, vmax=None, title=None, add_liaise=False, poly=None):
    """
    Maps the mean of a variable from a MesoNH dataset.
    """
    if 'time' not in ds[var].dims:
        mean_ds = ds[var]
    else:
        mean_ds = ds[var].mean(dim='time')
    title = title or f'{var} mean'
    label = f'{var} ({ds[var].attrs.get("units", "")})'
    nice_map_mesoNH(mean_ds, cmap=cmap, vmin=vmin, vmax=vmax, title=title, label=label, add_liaise=add_liaise, poly=poly)

def map_mesoNH_mean_restrict(ds, var, cmap='viridis', vmin=None, vmax=None, title=None, add_liaise=False, poly=None,
                             lon_min=None, lon_max=None, lat_min=None, lat_max=None, label=None):
    """
    Maps the mean of a variable from a MesoNH dataset, with limitations on the spatial extent.
    """
    if 'time' not in ds[var].dims:
        data_to_plot_mean_selected = ds[var]
    else:
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
    if label is None:
        # Use the variable's units from attributes if available
        label = f'{var} ({ds[var].attrs.get("units", "")})'
    nice_map_mesoNH(data_to_plot, cmap=cmap, vmin=vmin, vmax=vmax, title=title, label=label, add_liaise=add_liaise, poly=poly)

def map_wind_mesoNH(ds, ax=None, extra_var='wind speed', extra_ds=None, height='10m', title=None, vmin=None, vmax=None, figsize=default_map_figsize, cmap=reds, dist=6, scale=100, clabel=None,
                    lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    if ax==None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        ax=ax

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
        data_to_plot = ds.where(spatial_mask)
    else:
        # If no subsetting is requested, use the time-selected data directly
        data_to_plot = ds

    if 'time' in data_to_plot.dims:
        windvar_u=data_to_plot['u'+height].mean(dim='time')
        windvar_v=data_to_plot['v'+height].mean(dim='time')
    else:
        windvar_u=data_to_plot['u'+height]
        windvar_v=data_to_plot['v'+height]

    #show extra_var in background color
    if extra_var=='wind speed':
        plotvar = (windvar_u**2 + windvar_v**2 ) ** (1/2)
    else:
        if 'time' in extra_ds.dims:
            extra_var = extra_ds[extra_var].mean(dim='time')
        plotvar = extra_ds[extra_var]
    
    nice_map_mesoNH(plotvar, ax, cmap=cmap,  vmin=vmin, vmax=vmax, label=clabel, poly=both_cells, title='off')

    #plot wind vectors
    windx = windvar_u[::dist,::dist]
    # print(windx)
    windy = windvar_v[::dist,::dist]
    longi=data_to_plot['lon'][::dist,::dist]
    # print(longi)
    lati=data_to_plot['lat'][::dist,::dist]
    quiver = ax.quiver(longi, lati, windx, windy, width=0.005, transform=ccrs.PlateCarree(), scale=scale)

    quiverkey_scale = scale/10
    # plt.quiverkey(quiver, X=0.93, Y=0.08, U=quiverkey_scale, label='{} m s⁻¹'.format(quiverkey_scale), labelpos='S',
    #                fontproperties={'weight': 'bold', 'size': 14})
    plt.quiverkey(quiver, X=0.77, Y=0.085, U=quiverkey_scale,
                   label='{} m s⁻¹'.format(quiverkey_scale), labelpos='S',
                #    fontproperties={'weight': 'bold', 'size': 14},
                   coordinates='figure')
    if title:
        if title == 'off':
            pass
        elif title is None:     
            if (height == '10m'):
                plt.title('10m wind (m s⁻¹) and {}'.format(extra_var))
            else :
                plt.title('{} hPa wind (m s⁻¹) and {}'.format(height, extra_var))
        else:
            plt.title(title)


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
    

## HISTOGRAMS ##
def bins_timestamp(ds, var, timestamp, nbins=10,
                   xmin=None, xmax=None, ylim=None,
                   ds_list=None, site=None,
                   xlabel=None, title=None,
                   force_xticks=False
                   ):
    """
    Make a histogram of the values of a variable at a given timestamp.
    """
    ds = ds.sel(time=timestamp)
    values = ds[var].values.flatten()
    values = values[~np.isnan(values)]

    # Set xmin and xmax
    if xmin is None:
        xmin = np.min(values)
    if xmax is None:
        xmax = np.max(values)

    # Create bins
    bins = np.linspace(xmin, xmax, nbins + 1)

    # Create histogram
    hist, edges = np.histogram(values, bins=bins)
    hist = hist / np.sum(hist) * 100.0

    # Plot histogram
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor='orange', align='edge',
            color='orange', alpha=0.4)

    if ylim is not None:
        plt.ylim(ylim)

    if xlabel is None:
        xlabel=var
    elif xlabel == False:
        xlabel = ''
    else:
        xlabel = xlabel  # use the provided xlabel
    plt.xlabel(xlabel)
    plt.ylabel('Frequency (%)')
    plt.grid(which='major', linestyle='-', linewidth=0.5)

    if title is None:
        title = f'Distribution of {var} at {timestamp} {site}'
    elif title == 'off':
        title = ''
    else:
        title = title  # use the provided title
    plt.title(title)

    # Set x-axis ticks: major every 100, minor every 25
    if force_xticks:
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(25))

    # Show the mean
    ds_mean = ds[var].mean(dim=['ni', 'nj']).values
    plt.scatter(ds_mean, 1, color=ds.attrs['plot_color'], marker='o')


    if ds_list is not None:
        for ds1 in ds_list:
            if var not in ds1:
                continue
            else:
                ds1_val = ds1.sel(time=timestamp, method='nearest')[var].values
                color = ds1.attrs['plot_color']
                plt.scatter(ds1_val, 1, color=color, marker='o')
            
def bins_hour(ds, var, hour, nbins=10,
                   xmin=None, xmax=None, title=None, xlabel=None):
    """
    Make a histogram of the values of a variable at a given timestamp.
    """
    ds = ds.where(ds.time.dt.hour==hour).mean(dim='time')
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
    # Convert histogram to percentage
    hist = hist / np.sum(hist) * 100.0
    

    # Plot histogram
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor='black', align='edge')
    plt.xlabel(var)
    plt.ylabel('Frequency (%)')
    if not xlabel:
        xlabel = f'{var} ({ds[var].attrs["units"]})'
    plt.xlabel(xlabel)
    if not title:
        plt.title(f'Distribution of {var} at {hour}UTC (14-31/07/2021)')
    else:
        plt.title(title)
    plt.grid()
    plt.show()