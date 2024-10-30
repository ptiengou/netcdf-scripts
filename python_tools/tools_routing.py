from tools import *

#dict from JP_csv filtered with stations in obs
#keeping only data after 2010
stations_dict_filtered={
    6212700: {'name': 'Peral De Arlanza', 'river': 'Arlanza, Rio', 'lat_grid': 42.07500076293945, 'lon_grid': -4.074992656707764, 'last_record': '1984-08-15'},
    6217700: {'name': 'Pinos Puente', 'river': 'Frailes, Rio', 'lat_grid': 37.27499771118164, 'lon_grid': -3.758326292037964, 'last_record': '1984-08-15'}, 
    # 6217110: {'name': 'Cantillana', 'river': 'Guadalquivir, Rio', 'lat_grid': 37.59166717529297, 'lon_grid': -5.824993133544922, 'last_record': '1992-11-15'}, 
    # 6227100: {'name': 'Bobadilla', 'river': 'Guadalhorce, Rio', 'lat_grid': 37.04166412353516, 'lon_grid': -4.70832633972168, 'last_record': '1984-08-15'}, 
    # 6217120: {'name': 'Sevilla', 'river': 'Guadalquivir, Rio', 'lat_grid': 37.39166641235352, 'lon_grid': -6.008326530456543, 'last_record': '1964-11-15'}, 
    # 6227300: {'name': 'El Chono', 'river': 'Nacimiento, Rio', 'lat_grid': 37.07500076293945, 'lon_grid': -2.6416594982147217, 'last_record': '1984-08-15'}, 
    6217600: {'name': 'El Doctor', 'river': 'Guardal, Rio', 'lat_grid': 37.85833358764648, 'lon_grid': -4.691659450531006, 'last_record': '1984-08-15'}, 
    # 6211500: {'name': 'San Pedro', 'river': 'Sil, Rio', 'lat_grid': 42.44166564941406, 'lon_grid': -7.70832633972168, 'last_record': '1993-09-15'}, 
    6213700: {'name': 'Talavera', 'river': 'Tagus River', 'lat_grid': 39.95833206176758, 'lon_grid': -4.824993133544922, 'last_record': '2013-06-15'}, 
    6226800: {'name': 'Tortosa', 'river': 'Ebro, Rio', 'lat_grid': 40.82500076293945, 'lon_grid': 0.5250073671340942, 'last_record': '2013-09-15'}, 
    6226700: {'name': 'Lecina De Barcabo', 'river': 'Vero, Rio', 'lat_grid': 42.20833206176758, 'lon_grid': 0.0416740216314792, 'last_record': '1984-08-15'}, 
    # 6212420: {'name': 'Puente Pino', 'river': 'Douro, Rio', 'lat_grid': 41.57500076293945, 'lon_grid': -6.191659450531006, 'last_record': '1991-11-15'}, 
    6226300: {'name': 'Castejon', 'river': 'Ebro, Rio', 'lat_grid': 42.17499923706055, 'lon_grid': -1.691659450531006, 'last_record': '2013-09-15'}, 
    6216800: {'name': 'Quintanar', 'river': 'Giguela, Rio', 'lat_grid': 39.64166641235352, 'lon_grid': -3.0749928951263428, 'last_record': '2012-09-15'}, 
    6226650: {'name': 'Fraga', 'river': 'Cinca, Rio', 'lat_grid': 41.52499771118164, 'lon_grid': 0.3416740298271179, 'last_record': '2013-09-15'}, 
    # 6226500: {'name': 'Pitarque', 'river': 'Portanete, Rio', 'lat_grid': 40.625, 'lon_grid': -0.5916593670845032, 'last_record': '1984-08-15'}, 
    # 6216700: {'name': 'Puente Morena', 'river': 'Jabalon, Rio', 'lat_grid': 38.89166641235352, 'lon_grid': -4.024992942810059, 'last_record': '1984-08-15'}, 
    # 6217100: {'name': 'Alcala Del Rio', 'river': 'Guadalquivir, Rio', 'lat_grid': 37.52499771118164, 'lon_grid': -5.974992752075195, 'last_record': '1995-09-15'}, 
    # 6213750: {'name': 'Aranjuez (P. Largo)', 'river': 'Jarama, Rio', 'lat_grid': 40.09166717529297, 'lon_grid': -3.608326196670532, 'last_record': '2009-10-15'}, 
    # 6210100: {'name': 'Echevarri', 'river': 'Nervion, Rio', 'lat_grid': 43.24166488647461, 'lon_grid': -2.9083261489868164, 'last_record': '1990-06-15'}, 
    6226600: {'name': 'Seros', 'river': 'Segre', 'lat_grid': 41.45833206176758, 'lon_grid': 0.4250073730945587, 'last_record': '2013-09-15'}, 
    # 6212500: {'name': 'Benamariel', 'river': 'Esla, Rio', 'lat_grid': 42.375, 'lon_grid': -5.558326244354248, 'last_record': '1984-08-15'}, 
    6212410: {'name': 'Tore', 'river': 'Douro, Rio', 'lat_grid': 41.508331298828125, 'lon_grid': -5.474992752075195, 'last_record': '1984-11-15'}, 
    6212510: {'name': 'Breto', 'river': 'Esla, Rio', 'lat_grid': 41.875, 'lon_grid': -5.758326053619385, 'last_record': '2011-09-15'}, 
    6227500: {'name': 'Masia De Pompo', 'river': 'Jucar, Rio', 'lat_grid': 39.15833282470703, 'lon_grid': -0.6583260297775269, 'last_record': '1987-08-15'}, 
    # 6213600: {'name': 'Alcantara', 'river': 'Tagus River', 'lat_grid': 39.72499847412109, 'lon_grid': -6.891659736633301, 'last_record': '1986-09-15'}, 
    # 6216500: {'name': 'Puente De Palmas', 'river': 'Guadiana, Rio', 'lat_grid': 38.875, 'lon_grid': -6.9749932289123535, 'last_record': '1992-11-15'}, 
    # 6211100: {'name': 'Orense', 'river': 'Mino, Rio', 'lat_grid': 42.34166717529297, 'lon_grid': -7.874992847442627, 'last_record': '1990-11-15'}, 
    6213900: {'name': 'Peralejos', 'river': 'Tagus River', 'lat_grid': 40.59166717529297, 'lon_grid': -1.9249927997589111, 'last_record': '2013-09-15'}, 
    6226400: {'name': 'Zaragoza', 'river': 'Ebro, Rio', 'lat_grid': 41.67499923706055, 'lon_grid': -0.9083260297775269, 'last_record': '1984-11-15'}, 
    # 6212400: {'name': 'Villachica', 'river': 'Douro, Rio', 'lat_grid': 41.508331298828125, 'lon_grid': -5.474992752075195, 'last_record': '1979-11-15'}, 
    6213800: {'name': 'Trillo', 'river': 'Tagus River', 'lat_grid': 40.70833206176758, 'lon_grid': -2.5749928951263428, 'last_record': '2013-09-15'}
    }

#display stations on map
def stations_map_xy(x_values, y_values, title='Location of selected stations'):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.RIVERS)
    ax.set_extent([-10, 6, 35, 45])
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.ylocator = plt.MaxNLocator(5)
    gl.right_labels = False
    gl.top_labels = False
    plt.scatter(x_values, y_values, s=30, marker='o')
    plt.title(title)

def stations_map_dict(stations_dict, river_cond=None, name_cond=None, title='Location of selected stations', legend=True):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.RIVERS)
    ax.set_extent([-10, 6, 35, 45])
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.ylocator = plt.MaxNLocator(5)
    gl.right_labels = False
    gl.top_labels = False
    for key, value in stations_dict.items():
        if river_cond:
            if value['river'] == river_cond:
                plt.scatter(value['lon_grid'], value['lat_grid'], s=30, label=value['name'], marker='o')
        elif name_cond:
            if value['name'] == name_cond:
                plt.scatter(value['lon_grid'], value['lat_grid'], s=30, label=value['name'], marker='o')
        else:
            plt.scatter(value['lon_grid'], value['lat_grid'], s=30, label=value['name'], marker='o')
    plt.title(title)
    if legend:
        plt.legend()

#discharge time plots
def discharge_coord_ts(ds_list, coord_dict, var='hydrographs', figsize=(20, 10), year_min=2010, year_max=2022, title=None, ylabel=None, xlabel=None):
    ncols = len(coord_dict)//2
    fig, axes = plt.subplots(2, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (key, coord) in enumerate(coord_dict.items()):
        ax=axes[i]
        ax.grid()
        lon = coord['lon_grid']
        lat = coord['lat_grid']
        name=coord['name']
        year_min=coord['year_min']
        year_max=coord['year_max']
        # mdate='{}-01-01'.format(year_min)
        # max_date='{}-12-31'.format(year_max)

        for ds in ds_list:
            plotvar=ds[var].where(ds['time.year'] >= year_min, drop=True).where(ds['time.year'] <= year_max, drop=True)
            plotvar=plotvar.sel(lon=lon, lat=lat, method='nearest')
            nice_time_plot(plotvar,ax,label=ds.name, title=name, ylabel=ylabel, xlabel=xlabel)
    
    if not title:
        fig_title = var + (' time series ({})'.format(ds_list[0][var].attrs['units']))
    fig.suptitle(fig_title)
    plt.tight_layout()

def discharge_coord_sc(ds_list, coord_dict, var='hydrographs', figsize=(20, 10), year_min=2010, year_max=2022, title=None, ylabel=None, xlabel=None):
    ncols = len(coord_dict)//2
    fig, axes = plt.subplots(2, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (key, coord) in enumerate(coord_dict.items()):
        ax=axes[i]
        ax.grid()
        ax.set_xticks(np.arange(1,13))
        ax.set_xticklabels(months_name_list)
        lon = coord['lon_grid']
        lat = coord['lat_grid']
        name=coord['name']
        year_min=coord['year_min']
        year_max=coord['year_max']
        # mdate='{}-01-01'.format(year_min)
        # max_date='{}-12-31'.format(year_max)

        for ds in ds_list:
            plotvar=ds[var].where(ds['time.year'] >= year_min, drop=True).where(ds['time.year'] <= year_max, drop=True)
            plotvar=plotvar.sel(lon=lon, lat=lat, method='nearest').groupby('time.month').mean(dim='time')
            nice_time_plot(plotvar,ax,label=ds.name, title=name, ylabel=ylabel, xlabel=xlabel)
    
    if not title:
        fig_title = var + (' time series ({})'.format(ds_list[0][var].attrs['units']))
    fig.suptitle(fig_title)
    plt.tight_layout()

#for stations
def ts_station(stations_ds, ax, station_id, name=None, var='runoff_mean', year_min=2010, year_max=2022, ylabel=None, xlabel=None):
    ax.grid()
    plotds = stations_ds.sel(id=station_id)
    plotvar=plotds[var].where(stations_ds['time.year'] >= year_min, drop=True).where(stations_ds['time.year'] <= year_max, drop=True)
    nice_time_plot(plotvar,ax,label='obs', title=name, ylabel=ylabel, xlabel=xlabel, color='black')

def ts_with_obs(ds_list, stations_ds, ax, station_id, station_data, var='hydrographs', year_min=2010, year_max=2022, ylabel=None, xlabel=None):
    ax.grid()
    station = stations_ds.sel(id=station_id)
    station = station.where(station['time.year'] >= year_min, drop=True).where(station['time.year'] <= year_max, drop=True)
    mask = station['runoff_mean'].notnull()

    lon = station_data['lon_grid']
    lat = station_data['lat_grid']

    for ds in ds_list:
        plotvar=ds[var].sel(lon=lon, lat=lat, method='nearest')
        plotvar=plotvar.where(mask)
        nice_time_plot(plotvar,ax,label=ds.name, title=station_data['name'], ylabel=ylabel, xlabel=xlabel)

def sc_station(stations_ds, ax, station_id, name=None, var='runoff_mean', year_min=2010, year_max=2022, ylabel=None, xlabel=None):
    ax.grid()
    ax.set_xticks(np.arange(1,13))
    ax.set_xticklabels(months_name_list)
    plotds = stations_ds.sel(id=station_id)
    plotvar=plotds[var].where(stations_ds['time.year'] >= year_min, drop=True).where(stations_ds['time.year'] <= year_max, drop=True)
    plotvar=plotvar.groupby('time.month').mean(dim='time')
    nice_time_plot(plotvar,ax,label='obs', title=name, ylabel=ylabel, xlabel=xlabel, color='black')
    
def sc_with_obs(ds_list, stations_ds, ax, station_id, station_data, var='hydrographs', year_min=2010, year_max=2022, ylabel=None, xlabel=None):
    ax.grid()
    ax.set_xticks(np.arange(1,13))
    ax.set_xticklabels(months_name_list)
    station = stations_ds.sel(id=station_id)
    station = station.where(station['time.year'] >= year_min, drop=True).where(station['time.year'] <= year_max, drop=True)
    mask = station['runoff_mean'].notnull()

    lon = station_data['lon_grid']
    lat = station_data['lat_grid']

    for ds in ds_list:
        plotvar=ds[var].sel(lon=lon, lat=lat, method='nearest')
        plotvar=plotvar.where(mask)
        plotvar=plotvar.groupby('time.month').mean(dim='time')
        nice_time_plot(plotvar,ax,label=ds.name, title=station_data['name'], ylabel=ylabel, xlabel=xlabel)

#metrics definition
def metric_sim_module(sim: xr.DataArray, obs: xr.DataArray) -> float:
    """
    Formatted as other metrics but gives only module of sim

    Parameters:
    sim (xr.DataArray): Simulation time series.
    obs (xr.DataArray): Reference time series (observation).

    Returns:
    float: Simulation module.
    """
    # Define function attributes
    metric_sim_module.__name__ = 'Simulated module'
    metric_sim_module.__short_name__ = 'Module (sim, m³/s)'
    metric_sim_module.__colormap__ = wet
    metric_sim_module.__min_value__= 0
    metric_sim_module.__max_value__= None

    # Calculate the module
    module = sim.mean(dim='time')

    return np.round(module.values, 2)  # Extract the scalar value from the xarray object

def metric_obs_module(sim: xr.DataArray, obs: xr.DataArray) -> float:
    """
    Formatted as other metrics but gives only module of obs

    Parameters:
    sim (xr.DataArray): Simulation time series.
    obs (xr.DataArray): Reference time series (observation).

    Returns:
    float: Observation module.
    """
    # Define function attributes
    metric_obs_module.__name__ = 'Observed module'
    metric_obs_module.__short_name__ = 'Module (obs, m³/s)'
    metric_obs_module.__colormap__ = wet
    metric_obs_module.__min_value__= 0
    metric_obs_module.__max_value__= None

    # Calculate the module
    module = obs.mean(dim='time')

    return np.round(module.values, 2)  # Extract the scalar value from the xarray object

def metric_bias(sim: xr.DataArray, obs: xr.DataArray) -> float:
    """
    Calculate the bias between two xarray time series.

    Parameters:
    sim (xr.DataArray): Simulation time series.
    obs (xr.DataArray): Reference time series (observation).

    Returns:
    float: Bias.
    """
    # Define function attributes
    metric_bias.__name__ = 'Bias'
    metric_bias.__short_name__ = 'Bias (m³/s)'
    metric_bias.__colormap__ = emb
    metric_bias.__min_value__= None
    metric_bias.__max_value__= None

    # Calculate the bias
    bias = (sim - obs).mean(dim='time')

    return np.round(bias.values, 2)  # Extract the scalar value from the xarray object

def metric_tcorr(sim: xr.DataArray, obs: xr.DataArray) -> float:
    """
    Calculate the correlation coefficient between two xarray time series.

    Parameters:
    sim (xr.DataArray): Simulation time series.
    obs (xr.DataArray): Reference time series (observation).

    Returns:
    float: Correlation coefficient.
    """
    # Define function attributes
    metric_tcorr.__name__ = 'Pearson correlation coefficient'
    metric_tcorr.__short_name__ = 'r'
    metric_tcorr.__colormap__ = bad_good
    metric_tcorr.__min_value__= -1
    metric_tcorr.__max_value__= 1

    # Calculate the correlation coefficient
    correlation = xr.corr(sim, obs)

    return np.round(correlation.values, 2)  # Extract the scalar value from the xarray object

def metric_nse(sim: xr.DataArray, obs: xr.DataArray) -> float:
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE) between two xarray time series.

    Parameters:
    sim (xr.DataArray): Simulation time series.
    obs (xr.DataArray): Reference time series (observation).

    Returns:
    float: Nash-Sutcliffe Efficiency.
    """

    # Define function attributes
    metric_nse.__name__ = 'Nash-Sutcliffe Efficiency'
    metric_nse.__short_name__ = 'NSE'
    metric_nse.__colormap__ = bad_good
    metric_nse.__min_value__= 0
    metric_nse.__max_value__= 1

    # Calculate the Nash-Sutcliffe Efficiency
    nse = 1 - ( ((sim - obs)**2).sum(dim='time') / ((obs - obs.mean(dim='time'))**2).sum(dim='time') )

    return np.round(nse.values, 2)  # Extract the scalar value from the xarray object

def metric_kge(sim: xr.DataArray, obs: xr.DataArray) -> float:
    """
    Calculate the Kling-Gupta Efficiency (KGE) between two xarray time series.

    Parameters:
    sim (xr.DataArray): Simulation time series.
    obs (xr.DataArray): Reference time series (observation).

    Returns:
    float: Kling-Gupta Efficiency.
    """
    # Define function attributes
    metric_kge.__name__ = 'Kling-Gupta Efficiency'
    metric_kge.__short_name__ = 'KGE'
    metric_kge.__colormap__ = bad_good
    metric_kge.__min_value__= -1
    metric_kge.__max_value__= 1

    # Calculate the Kling-Gupta Efficiency
    kge = 1 - np.sqrt((xr.corr(sim, obs) - 1)**2 + (sim.std(dim='time') / obs.std(dim='time') - 1)**2 + (sim.mean(dim='time') / obs.mean(dim='time') - 1)**2)

    return np.round(kge.values, 2)  # Extract the scalar value from the xarray object

def metric_rmse(sim: xr.DataArray, obs: xr.DataArray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between two xarray time series.

    Parameters:
    sim (xr.DataArray): Simulation time series.
    obs (xr.DataArray): Reference time series (observation).

    Returns:
    float: Root Mean Squared Error.
    """
    # Define function attributes
    metric_rmse.__name__ = 'Root Mean Squared Error'
    metric_rmse.__short_name__ = 'RMSE'
    metric_rmse.__colormap__ = good_bad
    metric_rmse.__min_value__= 0
    metric_rmse.__max_value__= None

    # Calculate the Root Mean Squared Error
    rmse = np.sqrt(((sim - obs)**2).mean(dim='time'))

    return np.round(rmse.values, 2)  # Extract the scalar value from the xarray object

#metrics computation and display
def compute_metric_obs(sim_ds, obs_ds, metric_func) -> float:
    """
    Compute a specified metric for two xarray time series.
    Filters the simulated series based on the obs series.

    Parameters:
    metric_func (function): A function that takes two aligned xarray DataArrays and returns a float.

    Returns:
    float: Result of the metric function applied to the two time series.
    """
    obs=obs_ds['runoff_mean']
    mask=obs.notnull()

    sim=sim_ds['hydrographs']
    sim_filtered=sim.where(mask, drop=True)


    # Ensure the time series have the same time dimension
    sim, obs = xr.align(sim_filtered, obs)

    # Apply the metric function to the aligned time series
    result = metric_func(sim, obs)

    return result

def compute_metric_station(sim_ds, obs_ds, station_id, station, metric_func) -> float:
    """
    Compute a metric for a given station

    Parameters:
    metric_func (function): A function that takes two aligned xarray DataArrays and returns a float.

    Returns:
    float: Result of the metric function applied to the two time series.
    """
    obs=obs_ds.sel(id=station_id)


    lon=station['lon_grid']
    lat=station['lat_grid']
    sim=sim_ds.sel(lon=lon, lat=lat, method='nearest')

    result = compute_metric_obs(sim, obs, metric_func)

    return np.round(result, 2)
    # return result

def display_metric_map(sim_ds: xr.DataArray, obs_ds: xr.DataArray, stations_dict, metric_func, metric_min=None, metric_max=None, title=None, legend=False):
    """
    Calculate a metric for a given list of stations
    Display each station on a map with a color that corresponds to the metric value
    The metric value scale is given by metric_min and metric_max
    """
    fig = plt.figure(figsize=(15, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.RIVERS)
    ax.set_extent([-10, 4, 35, 45])
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.ylocator = plt.MaxNLocator(5)
    gl.right_labels = False
    gl.top_labels = False
    
    for key, station in stations_dict.items():
        metric = compute_metric_station(sim_ds, obs_ds, key, station, metric_func)
        
        cmap = metric_func.__colormap__
        if not metric_min:
            metric_min = metric_func.__min_value__
        if not metric_max:
            metric_max = metric_func.__max_value__
        plt.scatter(station['lon_grid'], station['lat_grid'], label=station['name'], 
                    c=metric, cmap=cmap, 
                    vmin=metric_min, vmax=metric_max, 
                    s=120,marker='o', 
                    edgecolors='black',  # Add black edges around markers
                    linewidths=0.5,  # Edge line width
                    )
    # plt.scatter([station['lon_grid'] for key, station in stations_dict.items()],
    #             [station['lat_grid'] for key, station in stations_dict.items()],
    #             label=[station['name'] for key, station in stations_dict.items()],
    #             s=30, c=metric_values, cmap=cmap, vmin=metric_min, vmax=metric_max, marker='o')
    
    plt.colorbar(label='{}'.format(metric_func.__short_name__))
    if not title:
        plt.title('{}'.format(metric_func.__name__))
    else:
        plt.title(title)
    #add legend outside of the plot
    if legend:
        plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1))

def display_metric_diff_map(sim1_ds: xr.DataArray, sim2_ds:xr.DataArray, obs_ds: xr.DataArray, stations_dict, metric_func, metric_min=None, metric_max=None, title=None, legend=False):
    """
    Calculate a metric for a given list of stations
    Substract the metric values of sim2 from sim1
    Display each station on a map with a color that corresponds to the metric value
    The metric value scale is given by metric_min and metric_max
    """
    fig = plt.figure(figsize=(15, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.RIVERS)
    ax.set_extent([-10, 4, 35, 45])
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.ylocator = plt.MaxNLocator(5)
    gl.right_labels = False
    gl.top_labels = False
    
    for key, station in stations_dict.items():
        metric1 = compute_metric_station(sim1_ds, obs_ds, key, station, metric_func)
        metric2 = compute_metric_station(sim2_ds, obs_ds, key, station, metric_func)
        metric = metric1 - metric2
        cmap = metric_func.__colormap__
        # if not metric_min:
        #     metric_min = metric_func.__min_value__
        # if not metric_max:
        #     metric_max = metric_func.__max_value__
        plt.scatter(station['lon_grid'], station['lat_grid'], label=station['name'], 
                    c=metric, cmap=cmap, 
                    vmin=metric_min, vmax=metric_max, 
                    s=120,marker='o', 
                    edgecolors='black',  # Add black edges around markers
                    linewidths=0.5,  # Edge line width
                    )
    # plt.scatter([station['lon_grid'] for key, station in stations_dict.items()],
    #             [station['lat_grid'] for key, station in stations_dict.items()],
    #             label=[station['name'] for key, station in stations_dict.items()],
    #             s=30, c=metric_values, cmap=cmap, vmin=metric_min, vmax=metric_max, marker='o')
    
    plt.colorbar(label='{} variation'.format(metric_func.__short_name__))
    if not title:
        plt.title('{} difference ({} - {})'.format(metric_func.__name__, sim1_ds.name, sim2_ds.name))
    else:
        plt.title(title)
    #add legend outside of the plot
    if legend:
        plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1))