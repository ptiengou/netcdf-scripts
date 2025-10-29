from tools import *
from tools_mapping import *

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

proper_stations_dict = {
    6226800: {'name': 'Tortosa',            'river': 'Ebro',            'lat_grid': 40.82500,   'lon_grid': 0.525007,   'station_nb': 1},
    6226400: {'name': 'Zaragoza',           'river': 'Ebro',            'lat_grid': 41.67499,   'lon_grid': -0.90832,   'station_nb': 2},
    6226300: {'name': 'Castejon',           'river': 'Ebro',            'lat_grid': 42.17499,   'lon_grid': -1.69165,   'station_nb': 3},
    6226600: {'name': 'Seros',              'river': 'Segre',           'lat_grid': 41.45833,   'lon_grid': 0.425007,   'station_nb': 4},
    6226650: {'name': 'Fraga',              'river': 'Cinca',           'lat_grid': 41.52499,   'lon_grid': 0.341674,   'station_nb': 5},
    6212410: {'name': 'Tore',               'river': 'Douro',           'lat_grid': 41.50833,   'lon_grid': -5.47499,   'station_nb': 6},
    6212700: {'name': 'Peral De Arlanza',   'river': 'Arlanza',         'lat_grid': 42.07500,   'lon_grid': -4.07499,   'station_nb': 7},
    6213700: {'name': 'Talavera',           'river': 'Tagus',           'lat_grid': 39.95833,   'lon_grid': -4.82499,   'station_nb': 8},
    6213800: {'name': 'Trillo',             'river': 'Tagus',           'lat_grid': 40.70833,   'lon_grid': -2.57499,   'station_nb': 9},
    6213900: {'name': 'Peralejos',          'river': 'Tagus',           'lat_grid': 40.59166,   'lon_grid': -1.92499,   'station_nb': 10},
    6216510: {'name': 'Azud de Badajoz',    'river': 'Guadiana',        'lat_grid': 38.86199,   'lon_grid': -7.01,      'station_nb': 11}, 
    6116200: {'name': 'Pulo do Lobo',       'river': 'Guadiana',        'lat_grid': 37.803,     'lon_grid': -7.633,     'station_nb': 12},         
    6216530: {'name': 'La Cubeta',          'river': 'Guadiana',        'lat_grid': 38.975,     'lon_grid': -2.895,     'station_nb': 13},         
    6216520: {'name': 'Villarubia',         'river': 'Guadiana',        'lat_grid': 39.125,     'lon_grid': -3.59073,   'station_nb': 14},      
    6216800: {'name': 'Quintanar',          'river': 'Giguela',         'lat_grid': 39.64166,   'lon_grid': -3.07499,   'station_nb': 15},
    6217140: {'name': 'Mengibar',           'river': 'Guadalquivir',    'lat_grid': 37.98425,   'lon_grid': -3.79939,   'station_nb': 16},     
    6217200: {'name': 'Arroyo Maria',       'river': 'Guadalquivir',    'lat_grid': 38.17905,   'lon_grid': -2.83594,   'station_nb': 17}, 
    6217700: {'name': 'Pinos Puente',       'river': 'Frailes',         'lat_grid': 37.27499,   'lon_grid': -3.75832,   'station_nb': 18},
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

def stations_map_dict(stations_dict, river_cond=None, name_cond=None, title='Location of selected stations', legend=True, extent=[-10, 2.5, 35, 45], dams_df=None, dam_nb=50):
    """
    Plots the stations from a dictionary on a map, filtering by river or name if specified.
    Automatically assigns unique colors to each station based on the number of stations plotted.

    Args:
        stations_dict (dict): Dictionary containing station information.
        river_cond (str, optional): Filter stations by river name.
        name_cond (str, optional): Filter stations by station name.
        title (str, optional): Title of the map. Defaults to 'Location of selected stations'.
        legend (bool, optional): Whether to include a legend. Defaults to True.
    """
    # Filter stations based on conditions
    filtered_stations = {
        key: value for key, value in stations_dict.items()
        if (river_cond and value['river'] == river_cond) or
           (name_cond and value['name'] == name_cond) or
           (not river_cond and not name_cond)
    }

    # Generate colors based on the number of filtered stations
    num_stations = len(filtered_stations)
    cmap = plt.get_cmap('tab20')  # Choose your colormap
    colors = [cmap(i) for i in np.linspace(0, 1, num_stations)]

    # Setup map
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAND, color='lightyellow', edgecolor='black')
    ax.grid()
    ax.set_extent(extent)

    # Overlay river names as labels
    river_data = [
        {"name": "Ebro", "lon": -2.4, "lat": 42.75},
        {"name": "Douro", "lon": -4.2, "lat": 41.35},
        {"name": "Tagus", "lon": -6.7, "lat": 39.5},
        {"name": "Guadiana", "lon": -4.6, "lat": 38.7},
        {"name": "Guadalquivir", "lon": -4.9, "lat": 37.45},
        # {"name": "Cinca", "lon": 0.34, "lat": 41.52},
        # {"name": "Segre", "lon": 0.41, "lat": 41.45},
        # {"name": "Arlanza", "lon": -4.07, "lat": 42.08},
        # {"name": "Frailes", "lon": -3.76, "lat": 37.25},
        # {"name": "Giguela", "lon": -3.08, "lat": 39.64},
    ]

    # Plot river names
    for river in river_data:
        ax.text(
            river["lon"], river["lat"], river["name"],
            fontsize=14, 
            # color=(180/255, 200/255, 250/255, 1),
            color='steelblue',
            weight="normal",
            transform=ccrs.PlateCarree(),
            ha="center", va="center",
        )
    
    if dams_df is not None:
        # Plot each dam
        dams_df=dams_df.nlargest(dam_nb, 'capacity')
        for _, row in dams_df.iterrows():
            name=row.Name
            # number=row.ID
            label=name
            plt.scatter(
                row.lon, row.lat,
                s=50, marker='D', color='silver'
            )

    # Gridlines
    # gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    # gl.ylocator = plt.MaxNLocator(5)
    # gl.right_labels = False
    # gl.top_labels = False

    # Plot each station with its unique color
    for idx, (key, value) in enumerate(filtered_stations.items()):
        name=value['name']
        number=value['station_nb']
        label='{} ({})'.format(number, name)
        plt.scatter(
            value['lon_grid'], value['lat_grid'],
            s=100, label=label, marker='o', color=colors[idx]
        )
        # Add text next to the point
        plt.text(
            value['lon_grid'] + 0.11, value['lat_grid'] - 0.15,  # Adjust the offset for better placement
            str(number), fontsize=15, color='black'
        )


    # Add title and legend
    if title:
        plt.title(title)
    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.5)

def dams_map(df, title='Location of selected dams', legend=False, extent=[-10, 2.5, 35, 45]):
    """
    Plots the dams from a dictionary on a ma
    """

    # Setup map
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    # ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.NaturalEarthFeature(
        'physical', 'rivers_lake_centerlines', '10m'),
        facecolor='none', edgecolor='blue', linewidth=0.5)
    ax.set_extent(extent)

    # Overlay river names as labels
    river_data = [
        {"name": "Ebro", "lon": -2.4, "lat": 42.75},
        {"name": "Douro", "lon": -3.2, "lat": 41.35},
        {"name": "Tagus", "lon": -6.7, "lat": 39.5},
        {"name": "Guadiana", "lon": -5, "lat": 38.8},
        {"name": "Guadalquivir", "lon": -4.9, "lat": 37.45},
        # {"name": "Cinca", "lon": 0.34, "lat": 41.52},
        # {"name": "Segre", "lon": 0.41, "lat": 41.45},
        # {"name": "Arlanza", "lon": -4.07, "lat": 42.08},
        # {"name": "Frailes", "lon": -3.76, "lat": 37.25},
        # {"name": "Giguela", "lon": -3.08, "lat": 39.64},
    ]

    # Plot river names
    for river in river_data:
        ax.text(
            river["lon"], river["lat"], river["name"],
            fontsize=14, 
            # color=(180/255, 200/255, 250/255, 1),
            color='steelblue',
            weight="normal",
            transform=ccrs.PlateCarree(),
            ha="center", va="center",
        )

    # Gridlines
    # gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    # gl.ylocator = plt.MaxNLocator(5)
    # gl.right_labels = False
    # gl.top_labels = False

    # Plot each dam
    for _, row in df.iterrows():
        name=row.Name
        # number=row.ID
        label=name
        plt.scatter(
            row.lon, row.lat,
            s=50, label=label, marker='D', color='grey'
        )

    # Add title and legend
    # if title:
    #     plt.title(title)
    # if legend:
    #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.5)

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
def ts_station(stations_ds, ax, station_id, name=None, var='runoff_mean', year_min=2010, year_max=2022, ylabel=None, xlabel=None, polcher_ds=False):
    ax.grid()
    if polcher_ds:
        mask = (stations_ds['id'] == station_id)
        plotds = stations_ds.sel(stations=mask)
    else:
        plotds = stations_ds.sel(id=station_id)
    plotvar=plotds[var].where(stations_ds['time.year'] >= year_min, drop=True).where(stations_ds['time.year'] <= year_max, drop=True)
    nice_time_plot(plotvar,ax,label='obs', title=name, ylabel=ylabel, xlabel=xlabel, color='black')

def ts_with_obs(ds_list, stations_ds, ax, station_id, station_data, var='hydrographs', year_min=2010, year_max=2022, ylabel=None, xlabel=None, plot_all_sim=False, polcher_ds=False, title_letter=None):
    ax.grid()
    if polcher_ds:
        mask = (stations_ds['id'] == station_id)
        station = stations_ds.sel(stations=mask)
    else:
        station = stations_ds.sel(id=station_id)
    station = station.where(station['time.year'] >= year_min, drop=True).where(station['time.year'] <= year_max, drop=True)
    mask = station['runoff_mean'].notnull()

    lon = station_data['lon_grid']
    lat = station_data['lat_grid']

    for ds in ds_list:
        plotvar=ds[var].sel(lon=lon, lat=lat, method='nearest')
        if not plot_all_sim:
            plotvar=plotvar.where(mask)
        else:
            plotvar = plotvar.where(ds['time.year'] >= year_min, drop=True).where(ds['time.year'] <= year_max, drop=True)
        name=station_data['name']
        nb=station_data['station_nb']
        river=station_data['river']
        if title_letter:
            title= '({}) Station {} ({}, on river {})'.format(title_letter, nb, name, river)
        else:
            title= 'Station {} ({}, on river {})'.format(nb, name, river)
        nice_time_plot(plotvar,ax,label=ds.name, title=title, color=ds.attrs['plot_color'], ylabel=ylabel, xlabel=xlabel)

def sc_station(stations_ds, ax, station_id, name=None, var='runoff_mean', year_min=2010, year_max=2022, ylabel=None, xlabel=None, polcher_ds=False, title=None):
    ax.grid()
    ax.set_xticks(np.arange(1,13))
    ax.set_xticklabels(months_name_list)
    if polcher_ds:
        mask = (stations_ds['id'] == station_id)
        plotds = stations_ds.sel(stations=mask)
    else:
        plotds = stations_ds.sel(id=station_id)
    plotvar=plotds[var].where(stations_ds['time.year'] >= year_min, drop=True).where(stations_ds['time.year'] <= year_max, drop=True)
    plotvar=plotvar.groupby('time.month').mean(dim='time')
    if not title:
        title = name
    nice_time_plot(plotvar,ax,label='obs', title=title, ylabel=ylabel, xlabel=xlabel, color='black')
    
def sc_with_obs(ds_list, stations_ds, ax, station_id, station_data, var='hydrographs', year_min=2010, year_max=2022, ylabel=None, xlabel=None, title_letter=None, polcher_ds=False, plot_all_sim=False, title=None, title_number=True):
    ax.grid()
    ax.set_xticks(np.arange(1,13))
    ax.set_xticklabels(months_name_list)
    if polcher_ds:
        mask = (stations_ds['id'] == station_id)
        station = stations_ds.sel(stations=mask)
    else:
        station = stations_ds.sel(id=station_id)
    station = station.where(station['time.year'] >= year_min, drop=True).where(station['time.year'] <= year_max, drop=True)
    mask = station['runoff_mean'].notnull()

    lon = station_data['lon_grid']
    lat = station_data['lat_grid']

    for ds in ds_list:
        plotvar=ds[var].sel(lon=lon, lat=lat, method='nearest')
        if not plot_all_sim:
            plotvar=plotvar.where(mask)
        plotvar=plotvar.groupby('time.month').mean(dim='time')
        name=station_data['name']
        nb=station_data['station_nb']
        river=station_data['river']
        if not title:
            if title_letter:
                if title_number:
                    title= '({}) Station {} ({}, on river {})'.format(title_letter, nb, name, river)
                else:
                    title= '({}) Station {}, on river {}'.format(title_letter, name, river)
            else:
                if title_number:
                    title= 'Station {} ({}, on river {})'.format(nb, name, river)
                else:
                    title= 'Station {}, on river {})'.format( name, river)
                    
        nice_time_plot(plotvar,ax,label=ds.name, title=title, color=ds.attrs['plot_color'], ylabel=ylabel, xlabel=xlabel)

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
    metric_sim_module.__diff_colormap__ = wet
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
    metric_obs_module.__diff_colormap__ = wet
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
    metric_bias.__colormap__ = emb_neutral
    metric_bias.__diff_colormap__ = emb_neutral
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
    metric_tcorr.__diff_colormap__ = bad_goodW
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
    metric_nse.__diff_colormap__ = bad_goodW
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
    metric_kge.__diff_colormap__ = bad_goodW
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
    metric_rmse.__diff_colormap__ = good_badW
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
        cmap = metric_func.__diff_colormap__
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

def percent_valid_coverage(ds, start_year, end_year, time_var='time', data_var=None):
    """
    Calculates the percentage of valid (non-NaN) monthly values in a NetCDF time series
    between start_year and end_year (inclusive), using an open xarray Dataset.

    Parameters:
        ds (xarray.Dataset): Opened xarray dataset.
        start_year (int): Start year (inclusive).
        end_year (int): End year (inclusive).
        time_var (str): Name of the time variable.
        data_var (str): Name of the data variable. If None, uses the first variable.

    Returns:
        float: Percentage coverage (0 to 100).
    """
    if data_var is None:
        data_var = list(ds.data_vars)[0]
    
    data = ds[data_var]
    time = ds[time_var]
    years = time.dt.year

    # Filter by year range
    mask = (years >= start_year) & (years <= end_year)
    filtered_data = data.sel({time_var: mask})
    filtered_time = time.sel({time_var: mask})

    valid_count = np.count_nonzero(~np.isnan(filtered_data))
    total_months = filtered_time.size

    if total_months == 0:
        return 0.0
    # print(valid_count, total_months)
    return round((valid_count / total_months) * 100, 2)