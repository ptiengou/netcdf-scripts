from tools import *

# times_correspondance={1.5: '01:30', 4.5: '04:30', 7.5: '07:30', 10.5: '10:30', 13.5: '13:30', 16.5: '16:30', 19.5: '19:30', 22.5: '22:30'}
# times_correspondance={0.0: '00:00', 0.5: '00:30', 1.0: '01:00', 1.5: '01:30', 2.0: '02:00', 2.5: '02:30', 3.0: '03:00', 3.5: '03:30', 4.0: '04:00', 4.5: '04:30', 5.0: '05:00', 5.5: '05:30', 6.0: '06:00', 6.5: '06:30', 7.0: '07:00', 7.5: '07:30', 8.0: '08:00', 8.5: '08:30', 9.0: '09:00', 9.5: '09:30', 10.0: '10:00', 10.5: '10:30', 11.0: '11:00', 11.5: '11:30', 12.0: '12:00', 12.5: '12:30', 13.0: '13:00', 13.5: '13:30', 14.0: '14:00', 14.5: '14:30', 15.0: '15:00', 15.5: '15:30', 16.0: '16:00', 16.5: '16:30', 17.0: '17:00', 17.5: '17:30', 18.0: '18:00', 18.5: '18:30', 19.0: '19:00', 19.5: '19:30', 20.0: '20:00', 20.5: '20:30', 21.0: '21:00', 21.5: '21:30', 22.0: '22:00', 22.5: '22:30', 23.0: '23:00', 23.5: '23:30'}
times_correspondance_1h={0.5: '00:00',1.5: '01:00',2.5: '02:00',3.5: '03:00',4.5: '04:00',5.5: '05:00',6.5: '06:00',7.5: '07:00',8.5: '08:00',9.5: '09:00',10.5: '10:00',11.5: '11:00',12.5: '12:00',13.5: '13:00',14.5: '14:00',15.5: '15:00',16.5: '16:00',17.5: '17:00',18.5: '18:00',19.5: '19:00',20.5: '20:00',21.5: '21:00',22.5: '22:00',23.5: '23:00'}
times_correspondance_30mn = {0.25: '00:00',0.75: '00:30',1.25: '01:00',1.75: '01:30',2.25: '02:00',2.75: '02:30',3.25: '03:00',3.75: '03:30',4.25: '04:00',4.75: '04:30',5.25: '05:00',5.75: '05:30',6.25: '06:00',6.75: '06:30',7.25: '07:00',7.75: '07:30',8.25: '08:00',8.75: '08:30',9.25: '09:00',9.75: '09:30',10.25: '10:00',10.75: '10:30',11.25: '11:00',11.75: '11:30',12.25: '12:00',12.75: '12:30',13.25: '13:00',13.75: '13:30',14.25: '14:00',14.75: '14:30',15.25: '15:00',15.75: '15:30',16.25: '16:00',16.75: '16:30',17.25: '17:00',17.75: '17:30',18.25: '18:00',18.75: '18:30',19.25: '19:00',19.75: '19:30',20.25: '20:00',20.75: '20:30',21.25: '21:00',21.75: '21:30',22.25: '22:00',22.75: '22:30',23.25: '23:00',23.75: '23:30'}
nice_day_print={
    '1507':'15 July 2021',
    '1607':'16 July 2021',
    '1707':'17 July 2021',
    '2007':'20 July 2021',
    '2107':'21 July 2021',
    '2207':'22 July 2021',
    '2707':'27 July 2021',
}

## DIURNAL CYCLES ##
#generic function
def diurnal_cycle_ax(ax, title=None, ylabel=None, xlabel=None, vmin=None, vmax=None):
    ax.grid()
    ax.set_xticks(np.arange(1, 24, 3))
    tick_positions = np.arange(1, 24, 2)
    tick_labels = [f"{int(t)}:{int((t % 1) * 60):02d}" for t in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if vmin is not None and vmax is not None:
        ax.set_ylim(vmin, vmax)

#for one var, several datasets, averaged over lon and lat
def diurnal_cycle_ave(ds_list, var, figsize=(6, 6), ds_colors=False, title=None, ylabel=None, xlabel=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=figsize)
    if not title:
        title = var + (' ({})'.format(ds_list[0][var].attrs['units']))
    for ds in ds_list:        
        # Check if 'lon' and 'lat' are dimensions in the dataset
        if 'lon' in ds.dims and 'lat' in ds.dims:
            plotvar = ds[var].mean(dim=['lon', 'lat']).groupby('time_decimal').mean(dim='time')
        else:
            plotvar = ds[var].groupby('time_decimal').mean(dim='time')
        nice_time_plot(plotvar, ax, label=ds.name, color=ds.attrs["plot_color"] if ds_colors else None)
    diurnal_cycle_ax(ax, title=title, ylabel=ylabel, xlabel=xlabel, vmin=vmin, vmax=vmax)

#for one var, several datasets, at a specific lon and lat
def diurnal_cycle_lonlat(ds_list, var, lon, lat, figsize=(6, 6), ds_colors=False, title=None, ylabel=None, xlabel=None):
    fig, ax = plt.subplots(figsize=figsize)
    if not title:
        title = ds_list[0][var].attrs['long_name'] + (' at ({}°N,{}°W) ({})'.format(lon, lat, ds_list[0][var].attrs['units']))
    for ds in ds_list:
        plotvar = ds[var].sel(lon=lon, lat=lat, method='nearest').groupby('time_decimal').mean(dim='time')
        nice_time_plot(plotvar, ax, label=ds.name, color=ds.attrs["plot_color"] if ds_colors else None)
    diurnal_cycle_ax(ax, title=title, ylabel=ylabel, xlabel=xlabel)

##Energy budget plots##
# Define colors for each variable in energy budget
nrg_colors = {
    'sens': 'red',
    'flat': 'blue',
    'swnet': 'orange',
    'lwnet': 'green',
    'Qg': 'brown'
}

def energy_budget_dc(ds_lmdz1, ds_lmdz2, ds_orc1, ds_orc2, title=None, lab1=None, lab2=None):
    vars_lmdz = ['sens', 'flat']
    var_orc = ['swnet', 'lwnet', 'Qg']
    fig, ax = plt.subplots(figsize=(8, 6))

    ylabel = 'W/m²'
    xlabel = 'Time (UTC)'

    for ds_lmdz, ds_orc in zip([ds_lmdz1, ds_lmdz2], [ds_orc1, ds_orc2]):
        for var in vars_lmdz:
            label = var if ds_lmdz is ds_lmdz2 else ""
            linestyle = '-' if ds_lmdz is ds_lmdz1 else '--'
            plotvar = ds_lmdz[var].groupby('time_decimal').mean(dim='time')
            nice_time_plot(plotvar, ax=ax, label=label, linestyle=linestyle, color=nrg_colors[var])

        for var in var_orc:
            label = var if ds_orc is ds_orc2 else ""
            linestyle = '-' if ds_orc is ds_orc1 else '--'
            plotvar = ds_orc[var].groupby('time_decimal').mean(dim='time')
            nice_time_plot(plotvar, ax=ax, label=label, linestyle=linestyle, color=nrg_colors[var])

    # Add legend only once
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())
    # Add custom legend entries for line styles
    custom_lines = [
        plt.Line2D([0], [0], color='black', linestyle='-', label=lab1),
        plt.Line2D([0], [0], color='black', linestyle='--', label=lab2)
    ]

    # Combine custom lines with unique variable labels
    ax.legend(handles=list(unique_labels.values()) + custom_lines)

    diurnal_cycle_ax(ax, title=title, ylabel=ylabel, xlabel=xlabel)
    ax.grid()

## VERTICAL PROFILES ##

#generic function
def profile_altitudes_ax(ax, x_var, y_coord, nbins=None, plot_label='', title=None, xlabel=None, ylabel=None, xmin=None, xmax=None, ymin=None, ymax=None, linestyle=None, color=None, alpha=1):
    # Ensure x_var and y_coord are numpy arrays for masking
    x_var = np.asarray(x_var)
    y_coord = np.asarray(y_coord)

    # Check if x_var and y_coord have the same size
    if x_var.shape != y_coord.shape:
        raise ValueError("x_var and y_coord must have the same shape.")

    # Create a mask for y_coord values within the specified range
    if ymin is not None and ymax is not None:
        mask = (y_coord >= ymin) & (y_coord <= ymax)
    elif ymin is not None:
        mask = (y_coord >= ymin)
    elif ymax is not None:
        mask = (y_coord <= ymax)
    else:
        mask = np.ones_like(y_coord, dtype=bool)  # No masking if ymin and ymax are None

    # Apply the mask to both x_var and y_coord
    x_var = x_var[mask]
    y_coord = y_coord[mask]
    
    #plotting
    ax.plot(x_var, y_coord, label=plot_label, color=color, linestyle=linestyle, alpha=alpha)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Adjust x-axis tick labels to avoid overlap
    if nbins is not None:
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=nbins))
    # plt.xticks(rotation=45)  # Rotate labels
    ax.legend()
    ax.grid(True)

#plot a profile from 3D variable on pressure levels (not used)
def profile_preslevs_ave(ds_list, var, figsize=(6,8), preslevelmax=20, title=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.grid()
    plt.gca().invert_yaxis()
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(var + ' vertical profile')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_xlabel(var + ' ({})'.format(ds_list[0][var].attrs['units']))
    pressure=ds_list[0]['presnivs'][0:preslevelmax]/100 #select levels and convert from Pa to hPa
    for ds in ds_list:
        plotvar = ds[var].mean(dim=['time', 'lon', 'lat'])[0:preslevelmax]
        ax.plot(plotvar, pressure, label=ds.name) 
    ax.legend()

def profile_preslevs_local(ds_list, var, figsize=(6,8), preslevelmax=20, title=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.grid()
    plt.gca().invert_yaxis() #for pressure levels
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(var + ' vertical profile')
    pressure=ds_list[0]['presnivs'][0:preslevelmax]/100 #select levels and convert from Pa to hPa
    for ds in ds_list:
        plotvar = ds[var].mean(dim=['time'])[0:preslevelmax]
        ax.plot(plotvar, pressure, label=ds.name) 
    ax.set_ylabel('Pressure (hPa)')
    ax.set_xlabel(var + ' ({})'.format(ds_list[0][var].attrs['units']))
    ax.legend()

#plot a profile from 3D variable with altitude as y_coord
def profile_altitude_local_mean(ds_list, var, obs_ds_list=None, figsize=(6,8), ax=None, title=None, altmin=-0, altmax=2000, nbins=None, substract_gl=True, xmin=None, xmax=None, alpha=1.):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for ds in ds_list:
        if 'time' in ds.dims:
            # print('Computing temporal mean')
            # Compute the mean if there are multiple time points
            if substract_gl:
                altitude = ds['geoph'].mean(dim='time').compute() - ds['ground_level'].mean(dim='time').compute().item()
            else:
                altitude = ds['geoph'].mean(dim='time').compute()

            plotvar = ds[var].mean(dim='time').compute()
        else:
            # Use the data directly if 'time' dimension is not present
            # print('No time dimension used')
            if substract_gl:
                altitude = ds['geoph'] - ds['ground_level'].item()
            else:
                altitude = ds['geoph']

            plotvar = ds[var]

        x_var = plotvar.values
        y_coord = altitude.values
        if 'name' in ds.attrs:
            plot_label = ds.name
        else:
            plot_label = None
        if 'units' in ds[var].attrs:
            xlabel = f"{var} ({ds[var].attrs['units']})"
        else:
            xlabel = var
        profile_altitudes_ax(ax, x_var, y_coord, nbins=nbins, plot_label=plot_label, title=title,
                             xlabel=xlabel, 
                             ylabel=('Height agl (m)' if substract_gl else 'Altitude (m)'),
                             xmin=xmin, xmax=xmax, ymin=altmin, ymax=altmax,
                             linestyle=ds.attrs.get('linestyle', None),
                             color=ds.attrs.get('plot_color', None),
                             alpha=alpha)
        
    if obs_ds_list is not None:
        profile_altitude_obs(obs_ds_list, var, ax=ax, title=title, altmin=altmin, altmax=altmax, substract_gl=substract_gl, nbins=nbins, xmin=xmin, xmax=xmax)

def profile_altitude_local_timestamp(ds_list, var, timestamp,  obs_ds_list=None, figsize=(6, 8), ax=None, title=None, altmin=0, altmax=2000, nbins=None, substract_gl=True, xmin=None, xmax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for ds in ds_list:
        # Filter the dataset by the specified timestamp
        ds = ds.sel(time=timestamp, method='nearest')

        if substract_gl:
            altitude = ds['geoph'] - ds['ground_level']
        else:
            altitude = ds['geoph']
        plotvar = ds[var]

        x_var=plotvar.values
        y_coord=altitude.values

        profile_altitudes_ax(ax, x_var, y_coord,
                             nbins=nbins,
                             plot_label=ds.name,
                             title=title,
                             xlabel=f"{var} ({ds_list[0][var].attrs['units']})",
                             ylabel=('Height agl (m)' if substract_gl else 'Altitude (m)'),
                             xmin=xmin, xmax=xmax, ymin=altmin, ymax=altmax)
    if obs_ds_list is not None:
        profile_altitude_obs(obs_ds_list, var, ax=ax, title=title, altmin=altmin, altmax=altmax, substract_gl=substract_gl, nbins=nbins)
    
def profile_altitude_multipletimes_mean(ds_list, var, times, altmin=0, altmax=2000, xmin=None, xmax=None, substract_gl=True, simfreq='1h'):
    n_ax = len(times)
    fig, axs = plt.subplots(1, n_ax, figsize=(5.5*n_ax, 6))
    # Flatten axs only if it's an array (i.e., more than one subplot)
    axes = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
    for i, time in enumerate(times):
        if simfreq=='1h':
            hour=times_correspondance_1h[time]
        elif simfreq=='30mn':
            hour=times_correspondance_30mn[time]
        hour=times_correspondance[time]
        title = f"{var} at {hour}"
        
        # Filter datasets by the specified time and plot
        ds_list_tmp = [ds.where(ds['time_decimal']==time) for ds in ds_list]
        profile_altitude_local_mean(ds_list_tmp,
                                    var,
                                    ax=axes[i],
                                    title=title,
                                    altmin=altmin,
                                    altmax=altmax,
                                    substract_gl=substract_gl,
                                    xmin=xmin, 
                                    xmax=xmax
                                    )
        plt.tight_layout()

def profile_altitude_multipletimes_mean_singleplot(ds, var, times, altmin=0, altmax=2000, xmin=None, xmax=None, substract_gl=True, title=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define a transparency scale based on the number of times
    alpha_scale = np.linspace(0.2, 1, len(times))
    title = f"{var} evolution in {ds.attrs['name']}" if title is None else title

    for i, time in enumerate(times):
        hour = times_correspondance[time]

        # Filter datasets by the specified time and plot
        ds_list_tmp = [ds.where(ds['time_decimal'] == time)]
        profile_altitude_local_mean(ds_list_tmp,
                                    var,
                                    ax=ax,
                                    title=title if i == 0 else None,  # Only set title once
                                    altmin=altmin,
                                    altmax=altmax,
                                    substract_gl=substract_gl,
                                    xmin=xmin,
                                    xmax=xmax,
                                    alpha=alpha_scale[i]  # Set transparency
                                    )

  # Create custom legend entries for transparency scale
    transparency_handles = [
        plt.Line2D([], [], color='gray', alpha=alpha, label=f'{times_correspondance[time]}')
        for alpha, time in zip(alpha_scale, times)
    ]

    # Add the first legend (existing one)
    # ax.legend(loc='upper left', title='Data')

    # Add the second legend for transparency scale
    transparency_legend = plt.legend(handles=transparency_handles, 
                                    #  loc='upper right', 
                                     title='Times')
    ax.add_artist(transparency_legend)

    plt.tight_layout()

def profile_altitude_obs(ds_list, var, figsize=(6,8), ax=None, title=None, altmin=-0, altmax=2000, substract_gl=True, nbins=None, xmin=None, xmax=None, altsite=0):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for ds in ds_list:
        if substract_gl:
            ds['ground_level'] = ds['altitude'][0]
            altitude = ds['altitude'] - ds['ground_level']
        else:
            altitude = ds['altitude'] + altsite
        plotvar = ds[var]

        x_var = plotvar.values
        y_coord = altitude.values
        if 'name' in ds.attrs:
            plot_label = ds.name
        else:
            plot_label = None
        if 'units' in ds[var].attrs:
            xlabel = f"{var} ({ds[var].attrs['units']})"
        else:
            xlabel = var
        profile_altitudes_ax(ax, x_var, y_coord, nbins=nbins, plot_label=plot_label, title=title,
                             xlabel=xlabel, 
                             ylabel=('Height agl (m)' if substract_gl else 'Altitude (m)'),
                             xmin=xmin, xmax=xmax, ymin=altmin, ymax=altmax,
                             linestyle=ds.attrs.get('linestyle', None),
                             color=ds.attrs.get('plot_color', None))
        
def profile_altitude_multipletimes_obs(ds_list, obs_dict, var, times, altmin=0, altmax=2000, xmin=None, xmax=None, substract_gl=True, simfreq='1h', title=None, altsite=0):
    n_ax = len(times)
    fig, axs = plt.subplots(1, n_ax, figsize=(5.5*n_ax, 6))
    fig.suptitle(title)
    # Flatten axs only if it's an array (i.e., more than one subplot)
    axes = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
    if simfreq=='1h':
        times_correspondance=times_correspondance_1h
        offset=0.5
    elif simfreq=='30mn':
        times_correspondance=times_correspondance_30mn
        offset=0.25
    for i, time in enumerate(times):
        hour=times_correspondance[time]
        title = f"{var} at {hour}"
        
        # Filter datasets by the specified time and plot
        ds_list_tmp = [ds.where(ds['time_decimal']==time) for ds in ds_list]
        profile_altitude_local_mean(ds_list_tmp,
                                    var,
                                    ax=axes[i],
                                    title=title,
                                    altmin=altmin,
                                    altmax=altmax,
                                    substract_gl=substract_gl,
                                    xmin=xmin, 
                                    xmax=xmax
                                    )
        # Add observations
        obs_ds_list = [obs_dict[time-offset]]
        profile_altitude_obs(obs_ds_list,
                             var,
                             ax=axes[i],
                             altmin=altmin,
                             altmax=altmax,
                             substract_gl=substract_gl,
                             xmin=xmin, 
                             xmax=xmax,
                             altsite=altsite
                             )
        plt.tight_layout()