from tools import *

times_correspondance={1.5: '01:30', 4.5: '04:30', 7.5: '07:30', 10.5: '10:30', 13.5: '13:30', 16.5: '16:30', 19.5: '19:30', 22.5: '22:30'}

##LIAISE OBS##
def txtRS_to_xarray(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract metadata
    metadata = {}
    for line in lines[:7]:  # Assuming the first 7 lines are metadata
        key, value = line.split('\n')
        metadata[key.strip()] = value.strip()

    # Extract time series data
    data_start_line = 10  # Assuming the data starts from the 10th line
    data_lines = lines[data_start_line:]

    # Extract variable names and units
    variable_names = lines[8].split()
    variable_units = lines[9].split()
    # Fix split unit 'deg C'
    fixed_units = []
    i = 0
    while i < len(variable_units):
        if variable_units[i] == 'deg' and i + 1 < len(variable_units) and variable_units[i + 1] == 'C':
            fixed_units.append('deg C')
            i += 2  # Skip the next part
        else:
            fixed_units.append(variable_units[i])
            i += 1

    # Convert data lines to a DataFrame
    df = pd.read_csv(StringIO(''.join(data_lines)), delim_whitespace=True, names=variable_names)

    # Convert DataFrame to xarray.Dataset
    ds = xr.Dataset.from_dataframe(df.set_index('Time'))

    # Add metadata as attributes
    ds.attrs = metadata

    for var, unit in zip(variable_names[1:], fixed_units[1:]):  # Skip 'Time'
        ds[var].attrs['units'] = unit
    
    return ds

## DIURNAL CYCLES ##
#generic function
def diurnal_cycle_ax(ax, title=None, ylabel=None, xlabel=None):
    ax.grid()
    ax.set_xticks(np.arange(1, 24, 3))
    tick_positions = np.arange(1.5, 24, 3)
    tick_labels = [f"{int(t)}:{int((t % 1) * 60):02d}" for t in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)

#for one var, several datasets, averaged over lon and lat
def diurnal_cycle_ave(ds_list, var, figsize=(6, 6), ds_colors=False, title=None, ylabel=None, xlabel=None):
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
    diurnal_cycle_ax(ax, title=title, ylabel=ylabel, xlabel=xlabel)

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

#generic basic function
def profile_altitudes_ax(ax, x_var, y_coord, nbins=None, plot_label='', title=None, xlabel=None, ylabel=None, xmin=None, xmax=None, ymin=None, ymax=None, linestyle=None, color=None):
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
    ax.plot(x_var, y_coord, label=plot_label, color=color, linestyle=linestyle)
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
def profile_altitude_local_mean(ds_list, var, obs_ds_list=None, figsize=(6,8), ax=None, title=None, altmin=-0, altmax=2000, nbins=None, substract_gl=True, xmin=None, xmax=None):
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
                             color=ds.attrs.get('plot_color', None))
        
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
    
def profile_altitude_multipletimes_mean(ds_list, var, times, altmin=0, altmax=2000, xmin=None, xmax=None, substract_gl=True):
    n_ax = len(times)
    fig, axs = plt.subplots(1, n_ax, figsize=(5.5*n_ax, 6))
    # Flatten axs only if it's an array (i.e., more than one subplot)
    axes = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
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
        plt.tight_layout()

def profile_altitude_obs(ds_list, var, figsize=(6,8), ax=None, title=None, altmin=-0, altmax=2000, substract_gl=True, nbins=None, xmin=None, xmax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for ds in ds_list:
        if substract_gl:
            ds['ground_level'] = ds['altitude'][0]
            altitude = ds['altitude'] - ds['ground_level']
        else:
            altitude = ds['altitude']
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