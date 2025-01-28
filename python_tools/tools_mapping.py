from tools import *

lon_min=-10
lon_max=4
lat_min=35.5
lat_max=44

default_map_figsize = (8.5,4)
### maps ###
# plt.rcParams['hatch.linewidth'] = 6
# plt.bar(0,2,hatch='//' , edgecolor = None)

def nice_map(plotvar, ax, cmap=myvir, vmin=None, vmax=None, poly=None, sig_mask=None, hatch='//', sig_viz=6, clabel=None, cbar_on=True, left_labels=True):
    ax.coastlines()
    # ax.add_feature(rivers)
    ax.add_feature(cfeature.RIVERS)
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, alpha=0.8)
    gl.right_labels = False
    gl.left_labels = left_labels
    gl.top_labels = False
    gl.xlocator = plt.MaxNLocator(8)
    gl.ylocator = plt.MaxNLocator(9)

    if sig_mask is None:
        # Plot the main variable using xarray plot method
        plot_obj = plotvar.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Plot the main variable using pcolormesh
        # mesh = ax.pcolormesh(
        #     plotvar['lon'], plotvar['lat'], plotvar, 
        #     transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
        # )    
        # Add hatching or other visualizations for significant areas if provided
    elif sig_mask is not None:
        if sig_viz == 0:  # Hiding non-sig values
            plotvar = plotvar.where(sig_mask)
            plot_obj = plotvar.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            plot_obj = plotvar.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
            if sig_viz == 1:  # Hatching using pcolormesh
                #probably does not work with xarray plot method
                ax.pcolormesh(
                    plotvar['lon'], plotvar['lat'], sig_mask, 
                    transform=ccrs.PlateCarree(), hatch=hatch,
                    alpha=0, edgecolor='black', shading='auto'
                )
            elif sig_viz == 2:  # Overlay a gray mask on non-significant areas
                ax.pcolormesh(
                    plotvar['lon'], plotvar['lat'], sig_mask, 
                    transform=ccrs.PlateCarree(), alpha=0.1, cmap='gray'
                )
            elif sig_viz == 3:  # Contours around significant regions
                ax.contour(
                    plotvar['lon'], plotvar['lat'], sig_mask, 
                    levels=[0.5], colors='red', linewidths=1, linestyles='dashed', transform=ccrs.PlateCarree()
                )
            elif sig_viz == 4:  # Transparent contourf with hatches
                #probably not working at all
                ax.contourf(
                    plotvar['lon'], plotvar['lat'], sig_mask, 
                    levels=[0, 1], colors='none', hatches=[hatch], alpha=0, transform=ccrs.PlateCarree()
                )
            elif sig_viz == 5:  # Custom scatter for significant points
                significant_lon, significant_lat = np.meshgrid(plotvar['lon'], plotvar['lat'])
                sig_points = sig_mask
                ax.scatter(
                    significant_lon[sig_points], significant_lat[sig_points], 
                    color='grey', marker='*', s=20, transform=ccrs.PlateCarree(), label='Significant'
                )
            elif sig_viz == 6:  # Overlay hatching for significant regions
                masked_data = np.ma.masked_where(sig_mask, plotvar)
                ax.contourf(
                    plotvar['lon'], plotvar['lat'], masked_data, 
                    levels=np.linspace(vmin if vmin else plotvar.min(), 
                                    vmax if vmax else plotvar.max(), 11),
                    colors='none',  # Transparent so only the hatch shows
                    hatches=[hatch] * len(np.unique(masked_data)),  # Add hatching pattern
                    transform=ccrs.PlateCarree()
                )

    if cbar_on:
        cbar = plot_obj.colorbar
        # cbar = plt.colorbar(plot_obj,ax=ax,shrink=0.8)
        # cbar.shrink(0.8) 
        cbar.set_label(clabel)
        ticks = np.linspace(vmin if vmin is not None else plotvar.min().values, 
                            vmax if vmax is not None else plotvar.max().values, 
                            6)
        cbar.set_ticks(ticks)
        cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        cbar.ax.yaxis.get_major_formatter().set_scientific(True)
        cbar.ax.yaxis.get_major_formatter().set_powerlimits((-2, 4))
    else:
        plot_obj.colorbar.remove()

    # Optional: plot polygon
    if poly:
        plot_polygon(poly, ax)

    plt.tight_layout()

def map_plotvar(plotvar, vmin=None, vmax=None, cmap=myvir, figsize=default_map_figsize, title=None, poly=None, hex=False, hex_center=False):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    nice_map(plotvar, ax, cmap, vmin, vmax, poly=poly)
    if hex:
        plot_hexagon(ax, show_center=hex_center)
    plt.title(title)

def map_ave(ds, var, vmin=None, vmax=None, cmap=myvir, multiplier=1, figsize=default_map_figsize, clabel=None, hex=False, hex_center=False, title=None, poly=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    plotvar = ds[var].mean(dim='time') * multiplier
    nice_map(plotvar, ax, cmap=cmap, vmin=vmin, vmax=vmax, poly=poly, clabel=clabel)
    if hex:
        plot_hexagon(ax, show_center=hex_center)
    if title=="off":
        pass
    elif title:
        plt.title(title)
    else:
        # plt.title(var + ' (' + ds[var].attrs['units'] + ')')
        #check if var has attribute long_name 
        if 'long_name' in ds[var].attrs:
            plt.title(ds[var].attrs['long_name'] + ' (' + ds[var].attrs['units'] + ')')
        else:
            plt.title(var + ' (' + ds[var].attrs['units'] + ')')

def map_diff_ave(ds1, ds2, var, vmin=None, vmax=None, cmap=emb, figsize=default_map_figsize, title=None, clabel=None, hex=False, hex_center=False,
                 sig=False, sig_method=0 , pvalue=0.05, hatch='//', sig_viz=6, check_norm=False):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    diff = (ds1[var]-ds2[var]).mean(dim='time')

    if sig:
        sig_mask = compute_sig_mask(ds1, ds2, var, check_norm=check_norm, method=sig_method, pvalue=pvalue)
        nice_map(diff, ax, cmap=cmap, vmin=vmin, vmax=vmax, sig_mask=sig_mask, hatch=hatch, sig_viz=sig_viz, clabel=clabel)

    else:
        print('No significance mask applied')
        nice_map(diff, ax, cmap=cmap, vmin=vmin, vmax=vmax, clabel=clabel)
    if hex:
        plot_hexagon(ax, show_center=hex_center)
    if title=="off":
        pass
    elif title:
        plt.title(title)
    else:
        plt.title(var + ' difference (' + ds1.name + ' - ' + ds2.name + ', ' + ds1[var].attrs['units'] + ')')

def compute_sig_mask(ds1, ds2, var, check_norm, method, pvalue):
    diff=ds1[var]-ds2[var]
    if method == 0:
        print('Significance method 0: two-sample t-test')
        if check_norm:
            check_normality(ds1, var, pvalue)
            check_normality(ds2, var, pvalue)
        # Two-sample t-test
        _, sample_pvalues = stats.ttest_ind(ds1[var], ds2[var], axis=0)

        #alternative : related samples t-test
        # _, pvalue = stats.ttest_rel(ds1[var1], ds2[var2], axis=0)    
    elif method == 0.1:
        #initial version used (independent samples ttest)
        if check_norm:
            check_normality(ds1, var, pvalue)
            check_normality(ds2, var, pvalue)
        sample_pvalues = xr.apply_ufunc(
            lambda x, y: ttest_ind(x, y, axis=0, nan_policy='omit').pvalue, 
            ds1[var], ds2[var],
            input_core_dims=[['time'], ['time']],
            output_core_dims=[[]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )
    elif method == 1:
        print('Significance method 1: single sample t-test')
        if check_norm:
            check_normality(ds1-ds2, var, pvalue)
        # Single sample t-test on diff
        _, sample_pvalues = stats.ttest_1samp(diff, 0, axis=0)
    elif method == 2:
        # Wilcoxon signed-rank test
        print('Significance method 2: Wilcoxon signed-rank test')
        _, sample_pvalues = stats.wilcoxon(ds1[var], ds2[var], axis=0)
    elif method == 3:
        # Mann-Whitney U test
        print('Significance method 3: Mann-Whitney U test')
        _, sample_pvalues = stats.mannwhitneyu(ds1[var], ds2[var], axis=0)
    return sample_pvalues < pvalue

def check_normality(ds, var, pvalue=0.05, method='shapiro'):
    """
    Check the normality of a variable in a dataset using different methods.

    Parameters:
        data (pd.DataFrame or np.ndarray): Input dataset.
        variable (str): Column name for the variable to test (if DataFrame).
                        If np.ndarray, it assumes the array is the variable itself.
        pvalue (float): Significance level for the test (default: 0.05).
        method (str): Method for testing normality. Options:
                      - 'shapiro': Shapiro-Wilk Test
                      - 'kstest': Kolmogorov-Smirnov Test
                      - 'anderson': Anderson-Darling Test
                      Default is 'shapiro'.

    Returns:

    Raises:
        ValueError: If the specified method is not recognized.
    """
    print("Checking if distributions are normal")
    if method == 'shapiro':
        # Shapiro-Wilk Test
        pvalues, percentage = compute_shapiro_pvalues(ds, var, pvalue)
    elif method == 'kstest':
        # Kolmogorov-Smirnov Test
        pvalues, percentage = compute_kstest_pvalues(ds, var, pvalue)
    elif method == 'anderson':
        print("not implemented")
        pass
    else:
        raise ValueError("Unsupported method. Choose from 'shapiro', 'kstest', or 'anderson'.")
    # print(result)
    return pvalues, percentage

def compute_shapiro_pvalues(ds, var, pvalue):
    # Define a helper function for Shapiro-Wilk test
    def shapiro_pvalue(time_series):
        if len(time_series) < 3:  # Shapiro requires at least 3 observations
            print("Warning: Not enough data for Shapiro-Wilk test.")
            return float("nan")
        _, p_value = stats.shapiro(time_series)
        return p_value

    # Apply the function along the 'time' dimension
    p_values = xr.apply_ufunc(
        shapiro_pvalue,
        ds[var],
        input_core_dims=[["time"]],  # Apply along 'time' dimension
        vectorize=True,             # Enable vectorization for efficiency
        dask="parallelized",        # Use Dask if data is chunked
        output_dtypes=[float],      # Output is a float (p-value)
        dask_gufunc_kwargs={"allow_rechunk": True}  # Allow rechunking
    )
    #count number of significant p-values (takes time...)
    non_sig_cellnb = np.sum(p_values>=pvalue).values
    total_cellnb = np.sum(p_values>0.0).values
    percentage_non_sig=100*non_sig_cellnb/(total_cellnb + 1e-16)
    print('Number of non-significant cells for Shapiro (pvalue={}): {} ({:.2f}%)'.format(pvalue, non_sig_cellnb, percentage_non_sig))
    return p_values, percentage_non_sig

def compute_kstest_pvalues(ds, var, pvalue):
    # Define a helper function for Shapiro-Wilk test
    def kstest_pvalue(time_series):
        if len(time_series) < 3:  # Shapiro requires at least 3 observations
            print("Warning: Not enough data for Shapiro-Wilk test.")
            return float("nan")
        _, p_value = stats.kstest(time_series)
        return p_value

    # Apply the function along the 'time' dimension
    p_values = xr.apply_ufunc(
        kstest_pvalue,
        ds[var],
        input_core_dims=[["time"]],  # Apply along 'time' dimension
        vectorize=True,             # Enable vectorization for efficiency
        dask="parallelized",        # Use Dask if data is chunked
        output_dtypes=[float],      # Output is a float (p-value)
        dask_gufunc_kwargs={"allow_rechunk": True}  # Allow rechunking
    )
    #count number of significant p-values
    non_sig_cellnb = np.sum(p_values>=pvalue).values
    total_cellnb = np.sum(p_values>0.0).values
    percentage_non_sig=non_sig_cellnb/(total_cellnb + 1e-10)
    print('Number of non-significant cells for Kolmogorov-Smirnov (pvalue={}): {} ({:.2f}%)'.format(pvalue, non_sig_cellnb, percentage_non_sig))
    return p_values, percentage_non_sig

def map_rel_diff_ave(ds1, ds2, var, vmin=None, vmax=None, cmap=emb, title=None, clabel=None, figsize=default_map_figsize, hex=False, hex_center=False):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    rel_diff = ((ds1[var]-ds2[var]).mean(dim='time') / (ds2[var].mean(dim='time'))) * 100

    nice_map(rel_diff, ax, cmap=cmap, vmin=vmin, vmax=vmax, clabel=clabel)
    if hex:
        plot_hexagon(ax, show_center=hex_center)
    if title=="off":
        pass
    elif title:
        plt.title(title)
    else:
        plt.title(var + ' relative difference (' + ds1.name + ' - ' + ds2.name + ' ; %)')

def map_two_ds(ds1, ds2, var, vmin=None, vmax=None, cmap=reds, figsize=(15,6), hex=False, hex_center=False):
    fig, axs = plt.subplots(1, 2, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(var + ' ({})'.format(ds1[var].units))
    plotvar_1 = ds1[var].mean(dim='time')
    plotvar_2 = ds2[var].mean(dim='time')
    nice_map(plotvar_1, axs[0], cmap, vmin, vmax)
    axs[0].set_title(ds1.name)
    if hex:
        plot_hexagon(axs[0], show_center=hex_center)
    nice_map(plotvar_2, axs[1], cmap, vmin, vmax)
    axs[1].set_title(ds2.name)
    if hex:
        plot_hexagon(axs[1], show_center=hex_center)

def map_seasons(plotvar, vmin=None, vmax=None, cmap=myvir, figsize=(12,7), hex=False, hex_center=False, title=None):
    fig, axs = plt.subplots(2, 2, figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(title)
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    for i, season in enumerate(seasons):
        plotvar_season = plotvar.where(plotvar['time.season']==season, drop=True)
        nice_map(plotvar_season.mean(dim='time'), axs.flatten()[i], cmap, vmin, vmax)
        axs.flatten()[i].set_title(season)
        if hex:
            plot_hexagon(axs.flatten()[i], show_center=hex_center)

## quiver plots for transport ##
def map_wind(ds, extra_var='wind speed', height='10m', vmin=None, vmax=None, figsize=default_map_figsize, cmap=reds, dist=6, scale=100):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    windvar_u=ds['u'+height].mean(dim='time')
    windvar_v=ds['v'+height].mean(dim='time')
    #show extra_var in background color
    if extra_var=='wind speed':
        plotvar = (windvar_u**2 + windvar_v**2 ) ** (1/2)
    else:
        plotvar = ds[extra_var].mean(dim='time')
    nice_map(plotvar, ax, cmap=cmap, vmin=vmin, vmax=vmax)

    #plot wind vectors
    windx = windvar_u[::dist,::dist]
    windy = windvar_v[::dist,::dist]
    longi=ds['lon'][::dist]
    lati=ds['lat'][::dist]
    quiver = ax.quiver(longi, lati, windx, windy, transform=ccrs.PlateCarree(), scale=scale)

    quiverkey_scale = scale/10
    plt.quiverkey(quiver, X=0.93, Y=0.08, U=quiverkey_scale, label='{} m/s'.format(quiverkey_scale), labelpos='S')

    if (height == '10m'):
        plt.title('10m wind (m/s) and {}'.format(extra_var))
    else :
        plt.title('{} hPa wind (m/s) and {}'.format(height, extra_var))

def map_wind_diff(ds1, ds2, height='10m', figsize=default_map_figsize, vmin=None, vmax=None, cmap=emb, dist=6, scale=100, hex=False, hex_center=False):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    windvar_u1=ds1['u'+height].mean(dim='time')
    windvar_v1=ds1['v'+height].mean(dim='time')
    windvar_u2=ds2['u'+height].mean(dim='time')
    windvar_v2=ds2['v'+height].mean(dim='time')
    #show wind speed in background color
    wind_speed1 = (windvar_u1**2 + windvar_v1**2 ) ** (1/2)
    wind_speed2 = (windvar_u2**2 + windvar_v2**2 ) ** (1/2)
    wind_speed_diff = wind_speed1 - wind_speed2
    nice_map(wind_speed_diff, ax, cmap=cmap, vmin=vmin, vmax=vmax)
    if hex:
        plot_hexagon(ax, show_center=hex_center)
    #plot wind vectors
    windx = (windvar_u1-windvar_u2)[::dist,::dist]
    windy = (windvar_v1-windvar_v2)[::dist,::dist]
    longi=ds1['lon'][::dist]
    lati=ds1['lat'][::dist]
    quiver = ax.quiver(longi, lati, windx, windy, transform=ccrs.PlateCarree(), scale=scale)

    quiverkey_scale = scale/10
    plt.quiverkey(quiver, X=0.93, Y=0.08, U=quiverkey_scale, label='{} m/s'.format(quiverkey_scale), labelpos='S')

    if (height == '10m'):
        plt.title('10m wind speed (m/s) and direction change ({} - {})'.format(ds1.name, ds2.name))
    else :
        plt.title('{} hPa wind speed (m/s) and direction change ({} - {})'.format(height, ds1.name, ds2.name))

def map_moisture_transport(ds, extra_var='norm', figsize=default_map_figsize, cmap=reds, vmin=None, vmax=None, dist=6, scale=100, poly=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    windvar_u=ds['uq'].mean(dim='time')
    windvar_v=ds['vq'].mean(dim='time')
    #show extra_var in background color
    if extra_var=='norm':
        plotvar = (windvar_u**2 + windvar_v**2 ) ** (1/2)
    else:
        plotvar = ds[extra_var].mean(dim='time')
    nice_map(plotvar, ax, cmap=cmap, vmin=vmin, vmax=vmax, poly=poly)

    #plot wind vectors
    windx = windvar_u[::dist,::dist]
    windy = windvar_v[::dist,::dist]
    longi=ds['lon'][::dist]
    lati=ds['lat'][::dist]
    quiver = ax.quiver(longi, lati, windx, windy, transform=ccrs.PlateCarree(), scale=scale)

    quiverkey_scale = scale/10
    plt.quiverkey(quiver, X=0.95, Y=0.05, U=quiverkey_scale, label='{} kg/m/s'.format(quiverkey_scale), labelpos='S')

    plt.title('Moisture transport ({})'.format(extra_var))

#hexagons
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

# def plot_hexagon(ax, center_lon=-3.7, center_lat=40.43, sim_radius_km=825, nbp=40, forced=5, nudged=8, show_center=False):
def plot_hexagon(ax, center_lon=-3.7, center_lat=40.43, sim_radius_km=1250, nbp=40, forced=5, nudged=8, show_center=False):
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

#native ICO grid

def plot_ICO_grid(ax, lon, lat, color='black', linewidth=0.5):
    return()
