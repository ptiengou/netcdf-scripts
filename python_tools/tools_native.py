from tools import *

## MAPPING
#
def plot_ICO_from_netcdf(file, var, timestep=0, vmin=0.0, vmax=None, cmap=wet, 
                         projection='ortho', title=None,
                         lon_min=None, lat_min=None, lon_max=None, lat_max=None,
                         show_grid=True, figsize=(10,6)):
    """
    Plots a map from a NetCDF file using psyplot.

    Parameters:
    - file: str, path to the NetCDF file.
    - var: str, variable name to plot.
    - vmin: float, minimum value for the color scale.
    - vmax: float, maximum value for the color scale.
    - cmap: str, colormap to use.
    - projection: str, map projection to use.
    - title: str, title of the plot.
    - lon_min, lat_min, lon_max, lat_max: float, optional, bounding box for the map.
    - grid: bool, whether to show longitude-latitude grid (default: False).
    """

    if vmin is None or vmax is None:
        bounds = None
    else:
        pas = (vmax - vmin) / 10
        bounds = np.arange(vmin, vmax + pas, pas)

    map_extent = None
    if None not in (lon_min, lat_min, lon_max, lat_max):
        map_extent = [lon_min, lon_max, lat_min, lat_max]
    
    plot=psy.plot.mapplot(
        file,
        # time=timestep,
        name=var,
        datagrid=dict(color='k', linewidth=0.2),
        cbar='r',
        tight=True,
        lsm='50m',
        cmap=cmap,
        extend='both',
        projection=projection,
        title=title if title else var,
        bounds=bounds,
        map_extent=map_extent,
        ygrid=show_grid,
        xgrid=show_grid
    )
    # Resize the current figure after psyplot creates it
    fig = plt.gcf()  # Get the current figure used by psyplot
    fig.set_size_inches(figsize)  # Set new figure size