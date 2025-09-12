from tools import *

##DATASETS##

def sel_closest(ds, lon, lat, r=None):
    # from A. Borella
    dist = haversine(ds.lat, ds.lon, lat, lon)
    if r is None:
        # sel the closest
        dict_closest = dist.argmin(...)
    else:
        # sel all that are closer than r (in km)
        dict_closest = dict(cell=(dist <= r * 1e3))

    ds = ds.isel(**dict_closest)

    return ds

def haversine(lat1, lon1, lat2, lon2):
    # from A. Borella
    '''Computes the Haversine distance between two points in m'''
    dlon_rad = np.deg2rad(lon2 - lon1)
    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    arc = np.sin((lat2_rad - lat1_rad) / 2.)**2. + np.cos(lat1_rad) \
            * np.cos(lat2_rad) * np.sin(dlon_rad / 2.)**2.
    c = 2. * np.arcsin(np.sqrt(arc))
    R_E = 6372800.0 #see http://rosettacode.org/wiki/Haversine_formula
    return R_E * c

## MAPPING
#
def plot_ICO_from_netcdf(file, var, timestep=0, vmin=0.0, vmax=None, cmap=wet, 
                         projection='ortho', title=None, clabel=None,
                         lon_min=None, lat_min=None, lon_max=None, lat_max=None,
                         show_grid=True, figsize=(8,4)):
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
        datagrid=dict(color='k', linewidth=0.2, alpha=1.),
        cbar='r',
        tight=True,
        lsm='50m',
        cmap=cmap,
        # extend='both',
        projection=projection,
        # title=title if title is not None else var,
        bounds=bounds,
        map_extent=map_extent,
        # ygrid=show_grid,
        # xgrid=show_grid,
        ygrid=False,
        xgrid=False,
        clabel=clabel,

        # decode_times=False
    )

    # Resize the current figure after psyplot creates it
    fig = plt.gcf()  # Get the current figure used by psyplot
    fig.set_size_inches(figsize)  # Set new figure size