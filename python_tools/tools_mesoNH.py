from tools import *
from tools_mapping import *
from tools_LIAISE import *
from tools_hf import *


##DATASETS##


## MAPS ##

def nice_map_mesoNH(data_to_plot, vmin=None, vmax=None, cmap='viridis', 
                         add_liaise=False, title=None, label=None):
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
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)

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

    plt.tight_layout()
    plt.show()

def map_mesoNH_timestamp(ds, var, vmin=None, vmax=None, cmap='viridis', 
                         add_liaise=False,
                         record_index=0, timestamp='2021-07-14T01:00:00'):
    # data_to_plot = ds[var].isel(record=record_index).squeeze()
    data_to_plot = ds[var].sel(time=timestamp, method='nearest')
    title=f'{var} on {timestamp}'
    label = f'{var} ({ds[var].attrs.get("units", "")})'
    nice_map_mesoNH(data_to_plot, vmin=vmin, vmax=vmax, cmap=cmap, 
                         add_liaise=add_liaise, title=title, label=label)
    
def map_mesoNH_mean(ds, var, cmap='viridis', vmin=None, vmax=None, title=None, add_liaise=False):
    """
    Maps the mean of a variable from a MesoNH dataset.
    """
    mean_ds = ds[var].mean(dim='time')
    title = title or f'{var} mean'
    label = f'{var} ({ds[var].attrs.get("units", "")})'
    nice_map_mesoNH(mean_ds, cmap=cmap, vmin=vmin, vmax=vmax, title=title, label=label, add_liaise=add_liaise)