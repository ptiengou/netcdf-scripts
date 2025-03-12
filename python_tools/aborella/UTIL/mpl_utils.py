def createBasemap(ax, lon_bounds=None, lat_bounds=None, xticks=None, yticks=None):

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.coastlines(zorder=1)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), zorder=2)
    gl.xlocator = mticker.FixedLocator(np.arange(-180., 181., 30.))
    gl.ylocator = mticker.FixedLocator(np.arange(-90., 91., 30.))
    gl.top_labels = False
    gl.right_labels = False

    ax.set_xticks([-180., 0., 180.], crs=ccrs.PlateCarree())
    ax.set_yticks([-60., 0., 90.])

    ax.xaxis.set_major_formatter(LongitudeFormatter(dateline_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    return ax

