import xarray as xr
import numpy as np
import warnings

def relabelise(
        xr_in: 'xr.Dataset',
        ) -> tuple['xr.Dataset', dict]:
    """This function relabelise the dimensions of a dataset. The final dimensions
    will be named 'time', 'lev', 'lon', 'lat'.

    Parameters:
    - xr_in: xr.DataArray or xr.Dataset.

    Returns a tuple (xr_out, dim_labels), where xr_out is the relabelised dataset/dataarray,
    and dim_labels is a dictionnary with the old dimension labels.
    """

    if 'time' in xr_in.dims:
        timelabel = 'time'
    elif 'time_counter' in xr_in.dims:
        timelabel = 'time_counter'
        xr_in = xr_in.rename(time_counter='time')
    elif 'valid_time' in xr_in.dims:
        timelabel = 'valid_time'
        xr_in = xr_in.rename(valid_time='time')
    else:
        timelabel = None

    if 'lev' in xr_in.dims:
        levlabel = 'lev'
    elif 'level' in xr_in.dims:
        levlabel = 'level'
        xr_in = xr_in.rename(level='lev')
    elif 'presnivs' in xr_in.dims:
        levlabel = 'presnivs'
        xr_in = xr_in.rename(presnivs='lev')
    elif 'pressure_level' in xr_in.dims:
        levlabel = 'pressure_level'
        xr_in = xr_in.rename(pressure_level='lev')
    else:
        levlabel = None

    if 'lon' in xr_in.dims:
        lonlabel = 'lon'
    elif 'lons' in xr_in.dims:
        lonlabel = 'lons'
        xr_in = xr_in.rename(lons='lon')
    elif 'longitude' in xr_in.dims:
        lonlabel = 'longitude'
        xr_in = xr_in.rename(longitude='lon')
    elif 'longitudes' in xr_in.dims:
        lonlabel = 'longitudes'
        xr_in = xr_in.rename(longitudes='lon')
    else:
        lonlabel = None

    if 'lat' in xr_in.dims:
        latlabel = 'lat'
    elif 'lats' in xr_in.dims:
        latlabel = 'lats'
        xr_in = xr_in.rename(lats='lat')
    elif 'latitude' in xr_in.dims:
        latlabel = 'latitude'
        xr_in = xr_in.rename(latitude='lat')
    elif 'latitudes' in xr_in.dims:
        latlabel = 'latitudes'
        xr_in = xr_in.rename(latitudes='lat')
    else:
        latlabel = None

    dimlabels = dict(time=timelabel, lev=levlabel, lon=lonlabel, lat=latlabel)

    return xr_in, dimlabels


def delabelise(
        xr_in: 'xr.Dataset',
        dim_labels: dict,
        ) -> 'xr.Dataset':
    """This function relabelise the dimensions of a dataset according to the names contained
    in the dictionary dim_labels.

    Parameters:
    - xr_in: xr.DataArray or xr.Dataset. The dataset to relabelise.
    - dim_labels: a dictionary containing the new labels. Typically, as outputed by
                  the function relabelise.

    Returns a dataset xr_out, dim_labels, where xr_out is the relabelised dataset/dataarray.
    """
    
    timelabel = dim_labels['time']
    levlabel = dim_labels['lev']
    lonlabel = dim_labels['lon']
    latlabel = dim_labels['lat']
    if 'time' in xr_in.dims and timelabel != 'time': xr_in = xr_in.rename(time=timelabel)
    if 'lev' in xr_in.dims and levlabel != 'lev': xr_in = xr_in.rename(lev=levlabel)
    if 'lon' in xr_in.dims and lonlabel != 'lon': xr_in = xr_in.rename(lon=lonlabel)
    if 'lat' in xr_in.dims and latlabel != 'lat': xr_in = xr_in.rename(lat=latlabel)

    return xr_in


def reorder_lonlat(
        xr_in: xr.Dataset | xr.DataArray,
        ) -> xr.Dataset | xr.DataArray:

    xr_in, dimlabels = relabelise(xr_in)

    xr_in['lon'] = xr_in.lon % 360. - 360. * ( (xr_in.lon % 360.) // 180. )
    if 'lon' in xr_in.dims: xr_in = xr_in.sortby('lon')
    if 'lat' in xr_in.dims: xr_in = xr_in.sortby('lat')

    xr_in = delabelise(xr_in, dimlabels)

    return xr_in


def lazy_selection(
        xr_in: 'xr.Dataset',
        dt_bounds: tuple[np.datetime64, np.datetime64] = None,
        lev_bounds: tuple[float, float] = None,
        lon_bounds_oriented: tuple[float, float] = None,
        lat_bounds: tuple[float, float] = None,
        enlarge: bool = True,
        ok_warn: bool = True,
        ) -> 'xr.Dataset':
    """Lazily reduce a 4D xarray dataset size. This function permits to lazily select met data,
    without having time / memory issues when loading the data.
    For all the 4 bounds, lower and upper bounds are included in the selection.
    If enlarge is set to True, the selection will wrap the bounds, if not, the bounds
    will be left out of the selection (if they do not match a value in the dataset).
    If one bound tuple is set to None, there is no selection on the corresponding variable.

    Parameters:
    - xr_in: xr.DataArray or xr.Dataset.
    - dt_bounds: tuple of np.datetime64 or None. Bounds for the time selection.
    - lev_bounds: tuple of float or None. Bounds for the level selection.
    - lon_bounds_oriented: tuple of float or None. Bounds for the longitude selection. Must
            be ordered. The selection is done using the first bound as the 'left' or 'west'
            bound of the window, and the second bounds as the 'right' or 'east' bound of the
            window.
            E.g., lon_bounds_oriented = (-40, 25) will select data passing by the longitude 0,
            while lon_bounds_oriented = (25, -40) will select data passing by the longitude 180.
            The bounds can be outside of the [-180, 180) range.
    - lat_bounds: tuple of float or None. Bounds for the latitude selection.
            The bounds can be outside of the [-90, 90] range.
    - enlarge: bool. If True, the selection is the most little one to comprise all the bounds.
            If False, the selection is the most big one so that bounds are outside or on the
            edge of the selection. Default to True.
    - ok_warn: bool.

    Returns xr_out, the reduced dataset. Data are not load.
    """
#    TODO a priori encore quelques pb avec enlarge dans les cas ou les bornes
#    sont proches de 0 ou 180 deg, a cause de la periodicite. A recheck en tout cas

    # change the labels of the input dataset to time, lev, lon, lat
    xr_in, dimlabels = relabelise(xr_in)
    
    # select time data
    if not dt_bounds is None and 'time' in xr_in.dims:
        if np.isscalar(dt_bounds):
            # dt_bounds is simply a np.datetime64, we choose the closest
            xr_in = xr_in.sel(time=dt_bounds, method='nearest')

        else:
            time = xr_in.time
            dt_bound_min, dt_bound_max = min(dt_bounds), max(dt_bounds)
            mask_upper = (time >= dt_bound_min)
            mask_lower = (time <= dt_bound_max)

            if enlarge:
                # first, we apply the extension on mask_upper
                # we check that the left bound is not included in the selection
                if np.nanmin(time[mask_upper]) != dt_bound_min and np.min(time) < dt_bound_min:
                    # we select all the times that were excluded
                    # as 'NaT' are not supported by nanargmin/nanargmax, we put the
                    # lowest value instead of a 'NaT'
                    time_excluded = np.where(mask_upper, np.min(time), time)
                    # we find the highest value and demask it
                    mask_upper[np.nanargmax(time_excluded)] = True
                
                # we do the same on mask_lower
                # we check that the right bound is not included in the selection
                if np.nanmax(time[mask_lower]) != dt_bound_max and np.max(time) > dt_bound_max:
                    # we select all the times that were excluded
                    # as 'NaT' are not supported by nanargmin/nanargmax, we put the
                    # highest value instead of a 'NaT'
                    time_excluded = np.where(mask_lower, np.max(time), time)
                    # we find the lowest value and demask it
                    mask_lower[np.nanargmin(time_excluded)] = True

            mask = mask_upper & mask_lower
            xr_in = xr_in.isel(time=mask)
    
    # select lev data
    if not lev_bounds is None and 'lev' in xr_in.dims:
        if np.isscalar(lev_bounds):
            # lev_bounds is simply a scalar, we choose the closest
            xr_in = xr_in.sel(lev=lev_bounds, method='nearest')

        else:
            lev = xr_in.lev
            lev_bound_min, lev_bound_max = min(lev_bounds), max(lev_bounds)
            mask_upper = (lev >= lev_bound_min)
            mask_lower = (lev <= lev_bound_max)

            if enlarge:
                # first, we apply the extension on mask_upper
                # we check that the left bound is not included in the selection
                if np.nanmin(lev[mask_upper]) != lev_bound_min and np.min(lev) < lev_bound_min:
                    # we select all the levs that were excluded
                    lev_excluded = np.where(mask_upper, np.nan, lev)
                    # we find the highest value and demask it
                    mask_upper[np.nanargmax(lev_excluded)] = True
                
                # we do the same on mask_lower
                # we check that the right bound is not included in the selection
                if np.nanmax(lev[mask_lower]) != lev_bound_max and np.max(lev) > lev_bound_max:
                    # we select all the levs that were excluded
                    lev_excluded = np.where(mask_lower, np.nan, lev)
                    # we find the lowest value and demask it
                    mask_lower[np.nanargmin(lev_excluded)] = True

            mask = mask_upper & mask_lower
            xr_in = xr_in.isel(lev=mask)
    
    # select lat data
    if not lat_bounds is None and 'lat' in xr_in.dims:
        if np.isscalar(lat_bounds):
            # lat_bounds is simply a scalar, we choose the closest
            xr_in = xr_in.sel(lat=lat_bounds, method='nearest')

        else:
            lat = xr_in.lat
            lat_bound_min, lat_bound_max = min(lat_bounds), max(lat_bounds)
            mask_upper = (lat >= lat_bound_min)
            mask_lower = (lat <= lat_bound_max)

            if enlarge:
                # first, we apply the extension on mask_upper
                # we check that the left bound is not included in the selection
                if np.nanmin(lat[mask_upper]) != lat_bound_min and np.min(lat) < lat_bound_min:
                    # we select all the lats that were excluded
                    lat_excluded = np.where(mask_upper, np.nan, lat)
                    # we find the highest value and demask it
                    mask_upper[np.nanargmax(lat_excluded)] = True
                
                # we do the same on mask_lower
                # we check that the right bound is not included in the selection
                if np.nanmax(lat[mask_lower]) != lat_bound_max and np.max(lat) > lat_bound_max:
                    # we select all the lats that were excluded
                    lat_excluded = np.where(mask_lower, np.nan, lat)
                    # we find the lowest value and demask it
                    mask_lower[np.nanargmin(lat_excluded)] = True

            mask = mask_upper & mask_lower
            xr_in = xr_in.isel(lat=mask)

    # select lon data
    if not lon_bounds_oriented is None and 'lon' in xr_in.dims:
        # we save the longitudes and check whether the breaking happens at 0deg or 180deg
        lon = xr_in.lon
        breaking_greenwich = np.any(lon >= 180.)

        if np.isscalar(lon_bounds_oriented):
            # lon_bounds_oriented is simply a scalar, we choose the closest
            lon_sel = lon_bounds_oriented

            # the longitude bounds are transformed to be in the same plane as the input dataset
            if lon_sel < 0. and breaking_greenwich:
                lon_sel = lon_sel % 360.
            else:
                # either lon_sel > 180 and the input dataset lives in [-180, 180),
                # or lon_sel is within [0, 180) and no change has to be made
                lon_sel = lon_sel % 360. - 360. * ((lon_sel % 360.) // 180.)
            xr_in = xr_in.sel(lon=lon_sel, method='nearest')

        else:
            # the longitude bounds are transformed to be in the same plane as the input dataset
            lon_left, lon_right = lon_bounds_oriented
            if lon_left < 0. and breaking_greenwich:
                lon_left = lon_left % 360.
            else:
                # either lon_left > 180 and the input dataset lives in [-180, 180),
                # or lon_left is within [0, 180) and no change has to be made
                lon_left = lon_left % 360. - 360. * ((lon_left % 360.) // 180.)
            if lon_right < 0. and breaking_greenwich:
                lon_right = lon_right % 360.
            else:
                # either lon_right > 180 and the input dataset lives in [-180, 180),
                # or lon_right is within [0, 180) and no change has to be made
                lon_right = lon_right % 360. - 360. * ((lon_right % 360.) // 180.)
            # check for the direction of the selection
            if lon_left <= lon_right:
                # the selection is not passing by the breaking longitude
                mask = (lon >= lon_left) & (lon <= lon_right)
                if enlarge:
                    idx_mask = np.nonzero(mask.values)[0]
                    idx_tmp = idx_mask[0]
                    if lon[idx_tmp] != lon_left and lon[idx_tmp] != lon_right:
                        mask[idx_tmp-1] = True
                    idx_tmp = idx_mask[-1]
                    if lon[idx_tmp] != lon_left and lon[idx_tmp] != lon_right:
                        if idx_tmp == lon.size-1: idx_tmp = -1
                        mask[idx_tmp+1] = True
                
                xr_in = xr_in.isel(lon=mask)

            else:
                # the selection is passing by the breaking longitude
                # note: we cannot keep a singular file with lazy selection in this case
                # first, we select the western window
                mask_west = (lon >= lon_left)
                if enlarge:
                    idx_mask = np.nonzero(mask_west.values)[0]
                    idx_tmp = idx_mask[0]
                    if lon[idx_tmp] != lon_left:
                        mask_west[idx_tmp-1] = True

                # then, we select the eastern window
                mask_east = (lon <= lon_right)
                if enlarge:
                    idx_mask = np.nonzero(mask_east.values)[0]
                    idx_tmp = idx_mask[-1]
                    if lon[idx_tmp] != lon_right:
                        mask_east[idx_tmp+1] = True

                xr_in_west = xr_in.isel(lon=mask_west)
                xr_in_east = xr_in.isel(lon=mask_east)

                # in this case, return two datasets
                xr_in_west = delabelise(xr_in_west, dimlabels)
                xr_in_east = delabelise(xr_in_east, dimlabels)

                left = 0 if breaking_greenwich else -180
                right = 360 if breaking_greenwich else 180
                warn_msg = ('The selection you want includes a break in longitudes '
                           f'between {left}° and {right}°. xarray cannot at the time lazily '
                            'select this window. This function will therefore return '
                            'two lazily selected datasets, one for the selection at '
                           f'the east of {right}°, and one at the west of {left}°.\n'
                            'You can either load then recombine the data with the '
                            'xr.concat function. Alternatively, you can use the '
                            'xr_utils.smart_load function, that returns a single '
                            'loaded dataset but with fast loading.\n'
                            'To suppress this warning, pass ok_warn=False as an argument.')
                if ok_warn: warnings.warn(warn_msg)

                return xr_in_east, xr_in_west
    

    # rename the labels of the input dataset with their original name
    xr_in = delabelise(xr_in, dimlabels)

    return xr_in


def smart_load(
        xr_in: 'xr.Dataset',
        dt_bounds: tuple[np.datetime64, np.datetime64] = None,
        lev_bounds: tuple[float, float] = None,
        lon_bounds_oriented: tuple[float, float] = None,
        lat_bounds: tuple[float, float] = None,
        enlarge: bool = True,
        ) -> 'xr.Dataset':
    """Smartly load a 4D xarray dataset size. This function permits to load met data,
    without having time / memory issues when loading the data.
    For all the 4 bounds, lower and upper bounds are included in the selection.
    If enlarge is set to True, the selection will wrap the bounds, if not, the bounds
    will be left out of the selection (if they do not match a value in the dataset).
    If one bound tuple is set to None, there is no selection on the corresponding variable.

    Parameters:
    - xr_in: xr.DataArray or xr.Dataset.
    - dt_bounds: tuple of np.datetime64 or None. Bounds for the time selection.
    - lev_bounds: tuple of float or None. Bounds for the level selection.
    - lon_bounds_oriented: tuple of float or None. Bounds for the longitude selection. Must
            be ordered. The selection is done using the first bound as the 'left' or 'west'
            bound of the window, and the second bounds as the 'right' or 'east' bound of the
            window.
            E.g., lon_bounds_oriented = (-40, 25) will select data passing by the longitude 0,
            while lon_bounds_oriented = (25, -40) will select data passing by the longitude 180.
            The bounds can be outside of the [-180, 180) range.
    - lat_bounds: tuple of float or None. Bounds for the latitude selection.
            The bounds can be outside of the [-90, 90] range.
    - enlarge: bool. If True, the selection is the most little one to comprise all the bounds.
            If False, the selection is the most big one so that bounds are outside or on the
            edge of the selection. Default to True.

    Returns xr_out, the loaded dataset.
    """

    out = lazy_selection(xr_in,
            dt_bounds=dt_bounds,
            lev_bounds=lev_bounds,
            lon_bounds_oriented=lon_bounds_oriented,
            lat_bounds=lat_bounds,
            enlarge=enlarge,
            ok_warn=False,
            )

    if isinstance(out, tuple):
        xr_out_1 = out[0].load()
        xr_out_2 = out[1].load()
        xr_out_1.close()
        xr_out_2.close()

        xr_out_1, dimlabels = relabelise(xr_out_1)
        xr_out_2, dimlabels = relabelise(xr_out_2)

        xr_out = xr.concat((xr_out_1, xr_out_2), dim='lon')

        xr_out = delabelise(xr_out, dimlabels)
    else:
        xr_out = out.load()
        xr_out.close()

    return xr_out
