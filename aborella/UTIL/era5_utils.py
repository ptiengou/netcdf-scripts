import os, sys
import hashlib
import xarray as xr
import pandas as pd
import numpy as np

from pycontrails import MetDataset
from pycontrails.datalib.ecmwf import ERA5

sys.path.append('/home/aborella/UTIL')
import xr_utils

# max age of a contrail - for cocip
max_age = np.timedelta64(19, 'h') + np.timedelta64(45, 'm')


def get_paths(dt_hour, models=None, realisation=None):

    models = 'perf_cocip_accf' if models is None else models
    realisation = 'HRES' if realisation is None else realisation

    timestamp = pd.Timestamp(dt_hour)
    yr = timestamp.year
    mth = str(timestamp.month).zfill(2)

    path_t = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{yr}/ta.{yr}{mth}.ap1e5.GLOBAL_025.nc'
    path_q = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{yr}/q.{yr}{mth}.ap1e5.GLOBAL_025.nc'
    path_z = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{yr}/geopt.{yr}{mth}.ap1e5.GLOBAL_025.nc'
    path_r = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{yr}/r.{yr}{mth}.ap1e5.GLOBAL_025.nc'
    path_u = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{yr}/u.{yr}{mth}.ap1e5.GLOBAL_025.nc'
    path_v = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{yr}/v.{yr}{mth}.ap1e5.GLOBAL_025.nc'
    #path_cc = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{yr}/cc.{yr}{mth}.ap1e5.GLOBAL_025.nc'

#    path_pv = f'/projsu/cmip-work/oboucher/ERA5/pv.{yr}.GLOBAL.nc'  
#    path_ciwc = f'/projsu/cmip-work/oboucher/ERA5/ciwc.{yr}.GLOBAL.nc'
#    path_w = f'/projsu/cmip-work/oboucher/ERA5/w.{yr}.GLOBAL.nc'
#
#    path_tsr = f'/projsu/cmip-work/oboucher/ERA5/tsr.{yr}.GLOBAL.nc'
#    path_ttr = f'/projsu/cmip-work/oboucher/ERA5/ttr.{yr}.GLOBAL.nc'
#    path_ssrd = f'/projsu/cmip-work/oboucher/ERA5/ssrd.{yr}.GLOBAL.nc'

    path_pv = f'/scratchx/aborella/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{yr}/pv.{yr}{mth}.ap1e5.GLOBAL_025.UTLS.nc'    
    path_ciwc = f'/scratchx/aborella/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{yr}/ciwc.{yr}{mth}.ap1e5.GLOBAL_025.UTLS.nc'
    path_w = f'/scratchx/aborella/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/{yr}/w.{yr}{mth}.ap1e5.GLOBAL_025.UTLS.nc'

    path_tsr = f'/scratchx/aborella/ERA5/NETCDF/GLOBAL_025/hourly/AN_SL/{yr}/tsr.{yr}{mth}.ap1e5.GLOBAL_025.nc'
    path_ttr = f'/scratchx/aborella/ERA5/NETCDF/GLOBAL_025/hourly/AN_SL/{yr}/ttr.{yr}{mth}.ap1e5.GLOBAL_025.nc'
    path_ssrd = f'/scratchx/aborella/ERA5/NETCDF/GLOBAL_025/hourly/AN_SL/{yr}/ssrd.{yr}{mth}.ap1e5.GLOBAL_025.nc'


    variables_met = ()
    variables_rad = ()
    paths_met = ()
    paths_rad = ()

    if 'perf' in models:
        variables_met += ('air_temperature', 'eastward_wind', 'northward_wind',)
        paths_met += (path_t, path_u, path_v,)
    if 'cocip' in models:
        variables_met += ('air_temperature', 'specific_humidity', 'eastward_wind',
                'northward_wind', 'lagrangian_tendency_of_air_pressure',
                'specific_cloud_ice_water_content', 'geopotential',)
        paths_met += (path_t, path_q, path_u, path_v, path_w, path_ciwc, path_z,)
        variables_rad += ('top_net_solar_radiation', 'top_net_thermal_radiation',)
        paths_rad += (path_tsr, path_ttr,)
    if 'accf' in models:
        variables_met += ('air_temperature', 'specific_humidity', 'potential_vorticity',
                'geopotential', 'relative_humidity', 'northward_wind', 'eastward_wind',)
        paths_met += (path_t, path_q, path_pv, path_z, path_r, path_v, path_u,)
        variables_rad += ('surface_solar_downward_radiation', 'top_net_thermal_radiation',)
        paths_rad += (path_ssrd, path_ttr,)

    variables_met = tuple(set(variables_met))
    variables_rad = tuple(set(variables_rad))
    paths_met = tuple(set(paths_met))
    paths_rad = tuple(set(paths_rad))


    paths = [paths_met, paths_rad]
    variables = [variables_met, variables_rad]

    return paths, variables


def get_reanalysis(dt_bounds, lev_bounds, lon_bounds_oriented, lat_bounds, models):

    # fetch needed meteorological data and load MetDatasets from it
    paths, variables = get_paths(dt_bounds[0], models=models)

    paths_met, paths_rad = paths
    variables_met, variables_rad = variables


    met_dataset = None
    rad_dataset = None

    for path in paths_met:
#        util.info(f'Fetch {path}')
        da_tmp = xr.open_dataset(path)

        da_tmp = xr_utils.smart_load(da_tmp,
                dt_bounds=dt_bounds,
                lev_bounds=lev_bounds,
                lon_bounds_oriented=lon_bounds_oriented,
                lat_bounds=lat_bounds,
                )


        da_tmp['longitude'] = da_tmp.longitude % 360. \
                            - 360. * ( ( da_tmp.longitude % 360. ) // 180.)
        da_tmp = da_tmp.sortby('longitude').sortby('latitude')


        if met_dataset is None:
            met_dataset = da_tmp
        else:
            met_dataset = xr.merge([met_dataset, da_tmp])

    if 'air_temperature' in variables_met: met_dataset = met_dataset.rename(ta='t')
    if 'geopotential' in variables_met: met_dataset = met_dataset.rename(geopt='z')


    for path in paths_rad:
#        util.info(f'Fetch {path}')
        da_tmp = xr.open_dataset(path)

        # we add one hour because of the shift radiation time of 30 minutes backward
        dt_bounds_rad = (dt_bounds[0],
                         dt_bounds[1] + np.timedelta64(1, 'h'))
        da_tmp = xr_utils.smart_load(da_tmp,
                dt_bounds=dt_bounds_rad,
                lon_bounds_oriented=lon_bounds_oriented,
                lat_bounds=lat_bounds,
                )


        da_tmp['longitude'] = da_tmp.longitude % 360. \
                            - 360. * ( ( da_tmp.longitude % 360. ) // 180.)
        da_tmp = da_tmp.sortby('longitude').sortby('latitude')


        if rad_dataset is None:
            rad_dataset = da_tmp
        else:
            rad_dataset = xr.merge([rad_dataset, da_tmp])


#    util.newline()

    return met_dataset, rad_dataset


def create_datasets(dt_bounds, met_ds, rad_ds, models):

    paths, variables = get_paths(dt_bounds[0], models=models)


    pressure_levels = met_ds['level'].values
    variables_met, variables_rad = variables

    era5pl = ERA5(
            time=dt_bounds,
            variables=variables_met,
            pressure_levels=pressure_levels,
            product_type='reanalysis',
            )
    # we add one hour because of the shift radiation time of 30 minutes backward
    dt_bounds_rad = (dt_bounds[0],
                     dt_bounds[1] + np.timedelta64(1, 'h'))
    era5sl = ERA5(
            time=dt_bounds_rad,
            variables=variables_rad,
            product_type='reanalysis',
            )

    met = era5pl.open_metdataset(dataset=met_ds)

    if len(variables_rad) == 0:
        rad = None
    else:
        rad = era5sl.open_metdataset(dataset=rad_ds)

    return met, rad
