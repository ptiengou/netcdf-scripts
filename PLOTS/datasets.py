import sys, os
import glob
import xarray as xr
import numpy as np
import scipy.spatial as sspatial
import matplotlib.pyplot as plt
import pandas as pd
from pyhdf.SD import SD, SDC
import psyplot.project as psy

sys.path.append('/home/aborella/UTIL')
import thermodynamic as thermo
import xr_utils

import calipso_mod
import modis_mod
import iasi_mod
import radflux_mod

alt_min, alt_max = 0., 12000.
alt_bounds = (alt_min, alt_max)
lev_min, lev_max = 100., 999999.
lev_bounds = (lev_min, lev_max)

ERA5_levels = np.array([   1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,  125,  150,  175,  200,  225,  250,  300,  350,  400,  450,  500,  550,  600,  650,  700,  750,  775,  800,  825,  850,  875,  900,  925,  950,  975, 1000])
ERA5_levels_bounds = np.concatenate(([0], (ERA5_levels[1:] + ERA5_levels[:-1]) / 2., [999999]))


def sel_closest(ds, lon, lat, r=None):

    # NB. this method is much slower than calculating the haversine distance
    #tree = sspatial.KDTree(list(zip(ds.lon, ds.lat)))
    #_, icell = tree.query((lon, lat))

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
    '''Computes the Haversine distance between two points in m'''
    dlon_rad = np.deg2rad(lon2 - lon1)
    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    arc = np.sin((lat2_rad - lat1_rad) / 2.)**2. + np.cos(lat1_rad) \
            * np.cos(lat2_rad) * np.sin(dlon_rad / 2.)**2.
    c = 2. * np.arcsin(np.sqrt(arc))
    R_E = 6372800.0 #see http://rosettacode.org/wiki/Haversine_formula
    return R_E * c


def get_lmdz_filename(simu, cosp=False):
    cosp_str = 'COSP' if cosp else ''
#    print('WARNING, CTRL IS REPLACED WITH CTRL7.0.1')
    if simu == 'CTRL':
        filename = f'/scratchu/aborella/LMDZ/TGCC/PARIS-CTRL/PARIS-CTRL_20221221_20221228_IN_histins{cosp_str}.nc'
#        filename = f'/scratchu/aborella/LMDZ/TGCC/PARIS-CTRL7.0.1/PARIS-CTRL7.0.1_20221221_20221228_IN_histins{cosp_str}.nc'
    elif simu == 'OVLP':
        filename = f'/scratchu/aborella/LMDZ/TGCC/PARIS-OVLP/ICOLMDZ-PARIS-OVLP_20221221_20221228_IN_histins{cosp_str}.nc'
    else:
        filename = f'/scratchu/aborella/LMDZ/TGCC/PARIS-ISSR-{simu}/PARIS-ISSR-{simu}_20221221_20221228_IN_histins{cosp_str}.nc'

    return filename


def get_era5_filename(varname):
    filename = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/2022/{varname}.202212.ap1e5.GLOBAL_025.nc'
    if not os.path.exists(filename):
        filename = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_SF/2022/{varname}.202212.as1e5.GLOBAL_025.nc'
        if not os.path.exists(filename):
            filename = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/FC_SF/2022/12/{varname}.20221226.1800.fs1e5.GLOBAL_025.nc'
            if not os.path.exists(filename):
                raise ValueError

    return filename


def get_era5_average(varname, **bounds):

    if varname is None: return None

    filename = get_era5_filename(varname)
    ds = xr.open_dataset(filename)
    ds, _ = xr_utils.relabelise(ds)

    ds = ds[varname]

    ds = xr_utils.smart_load(ds, dt_bounds=bounds.get('time'),
            lat_bounds=bounds.get('lat'), lon_bounds_oriented=bounds.get('lon'))
    ds = xr_utils.smart_load(ds, lev_bounds=bounds.get('lev'), enlarge=False)
    ds = xr_utils.reorder_lonlat(ds)

    # circle definition
#    center = (2., 48.)
#    radius = 300. # km

#    dist = haversine(ds.lat, ds.lon, center[1], center[0])
#    mask = dist <= radius * 1e3
    # area definition (bis)
    mask = (ds.lon <= 10.) & (ds.lat <= 50.5) & (ds.lat >= 44.) \
         & (ds.lat <= (50.5 - 44) / (-5 + 10) * (ds.lon + 10) + 44)

    ds = ds.where(mask, drop=True)
    mask = mask.where(mask, False, drop=True)

    weights = np.cos(np.deg2rad(ds.lat))
    weights = weights.expand_dims(lon=ds.lon)
    weights = weights.where(mask, 0.)

    ds = ds.weighted(weights)
    ds = ds.mean(('lon', 'lat'))

    return ds


def get_ml_era5_average(**bounds):

    ds_qt = xr.open_dataset('/scratchu/aborella/OBS/ERA5/q_t_ERA5_ML_case_study.nc')
    ds_cc = xr.open_dataset('/scratchu/aborella/OBS/ERA5/cc_ERA5_ML_case_study.nc')
    ds_ciwc = xr.open_dataset('/scratchu/aborella/OBS/ERA5/ciwc_ERA5_ML_case_study.nc')
    ds_sp = xr.open_dataset('/scratchu/aborella/OBS/ERA5/sp_ERA5_ML_case_study.nc')
    ml_to_pl = pd.read_csv('/scratchu/aborella/OBS/ERA5/ML_to_PL.txt', sep=r'\s+',
                           names=['model_level', 'a', 'b'])
    a = xr.DataArray(ml_to_pl['a'], dims=('model_level',), coords={'model_level': ml_to_pl['model_level']})
    b = xr.DataArray(ml_to_pl['b'], dims=('model_level',), coords={'model_level': ml_to_pl['model_level']})

    ds_qt, _ = xr_utils.relabelise(ds_qt)
    ds_qt = xr_utils.smart_load(ds_qt, dt_bounds=bounds.get('time'),
               lat_bounds=bounds.get('lat'), lon_bounds_oriented=bounds.get('lon'))
    ds_qt = xr_utils.reorder_lonlat(ds_qt)

    ds_cc, _ = xr_utils.relabelise(ds_cc)
    ds_cc = xr_utils.smart_load(ds_cc, dt_bounds=bounds.get('time'),
               lat_bounds=bounds.get('lat'), lon_bounds_oriented=bounds.get('lon'))
    ds_cc = xr_utils.reorder_lonlat(ds_cc)

    ds_ciwc, _ = xr_utils.relabelise(ds_ciwc)
    ds_ciwc = xr_utils.smart_load(ds_ciwc, dt_bounds=bounds.get('time'),
               lat_bounds=bounds.get('lat'), lon_bounds_oriented=bounds.get('lon'))
    ds_ciwc = xr_utils.reorder_lonlat(ds_ciwc)

    ds_sp, _ = xr_utils.relabelise(ds_sp)
    ds_sp = xr_utils.smart_load(ds_sp, dt_bounds=bounds.get('time'),
               lat_bounds=bounds.get('lat'), lon_bounds_oriented=bounds.get('lon'))
    ds_sp = xr_utils.reorder_lonlat(ds_sp)

    ds_sp = ds_sp.isel(model_level=0).reset_coords(drop=True)
    pres = a + b * np.exp(ds_sp.lnsp)

    da_rhi = thermo.speHumToRH(ds_qt.q, ds_qt.t, pres, phase='ice')
    da_rhi.name = 'rhi'
    pres = pres / 1e2

    ds = xr.merge((ds_cc, ds_ciwc, da_rhi))

    ds = ds.assign_coords(lev=pres.mean(('lon', 'lat', 'time')))
    ds = ds.swap_dims(model_level='lev')
    ds = xr_utils.smart_load(ds, lev_bounds=bounds.get('lev'), enlarge=False)

    # circle definition
#    center = (2., 48.)
#    radius = 300. # km

#    dist = haversine(ds.lat, ds.lon, center[1], center[0])
#    mask = dist <= radius * 1e3
    # area definition (bis)
    mask = (ds.lon <= 10.) & (ds.lat <= 50.5) & (ds.lat >= 44.) \
         & (ds.lat <= (50.5 - 44) / (-5 + 10) * (ds.lon + 10) + 44)

    ds['qiceincld'] = ds.ciwc / ds.cc
    ds['qiceincld'] = ds.qiceincld.where(ds.cc >= 1e-2)

    ds = ds.where(mask, drop=True)
    mask = mask.where(mask, False, drop=True)

    weights = np.cos(np.deg2rad(ds.lat))
    weights = weights.expand_dims(lon=ds.lon)
    weights = weights.where(mask, 0.)

    ds = ds.isel(lev=0)
    ds = ds.weighted(weights)
    ds = ds.mean(('lon', 'lat'))

    return ds


def get_lmdz_average(vartype, varnames, simu, **bounds):

    cosp = 'modis' in vartype or 'calipso' in vartype or 'isccp' in vartype or 'cloudsat' in vartype
    filename = get_lmdz_filename(simu, cosp=cosp)
    ds = xr.open_dataset(filename)
    ds, _ = xr_utils.relabelise(ds)
    if 'lev' in ds.dims: ds['lev'] = ds.lev / 1e2

    if vartype == 'hcc-modis':
        ds = ds['clhmodis']
#        ds = ds[['pctmodis', 'cltmodis']]
    elif vartype == 'hcc-calipso':
        ds = ds['clhcalipso']
    elif vartype == 'hcc-isccp':
        ds = ds[['ctpisccp', 'tclisccp']]
    elif vartype == 'hcod-modis':
        ds = ds['tautmodis']
    elif vartype == 'hcod-isccp':
        ds = ds['tauisccp']
    elif vartype == 'iwp-modis':
        ds = ds['iwpmodis']
    elif vartype == 'lwp-modis':
        ds = ds['lwpmodis']
    else:
        if varnames is None: return None
        delta_pres = np.abs(ds.presinter.diff('presinter')).rename(presinter='lev')
        delta_pres['lev'] = ds.lev
        delta_pres.name = delta_pres
        ds = ds[varnames]
        if 'lev' in ds.dims: ds = ds.assign_coords(delta_pres=delta_pres)

    if vartype == 'tendice':
        lev_bounds = (bounds['lev'][0], bounds['lev'][1] - 25.)
        pr_lsc_i = xr_utils.smart_load(ds.pr_lsc_i, dt_bounds=bounds.get('time'),
                lev_bounds=lev_bounds, lat_bounds=bounds.get('lat'),
                lon_bounds_oriented=bounds.get('lon'))

    ds = xr_utils.smart_load(ds, dt_bounds=bounds.get('time'),
            lat_bounds=bounds.get('lat'), lon_bounds_oriented=bounds.get('lon'))
    ds = xr_utils.smart_load(ds, lev_bounds=bounds.get('lev'), enlarge=False)
    ds = xr_utils.reorder_lonlat(ds)

    # circle definition
#    center = (2., 48.)
#    radius = 300. # km

#    dist = haversine(ds.lat, ds.lon, center[1], center[0])
#    mask = dist <= radius * 1e3
    # area definition (bis)
    mask = (ds.lon <= 10.) & (ds.lat <= 50.5) & (ds.lat >= 44.) \
         & (ds.lat <= (50.5 - 44) / (-5 + 10) * (ds.lon + 10) + 44)
#    mask = (ds.lon <= 3.2) & (ds.lon >= 1.2) & (ds.lat <= 49.7) & (ds.lat >= 47.7)
#    mask = (ds.lon <= 2.3026) & (ds.lon >= 2.3025) & (ds.lat <= 47.9017) & (ds.lat >= 47.9016)

    ds = ds.isel(cell=mask)

    if vartype == 'tendice': pr_lsc_i = pr_lsc_i.isel(cell=mask)

    filename = '/scratchu/aborella/LMDZ/TGCC/PARIS-ISSR-REF/PARIS-ISSR-REF_20221221_20221228_1D_histday.nc'
    ds_1D = xr.open_dataset(filename)
    weights = ds_1D.aire
    weights = weights.isel(cell=mask)

    if vartype == 'hcc-modis':
        ds = ds / 100.
#        ds = ds.cltmodis.where(ds.pctmodis < 44000.) / 100.
    elif vartype == 'hcc-isccp':
        ds = ds.tclisccp.where(ds.ctpisccp < 44000.) / 100.
    elif vartype == 'hcod-modis':
        ds = np.log10(ds)
    elif vartype == 'hcod-isccp':
        ds = np.log10(ds)
    elif vartype == 'frac':
        ds['subfra'] = 1. - ds.cfseri - ds.issrfra
    elif vartype == 'rhum':
        eps = 1e-10
        subfra = 1. - ds.cfseri - ds.issrfra
        subfra = subfra.where(subfra >= eps)
        clrfra = 1. - ds.cfseri
        clrfra = clrfra.where(clrfra >= eps)
        issrfra = ds.issrfra
        issrfra = issrfra.where(issrfra >= eps)
        cldfra = ds.cfseri
        cldfra = cldfra.where(cldfra >= eps)
        qsub = ds.ovap * (1. - ds.rvcseri) - ds.qissr
        qsub = qsub.where(qsub >= eps)
        qclr = ds.ovap * (1. - ds.rvcseri)
        qclr = qclr.where(qclr >= eps)
        qissr = ds.qissr
        qissr = qissr.where(qissr >= eps)
        qcld = ds.qcld
        qcld = qcld.where(qcld >= eps)
        qvc = ds.rvcseri * ds.ovap
        qvc = qvc.where(qvc >= eps)
        ds['rhi_cld']    = ds.rhi / ds.ovap * qcld  / cldfra
        ds['rhi_cldvap'] = ds.rhi / ds.ovap * qvc   / cldfra
        ds['rhi_issr']   = ds.rhi / ds.ovap * qissr / issrfra
        ds['rhi_clr']    = ds.rhi / ds.ovap * qclr  / clrfra
        ds['rhi_sub']    = ds.rhi / ds.ovap * qsub  / subfra
    elif vartype == 'spehum':
        ds['qsub'] = ds.ovap * (1. - ds.rvcseri) - ds.qissr
        ds['qclr'] = ds.ovap * (1. - ds.rvcseri)
        ds['qissr'] = ds.qissr
        ds['qcld'] = ds.qcld
        ds['qvc'] = ds.rvcseri * ds.ovap
        ds['qice'] = ds.ocond - ds.oliq
        ds['qvap'] = ds.ovap
    elif vartype == 'tendice':
        delta_pres = - pr_lsc_i.lev.diff('lev', label='lower') * 1e2
        delta_flux = pr_lsc_i.diff('lev', label='lower')
        timestep = 900.
        dqised = delta_flux * 9.8065 * timestep / delta_pres * 1e-3
        dqised = dqised.sel(lev=ds.lev)
        ds['dqised'] = dqised - ds.dqssub
    if 'rhi_clr_CTRL' in vartype:
        ds['rhi_clr'] = (ds.rhi - 100. * ds.rnebls) / (1. - ds.rnebls)
    if 'rhi_clr' in vartype:
        ds['rhi_clr'] = ds.rhi * (1. - ds.rvcseri) / (1. - ds.cfseri)
        ds['rhi_clr'] = ds.rhi_clr.where(ds.cfseri < 1.)
    if 'rhi_issr' in vartype:
        ds['rhi_issr'] = ds.rhi / ds.ovap * ds.qissr / ds.issrfra
        ds['rhi_issr'] = ds.rhi_issr.where(ds.issrfra >= 1e-2)
    if 'oice_CTRL' in vartype:
        ds['oice'] = ds.ocond - ds.oliq
        ds['qiceincld'] = ds.oice / ds.rnebls
        ds['qiceincld'] = ds.qiceincld.where(ds.rnebls >= 1e-2)
    if 'oice' in vartype:
        ds['oice'] = ds.ocond - ds.oliq
        ds['qiceincld'] = ds.oice / ds.cfseri
        ds['qiceincld'] = ds.qiceincld.where(ds.cfseri >= 1e-2)

#   ds = ds.quantile(0.05, dim=('cell', 'lev'), skipna=True)

    if len(bounds.get('lev')) == 2 and 'lev' in ds.dims:
        ds = ds.weighted(weights * ds.delta_pres)
        ds = ds.mean(dim=('cell', 'lev'), skipna=True)
    else:
        ds = ds.weighted(weights)
        ds = ds.mean('cell')

    return ds


def get_MSG_average(varname, **bounds):

    if varname is None: return None

    da_merge = pre_mask = mask = None
    step = np.timedelta64(15, 'm')
    for time in np.arange(bounds['time'][0], bounds['time'][1] + step / 2, step):

        # NO DATA FOR THESE POINTS
        if time == np.datetime64('2022-12-27T06:30') or time == np.datetime64('2022-12-27T07'):
            da = xr.DataArray([np.nan], dims=('time',), coords=dict(time=[time]))
            da_merge = xr.concat((da_merge, da), dim='time')
            continue

        filename = '/scratchu/aborella/OBS/MSG/202212/MSG4-SEVI-MSGOCAE-0101-0101-202212'
        filenames = glob.glob(filename + pd.to_datetime(time).strftime('%d%H%M') + '*.nc')
        print(filenames)
        assert len(filenames) == 1
        filename = filenames[0]

        ds = xr.open_dataset(filename)
        ds = ds.rename(longitude='lon', latitude='lat')

        ds = ds[['Upper_Layer_Cloud_Top_Pressure', 'Upper_Layer_Cloud_Optical_Depth', 'Lower_Layer_Cloud_Optical_Depth']]

        if pre_mask is None:
            pre_mask = (ds.lon >= bounds['lon'][0]) & (ds.lon <= bounds['lon'][1]) \
                     & (ds.lat >= bounds['lat'][0]) & (ds.lat <= bounds['lat'][1])
        ds = ds.where(pre_mask, drop=True)

        if mask is None:
            # circle definition
#            center = (2., 48.)
#            radius = 300. # km

#            dist = haversine(ds.lat, ds.lon, center[1], center[0])
#            mask = dist <= radius * 1e3

            # area definition (bis)
            mask = (ds.lon <= 10.) & (ds.lat <= 50.5) & (ds.lat >= 44.) \
                 & (ds.lat <= (50.5 - 44) / (-5 + 10) * (ds.lon + 10) + 44)

        weights = np.cos(np.deg2rad(ds.lat))
        ds = ds.where(mask, drop=True)
        weights = weights.where(mask, 0., drop=True)

        if varname == 'hcc':
            da = ds.Upper_Layer_Cloud_Top_Pressure < 44000.
        elif varname == 'hcod':
            da = 10**ds.Upper_Layer_Cloud_Optical_Depth + 10**ds.Lower_Layer_Cloud_Optical_Depth
            da = np.log10(da)
#            da = da.where(da < 44000.)
        else:
            raise ValueError

        da = da.weighted(weights)
        da = da.mean(('x', 'y'))

        da_merge = da if da_merge is None else xr.concat((da_merge, da), dim='time')

    return da_merge


def get_era5_profile(variable, times_data, levs_data, lons_data, lats_data):

    if variable is None: return None, None

    filename = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/2022/{variable}.202212.ap1e5.GLOBAL_025.nc'
    if not os.path.exists(filename):
        filename = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_SF/2022/{variable}.202212.as1e5.GLOBAL_025.nc'
        if not os.path.exists(filename):
            filename = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/FC_SF/2022/12/{variable}.20221226.1800.fs1e5.GLOBAL_025.nc'
            if not os.path.exists(filename):
                raise ValueError

    ds_era5 = xr.open_dataset(filename)
    ds_geop = xr.open_dataset(f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/2022/geopt.202212.ap1e5.GLOBAL_025.nc')
    ds_geopsfc = xr.open_dataset(f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_SF/2022/geopt.202212.as1e5.GLOBAL_025.nc')
    ds_geopsfc = ds_geopsfc.rename(geopt='geoptsfc')
    ds_era5 = ds_era5.merge(ds_geop)
    ds_era5 = ds_era5.merge(ds_geopsfc)

    ds_era5, _ = xr_utils.relabelise(ds_era5)
    ds_era5 = xr_utils.reorder_lonlat(ds_era5)

    mask_time = (ds_era5.time >= np.min(times_data) - np.timedelta64(1, 'h')) \
              & (ds_era5.time <= np.max(times_data) + np.timedelta64(1, 'h'))
    mask_lev = (ds_era5.lev >= lev_min) & (ds_era5.lev <= lev_max)
    mask_lon = (ds_era5.lon >= np.min(lons_data) - 0.25) & (ds_era5.lon <= np.max(lons_data) + 0.25)
    mask_lat = (ds_era5.lat >= np.min(lats_data) - 0.25) & (ds_era5.lat <= np.max(lats_data) + 0.25)
    ds_era5 = ds_era5.isel(time=mask_time, lev=mask_lev, lon=mask_lon, lat=mask_lat)

    n_data = max(np.size(times_data), np.size(levs_data), np.size(lons_data), np.size(lats_data), 2)
    points_coords = dict()

    if np.size(times_data) == n_data:
        da_times = xr.DataArray(np.array(times_data, dtype='datetime64[ns]'), dims='points')
        points_coords['time'] = da_times
    elif np.size(times_data) == 1:
        ds_era5 = ds_era5.sel(time=times_data, method='nearest')
    else:
        raise ValueError

    if not levs_data is None:
        if np.size(levs_data) == n_data:
            da_levs = xr.DataArray(np.array(levs_data), dims='points')
            points_coords['lev'] = da_levs
        elif np.size(levs_data) == 1:
            ds_era5 = ds_era5.sel(lev=levs_data, method='nearest')
        else:
            raise ValueError

    da_lons = None
    if np.size(lons_data) == n_data:
        da_lons = xr.DataArray(np.array(lons_data), dims='points')
        points_coords['lon'] = da_lons
    elif np.size(lons_data) == 1:
        ds_era5 = ds_era5.sel(lon=lons_data, method='nearest')
    else:
        raise ValueError

    da_lats = None
    if np.size(lats_data) == n_data:
        da_lats = xr.DataArray(np.array(lats_data), dims='points')
        points_coords['lat'] = da_lats
    elif np.size(lats_data) == 1:
        ds_era5 = ds_era5.sel(lat=lats_data, method='nearest')
    else:
        raise ValueError

    ds_era5 = ds_era5.sel(**points_coords, method='nearest')

    da = ds_era5[variable]
    geoph = (ds_era5['geopt'] - ds_era5['geoptsfc']) / 9.80665

    return da, geoph


def get_ml_era5_r_profile(times_data, levs_data, lons_data, lats_data):

    ds_qt = xr.open_dataset('/scratchu/aborella/OBS/ERA5/q_t_ERA5_ML_case_study.nc')
    ds_sp = xr.open_dataset('/scratchu/aborella/OBS/ERA5/sp_ERA5_ML_case_study.nc')
    ml_to_pl = pd.read_csv('/scratchu/aborella/OBS/ERA5/ML_to_PL.txt', sep=r'\s+',
                           names=['model_level', 'a', 'b'])
    a = xr.DataArray(ml_to_pl['a'], dims=('model_level',), coords={'model_level': ml_to_pl['model_level']})
    b = xr.DataArray(ml_to_pl['b'], dims=('model_level',), coords={'model_level': ml_to_pl['model_level']})

    ds_qt, _ = xr_utils.relabelise(ds_qt)
    ds_qt = xr_utils.reorder_lonlat(ds_qt)
    mask_time = (ds_qt.time >= np.min(times_data) - np.timedelta64(1, 'h')) \
              & (ds_qt.time <= np.max(times_data) + np.timedelta64(1, 'h'))
    mask_lon = (ds_qt.lon >= np.min(lons_data) - 0.25) & (ds_qt.lon <= np.max(lons_data) + 0.25)
    mask_lat = (ds_qt.lat >= np.min(lats_data) - 0.25) & (ds_qt.lat <= np.max(lats_data) + 0.25)
    ds_qt = ds_qt.isel(time=mask_time, lon=mask_lon, lat=mask_lat)

    ds_sp, _ = xr_utils.relabelise(ds_sp)
    ds_sp = xr_utils.reorder_lonlat(ds_sp)
    mask_time = (ds_sp.time >= np.min(times_data) - np.timedelta64(1, 'h')) \
              & (ds_sp.time <= np.max(times_data) + np.timedelta64(1, 'h'))
    mask_lon = (ds_sp.lon >= np.min(lons_data) - 0.25) & (ds_sp.lon <= np.max(lons_data) + 0.25)
    mask_lat = (ds_sp.lat >= np.min(lats_data) - 0.25) & (ds_sp.lat <= np.max(lats_data) + 0.25)
    ds_sp = ds_sp.isel(time=mask_time, lon=mask_lon, lat=mask_lat)

    ds_sp = ds_sp.isel(model_level=0).reset_coords(drop=True)
    pres = a + b * np.exp(ds_sp.lnsp)
    q, t = ds_qt.q, ds_qt.t

    ds_era5 = thermo.speHumToRH(q, t, pres, phase='ice')
    pres = pres / 1e2

    ds_era5 = ds_era5.assign_coords(lev=pres.mean(('lon', 'lat', 'time')))
    ds_era5 = ds_era5.swap_dims(model_level='lev')


    n_data = max(np.size(times_data), np.size(levs_data), np.size(lons_data), np.size(lats_data), 2)
    points_coords = dict()

    if np.size(times_data) == n_data:
        da_times = xr.DataArray(np.array(times_data, dtype='datetime64[ns]'), dims='points')
        points_coords['time'] = da_times
    elif np.size(times_data) == 1:
        ds_era5 = ds_era5.sel(time=times_data, method='nearest')
    else:
        raise ValueError

    if not levs_data is None:
        if np.size(levs_data) == n_data:
            da_levs = xr.DataArray(np.array(levs_data), dims='points')
            points_coords['lev'] = da_levs
        elif np.size(levs_data) == 1:
            ds_era5 = ds_era5.sel(lev=levs_data, method='nearest')
        else:
            raise ValueError

    da_lons = None
    if np.size(lons_data) == n_data:
        da_lons = xr.DataArray(np.array(lons_data), dims='points')
        points_coords['lon'] = da_lons
    elif np.size(lons_data) == 1:
        ds_era5 = ds_era5.sel(lon=lons_data, method='nearest')
    else:
        raise ValueError

    da_lats = None
    if np.size(lats_data) == n_data:
        da_lats = xr.DataArray(np.array(lats_data), dims='points')
        points_coords['lat'] = da_lats
    elif np.size(lats_data) == 1:
        ds_era5 = ds_era5.sel(lat=lats_data, method='nearest')
    else:
        raise ValueError

    ds_era5 = ds_era5.sel(**points_coords, method='nearest')

    return ds_era5, None


def get_lmdz_profile(simu, variable, times_data, levs_data, lons_data, lats_data, cosp=False):

    filename = get_lmdz_filename(simu, cosp=cosp)
    ds_lmdz = xr.open_dataset(filename)
    ds_lmdz, _ = xr_utils.relabelise(ds_lmdz)

    if 'lev' in ds_lmdz.dims:
        ds_lmdz['lev'] = ds_lmdz.lev / 1e2
        if isinstance(variable, list):
            ds_lmdz = ds_lmdz[[*variable, 'geop', 'phis']]
        else:
            ds_lmdz = ds_lmdz[[variable, 'geop', 'phis']]
        ok_lev = True
    else:
        if isinstance(variable, list):
            ds_lmdz = ds_lmdz[[*variable]]
        else:
            ds_lmdz = ds_lmdz[[variable]]
        ok_lev = False

    mask_time = (ds_lmdz.time >= np.min(times_data) - np.timedelta64(15, 'm')) \
              & (ds_lmdz.time <= np.max(times_data) + np.timedelta64(15, 'm'))
    ds_lmdz = ds_lmdz.isel(time=mask_time)

    if ok_lev:
        mask_lev = (ds_lmdz.lev >= lev_min) & (ds_lmdz.lev <= lev_max)
        ds_lmdz = ds_lmdz.isel(lev=mask_lev)

    mask_cell = (ds_lmdz.lon >= np.min(lons_data) - 2.) & (ds_lmdz.lon <= np.max(lons_data) + 2.) \
              & (ds_lmdz.lat >= np.min(lats_data) - 2.) & (ds_lmdz.lat <= np.max(lats_data) + 2.)
    ds_lmdz = ds_lmdz.isel(cell=mask_cell)


    n_data = max(np.size(times_data), np.size(levs_data), np.size(lons_data), np.size(lats_data), 2)
    points_coords = dict()

    if np.size(times_data) == n_data:
        da_times = xr.DataArray(np.array(times_data, dtype='datetime64[ns]'), dims='points')
        points_coords['time'] = da_times
    elif np.size(times_data) == 1:
        ds_lmdz = ds_lmdz.sel(time=times_data, method='nearest')
    else:
        raise ValueError

    if not levs_data is None:
        if np.size(levs_data) == n_data:
            da_levs = xr.DataArray(np.array(levs_data), dims='points')
            points_coords['lev'] = da_levs
        elif np.size(levs_data) == 1:
            ds_lmdz = ds_lmdz.sel(lev=levs_data, method='nearest')
        else:
            raise ValueError

    if np.size(lons_data) == n_data and np.size(lats_data) == n_data:
        lons_data = xr.DataArray(np.array(lons_data), dims=('points',))
        lats_data = xr.DataArray(np.array(lats_data), dims=('points',))
        dist = haversine(lats_data, lons_data, ds_lmdz.lat, ds_lmdz.lon)
        da_cell = dist.argmin('cell')
        points_coords['cell'] = da_cell
        ds_lmdz = ds_lmdz.assign_coords(cell=np.arange(ds_lmdz.cell.size))
    elif np.size(lons_data) == 1 and np.size(lats_data) == 1:
        ds_lmdz = sel_closest(ds_lmdz, lons_data, lats_data)
    else:
        raise ValueError

    ds_lmdz = ds_lmdz.sel(**points_coords, method='nearest')

    da = ds_lmdz[variable]
    if ok_lev:
        geoph = (ds_lmdz['geop'] - ds_lmdz['phis']) / 9.80665
    else:
        geoph = None

    return da, geoph


def get_era5_map(variable, level_snapshot, time_snapshot):

    filename = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/2022/{variable}.202212.ap1e5.GLOBAL_025.nc'
    if not os.path.exists(filename):
        filename = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_SF/2022/{variable}.202212.as1e5.GLOBAL_025.nc'
        if not os.path.exists(filename):
            filename = f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/FC_SF/2022/12/{variable}.20221226.1800.fs1e5.GLOBAL_025.nc'
            if not os.path.exists(filename):
                return None

    ds_era5 = psy.open_dataset(filename)
    ds_era5 = xr_utils.reorder_lonlat(ds_era5)

    ds_era5 = ds_era5.sel(time=time_snapshot, method='nearest')
    if 'level' in ds_era5.dims: ds_era5 = ds_era5.sel(level=level_snapshot, method='nearest')

    return ds_era5


def get_ml_era5_ciwc_map(level_snapshot, time_snapshot):

    ds_era5 = xr.open_dataset('/scratchu/aborella/OBS/ERA5/ciwc_ERA5_ML_case_study.nc')
    ds_cc = xr.open_dataset('/scratchu/aborella/OBS/ERA5/cc_ERA5_ML_case_study.nc')
    ds_sp = xr.open_dataset('/scratchu/aborella/OBS/ERA5/sp_ERA5_ML_case_study.nc')
    ml_to_pl = pd.read_csv('/scratchu/aborella/OBS/ERA5/ML_to_PL.txt', sep=r'\s+',
                           names=['model_level', 'a', 'b'])
    a = xr.DataArray(ml_to_pl['a'], dims=('model_level',), coords={'model_level': ml_to_pl['model_level']})
    b = xr.DataArray(ml_to_pl['b'], dims=('model_level',), coords={'model_level': ml_to_pl['model_level']})

    ds_era5, _ = xr_utils.relabelise(ds_era5)
    ds_era5 = xr_utils.reorder_lonlat(ds_era5)
    ds_era5 = ds_era5.sel(time=time_snapshot, method='nearest')

    ds_cc, _ = xr_utils.relabelise(ds_cc)
    ds_cc = xr_utils.reorder_lonlat(ds_cc)
    ds_cc = ds_cc.sel(time=time_snapshot, method='nearest')

    ds_sp, _ = xr_utils.relabelise(ds_sp)
    ds_sp = xr_utils.reorder_lonlat(ds_sp)
    ds_sp = ds_sp.sel(time=time_snapshot, method='nearest')

    ds_sp = ds_sp.isel(model_level=0).reset_coords(drop=True)
    pres = a + b * np.exp(ds_sp.lnsp)

    ds_era5['cc'] = ds_cc.cc
    ds_era5['qiceincld'] = ds_era5.ciwc / ds_era5.cc

    pres = pres / 1e2
    ds_era5 = ds_era5.assign_coords(lev=pres.mean(('lon', 'lat')))
    ds_era5 = ds_era5.swap_dims(model_level='lev')
    if 'lev' in ds_era5.dims: ds_era5 = ds_era5.sel(lev=level_snapshot, method='nearest')

    return ds_era5


def get_lmdz_map(simu, level_snapshot, time_snapshot, cosp=False):

    filename = get_lmdz_filename(simu, cosp=cosp)
    ds_lmdz = psy.open_dataset(filename)

    if 'presnivs' in ds_lmdz.dims:
        ds_lmdz['presnivs'] = ds_lmdz.presnivs / 1e2
        ds_lmdz = ds_lmdz.sel(presnivs=level_snapshot, method='nearest')
    ds_lmdz = ds_lmdz.sel(time_counter=time_snapshot, method='nearest')

    return ds_lmdz


def get_era5_radflux(varname, times_data, levs_data, lons_data, lats_data):

    filenames = []
    if varname == 'lwup':
        variable = 'str'
        for dt in '20221226.1800', '20221227.600':
            filenames.append(f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/FC_SF/2022/12/{variable}.{dt}.fs1e5.GLOBAL_025.nc')
        variable = 'strd'
        for dt in '20221226.1800', '20221227.600':
            filenames.append(f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/FC_SF/2022/12/{variable}.{dt}.fs1e5.GLOBAL_025.nc')
    elif varname == 'lwdn':
        variable = 'strd'
        for dt in '20221226.1800', '20221227.600':
            filenames.append(f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/FC_SF/2022/12/{variable}.{dt}.fs1e5.GLOBAL_025.nc')
    elif varname == 'swup':
        variable = 'ssr'
        for dt in '20221226.1800', '20221227.600':
            filenames.append(f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/FC_SF/2022/12/{variable}.{dt}.fs1e5.GLOBAL_025.nc')
        variable = 'ssrd'
        for dt in '20221226.1800', '20221227.600':
            filenames.append(f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/FC_SF/2022/12/{variable}.{dt}.fs1e5.GLOBAL_025.nc')
    elif varname == 'swdn':
        variable = 'ssrd'
        for dt in '20221226.1800', '20221227.600':
            filenames.append(f'/bdd/ERA5/NETCDF/GLOBAL_025/hourly/FC_SF/2022/12/{variable}.{dt}.fs1e5.GLOBAL_025.nc')


    preprocess = lambda ds: ds.isel(time=np.arange(12))
    ds_era5 = xr.open_mfdataset(filenames, preprocess=preprocess)

    ds_era5, _ = xr_utils.relabelise(ds_era5)
    ds_era5 = xr_utils.reorder_lonlat(ds_era5)

    da_times = xr.DataArray(np.array(times_data, dtype='datetime64[ns]'), dims='points')
    ds_era5 = ds_era5.sel(time=da_times, lon=lons_data, lat=lats_data, method='nearest')

    if varname == 'lwup':
        da = ds_era5['strd'] - ds_era5['str']
    elif varname == 'lwdn':
        da = ds_era5['strd']
    elif varname == 'swup':
        da = ds_era5['ssrd'] - ds_era5['ssr']
    elif varname == 'swdn':
        da = ds_era5['ssrd']

    # Joules to Watts
    da = da / 3600.

    return da


def get_basta_sirta(time_bounds):
    time_beg, time_end = time_bounds

    basta_nc = xr.open_mfdataset(['/bdd/SIRTA/pub/basesirta/1a/basta/2022/12/26/basta_1a_cldradLz1Lb87R100m-18km_v03_20221226_000000_1440.nc',
                                  '/bdd/SIRTA/pub/basesirta/1a/basta/2022/12/27/basta_1a_cldradLz1Lb87R100m-18km_v03_20221227_000000_1440.nc'])
    reflectivity = basta_nc['reflectivity'].rename(range='lev')
    velocity = basta_nc['velocity'].rename(range='lev')
    background_mask = basta_nc['background_mask'].rename(range='lev')
    alt = basta_nc['range'].rename(range='lev')
    time = basta_nc['time']
    
    mask_time = (time >= time_beg) & (time <= time_end)
    mask_alt = (alt >= alt_min) & (alt <= alt_max)
    
    reflectivity = reflectivity.isel(time=mask_time, lev=mask_alt)
    velocity = velocity.isel(time=mask_time, lev=mask_alt)
    background_mask = background_mask.isel(time=mask_time, lev=mask_alt)
    alt = alt.isel(lev=mask_alt).expand_dims(time=velocity.time)

    reflectivity = reflectivity.where(background_mask != 0, -59.9)
    reflectivity = reflectivity.where(background_mask >= 0)

    return reflectivity, velocity, alt


def get_chm15k_sirta(time_bounds):
    time_beg, time_end = time_bounds
    chm15k_nc = xr.open_dataset('/bdd/SIRTA/pub/basesirta/1a/chm15k/2022/12/27/chm15k_1a_Lz1Lb87Ppr2R15mF15s_v01_20221227_000000_1440.nc')
    cloud_base_height = chm15k_nc['cloud_base_height']
    cloud_depth = chm15k_nc['cdp']
    time = chm15k_nc['time']
    layer = chm15k_nc['layer']
    alt = chm15k_nc['range'].rename(range='lev')
    
    mask_time = (time >= time_beg) & (time <= time_end)
    
    cloud_base_height = cloud_base_height.isel(time=mask_time)
    cloud_depth = cloud_depth.isel(time=mask_time)
    time = time.isel(time=mask_time)

    cloud_presence = np.zeros((time.size, alt.size), dtype=np.bool_)
    for ilay in range(layer.size):
        for itime in range(time.size):
            cbh = cloud_base_height.isel(time=itime, layer=ilay)
            cd = cloud_depth.isel(time=itime, layer=ilay)
            if cd.values == 0 or cd.values == -1: continue
            icld_base = int(np.abs(alt - cbh).argmin())
            icld_top = int(np.abs(alt - (cbh + cd)).argmin())
            cloud_presence[itime, icld_base:icld_top+1] = True

    cloud_presence = xr.DataArray(cloud_presence, dims=('time', 'lev'),
            coords=dict(time=time, lev=alt))

    mask_alt = (alt >= alt_min) & (alt <= alt_max)
    cloud_presence = cloud_presence.isel(lev=mask_alt)
    alt = alt.isel(lev=mask_alt).expand_dims(time=time)

    return cloud_presence, alt


def get_chm15k_qualair(time_bounds):
    time_beg, time_end = time_bounds
    chm15k_nc = xr.open_dataset('/bdd/SIRTA/pub/basesirta/1a/chm15k/2022/12/27/chm15k_1a_LqualairPpr2R15mF15s_v01_20221227_000000_1440.nc')
    cloud_base_height = chm15k_nc['cloud_base_height']
    cloud_depth = chm15k_nc['cdp']
    time = chm15k_nc['time']
    layer = chm15k_nc['layer']
    alt = chm15k_nc['range'].rename(range='lev')
    
    mask_time = (time >= time_beg) & (time <= time_end)
    
    cloud_base_height = cloud_base_height.isel(time=mask_time)
    cloud_depth = cloud_depth.isel(time=mask_time)
    time = time.isel(time=mask_time)

    cloud_presence = np.zeros((time.size, alt.size), dtype=np.bool_)
    for ilay in range(layer.size):
        for itime in range(time.size):
            cbh = cloud_base_height.isel(time=itime, layer=ilay)
            cd = cloud_depth.isel(time=itime, layer=ilay)
            if cd.values == 0 or cd.values == -1: continue
            icld_base = int(np.abs(alt - cbh).argmin())
            icld_top = int(np.abs(alt - (cbh + cd)).argmin())
            cloud_presence[itime, icld_base:icld_top+1] = True

    cloud_presence = xr.DataArray(cloud_presence, dims=('time', 'lev'),
            coords=dict(time=time, lev=alt))

    mask_alt = (alt >= alt_min) & (alt <= alt_max)
    cloud_presence = cloud_presence.isel(lev=mask_alt)
    alt = alt.isel(lev=mask_alt).expand_dims(time=time)

    return cloud_presence, alt


def get_rs_trappes(ddhh):

    dd, hh = ddhh[:2], ddhh[2:]

    ds = xr.open_dataset(f'/bdd/GRUAN/RADIOSONDAGE/L2/TRP/2022/2022_12_{dd}/TRP-RS-01_2_M10-GDP-BETA_001_202212{dd}T{hh}0000_1-000-001.nc')

    mask_valid = np.isfinite(ds.wzon)

    time = ds.time[mask_valid]
    alt = ds.alt[mask_valid]
    lat = ds.lat[mask_valid]
    lon = ds.lon[mask_valid]
    pres = ds.press[mask_valid]
    temp = ds.temp[mask_valid]
    rhl = ds.rh[mask_valid]
    rhi = thermo.convertWater('rl', 'ri', rhl, temp, pres)
    u = ds.wzon[mask_valid]
    v = ds.wmeri[mask_valid]
    wind_force = ds.wspeed[mask_valid]
    wind_dir = ds.wdir[mask_valid]

    return time, lon, lat, alt, pres, temp, rhi, rhl, wind_dir, wind_force, u, v

    """ OLD CODE

    with open(f'/data/aborella/LMDZ/CASE_STUDY/07145.202212{ddhh}.HR.csv') as rs_data:

        _ = rs_data.readline()
        _, timestamp_1, size_1, _ = rs_data.readline().split(',')
        _ = rs_data.readline()
        timestamp_1 = pd.Timestamp(timestamp_1)
        size_1 = int(size_1)

        pres_1 = np.zeros(size_1)
        alt_1 = np.zeros(size_1)
        temp_1 = np.zeros(size_1)
        temp_dew_1 = np.zeros(size_1)
        wind_dir_1 = np.zeros(size_1)
        wind_force_1 = np.zeros(size_1)

        for i in range(size_1):
            pres_tmp, alt_tmp, temp_tmp, temp_dew_tmp, wind_dir_tmp, \
                    wind_force_tmp, _ = rs_data.readline().split(',')
            pres_1[i], alt_1[i], temp_1[i], temp_dew_1[i], wind_dir_1[i], wind_force_1[i] \
                    = float(pres_tmp), float(alt_tmp), float(temp_tmp), \
                    float(temp_dew_tmp), float(wind_dir_tmp), float(wind_force_tmp)


        _ = rs_data.readline()
        _, timestamp_2, size_2, _ = rs_data.readline().split(',')
        _ = rs_data.readline()
        timestamp_2 = pd.Timestamp(timestamp_2)
        size_2 = int(size_2)

        pres_2 = np.zeros(size_2)
        alt_2 = np.zeros(size_2)
        temp_2 = np.zeros(size_2)
        temp_dew_2 = np.zeros(size_2)
        wind_dir_2 = np.zeros(size_2)
        wind_force_2 = np.zeros(size_2)

        for i in range(size_2):
            pres_tmp, alt_tmp, temp_tmp, temp_dew_tmp, wind_dir_tmp, \
                    wind_force_tmp, _ = rs_data.readline().split(',')
            pres_2[i], alt_2[i], temp_2[i], temp_dew_2[i], wind_dir_2[i], wind_force_2[i] \
                    = float(pres_tmp), float(alt_tmp), float(temp_tmp), \
                    float(temp_dew_tmp), float(wind_dir_tmp), float(wind_force_tmp)

    alt_1 = xr.DataArray(data=alt_1, dims=('alt',), coords={'alt': alt_1})
    pres_1 = xr.DataArray(data=pres_1, dims=('alt',), coords={'alt': alt_1})
    temp_1 = xr.DataArray(data=temp_1, dims=('alt',), coords={'alt': alt_1})
    temp_dew_1 = xr.DataArray(data=temp_dew_1, dims=('alt',), coords={'alt': alt_1})
    wind_dir_1 = xr.DataArray(data=wind_dir_1, dims=('alt',), coords={'alt': alt_1})
    wind_force_1 = xr.DataArray(data=wind_force_1, dims=('alt',), coords={'alt': alt_1})

    rhl_1 = thermo.getSatPressure(temp_dew_1, phase='liq') / thermo.getSatPressure(temp_1, phase='liq') * 100.
    rhi_1 = thermo.convertWater('rl', 'ri', rhl_1, temp_1, pres_1)
    u_1 = wind_force_1 * np.cos(np.deg2rad(270. - wind_dir_1))
    v_1 = wind_force_1 * np.sin(np.deg2rad(270. - wind_dir_1))

    ialt_min = int(np.abs(alt_1 - alt_min).argmin())
    ialt_max = int(np.abs(alt_1 - alt_max).argmin())
    alt_1 = alt_1.isel(alt=slice(ialt_min,ialt_max+1))
    pres_1 = pres_1.isel(alt=slice(ialt_min,ialt_max+1))
    temp_1 = temp_1.isel(alt=slice(ialt_min,ialt_max+1))
    rhl_1 = rhl_1.isel(alt=slice(ialt_min,ialt_max+1))
    rhi_1 = rhi_1.isel(alt=slice(ialt_min,ialt_max+1))
    wind_dir_1 = wind_dir_1.isel(alt=slice(ialt_min,ialt_max+1))
    wind_force_1 = wind_force_1.isel(alt=slice(ialt_min,ialt_max+1))
    u_1 = u_1.isel(alt=slice(ialt_min,ialt_max+1))
    v_1 = v_1.isel(alt=slice(ialt_min,ialt_max+1))


    pres_2 = xr.DataArray(data=pres_2, dims=('alt',), coords={'alt': alt_2})
    alt_2 = xr.DataArray(data=alt_2, dims=('alt',), coords={'alt': alt_2})
    temp_2 = xr.DataArray(data=temp_2, dims=('alt',), coords={'alt': alt_2})
    temp_dew_2 = xr.DataArray(data=temp_dew_2, dims=('alt',), coords={'alt': alt_2})
    wind_dir_2 = xr.DataArray(data=wind_dir_2, dims=('alt',), coords={'alt': alt_2})
    wind_force_2 = xr.DataArray(data=wind_force_2, dims=('alt',), coords={'alt': alt_2})

    rhl_2 = thermo.getSatPressure(temp_dew_2, phase='liq') / thermo.getSatPressure(temp_2, phase='liq') * 100.
    rhi_2 = thermo.convertWater('rl', 'ri', rhl_2, temp_2, pres_2)
    u_2 = wind_force_2 * np.cos(np.deg2rad(270. - wind_dir_2))
    v_2 = wind_force_2 * np.sin(np.deg2rad(270. - wind_dir_2))

    ialt_min = int(np.abs(alt_2 - alt_min).argmin())
    ialt_max = int(np.abs(alt_2 - alt_max).argmin())
    alt_2 = alt_2.isel(alt=slice(ialt_min,ialt_max+1))
    pres_2 = pres_2.isel(alt=slice(ialt_min,ialt_max+1))
    temp_2 = temp_2.isel(alt=slice(ialt_min,ialt_max+1))
    rhl_2 = rhl_2.isel(alt=slice(ialt_min,ialt_max+1))
    rhi_2 = rhi_2.isel(alt=slice(ialt_min,ialt_max+1))
    wind_dir_2 = wind_dir_2.isel(alt=slice(ialt_min,ialt_max+1))
    wind_force_2 = wind_force_2.isel(alt=slice(ialt_min,ialt_max+1))
    u_2 = u_2.isel(alt=slice(ialt_min,ialt_max+1))
    v_2 = v_2.isel(alt=slice(ialt_min,ialt_max+1))

    return alt_1, pres_1, temp_1, rhi_1, rhl_1, wind_dir_1, wind_force_1, u_1, v_1, \
           alt_2, pres_2, temp_2, rhi_2, rhl_2, wind_dir_2, wind_force_2, u_2, v_2
    """


def get_iagos(traj):

    if traj == 'Douala_to_Paris':
        iagos = xr.open_dataset('/bdd/IAGOS/netcdf/202212/IAGOS_timeseries_2022122622563915_L2_3.1.1.nc4')
    elif traj == 'Vancouver_to_Francfort':
        iagos = xr.open_dataset('/bdd/IAGOS/netcdf/202212/IAGOS_timeseries_2022122623300606_L2_3.1.0.nc4')

    time = iagos['UTC_time'].rename(dict(UTC_time='time', baro_alt_AC='alt'))
    alt = iagos['baro_alt_AC'].rename(dict(UTC_time='time', baro_alt_AC='alt'))
    pres = iagos['air_press_AC'].rename(dict(UTC_time='time', baro_alt_AC='alt'))
    lon = iagos['lon'].rename(dict(UTC_time='time', baro_alt_AC='alt'))
    lat = iagos['lat'].rename(dict(UTC_time='time', baro_alt_AC='alt'))
    temp = iagos['air_temp_P1'].rename(dict(UTC_time='time', baro_alt_AC='alt'))
    rhi = iagos['RHI_P1'].rename(dict(UTC_time='time', baro_alt_AC='alt'))
    rhl = iagos['RHL_P1'].rename(dict(UTC_time='time', baro_alt_AC='alt'))
    u = iagos['zon_wind_AC'].rename(dict(UTC_time='time', baro_alt_AC='alt'))
    v = iagos['mer_wind_AC'].rename(dict(UTC_time='time', baro_alt_AC='alt'))

    return time, alt, pres, lon, lat, temp, rhi, rhl, u, v


def get_radflux(varname, time_bounds):

    da = None
    step = np.timedelta64(1, 'D')
    for day in np.arange(time_bounds[0].astype('datetime64[D]'), time_bounds[1].astype('datetime64[D]') + step, step):
        if day == np.datetime64('2022-12-26'):
            filename = '26/radfluxupdown_1a_mat10mLz1LpnN10mF1minDp_v02_20221226_000000_1440.dat'
        elif day == np.datetime64('2022-12-27'):
            filename = '27/radfluxupdown_1a_mat10mLz1LpnN10mF1minDp_v02_20221227_000000_1440.dat'
        filename = '/bdd/SIRTA/pub/basesirta/1a/radfluxupdown/2022/12/' + filename

        ds = radflux_mod.load_data(filename)
        da = ds[varname] if da is None else da.combine_first(ds[varname])

    return da


def get_MSG(time, lon_bounds, lat_bounds):

    if time == np.datetime64('2022-12-27T07'): time = np.datetime64('2022-12-27T07:15')

    filename = '/scratchu/aborella/OBS/MSG/202212/MSG4-SEVI-MSGOCAE-0101-0101-202212'
    filenames = glob.glob(filename + pd.to_datetime(time).strftime('%d%H%M') + '*.nc')
    print(filenames)
    assert len(filenames) == 1
    filename = filenames[0]

    ds = xr.open_dataset(filename)
    ds = ds.isel(time=0)
    mask = (ds.longitude >= lon_bounds[0]) & (ds.longitude <= lon_bounds[1]) \
         & (ds.latitude  >= lat_bounds[0]) & (ds.latitude  <= lat_bounds[1])

#    ds = ds[['Upper_Layer_Cloud_Optical_Depth', 'Pixel_Scene_Type']]
#    ds = ds.where(mask)
#    da = ds.Upper_Layer_Cloud_Optical_Depth
#    da = da.where(ds.Pixel_Scene_Type == 112)

    ds = ds[['Upper_Layer_Cloud_Top_Pressure', 'Upper_Layer_Cloud_Optical_Depth']]
    ds = ds.where(mask, drop=True)
    da = ds.Upper_Layer_Cloud_Optical_Depth
    da = da.where(ds.Upper_Layer_Cloud_Top_Pressure < 44000.)

    return da


def get_calipso(varname, lon_bounds, lat_bounds):

    filename = '/scratchu/aborella/OBS/CALIPSO/202212/CAL_LID_L2_05kmCPro-Standard-V4-51.2022-12-27T03-12-10ZN.hdf'
    ds = calipso_mod.load_cpro(filename)

    mask = (ds.lon >= lon_bounds[0]) & (ds.lon <= lon_bounds[1]) & \
           (ds.lat >= lat_bounds[0]) & (ds.lat <= lat_bounds[1])
    ds = ds.isel(points=mask)

    return ds


def get_MODIS(MODIS_ID, lon_bounds, lat_bounds):

    filenames = ['/scratchu/aborella/OBS/MODIS/202212/MODATML2.A2022360.2215.061.2022362035441.hdf',
                 '/scratchu/aborella/OBS/MODIS/202212/MYDATML2.A2022361.0200.061.2022362121650.hdf',
                 '/scratchu/aborella/OBS/MODIS/202212/MODATML2.A2022361.1010.061.2022362093516.hdf']
    times = [np.datetime64('2022-12-26T22:16:00'),
             np.datetime64('2022-12-27T02:00:00'),
             np.datetime64('2022-12-27T10:10:00')]
    filename, time = filenames[MODIS_ID], times[MODIS_ID]

    ds = modis_mod.load_data(filename)
    ds = ds.assign_coords(time=time)

    mask = (ds.lon >= lon_bounds[0]) & (ds.lon <= lon_bounds[1]) \
         & (ds.lat >= lat_bounds[0]) & (ds.lat <= lat_bounds[1])
    ds = ds.where(mask)

    return ds


# unused
def get_MLS(lon_bounds, lat_bounds):

    import h5py
    f = h5py.File('/data/aborella/LMDZ/CASE_STUDY/MLS-Aura_L2GP-RHI_v05-01-c01_2022d361.he5', 'r')
    ds = f['HDFEOS']['SWATHS']['RHI']

    time = ds['Geolocation Fields']['Time'][:]
    time = np.datetime64('1993-01-01T00:00:00') + np.array(time, dtype='timedelta64[s]')
    time = time.astype('datetime64[ns]')
    pres = ds['Geolocation Fields']['Pressure'][:]

    lon = ds['Geolocation Fields']['Longitude'][:]
    lon = xr.DataArray(lon, dims=('time',), coords=dict(time=time))
    lat = ds['Geolocation Fields']['Latitude'][:]
    lat = xr.DataArray(lat, dims=('time',), coords=dict(time=time))
    status = ds['Data Fields']['Status'][:]
    status = xr.DataArray(status, dims=('time',), coords=dict(time=time))
    quality = ds['Data Fields']['Quality'][:]
    quality = xr.DataArray(quality, dims=('time',), coords=dict(time=time))
    convergence = ds['Data Fields']['Convergence'][:]
    convergence = xr.DataArray(convergence, dims=('time',), coords=dict(time=time))
    rhi = ds['Data Fields']['RHI'][:]
    rhi = xr.DataArray(rhi, dims=('time','lev'), coords=dict(time=time, lev=pres))
    prec = ds['Data Fields']['RHIPrecision'][:]
    prec = xr.DataArray(prec, dims=('time','lev'), coords=dict(time=time, lev=pres))

    rhi = rhi.assign_coords(lon=lon, lat=lat)

    rhi = rhi.where(status % 2 == 0)
    rhi = rhi.where(status != 16)
    rhi = rhi.where(status != 32)
    rhi = rhi.where(prec > 0.)
    rhi = rhi.where(convergence <= 2.)
    rhi = rhi.where(quality >= 0.7)
    rhi = rhi.where(rhi.lev <= 317., drop=True)
    rhi = rhi.where(rhi.lev >= 100., drop=True)
    rhi = rhi.dropna('time')

    rhi = rhi.where(rhi.lon >= lon_bounds[0], drop=True)
    rhi = rhi.where(rhi.lon <= lon_bounds[1], drop=True)
    rhi = rhi.where(rhi.lat >= lat_bounds[0], drop=True)
    rhi = rhi.where(rhi.lat <= lat_bounds[1], drop=True)

    # 27/12 entre 3h10 et 3h19
    # 27/12 entre 1h30 et 1h40

    return rhi


def get_IASI(IASI_ID, level_snapshot, lon_bounds, lat_bounds):

    filenames = ['/scratchu/aborella/OBS/IASI/202212/W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,METOPB+IASI_C_EUMP_20221226190256_53306_eps_o_l2.nc',
                 '/scratchu/aborella/OBS/IASI/202212/W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,METOPB+IASI_C_EUMP_20221226204456_53307_eps_o_l2.nc',
                 '/scratchu/aborella/OBS/IASI/202212/W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,METOPB+IASI_C_EUMP_20221227102056_53315_eps_o_l2.nc',
                 '/scratchu/aborella/OBS/IASI/202212/W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,METOPB+IASI_C_EUMP_20221227115952_53316_eps_o_l2.nc',
                 '/scratchu/aborella/OBS/IASI/202212/W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,METOPC+IASI_C_EUMP_20221226195357_21464_eps_o_l2.nc',
                 '/scratchu/aborella/OBS/IASI/202212/W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,METOPC+IASI_C_EUMP_20221227093253_21472_eps_o_l2.nc']
    filename = filenames[IASI_ID]

    ds = iasi_mod.load_data(filename)
    ds = ds.drop_dims(('nerr', 'nerrt', 'nerrw'))

    ds = ds.sel(pressure_levels=level_snapshot, method='nearest')

    mask = (ds.lon >= lon_bounds[0]) & (ds.lon <= lon_bounds[1]) \
         & (ds.lat >= lat_bounds[0]) & (ds.lat <= lat_bounds[1])
    ds = ds.where(mask, drop=True)

    ds = ds.where(~np.isnat(ds.record_start_time), drop=True)
    time = ds.record_start_time.isel(along_track=ds.along_track.size // 2, across_track=ds.across_track.size // 2).values
    ds = ds.drop_vars(('record_start_time', 'record_stop_time'))
    ds = ds.assign_coords(time=time)

    return ds


def get_IASI_profile(IASI_ID):

    filenames = ['/scratchu/aborella/OBS/IASI/202212/W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,METOPB+IASI_C_EUMP_20221226190256_53306_eps_o_l2.nc',
                 '/scratchu/aborella/OBS/IASI/202212/W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,METOPB+IASI_C_EUMP_20221226204456_53307_eps_o_l2.nc',
                 '/scratchu/aborella/OBS/IASI/202212/W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,METOPB+IASI_C_EUMP_20221227102056_53315_eps_o_l2.nc',
                 '/scratchu/aborella/OBS/IASI/202212/W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,METOPB+IASI_C_EUMP_20221227115952_53316_eps_o_l2.nc',
                 '/scratchu/aborella/OBS/IASI/202212/W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,METOPC+IASI_C_EUMP_20221226195357_21464_eps_o_l2.nc',
                 '/scratchu/aborella/OBS/IASI/202212/W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,METOPC+IASI_C_EUMP_20221227093253_21472_eps_o_l2.nc']
    filename = filenames[IASI_ID]

    ds = iasi_mod.load_data(filename)
    ds = ds.drop_dims(('nerr', 'nerrt', 'nerrw'))

#    mask = (ds.lon <= 10.) & (ds.lat <= 50.5) & (ds.lat >= 44.) \
#         & (ds.lat <= (50.5 - 44) / (-5 + 10) * (ds.lon + 10) + 44)
#    ds = ds.where(mask, drop=True)
    ds['lon'] = ds.lon.where(ds.flag_cldnes <= 2)
    lon_sirta, lat_sirta = 2.2, 48.7
    ds = sel_closest(ds, lon_sirta, lat_sirta)

#    ds = ds.where(~np.isnat(ds.record_start_time), drop=True)
#    time = ds.record_start_time.isel(along_track=ds.along_track.size // 2, across_track=ds.across_track.size // 2).values
#    ds = ds.drop_vars(('record_start_time', 'record_stop_time'))
#    ds = ds.assign_coords(time=time)
    time = ds.record_start_time.values
    ds = ds.drop_vars(('record_start_time', 'record_stop_time'))
    ds = ds.assign_coords(time=time)

#    ds = ds.mean(('lon', 'lat'))

    return ds
