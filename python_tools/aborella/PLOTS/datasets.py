import sys, os
import glob
import xarray as xr
import numpy as np
import scipy.spatial as sspatial
import matplotlib.pyplot as plt
import pandas as pd
from pyhdf.SD import SD, SDC
import psyplot.project as psy

# sys.path.append('/home/aborella/UTIL')
# import thermodynamic as thermo
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
