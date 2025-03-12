import sys, os
import warnings
warnings.filterwarnings("ignore", message="Converting non-nanosecond precision")

import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import matplotlib.colors as mplc
import matplotlib.cm as mplcm
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import psyplot.project as psy
from cycler import cycler

import numpy as np
import xarray as xr
import great_circle_calculator.great_circle_calculator as gcc

sys.path.append('/home/aborella/UTIL')
import seaborn_tmp as sns
import thermodynamic as thermo
import datasets
import xr_utils

#path_output = '/data/aborella/LMDZ/CASE_STUDY/plots_explicit_cloud_formation'
#path_output = '/data/aborella/LMDZ/CASE_STUDY/plots_explicit_cloud_formation_reice_sun_corrected'
#path_output = '/data/aborella/LMDZ/CASE_STUDY/plots_rmice_ecf'
path_output = '/data/aborella/LMDZ/CASE_STUDY/plots_set3'
map_proj = ccrs.PlateCarree()
map_proj._threshold /= 100.

def plot_map_datasets(level_era5, time_era5, **kwargs):
    """UNUSED FUN - TO DELETE
    """

    cc_era5 = datasets.get_era5_map('cc', level_era5, time_era5, lon_bounds, lat_bounds)
    time_iagos_1, altitude_iagos_1, pres_iagos_1, lon_iagos_1, lat_iagos_1, temp_iagos_1, \
            rhi_iagos_1, rhl_iagos_1, u_iagos_1, v_iagos_1 = datasets.get_iagos('Douala_to_Paris')
    time_iagos_2, altitude_iagos_2, pres_iagos_2, lon_iagos_2, lat_iagos_2, temp_iagos_2, \
            rhi_iagos_2, rhl_iagos_2, u_iagos_2, v_iagos_2 = datasets.get_iagos('Vancouver_to_Francfort')
    time_calipso, lon_calipso, lat_calipso, _, _ = datasets.get_calipso('cc', lon_bounds, lat_bounds)

    idx_time_iagos_1 = []
    for i in range(time_iagos_1[-1].values.astype('datetime64[h]').astype(np.int32) - \
                   time_iagos_1[0].values.astype('datetime64[h]').astype(np.int32) + 1):
        dt = time_iagos_1[0].values.astype('datetime64[h]') + np.timedelta64(i, 'h')
        i_dt = np.abs(time_iagos_1 - dt).argmin()
        idx_time_iagos_1.append(i_dt)
    idx_time_iagos_1 = np.array(idx_time_iagos_1)
    time_key_iagos_1 = time_iagos_1[idx_time_iagos_1].values.astype('datetime64[h]')
    pres_key_iagos_1 = pres_iagos_1[idx_time_iagos_1].values / 1e2
    lon_key_iagos_1 = lon_iagos_1[idx_time_iagos_1].values
    lat_key_iagos_1 = lat_iagos_1[idx_time_iagos_1].values

    idx_time_iagos_2 = []
    for i in range(time_iagos_2[-1].values.astype('datetime64[h]').astype(np.int32) - \
                   time_iagos_2[0].values.astype('datetime64[h]').astype(np.int32) + 2):
        dt = time_iagos_1[0].values.astype('datetime64[h]') + np.timedelta64(i, 'h')
        i_dt = np.abs(time_iagos_2 - dt).argmin()
        idx_time_iagos_2.append(i_dt)
    idx_time_iagos_2 = np.array(idx_time_iagos_2)

    idx_time_calipso = []
    for i in range(time_calipso[-1].values.astype('datetime64[h]').astype(np.int32) - \
                   time_calipso[0].values.astype('datetime64[h]').astype(np.int32) + 2):
        dt = time_calipso[0].values.astype('datetime64[h]') + np.timedelta64(i, 'h')
        i_dt = np.abs(time_calipso - dt).argmin()
        idx_time_calipso.append(i_dt)
    idx_time_calipso = np.array(idx_time_calipso)
    
    ax = plt.axes(projection=map_proj)
    ax.set_xlim(*lon_bounds)
    ax.set_ylim(*lat_bounds)
    ax.coastlines()
    
    # era5
    pco = ax.pcolor(cc_era5.lon, cc_era5.lat, cc_era5, cmap='plasma')
    
    # iagos
    ax.plot(lon_iagos_1, lat_iagos_1, transform=map_proj, lw=3., label='IAGOS 1')
    ax.plot(lon_iagos_2, lat_iagos_2, transform=map_proj, lw=3., label='IAGOS 2')
    
    # calipso
    ax.plot(lon_calipso, lat_calipso, transform=map_proj, lw=3., label='CALIPSO')
    
#    for i in range(idx_time_iagos_1.size):
#        ax.plot(
    ax.legend()
    plt.savefig(f'{path_output}/map_datasets.png', dpi=200.)
    plt.close()
    return


def plot_map_models(level_snapshot, time_snapshot, varname, simu_issr, ok_contour):

    dict_vars = dict(
            u=('u', 'vitu', np.arange(-50., 51., 5.), 'coolwarm', 'U component of wind', 'm/s', False),
            v=('v', 'vitv', np.arange(-30., 31., 5.), 'coolwarm', 'V component of wind', 'm/s', False),
            cc=('cc', 'cfseri', np.arange(0., 1.01, 0.1), 'plasma', 'Cloud fraction', '-', False),
            tcc=('tcc', 'cldt', np.arange(0., 1.01, 0.1), 'Greys', 'Total cloud cover', '-', False),
            hcc=('hcc', 'cldh', np.arange(0., 1.01, 0.1), 'Greys', 'High cloud cover', '-', False),
            mcc=('mcc', 'cldm', np.arange(0., 1.01, 0.1), 'Greys', 'Mid cloud cover', '-', False),
            lcc=('lcc', 'cldl', np.arange(0., 1.01, 0.1), 'Greys', 'Low cloud cover', '-', False),
            q=('q', 'ovap', np.arange(1.e-5, 1.7e-4, 1.e-5), 'Blues', 'Specific humidity', 'kg/kg', False),
            qi=(None, 'ocond', np.arange(0., 1.01e-5, 1.e-6), 'Blues', 'Ice water content', 'kg/kg', False),
            pr=(None, 'pr_lsc_i', np.arange(0., 5.01e-6, 5.e-7), 'Blues', 'Snow flux', 'kg/m2/s', False),
            r=('r', 'rhi', np.arange(0., 145., 10.), 'Blues', 'Relative humidity', '%', False),
            t=('ta', 'temp', np.arange(215., 235., 1.), 'Oranges', 'Air temperature', 'K', False),
            dbz=(None, 'dbze94', np.arange(-60., 0., 5.), 'Greys', 'Reflectivity', 'dBZ', True),
            clhcalipso=(None, 'clhcalipso', np.arange(0., 1.01, 0.1), 'Greys', 'HCC', '-', True),
            )
    name_era5, name_lmdz, bounds, cmap, longname, unit, cosp = dict_vars[varname]
#    lat_bounds=(40.,58.)
#    lon_bounds=(-7.,11.)

    data = (level_snapshot, time_snapshot)
    ds_era5 = datasets.get_era5_map(name_era5, *data)
    ds_ctrl = datasets.get_lmdz_map('CTRL', *data, cosp=cosp)
    ds_issr = datasets.get_lmdz_map(simu_issr, *data, cosp=cosp)
    if varname == 'cc': ds_ctrl = ds_ctrl.rename(rneb='cfseri')
    if varname == 'dbz':
        ds_ctrl = ds_ctrl.sel(height_mlev=level_snapshot, method='nearest')
        ds_ctrl[name_lmdz] = ds_ctrl.dbze94.where(ds_ctrl.dbze94 >= -60.).mean('column')
        bounds_lon, bounds_lat = ds_ctrl.bounds_lon, ds_ctrl.bounds_lat
        ds_ctrl = ds_ctrl[[name_lmdz]].assign_coords(bounds_lon=bounds_lon, bounds_lat=bounds_lat)
        ds_ctrl.to_netcdf('tmp_ctrl.nc')
        ds_ctrl = xr.open_dataset('tmp_ctrl.nc')
        os.system('rm tmp_ctrl.nc')

        ds_issr = ds_issr.sel(height_mlev=level_snapshot, method='nearest')
        ds_issr[name_lmdz] = ds_issr.dbze94.where(ds_issr.dbze94 >= -60.).mean('column')
        bounds_lon, bounds_lat = ds_issr.bounds_lon, ds_issr.bounds_lat
        ds_issr = ds_issr[[name_lmdz]].assign_coords(bounds_lon=bounds_lon, bounds_lat=bounds_lat)
        ds_issr.to_netcdf('tmp_issr.nc')
        ds_issr = xr.open_dataset('tmp_issr.nc')
        os.system('rm tmp_issr.nc')

    fig, axs = plt.subplots(nrows=1,ncols=3,
                            subplot_kw={'projection': map_proj},
                            figsize=(15,5))
    for ax in axs:
        # Longitude labels
        ax.set_xticks(np.arange(lon_bounds[0], lon_bounds[1]+0.1, 9), crs=map_proj)
        lon_formatter = cticker.LongitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)

        # Latitude labels
        ax.set_yticks(np.arange(lat_bounds[0], lat_bounds[1]+0.1, 6), crs=map_proj)
        lat_formatter = cticker.LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)

        # Circle tuning
        p_center = (2., 49.)
        height = 2. * abs(gcc.point_given_start_and_bearing(p_center, 0., 2e6)[1] - 49.)
        width = 2. * abs(gcc.point_given_start_and_bearing(p_center, 90., 2e6)[0] - 2.)
        tuning_zone = mpl_patches.Ellipse(p_center, width, height,
                lw=2., color='red', fill=False, transform=map_proj, zorder=20)
        ax.add_patch(tuning_zone)

        # Make the panels squares
        ax.set_aspect('auto')

    plot_method = 'contourf' if ok_contour else 'mesh'
    for ax, ds, name, ok_unstruct in zip(axs,
            (ds_era5, ds_ctrl, ds_issr),
            (name_era5, name_lmdz, name_lmdz),
            (False, True, True)):

        if ds is None: continue

        ds.psy.plot.mapplot(name=name, bounds=bounds, plot=plot_method,
                            cbar=None, ax=ax, cmap=cmap, map_extent=[*lon_bounds, *lat_bounds])

        if varname == 'r':
            ds, _ = xr_utils.relabelise(ds)
            contour_fun = ax.tricontour if ok_unstruct else ax.contour
            contour_fun(ds.lon, ds.lat, ds[name],
                    levels=[100.,], colors='black', transform=map_proj)

    ax_era5, ax_ctrl, ax_issr = axs
    ax_era5.set(title='ERA5')
    ax_ctrl.set(title=f'LMDZ - CTRL')
    ax_issr.set(title=f'LMDZ - ISSR {simu_issr}')

    norm = mplc.Normalize(bounds.min(), bounds.max())
    cmap = mpl.colormaps[cmap].resampled(bounds.size - 1)
    cbar = fig.colorbar(mplcm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axs, orientation='horizontal')

    fig.suptitle(f'{longname} [{unit}] at {level_snapshot:.0f} hPa and {time_snapshot}')
    plt.savefig(f'{path_output}/map_{varname}_{level_snapshot:.0f}_hPa_{time_snapshot}_ISSR_{simu_issr}.png', dpi=200.)
#    plt.show()

    plt.close()
    return


def plot_profile(ref, varname, simu_issr):

    dict_vars = dict(
            u=('u', 'vitu', np.arange(-50., 51., 5.), 'coolwarm', 'U component of wind', 'm/s'),
            v=('v', 'vitv', np.arange(-30., 31., 5.), 'coolwarm', 'V component of wind', 'm/s'),
            cc=('cc', 'cfseri', np.arange(0., 1.01, 0.1), 'plasma', 'Cloud fraction', '-'),
            pr=(None, 'pr_lsc_i', np.arange(0., 1.e-5, 1.e-6), 'plasma', 'Snow flux', 'kg/m2/s'),
            q=('q', 'ovap', np.arange(1.e-5, 1.7e-4, 1.e-5), 'Blues', 'Specific humidity', 'kg/kg'),
            qi=('ciwc', 'ocond', np.arange(1.e-5, 1.7e-4, 1.e-5), 'Blues', 'Ice water content', 'kg/kg'),
            r=('r', 'rhi', np.arange(0., 150., 10.), 'Blues', 'Relative humidity', '%'),
            t=('ta', 'temp', np.arange(215., 235., 1.), 'Oranges', 'Air temperature', 'K'),
            )
    name_era5, name_lmdz, clevs, cmap, longname, unit = dict_vars[varname]

    if varname == 'cc':
        if ref == 'CHM15K-SIRTA':
            da_ref, geoph_ref = datasets.get_chm15k_sirta((time_beg, time_end))
            times = np.arange(time_beg.astype('datetime64[m]'), time_end.astype('datetime64[m]')+1, 15)
            lons, lats = lon_sirta, lat_sirta
        elif ref == 'BASTA-SIRTA':
            reflectivity, velocity, geoph_ref = datasets.get_basta_sirta((time_beg, time_end))
            da_ref = reflectivity
            times = np.arange(time_beg.astype('datetime64[m]'), time_end.astype('datetime64[m]')+1, 15)
            lons, lats = lon_sirta, lat_sirta
        elif ref == 'CHM15K-QUALAIR':
            da_ref, geoph_ref = datasets.get_chm15k_qualair((time_beg, time_end))
            times = np.arange(time_beg.astype('datetime64[m]'), time_end.astype('datetime64[m]')+1, 15)
            lons, lats = lon_qualair, lat_qualair
        else:
            raise ValueError
    else:
        times = np.arange(time_beg.astype('datetime64[m]'), time_end.astype('datetime64[m]')+1, 15)
        lons, lats = lon_sirta, lat_sirta
        da_ref, time_ref, geoph_ref = None, None, None


    data = (times, None, lons, lats)

#    da_era5 = geoph_era5 = None
    da_era5, geoph_era5 = datasets.get_era5_profile(name_era5, *data)
    da_npar, geoph_npar = datasets.get_lmdz_profile(simu_issr, name_lmdz, *data)
    if varname == 'cc': name_lmdz = 'rnebls'
    da_ctrl, geoph_ctrl = datasets.get_lmdz_profile('CTRL', name_lmdz, *data)

    da_era5 = da_era5.swap_dims(points='time').sortby('lev', ascending=False)
    da_ctrl = da_ctrl.swap_dims(points='time').sortby('lev', ascending=False)
    da_npar = da_npar.swap_dims(points='time').sortby('lev', ascending=False)
    geoph_era5 = geoph_era5.swap_dims(points='time').sortby('lev', ascending=False)
    geoph_ctrl = geoph_ctrl.swap_dims(points='time').sortby('lev', ascending=False)
    geoph_npar = geoph_npar.swap_dims(points='time').sortby('lev', ascending=False)


    fig, axs = plt.subplots(nrows=2, ncols=2, layout='constrained')
    axs = axs.flatten()

    for ax in axs:
        ax.set_xlim(time_beg, time_end)
        ax.set_ylim(*datasets.alt_bounds)

        # Time labels
        ax.set_xticks(np.arange(time_beg, time_end+1, 3))
        ax.set_xticklabels(['00h', '03h', '06h', '09h', '12h'])

        # Altitude labels
        ax.set_yticks(np.arange(datasets.alt_bounds[0], datasets.alt_bounds[1] + 1., 2000.))
        ax.set_yticklabels(np.arange(datasets.alt_bounds[0] / 1e3, datasets.alt_bounds[1] / 1e3 + 1, 2, dtype=int))


    for ax, geoph, da in zip(axs,
             (geoph_ref, geoph_era5, geoph_ctrl, geoph_npar),
             (da_ref, da_era5, da_ctrl, da_npar)):

        if da is None: continue

        time = da.time.expand_dims(lev=da.lev).transpose('time', 'lev')
        geoph = geoph.transpose('time', 'lev')
        da = da.transpose('time', 'lev')

        cf = ax.contourf(time, geoph, da, clevs, cmap=cmap)
        if varname == 'r':
            ax.contour(time, geoph, da, [100.], colors='black')
#        cf = ax.contourf(time, geoph, da, cmap=cmap)
#        cbar = fig.colorbar(cf, ax=ax)


    ax_ref, ax_era5, ax_ctrl, ax_npar = axs
    if not da_ref is None: ax_ref.set(title=ref)
    ax_era5.set(title='ERA5')
    ax_ctrl.set(title=f'LMDZ - CTRL')
    ax_npar.set(title=f'LMDZ - ISSR {simu_issr}')

    cbar = fig.colorbar(cf, ax=axs)
    fig.suptitle(f'{longname} [{unit}]')
    if da_ref is None:
        plt.savefig(f'{path_output}/profile_{varname}_ISSR_{simu_issr}.png', dpi=200.)
    else:
        plt.savefig(f'{path_output}/profile_{varname}_{ref}_ISSR_{simu_issr}.png', dpi=200.)
#    plt.show()

    plt.close()
    return


def plot_radiative(varname, simu_issr):

    dict_vars = dict(
            lwup=('LW_up', 'LWupSFC', 'Longwave upwelling flux'),
            lwdn=('LW_down', 'LWdnSFC', 'Longwave downwelling flux'),
            swup=('SW_up', 'SWupSFC', 'Shortwave upwelling flux'),
            swdn=('SW_down', 'SWdnSFC', 'Shortwave downwelling flux'),
            )
    name_obs, name_lmdz, longname = dict_vars[varname]
    time_bounds = (np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T15'))

    step = np.timedelta64(15, 'm')
    times = np.arange(time_bounds[0], time_bounds[1] + step, step)
#    time_rad = np.arange(np.datetime64('2022-12-26T21:15'), np.datetime64('2022-12-27T15'), np.timedelta64(90, 'm'))
#    times = time_rad
    data = (times, None, lon_sirta, lat_sirta)

    da_obs = datasets.get_radflux(name_obs, time_bounds)
    da_era5 = datasets.get_era5_radflux(varname, *data)
#    da_era5, _ = datasets.get_era5_profile('cc', *data)
#    name_lmdz = 'rnebls'
    da_ctrl, _ = datasets.get_lmdz_profile('CTRL', name_lmdz, *data)
#    name_lmdz = 'cfseri'
    da_issr, _ = datasets.get_lmdz_profile(simu_issr, name_lmdz, *data)
#    da_era5 = da_era5.isel(points=8)
#    da_ctrl = da_ctrl.isel(points=32)
#    da_issr = da_issr.isel(points=32)
#    da_ctrl = da_ctrl.sel(lev=980., method='nearest')
#    da_issr = da_issr.sel(lev=980., method='nearest')
#    print(da_era5, da_ctrl, da_issr)

    fig, ax = plt.subplots()

    ax.plot(da_obs.time, da_obs, label='MSG', color='grey')
    ax.plot(da_era5.time, da_era5, label='ERA5', color='black')
    ax.plot(da_ctrl.time, da_ctrl, label='CTRL', color='tab:blue')
    ax.plot(da_issr.time, da_issr, label='ISSR', color='tab:orange')

    ax.set(ylabel=f'{longname} [W/m2]', title=longname)

#    ax.set(xlabel='Time', xlim=time_bounds)
#    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=15, ha='right')
    ax.legend()

    plt.savefig(f'{path_output}_TS/timeseries_{varname}_ISSR_{simu_issr}.png', dpi=200.)
#    plt.show()
    plt.close()
    return


def plot_radiosounding(ddhh, varnames, simu_issr):

    time_rs, lon_rs, lat_rs, alt_rs, pres_rs, temp_rs, rhi_rs, rhl_rs, \
            wind_dir_rs, wind_force_rs, u_rs, v_rs = datasets.get_rs_trappes(ddhh)

    data = (time_rs, pres_rs, lon_rs, lat_rs)

    u_era5, alt_era5 = datasets.get_era5_profile('u', *data)
    u_ctrl, alt_ctrl = datasets.get_lmdz_profile('CTRL', 'vitu', *data)
    u_npar, alt_npar = datasets.get_lmdz_profile(simu_issr, 'vitu', *data)

    _, unique_era5 = np.unique(alt_era5, return_index=True)
    _, unique_lmdz = np.unique(alt_ctrl, return_index=True)
    alt_era5 = alt_era5[unique_era5]
    alt_ctrl = alt_ctrl[unique_lmdz]
    alt_npar = alt_npar[unique_lmdz]
    lev_era5 = alt_era5.lev * 1e2

    u_era5 = u_era5[unique_era5]
    u_ctrl = u_ctrl[unique_lmdz]
    u_npar = u_npar[unique_lmdz]

    v_era5, _ = datasets.get_era5_profile('v', *data)
    v_ctrl, _ = datasets.get_lmdz_profile('CTRL', 'vitv', *data)
    v_npar, _ = datasets.get_lmdz_profile(simu_issr, 'vitv', *data)
    v_era5 = v_era5[unique_era5]
    v_ctrl = v_ctrl[unique_lmdz]
    v_npar = v_npar[unique_lmdz]


    wind_force_era5 = np.sqrt(u_era5**2. + v_era5**2.)
    wind_dir_era5 = (270. - np.rad2deg(np.arctan2(v_era5, u_era5))) % 360
    wind_force_ctrl = np.sqrt(u_ctrl**2. + v_ctrl**2.)
    wind_dir_ctrl = (270. - np.rad2deg(np.arctan2(v_ctrl, u_ctrl))) % 360
    wind_force_npar = np.sqrt(u_npar**2. + v_npar**2.)
    wind_dir_npar = (270. - np.rad2deg(np.arctan2(v_npar, u_npar))) % 360


    fig, axs = plt.subplots(nrows=2, ncols=2, layout='constrained')
    axs = axs.flatten()

    for (ax, var) in zip(axs, varnames):

        if var == 'wind_force':
            da_era5 = wind_force_era5
            da_ctrl = wind_force_ctrl
            da_npar = wind_force_npar
            da_rs = wind_force_rs

            ax.set(title='Wind force [m/s]', xlim=(0., 50.))

        elif var == 'wind_dir':
            da_era5 = wind_dir_era5
            da_ctrl = wind_dir_ctrl
            da_npar = wind_dir_npar
            da_rs = wind_dir_rs

            ax.set(title='Wind dir. [deg]')#, xlim=(0.,360.))

        elif var == 'u':
            da_era5 = u_era5
            da_ctrl = u_ctrl
            da_npar = u_npar
            da_rs = u_rs

            ax.set(title='U comp. of wind [m/s]')

        elif var == 'v':
            da_era5 = v_era5
            da_ctrl = v_ctrl
            da_npar = v_npar
            da_rs = v_rs

            ax.set(title='V comp. of wind [m/s]')

        elif var == 't':
            da_era5, _ = datasets.get_era5_profile('ta', *data)
            da_ctrl, _ = datasets.get_lmdz_profile('CTRL', 'temp', *data)
            da_npar, _ = datasets.get_lmdz_profile(simu_issr, 'temp', *data)
            da_era5 = da_era5[unique_era5]
            da_ctrl = da_ctrl[unique_lmdz]
            da_npar = da_npar[unique_lmdz]
            da_rs = temp_rs

            ax.set(title='Temperature [K]')

        elif var == 'rhi':
            ovap_era5, _ = datasets.get_era5_profile('q', *data)
            temp_era5, _ = datasets.get_era5_profile('ta', *data)
            ovap_era5 = ovap_era5[unique_era5]
            temp_era5 = temp_era5[unique_era5]
            da_era5 = thermo.speHumToRH(ovap_era5, temp_era5, lev_era5, phase='ice')
            da_ctrl, _ = datasets.get_lmdz_profile('CTRL', 'rhi', *data)
            da_npar, _ = datasets.get_lmdz_profile(simu_issr, 'rhi', *data)
            da_ctrl = da_ctrl[unique_lmdz]
            da_npar = da_npar[unique_lmdz]
            da_rs = rhi_rs

            ax.set(title='RH w.r.t. ice [%]')

        elif var == 'rhl':
            ovap_era5, _ = datasets.get_era5_profile('q', *data)
            temp_era5, _ = datasets.get_era5_profile('ta', *data)
            ovap_era5 = ovap_era5[unique_era5]
            temp_era5 = temp_era5[unique_era5]
            da_era5 = thermo.speHumToRH(ovap_era5, temp_era5, lev_era5, phase='liq')
            da_ctrl, _ = datasets.get_lmdz_profile('CTRL', 'rhl', *data)
            da_npar, _ = datasets.get_lmdz_profile(simu_issr, 'rhl', *data)
            da_ctrl = da_ctrl[unique_lmdz]
            da_npar = da_npar[unique_lmdz]
            da_rs = rhl_rs

            ax.set(title='RH w.r.t. liq. [%]')

        ax.plot(da_era5, alt_era5, label='ERA5', lw=2., ls='-.', color='black')
        ax.plot(da_ctrl, alt_ctrl, label=f'CTRL', lw=2., color=sns.color_palette()[0])
        ax.plot(da_npar, alt_npar, label=f'ISSR {simu_issr}', lw=2., color=sns.color_palette()[1])
        ax.plot(da_rs, alt_rs, label='RS', lw=2., ls='--', color=sns.color_palette()[2])

        ax.set(ylim=(1000.,12000.), ylabel='Altitude [m]')
    axs[0].legend()
    plt.suptitle(f'Trappes RS at {np.datetime64(f"2022-12-{ddhh[:2]}T{ddhh[2:]}")}')
#    plt.show()

    plt.savefig(f'{path_output}/profile_RS_{ddhh[:2]}T{ddhh[2:]}_ISSR_{simu_issr}.png', dpi=200.)

    plt.close()
    return


def plot_timeseries(vartype, simu_issr):

    # possible levs: 400, 375, 350, 325, 300, 280, 250, 235, 215, 195
    bounds = dict(time=(np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T15')),
                  lev=(350., 225.), lon=(-11.,11.), lat=(43.,51.))
    time_rad = np.arange(np.datetime64('2022-12-26T21:15'), np.datetime64('2022-12-27T15'), np.timedelta64(90, 'm'))
    time_rad = np.arange(np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T15'), np.timedelta64(15, 'm'))

    varnames_obs = varnames_era5 = varnames_ctrl = varnames_issr = None
    if vartype == 'frac':
        varnames_issr = ['cfseri', 'issrfra']
    elif vartype == 'rhum':
        varnames_issr = ['rhi', 'qcld', 'cfseri', 'issrfra', 'ovap', 'qissr', 'rvcseri', 'ocond']
    elif vartype == 'spehum':
        varnames_issr = ['qcld', 'ovap', 'qissr', 'rvcseri', 'ocond', 'oliq']
    elif vartype == 'tendfrac':
        varnames_issr = ['dcfcon', 'dcfdyn', 'dcfmix', 'dcfsub']
    elif vartype == 'tendice':
        varnames_issr = ['dqicon', 'dqsdyn', 'dqisub', 'dqiadj', 'pr_lsc_i']
    elif vartype == 'tendcloudvap':
        varnames_issr = ['dqvccon', 'dqvcsub', 'dqvcmix', 'dqvcadj', 'drvcdyn', 'ovap']
    elif vartype == 'cldh':
        varnames_obs = varnames_era5 = 'hcc'
        varnames_ctrl = varnames_issr = 'cldh'
    elif vartype == 'cldm':
        varnames_era5 = 'mcc'
        varnames_ctrl = varnames_issr = 'cldm'
    elif vartype == 'cldl':
        varnames_era5 = 'lcc'
        varnames_ctrl = varnames_issr = 'cldl'
    elif 'hcc' in vartype:
        varnames_obs = varnames_era5 = 'hcc'
    elif 'iwp' in vartype:
        varnames_era5 = 'tciw'
        varnames_ctrl = varnames_issr = vartype
    elif 'lwp' in vartype:
        varnames_era5 = 'tclw'
        varnames_ctrl = varnames_issr = vartype
    elif vartype == 'vwp':
        raise NotImplementedError
        varnames_era5 = 'tcwv'
    elif 'hcod' in vartype:
        varnames_obs = 'hcod'

    da_obs  = datasets.get_MSG_average(varnames_obs, **bounds)
    da_era5 = datasets.get_era5_average(varnames_era5, **bounds)
    da_ctrl = datasets.get_lmdz_average(vartype, varnames_ctrl, 'CTRL', **bounds)
    da_issr = datasets.get_lmdz_average(vartype, varnames_issr, simu_issr, **bounds)


    fig, ax = plt.subplots()
    if 'hcc' in vartype or vartype == 'cldh':
        da_ctrl = da_ctrl.sel(time=time_rad, method='nearest')
        da_issr = da_issr.sel(time=time_rad, method='nearest')
        ax.plot(da_obs.time, da_obs, label='MSG', color='grey')
        ax.plot(da_era5.time, da_era5, label='ERA5', color='black')
        ax.plot(da_ctrl.time, da_ctrl, label='CTRL', color='tab:blue')
        ax.plot(da_issr.time, da_issr, label='ISSR', color='tab:orange')

        ax.set(ylabel='High cloud cover [-]', title='High cloud cover')

    elif vartype == 'cldm':
        da_ctrl = da_ctrl.sel(time=time_rad, method='nearest')
        da_issr = da_issr.sel(time=time_rad, method='nearest')
        ax.plot(da_era5.time, da_era5, label='ERA5', color='black')
        ax.plot(da_ctrl.time, da_ctrl, label='CTRL', color='tab:blue')
        ax.plot(da_issr.time, da_issr, label='ISSR', color='tab:orange')

        ax.set(ylabel='Mid cloud cover [-]', title='Mid cloud cover')

    elif vartype == 'cldl':
        da_ctrl = da_ctrl.sel(time=time_rad, method='nearest')
        da_issr = da_issr.sel(time=time_rad, method='nearest')
        ax.plot(da_era5.time, da_era5, label='ERA5', color='black')
        ax.plot(da_ctrl.time, da_ctrl, label='CTRL', color='tab:blue')
        ax.plot(da_issr.time, da_issr, label='ISSR', color='tab:orange')

        ax.set(ylabel='Low cloud cover [-]', title='Low cloud cover')

    elif 'hcod' in vartype:
        da_ctrl = da_ctrl.sel(time=time_rad, method='nearest')
        da_issr = da_issr.sel(time=time_rad, method='nearest')
        ax.plot(da_obs.time, da_obs, label='MSG', color='grey')
        ax.plot(da_ctrl.time, da_ctrl, label='CTRL', color='tab:blue')
        ax.plot(da_issr.time, da_issr, label='ISSR', color='tab:orange')

        ax.set(ylabel='COD log10 [-]', title='Cloud optical depth (log10)')

    elif 'iwp' in vartype:
        da_ctrl = da_ctrl.sel(time=time_rad, method='nearest')
        da_issr = da_issr.sel(time=time_rad, method='nearest')
        ax.plot(da_era5.time, da_era5, label='ERA5', color='black')
        ax.plot(da_ctrl.time, da_ctrl, label='CTRL', color='tab:blue')
        ax.plot(da_issr.time, da_issr, label='ISSR', color='tab:orange')

        ax.set(ylabel='Ice water path [kg/m2]', title='Ice water path')

    elif 'lwp' in vartype:
        da_ctrl = da_ctrl.sel(time=time_rad, method='nearest')
        da_issr = da_issr.sel(time=time_rad, method='nearest')
        ax.plot(da_era5.time, da_era5, label='ERA5', color='black')
        ax.plot(da_ctrl.time, da_ctrl, label='CTRL', color='tab:blue')
        ax.plot(da_issr.time, da_issr, label='ISSR', color='tab:orange')

        ax.set(ylabel='Liquid water path [kg/m2]', title='Liquid water path')

    elif vartype == 'frac':
        ax.plot(da_issr.time, da_issr.cfseri, label=f'Cloud fraction', color='tab:orange')
        ax.plot(da_issr.time, da_issr.issrfra, label=f'ISSR fraction', color='tab:green')
        ax.plot(da_issr.time, da_issr.subfra, label=f'Subsat. sky fraction', color='tab:blue')

        ax.set(ylabel='Fraction [-]', title='Fractions')
    
    elif vartype == 'rhum':
        ax.plot(da_issr.time, da_issr.rhi, label=f'All water vapor', color='black')
        ax.plot(da_issr.time, da_issr.rhi_cld, label=f'Total cloud content', color='tab:orange')
        ax.plot(da_issr.time, da_issr.rhi_cldvap, label=f'Cloud water vapor', color='tab:red')
        ax.plot(da_issr.time, da_issr.rhi_issr, label=f'ISSR', color='tab:green')
        ax.plot(da_issr.time, da_issr.rhi_clr, label=f'Clear sky (incl. ISSR)', color='lightblue')
        ax.plot(da_issr.time, da_issr.rhi_sub, label=f'Subsat. sky', color='tab:blue')

        ax.set(ylabel='Relative humidity w.r.t. [%]', title='Relative humidities w.r.t. ice')

    elif vartype == 'spehum':
        ax.plot(da_issr.time, da_issr.qcld , label=f'Total cloud content', color='tab:orange')
        ax.plot(da_issr.time, da_issr.qvc  , label=f'Cloud water vapor', color='tab:red')
        ax.plot(da_issr.time, da_issr.qice , label=f'Ice content', color='tab:purple')
        ax.plot(da_issr.time, da_issr.qissr, label=f'ISSR', color='tab:green')
        ax.plot(da_issr.time, da_issr.qvap , label=f'Total vapor', color='black')
        ax.plot(da_issr.time, da_issr.qsub , label=f'Subsat. sky', color='tab:blue')
        ax.plot(da_issr.time, da_issr.qclr , label=f'Clear sky (incl. ISSR)', color='lightblue')

        ax.set(ylabel='Specific humidity [kg/kg]', title='Specific humidities')

    elif vartype == 'tendfrac':
        ax.plot(da_issr.time, da_issr.dcfcon, label=f'Cond.', color='tab:blue')
        ax.plot(da_issr.time, da_issr.dcfsub, label=f'Subl.', color='tab:orange')
        ax.plot(da_issr.time, da_issr.dcfmix, label=f'Mix.', color='tab:red')
        ax.plot(da_issr.time, da_issr.dcfdyn, label=f'Adv.', color='black')

        ax.set(ylabel='Fraction tendency [s-1]', title='Fraction tendencies')

    elif vartype == 'tendice':
        ax.plot(da_issr.time, da_issr.dqicon, label=f'Cond.', color='tab:blue')
        ax.plot(da_issr.time, da_issr.dqisub, label=f'Subl.', color='tab:orange')
        ax.plot(da_issr.time, da_issr.dqised, label=f'Sed.', color='tab:purple')
        ax.plot(da_issr.time, da_issr.dqiadj, label=f'Adj.', color='tab:green')
        ax.plot(da_issr.time, da_issr.dqsdyn, label=f'Adv.', color='black')

        ax.set(ylabel='Cloud ice tendency [kg/kg/s]', title='Cloud ice tendencies')

    elif vartype == 'tendcloudvap':
        ax.plot(da_issr.time, da_issr.dqvccon, label=f'Cond.', color='tab:blue')
        ax.plot(da_issr.time, da_issr.dqvcsub, label=f'Subl.', color='tab:orange')
        ax.plot(da_issr.time, da_issr.dqvcadj, label=f'Adj.', color='tab:green')
        ax.plot(da_issr.time, da_issr.dqvcmix, label=f'Mix.', color='tab:red')
        ax.plot(da_issr.time, da_issr.drvcdyn * da_issr.ovap, label=f'Adv.', color='black')

        ax.set(ylabel='Cloud vapor tendency [kg/kg/s]', title='Cloud vapor tendencies')

    ax.set(xlabel='Time', xlim=bounds.get('time'))
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=15, ha='right')
    ax.legend()

    add_str = ''
    if vartype in ('frac', 'rhum', 'spehum', 'tendfrac', 'tendice', 'tendcloudvap'):
        if len(bounds.get('lev')) == 2:
            add_str = f' - {bounds.get("lev")[0]:.0f} to {bounds.get("lev")[1]:.0f} hPa'
        else:
            add_str = f' - {bounds.get("lev")[0]:.0f} hPa'
    else:
        if 'modis' in vartype:
            add_str = ' - COSP MODIS simulator'
        elif 'calipso' in vartype:
            add_str = ' - COSP CALIPSO simulator'
        elif 'isccp' in vartype:
            add_str = ' - COSP ISCCP simulator'
    ax.set_title(ax.title.get_text() + add_str)

    plt.savefig(f'{path_output}_TS/timeseries_{simu_issr}_{vartype}.png', dpi=200.)
#    plt.show()
    plt.close()
    return


# for maps and other
lon_bounds = (-25., 29.)
lat_bounds = (31., 67.)
time_snapshot = np.datetime64('2022-12-27T03')
level_snapshot = 300.

# for profiles
lon_sirta = 2.2
lat_sirta = 48.7
lon_qualair = 2.4
lat_qualair = 48.8
lon_trappes = 2.009722
lat_trappes = 48.774444
time_beg = np.datetime64('2022-12-27T00')
time_end = np.datetime64('2022-12-27T12')


# psyplot params
psy.rcParams['plotter.maps.map_extent'] = [*lon_bounds, *lat_bounds]
psy.rcParams['plotter.maps.projection'] = 'cyl' # PlateCarree
psy.rcParams['plotter.maps.xgrid'] = False
psy.rcParams['plotter.maps.ygrid'] = False
# allowed resolutions: ('110m', '50m', '10m')
psy.rcParams['plotter.maps.lsm'] = dict(res='110m', linewidth=0.5, coast='darkgrey')

plt.rcParams.update({
                     'figure.dpi': 100,
                     'figure.constrained_layout.use': True,
                     'figure.constrained_layout.h_pad': 5./72.,
                     'figure.constrained_layout.w_pad': 5./72.,
                     'figure.figsize': (6.4, 4.8),
                     'lines.linewidth': 3.,
                     'axes.prop_cycle': cycler(color=sns.color_palette()),
                     'axes.labelsize': 15,
                     'axes.titlesize': 15,
                     'legend.fontsize': 12,
                     'xtick.labelsize': 12,
                     'ytick.labelsize': 12,
                     'font.size': 15,
                     'font.family': 'Lato',
                    })
