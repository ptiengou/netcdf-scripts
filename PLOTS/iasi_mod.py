import warnings
warnings.filterwarnings('ignore', message='Converting non-nanosecond precision timedelta values to nanosecond precision. This behavior can eventually be relaxed in xarray, as it is an artifact from pandas which is now beginning to support non-nanosecond precision values. This warning is caused by passing non-nanosecond np.datetime64 or np.timedelta64 values to the DataArray or Variable constructor; it can be silenced by converting the values to nanosecond precision ahead of time.')

import numpy as np
import xarray as xr

import sys
sys.path.append('/home/aborella/UTIL')
import thermodynamic as thermo

def load_data(filename):

    # see https://user.eumetsat.int/s3/eup-strapi-media/IASI_Level_2_Product_Guide_8f61a2369f.pdf

    ds = xr.open_dataset(filename, decode_times=False)
#    ds = ds.isel(along_track=ds.along_track.size // 2, across_track=ds.across_track.size // 2)
#    npct, npcw = ds.npct.size, ds.npcw.size
#    npct, npcw = 28, 18
#    print(f'{npct=}, {npcw=}')
    ds = ds.drop_vars(('forli_layer_heights_co','forli_layer_heights_hno3','forli_layer_heights_o3','brescia_altitudes_so2','degraded_ins_MDR','degraded_proc_MDR','solar_zenith','satellite_zenith','solar_azimuth','satellite_azimuth','fg_atmospheric_ozone','fg_surface_temperature','fg_qi_atmospheric_ozone','fg_qi_surface_temperature','atmospheric_ozone','surface_temperature','integrated_ozone','integrated_n2o','integrated_co','integrated_ch4','integrated_co2','surface_emissivity','number_cloud_formations','cloud_phase','surface_pressure','instrument_mode','spacecraft_altitude','flag_amsubad','flag_avhrrbad','flag_cdlfrm','flag_cdltst','flag_daynit','flag_dustcld','flag_fgcheck','flag_iasibad','flag_initia','flag_itconv','flag_landsea','flag_mhsbad','flag_numit','flag_nwpbad','flag_physcheck','flag_retcheck','flag_satman','flag_sunglnt','flag_thicir','nerr_values','error_data_index','ozone_error','surface_z','co_qflag','co_bdiv','co_npca','co_nfitlayers','co_nbr_values','co_cp_air','co_cp_co_a','co_x_co','co_h_eigenvalues','co_h_eigenvectors','hno3_qflag','hno3_bdiv','hno3_npca','hno3_nfitlayers','hno3_nbr_values','hno3_cp_air','hno3_cp_hno3_a','hno3_x_hno3','hno3_h_eigenvalues','hno3_h_eigenvectors','o3_qflag','o3_bdiv','o3_npca','o3_nfitlayers','o3_nbr_values','o3_cp_air','o3_cp_o3_a','o3_x_o3','o3_h_eigenvalues','o3_h_eigenvectors','so2_qflag','so2_col_at_altitudes','so2_altitudes','so2_col','so2_bt_difference','pressure_levels_ozone','surface_emissivity_wavelengths'))
#    water_vapour_error = ds.water_vapour_error
#    temperature_error = ds.temperature_error
#    print(water_vapour_error)
#    print(temperature_error)
    
    ds['record_start_time'] = ds.record_start_time.astype('timedelta64[s]') + np.datetime64('2000-01-01')
    ds['record_stop_time'] = ds.record_stop_time.astype('timedelta64[s]') + np.datetime64('2000-01-01')

    ds = ds.isel(cloud_formations=0)


    ds['cloud_top_pressure'] = ds.cloud_top_pressure.where(ds.cloud_top_pressure < 120000.)
    ds['integrated_water_vapor'] = ds.integrated_water_vapor.where(ds.integrated_water_vapor < 350.)

    ds['atmospheric_temperature'] = ds.atmospheric_temperature.where(ds.atmospheric_temperature < 350.)
    ds['fg_atmospheric_temperature'] = ds.fg_atmospheric_temperature.where(ds.fg_atmospheric_temperature < 350.)
    ds['fg_atmospheric_temperature'] = ds.fg_atmospheric_temperature.where(ds.fg_qi_atmospheric_temperature >= 0.9)
    ds['atmospheric_temperature'] = ds.atmospheric_temperature.where(~ds.atmospheric_temperature.isnull(), ds.fg_atmospheric_temperature)
    ds['atmospheric_temperature'] = ds.atmospheric_temperature.where(ds.flag_cldnes <= 2)
    ds = ds.drop_vars(('fg_atmospheric_temperature', 'fg_qi_atmospheric_temperature'))

    ds['atmospheric_water_vapor'] = ds.atmospheric_water_vapor.where(ds.atmospheric_water_vapor < 350.)
    ds['fg_atmospheric_water_vapor'] = ds.fg_atmospheric_water_vapor.where(ds.fg_atmospheric_water_vapor < 350.)
    ds['fg_atmospheric_water_vapor'] = ds.fg_atmospheric_water_vapor.where(ds.fg_qi_atmospheric_water_vapour >= 0.9)
    ds['atmospheric_water_vapor'] = ds.atmospheric_water_vapor.where(~ds.atmospheric_water_vapor.isnull(), ds.fg_atmospheric_water_vapor)
    ds['atmospheric_water_vapor'] = ds.atmospheric_water_vapor.where(ds.flag_cldnes <= 2)
    ds = ds.drop_vars(('fg_atmospheric_water_vapor', 'fg_qi_atmospheric_water_vapour'))


    ds['nlq'] = ds.pressure_levels_humidity / 1e2
    ds['nlt'] = ds.pressure_levels_temp / 1e2
    ds = ds.drop_vars(('pressure_levels_humidity', 'pressure_levels_temp'))
    ds = ds.rename(nlq='pressure_levels_humidity', nlt='pressure_levels_temp')

    ds['atmospheric_temperature'] = ds.atmospheric_temperature.rename(pressure_levels_temp='pressure_levels_humidity')
    ds = ds.drop_dims('pressure_levels_temp')
    ds = ds.rename(pressure_levels_humidity='pressure_levels')


#    hcc = np.where(ds.cloud_top_pressure <= 44000., ds.fractional_cloud_cover, 0.) / 100.
#    hcc[np.isnan(ds.fractional_cloud_cover)] = np.nan
#    ds['hcc'] = xr.DataArray(hcc, dims=('along_track', 'across_track'))
#    ds = ds.drop_vars(('cloud_top_temperature', 'cloud_top_pressure', 'fractional_cloud_cover'))

    # REARRANGE DATA TO HAVE A REGULAR GRID

    sort_axis = ds.lon.get_axis_num('across_track')

    mask = (np.arange(ds.across_track.size) % 4 == 0) | (np.arange(ds.across_track.size) % 4 == 3)
    tmp1 = ds.isel(across_track=mask)
    tmp2 = ds.isel(across_track=~mask)
    sort_inds_1 = tmp1.lon.argsort(axis=sort_axis)
    sort_inds_2 = tmp2.lon.argsort(axis=sort_axis)
    tmp1 = tmp1.isel({'across_track': sort_inds_1})
    tmp2 = tmp2.isel({'across_track': sort_inds_2})

    along_track = np.arange(tmp1.along_track.size)
    across_track = np.arange(tmp1.across_track.size)
    along_track_1 = along_track * 2
    along_track_2 = along_track * 2 + 1

    tmp1 = tmp1.assign_coords(along_track=along_track_1, across_track=across_track)
    tmp2 = tmp2.assign_coords(along_track=along_track_2, across_track=across_track)

    ds_new = tmp1.merge(tmp2)

    ds_new['relative_humidity'] = thermo.massMixRatioToRH(
            ds_new.atmospheric_water_vapor, ds_new.atmospheric_temperature,
            ds_new.pressure_levels * 100., phase='ice')

    return ds_new


if __name__ == '__main__':
    #from satpy import Scene
    #import glob
    import eccodes as ec
    
    filename = '/bdd/metopc/l2/iasi/2022/12/27/so2/W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPC+IASI_C_EUMR_20221227235957_21480_eps_o_so2_l2.bin'
    filename = '/bdd/metopc/l2/iasi/2022/12/27/twt/W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPC+IASI_C_EUMR_20221227235957_21480_eps_o_twt_l2.bin'
    filename = '/bdd/metopc/l2/iasi/2022/12/27/cox/W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPC+IASI_C_EUMR_20221227235957_21480_eps_o_cox_l2.bin'
    filename = '/bdd/metopc/l2/iasi/2022/12/27/clp/W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPC+IASI_C_EUMR_20221227235957_21480_eps_o_clp_l2.bin'
    #filename = '/bdd/metopc/l2/iasi/2022/12/27/trg/W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPC+IASI_C_EUMR_20221227235957_21480_eps_o_trg_l2.bin'
    #filename = '/bdd/metopc/l2/iasi/2022/12/27/pmap/W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPC+GOME_C_EUMR_20221227235957_21481_eps_o_pmap_l2.nc'
    #filename = '/bdd/metopc/l2/iasi/2022/12/27/nac/W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPC+IASI_C_EUMR_20221227235957_21480_eps_o_nac_l2.bin'
    #filename = '/bdd/metopc/l2/iasi/2022/12/27/err/W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPC+IASI_C_EUMR_20221227235957_21480_eps_o_err_l2.bin'
    #filename = '/bdd/metopc/l2/iasi/2022/12/27/ems/W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPC+IASI_C_EUMR_20221227235957_21480_eps_o_ems_l2.bin'
    
    
    fh = open(filename, "rb")
    bufr = ec.codes_bufr_new_from_file(fh)
    
    #if bufr is None:
    #    break
    iterid = ec.codes_keys_iterator_new(bufr)
    while True:
        ec.codes_keys_iterator_next(iterid)
        key = ec.codes_keys_iterator_get_name(iterid)
        print('\n', key)
        print(ec.codes_get_native_type(bufr, key))
        if key in ('expandedAbbreviations', 'expandedTypes', 'expandedNames', 'expandedUnits'):
            print(ec.codes_get_array(bufr, key))
        if key == '7777':
            break
    ec.codes_keys_iterator_delete(iterid)
    
    ec.codes_set(bufr, "unpack", 1)
    #print(ec.codes_get_array(bufr, 'sulphurDioxide'))
    iterid = ec.codes_keys_iterator_new(bufr)
    while True:
        ec.codes_keys_iterator_next(iterid)
        key = ec.codes_keys_iterator_get_name(iterid)
        print('\n', key)
        print(ec.codes_get_native_type(bufr, key))
        if key in ('mixingRatio','pressure'):
            print(ec.codes_get_array(bufr, key))
        if key == '7777':
            break
    ec.codes_keys_iterator_delete(iterid)
    
    
    #ec.codes_set(bufr, "unpack", 1)
    #year = ec.codes_get(bufr, "year")
    #month = ec.codes_get(bufr, "month")
    #day = ec.codes_get(bufr, "day")
    #hour = ec.codes_get(bufr, "hour")
    #minute = ec.codes_get(bufr, "minute")
    #second = ec.codes_get(bufr, "second")
    #ec.codes_release(bufr)
    #
    #ec.codes_get_message(bufr)
