import numpy as np
import xarray as xr
from pyhdf.SD import SD, SDC

def convert_time(time_in):

    time_converted = []

    for time_val in time_in:

        time_str = str(time_val)
        yr = '20' + time_str[0:2]
        mth = time_str[2:4]
        day = time_str[4:6]

        tot_sec = round(float(time_str[6:]) * 24. * 3600.)
        sec = str(int(tot_sec % 60.)).zfill(2)
        tot_minu = tot_sec // 60.
        minu = str(int(tot_minu % 60.)).zfill(2)
        hours = str(int(tot_minu // 60.)).zfill(2)

        time_converted.append(np.datetime64(f'{yr}-{mth}-{day}T{hours}:{minu}:{sec}'))

    time_converted = np.array(time_converted, dtype='datetime64[ns]')

    return time_converted


def convert_feature_flag(feature):

    # see https://www-calipso.larc.nasa.gov/resources/calipso_users_guide/data_desc/cal_lid_l2_vfm_v4-51_desc.php

    # Feature type
    # 0 = invalid (bad or missing data)
    # 1 = "clear air"
    # 2 = cloud
    # 3 = tropospheric aerosol
    # 4 = stratospheric aerosol
    # 5 = surface
    # 6 = subsurface
    # 7 = no signal (totally attenuated)
    feature_type = feature % 8
    feature = feature // 8

    # Feature type quality assessment
    # 0 = none
    # 1 = low
    # 2 = medium
    # 3 = high
    feature_type_qa = feature % 4
    feature = feature // 4

    # Water phase
    # 0 = unknown / not determined
    # 1 = ice
    # 2 = water
    # 3 = oriented ice crystals
    water_phase = feature % 4
    feature = feature // 4

    # Water phase quality assessment
    # 0 = none
    # 1 = low
    # 2 = medium
    # 3 = high
    water_phase_qa = feature % 4
    feature = feature // 4

    # Cloud subtype (NB. only valid for clouds, different types for aerosols/merged)
    # 0 = low overcast, transparent
    # 1 = low overcast, opaque
    # 2 = transition stratocumulus
    # 3 = low, broken cumulus
    # 4 = altocumulus (transparent)
    # 5 = altostratus (opaque)
    # 6 = cirrus (transparent)
    # 7 = deep convective (opaque)
    feature_subtype = feature % 8
    feature = feature // 8

    # Cloud / Troposheric Aerosol / Stratospheric Aerosol QA
    # 0 = not confident
    # 1 = confident
    feature_qa = feature % 2
    feature = feature // 2

    # Horizontal averaging, for layers
    # 0 = not applicable
    # 1 = 1/3 km
    # 2 = 1 km
    # 3 = 5 km
    # 4 = 20 km
    # 5 = 80 km
    # NB. this flag is different for profiles
    horiz_averaging = feature

    return feature_type, feature_type_qa, water_phase, \
           water_phase_qa, feature_subtype, feature_qa, horiz_averaging


def screen_iwc(iwc, iwc_uncer, ext_qcflag, feature):

    # see https://www-calipso.larc.nasa.gov/resources/calipso_users_guide/data_desc/cal_lid_l2_profile_v4-51_desc.php

    mask = (ext_qcflag == 0)  | (ext_qcflag == 1) | (ext_qcflag == 2) \
         | (ext_qcflag == 16) | (ext_qcflag == 18)
    feature_type, _, water_phase, _, _, _, _ = convert_feature_flag(feature)
    mask = mask & (feature_type == 2)
    mask = mask & ((water_phase == 1) | (water_phase == 3))
    
    iwc = iwc.where(mask).where(iwc >= 0.)
    iwc_uncer = iwc.where(mask).where(iwc_uncer >= 0.)

    return iwc, iwc_uncer


def get_keys(filename):
    hdf = SD(filename, SDC.READ)
    
    for key in hdf.datasets().keys():
        data = hdf.select(key)
        print(key, data[:].shape, data.attributes())

    return


def load_clay(filename):

    # see https://www-calipso.larc.nasa.gov/resources/calipso_users_guide/data_desc/cal_lid_l2_layer_v4-51_desc.php

    hdf = SD(filename, SDC.READ)
    
    time = hdf.select('Profile_UTC_Time')[:,1]
    lon = hdf.select('Longitude')[:,1]
    lat = hdf.select('Latitude')[:,1]
    cod = hdf.select('Column_Optical_Depth_Cloud_532')[:,0] # no unit
    cod_uncer = hdf.select('Column_Optical_Depth_Cloud_Uncertainty_532')[:,0] # no unit
    
    n_lays = hdf.select('Number_Layers_Found')[:,0]
    lay_top = hdf.select('Layer_Top_Altitude')[:] # km
    lay_base = hdf.select('Layer_Base_Altitude')[:] # km
    lay_cod = hdf.select('Feature_Optical_Depth_532')[:]
    lay_cod_uncer = hdf.select('Feature_Optical_Depth_Uncertainty_532')[:]
    lay_iwp = hdf.select('Ice_Water_Path')[:] # g/m2
    lay_iwp_uncer = hdf.select('Ice_Water_Path_Uncertainty')[:] # g/m2
    lay_feature = hdf.select('Feature_Classification_Flags')[:]
    
    dims = ('points',)
    time = xr.DataArray(convert_time(time), dims=dims)
    lon = xr.DataArray(lon, dims=dims)
    lat = xr.DataArray(lat, dims=dims)
    lon = lon.where(lon != -9999.)
    lat = lat.where(lat != -9999.)
    
    coords = dict(time=time, lon=lon, lat=lat)
    cod = xr.DataArray(cod, dims=dims, coords=coords, name='cod')
    cod_uncer = xr.DataArray(cod_uncer, dims=dims, coords=coords, name='cod_uncer')
    cod = cod.where(cod >= 0.)
    cod_uncer = cod_uncer.where(cod_uncer >= 0.)
    n_lays = xr.DataArray(n_lays, dims=dims, coords=coords, name='n_lays')
    
    dims = ('points', 'layer')
    lay_top = xr.DataArray(lay_top, dims=dims, coords=coords, name='lay_top')
    lay_base = xr.DataArray(lay_base, dims=dims, coords=coords, name='lay_base')
    lay_cod = xr.DataArray(lay_cod, dims=dims, coords=coords, name='lay_cod')
    lay_cod_uncer = xr.DataArray(lay_cod_uncer, dims=dims, coords=coords, name='lay_cod_uncer')
    lay_iwp = xr.DataArray(lay_iwp, dims=dims, coords=coords, name='lay_iwp')
    lay_iwp_uncer = xr.DataArray(lay_iwp_uncer, dims=dims, coords=coords, name='lay_iwp_uncer')
    lay_feature = xr.DataArray(lay_feature, dims=dims, coords=coords, name='lay_feature')
    lay_top = lay_top.where(lay_top != -9999.)
    lay_base = lay_base.where(lay_base != -9999.)
    lay_cod = lay_cod.where(lay_cod >= 0.)
    lay_cod_uncer = lay_cod_uncer.where(lay_cod_uncer >= 0.)
    lay_iwp = lay_iwp.where(lay_iwp >= 0.)
    lay_iwp_uncer = lay_iwp_uncer.where(lay_iwp_uncer >= 0.)
    
    ds = xr.merge((cod, cod_uncer, n_lays, lay_top, lay_base, lay_cod, lay_cod_uncer,
                   lay_iwp, lay_iwp_uncer, lay_feature))

    return ds


def load_cpro(filename):

    # see https://www-calipso.larc.nasa.gov/resources/calipso_users_guide/data_desc/cal_lid_l2_profile_v4-51_desc.php

    hdf = SD(filename, SDC.READ)
    
    time = hdf.select('Profile_UTC_Time')[:,1]
    lon = hdf.select('Longitude')[:,1]
    lat = hdf.select('Latitude')[:,1]
    cod = hdf.select('Column_Optical_Depth_Cloud_532')[:,0] # no unit
    cod_uncer = hdf.select('Column_Optical_Depth_Cloud_Uncertainty_532')[:,0] # no unit

    cc = hdf.select('Cloud_Layer_Fraction')[:]
    iwc = hdf.select('Ice_Water_Content_Profile')[:] # g/m3
    iwc_uncer = hdf.select('Ice_Water_Content_Profile_Uncertainty')[:] # g/m3
    ext_qcflag = hdf.select('Extinction_QC_Flag_532')[:,:,0]
    feature = hdf.select('Atmospheric_Volume_Description')[:,:,0]
    print(ext_qcflag.shape, feature.shape)

    alt = np.concatenate((np.arange(30000., 20060., -180.),
                          np.arange(20060., -500., -60.)))
#    alt = np.arange(30000., -500., -60.)


    dims = ('points',)
    time = xr.DataArray(convert_time(time), dims=dims)
    lon = xr.DataArray(lon, dims=dims)
    lat = xr.DataArray(lat, dims=dims)
    alt = xr.DataArray(data=alt, dims=('alt',))
    lon = lon.where(lon != -9999.)
    lat = lat.where(lat != -9999.)

    coords = dict(time=time, lon=lon, lat=lat)
    cod = xr.DataArray(cod, dims=dims, coords=coords, name='cod')
    cod_uncer = xr.DataArray(cod_uncer, dims=dims, coords=coords, name='cod_uncer')
    cod = cod.where(cod >= 0.)
    cod_uncer = cod_uncer.where(cod_uncer >= 0.)

    dims = ('points', 'alt')
    coords = dict(time=time, lon=lon, lat=lat, alt=alt)
    cc = xr.DataArray(cc, dims=dims, coords=coords, name='cc')
    iwc = xr.DataArray(iwc, dims=dims, coords=coords, name='iwc')
    iwc_uncer = xr.DataArray(iwc_uncer, dims=dims, coords=coords, name='iwc_uncer')
    ext_qcflag = xr.DataArray(ext_qcflag, dims=dims, coords=coords)
    feature = xr.DataArray(feature, dims=dims, coords=coords)

    iwc, iwc_uncer = screen_iwc(iwc, iwc_uncer, ext_qcflag, feature)

    ds = xr.merge((cod, cod_uncer, cc, iwc, iwc_uncer))

    return ds
