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


def get_keys(filename):
    hdf = SD(filename, SDC.READ)
    
    str = ''
    for key in hdf.datasets().keys():
        data = hdf.select(key)
        str += key + ' '
#        print(key)#, data[:].shape, data.attributes())
    print(str)

    return


def load_data(filename):

    # see https://atmosphere-imager.gsfc.nasa.gov/products/joint-atm/format-content

    hdf = SD(filename, SDC.READ)

    lon = hdf.select('Longitude')[:]
    lat = hdf.select('Latitude')[:]

    wvp_NIR = hdf.select('Precipitable_Water_Near_Infrared_ClearSky')[:] # cm
    wvp_IR = hdf.select('Precipitable_Water_Infrared_ClearSky')[:] # cm
    ctp = hdf.select('Cloud_Top_Pressure')[:] # hPa
    cf = hdf.select('Cloud_Fraction')[:] # -
    # Phase key: [ 0 = Clear, 1 = Liquid Water Phase Cloud, 2 = Ice Phase Cloud, 3 = Mixed Phase Cloud, 6 = Undetermined Phase Cloud ]
    cphase = hdf.select('Cloud_Phase_Infrared')[:] # -
    cwp = hdf.select('Cloud_Water_Path')[:] # g/m2
    cwp_uncer = hdf.select('Cloud_Water_Path_Uncertainty')[:] # %
    cot = hdf.select('Cloud_Optical_Thickness')[:] # -
    cot_uncer = hdf.select('Cloud_Optical_Thickness_Uncertainty')[:] # %
    cer = hdf.select('Cloud_Effective_Radius')[:] # microns
    cer_uncer = hdf.select('Cloud_Effective_Radius_Uncertainty')[:] # %

    dims = ('along_swath', 'across_swath')
    lon = xr.DataArray(lon, dims=dims)
    lat = xr.DataArray(lat, dims=dims)
    lon = lon.where(lon != -999.)
    lat = lat.where(lat != -999.)

    coords = dict(lon=lon, lat=lat)
    wvp_NIR = xr.DataArray(wvp_NIR, dims=dims, coords=coords, name='wvp_NIR')
    wvp_IR = xr.DataArray(wvp_IR, dims=dims, coords=coords, name='wvp_IR')
    ctp = xr.DataArray(ctp, dims=dims, coords=coords, name='ctp')
    cf = xr.DataArray(cf, dims=dims, coords=coords, name='cf')
    cphase = xr.DataArray(cphase, dims=dims, coords=coords, name='cphase')
    cwp = xr.DataArray(cwp, dims=dims, coords=coords, name='cwp')
    cwp_uncer = xr.DataArray(cwp_uncer, dims=dims, coords=coords, name='cwp_uncer')
    cot = xr.DataArray(cot, dims=dims, coords=coords, name='cot')
    cot_uncer = xr.DataArray(cot_uncer, dims=dims, coords=coords, name='cot_uncer')
    cer = xr.DataArray(cer, dims=dims, coords=coords, name='cer')
    cer_uncer = xr.DataArray(cer_uncer, dims=dims, coords=coords, name='cer_uncer')

    wvp_NIR = wvp_NIR.where(wvp_NIR != -9999.)
    wvp_IR = wvp_IR.where(wvp_IR != -9999.)
    ctp = ctp.where(ctp > 0.)
    cf = cf.where(cf != -9999.)
    cphase = cphase
    cwp = cwp.where(cwp != -9999.)
    cwp_uncer = cwp_uncer.where(cwp_uncer != -9999.)
    cot = cot.where(cot != -9999.)
    cot_uncer = cot_uncer.where(cot_uncer != -9999.)
    cer = cer.where(cer != -9999.)
    cer_uncer = cer_uncer.where(cer_uncer != -9999.)

    wvp_NIR = wvp_NIR / 100.
    wvp_IR = wvp_IR / 100.
    ctp = ctp / 10.
    cf = cf / 100.
    cphase = cphase
    cwp = cwp
    cwp_uncer = cwp_uncer / 100.
    cot = cot / 100.
    cot_uncer = cot_uncer / 100.
    cer = cer / 100.
    cer_uncer = cer_uncer / 100.

    
    ds = xr.merge((wvp_NIR, wvp_IR, ctp, cf, cphase, cwp, cwp_uncer, cot, cot_uncer,
                   cer, cer_uncer))
    ds = ds.where(ds.cf <= 1.)

    return ds

#filename = '/scratchu/aborella/OBS/MODIS/202212/MODATML2.A2022360.2215.061.2022362035441.hdf'
#ds = load_data(filename)
#print(ds)

#import matplotlib.pyplot as plt
#ds.wvp_IR.plot() ; plt.show()
