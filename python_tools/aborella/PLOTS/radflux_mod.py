import xarray as xr
import pandas as pd
import numpy as np

def load_data(filename):

    df = pd.read_csv(filename, sep=r'\s+', comment='#', header=None)
    times = df[0].values
    lw_dn = df[1].values
    lw_up = df[2].values
    sw_dn = df[3].values
    sw_up = df[4].values
    alb = df[5].values
    lw_net = df[6].values
    sw_net = df[7].values
    net = df[8].values

    times = [np.datetime64(time[:-1]) for time in times]

    ds = xr.Dataset(dict(
            LW_down=(('time',), lw_dn,  dict(units='W/m2', long_name='Infrared downwelling flux')),
            LW_up  =(('time',), lw_up,  dict(units='W/m2', long_name='Infrared upwelling flux')),
            SW_down=(('time',), sw_dn,  dict(units='W/m2', long_name='Solar downwelling flux')),
            SW_up  =(('time',), sw_up,  dict(units='W/m2', long_name='Solar upwelling flux')),
            albedo =(('time',), alb,    dict(units='-',    long_name='Albedo')),
            LW_net =(('time',), lw_net, dict(units='W/m2', long_name='Infrared budget')),
            SW_net =(('time',), sw_net, dict(units='W/m2', long_name='Solar budget')),
            net    =(('time',), net,    dict(units='W/m2', long_name='Radiative budget')),
            time   =times,
            ))

    return ds
