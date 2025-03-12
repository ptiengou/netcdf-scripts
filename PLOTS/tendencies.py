import numpy as np
import matplotlib.pyplot as plt

import datasets

time_bounds = (np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T12'))
lev = 300.
lon = 2.2
lat = 48.7
nums_issr = (1, 2, 4)

dict_vars = dict(
        u=('u', 'vitu', (-50., 51.), 'U component of wind', 'm/s'),
        v=('v', 'vitv', (-30., 31.), 'V component of wind', 'm/s'),
        cc=('cc', 'rneb', (0., 1.01), 'Cloud fraction', '-'),
        q=('q', 'ovap', (1.e-5, 1.7e-4), 'Specific humidity', 'kg/kg'),
        r=('r', 'rhi', (0., 150.), 'Relative humidity', '%'),
        t=('ta', 'temp', (200., 246.), 'Air temperature', 'K'),
        )


def plot_timeseries(varname, num_ctrl, num_issr):
    
    name_era5, name_lmdz, ylims, longname, unit = dict_vars[varname]
    fig, axs = plt.subplots(nrows=1, ncols=1, layout='constrained')
    ax = axs

    ax.set(xlim=time_bounds, xlabel='Time', ylabel=f'{longname} [{unit}]')


    var_era5 = datasets.get_era5_timeseries(name_era5, time_bounds, lev, lon, lat)
    var_ctrl = datasets.get_lmdz_timeseries('CTRL', num_ctrl, name_lmdz, time_bounds, lev, lon, lat)
    var_npar = datasets.get_lmdz_timeseries('ISSR', num_issr, name_lmdz, time_bounds, lev, lon, lat)

    ax.plot(var_era5.time, var_era5, label='ERA5')
    ax.plot(var_ctrl.time, var_ctrl, label=f'CTRL {num_ctrl}')
    ax.plot(var_npar.time, var_npar, label=f'ISSR {num_issr}')

    ax.legend()
    return


def plot_tendency_all_params(varname):

    fig, axs = plt.subplots(nrows=1, ncols=1, layout='constrained')
    ax = axs

    for num_issr in nums_issr:
        var = datasets.get_lmdz_timeseries('ISSR', num_issr, varname, time_bounds, lev, lon, lat)
        ax.plot(var.time, var, label=f'ISSR {num_issr}')

    ax.set(xlim=time_bounds, xlabel='Time', ylabel=f'{var.long_name} [var.units]')

    ax.legend()
    return


def plot_tendencies(*vardata):

    fig, axs = plt.subplots(nrows=1, ncols=1, layout='constrained')
    ax = axs

    for var in vardata:
        if isinstance(var, str):
            var = datasets.get_lmdz_timeseries('ISSR', num_issr, var, time_bounds, lev, lon, lat)
            var.attrs['kwargs'] = dict()
        ax.plot(var.time, var, label=var.name, **var.kwargs)

    ax.set(xlim=time_bounds, xlabel='Time')

    ax.legend()
    return

def lmdz_var(varname):
    var = datasets.get_lmdz_timeseries('ISSR', num_issr, varname, time_bounds, lev, lon, lat)
    var.attrs['kwargs'] = dict()
    return var

num_issr = 1

#plot_timeseries('r', 1, 1)
plot_timeseries('cc', 1, 1)
#plot_tendency_all_params('dcfcon')

rhi = lmdz_var('rhi')
ovap = lmdz_var('ovap')
ocond = lmdz_var('ocond')
qcld = lmdz_var('qcld')
qissr = lmdz_var('qissr')
cldfra = lmdz_var('rneb')
issrfra = lmdz_var('issrfra')
dcfcon = lmdz_var('dcfcon')
dcfdyn = lmdz_var('dcfdyn')
dcfmix = lmdz_var('dcfmix')
dcfsub = lmdz_var('dcfsub')
dqicon = lmdz_var('dqicon')
dqidyn = lmdz_var('dqsdyn')
dqisub = lmdz_var('dqisub')
dqiadj = lmdz_var('dqiadj')
dqvccon = lmdz_var('dqvccon')
dqvcsub = lmdz_var('dqvcsub')
dqvcmix = lmdz_var('dqvcmix')
dqvcadj = lmdz_var('dqvcadj')
drvcdyn = lmdz_var('drvcdyn')

rhi.attrs['kwargs'] = dict()
ovap.attrs['kwargs'] = dict()
ocond.attrs['kwargs'] = dict()
qcld.attrs['kwargs'] = dict()
qissr.attrs['kwargs'] = dict()
cldfra.attrs['kwargs'] = dict()
issrfra.attrs['kwargs'] = dict()
dcfcon.attrs['kwargs'] = dict(color='tab:blue')
dcfdyn.attrs['kwargs'] = dict(color='grey')
dcfmix.attrs['kwargs'] = dict(color='tab:red')
dcfsub.attrs['kwargs'] = dict(color='tab:green')
dqicon.attrs['kwargs'] = dict(color='tab:blue')
dqidyn.attrs['kwargs'] = dict(color='grey')
dqisub.attrs['kwargs'] = dict(color='tab:green')
dqiadj.attrs['kwargs'] = dict(color='tab:orange')
dqvccon.attrs['kwargs'] = dict(color='tab:blue')
dqvcsub.attrs['kwargs'] = dict(color='tab:green')
dqvcmix.attrs['kwargs'] = dict(color='tab:red')
dqvcadj.attrs['kwargs'] = dict(color='tab:orange')
drvcdyn.attrs['kwargs'] = dict()

rhcld = rhi * qcld / cldfra / ovap
rhcld.name = 'rhcld'
rhcld.attrs['kwargs'] = dict(color='tab:blue')
rhissr = rhi * qissr / issrfra / ovap
rhissr.name = 'rhissr'
rhissr.attrs['kwargs'] = dict(color='tab:orange')
dqvcdyn = drvcdyn * ovap
dqvcdyn.name = 'dqvcdyn'
dqvcdyn.attrs['kwargs'] = dict(color='grey')

#plot_tendencies(dcfcon, dcfdyn, dcfmix, dcfsub)
#plot_tendencies(dqicon, dqidyn, dqisub, dqiadj)
#plot_tendencies(dqvccon, dqvcmix, dqvcsub, dqvcadj, dqvcdyn)
#plot_tendencies(cldfra, issrfra)
#plot_tendencies(rhcld, rhissr)

plt.show()
