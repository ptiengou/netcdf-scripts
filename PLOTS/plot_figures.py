import sys

import numpy as np
import xarray as xr
import scipy.stats as sstats
import great_circle_calculator.great_circle_calculator as gcc
from metpy.calc import height_to_pressure_std, pressure_to_height_std
from metpy.units import units

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mplc
import matplotlib.cm as mplcm
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cycler import cycler
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.mpl.geoaxes import GeoAxes
import psyplot.project as psy


import datasets

sys.path.append('/home/aborella/UTIL')
import xr_utils
import thermodynamic as thermo
import seaborn_tmp as sns

map_proj = ccrs.PlateCarree()
map_proj._threshold /= 100.

simu_issr = 'REF'
#simu_issr = 'OVLP'

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mplc.LinearSegmentedColormap(name, cdict)

    return newcmap


def scheme_param():

    x_figsize, y_figsize = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(x_figsize, y_figsize))

    ax.set(xticks=[], yticks=[])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)

#    x = np.linspace(0.15, 1., 1000)
#    a = 4. / 3.
#    pdf = sstats.gamma.pdf(x, a, loc=0.15, scale=0.15) / 4.
    x = np.linspace(0., 1., 1000)
    pdf = sstats.norm.pdf(x, loc=0.4, scale=0.09) / 5

    ax.plot(x, pdf, lw=4., color='firebrick')
#    ax.plot([0.25, 0.25], [0., 0.90], color='grey', ls='--')
    ax.plot([0.3, 0.3], [0., 0.50], color='grey', ls='--')
    ax.arrow(0., 0., 0., 1., zorder=10, lw=3., head_length=0.015, head_width=0.015)
    ax.arrow(0., 0., 1., 0., zorder=10, lw=3., head_length=0.015, head_width=0.015)
#    ax.fill_between(x, pdf, where=(x < 0.25), alpha=0.3, color='grey')
    ax.fill_between(x, pdf, where=(x < 0.3), alpha=0.3, color='grey')

    plt.savefig('distrib_incloud.png')
#    plt.show()
    plt.close()


    x_figsize, y_figsize = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(x_figsize, y_figsize))

    ax.set(xticks=[], yticks=[])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)

    x = np.linspace(0.01, 1., 1000)
    a = 8.
    pdf = sstats.weibull_min.pdf(x, a, loc=-0.1, scale=0.8) / 4. + 0.002

    ax.plot(x, pdf, lw=4., color='firebrick')
    ax.plot([0.8, 0.8], [0., 0.55], color='grey', ls='--')
    ax.arrow(0., 0., 0., 1., zorder=10, lw=3., head_length=0.015, head_width=0.015)
    ax.arrow(0., 0., 1., 0., zorder=10, lw=3., head_length=0.015, head_width=0.015)
    ax.fill_between(x, pdf, where=(x > 0.8), alpha=0.3, color='grey')

    plt.savefig('distrib_clear.png')
#    plt.show()
    plt.close()
    return


def scheme_mixing():

    # Paramètres de l'ellipse
    center = (0.5, 0.5)
    width = 0.85
    height = 0.5
    angle = 0
    thickness = 0.07
    facecolor = 'orange'
    edgecolor = 'orangered'
    facecolor = sns.color_palette('deep')[9]
    edgecolor = sns.color_palette('dark')[9]

    # Création de la figure et de l'axe
    x_figsize, y_figsize = plt.rcParams['figure.figsize']
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(2.*x_figsize, y_figsize))

    ax = axs[0]
    # Création de l'ellipse extérieure
    outer_ellipse = mpatches.Ellipse(xy=center, width=width, height=height,
                                    angle=angle, fc=facecolor, ec=edgecolor, alpha=0.5,
                                    ls='--', lw=3.)
    # Création de l'ellipse intérieure (trou)
    inner_ellipse = mpatches.Ellipse(xy=center, width=width-2*thickness, height=height-2*thickness,
                                    angle=angle, fc=facecolor, ec=edgecolor,
                                    lw=3.)
    # Ajout des ellipses à l'axe
    ax.add_patch(outer_ellipse)
    ax.add_patch(inner_ellipse)

    ax.text(0.5, 0.45, r'$\alpha_c$ before mixing', ha='center', va='center')
    ax.annotate(r'New fraction $\Delta \alpha_c|_\mathrm{turb}$',
                (0.8, 0.37), (0.7, 0.15), ha='center', va='bottom',
                arrowprops=dict(fc='black', ec='none'))

    ax.plot([0.5, 0.5], [0.5-height/2., 0.5-height/2.+thickness], color='k')
    ax.text(0.52, 0.5-height/2.+thickness/2., r'$L_{turb}$', va='center', ha='left')
    ax.plot([0.5-width/2., 0.5-width/2.+thickness], [0.5, 0.5], color='k')
    ax.text(0.5-width/2.+thickness/4., 0.51, r'$L_{turb}$', va='bottom', ha='left')

    ax.plot([0.5, 0.5], [0.5, 0.5+height/2.-thickness], color='dimgrey')
    ax.text(0.49, 0.5+height/4.-thickness/2., r'$b$', va='center', ha='right', color='dimgrey')
    ax.plot([0.5, 0.5+width/2.-thickness], [0.5, 0.5], color='dimgrey')
    ax.text(0.5+width/4.-thickness/2., 0.51, r'$a$', va='bottom', ha='center', color='dimgrey')
    
    # Suppression des axes
    ax.axis('off')
    
    # Ajustement des limites pour centrer l'ellipse
    ax.set(xlim=(0., 1.), ylim=(0.2, 0.8))
    ax.set_title('(a) Horizontal turbulent diffusion mixing', loc='left')
    
    # Égalisation de l'échelle des axes pour que l'ellipse ne soit pas déformée
    ax.set_aspect('equal', adjustable='box')


    ax = axs[1]
    center_tmp = (center[0] + 0.05, center[1])
    outer_ellipse = mpatches.Ellipse(xy=center_tmp, width=width-0.1, height=height-0.1,
                                    angle=angle, fc=facecolor, ec=edgecolor, alpha=0.5,
                                    ls='--', lw=3.)
    center_tmp = (center[0] - 0.05, center[1])
    inner_ellipse = mpatches.Ellipse(xy=center_tmp, width=width-0.1, height=height-0.1,
                                    angle=angle, fc=facecolor, ec=edgecolor,
                                    lw=3.)
    # Ajout des ellipses à l'axe
    ax.add_patch(outer_ellipse)
    ax.add_patch(inner_ellipse)

    ax.text(0.5-0.05, 0.5, r'$\alpha_c$ before mixing', ha='center', va='center')
    ax.annotate(r'New fraction $\Delta \alpha_c|_\mathrm{shear}$',
                (0.8, 0.39), (0.7, 0.15), ha='center', va='bottom',
                arrowprops=dict(fc='black', ec='none'))

    ax.plot([0.5-0.05+(width-0.1)/2., 0.5+0.05+(width-0.1)/2.], [0.5, 0.5], color='k')
    ax.text(0.5-0.05+(width-0.1)/2., 0.51, r'$L_{shear}$', va='bottom', ha='left')
    
#    arr = mpatches.FancyArrowPatch((0.25, 0.6), (0.5, 0.6),
#                               arrowstyle='->,head_width=.15', mutation_scale=20)
#    ax.add_patch(arr)
#    ax.annotate('Wind ', (.5, .5), xycoords=arr, ha='center', va='bottom')
    
    # Suppression des axes
    ax.axis('off')
    
    # Ajustement des limites pour centrer l'ellipse
    ax.set(xlim=(0., 1.), ylim=(0.2, 0.8))
    ax.set_title('(b) Vertical wind shear-induced mixing', loc='left')
    
    # Égalisation de l'échelle des axes pour que l'ellipse ne soit pas déformée
    ax.set_aspect('equal', adjustable='box')

    
    # Affichage de la figure
    plt.savefig('scheme_mixing.png')
    plt.savefig('scheme_mixing.pdf')
    plt.close()


def case_study_MSG():

    lon_bounds = (-10, 10.)
    lat_bounds = (44., 52.)
    times = [np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T00'),
             np.datetime64('2022-12-27T03'), np.datetime64('2022-12-27T06'),
             np.datetime64('2022-12-27T09'), np.datetime64('2022-12-27T12')]

    x_figsize, y_figsize = plt.rcParams['figure.figsize']
    fig, axs = plt.subplots(nrows=2, ncols=3,
                        subplot_kw={'projection': map_proj},
                        figsize=(3.*x_figsize, 2.*0.7*y_figsize))

    for i, (ax, time) in enumerate(zip(axs.flatten(), times)):

        da = datasets.get_MSG(time, lon_bounds, lat_bounds)
        lons_interp = np.arange(lon_bounds[0], lon_bounds[1] + 0.1 / 2., 0.1)
        lats_interp = np.arange(lat_bounds[0], lat_bounds[1] + 0.1 / 2., 0.1)
        da = np.power(10, da)
        pco = ax.pcolormesh(da.longitude, da.latitude, da,
                            norm=mplc.LogNorm(da.min(), da.max()),
                            cmap='YlOrBr', transform=map_proj)

        # this is usually set by psyplot
        ax.set(xlim=lon_bounds, ylim=lat_bounds)
        ax.coastlines(resolution='110m', linewidth=0.5, color='darkgrey')
        ax.grid(linewidth=0.5, color='lightgrey')

        # Longitude labels
        ax.set_xticks(np.round(np.linspace(*lon_bounds, 5, endpoint=True), 1), crs=map_proj)
        if i // 3 == 1:
            lon_formatter = cticker.LongitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
        else:
            ax.set_xticklabels([])

        # Latitude labels
        ax.set_yticks(np.round(np.linspace(*lat_bounds, 3, endpoint=True), 1), crs=map_proj)
        if i % 3 == 0:
            lat_formatter = cticker.LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)
        else:
            ax.set_yticklabels([])

        labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        time_label = str(time)[:10] + ' ' + str(time)[11:] + ':00'
        ax.set_title(labels[i] + ' ' + time_label, loc='left')

        # show averaging zone
        ax.plot([-10., 10., 10., -5., -10.], [44., 44., 50.5, 50.5, 44.], color='black', lw=3.)

        # show SIRTA
        ax.plot([2.2], [48.7], '*g', markersize=20)

        ax.tick_params(direction='inout', top=True, right=True)

    fig.suptitle('High cloud optical depth from MSG OCA [-]')
    cbar = fig.colorbar(pco, ax=axs, pad=0.03, shrink=0.85)

    plt.savefig('case_study_MSG.png')
#    plt.show()
    plt.close()
    return


def grid():

    lon_bounds = (-10, 14.)
    lat_bounds = (41., 57.)

    x_figsize, _ = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           subplot_kw={'projection': map_proj},
                           figsize=(1.5*x_figsize, 1.5*0.5*x_figsize))

    # Longitude labels                                                              
    ax.set_xticks(np.linspace(-180., 180., 5, endpoint=True), crs=map_proj)
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

    # Latitude labels
    ax.set_yticks(np.linspace(-90., 90., 5, endpoint=True), crs=map_proj)
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)

    # Circle tuning
    p_center = (2., 49.)
    height = 2. * abs(gcc.point_given_start_and_bearing(p_center, 0., 2e6)[1] - 49.)
    width = 2. * abs(gcc.point_given_start_and_bearing(p_center, 90., 2e6)[0] - 2.)
    tuning_zone = mpatches.Ellipse(p_center, width, height,
            lw=2., color='red', fill=False, transform=map_proj, zorder=20)
    ax.add_patch(tuning_zone)

    ax.plot([lon_bounds[0], lon_bounds[1], lon_bounds[1], lon_bounds[0], lon_bounds[0]],
            [lat_bounds[0], lat_bounds[0], lat_bounds[1], lat_bounds[1], lat_bounds[0]],
            'k', lw=1., zorder=20)

    ds_lmdz = datasets.get_lmdz_map('CTRL', 1., np.datetime64('1980-01-01'))
    ds_lmdz.snow[:] = 0.

    ds_lmdz.psy.plot.mapplot(name='snow', datagrid=dict(color='grey', linewidth=0.1),
            cbar=None, ax=ax, cmap=mplc.ListedColormap(['white']))

    ax.set_title('ICOLMDZ grid and nudging zone')
    ax.set(xlabel='Longitude [°]', ylabel='Latitude [°]')

#    ax_in = fig.add_axes([0.1, 0.1, 0.3, 0.3], projection=map_proj)
    ax_in = inset_axes(ax, width='100%', height='100%',
                   bbox_to_anchor=(-150., -70., 120., 70.),
                   bbox_transform=ax.transData,
                   loc='lower left',
                   borderpad=0,
                   axes_class=GeoAxes,
                   axes_kwargs=dict(projection=ccrs.PlateCarree()))
    ds_lmdz.psy.plot.mapplot(name='snow', datagrid=dict(color='grey', linewidth=0.1),
            cbar=None, ax=ax_in, cmap=mplc.ListedColormap(['white']),
            map_extent=[*lon_bounds, *lat_bounds],
            lsm=dict(res='50m', linewidth=0.5, coast='dimgrey'))

    ax.plot([-142., lon_bounds[0]], [0., lat_bounds[1]], 'k', lw=1., zorder=20)
    ax.plot([-37.5, lon_bounds[1]], [-70., lat_bounds[0]], 'k', lw=1., zorder=20)

    # Longitude labels                                                              
    ax_in.set_xticks(np.linspace(*lon_bounds, 3, endpoint=True), crs=map_proj)
    lon_formatter = cticker.LongitudeFormatter()
    ax_in.xaxis.set_major_formatter(lon_formatter)

    # Latitude labels
    ax_in.set_yticks(np.linspace(*lat_bounds, 3, endpoint=True), crs=map_proj)
    lat_formatter = cticker.LatitudeFormatter()
    ax_in.yaxis.set_major_formatter(lat_formatter)

    ax.tick_params(direction='inout', top=True, right=True)
    ax_in.tick_params(direction='inout', top=True, right=True)
    
    fontsize = plt.rcParams['axes.labelsize'] * 0.75
    for item in ax_in.get_xticklabels() + ax_in.get_yticklabels():
        item.set_fontsize(fontsize)
    

    plt.savefig('grid.png', dpi=150.)
#    plt.show()
    plt.close()
    return


def timeseries_cldh():
    print('go timeseries_cldh')

    msg_dict = dict(label='Obs.', color='grey')
    era5_dict = dict(label='ERA5', color='black', ls='-.')
    ctrl_dict = dict(label='CTRL', color=sns.color_palette()[0])
    issr_dict = dict(label='ISSR', color=sns.color_palette()[1])

    # possible levs: 400, 375, 350, 325, 300, 280, 250, 235, 215, 195
    bounds = dict(time=(np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T15')),
                  lev=(350., 225.), lon=(-11.,11.), lat=(43.,51.))

    varnames_obs = varnames_era5 = 'hcc'
    vartype = varnames_ctrl = varnames_issr = 'cldh'

#    da_obs  = datasets.get_MSG_average(varnames_obs, **bounds)
#    da_obs.to_netcdf('cldh_MSG.nc')
    da_obs  = xr.open_dataset('cldh_MSG.nc').Upper_Layer_Cloud_Top_Pressure
    da_era5 = datasets.get_era5_average(varnames_era5, **bounds)
    da_ctrl = datasets.get_lmdz_average(vartype, varnames_ctrl, 'CTRL', **bounds)
    da_issr = datasets.get_lmdz_average(vartype, varnames_issr, simu_issr, **bounds)


    x_figsize, y_figsize = plt.rcParams['figure.figsize']
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           figsize=(x_figsize, y_figsize))
    ax.plot(da_obs.time, da_obs, **msg_dict)
    ax.plot(da_era5.time, da_era5, **era5_dict)
    ax.plot(da_ctrl.time, da_ctrl, **ctrl_dict)
    ax.plot(da_issr.time, da_issr, **issr_dict)

    xticks = np.arange(np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T14'), np.timedelta64(3, 'h'))
    xticklabels = [r'$\ \ \ \ $21h', '00h', '03h', '06h', '09h', '12h']
    ax.set(xticks=xticks, xticklabels=xticklabels)
    ax.set(ylabel='High cloud cover [-]')
    ax.set(xlabel='Time UTC', xlim=(xticks[0], xticks[-1]))
#    ax.set_xticks(xticks, xticklabels, rotation=15, ha='right')
    ax.legend(fontsize=plt.rcParams['font.size'] * 0.66)
    ax.set_ylim(0., 0.45)

    ax.axvline(x=np.datetime64('2022-12-27T00'), lw=1., color='dimgrey')
    ax.axvline(x=np.datetime64('2022-12-27T09'), lw=1., color='dimgrey')
#    ax.text(x=np.datetime64('2022-12-27T01'), y=0., s='Start of the event',
#            ha='left', va='bottom')
#    ax.text(x=np.datetime64('2022-12-27T10'), y=0., s='End of the event',
#            ha='right', va='bottom')

    ax.tick_params(direction='in', top=True, right=True)

    plt.savefig('timeseries_cldh.png')
#    plt.show()
    plt.close()
    return


def map_models():
    print('go map_models')

    level_snapshot = 350. # hPa
#    level_snapshot = 280. # hPa
    time_snapshot = np.datetime64('2022-12-27T04')
    lon_bounds = (-25., 29.)
    lat_bounds = (31., 67.)

    data = (level_snapshot, time_snapshot)
    ds_era5_hcc = datasets.get_era5_map('hcc', *data)
    ds_era5_rhi = datasets.get_era5_map('r', *data)
#    ds_era5_rhi = datasets.get_ml_era5_ciwc_map(*data)
    ds_ctrl = datasets.get_lmdz_map('CTRL', *data)
    ds_issr = datasets.get_lmdz_map(simu_issr, *data)

    da_msg = datasets.get_MSG(time_snapshot, lon_bounds, lat_bounds)
    da_msg = xr.where(np.isnan(da_msg), 0., 1.)
    
    step = 0.75
    lon_bins = np.arange(lon_bounds[0]-step, lon_bounds[1]+step, step)
    lat_bins = np.arange(lat_bounds[0]-step, lat_bounds[1]+step, step)
    lon_grouper = xr.groupers.BinGrouper(lon_bins, labels=lon_bins[1:])
    lat_grouper = xr.groupers.BinGrouper(lat_bins, labels=lat_bins[1:])
    da_msg = da_msg.groupby(longitude=lon_grouper, latitude=lat_grouper).mean()
    da_msg = da_msg.rename(longitude_bins='lon', latitude_bins='lat').T


    x_figsize, _ = plt.rcParams['figure.figsize']
    fig, axs = plt.subplots(nrows=2, ncols=3, subplot_kw={'projection': map_proj},
                            figsize=(3.*x_figsize, 1.8*x_figsize))

    bounds = np.arange(0., 1.01, 0.1)
    cmap = 'Oranges'
    for ax, ds, name in zip(axs[0],
            (ds_era5_hcc, ds_ctrl, ds_issr),
            ('hcc', 'cldh', 'cldh')):

        ds.psy.plot.mapplot(name=name, bounds=bounds, cbar=None, ax=ax, cmap=cmap,
                            map_extent=[*lon_bounds, *lat_bounds])

        cs = ax.contour(da_msg.lon, da_msg.lat, da_msg,
                        levels=[0.3, 0.8], colors='black', transform=map_proj,
                        linewidths=plt.rcParams['lines.linewidth'] * 0.75,
                        linestyles=['dashed', 'solid'])
        ax.clabel(cs, fontsize=plt.rcParams['font.size'] * 0.5)

        ax.tick_params(direction='inout', top=True, right=True)

    axs[0,0].plot([], [], color='black', label='Obs.')
    axs[0,0].legend(loc='lower right').set_zorder(30)

    norm = mplc.Normalize(bounds.min(), bounds.max())
    cmap = mpl.colormaps[cmap].resampled(bounds.size - 1)
    cbar = fig.colorbar(mplcm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axs[0], orientation='horizontal', fraction=0.1, aspect=50)



    bounds = np.arange(0., 145., 10.)
    cmap = shiftedColorMap(mplcm.RdBu, midpoint=100. / 145.)
#    bounds = np.linspace(0., 1e-6, 10, endpoint=True)
#    cmap = 'plasma'
    for ax, ds, name in zip(axs[1],
            (ds_era5_rhi, ds_ctrl, ds_issr),
            ('r', 'rhi', 'rhi')):
#            ('ciwc', 'ocond', 'ocond')):

        ds.psy.plot.mapplot(name=name, bounds=bounds, cbar=None, ax=ax, cmap=cmap,
                            map_extent=[*lon_bounds, *lat_bounds])

        ds, _ = xr_utils.relabelise(ds)
        if name == 'r':
#        if name == 'ciwc':
            contour_fun = ax.contour
            name = 'hcc'
            ds, _ = xr_utils.relabelise(ds_era5_hcc)
        else:
            contour_fun = ax.tricontour
            name = 'cldh'
        cs = contour_fun(ds.lon, ds.lat, ds[name],
                         levels=[0.3, 0.8], colors='black', transform=map_proj,
                         linewidths=plt.rcParams['lines.linewidth'] * 0.75,
                         linestyles=['dashed', 'solid'])
        ax.clabel(cs, fontsize=plt.rcParams['font.size'] * 0.5)

        ax.tick_params(direction='inout', top=True, right=True)

    norm = mplc.Normalize(bounds.min(), bounds.max())
#    cmap = mpl.colormaps[cmap].resampled(bounds.size - 1)
    cmap = cmap.resampled(bounds.size - 1)
    cbar = fig.colorbar(mplcm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axs[1], orientation='horizontal', fraction=0.1, aspect=50)


    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    for ax in axs.flatten():
        # Longitude labels
        ax.set_xticks(np.arange(lon_bounds[0], lon_bounds[1]+0.1, 13.5), crs=map_proj)
        ax.xaxis.set_major_formatter(lon_formatter)

        # Latitude labels
        ax.set_yticks(np.arange(lat_bounds[0], lat_bounds[1]+0.1, 9), crs=map_proj)
        ax.yaxis.set_major_formatter(lat_formatter)

        # Circle tuning
        p_center = (2., 49.)
        height = 2. * abs(gcc.point_given_start_and_bearing(p_center, 0., 2e6)[1] - 49.)
        width = 2. * abs(gcc.point_given_start_and_bearing(p_center, 90., 2e6)[0] - 2.)
        tuning_zone = mpatches.Ellipse(p_center, width, height,
                lw=2., color='red', fill=False, transform=map_proj, zorder=20)
        ax.add_patch(tuning_zone)

    for ax in axs[0].flatten():
        ax.tick_params(labelbottom=False)

    for ax in axs[:,1:].flatten():
        ax.tick_params(labelleft=False)

    axs[0,0].set_title('(a) ERA5 high cloud cover [-]', loc='left')
    axs[0,1].set_title('(b) LMDZ - CTRL high cloud cover [-]', loc='left')
    axs[0,2].set_title('(c) LMDZ - ISSR high cloud cover [-]', loc='left')
    axs[1,0].set_title(r'(d) ERA5 RH$_{\mathrm{ice}}$ at 350 hPa [%]', loc='left')
    axs[1,1].set_title(r'(e) LMDZ - CTRL RH$_{\mathrm{ice}}$ at 350 hPa [%]', loc='left')
    axs[1,2].set_title(r'(f) LMDZ - ISSR RH$_{\mathrm{ice}}$ at 350 hPa [%]', loc='left')


    plt.savefig('map_models.png')
#    plt.show()
    plt.close()
    return


def profile_basta():
    print('go profile_basta')

    time_beg = np.datetime64('2022-12-26T21')
    time_end = np.datetime64('2022-12-27T12')
    lon_sirta, lat_sirta = 2.2, 48.7
    alt_bounds = (0., 12000.)

    da_ref, _, _ = datasets.get_basta_sirta((time_beg, time_end))

    times = np.arange(time_beg.astype('datetime64[m]'), time_end.astype('datetime64[m]')+1, 15)
    data = (times, None, lon_sirta, lat_sirta)

    da_ctrl, _ = datasets.get_lmdz_profile('CTRL', 'dbze94', *data, cosp=True)
    da_issr, _ = datasets.get_lmdz_profile(simu_issr, 'dbze94', *data, cosp=True)
    da_ctrl = da_ctrl.rename(height_mlev='lev')
    da_issr = da_issr.rename(height_mlev='lev')
    da_ctrl = np.log10(np.power(10., da_ctrl).mean('column'))
    da_ctrl = da_ctrl.where(np.isfinite(da_ctrl), -60.)
    da_issr = np.log10(np.power(10., da_issr).mean('column'))
    da_issr = da_issr.where(np.isfinite(da_issr), -60.)

    da_ctrl = da_ctrl.swap_dims(points='time').sortby('lev', ascending=True)
    da_issr = da_issr.swap_dims(points='time').sortby('lev', ascending=True)

    time_bins = np.concatenate(([np.datetime64('2022-12-26T00')], da_ctrl.time.values))
    lev_bins = np.concatenate(([0.], da_ctrl.lev.values))

    da_ref_coarse = np.power(10., da_ref)
    da_ref_coarse = da_ref_coarse.groupby_bins('time', time_bins, labels=da_ctrl.time.values).mean()
    da_ref_coarse = da_ref_coarse.groupby_bins('lev', lev_bins, labels=da_ctrl.lev.values).mean()
    da_ref_coarse = np.log10(da_ref_coarse)
    da_ref_coarse = da_ref_coarse.where(np.isfinite(da_ref_coarse), -60.)
    da_ref_coarse = da_ref_coarse.rename(time_bins='time', lev_bins='lev')


    x_figsize, y_figsize = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(2*x_figsize, 2.*y_figsize))
    gs = gridspec.GridSpec(2, 2, figure=fig)#, width_ratios=[0.1, 1, 1, 0.1], height_ratios=[1, 1])
#    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(2*x_figsize, 2.*y_figsize))
    axs = [[fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])],
           [fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])]]
    axs = np.array(axs)

    for ax in axs.flatten():
        ax.set_xlim(time_beg, time_end)
        ax.set_ylim(*alt_bounds)

        # Time labels
        ax.set_xticks(np.arange(time_beg, time_end+1, 3))
        ax.set_xticklabels([r'$\ \ \ \ $21h', '00h', '03h', '06h', '09h', '12h'])

        # Altitude labels
        alt = np.arange(alt_bounds[0], alt_bounds[1] + 1., 3000.)
        ax.set_yticks(alt)
        ax.set_yticklabels((alt / 1000.).astype(int))

    for ax in axs[0].flatten():
        ax.tick_params(labelbottom=False)

    for ax in axs[:,0].flatten():
        # Pressure labels
        secax = ax.secondary_yaxis(1.0)
        pres = [200, 300, 500, 700, 1000]
        equiv_height = pressure_to_height_std(pres * units.hPa).to('m').magnitude
        secax.set_yticks(equiv_height)
        secax.set_yticklabels(pres)
        secax.tick_params(labelright=False)

    for ax in axs[:,1].flatten():
        ax.tick_params(labelleft=False)

        # Pressure labels
        secax = ax.secondary_yaxis(1.0)
        pres = [200, 300, 500, 700, 1000]
        equiv_height = pressure_to_height_std(pres * units.hPa).to('m').magnitude
        secax.set_yticks(equiv_height)
        secax.set_yticklabels(pres)


    levels = np.arange(-60., 0.01, 5.)
    cmap = 'plasma'
    for ax, da in zip(axs.flatten(), (da_ref, da_ref_coarse, da_ctrl, da_issr)):
        cf = ax.contourf(da.time, da.lev, da.transpose('lev', 'time'), levels, cmap=cmap)

    da_issr, geoph_issr = datasets.get_lmdz_profile(simu_issr, 'temp', *data)
    da_issr = da_issr.swap_dims(points='time').sortby('lev', ascending=False)
    da_issr = da_issr - 273.15
    time = da_issr.time.expand_dims(lev=da_issr.lev)
    cs = axs[1,1].contour(time.T, geoph_issr, da_issr, levels=[-38.],
                          colors=['white'], linewidths=2.)
    axs[1,1].clabel(cs, fontsize=plt.rcParams['font.size'] * 0.5)


    axs[0,0].set_title('(a) BASTA-SIRTA', loc='left')
    axs[0,1].set_title('(b) Regridded BASTA-SIRTA', loc='left')
    axs[1,0].set_title('(c) LMDZ - CTRL with COSP', loc='left')
    axs[1,1].set_title('(d) LMDZ - ISSR with COSP', loc='left')

#    cbar = fig.colorbar(cf, ax=axs, label='Radar reflectivity factor [dBZ]')
    cbar = fig.colorbar(cf, ax=axs, orientation='horizontal', label='Radar reflectivity factor [dBZ]')

    ax = fig.add_subplot(gs[:,:])
    ax.set(xticks=[10], xlim=(9,15), yticks=[10], ylim=(9,15))
    [ax.spines[side].set_visible(False) for side in ('left', 'top', 'right', 'bottom')]
    ax.patch.set_visible(False)
    ax.tick_params(axis='both', colors=[0,0,0,0])
    ax.set_xlabel('Hour of the day on the 27 December 2022')
    ax.set_ylabel('Altitude [km]')

    ax = fig.add_subplot(gs[:,:])
    ax.set(yticks=[1000], ylim=(1500,200))
    [ax.spines[side].set_visible(False) for side in ('left', 'top', 'right', 'bottom')]
    ax.patch.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')
    ax.tick_params(axis='both', colors=[0,0,0,0])
    ax.set_ylabel('Equivalent standard pressure [hPa]')

    plt.savefig('profile_basta.png')
#    plt.show()
    plt.close()
    return


def iasi_running_mean(da):

    alt = pressure_to_height_std(da.lev.values * units.hPa).to('m').magnitude
    da_ave = da.load().copy()

    for i in range(len(da)):
        mask = abs(alt - alt[i]) < 1000.
        da_ave[i] = da[mask].mean()

    return da_ave, alt


def profile_rhi():
    print('go profile_rhi')

    iasi_dict = dict(label='Obs.', color='grey')
    rs_dict = dict(label='Obs.', color='grey')
    era5_dict = dict(label='ERA5', color='black', ls='-.')
    ctrl_dict = dict(label='CTRL', color=sns.color_palette()[0])
    issr_dict = dict(label='ISSR', color=sns.color_palette()[1])

    
    alt_bounds = (6000., 12000.)
    alt_ticks = [6000., 8000., 10000., 12000.]
    alt_ticklabels = [6, 8, 10, 12]
    lev_bounds = height_to_pressure_std(alt_bounds * units.m).to('hPa').magnitude
    lev_ticks = [450., 400., 325., 250., 200.]
    lev_ticks_from_alt = pressure_to_height_std(lev_ticks * units.hPa).to('m').magnitude
    lev_ticklabels = np.array(lev_ticks, dtype=int)


    x_figsize, y_figsize = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(2*x_figsize, 2.*y_figsize))
    gs = gridspec.GridSpec(2, 3, figure=fig)


    IASI_ID = 0
    ax = fig.add_subplot(gs[0,0])

    ds_iasi = datasets.get_IASI_profile(IASI_ID)
    da_iasi = ds_iasi['relative_humidity']
    data = (ds_iasi.time.values, None, ds_iasi.lon.values, ds_iasi.lat.values)

#    da_era5, _ = datasets.get_era5_profile('r', *data)
    da_era5, _ = datasets.get_ml_era5_r_profile(*data)
    da_ctrl, _ = datasets.get_lmdz_profile('CTRL', 'rhi', *data)
    da_issr, _ = datasets.get_lmdz_profile(simu_issr, 'rhi', *data)

    ax.plot(da_iasi, pressure_to_height_std(da_iasi.pressure_levels.values * units.hPa).to('m').magnitude, **iasi_dict)
    ax.plot(*iasi_running_mean(da_era5), **era5_dict)
    ax.plot(*iasi_running_mean(da_ctrl), **ctrl_dict)
    ax.plot(*iasi_running_mean(da_issr), **issr_dict)

    ax.set_title('(a) IASI at 20h31 UTC', loc='left')

    ax.set(ylim=alt_bounds, yticks=alt_ticks, yticklabels=alt_ticklabels)
    ax.set(xlim=(0., 120.))
    ax.tick_params(direction='in', top=True)

    ax.legend(fontsize=plt.rcParams['font.size'] * 0.66)

    secax = ax.secondary_yaxis('right')
    secax.set(yticks=lev_ticks_from_alt, yticklabels=[])
    secax.tick_params(direction='in')


    IASI_ID = 4
    ax = fig.add_subplot(gs[0,1])

    ds_iasi = datasets.get_IASI_profile(IASI_ID)
    da_iasi = ds_iasi['relative_humidity']
    data = (ds_iasi.time.values, None, ds_iasi.lon.values, ds_iasi.lat.values)

#    da_era5, _ = datasets.get_era5_profile('r', *data)
    da_era5, _ = datasets.get_ml_era5_r_profile(*data)
    da_ctrl, _ = datasets.get_lmdz_profile('CTRL', 'rhi', *data)
    da_issr, _ = datasets.get_lmdz_profile(simu_issr, 'rhi', *data)

    ax.plot(da_iasi, pressure_to_height_std(da_iasi.pressure_levels.values * units.hPa).to('m').magnitude, **iasi_dict)
    ax.plot(*iasi_running_mean(da_era5), **era5_dict)
    ax.plot(*iasi_running_mean(da_ctrl), **ctrl_dict)
    ax.plot(*iasi_running_mean(da_issr), **issr_dict)

    ax.set_title('(b) IASI at 21h23 UTC', loc='left')

    ax.set(ylim=alt_bounds, yticks=alt_ticks, yticklabels=[])
    ax.set(xlim=(0., 120.))
    ax.tick_params(direction='in', top=True)

    secax = ax.secondary_yaxis('right')
    secax.set(yticks=lev_ticks_from_alt, yticklabels=[])
    secax.tick_params(direction='in')


    ddhh = '2700'
    ax = fig.add_subplot(gs[0,2])

    time_rs, lon_rs, lat_rs, alt_rs, pres_rs, temp_rs, rhi_rs, rhl_rs, \
        wind_dir_rs, wind_force_rs, u_rs, v_rs = datasets.get_rs_trappes(ddhh)
    data = (time_rs, pres_rs, lon_rs, lat_rs)

    da_rs = rhi_rs

    da_era5, _ = datasets.get_ml_era5_r_profile(*data)
    da_ctrl, alt_ctrl = datasets.get_lmdz_profile('CTRL', 'rhi', *data)
    da_issr, alt_issr = datasets.get_lmdz_profile(simu_issr, 'rhi', *data)
#    variables = ['qissr', 'issrfra', 'rhi', 'ovap']
#    ds_issr, alt_issr = datasets.get_lmdz_profile(simu_issr, variables, *data)
#    da_issr = ds_issr.rhi / ds_issr.ovap * ds_issr.qissr / ds_issr.issrfra
#    da_issr = da_issr.where(ds_issr.issrfra > 0.)


#    _, unique_era5, counts_era5 = np.unique(da_era5.lev, return_index=True, return_counts=True)
    _, unique_lmdz, counts_lmdz = np.unique(da_ctrl.lev, return_index=True, return_counts=True)
#    unique_era5 += counts_era5 // 2
    unique_lmdz += counts_lmdz // 2
#    da_era5 = da_era5[unique_era5]
#    alt_era5 = alt_era5[unique_era5]
    da_ctrl = da_ctrl[unique_lmdz]
    alt_ctrl = alt_ctrl[unique_lmdz]
    da_issr = da_issr[unique_lmdz]
    alt_issr = alt_issr[unique_lmdz]

    da_era5 = da_era5[unique_lmdz]
    alt_era5 = alt_ctrl

    ax.plot(da_rs, alt_rs, **rs_dict)
    ax.plot(da_era5, alt_era5, **era5_dict)
    ax.plot(da_ctrl, alt_ctrl, **ctrl_dict)
    ax.plot(da_issr, alt_issr, **issr_dict)

    ax.set_title('(c) RS around 00h UTC', loc='left')

    ax.set(ylim=alt_bounds, yticks=alt_ticks, yticklabels=[])
    ax.set(xlim=(0., 120.))
    ax.tick_params(direction='in', top=True)

    secax = ax.secondary_yaxis('right')
    secax.set(yticks=lev_ticks_from_alt, yticklabels=lev_ticklabels)
    secax.tick_params(direction='in')


    IASI_ID = 5
    ax = fig.add_subplot(gs[1,0])

    ds_iasi = datasets.get_IASI_profile(IASI_ID)
    da_iasi = ds_iasi['relative_humidity']
    data = (ds_iasi.time.values, None, ds_iasi.lon.values, ds_iasi.lat.values)

#    da_era5, _ = datasets.get_era5_profile('r', *data)
    da_era5, _ = datasets.get_ml_era5_r_profile(*data)
    da_ctrl, _ = datasets.get_lmdz_profile('CTRL', 'rhi', *data)
    da_issr, _ = datasets.get_lmdz_profile(simu_issr, 'rhi', *data)

    ax.plot(da_iasi, pressure_to_height_std(da_iasi.pressure_levels.values * units.hPa).to('m').magnitude, **iasi_dict)
    ax.plot(*iasi_running_mean(da_era5), **era5_dict)
    ax.plot(*iasi_running_mean(da_ctrl), **ctrl_dict)
    ax.plot(*iasi_running_mean(da_issr), **issr_dict)

    ax.set_title('(d) IASI at 09h36 UTC', loc='left')

    ax.set(ylim=alt_bounds, yticks=alt_ticks, yticklabels=alt_ticklabels)
    ax.set(xlim=(0., 120.))
    ax.tick_params(direction='in', top=True)

    secax = ax.secondary_yaxis('right')
    secax.set(yticks=lev_ticks_from_alt, yticklabels=[])
    secax.tick_params(direction='in')


    IASI_ID = 2
    ax = fig.add_subplot(gs[1,1])

    ds_iasi = datasets.get_IASI_profile(IASI_ID)
    da_iasi = ds_iasi['relative_humidity']
    data = (ds_iasi.time.values, None, ds_iasi.lon.values, ds_iasi.lat.values)

#    da_era5, _ = datasets.get_era5_profile('r', *data)
    da_era5, _ = datasets.get_ml_era5_r_profile(*data)
    da_ctrl, _ = datasets.get_lmdz_profile('CTRL', 'rhi', *data)
    da_issr, _ = datasets.get_lmdz_profile(simu_issr, 'rhi', *data)

    ax.plot(da_iasi, pressure_to_height_std(da_iasi.pressure_levels.values * units.hPa).to('m').magnitude, **iasi_dict)
    ax.plot(*iasi_running_mean(da_era5), **era5_dict)
    ax.plot(*iasi_running_mean(da_ctrl), **ctrl_dict)
    ax.plot(*iasi_running_mean(da_issr), **issr_dict)

    ax.set_title('(e) IASI at 10h24 UTC', loc='left')

    ax.set(ylim=alt_bounds, yticks=alt_ticks, yticklabels=[])
    ax.set(xlim=(0., 120.))
    ax.tick_params(direction='in', top=True)

    secax = ax.secondary_yaxis('right')
    secax.set(yticks=lev_ticks_from_alt, yticklabels=[])
    secax.tick_params(direction='in')

    ax.set(xlabel='Relative humidity w.r.t. ice [%]')


    ddhh = '2712'
    ax = fig.add_subplot(gs[1,2])

    time_rs, lon_rs, lat_rs, alt_rs, pres_rs, temp_rs, rhi_rs, rhl_rs, \
        wind_dir_rs, wind_force_rs, u_rs, v_rs = datasets.get_rs_trappes(ddhh)
    data = (time_rs, pres_rs, lon_rs, lat_rs)

    da_rs = rhi_rs

    da_era5, _ = datasets.get_ml_era5_r_profile(*data)
    da_ctrl, alt_ctrl = datasets.get_lmdz_profile('CTRL', 'rhi', *data)
    da_issr, alt_issr = datasets.get_lmdz_profile(simu_issr, 'rhi', *data)

    _, unique_lmdz, counts_lmdz = np.unique(da_ctrl.lev, return_index=True, return_counts=True)
    unique_lmdz += counts_lmdz // 2
    da_ctrl = da_ctrl[unique_lmdz]
    alt_ctrl = alt_ctrl[unique_lmdz]
    da_issr = da_issr[unique_lmdz]
    alt_issr = alt_issr[unique_lmdz]

    da_era5 = da_era5[unique_lmdz]
    alt_era5 = alt_ctrl

    ax.plot(da_rs, alt_rs, **rs_dict)
    ax.plot(da_era5, alt_era5, **era5_dict)
    ax.plot(da_ctrl, alt_ctrl, **ctrl_dict)
    ax.plot(da_issr, alt_issr, **issr_dict)

    ax.set_title('(f) RS around 12h UTC', loc='left')

    ax.set(ylim=alt_bounds, yticks=alt_ticks, yticklabels=[])
    ax.set(xlim=(0., 120.))
    ax.tick_params(direction='in', top=True)

    secax = ax.secondary_yaxis('right')
    secax.set(yticks=lev_ticks_from_alt, yticklabels=lev_ticklabels)
    secax.tick_params(direction='in')


    ax = fig.add_subplot(gs[:,:])
    ax.set(yticks=[10], ylim=(9,15))
    [ax.spines[side].set_visible(False) for side in ('left', 'top', 'right', 'bottom')]
    ax.patch.set_visible(False)
    ax.tick_params(axis='both', colors=[0,0,0,0])
    ax.set_ylabel('Altitude [km]')

    ax = fig.add_subplot(gs[:,:])
    ax.set(yticks=[450], ylim=(500,200))
    [ax.spines[side].set_visible(False) for side in ('left', 'top', 'right', 'bottom')]
    ax.patch.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')
    ax.tick_params(axis='both', colors=[0,0,0,0])
    ax.set_ylabel('Equivalent standard pressure [hPa]')


    plt.savefig('profile_rhi.png')
#    plt.show()
    plt.close()
    return


def timeseries_rad():
    print('go timeseries_rad')

    obs_dict = dict(label='Obs.', color='grey')
    era5_dict = dict(label='ERA5', color='black', ls='-.')
    ctrl_dict = dict(label='CTRL', color=sns.color_palette()[0])
    issr_dict = dict(label='ISSR', color=sns.color_palette()[1])

    time_bounds = (np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T15'))
    step = np.timedelta64(15, 'm')
    times = np.arange(time_bounds[0], time_bounds[1] + step, step)

    lon_sirta, lat_sirta = 2.2, 48.7

    data = (times, None, lon_sirta, lat_sirta)


    x_figsize, y_figsize = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(x_figsize, 1.*y_figsize))
    gs = gridspec.GridSpec(1, 1, figure=fig)
    axs = []


    ax = fig.add_subplot(gs[0])
    axs.append(ax)

    da_obs  = datasets.get_radflux('LW_down', time_bounds)
    da_era5 = datasets.get_era5_radflux('lwdn', *data)
    da_ctrl, _ = datasets.get_lmdz_profile('CTRL', 'rld', *data)
    da_issr, _ = datasets.get_lmdz_profile(simu_issr, 'rld', *data)
    da_ctrl_cs, _ = datasets.get_lmdz_profile('CTRL', 'rldcs', *data)
    da_issr_cs, _ = datasets.get_lmdz_profile(simu_issr, 'rldcs', *data)

    levs_to_sel = xr.where(da_ctrl.time <= np.datetime64('2022-12-27T04'),
                           70000., 95000.)
    levs_to_sel = 70000.

    da_ctrl = da_ctrl_cs.sel(presinter=levs_to_sel, method='nearest') \
            - da_ctrl_cs.sel(presinter=999999., method='nearest') \
            - da_ctrl.sel(presinter=levs_to_sel, method='nearest')
    da_issr = da_issr_cs.sel(presinter=levs_to_sel, method='nearest') \
            - da_issr_cs.sel(presinter=999999., method='nearest') \
            - da_issr.sel(presinter=levs_to_sel, method='nearest')

    ax.plot(da_obs.time, da_obs, **obs_dict)
    ax.plot(da_era5.time, da_era5, **era5_dict)
    ax.plot(da_ctrl.time, da_ctrl, **ctrl_dict)
    ax.plot(da_issr.time, da_issr, **issr_dict)

#    ax.set_title('(a) Longwave', loc='left')

    xticks = np.arange(np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T14'), np.timedelta64(3, 'h'))
    xticklabels = [r'$\ \ \ \ $21h', '00h', '03h', '06h', '09h', '12h']
    ax.set(xticks=xticks, xticklabels=xticklabels)
#    ax.set(xlim=(xticks[0], xticks[-1]))
    ax.set(xlabel='Time UTC', xlim=(xticks[0], xticks[-1]))
    ax.legend(fontsize=plt.rcParams['font.size'] * 0.66)
    ax.set_ylim(220., 310.)

    ax.axvline(x=np.datetime64('2022-12-27T00'), lw=1., color='dimgrey')
    ax.axvline(x=np.datetime64('2022-12-27T09'), lw=1., color='dimgrey')

    ax.tick_params(direction='in', top=True, right=True)


#    ax = fig.add_subplot(gs[1])
#    axs.append(ax)
#
#    da_obs  = datasets.get_radflux('SW_down', time_bounds)
#    da_era5 = datasets.get_era5_radflux('swdn', *data)
#    da_ctrl, _ = datasets.get_lmdz_profile('CTRL', 'rsd', *data)
#    da_issr, _ = datasets.get_lmdz_profile(simu_issr, 'rsd', *data)
#
#    da_ctrl_cs, _ = datasets.get_lmdz_profile('CTRL', 'rsdcs', *data)
#    da_issr_cs, _ = datasets.get_lmdz_profile(simu_issr, 'rsdcs', *data)
#
#    levs_to_sel = 95000.
#
#    da_ctrl = da_ctrl_cs.sel(presinter=999999., method='nearest') \
#            - da_ctrl_cs.sel(presinter=levs_to_sel, method='nearest') \
#            + da_ctrl.sel(presinter=levs_to_sel, method='nearest')
#    da_issr = da_issr_cs.sel(presinter=999999., method='nearest') \
#            - da_issr_cs.sel(presinter=levs_to_sel, method='nearest') \
#            + da_issr.sel(presinter=levs_to_sel, method='nearest')
#
#    ax.plot(da_obs.time, da_obs, **obs_dict)
#    ax.plot(da_era5.time, da_era5, **era5_dict)
#    ax.plot(da_ctrl.time, da_ctrl, **ctrl_dict)
#    ax.plot(da_issr.time, da_issr, **issr_dict)
#
#    ax.set_title('(b) Shortwave', loc='left')
#
#    xticks = np.arange(np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T14'), np.timedelta64(3, 'h'))
#    xticklabels = [r'$\ \ \ \ $21h', '00h', '03h', '06h', '09h', '12h']
#    ax.set(xticks=xticks, xticklabels=xticklabels)
#    ax.set(xlabel='Time UTC', xlim=(xticks[0], xticks[-1]))
#    ax.legend(fontsize=plt.rcParams['font.size'] * 0.66, loc='upper left')
#    ax.set_ylim(0., 325.)
#
#    ax.axvline(x=np.datetime64('2022-12-27T00'), lw=1., color='dimgrey')
#    ax.axvline(x=np.datetime64('2022-12-27T09'), lw=1., color='dimgrey')
#
#    ax.tick_params(direction='in', top=True, right=True)


    ax = fig.add_subplot(gs[:])
    ax.set(yticks=[350], ylim=(349, 350))
    [ax.spines[side].set_visible(False) for side in ('left', 'top', 'right', 'bottom')]
    ax.patch.set_visible(False)
    ax.tick_params(axis='both', colors=[0,0,0,0])
    ax.set_ylabel('Radiative flux [W/m2]')


    plt.savefig('timeseries_rad.png')
#    plt.show()
    plt.close()
    return


def tendencies():
    print('go tendencies')

    # possible levs: 400, 375, 350, 325, 300, 280, 259, 236, 215, 195
    bounds = dict(time=(np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T15')),
                  lev=(350., 250.), lon=(-11.,11.), lat=(43.,51.))
#    bounds = dict(time=(np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T15')),
#                  lev=(290., 270.), lon=(-11.,11.), lat=(43.,51.))

    xticks = np.arange(np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T14'), np.timedelta64(3, 'h'))
    xticklabels = [r'$\ \ \ \ $21h', '00h', '03h', '06h', '09h', '12h']
    colors_dict = dict(con=sns.color_palette()[0], dis=sns.color_palette()[1],
                       adj=sns.color_palette()[2], mix=sns.color_palette()[3],
                       net=sns.color_palette()[7], sed=sns.color_palette()[9],
                       adv='black')


    x_figsize, y_figsize = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(3.*x_figsize, y_figsize))
    gs = gridspec.GridSpec(1, 3, figure=fig)
    axs = []

    
    ax = fig.add_subplot(gs[0])
    vartype = 'tendfrac'
    varnames = ['dcfcon', 'dcfdyn', 'dcfmix', 'dcfsub']
    da_issr = datasets.get_lmdz_average(vartype, varnames, simu_issr, **bounds)

    net = da_issr.dcfcon + da_issr.dcfsub + da_issr.dcfmix + da_issr.dcfdyn
#    net = da_issr.dcfcon + da_issr.dcfsub + da_issr.dcfmix
    ax.plot(da_issr.time, da_issr.dcfsub * 1e6, zorder=8, label='Diss.', color=colors_dict['dis'])
    ax.plot(da_issr.time, da_issr.dcfcon * 1e6, zorder=8, label='Form.', color=colors_dict['con'])
    ax.plot(da_issr.time, da_issr.dcfmix * 1e6, zorder=7, label='Mix.', color=colors_dict['mix'])
    ax.plot(da_issr.time, da_issr.dcfdyn * 1e6, zorder=6, label='Adv.', color=colors_dict['adv'])
    ax.plot(da_issr.time, net * 1e6, zorder=6, label='Net', color=colors_dict['net'], ls='--')

    ax.set(ylabel=r'Tendency [$\times 10^{-6}$ s$^{-1}$]')
    ax.set_title('(a) Cloud fraction tendencies', loc='left')
    ax.set_ylim(-9., 9.)

    ax.legend(fontsize=plt.rcParams['font.size'] * 0.66, loc='lower left', ncols=2).set_zorder(10)

    axs.append(ax)


    ax = fig.add_subplot(gs[2])
    vartype = 'tendice'
    varnames = ['dqicon', 'dqsdyn', 'dqisub', 'dqiadj', 'dqimix', 'pr_lsc_i', 'dqssub']
    da_issr = datasets.get_lmdz_average('tendice', varnames, simu_issr, **bounds)

    net = da_issr.dqicon + da_issr.dqisub + da_issr.dqimix + da_issr.dqised + da_issr.dqiadj + da_issr.dqsdyn
#    net = da_issr.dqicon + da_issr.dqisub + da_issr.dqised + da_issr.dqiadj
    ax.plot(da_issr.time, da_issr.dqiadj * 1e10, zorder=8, label='Adj.', color=colors_dict['adj'])
    ax.plot(da_issr.time, da_issr.dqicon * 1e10, zorder=7, label='Form.', color=colors_dict['con'])
    ax.plot(da_issr.time, da_issr.dqimix * 1e10, zorder=7, label='Mix.', color=colors_dict['mix'])
    ax.plot(da_issr.time, da_issr.dqised * 1e10, zorder=8, label='Auto.', color=colors_dict['sed'])
    ax.plot(da_issr.time, da_issr.dqsdyn * 1e10, zorder=6, label='Adv.', color=colors_dict['adv'])
    ax.plot(da_issr.time, net * 1e10, zorder=6, label='Net', color=colors_dict['net'], ls='--')

    ax.set(ylabel=r'Tendency [$\times 10^{-10}$ kg/kg/s]')
    ax.set_title('(c) Ice water content tendencies', loc='left')
    ax.set_ylim(-2.5, 2.5)

    ax.legend(fontsize=plt.rcParams['font.size'] * 0.66, ncols=2).set_zorder(10)

    axs.append(ax)


    ax = fig.add_subplot(gs[1])
    vartype = 'tendcloudvap'
    varnames = ['dqvccon', 'dqvcsub', 'dqvcmix', 'dqvcadj', 'drvcdyn', 'ovap']
    da_issr = datasets.get_lmdz_average(vartype, varnames, simu_issr, **bounds)

    net = da_issr.dqvccon + da_issr.dqvcsub + da_issr.dqvcadj + da_issr.dqvcmix + da_issr.drvcdyn * da_issr.ovap
#    net = da_issr.dqvccon + da_issr.dqvcsub + da_issr.dqvcadj + da_issr.dqvcmix
    ax.plot(da_issr.time, da_issr.dqvcadj * 1e10, zorder=8, label='Adj.', color=colors_dict['adj'])
    ax.plot(da_issr.time, da_issr.dqvcsub * 1e10, zorder=8, label='Diss.', color=colors_dict['dis'])
    ax.plot(da_issr.time, da_issr.dqvccon * 1e10, zorder=8, label='Form.', color=colors_dict['con'])
    ax.plot(da_issr.time, da_issr.dqvcmix * 1e10, zorder=8, label='Mix.', color=colors_dict['mix'])
    ax.plot(da_issr.time, da_issr.drvcdyn * 1e10 * da_issr.ovap, zorder=6, label='Adv.', color=colors_dict['adv'])
    ax.plot(da_issr.time, net * 1e10, zorder=6, label='Net', color=colors_dict['net'], ls='--')

    ax.set(ylabel=r'Tendency [$\times 10^{-10}$ kg/kg/s]')
    ax.set_title('(b) Cloud water vapor tendencies', loc='left')
    ax.set(xlabel='Time UTC')
    ax.set_ylim(-6., 6.)

    ax.legend(fontsize=plt.rcParams['font.size'] * 0.66, loc='lower left', ncols=2).set_zorder(10)

    axs.append(ax)


    for ax in axs:
        ax.set(xlim=(xticks[0], xticks[-1]), xticks=xticks, xticklabels=xticklabels)

        ax.axhline(y=0., lw=1., color='black')
        ax.axvline(x=np.datetime64('2022-12-27T00'), lw=1., color='dimgrey')
        ax.axvline(x=np.datetime64('2022-12-27T09'), lw=1., color='dimgrey')
    
        ax.tick_params(direction='in', top=True, right=True)


    plt.savefig('tendencies.png')
#    plt.show()
    plt.close()
    return


def sensitivity():
    print('go sensitivity')

    list_simus = {
            'ERA5' : dict(label='ERA5',      panel=0, color='black',                ls='-.', lw=4.),
            'CTRL' : dict(label='CTRL',      panel=0, color=sns.color_palette('dark')[0], ls='-' , lw=4.),
            'REF'  : dict(label='ISSR',      panel=0, color=sns.color_palette('dark')[3], ls='-' , lw=4.),
            'SIM01': dict(label='CAPA-',     panel=1, color=sns.color_palette()[0], ls='-' , lw=1.5),
            'SIM02': dict(label='CAPA+',     panel=1, color=sns.color_palette()[1], ls='-' , lw=1.5),
#            'SIM03': dict(label='EPMAX-',    panel=2, color=sns.color_palette()[0], ls='-.', lw=1.5),
#            'SIM04': dict(label='EPMAX+',    panel=2, color=sns.color_palette()[1], ls='-.', lw=1.5),
            'SIM05': dict(label='DISS-',     panel=1, color=sns.color_palette()[2], ls='-.', lw=1.5),
            'SIM06': dict(label='DISS+',     panel=1, color=sns.color_palette()[3], ls='-.', lw=1.5),
            'SIM07': dict(label='HETFREEZ',  panel=2, color=sns.color_palette()[0], ls='--', lw=1.5),
            'SIM08': dict(label='HOMFREEZ',  panel=2, color=sns.color_palette()[1], ls='--', lw=1.5),
#            'SIM09': dict(label='WIDTH--',   panel=2, color=sns.color_palette()[3], ls='-.', lw=1.5),
            'SIM10': dict(label='WIDTH-',    panel=2, color=sns.color_palette()[2], ls='--', lw=1.5),
            'SIM11': dict(label='WIDTH+',    panel=2, color=sns.color_palette()[3], ls='--', lw=1.5),
            'SIM12': dict(label='MIX-',      panel=3, color=sns.color_palette()[2], ls='-' , lw=1.5),
            'SIM13': dict(label='MIX+',      panel=3, color=sns.color_palette()[3], ls='-' , lw=1.5),
#            'SIM14': dict(label='NOSHEAR',   panel=2, color=sns.color_palette()[2], ls='-.', lw=1.5),
#            'SIM15': dict(label='SHEAR+',    panel=2, color=sns.color_palette()[3], ls='-.', lw=1.5),
#            'SIM16': dict(label='FALLV0.2',  panel=3, color=sns.color_palette()[0], ls=':' , lw=1.5),
            'SIM17': dict(label='FALLV-',    panel=3, color=sns.color_palette()[0], ls=':' , lw=1.5),
            'SIM18': dict(label='FALLV+',    panel=3, color=sns.color_palette()[1], ls=':' , lw=1.5),
#            'SIM19': dict(label='FALLV4.0',  panel=3, color=sns.color_palette()[3], ls=':' , lw=1.5),
#            'SIM20': dict(label='ADJCLD',    panel=0, color=sns.color_palette()[5], ls='-' , lw=1.5),
#            'SIM21': dict(label='WEIB1',     panel=3, color=sns.color_palette()[5], ls=':' , lw=1.5),
#            'SIM22': dict(label='WEIB2',     panel=3, color=sns.color_palette()[6], ls=':' , lw=1.5),
#            'SIM23': dict(label='EXP-RND',   panel=3, color=sns.color_palette()[7], ls=':' , lw=1.5),
#            'SIM24': dict(label='POP',       panel=3, color=sns.color_palette()[0], ls='-.', lw=1.5),
#            'SIM25': dict(label='POP/TAU-',  panel=3, color=sns.color_palette()[1], ls='-.', lw=1.5),
#            'SIM26': dict(label='POP/TAU+',  panel=3, color=sns.color_palette()[2], ls='-.', lw=1.5),
#            'SIM27': dict(label='POP/CLD-',  panel=3, color=sns.color_palette()[3], ls='-.', lw=1.5),
#            'SIM28': dict(label='POP/CLD+',  panel=3, color=sns.color_palette()[4], ls='-.', lw=1.5),
            }

    # possible levs: 400, 375, 350, 325, 300, 280, 250, 235, 215, 195
    bounds = dict(time=(np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T15')),
                  lev=(350., 250.), lon=(-11.,11.), lat=(43.,51.))
#    bounds = dict(time=(np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T15')),
#                  lev=(290., 270.), lon=(-11.,11.), lat=(43.,51.))

    xticks = np.arange(np.datetime64('2022-12-26T21'), np.datetime64('2022-12-27T14'), np.timedelta64(3, 'h'))
    xticklabels = [r'$\ \ \ \ $21h', '00h', '03h', '06h', '09h', '12h']
    colors_dict = dict(con=sns.color_palette()[0], dis=sns.color_palette()[1],
                       adj=sns.color_palette()[2], mix=sns.color_palette()[3],
                       net=sns.color_palette()[7], sed=sns.color_palette()[9],
                       adv='black')


    x_figsize, y_figsize = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(2.*x_figsize, 2.*y_figsize))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    axs = [ax00, ax01, ax10, ax11]


    print('ERA5')
    ds_era5 = datasets.get_ml_era5_average(**bounds)
    plot_dict = list_simus['ERA5']
    plot_dict.pop('panel')
    ax00.plot(ds_era5.time, ds_era5.cc, **plot_dict)
    plot_dict.pop('label')
    ax10.plot(ds_era5.time, ds_era5.rhi, **plot_dict)
    ax11.plot(ds_era5.time, ds_era5.ciwc * 1e7, **plot_dict)
#    ax11.plot(ds_era5.time, ds_era5.qiceincld * 1e7, **plot_dict)
#    ax11.plot(ds_era5.time, ds_era5.ciwc * 1e6 / ds_era5.cc, **plot_dict)


    print('CTRL')
    vartype = ['oice_CTRL', 'rhi_clr_CTRL']
    varnames = ['rnebls', 'rhi', 'ocond', 'oliq']
    da_issr = datasets.get_lmdz_average(vartype, varnames, 'CTRL', **bounds)
    plot_dict = list_simus['CTRL']
    plot_dict.pop('panel')
    ax00.plot(da_issr.time, da_issr.rnebls, **plot_dict)
    plot_dict.pop('label')
    ax01.axhline(0., **plot_dict)
    ax10.plot(da_issr.time, da_issr.rhi, **plot_dict)
    ax11.plot(da_issr.time, da_issr.oice * 1e7, **plot_dict)
#    ax11.plot(da_issr.time, da_issr.qiceincld * 1e7, **plot_dict)
#    ax11.plot(da_issr.time, da_issr.oice * 1e6 / da_issr.rnebls, **plot_dict)


    vartype = ['oice', 'rhi_clr', 'rhi_issr']
    varnames = ['cfseri', 'issrfra', 'rhi', 'rvcseri', 'ocond', 'oliq', 'ovap', 'qissr', 'temp']
    for simu_issr, plot_dict in list_simus.items():
        if simu_issr in ('ERA5', 'CTRL'): continue
        print(simu_issr)

        da_issr = datasets.get_lmdz_average(vartype, varnames, simu_issr, **bounds)

        panel, label = plot_dict.pop('panel'), plot_dict.pop('label')

        if panel == 0:
            ax00.plot(da_issr.time, da_issr.cfseri, **plot_dict, label=label)
        else:
            ax00.plot(da_issr.time, da_issr.cfseri, **plot_dict)
#        if panel == 0:
#            ax00.plot(da_issr.time, da_issr.rhi_issr, **plot_dict, label=label)
#        else:
#            ax00.plot(da_issr.time, da_issr.rhi_issr, **plot_dict)

        if panel == 1:
            ax01.plot(da_issr.time, da_issr.issrfra, **plot_dict, label=label)
        else:
            ax01.plot(da_issr.time, da_issr.issrfra, **plot_dict)
#        if panel == 1:
#            ax01.plot(da_issr.time, da_issr.ovap, **plot_dict, label=label)
#        else:
#            ax01.plot(da_issr.time, da_issr.ovap, **plot_dict)

        if panel == 2:
            ax10.plot(da_issr.time, da_issr.rhi, **plot_dict, label=label)
        else:
            ax10.plot(da_issr.time, da_issr.rhi, **plot_dict)
#        if panel == 2:
#            ax10.plot(da_issr.time, da_issr.rhi, **plot_dict, label=label)
#        else:
#            ax10.plot(da_issr.time, da_issr.rhi, **plot_dict)

        if panel == 3:
            ax11.plot(da_issr.time, da_issr.oice * 1e7, **plot_dict, label=label)
        else:
            ax11.plot(da_issr.time, da_issr.oice * 1e7, **plot_dict)
#        if panel == 3:
#            ax11.plot(da_issr.time, da_issr.qiceincld * 1e7, **plot_dict, label=label)
#        else:
#            ax11.plot(da_issr.time, da_issr.qiceincld * 1e7, **plot_dict)
#        if panel == 3:
#            ax11.plot(da_issr.time, da_issr.oice * 1e6 / da_issr.cfseri, **plot_dict, label=label)
#        else:
#            ax11.plot(da_issr.time, da_issr.oice * 1e6 / da_issr.cfseri, **plot_dict)


    ax00.set(ylabel=r'Cloud fraction $\alpha_c$ [-]')
    ax00.set_title('(a) Cloud fraction', loc='left')
    ax00.set_yticks([0., 0.05, 0.10, 0.15])
    ax00.set_ylim(0., 0.15)
    ax00.legend(fontsize=plt.rcParams['font.size'] * 0.66, loc='upper left')

    ax01.set(ylabel=r'ISSR fraction [-]')
    ax01.set_title('(b) Ice supersaturated region fraction', loc='left')
    ax01.set_ylim(0., None)
    ax01.legend(fontsize=plt.rcParams['font.size'] * 0.66, loc='upper left')

    ax10.set(ylabel=r'$r_i$ [%]')
    ax10.set_title('(c) Relative humidity w.r.t. ice', loc='left')
    ax10.set_ylim(40., 90.)
    ax10.legend(fontsize=plt.rcParams['font.size'] * 0.66, loc='lower right')

    ax11.set(ylabel=r'CIWC [$\times 10^{-7}$ kg/kg]')
    ax11.set_title('(d) Cloud ice water content', loc='left')
#    ax11.set_ylim(0., 13.)
    ax11.set_ylim(0., 12.)
    ax11.legend(fontsize=plt.rcParams['font.size'] * 0.66, loc='upper right')


    for ax in axs:
        ax.set(xlim=(xticks[0], xticks[-1]), xticks=xticks, xticklabels=xticklabels)

        ax.axvline(x=np.datetime64('2022-12-27T00'), lw=1., color='dimgrey')
        ax.axvline(x=np.datetime64('2022-12-27T09'), lw=1., color='dimgrey')
    
        ax.tick_params(direction='in', top=True, right=True)

    ax = fig.add_subplot(gs[:])
    ax.set(xlim=(xticks[0], xticks[-1]), xticks=xticks)
    [ax.spines[side].set_visible(False) for side in ('left', 'top', 'right', 'bottom')]
    ax.patch.set_visible(False)
    ax.tick_params(axis='both', colors=[0,0,0,0])
    ax.set_xlabel('Time UTC')

    plt.savefig('sensitivity.png')
#    plt.show()
    plt.close()
    return



plt.rcParams.update({
                     'figure.dpi': 200,
                     'figure.constrained_layout.use': True,
                     'figure.constrained_layout.h_pad': 5./72.,
                     'figure.constrained_layout.w_pad': 5./72.,
                     'figure.figsize': (6.4, 4.8),
                     'lines.linewidth': 3.,
                     'axes.prop_cycle': cycler(color=sns.color_palette()),
                     'axes.labelsize': 22,
                     'axes.titlesize': 22,
                     'legend.fontsize': 22,
                     'xtick.labelsize': 22,
                     'ytick.labelsize': 22,
                     'font.size': 25,
                     'font.family': 'Lato',
                    })

# psyplot params                                                                    
#psy.rcParams['plotter.maps.map_extent'] = [*lon_bounds, *lat_bounds]
psy.rcParams['plotter.maps.projection'] = 'cyl' # PlateCarree
psy.rcParams['plotter.maps.xgrid'] = False
psy.rcParams['plotter.maps.ygrid'] = False
# allowed resolutions: ('110m', '50m', '10m')
psy.rcParams['plotter.maps.lsm'] = dict(res='110m', linewidth=0.5, coast='dimgrey')


#scheme_param()
#scheme_mixing()
#case_study_MSG()
#grid()
timeseries_cldh()
#map_models()
#profile_basta()
#profile_rhi()
#timeseries_rad()
#tendencies()
#sensitivity()
