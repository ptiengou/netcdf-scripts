import numpy as np

import plot_mod
#import seviri

time_snapshot = np.datetime64('2022-12-27T05')
level_snapshot = 250.
simu_issr = 'REF'
ok_contour = False

#plot_mod.plot_map_models(level_snapshot=level_snapshot, time_snapshot=time_snapshot,
#                         varname='cc', simu_issr=simu_issr, ok_contour=ok_contour)

#times = [np.datetime64('2022-12-27T03'), np.datetime64('2022-12-27T06'), np.datetime64('2022-12-27T09'), np.datetime64('2022-12-27T12')]
#plot_mod.plot_map_MSG_evolution(times)

#plot_mod.plot_timeseries('hcc-isccp', 'REF')
#plot_mod.plot_timeseries('cldh', 'REF')
#plot_mod.plot_timeseries('iwp', 'REF')
#plot_mod.plot_timeseries('tendice', 'REF')

#plot_mod.plot_map_MODIS(MODIS_ID=1, simu_issr=simu_issr)

plot_mod.plot_map_IASI(IASI_ID=2, level_snapshot=level_snapshot,
                       varname='hcc', simu_issr=simu_issr)

#plot_mod.plot_profile(ref='', varname='r', simu_issr=simu_issr)
#plot_mod.plot_profile(ref='CHM15K-SIRTA', varname='cc', simu_issr=simu_issr)
#plot_mod.plot_profile(ref='BASTA-SIRTA', varname='cc', simu_issr=simu_issr)
#plot_mod.plot_profile(ref='CHM15K-QUALAIR', varname='cc', simu_issr=simu_issr)

#plot_mod.plot_profile_calipso(varname='cc', simu_issr=simu_issr)

#plot_mod.plot_radiosounding(ddhh='2700', varnames=('wind_force', 'wind_dir', 't', 'rhi'),
#                            simu_issr=simu_issr)

del simu_issr
#simu_issr = 'REF'
#simu_issr = 'SIM01'
#simu_issr = 'SIM02'
#simu_issr = 'SIM03'
#simu_issr = 'SIM04'
#simu_issr = 'SIM05'
#simu_issr = 'SIM06'
#simu_issr = 'SIM07'
#simu_issr = 'SIM08'
#simu_issr = 'SIM09'
#simu_issr = 'SIM10'
#simu_issr = 'SIM11'
#simu_issr = 'SIM12'
#simu_issr = 'SIM13'
#simu_issr = 'SIM14'
#simu_issr = 'SIM15'
#simu_issr = 'SIM16'
#simu_issr = 'SIM17'
#simu_issr = 'SIM18'
#simu_issr = 'SIM19'

#simu_issrs = ( \
#               'REF',
#              'SIM01',
#              'SIM02',
#              'SIM03',
#              'SIM04',
#              'SIM05',
#              'SIM06',
#              'SIM07',
#              'SIM08',
#              'SIM09',
#              'SIM10',
#              'SIM11',
#              'SIM12',
#              'SIM13',
#              'SIM14',
#              'SIM15',
#              'SIM16',
#              'SIM17',
#              'SIM18',
#              'SIM19',
#              )

#simu_issr = 'SIM20'
#simu_issr = 'SIM21'
#simu_issr = 'SIM22'
#simu_issr = 'SIM23'
#simu_issr = 'SIM24'
#simu_issr = 'SIM25'

print(simu_issr)

# vis outputs from models
print('map_models')
#for varname in 'r', 'cc':#, 't':
#    for level_snapshot in 200., 250., 300., 350.:
#        for time_snapshot in np.datetime64('2022-12-27T00'), np.datetime64('2022-12-27T03'), np.datetime64('2022-12-27T06'):
#            plot_mod.plot_map_models(level_snapshot=level_snapshot, time_snapshot=time_snapshot,
#                                     varname=varname, simu_issr=simu_issr, ok_contour=ok_contour)


# vis high cloud cover vs MSG
#IR_val = 'IR_120'
#print('map_MSG')
#for time_snapshot in ( \
#        np.datetime64('2022-12-27T00'), \
#        np.datetime64('2022-12-27T01'), \
#        np.datetime64('2022-12-27T02'), \
#        np.datetime64('2022-12-27T03'), \
#        np.datetime64('2022-12-27T04'), \
#        np.datetime64('2022-12-27T05'), \
#        np.datetime64('2022-12-27T06'),):
#    lons, lats, data = seviri.get_data(time_snapshot, IR_val)
#    plot_mod.plot_map_MSG(time_snapshot=time_snapshot, simu_issr=simu_issr, IR_val=IR_val,
#                          lon_obs=lons, lat_obs=lats, obs=data)
#    del lons, lats, data

# vis high cloud cover vs MODIS
print('map_MODIS')
#for ddhh in '2622', '2702', '2710':
#    plot_mod.plot_map_modis(ddhh=ddhh, simu_issr=simu_issr)

# vis profile from models
print('profile')
#for varname in 'r':#, 't':
#    plot_mod.plot_profile(ref='', varname=varname, simu_issr=simu_issr)

# vis cloud fraction profile from models vs obs
print('TODO utiliser radar 95 GHz COSP')
print('profile_obs')
#for ref in ('CHM15K-SIRTA',):# 'BASTA-SIRTA', 'CHM15K-QUALAIR'):
#    plot_mod.plot_profile(ref=ref, varname='cc', simu_issr=simu_issr)

# vis profiles from models vs CALIPSO
print('profile_calipso')
#varname = 'cc'
#plot_mod.plot_profile_calipso(varname=varname, simu_issr=simu_issr)

# vis profiles from models vs IASI
print('TODO kernel q pour IASI + hcc COSP ?')
print('profile_IASI')
#for IASI_ID in (2, 4, 5):
    #    for varname in 'q':#, 't':
#        for level_snapshot in 200., 250., 300., 350.:
#            plot_mod.plot_map_IASI(IASI_ID, level_snapshot, varname=varname, simu_issr=simu_issr)
#    plot_mod.plot_map_IASI(IASI_ID, level_snapshot, varname='hcc', simu_issr=simu_issr)

# vis profiles from models vs TRP RS
print('profile_radiosounding')
#for ddhh in '2612', '2700', '2712', '2800':
#    plot_mod.plot_radiosounding(ddhh=ddhh, varnames=('wind_force', 'wind_dir', 't', 'rhi'),
#                                simu_issr=simu_issr)

# vis timeseries
print('timeseries')
for simu_issr in simu_issrs:
    print(simu_issr)
    for vartype in ('frac', 'rhum', 'spehum', 'tendfrac', 'tendice', 'tendcloudvap',
                    'cldh', 'hcc-modis', 'hcc-calipso', 'hcc-isccp',
                    'hcod-modis', 'hcod-isccp',
                    'iwp', 'iwp-modis', 'lwp', 'lwp-modis'):
#    for vartype in ('cldh',):
        plot_mod.plot_timeseries(vartype, simu_issr)
