from tools import *
from tools_native import *

## CAMPAIGN INFO ##
RS_days_list=['1507','1607','1707','2007','2107','2207','2707']

SOP_start_time = '2021-07-15T00:00:00.000000000'
SOP_end_time='2021-07-28T00:00:00.000000000'

day1_start_time = '2021-07-15T00:00:00.000000000'
day1_end_time = '2021-07-16T00:00:00.000000000'

day2_start_time = '2021-07-16T00:00:00.000000000'
day2_end_time = '2021-07-17T00:00:00.000000000'

day3_start_time = '2021-07-17T00:00:00.000000000'
day3_end_time = '2021-07-18T00:00:00.000000000'

day4_start_time = '2021-07-20T00:00:00.000000000'
day4_end_time = '2021-07-21T00:00:00.000000000'

day5_start_time = '2021-07-21T00:00:00.000000000'
day5_end_time = '2021-07-22T00:00:00.000000000'

day6_start_time = '2021-07-22T00:00:00.000000000'
day6_end_time = '2021-07-23T00:00:00.000000000'

day7_start_time = '2021-07-27T00:00:00.000000000'
day7_end_time = '2021-07-28T00:00:00.000000000'

#actual data from LIAISE db
Cendrosa_Latitude=41.69336
Cendrosa_Longitude=0.928538
Cendrosa_altitude= 240

ElsPlans_Latitude = 41.590111
ElsPlans_Longitude = 1.029363
ElsPlans_altitude = 334

#Associated points on native grid
Cendrosa_lon=0.755
Cendrosa_lat=41.63
ElsPlans_lon=0.996
ElsPlans_lat=41.51

## FUNCTIONS FOR LIAISE SITES AND SOP##
def add_liaise_site_loc(ax=None):
    if ax is not None:
        # ax.plot(Cendrosa_lon, Cendrosa_lat, 'go', markersize=10, transform=ccrs.Geodetic())
        # ax.plot(ElsPlans_lon, ElsPlans_lat, 'go', markersize=10, transform=ccrs.Geodetic())
        ax.plot(Cendrosa_Longitude, Cendrosa_Latitude, 'ro', markersize=10, transform=ccrs.Geodetic())
        ax.plot(ElsPlans_Longitude, ElsPlans_Latitude, 'ro', markersize=10, transform=ccrs.Geodetic())
    else:
        # plt.plot(Cendrosa_lon, Cendrosa_lat, 'go', markersize=10, transform=ccrs.Geodetic())
        # plt.plot(ElsPlans_lon, ElsPlans_lat, 'go', markersize=10, transform=ccrs.Geodetic())
        plt.plot(Cendrosa_Longitude, Cendrosa_Latitude, 'ro', markersize=10, transform=ccrs.Geodetic())
        plt.plot(ElsPlans_Longitude, ElsPlans_Latitude, 'ro', markersize=10, transform=ccrs.Geodetic())

def select_liaise_sites_sop(ds, name_suffix, linestyle_cen=None, linestyle_els=None, sop=True):
    cen=sel_closest(ds, Cendrosa_lon, Cendrosa_lat)
    els=sel_closest(ds, ElsPlans_lon, ElsPlans_lat)
    cen.attrs['name']=name_suffix
    els.attrs['name']=name_suffix
    if linestyle_cen is not None and linestyle_els is not None:
        cen.attrs['linestyle']=linestyle_cen
        els.attrs['linestyle']=linestyle_els
    else:
        cen.attrs['linestyle']='-'
        els.attrs['linestyle']='--'
        
    if sop:
        sop_cen = filter_xarray_by_timestamps(cen, SOP_start_time, SOP_end_time)
        sop_els = filter_xarray_by_timestamps(els, SOP_start_time, SOP_end_time)
        return cen, els, sop_cen, sop_els
    else:
        return cen, els

## OBS FILES ##
## RS Cendrosa
cendrosa_files_1507=[
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210715-0420_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210715-0554_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210715-0800_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210715-0902_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210715-1001_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210715-1124_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210715-1207_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210715-1258_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210715-1400_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210715-1500_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210715-1559_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210715-1722_V2.nc",
]
cen_1507_times=[4,6,8,9,10,11,12,13,14,15,16,17]
cendrosa_files_1607=[
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210716-0358_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210716-0456_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210716-0603_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210716-0721_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210716-0800_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210716-0859_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210716-0959_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210716-1057_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210716-1157_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210716-1258_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210716-1359_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210716-1459_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210716-1600_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210716-1700_V2.nc"
]
cen_1607_times=[4,5,6,7,8,9,10,11,12,13,14,15,16,17]
cendrosa_files_1707=[
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210717-0408_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210717-0501_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210717-0606_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210717-0656_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210717-0809_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210717-0901_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210717-1003_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210717-1056_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210717-1158_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210717-1303_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210717-1401_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210717-1500_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210717-1602_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210717-1700_V2.nc"
]
cen_1707_times=[4,5,6,7,8,9,10,11,12,13,14,15,16,17]
cendrosa_files_2007=[
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210720-0400_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210720-0459_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210720-0600_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210720-0702_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210720-0800_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210720-0900_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210720-1011_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210720-1100_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210720-1200_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210720-1300_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210720-1401_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210720-1501_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210720-1600_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210720-1700_V2.nc"
]
cen_2007_times=[4,5,6,7,8,9,10,11,12,13,14,15,16,17]
cendrosa_files_2107=[
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210721-0401_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210721-0501_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210721-0601_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210721-0702_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210721-0806_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210721-0900_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210721-1011_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210721-1103_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210721-1204_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210721-1302_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210721-1358_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210721-1501_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210721-1557_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210721-1700_V2.nc"
]
cen_2107_times=[4,5,6,7,8,9,10,11,12,13,14,15,16,17]
cendrosa_files_2207=[    
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-0400_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-0500_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-0559_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-0659_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-0800_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-0900_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-1001_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-1100_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-1200_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-1300_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-1359_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-1501_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-1600_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-1746_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-1906_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-2117_V2.nc"
]
cen_2207_times=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21]
cendrosa_files_2707=[
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-0401_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-0459_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-0558_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-0700_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-0800_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-0859_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-1000_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-1059_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-1159_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-1259_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-1400_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-1500_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-1600_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-1703_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-1858_V2.nc",
    "LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210727-2103_V2.nc"
]
cen_2707_times=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21]

## RS Els Plans
els_files_1507=[
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210715-040009_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210715-050013_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210715-060005_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210715-070005_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210715-080123_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210715-090007_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210715-100006_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210715-110004_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210715-120005_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210715-130035_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210715-140029_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210715-145947_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210715-160203_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210715-170029_V1.0.txt"
]
els_1507_times=[4,5,6,7,8,9,10,11,12,13,14,15,16,17]
els_files_1607=[
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-040006_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-050015_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-060007_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-070006_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-080006_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-090014_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-100007_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-110028_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-120009_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-130012_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-140007_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-160017_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-165958_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-180007_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-190008_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-200014_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210716-210011_V1.0.txt"
]
els_1607_times=[4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21]
els_files_1707=[
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210717-040019_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210717-050012_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210717-060007_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210717-070008_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210717-080009_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210717-090007_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210717-100010_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210717-110010_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210717-120043_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210717-130011_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210717-140006_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210717-145941_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210717-160031_V1.0.txt"
]
els_1707_times=[4,5,6,7,8,9,10,11,12,13,14,16,17]
els_files_2007=[
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-040010_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-050011_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-060007_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-070005_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-080009_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-090009_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-100006_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-110547_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-120007_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-130014_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-140246_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-150017_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-160016_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-170012_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-190005_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210720-210013_V1.0.txt"
]
els_2007_times=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21]
els_files_2107=[
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-040008_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-050014_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-060007_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-070006_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-080011_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-090008_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-100006_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-110120_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-120247_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-130018_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-140032_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-150048_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-160010_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-170010_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-190058_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210721-210029_V1.0.txt"
]
els_2107_times=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21]
els_files_2207=[
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-040029_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-050013_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-060005_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-070034_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-080006_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-090001_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-100009_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-110011_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-120040_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-130007_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-140158_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-150035_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-160021_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-170013_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-190010_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-200336_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210722-210010_V1.0.txt"
]
els_2207_times=[4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21]
els_files_2707=[
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-050002_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-060010_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-070006_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-080004_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-090012_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-100007_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-110011_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-120026_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-130008_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-140018_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-150015_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-160010_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-170011_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-180019_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-190018_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-200015_V1.0.txt",
    "LIAISE_ELS-PLANS_UKMO_RADIOSONDEPROFILE_L1-20210727-210012_V1.0.txt"
]
els_2707_times=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]



## FUNCTIONS FOR EXTRACTING DATA ##

## RS
def txtRS_to_xarray(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract metadata
    metadata = {}
    for line in lines[:7]:  # Assuming the first 7 lines are metadata
        key, value = line.split('\n')
        metadata[key.strip()] = value.strip()

    # Extract time series data
    data_start_line = 10  # Assuming the data starts from the 10th line
    data_lines = lines[data_start_line:]

    # Extract variable names and units
    variable_names = lines[8].split()
    variable_units = lines[9].split()
    # Fix split unit 'deg C'
    fixed_units = []
    i = 0
    while i < len(variable_units):
        if variable_units[i] == 'deg' and i + 1 < len(variable_units) and variable_units[i + 1] == 'C':
            fixed_units.append('deg C')
            i += 2  # Skip the next part
        else:
            fixed_units.append(variable_units[i])
            i += 1

    # Convert data lines to a DataFrame
    df = pd.read_csv(StringIO(''.join(data_lines)), sep='\s+', names=variable_names)

    # Convert DataFrame to xarray.Dataset
    ds = xr.Dataset.from_dataframe(df.set_index('Time'))

    # Add metadata as attributes
    ds.attrs = metadata

    for var, unit in zip(variable_names[1:], fixed_units[1:]):  # Skip 'Time'
        ds[var].attrs['units'] = unit
    
    return ds

def read_varnames_from_dat(file_path, height_line, name_line):
    # print(f"Loading file: {file_path}")

    # Read file into list
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract and clean height values from line 152 (index 151)
    heights = lines[height_line].lstrip("!").strip().split()
    # print(f"Extracted heights: {heights}")

    # Extract and clean variable names from line 153 (index 152)
    variable_names = lines[name_line].lstrip("!").strip().split()
    # print(f"Extracted variable names: {variable_names}")

    # Ensure lengths match
    if len(variable_names) != len(heights):
        raise ValueError("Mismatch between variable names and height values!")

    # Create full variable names by concatenating variable name with height
    full_variable_names = [f"{var}_{height}" for var, height in zip(variable_names, heights)]
    # print(f"Full variable names:\n{full_variable_names}")

    return(full_variable_names)
    
def dat_to_xarray(file_path, height_line, name_line):
    variable_names=read_varnames_from_dat(file_path, height_line, name_line)

    # print(f"Extracting data from: {file_path}")

    # Read file into list
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract data lines starting from line 154 (index 153)
    data_lines = lines[name_line+1:]

    # Process each line
    data = []
    for i, line in enumerate(data_lines):
        # Split and filter out non-numeric symbols
        values = [val for val in line.split() if val not in {"|", "X", "m", "D", 'c', '?', '–'}]

        # Ensure correct number of values
        if len(values) != len(variable_names):
            print(f"Warning: Skipping line {i + 154}, expected {len(variable_names)} values but got {len(values)}")
            continue
        
        # Convert values to float and store
        data.append([float(v) for v in values])

    # Create DataFrame
    df = pd.DataFrame(data, columns=variable_names)
    # print(f"Extracted DataFrame:\n{df.head()}")

    # Convert DataFrame to xarray with HOUR_time as the dimension
    df = df.set_index('HOUR_time')
    ds=df.to_xarray()

    # print(f"Converted xarray dataset:\n{ds}")

    return ds

def format_Cendrosa_RS(filename, name='obs'):
    Cendrosa_RS = xr.open_mfdataset(filename)
    Cendrosa_RS.attrs['name'] = name
    Cendrosa_RS.attrs['plot_color']='black'

    # Rename vars
    rename_dict = {'windSpeed':'wind_speed', 
                'windDirection':'wind_direction',
                }
    Cendrosa_RS = Cendrosa_RS.rename(rename_dict)

    # Calculate potential temperature
    Cendrosa_RS['theta'] = Cendrosa_RS['temperature'] * (1000 / Cendrosa_RS['pressure']) ** 0.286
    Cendrosa_RS['theta'].attrs['units'] = 'K'

    # Calculate specific humidity
    Cendrosa_RS['ovap'] = 1000* Cendrosa_RS['mixingRatio'] / (1000 + Cendrosa_RS['mixingRatio'])
    Cendrosa_RS['ovap'].attrs['units'] = 'g kg⁻¹'

    Cendrosa_RS['altitude_agl'] = Cendrosa_RS['altitude']

    return Cendrosa_RS

def format_ElsPlans_RS(filename, name='obs'):
    ElsPlans_RS = txtRS_to_xarray(filename)
    ElsPlans_RS.attrs['name'] = name
    ElsPlans_RS.attrs['plot_color']='black'
    rename_dict = {
        'Time':'time',
        'Height':'altitude',
        'Pres':'pressure',
        'Temp':'temperature',
        'RH':'humidity',
        'Dewp':'dewPoint',
        'MixR':'mixingRatio',
        'Dir':'wind_direction',
        'Speed':'wind_speed',
    }
    ElsPlans_RS = ElsPlans_RS.rename(rename_dict)

    # Calculate potential temperature
    ElsPlans_RS['temperature'] = ElsPlans_RS['temperature'] + 273.15
    ElsPlans_RS['temperature'].attrs['units'] = 'K'

    ElsPlans_RS['theta'] = ElsPlans_RS['temperature'] * (1000 / ElsPlans_RS['pressure']) ** 0.286
    ElsPlans_RS['theta'].attrs['units'] = 'K'

    # Calculate specific humidity
    ElsPlans_RS['ovap'] = 1000* ElsPlans_RS['mixingRatio'] / (1000 + ElsPlans_RS['mixingRatio'])
    ElsPlans_RS['ovap'].attrs['units'] = 'g kg⁻¹'

    # Add altitude above ground level
    ElsPlans_RS['altitude_agl'] = ElsPlans_RS['altitude']

    return ElsPlans_RS

def RS_extract_full_day(file_list, times_list, site_dir, format_function):
    #extract full day of radiosonde data
    full_day={}
    #loop over the files
    for i, file in enumerate(file_list):
        #read the data
        file_path=f'{site_dir}/{file}'
        ds=format_function(file_path)
        #add the time
        ds.attrs['rs_time'] = times_list[i]
        #add the dataset to the dictionary with the corresponding time as the key
        full_day[times_list[i]] = ds
    return full_day

## Masts

def format_Cendrosa_obs(filename):
    ds = xr.open_mfdataset(filename)
    ds = ds.assign_coords(time_decimal=ds.time.dt.hour + ds.time.dt.minute / 60)
    #add name and plot color
    ds.attrs['name'] = 'obs'
    ds.attrs['plot_color'] = 'black'
    ds.attrs['linestyle']='-'

    rename_dict = {
        #lmdz vars
        'ta_2':'t2m',
        'shf_1':'sens',
        'lhf_1':'flat',
        'hur_2':'rh2m',
        'hus_2':'q2m',
        'ws_2':'wind_speed_10m',
        'wd_2':'wind_direction_10m',
        'pa':'psol',
        'swup':'SWupSFC',
        'lwup':'LWupSFC',
        'swd':'SWdnSFC',
        'lwd':'LWdnSFC',
        'rain_cumul':'precip',

        'ta_3':'t10m',
        'ta_4':'t25m',
        'ta_5':'t50m',
        'hur_3':'rh10m',
        'hur_4':'rh25m',
        'hur_5':'rh50m',
        'hus_3':'q10m',
        'hus_4':'q25m',
        'hus_5':'q50m',
        'shf_2':'sens25m',
        'shf_3':'sens50m',
        'lhf_2':'flat50m',
        
        #orc vars
        'soil_temp_1':'soil_temp_06',
        'soil_temp_2':'soil_temp_12',
        'soil_heat_flux':'Qg',

    }
    ds = ds.rename(rename_dict)

    #change temperatures to K
    # ds['t2m'] = ds['t2m'] + 273.15
    # ds['t2m'].attrs['units'] = 'K'
    # ds['t10m'] = ds['t10m'] + 273.15
    # ds['t10m'].attrs['units'] = 'K'
    # ds['t25m'] = ds['t25m'] + 273.15
    # ds['t25m'].attrs['units'] = 'K'
    # ds['t50m'] = ds['t50m'] + 273.15
    # ds['t50m'].attrs['units'] = 'K'
    #change sign of Qg
    ds['Qg'] = -ds['Qg']
    #make netrad fluxes
    ds['lwnet'] = ds['LWdnSFC'] - ds['LWupSFC']
    ds['lwnet'].attrs['units'] = 'W/m2'
    ds['swnet'] = ds['SWdnSFC'] - ds['SWupSFC']
    ds['swnet'].attrs['units'] = 'W/m2'
    #humidity in top10cm
    alpha_SM=0.5
    ds['mrsos']=(alpha_SM * ds['soil_moisture_1'] + (1-alpha_SM) * ds['soil_moisture_2']) * 100
    ds['mrsos'].attrs['units'] = 'mm'

    ds['soil_moisture_1'] = ds['soil_moisture_1'] * 100
    ds['soil_moisture_2'] = ds['soil_moisture_2'] * 100
    ds['soil_moisture_3'] = ds['soil_moisture_3'] * 100

    # #precip in mm/d (initialy in mm/30mn)
    # ds['precip'] = ds['precip'] * 2 * 24
    # ds['precip'].attrs['units'] = 'mm/d'

    return(ds)

def format_ElsPlans_obs(ds, start_day):
    #add name and plot color
    ds.attrs['name'] = 'obs'
    ds.attrs['plot_color'] = 'black'
    ds.attrs['linestyle']='--'

    rename_dict = {
        'HOUR_time':'time',
        #lmdz vars
        'PRES_subsoil':'psol',
        'TEMP_2m':'t2m',
        'RHUM_2mB':'rh2m',
        'UTOT_10m':'wind_speed_10m',
        'DIR_10m':'wind_direction_10m',
        'SWUP_rad':'SWupSFC',
        'LWUP_rad':'LWupSFC',
        'SWDN_rad':'SWdnSFC',
        'LWDN_rad':'LWdnSFC',
        'RAIN_subsoil' : 'precip',
        'TEMP_50m': 't50m',
        'RHUM_50m': 'rh50m',
        'TEMP_25m': 't25m',
        'RHUM_25m': 'rh25m',
        'TEMP_10m': 't10m',

        #orc vars
        'SFLXA_subsoil':'Qg',
        'ST01_subsoil':'soil_temp_01',
        'ST04_subsoil':'soil_temp_03',
        'ST10_subsoil':'soil_temp_12',
    }
    ds = ds.rename(rename_dict)
    # ds = ds.assign_coords(time_decimal=ds.time.dt.hour + ds.time.dt.minute / 60)

    #convert time to hour:mn:s
    ds['time'] = pd.to_datetime(start_day) + pd.to_timedelta(ds['time'], unit='h')

    #add coord time_decimal for diurnal cycle
    ds = ds.assign_coords(time_decimal=ds['time'].dt.hour + ds['time'].dt.minute / 60)

    #remove all values of 1e11
    ds = ds.where(ds != 1e11)
    # change temperatures to K
    # ds['t2m'] = ds['t2m'] + 273.15
    # ds['t2m'].attrs['units'] = 'K'
    # ds['t10m'] = ds['t10m'] + 273.15
    # ds['t10m'].attrs['units'] = 'K'
    # ds['t25m'] = ds['t25m'] + 273.15
    # ds['t25m'].attrs['units'] = 'K'
    # ds['t50m'] = ds['t50m'] + 273.15
    # ds['t50m'].attrs['units'] = 'K'
    #get turbulent fluxes
    # Latent heat of vaporization of water in J/kg
    lambda_v = 2.5e6
    ds['flat'] = ds['WQ_2m'] * lambda_v
    ds['flat'].attrs['units'] = 'W/m2'
    # air density in kg/m3 and specific heat capacity of air in J/kg/K
    rho=1.225
    cp=1004.67
    ds['sens'] = ds['WT_2m'] * rho * cp
    ds['sens'].attrs['units'] = 'W/m2'
    # make net rad fluxes
    ds['lwnet'] = ds['LWdnSFC'] - ds['LWupSFC']
    ds['lwnet'].attrs['units'] = 'W/m2'
    ds['swnet'] = ds['SWdnSFC'] - ds['SWupSFC']
    ds['swnet'].attrs['units'] = 'W/m2'
    #change sign of Qg
    ds['Qg'] = -ds['Qg']
    #humidity in top10cm
    # alpha_SM=0.5
    # ds['mrsos']=(alpha_SM * ds['SM02_subsoil'] + (1-alpha_SM) * ds['SM10_subsoil']) * 10
    ds['mrsos'] = ds['SM10_subsoil']
    ds['mrsos'].attrs['units'] = 'mm'
    #add specific humidity
    ds = add_specific_humidity_2m(ds, t2m_in_kelvin=False, psol_Pa=False)

    return(ds)