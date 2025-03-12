import numpy as np
from osgeo import gdal
from osgeo import osr
import os
import pyresample as pr
from satpy import Scene
import matplotlib.pyplot as plt

#filename = 'MSG4-SEVI-MSG15-0100-NA-20221226121243.443000000Z-NA.nat'
#filename = 'MSG4-SEVI-MSG15-0100-NA-20221227051243.714000000Z-NA.nat'
# define reader
reader = 'seviri_l1b_native'
# read the file
#scn = Scene(filenames = {reader:['/scratchu/aborella/SEVIRI/' + filename]})
# extract data set names
#dataset_names = scn.all_dataset_names()
# print available datasets
#print('\n'.join(map(str, dataset_names)))

def get_filename(time_snapshot):

    if time_snapshot == np.datetime64('2022-12-27T00'):
        filename = 'MSG4-SEVI-MSG15-0100-NA-20221226235743.034000000Z-NA.nat'
    elif time_snapshot == np.datetime64('2022-12-27T01'):
        filename = 'MSG4-SEVI-MSG15-0100-NA-20221227005742.584000000Z-NA.nat'
    elif time_snapshot == np.datetime64('2022-12-27T02'):
        filename = 'MSG4-SEVI-MSG15-0100-NA-20221227015743.341000000Z-NA.nat'
    elif time_snapshot == np.datetime64('2022-12-27T03'):
        filename = 'MSG4-SEVI-MSG15-0100-NA-20221227025742.896000000Z-NA.nat'
    elif time_snapshot == np.datetime64('2022-12-27T04'):
        filename = 'MSG4-SEVI-MSG15-0100-NA-20221227035742.453000000Z-NA.nat'
    elif time_snapshot == np.datetime64('2022-12-27T05'):
        filename = 'MSG4-SEVI-MSG15-0100-NA-20221227045743.823000000Z-NA.nat'
    elif time_snapshot == np.datetime64('2022-12-27T06'):
        filename = 'MSG4-SEVI-MSG15-0100-NA-20221227055743.386000000Z-NA.nat'
    else:
        raise ValueError(f'{time_snapshot} not linked')

    print(f'Linking {time_snapshot} with {filename}')

    return filename


# create some information on the reference system
area_id = 'France'
description = 'Geographical Coordinate System clipped on France'
proj_id = 'France'
# specifing some parameters of the projection
proj_dict = {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84'}
# calculate the width and height of the aoi in pixels
llx = -13. # lower left x coordinate in degrees
lly = 39. # lower left y coordinate in degrees
urx = 17. # upper right x coordinate in degrees
ury = 59. # upper right y coordinate in degrees
#llx = -25. # lower left x coordinate in degrees
#lly = 31. # lower left y coordinate in degrees
#urx = 29. # upper right x coordinate in degrees
#ury = 67. # upper right y coordinate in degrees
resolution = 0.005 # target resolution in degrees
# calculating the number of pixels
width = int((urx - llx) / resolution)
height = int((ury - lly) / resolution)
area_extent = (llx,lly,urx,ury)
# defining the area
area_def = pr.geometry.AreaDefinition(area_id, proj_id, description, proj_dict, width, height, area_extent)
#print(area_def)


def nat2tif(filename, calibration, area_def, dataset, reader, dtype, radius, epsilon, nodata):
    # open the file
    scn = Scene(filenames = {reader: ['/scratchu/aborella/OBS/MSG/202212/' + filename]})
    # let us check that the specified data set is actually available
    scn_names = scn.all_dataset_names()
    # raise exception if dataset is not present in available names
    if dataset not in scn_names:
        raise Exception('Specified dataset is not available.')
    # we need to load the data, different calibration can be chosen
    scn.load([dataset], calibration=calibration)
    # let us extract the longitude and latitude data
    lons, lats = scn[dataset].area.get_lonlats()
    # now we can apply a swath definition for our output raster
    swath_def = pr.geometry.SwathDefinition(lons=lons, lats=lats)
    # and finally we also extract the data
    values = scn[dataset].values
    # we will now change the datatype of the arrays
    # depending on the present data this can be changed
    lons = lons.astype(dtype)
    lats = lats.astype(dtype)
    values = values.astype(dtype)
    # now we can already resample our data to the area of interest
    values = pr.kd_tree.resample_nearest(swath_def, values,
                                               area_def,
                                               radius_of_influence=radius, # in meters
                                               epsilon=epsilon,
                                               fill_value=False)
    # let us join our filename based on the input file's basename
    outname = os.path.join('/scratchu/aborella/SEVIRI', os.path.basename(filename)[:-4] + '_' + str(dataset) + '.tif')
    # now we define some metadata for our raster file
    cols = values.shape[1]
    rows = values.shape[0]
    pixelWidth = (area_def.area_extent[2] - area_def.area_extent[0]) / cols
    pixelHeight = (area_def.area_extent[1] - area_def.area_extent[3]) / rows
    originX = area_def.area_extent[0]
    originY = area_def.area_extent[3]
    # here we actually create the file
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(outname, cols, rows, 1)
    # writing the metadata
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    # creating a new band and writting the data
    outband = outRaster.GetRasterBand(1)
    outband.SetNoDataValue(nodata) #specified no data value by user
    outband.WriteArray(np.array(values)) # writting the values
    outRasterSRS = osr.SpatialReference() # create CRS instance
    outRasterSRS.ImportFromEPSG(4326) # get info for EPSG 4326
    outRaster.SetProjection(outRasterSRS.ExportToWkt()) # set CRS as WKT
    # clean up
    outband.FlushCache()
    outband = None
    outRaster = None

    return


#dataset = 'HRV' # useless
#dataset = 'IR_016' # useless
#dataset = 'IR_039' # useless
#dataset = 'IR_087' # ok
#dataset = 'IR_097'
#dataset = 'IR_108'
#dataset = 'IR_120'
#dataset = 'IR_134'
#dataset = 'VIS006' # useless
#dataset = 'VIS008' # useless
#dataset = 'WV_062' # useless
#dataset = 'WV_073'


def get_data(time_snapshot, dataset):

    filename = get_filename(time_snapshot)

    outname = filename[:-4] + '_' + str(dataset) + '.tif'
    
    if not os.path.exists('/scratchu/aborella/SEVIRI/' + outname):
        nat2tif(filename = filename,
                calibration = 'radiance',
                area_def = area_def,
                dataset = dataset,
                reader = reader,
                dtype = 'float32',
                radius = 16000,
                epsilon = .5,
                nodata = -3.4E+38)
    
    
    ds = gdal.Open('/scratchu/aborella/SEVIRI/' + outname)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()

    lons, lats = area_def.get_lonlats()

    return lons, lats, data
