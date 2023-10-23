# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:52:51 2023

@author: HIDRAULICA-Dani
"""

#Convert NetCDF to ASCI files

import os
import pandas as pd
import numpy as np
from osgeo import gdal, osr
import netCDF4 as nc
import datetime
import matplotlib.pyplot as plt

# import math
# import matplotlib
# import matplotlib.patches as mpatches
# from matplotlib.ticker import FormatStrFormatter
# from matplotlib import cm


os.chdir('D:/DANI/2023/TESIS_QUIQUE/GIS/Raster/LIVNEH/')
os.listdir()

#Set spatial reference
sr = osr.SpatialReference()
sr.ImportFromEPSG(4326)

#Read tif files and get coordinates of each pixel
ds = gdal.Open("Livneh1.tif")
arr = np.array(ds.GetRasterBand(1).ReadAsArray())
width = ds.RasterXSize
height = ds.RasterYSize
gt = ds.GetGeoTransform()
minx = gt[0]
miny = gt[3] + width*gt[4] + height*gt[5]
maxx = gt[0] + width*gt[1] + height*gt[2]
maxy = gt[3]
resx = gt[1]
resy = gt[5]

ds.GetGeoTransform()

lon = np.arange(minx, maxx, resx)
lat = np.arange(maxy, miny, resy)
ilon = np.arange(0,318)
ilat = np.arange(0,183)

#Find closer pixel to specific lat and lon
# lon_x = -108.35
# lat_y = 27.65
# lon_id = (np.abs(lon_x-lon)).argmin()
# lat_id = (np.abs(lat_y-lat)).argmin()

#Get info about tiff file
# !gdalinfo Livneh1.tif

def write_geotiff(filename, arr, in_ds):
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    elif arr.dtype == np.float64:
        arr_type = gdal.GDT_Float64
    else:
        arr_type = gdal.GDT_Int32

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)

livneh_nc = nc.Dataset('D:/DANI/2023/TEMA_TORMENTAS/DATOS/LIVNEH/livneh_precip_1950-2013_mask.nc')
precip_liv = livneh_nc.variables['precipitation'][:].data
time_liv = livneh_nc.variables['time'][:].data
lat_liv = livneh_nc.variables['lat'][:].data
lon_liv = livneh_nc.variables['lon'][:].data

start = datetime.datetime(1900,1,1)
delta = datetime.timedelta(days=1)
dates_liv = start + delta*time_liv

# plt.matshow(precip_liv[0])

for i in range(len(precip_liv)):
    write_geotiff('./TIFF/'+dates_liv[i].strftime('%Y%m%d')+'.tif', precip_liv[i], ds)
    ds1 = gdal.Open('./TIFF/'+dates_liv[i].strftime('%Y%m%d')+'.tif')
    gdal.Translate('./ASC/'+dates_liv[i].strftime('%Y%m%d')+'.asc', ds1)


help(gdal.Translate)
