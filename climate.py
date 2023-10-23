# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:41:08 2023

@author: HIDRAULICA-Dani
"""

import os
import pandas as pd
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from osgeo import gdal

os.chdir('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CLIMAS/Map_KG-Global/')
os.listdir()

coord = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CLIMAS/Map_KG-Global/KG_1986-2010_5m.csv')
Lon = coord['lon'].unique()
Lat = coord['lat'].unique()

# lons = []
# lats = []
# kgs = [[],[]]

# for row, i in coord.iterrows():
#     print(row, i)
#     lon = i[0]
#     lat = i[1]
#     kg = i[2]
    
#     if (~np.isin(lon,lons)):
#         lons.append(lon)
    
#     if (~np.isin(lat,lats)):
#         lats.append(lat)
    
#     kgs.append(kg)
    
KG_grid = coord.pivot(index='lat', columns='lon', values='KG')
KG_grid.index.name = None
KG_grid.columns.name = None
KG_grid = KG_grid.sort_index(ascending=False)

KG_arr = KG_grid.values
# KG_arr[np.isnan(KG_arr)] = -9999

rows = KG_grid.index.values
cols = KG_grid.columns.values

KG_arr.shape

N = KG_grid.index[0]
# KG_grid.index[-1]
W = KG_grid.columns[0]
len(KG_grid.columns)

KG_grid.columns[1]-KG_grid.columns[2]
KG_grid.index[1]-KG_grid.index[2]


ras_filename = './KG_climate.tif'
new_transform = rasterio.transform.from_origin(west=W, north=N, xsize=0.08333333, ysize=0.08333333)


driver = gdal.GetDriverByName('GTiff')
driver.Register()
outds = driver.Create(ras_filename, xsize=KG_arr.shape[1], ysize=KG_arr.shape[0],
                    bands=1, eType=gdal.GDT_Float64)
outds.SetGeoTransform(new_transform)

driver = None
outds = None
raster = None




new_transform = rasterio.transform.from_origin(west=W, north=N, xsize=KG_arr.shape[1], ysize=KG_arr.shape[0])
new_transform

new_transform = rasterio.transform.from_origin(west=W, north=N, xsize=0.08333333, ysize=0.08333333)


rasterio.transform.xy(new_transform, rows, cols, zs=KG_arr, offset='center') #, **rpc_options)

plt.imshow(KG_arr)



KG = rasterio.open(
    './KG_climate.tif', 'w', 
    driver = 'GTiff',
    height = KG_arr.shape[0],
    width = KG_arr.shape[1],
    count = 1,
    nodata = -9999,
    dtype = KG_arr.dtype,
    crs = 4326,
    transform = new_transform,
    compress='lzw'
)

KG.close()

show(KG_arr)

KG_grid.values.dtype

show(KG_grid)

plt.scatter(coord['lon'], coord['lat'], s=1, marker=',')

lat.min()
lat.max()

lon.min()
lon.max()

len(coord)

len(lat)
len(lon)
