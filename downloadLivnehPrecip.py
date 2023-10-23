# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:11:45 2023

@author: HIDRAULICA-Dani
"""

import os
import pandas as pd
import numpy as np
import requests
import netCDF4 as nc
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
disable_warnings(InsecureRequestWarning)

os.chdir('D:/DANI/2023/TEMA_TORMENTAS/DATOS/LIVNEH')

years = np.arange(1950,2014).astype(str)
months = [x.zfill(2) for x in np.arange(1,13).astype(str)]

mxbounds = [[-118.36708718265577,14.532861395699475], [-86.71060890823804,32.71675888570315]]
lon1 = mxbounds[0][0]
lon2 = mxbounds[1][0]
lat1 = mxbounds[0][1]
lat2 = mxbounds[1][1]

#Download Livneh Precipitation Data
for year in years:
    # year = '1950'
    for month in months:
        # month = '02'
        print(year, month)
        file = 'livneh_NAmerExt_15Oct2014.'+year+month+'.nc'
        url = 'https://www.ncei.noaa.gov/data/oceans/ncei/archive/data/0129374/daily/'+file
        response = requests.get(url)
        open(file, 'wb').write(response.content)
        
        #Crop nc file to Mexico and extract Precipitation data
        liv = nc.Dataset(file)
        lat_all = liv.variables['lat'][:].data
        lon_all = liv.variables['lon'][:].data
        
        lat_mask = ((lat_all>lat1) & (lat_all<lat2))
        lon_mask = ((lon_all>lon1) & (lon_all<lon2))
        # s = pd.Series(lat_mask)
        # s = pd.Series(lon_mask)
        # grp = s.eq(False).cumsum()
        # arr = grp.loc[s.eq(True)].groupby(grp).apply(lambda x: [x.index.min(), x.index.max()])
        
        lat = lat_all[lat_mask]
        lon = lon_all[lon_mask]
        time = liv.variables['time'][:].data
        precip = liv.variables['Prec'][:].data[:,0:289,106:613]
        precip[precip==1e+20] = np.nan
        
        #Create netCDF4 file
        # f = nc.Dataset(file,'w', format='NETCDF4')
        
        #Create dimensions of the nc file (name and length)
        f.createDimension('lon', len(lon))
        f.createDimension('lat', len(lat))
        f.createDimension('time', len(time))
        
        #Create variables of the nc file (name, data type, dimensions)
        longitude = f.createVariable('lon', 'd', 'lon', fill_value='NaN')
        latitude = f.createVariable('lat', 'd', 'lat', fill_value='NaN')
        Time = f.createVariable('time', 'i8', 'time')
        precipitation = f.createVariable('precipitation', 'f', ('time', 'lat', 'lon'), fill_value='NaN')
        
        #Add data to the variables
        longitude[:] = lon
        latitude[:] = lat
        Time[:] = time
        precipitation[:] = precip
        
        #Add attributes to the variables
        Time.units = 'days since 1900-01-01 00:00:00'
        Time.calendar = 'standard'
        
        #Close nc file
        f.close()
        print('Done')
        

#Combine time and precip data in array
liv = nc.Dataset('livneh_NAmerExt_15Oct2014.195001.nc')
lat = liv.variables['lat'][:].data
lon = liv.variables['lon'][:].data
time_all = []
precip_all = []

for year in years:
    # year = '1950'
    for month in months:
        # month = '02'
        print(year, month)
        file = 'livneh_NAmerExt_15Oct2014.'+year+month+'.nc'
        liv = nc.Dataset(file)
        time = liv.variables['time'][:].data
        precip = liv.variables['precipitation'][:].data
        
        time_all.append(time)
        precip_all.append(precip)

time_flat = np.array([], dtype='int64')
for i in range(len(time_all)):
    time_flat = np.append(time_flat, time_all[i])

precip_flat = np.empty((len(time_flat),289,507), dtype='float32')
k = 0
for i in range(len(precip_all)):
    for j in range(len(precip_all[i])):
        print(k)
        precip_flat[k] = precip_all[i][j]
        k += 1

#Create netCDF4 file with all data
f = nc.Dataset('livneh_precip_1950-2013.nc','w', format='NETCDF4')

#Create dimensions of the nc file (name and length)
f.createDimension('lon', len(lon))
f.createDimension('lat', len(lat))
f.createDimension('time', len(time_flat))

#Create variables of the nc file (name, data type, dimensions)
longitude = f.createVariable('lon', 'd', 'lon', fill_value='NaN')
latitude = f.createVariable('lat', 'd', 'lat', fill_value='NaN')
Time = f.createVariable('time', 'i8', 'time')
precipitation = f.createVariable('precipitation', 'f', ('time', 'lat', 'lon'), fill_value='NaN')

#Add data to the variables
longitude[:] = lon
latitude[:] = lat
Time[:] = time_flat
precipitation[:] = precip_flat

#Add attributes to the variables
Time.units = 'days since 1900-01-01 00:00:00'
Time.calendar = 'standard'

#Close nc file
f.close()