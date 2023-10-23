# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:41:47 2023

@author: HIDRAULICA-Dani
"""

#Download ERA5-Land Precipitation for Mexico


import os
import cdsapi

os.chdir('D:/DANI/2023/TEMA_TORMENTAS/DATOS/ERA5_Land/')

start = 1950
end = 2023

for i in range(start, end+1):
    for j in range(1,13):
        # i = 2000
        # j = 1
        year = str(i)
        month = str(j).zfill(2)
        print(year, month)
        
        if not os.path.exists('./precip_era5_land_mexico_'+year+'_'+month+'.nc'):
    
            c = cdsapi.Client()
            
            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': [
                        'total_precipitation',
                    ],
                    'year': year,
                    'month': month,
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area': [
                        32.9, -118.3, 13.5,
                        -86.2,
                    ],
                    'format': 'netcdf',
                },
                './precip_era5_land_mexico_'+year+'_'+month+'.nc')
    

# 'year': [str(i) for i in range(start,end+1)]