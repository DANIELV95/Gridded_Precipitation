# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:05:32 2023

@author: HIDRAULICA-Dani
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patheffects as pe
import matplotlib as mpl
import proplot as pplt
import geopandas as gpd
from osgeo import gdal, osr
import rasterio
from rasterio import plot as rasterplot
import seaborn as sns

datasets_all = os.listdir('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/PRECIP/')

periods_dict = {0:'1981-1991', 1:'1991-2001', 2:'2001-2011', 3:'2011-2021'}
timesteps = ['d', 'm', 'mmax', 'y', 'ymax']
timesteps_names = {'d':'Daily', 'm':'Monthly', 'mmax':'Max Monthly', 'y':'Yearly', 'ymax':'Max Yearly'}

errors_list = ['nashsutcliffe', 'kge', 'rsr', 'pbias']
errors_dict = {'nashsutcliffe':'NSE', 'kge':'KGE', 'rsr':'RSR', 'pbias':'PBIAS'}
errors_colors = {'nashsutcliffe':'darkred', 'kge':'darkblue', 'rsr':'darkgreen', 'pbias':'orange'}

df_periods = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/len_ests.csv', index_col='Unnamed: 0')
ests25 = df_periods[df_periods['Am']>25].index

ests = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/Catalogo_estaciones.csv', encoding='latin-1')
ests.set_index('Unnamed: 0', inplace=True)
ests.index.name = None
ests.sort_index(inplace=True)

ests_loc = ests[['LON', 'LAT']][ests.index.isin(ests25)]

ds_order = ['Pchirps', 'Pdaymet', 'Pera', 'Pgldas', 'Pgpm', 'Plivneh', 'Ppersiann']
ds_names_dict = {'Pchirps':'CHIRPS', 'Pdaymet':'Daymet', 'Pera':'ERA5L', 'Pgldas':'GLDAS', 'Pgpm':'IMERG', 'Plivneh':'Livneh', 'Ppersiann':'PERSIANN'}
ds_names = ['CHIRPS', 'Daymet', 'ERA5L', 'GLDAS', 'IMERG', 'Livneh', 'PERSIANN']

shpMx = "D:/DANI/2023/TEMA_TORMENTAS/DATOS/GEE/SHP/Border_Mx.shp"
sMx = gpd.read_file(shpMx, bbox=None, mask=None, rows=None)
sMx.crs = {'init':"epsg:4326"}

data_est = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/GIS/Stations_Data_All.txt', index_col=['Field1'])
data_est.drop(['FID', 'NOMBRE', 'MUNICIPIO', 'ESTADO', 'ORG_CUENCA', 'CUENCA', 'SUBCUENCA',
               'INICIO', 'FIN', 'ALTURA', 'LAT', 'LON', 'CLASE_H', 'ENTIDAD', 'NOMBRE_12'], axis=1, inplace=True)
data_est.columns = ['Elevation', 'Z_Class', 'Climate_ID', 'Climate_Name', 'State_ID', 'State_Name', 'Basin_ID', 'Basin_Name']
data_est.index.name = 'id'

data_est.loc[data_est['Elevation']<100, 'Z_Class'] = 1
data_est.loc[((data_est['Elevation']>100) & (data_est['Elevation']<600)), 'Z_Class'] = 2
data_est.loc[((data_est['Elevation']>=600) & (data_est['Elevation']<1500)), 'Z_Class'] = 3
data_est.loc[((data_est['Elevation']>=1500) & (data_est['Elevation']<2000)), 'Z_Class'] = 4
data_est.loc[data_est['Elevation']>=2000, 'Z_Class'] = 5


# Analize All Data

# df_ALL = pd.merge(df_all_ts, data_est, on='id')
# df_ALL.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/All_Data_In_One.csv', encoding='latin-1', index=False)

dtypes_dict = {'id': int, 'timestep': object, 'period': object, 'dataset': object, 'nashsutcliffe': float,
               'nashsutcliffe_class': object, 'nashsutcliffe_outlier': bool, 'nashsutcliffe_alpha': float,
               'kge': float, 'kge_class': object, 'kge_outlier': bool, 'kge_alpha': float, 'rsr': float,
               'rsr_class': object, 'rsr_outlier': bool, 'rsr_alpha': float, 'pbias': float, 'pbias_class': object,
               'pbias_outlier': bool, 'pbias_alpha': float, 'Elevation': float, 'Z_Class': int,
               'Climate_ID': int, 'Climate_Name': object, 'State_ID': int, 'State_Name': object,
               'Basin_ID': int, 'Basin_Name': object}


ds_names_abbr = [ds_names[i][:3].upper() for i in range(len(ds_names))]

cmap_r2b = LinearSegmentedColormap.from_list('r2b',["red", "yellow", "cyan", "darkblue"], N=256)
cmap_b2r = LinearSegmentedColormap.from_list('r2b_r',["darkblue", "cyan", "yellow", "red"], N=256)
cmap_rbg = LinearSegmentedColormap.from_list('rbg',["red", "magenta", "darkblue", "cyan", "green"], N=256)
# cmap_rbg = LinearSegmentedColormap.from_list('rbg',["red", "darkblue", "green"], N=256)
cmap_r = LinearSegmentedColormap.from_list('red',["red", "red"], N=256)
# plt.cm.get_cmap("jet_r")

cmap_r2dr = LinearSegmentedColormap.from_list('r2k',["red", "black"], N=256)
cmap_g2dg = LinearSegmentedColormap.from_list('g2k',["green", "black"], N=256)
cmap_dr2r = LinearSegmentedColormap.from_list('k2r',["black", "red"], N=256)
cmap_dg2g = LinearSegmentedColormap.from_list('k2g',["black", "green"], N=256)

# cmap_r2dr = LinearSegmentedColormap.from_list('r2k',["red", "darkred"], N=256)
# cmap_g2dg = LinearSegmentedColormap.from_list('g2k',["green", "darkgreen"], N=256)
# cmap_dr2r = LinearSegmentedColormap.from_list('k2r',["darkred", "red"], N=256)
# cmap_dg2g = LinearSegmentedColormap.from_list('k2g',["darkgreen", "green"], N=256)

colors1 = cmap_dr2r(np.linspace(0., 1, 11))
colors2 = cmap_r2b(np.linspace(0, 1, 256))
# # combine them and build a new colormap
colors12 = np.vstack((colors1, colors2))
cmap_combined = LinearSegmentedColormap.from_list('combined', colors12)
fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cmap_rbg)

cmaps = {'nashsutcliffe':cmap_r2b, 'kge':cmap_r2b, 'rsr':cmap_b2r, 'pbias':cmap_rbg}
# vmin_dict = {'kge':-0.41, 'nashsutcliffe':0, 'correlationcoefficient':-1, 'pbias':-25, 'rsr':0, 'rmse':0, 'FB':0, 'POD':0, 'FAR':0, 'TS':0, 'ETS':0}
# vmax_dict = {'kge':1, 'nashsutcliffe':1, 'correlationcoefficient':1, 'pbias':25, 'rsr':1, 'rmse':25, 'FB':1, 'POD':1, 'FAR':1, 'TS':1, 'ETS':1}

vmin_dict = {'kge':-0.4, 'nashsutcliffe':0, 'correlationcoefficient':-1, 'pbias':-50, 'rsr':0, 'rmse':0, 'FB':0, 'POD':0, 'FAR':0, 'TS':0, 'ETS':0}
vmax_dict = {'kge':1, 'nashsutcliffe':1, 'correlationcoefficient':1, 'pbias':50, 'rsr':1.4, 'rmse':25, 'FB':1, 'POD':1, 'FAR':1, 'TS':1, 'ETS':1}


df = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/All_Data_In_One.csv', encoding='latin-1')
df = df.astype(dtypes_dict)
# df.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/All_Data_In_One.csv', encoding='latin-1', index=False)
# df.columns

df.loc[df['Elevation']<100, 'Z_Range'] = '<100'
df.loc[((df['Elevation']>=100) & (df['Elevation']<600)), 'Z_Range'] = '100-600'
df.loc[((df['Elevation']>=600) & (df['Elevation']<1500)), 'Z_Range'] = '600-1500'
df.loc[((df['Elevation']>=1500) & (df['Elevation']<2000)), 'Z_Range'] = '1500-2000'
df.loc[df['Elevation']>=2000, 'Z_Range'] = '>2000'

basins = df[['Basin_ID', 'Basin_Name']].drop_duplicates().sort_values(by='Basin_ID').reset_index(drop=True)
elevations = df[['Z_Class', 'Z_Range']].drop_duplicates().sort_values(by='Z_Class').reset_index(drop=True)
climates = df[['Climate_ID', 'Climate_Name']].drop_duplicates().sort_values(by='Climate_ID').reset_index(drop=True)
states = df[['State_ID', 'State_Name']].drop_duplicates().sort_values(by='State_ID').reset_index(drop=True)



