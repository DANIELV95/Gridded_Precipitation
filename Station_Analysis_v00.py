# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:40:19 2023

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

data_est.reset_index(inplace=True)

plt.hist(data_est['Z_Class'], bins=100)

#####################################################################################
#Outliers plot

#Set spatial reference
# sr = osr.SpatialReference()
# sr.ImportFromEPSG(4326)

#Read tif files and get coordinates of each pixel
# ds_mde = gdal.Open('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Raster/Slope1k.tif')
# arr = np.array(ds_mde.GetRasterBand(1).ReadAsArray())
# width = ds_mde.RasterXSize
# height = ds_mde.RasterYSize
# gt = ds_mde.GetGeoTransform()
# minx = gt[0]
# miny = gt[3] + width*gt[4] + height*gt[5]
# maxx = gt[0] + width*gt[1] + height*gt[2]
# maxy = gt[3]
# resx = gt[1]
# resy = gt[5]

# ds_mde.GetGeoTransform()

# lon = np.arange(minx, maxx, resx)
# lat = np.arange(maxy, miny, resy)
# ilon = np.arange(width)
# ilat = np.arange(height)
# tiff_extent = [minx, maxx, miny, maxy]


# cmap_r2dr = LinearSegmentedColormap.from_list('r2k',["red", "black"], N=256)
cmap_dr2r = LinearSegmentedColormap.from_list('k2r',["red", "magenta"], N=256)
cmap_g2dg = LinearSegmentedColormap.from_list('g2k',["yellow", "green"], N=256)
# cmap_dg2g = LinearSegmentedColormap.from_list('k2g',["white", "green"], N=256)

tiff = rasterio.open('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Raster/Slope1k_proj.tif')
# tiff = rasterio.open('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Raster/DEM1k.tif')
tiff_extent = [tiff.bounds[0], tiff.bounds[2], tiff.bounds[1], tiff.bounds[3]]

timesteps_pbias = ['d', 'mmax', 'ymax']
timesteps_pbias_names = ['Daily', 'Max Monthly', 'Max Yearly']

for error in errors_list:
    error = 'pbias'
    print(error)
    
for period in periods_dict:
    # period = 2
    print(periods_dict[period])

    fig, ax = pplt.subplots(ncols=3, nrows=7, share=False, figwidth=6, figheight=10, wspace=0.3, hspace=0.8)
    
    col = -1
    for timestep in timesteps_pbias:
        # timestep = 'mmax'
        print(timestep)
        col += 1
        row = -1
        
        error_df = pd.read_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/P_{periods_dict[period]}_E_{error}_T_{timestep}.csv', index_col=['Unnamed: 0'])
        df1 = pd.concat([ests_loc['LON'], ests_loc['LAT'], error_df], axis=1)
        df1.replace([np.inf, -np.inf], np.nan, inplace=True)
        df1.dropna(subset=error_df.columns, how='all', inplace=True)
        
        Q1 = df1[error_df.columns].quantile(0.25)
        Q3 = df1[error_df.columns].quantile(0.75)
        IQR = Q3 - Q1 #IQR is interquartile range.
        
        filter_lonlat = df1[['LON', 'LAT']].notnull()
        filter_outliers_low = df1[error_df.columns] <= Q1 - 1.5 * IQR
        filter_outliers_high = df1[error_df.columns] >= Q3 + 1.5 *IQR
    
        df_out_low = df1[pd.concat([filter_lonlat, filter_outliers_low], axis=1)]
        df_out_low.dropna(subset=error_df.columns, how='all', inplace=True)
        
        df_out_high = df1[pd.concat([filter_lonlat, filter_outliers_high], axis=1)]
        df_out_high.dropna(subset=error_df.columns, how='all', inplace=True)
        
        
        for ds in ds_order:
            # ds = ds_order[2]
            row += 1
            
            df_out1 = df_out_low[['LON', 'LAT', ds]].dropna(subset=[ds])
            df_out2 = df_out_high[['LON', 'LAT', ds]].dropna(subset=[ds])
            # print(df_out1[ds].min(), df_out2[ds].max())
            
            # fig, ax = pplt.subplots(nrows=2, ncols=2, sharey=False, figsize=(10, 8))
            sMx.boundary.plot(ax=ax[row,col], alpha=1, lw=0.25, color='k', zorder=2)
            # rasterplot.show(
            #     tiff,  # use tiff.read(1) with your data
            #     extent=tiff_extent,
            #     ax=ax[row,col],
            #     cmap='binary')
            df_plot_out1 = ax[row,col].scatter(df_out_low['LON'], df_out_low['LAT'], c=df_out_low[ds], cmap=cmap_dr2r, marker='o', mew=0.1,
                                               s=5, alpha=1, levels=pplt.arange(-200, 0, 20))
            df_plot_out2 = ax[row,col].scatter(df_out_high['LON'], df_out_high['LAT'], c=df_out_high[ds], cmap=cmap_g2dg, marker='o', mew=0.1,
                                               s=5, alpha=1, levels=pplt.arange(0, 200, 20))
            
            if row == 6:
                cbar = fig.colorbar(df_plot_out1, loc='b', col=(col+1), shrink=0.8)
                cbar = fig.colorbar(df_plot_out2, loc='b', col=(col+1), shrink=0.8)
                
    ax.format(grid=False,
            toplabels=timesteps_pbias_names,
            leftlabels=ds_names,
            xtickloc='none',
            xticklabelloc='none',
            ytickloc='none',
            yticklabelloc='none',
            xlabel='',
            ylabel='')
    
    ax[0, :].format(xtickloc='top', xticklabelloc='top')
    ax[-1, :].format(xtickloc='bottom', xticklabelloc='bottom')
    ax[:, 0].format(ytickloc='left', yticklabelloc='left')
    ax[:, -1].format(ytickloc='right', yticklabelloc='right')
    
    plt.figtext(0.01, 0.96, f'PBIAS\n{periods_dict[period]}', size='large', weight='bold')
    # fig.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Outliers/Outliers.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    fig.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Outliers/Outliers_{error}_{periods_dict[period]}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    plt.show()
    plt.close()

###########################################################################################
#Compare 3 errors in one plot

cols_order = ['id', 'Pchirps', 'Pdaymet', 'Pera', 'Pgldas', 'Pgpm', 'Plivneh', 'Ppersiann']
alpha_class = {'Unsatisfactory':0.25, 'Satisfactory':0.5, 'Good':0.75, 'Very Good':1}
pallete_class = {'Unsatisfactory':'darkred', 'Satisfactory':'cyan', 'Good':'blue', 'Very Good':'darkblue'}
ds_classes = ['nashsutcliffe_class', 'kge_class', 'rsr_class', 'pbias_class']

df_all_ts = pd.DataFrame()
for timestep in timesteps:
    # timestep = 'ymax'
    print(timestep)
    
    df_all_ds = pd.DataFrame()
    for ds in ds_order:
        # ds = ds_order[-1]
        df_all = pd.DataFrame()
        for period in periods_dict:
            # period = 2
            
            df = pd.DataFrame()
            for error in errors_list:
                #  error = 'pbias'
                error_df = pd.read_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/P_{periods_dict[period]}_E_{error}_T_{timestep}.csv')
                error_df.columns = cols_order
                
                df['id'] = error_df['id']
                df['timestep'] = timesteps_names[timestep]
                df['period'] = periods_dict[period]
                df['dataset'] = ds_names_dict[ds]
                df[f'{error}'] = error_df[ds]
                
                #Class
                if error == 'nashsutcliffe':
                    error_df[error_df>=0.75] = 4
                    error_df[((error_df>=0.65)&(error_df<0.75))] = 3
                    error_df[((error_df>=0.5)&(error_df<0.65))] = 2
                    error_df[error_df<0.5] = 1
                elif error == 'kge':
                    error_df[error_df>=0.75] = 4
                    error_df[((error_df>=0.5)&(error_df<0.75))] = 3
                    error_df[((error_df>=0.3)&(error_df<0.5))] = 2
                    error_df[error_df<0.3] = 1
                elif error == 'rsr':
                    error_df[error_df>0.7] = 1
                    error_df[((error_df>0.6)&(error_df<=0.7))] = 2
                    error_df[((error_df>0.5)&(error_df<=0.6))] = 3
                    error_df[error_df<=0.5] = 4
                elif error == 'pbias':
                    error_df[abs(error_df)<=10] = 4
                    error_df[((abs(error_df)>10)&(abs(error_df)<=15))] = 3
                    error_df[((abs(error_df)>15)&(abs(error_df)<=25))] = 2
                    error_df[abs(error_df)>25] = 1
                
                df[f'{error}_class'] = error_df[ds]
                                
                #Alpha
                df[f'{error}_alpha'] = (df[f'{error}_class']/4)**0.3
                
                #Outliers
                Q1 = df[f'{error}'].quantile(0.25)
                Q3 = df[f'{error}'].quantile(0.75)
                IQR = Q3 - Q1
                filter_outliers = (df[f'{error}'] >= Q1 - 1.5 * IQR) & (df[f'{error}'] <= Q3 + 1.5 *IQR)
                df[f'{error}_outlier'] = ~filter_outliers
                
            df_all = pd.concat([df_all, df], axis=0)
        df_all_ds = pd.concat([df_all_ds, df_all], axis=0)
    df_all_ts = pd.concat([df_all_ts, df_all_ds], axis=0)
    
df_all_ts.dropna(how='any', inplace=True)
df_all_ts.sort_values(by='nashsutcliffe', ascending=False, inplace=True)
df_all_ts.loc[:,ds_classes] = df_all_ts.loc[:,ds_classes].replace([1, 2, 3, 4], ['Unsatisfactory', 'Satisfactory', 'Good', 'Very Good'])
# df.replace([1, 2, 3, 4], ['Unsatisfactory', 'Satisfactory', 'Satisfactory', 'Satisfactory'], inplace=True)

# df_all_ts.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/All/Errors_all.csv', encoding='latin-1')

# NSE vs RSR
plt.scatter(df_all_ts['nashsutcliffe'], df_all_ts['rsr'], s=1)
plt.hlines([0.5,0.6,0.7], 0, 1)
plt.vlines([0.5,0.65,0.75], 0, 1)
plt.xlabel('NSE')
plt.ylabel('RSR')
plt.xlim(0,1)
plt.ylim(0,1)

# NSE vs PBIAS
plt.scatter(df_all_ts['nashsutcliffe'], df_all_ts['pbias'], s=1)
plt.hlines([-25,25], 0, 1)
plt.vlines([0.5,0.65,0.75], -100, 100)
plt.xlabel('NSE')
plt.ylabel('PBIAS')
plt.xlim(-1,1)
plt.ylim(-100,100)

kge_lims = [0.3, 0.5, 0.75]
nse_lims = [0.5, 0.65, 0.75]
pbias_lims = [10,15,25]
rsr_lims = [0.5,0.6,0.7]

# KGE vs PBIAS vs NSE_class
# ['darkblue', 'b', 'c', 'darkred']
fig, ax = plt.subplots(figsize=(8,8))
sns.scatterplot(x='kge', y='pbias', hue='nashsutcliffe_class', data=df_all_ts, ax=ax, palette=pallete_class, 
                s=1, linewidth=0, alpha=df_all_ts['rsr_alpha'], zorder=5)
for l in kge_lims:
    ax.axvspan(l, 1.0, facecolor='green', alpha=0.25, zorder=1)
for l in pbias_lims:
    ax.axhspan(l, -l, facecolor='green', alpha=0.25, zorder=1)
# ax.axvspan(-0.5, 0.3, facecolor='red', alpha=0.25, zorder=1)
# ax.axhspan(-75, -25, facecolor='red', alpha=0.25, zorder=1)
# ax.axhspan(75, 25, facecolor='red', alpha=0.25, zorder=1)
# ax.hlines([-25,-15,-10,10,15,25], -0.5, 1, colors=['orange', 'yellow', 'darkgreen', 'darkgreen', 'yellow', 'orange'], ls='--')
# ax.vlines([0.3, 0.5, 0.75], -75, 75, colors=['orange', 'yellow', 'darkgreen'], ls='--')
ax.set_xlabel('KGE')
ax.set_ylabel('PBIAS')
ax.set_xlim(-0.5,1)
ax.set_ylim(-75,75)
ax.set_xticks([-0.41,0,0.3,0.5,0.75,1])
ax.set_yticks([-75,-50,-25,-10,0,10,25,50,75])
ax.legend(loc='lower left', title='Performance NSE', title_fontsize=10, prop = {'size':9},
           markerscale=0.5, labelspacing=0.2, handletextpad=0.1, frameon=True, borderpad=0.5, borderaxespad=0.5)
# ax.text(0.6, 80, f'{timesteps_names[timestep]}', size='large', weight='bold')
# ax.text(0.6, 80, f'{ds_names_dict[ds]}\n{timesteps_names[timestep]}', size='large', weight='bold')
plt.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/KvPvNc_All.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/KvPvNc_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/KvPvNc_{ds}_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.close()


# KGE vs NSE vs PBIAS_class
fig, ax = plt.subplots(figsize=(8,8))
sns.scatterplot(x='kge', y='nashsutcliffe', hue='pbias_class', data=df_all_ts, ax=ax, palette=pallete_class,
                s=1, linewidth=0, alpha=df_all_ts['rsr_alpha'], zorder=5)
for l in kge_lims:
    ax.axvspan(l, 1.0, facecolor='green', alpha=0.25, zorder=1)
for l in nse_lims:
    ax.axhspan(l, 1.0, facecolor='green', alpha=0.25, zorder=1)
# ax.hlines([0.5,0.65,0.75], -0.5, 1, colors=['darkred', 'cyan', 'darkblue'], ls='--')
# ax.vlines([0.3, 0.5, 0.75], -0.5, 1, colors=['darkred', 'cyan', 'darkblue'], ls='--')
ax.set_xlabel('KGE')
ax.set_ylabel('NSE')
ax.set_xlim(-0.5,1)
ax.set_ylim(-0.5,1)
ax.set_xticks([-0.41,0,0.3,0.5,0.75,1])
ax.set_yticks([-0.5,0,0.5,0.65,0.75,1])
ax.legend(loc='lower left', title='Performance PBIAS', title_fontsize=10, prop = {'size':9},
           markerscale=0.5, labelspacing=0.2, handletextpad=0.1, frameon=True, borderpad=0.5, borderaxespad=0.5)
# ax.text(0.6, 80, f'{timesteps_names[timestep]}', size='large', weight='bold')
# ax.text(0.6, 80, f'{ds_names_dict[ds]}\n{timesteps_names[timestep]}', size='large', weight='bold')
plt.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/KvNvPc_All.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/KvNvPc_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/KvNvPc_{ds}_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.close()


# PBIAS vs NSE vs KGE_class
fig, ax = plt.subplots(figsize=(8,8))
sns.scatterplot(x='pbias', y='nashsutcliffe', hue='kge_class', data=df_all_ts, ax=ax, palette=pallete_class,
                s=1, linewidth=0, alpha=df_all_ts['rsr_alpha'], zorder=5)
for l in pbias_lims:
    ax.axvspan(l, -l, facecolor='green', alpha=0.25, zorder=1)
for l in nse_lims:
    ax.axhspan(l, 1.0, facecolor='green', alpha=0.25, zorder=1)
# ax.hlines([0.5,0.65,0.75], -75, 75, colors=['darkred', 'cyan', 'darkblue'], ls='--')
# ax.vlines([-25,-15,-10,10,15,25], -0.5, 1, colors=['darkred', 'cyan', 'darkblue', 'darkblue', 'cyan', 'darkred'], ls='--')
ax.set_xlabel('PBIAS')
ax.set_ylabel('NSE')
ax.set_xlim(-75,75)
ax.set_ylim(-0.5,1)
ax.set_xticks([-75,-50,-25,-10,0,10,25,50,75])
ax.set_yticks([-0.5,0,0.5,0.65,0.75,1])
ax.legend(loc='lower left', title='Performance KGE', title_fontsize=10, prop = {'size':9},
           markerscale=0.5, labelspacing=0.2, handletextpad=0.1, frameon=True, borderpad=0.5, borderaxespad=0.5)
# ax.text(0.6, 80, f'{timesteps_names[timestep]}', size='large', weight='bold')
# ax.text(0.6, 80, f'{ds_names_dict[ds]}\n{timesteps_names[timestep]}', size='large', weight='bold')
plt.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/PvNvKc_All.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/PvNvKc_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/PvNvKc_{ds}_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.close()

###########################################################################################
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
cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cmap_combined)

cmaps = {'nashsutcliffe':cmap_r2b, 'kge':cmap_r2b, 'rsr':cmap_b2r, 'pbias':cmap_rbg}
# vmin_dict = {'kge':-0.41, 'nashsutcliffe':0, 'correlationcoefficient':-1, 'pbias':-25, 'rsr':0, 'rmse':0, 'FB':0, 'POD':0, 'FAR':0, 'TS':0, 'ETS':0}
# vmax_dict = {'kge':1, 'nashsutcliffe':1, 'correlationcoefficient':1, 'pbias':25, 'rsr':1, 'rmse':25, 'FB':1, 'POD':1, 'FAR':1, 'TS':1, 'ETS':1}

vmin_dict = {'kge':-0.4, 'nashsutcliffe':0, 'correlationcoefficient':-1, 'pbias':-30, 'rsr':0, 'rmse':0, 'FB':0, 'POD':0, 'FAR':0, 'TS':0, 'ETS':0}
vmax_dict = {'kge':1, 'nashsutcliffe':1, 'correlationcoefficient':1, 'pbias':30, 'rsr':1.4, 'rmse':25, 'FB':1, 'POD':1, 'FAR':1, 'TS':1, 'ETS':1}


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

# filter_criteria = [basins, elevations, climates, states]
# for criteria in filter_criteria:

#############################################################################################
# MATSHOW for all errors and different elevations    

for period in periods_dict:
    # period = 2
    print(periods_dict[period])
    fig, ax = pplt.subplots(ncols=4, nrows=5, share=False, figwidth=11, figheight=11) #, wspace=0.05, hspace=0.2)
    
    for row, timestep in enumerate(timesteps):
        # timestep = 'd'
        print(row, timesteps_names[timestep])
    
        filter_timestep = (df['timestep'] == timesteps_names[timestep])
        # filter_dataset = (df['dataset'] == ds_names_dict[ds])
        filter_period = (df['period'] == periods_dict[period])
        
        df_filtered = df[(filter_timestep & filter_period)]
        # df_filtered.columns
        
        elev_errors_array = np.zeros((len(errors_list),len(elevations),len(ds_order)))
        # elev_errors_array = np.zeros((len(errors_list),len(basins),len(ds_order)))
        
        for i, ds in enumerate(ds_order):
            # ds = ds_order[0]
            # print(i, ds_names_dict[ds])
            filter_dataset = (df_filtered['dataset'] == ds_names_dict[ds])
            
            for j, elevation in elevations.iterrows():
            # for j, filter_criteria in criteria.iterrows():
                # elevation = elevations['Z_Range'][0]
                # print(elevation['Z_Class'], elevation['Z_Range'])
                filter_elevation = (df_filtered['Z_Class'] == elevation['Z_Class'])
                
                for k, error in enumerate(errors_list):
                    # k, error = 0, 'nashsutcliffe'
                    # print(k,j,i)
                    filter_outliers = (~df_filtered[f'{error}_outlier'])
                    df_elev = df_filtered[(filter_dataset & filter_outliers & filter_elevation)]
                    # df_elev = df_filtered[(filter_dataset & filter_elevation)]
                    df_elev = df_elev.replace([-np.inf, np.inf], np.nan)
                    df_elev.dropna(subset=[error], inplace=True)
                    
                    # pd.set_option('display.max_columns', None)
                    # plt.scatter(df_elev['id'], df_elev[f'{error}'])
                    
                    error_elev = df_elev[error].mean()
                    elev_errors_array[k,j,i] = error_elev

        # fig, ax = pplt.subplots(ncols=4, nrows=1, share=False, figwidth=16, figheight=4) #, wspace=0.05, hspace=0.2)
        for col, error in enumerate(errors_list):
            # i, error = 0, 'nashsutcliffe'
            # error_name = errors_dict[error]
            # print(col, error)
            
            arr = elev_errors_array[col]*1
            
            if error == 'pbias':
                # bias_lims = np.ceil(max(abs(np.nanmin(arr)), abs(np.nanmax(arr)))/10)*10
                arr1 = arr*1
                arr1[((arr1<vmin_dict[error])|(arr1>vmax_dict[error]))] = np.nan
                arr_out1 = arr*1
                arr_out1[arr_out1>=vmin_dict[error]] = np.nan
                arr_out2 = arr*1
                arr_out2[arr_out2<=vmax_dict[error]] = np.nan
                
                df_plot = ax[row,col].matshow(arr1, cmap=cmap_rbg,
                                        levels=pplt.arange(vmin_dict[error], vmax_dict[error], 4)) #2*bias_lims/15
                df_plot_out1 = ax[row,col].matshow(arr_out1, cmap=cmap_dr2r,
                                        levels=pplt.arange(-70, vmin_dict[error], 4))
                df_plot_out2 = ax[row,col].matshow(arr_out2, cmap=cmap_g2dg,
                                    levels=pplt.arange(vmax_dict[error], 70, 4))
                
                for (i, j), z in np.ndenumerate(arr):
                    if not np.isnan(z):
                        if abs(z) < 10:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='y', weight='bold', ha='center', va='center', fontsize=7)
                                    # border=True, borderwidth=0.4, bordercolor='k')
                        elif z in arr1:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='k', weight='bold', ha='center', va='center', fontsize=7)
                        else:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7)
                
            elif error == 'rsr':
                # avmin = np.round(np.nanmin(elev_errors_array[col]), 2)
                # avmax = np.round(np.nanmax(elev_errors_array[col]), 2)
                arr1 = arr*1
                arr1[arr1>vmax_dict[error]] = np.nan
                arr_out1 = arr*1
                arr_out1[arr_out1<=vmax_dict[error]] = np.nan
                
                df_plot = ax[row,col].matshow(arr1, cmap=cmap_b2r,
                                        levels=pplt.arange(vmin_dict[error], vmax_dict[error], 0.1)) #np.round((avmax-avmin)/15,2)
                df_plot_out1 = ax[row,col].matshow(arr_out1, cmap=cmap_r2dr, #, #, vmin=0) #,
                                        levels=pplt.arange(vmax_dict[error], 2.4, 0.1))
                
                for (i, j), z in np.ndenumerate(arr):
                    if not np.isnan(z):
                        if z < 0.5:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='y', weight='bold', ha='center', va='center', fontsize=7)
                        elif z in arr1:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='k', weight='bold', ha='center', va='center', fontsize=7)
                        else:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7)
                
            elif error == 'kge':
                # avmin = np.round(np.nanmin(elev_errors_array[col]), 2)
                # avmax = np.round(np.nanmax(elev_errors_array[col]), 2)
                arr1 = arr*1
                arr1[arr1<vmin_dict[error]] = np.nan
                arr_out1 = arr*1
                arr_out1[arr_out1>=vmin_dict[error]] = np.nan
                
                df_plot = ax[row,col].matshow(arr1, cmap=cmap_r2b,
                                        levels=pplt.arange(vmin_dict[error], vmax_dict[error], 0.1)) #np.round((avmax-avmin)/15,2)
                df_plot_out1 = ax[row,col].matshow(arr_out1, cmap=cmap_dr2r,
                                        levels=pplt.arange(-1.4, vmin_dict[error], 0.1))
                
                for (i, j), z in np.ndenumerate(arr):
                    if not np.isnan(z):
                        if z > 0.75:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='y', weight='bold', ha='center', va='center', fontsize=7)
                        elif z in arr1:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='k', weight='bold', ha='center', va='center', fontsize=7)
                        else:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7)
                                    
            elif error == 'nashsutcliffe':
                # avmin = np.round(np.nanmin(elev_errors_array[col]), 2)
                # avmax = np.round(np.nanmax(elev_errors_array[col]), 2)
                arr1 = arr*1
                arr1[arr1<vmin_dict[error]] = np.nan
                arr_out1 = arr*1
                arr_out1[arr_out1>=vmin_dict[error]] = np.nan
                
                df_plot = ax[row,col].matshow(arr1, cmap=cmap_r2b,
                                        levels=pplt.arange(vmin_dict[error], vmax_dict[error], (vmax_dict[error]-vmin_dict[error])/16)) #np.round((avmax-avmin)/15,2)
                df_plot_out1 = ax[row,col].matshow(arr_out1, cmap=cmap_dr2r,
                                        levels=pplt.arange(-4, vmin_dict[error], 0.4))
                
                for (i, j), z in np.ndenumerate(arr):
                    if not np.isnan(z):
                        if z > 0.75:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='y', weight='bold', ha='center', va='center', fontsize=7)
                        elif z in arr1:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='k', weight='bold', ha='center', va='center', fontsize=7)
                        else:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7)
            
            if row == 0:
                ax[row,col].xaxis.set_label_position('top')
                ax[row,col].set_xticks(range(7))
                ax[row,col].set_xticklabels(ds_names_abbr)
            elif row == 4:
                ax[row,col].xaxis.set_label_position('bottom')
                ax[row,col].set_xticks(range(7))
                ax[row,col].set_xticklabels(ds_names_abbr)
            
            # ax[row,col].set_xticks(range(7))
            # ax[row,col].set_xticklabels(ds_names_abbr)
            if col == 0:
                ax[row,col].yaxis.set_label_position('left')
                ax[row,col].set_yticks(range(5))
                ax[row,col].set_yticklabels(list(elevations['Z_Range'].values))
            elif col == 3:
                ax[row,col].yaxis.set_label_position('right')
                ax[row,col].set_yticks(range(5))
                ax[row,col].set_yticklabels(list(elevations['Z_Range'].values))
            
            # for (i, j), z in np.ndenumerate(arr):
            #     if not np.isnan(z):
            #         ax[row,col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7,
            #                 border=True, borderwidth=0.4, bordercolor='k')
            
            if row == 4:
                ticks_loc = np.linspace(vmin_dict[error], vmax_dict[error], 3, endpoint=True)
                cbar = fig.colorbar(df_plot, ticks=ticks_loc, loc='b', col=(col+1), shrink=0.8)
                if error == 'pbias':
                    cbar.ax.set_xticklabels([f'<{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'>{vmax_dict[error]}'])
                elif error == 'rsr':
                    cbar.ax.set_xticklabels([f'{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'>{vmax_dict[error]}'])
                else:
                    cbar.ax.set_xticklabels([f'<{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'{vmax_dict[error]}'])
                
                if error == 'pbias':
                    ticks_loc1 = np.linspace(-70, -30, 3, endpoint=True)
                    ticks_loc2 = np.linspace(30, 70, 3, endpoint=True)
                    cbar_out1 = fig.colorbar(df_plot_out1, ticks=ticks_loc1, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out2 = fig.colorbar(df_plot_out2, ticks=ticks_loc2, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out1.ax.set_xticklabels(['<-70', '-50', '-30'])
                    cbar_out2.ax.set_xticklabels(['30', '50', '>70'])
                elif error == 'rsr':
                    ticks_loc1 = np.linspace(vmax_dict[error], 2.4, 3, endpoint=True)
                    cbar_out1 = fig.colorbar(df_plot_out1, ticks=ticks_loc1, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out1.ax.set_xticklabels([f'{vmax_dict[error]}', '1.9', '>2.4'])
                elif error == 'kge':
                    ticks_loc1 = np.linspace(-1.4, vmin_dict[error], 3, endpoint=True)
                    cbar_out1 = fig.colorbar(df_plot_out1, ticks=ticks_loc1, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out1.ax.set_xticklabels(['<-1.4', '-0.9', f'{vmin_dict[error]}'])
                elif error == 'nashsutcliffe':
                    ticks_loc1 = np.linspace(-4, vmin_dict[error], 3, endpoint=True)
                    cbar_out1 = fig.colorbar(df_plot_out1, ticks=ticks_loc1, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out1.ax.set_xticklabels(['<-4', '-2', f'{vmin_dict[error]}'])
    
    ax.format(grid=False,
            toplabels=('NSE', 'KGE', 'RSR', 'PBIAS'),
            leftlabels=[ts_name+'\nElevation' for ts_name in timesteps_names.values()],
            leftlabelpad=12,
            leftlabels_kw={'ha':'center'},
            xtickloc='both',
            xminorlocator='none',
            xticklabelloc='none',
            ytickloc='both',
            yminorlocator='none',
            yticklabelloc='none',
            xlabel='',
            ylabel='')
    
    ax[0, :].format(xtickloc='both', xticklabelloc='top')
    ax[-1, :].format(xtickloc='both', xticklabelloc='bottom')
    ax[:, 0].format(ytickloc='both', yticklabelloc='left')
    ax[:, -1].format(ytickloc='both', yticklabelloc='right')
    
    plt.figtext(0.01, 0.96, f'Period:\n{periods_dict[period]}', size='large', weight='bold')
    fig.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Elevations/Error_Comparison_{periods_dict[period]}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    # fig.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Elevations/Error_Comparison_{periods_dict[period]}_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    plt.close()
    
# help(ax.format)
#############################################################################################
# MATSHOW for all datasets and different criteria
criteria_full_name = {'Basin':'Hydrologic Region', 'Climate':'Climatic Region', 'State':'Administrative Division'}
criteria_names = ['Basin', 'Climate', 'State']
#### Select Filter Criteria
filter_criteria_order = [basins, climates, states]
row = 0
for ix in range(3):
# ix = 0
    criteria_name = criteria_names[ix]
    criteria = filter_criteria_order[ix]
    ####
    
    for period in periods_dict:
        # period = 2
        print(periods_dict[period])
        for timestep in timesteps:
            # timestep = 'd'
            print(timesteps_names[timestep])
            
            filter_timestep = (df['timestep'] == timesteps_names[timestep])
            filter_period = (df['period'] == periods_dict[period])
            df_filtered = df[(filter_timestep & filter_period)]
            criteria_errors_array = np.zeros((len(errors_list),len(criteria),len(ds_order)))
            
            for i, ds in enumerate(ds_order):
                # ds = ds_order[0]
                # print(i, ds_names_dict[ds])
                filter_dataset = (df_filtered['dataset'] == ds_names_dict[ds])
                
                for j, filter_criteria in criteria.iterrows():
                    # elevation = elevations['Z_Range'][0]
                    # print(elevation['Z_Class'], elevation['Z_Range'])
                    filter_crit = (df_filtered[f'{criteria_name}_ID'] == filter_criteria[f'{criteria_name}_ID'])
                    
                    for k, error in enumerate(errors_list):
                        # k, error = 0, 'nashsutcliffe'
                        # print(k,j,i)
                        filter_outliers = (~df_filtered[f'{error}_outlier'])
                        df_crit = df_filtered[(filter_dataset & filter_outliers & filter_crit)]
                        # df_crit = df_filtered[(filter_dataset & filter_crit)]
                        df_crit = df_crit.replace([-np.inf, np.inf], np.nan)
                        df_crit.dropna(subset=[error], inplace=True)
                        
                        error_crit = df_crit[error].mean()
                        criteria_errors_array[k,j,i] = error_crit
            
            if criteria_name == 'Basin':
                fig, ax = pplt.subplots(ncols=4, nrows=1, share=False, figwidth=11, figheight=14.5) #Basins
            elif criteria_name == 'State':
                fig, ax = pplt.subplots(ncols=4, nrows=1, share=False, figwidth=11, figheight=13) #States
            elif criteria_name == 'Climate':
                fig, ax = pplt.subplots(ncols=4, nrows=1, share=False, figwidth=11, figheight=5.5) #Climates
            
            for col, error in enumerate(errors_list):
                # i, error = 0, 'nashsutcliffe'
                # error_name = errors_dict[error]
                # print(col, error)
                
                arr = criteria_errors_array[col]*1
                
                if error == 'pbias':
                    # bias_lims = np.ceil(max(abs(np.nanmin(arr)), abs(np.nanmax(arr)))/10)*10
                    arr1 = arr*1
                    arr1[((arr1<vmin_dict[error])|(arr1>vmax_dict[error]))] = np.nan
                    arr_out1 = arr*1
                    arr_out1[arr_out1>=vmin_dict[error]] = np.nan
                    arr_out2 = arr*1
                    arr_out2[arr_out2<=vmax_dict[error]] = np.nan
                    
                    df_plot = ax[row,col].matshow(arr1, cmap=cmap_rbg,
                                            levels=pplt.arange(vmin_dict[error], vmax_dict[error], 4)) #2*bias_lims/15
                    df_plot_out1 = ax[row,col].matshow(arr_out1, cmap=cmap_dr2r,
                                            levels=pplt.arange(-70, vmin_dict[error], 4))
                    df_plot_out2 = ax[row,col].matshow(arr_out2, cmap=cmap_g2dg,
                                        levels=pplt.arange(vmax_dict[error], 70, 4))
                    
                    for (i, j), z in np.ndenumerate(arr):
                        if not np.isnan(z):
                            if abs(z) < 10:
                                ax[row,col].text(j, i, '{:0.2f}'.format(z), color='y', weight='bold', ha='center', va='center', fontsize=7)
                                        # border=True, borderwidth=0.4, bordercolor='k')
                            elif z in arr1:
                                ax[row,col].text(j, i, '{:0.2f}'.format(z), color='k', weight='bold', ha='center', va='center', fontsize=7)
                            else:
                                ax[row,col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7)
                    
                elif error == 'rsr':
                    # avmin = np.round(np.nanmin(elev_errors_array[col]), 2)
                    # avmax = np.round(np.nanmax(elev_errors_array[col]), 2)
                    arr1 = arr*1
                    arr1[arr1>vmax_dict[error]] = np.nan
                    arr_out1 = arr*1
                    arr_out1[arr_out1<=vmax_dict[error]] = np.nan
                    
                    df_plot = ax[row,col].matshow(arr1, cmap=cmap_b2r,
                                            levels=pplt.arange(vmin_dict[error], vmax_dict[error], 0.1)) #np.round((avmax-avmin)/15,2)
                    df_plot_out1 = ax[row,col].matshow(arr_out1, cmap=cmap_r2dr, #, #, vmin=0) #,
                                            levels=pplt.arange(vmax_dict[error], 2.4, 0.1))
                    
                    for (i, j), z in np.ndenumerate(arr):
                        if not np.isnan(z):
                            if z < 0.5:
                                ax[row,col].text(j, i, '{:0.2f}'.format(z), color='y', weight='bold', ha='center', va='center', fontsize=7)
                            elif z in arr1:
                                ax[row,col].text(j, i, '{:0.2f}'.format(z), color='k', weight='bold', ha='center', va='center', fontsize=7)
                            else:
                                ax[row,col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7)
                    
                elif error == 'kge':
                    # avmin = np.round(np.nanmin(elev_errors_array[col]), 2)
                    # avmax = np.round(np.nanmax(elev_errors_array[col]), 2)
                    arr1 = arr*1
                    arr1[arr1<vmin_dict[error]] = np.nan
                    arr_out1 = arr*1
                    arr_out1[arr_out1>=vmin_dict[error]] = np.nan
                    
                    df_plot = ax[row,col].matshow(arr1, cmap=cmap_r2b,
                                            levels=pplt.arange(vmin_dict[error], vmax_dict[error], 0.1)) #np.round((avmax-avmin)/15,2)
                    df_plot_out1 = ax[row,col].matshow(arr_out1, cmap=cmap_dr2r,
                                            levels=pplt.arange(-1.4, vmin_dict[error], 0.1))
                    
                    for (i, j), z in np.ndenumerate(arr):
                        if not np.isnan(z):
                            if z > 0.75:
                                ax[row,col].text(j, i, '{:0.2f}'.format(z), color='y', weight='bold', ha='center', va='center', fontsize=7)
                            elif z in arr1:
                                ax[row,col].text(j, i, '{:0.2f}'.format(z), color='k', weight='bold', ha='center', va='center', fontsize=7)
                            else:
                                ax[row,col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7)
                                        
                elif error == 'nashsutcliffe':
                    # avmin = np.round(np.nanmin(elev_errors_array[col]), 2)
                    # avmax = np.round(np.nanmax(elev_errors_array[col]), 2)
                    arr1 = arr*1
                    arr1[arr1<vmin_dict[error]] = np.nan
                    arr_out1 = arr*1
                    arr_out1[arr_out1>=vmin_dict[error]] = np.nan
                    
                    df_plot = ax[row,col].matshow(arr1, cmap=cmap_r2b,
                                            levels=pplt.arange(vmin_dict[error], vmax_dict[error], (vmax_dict[error]-vmin_dict[error])/16)) #np.round((avmax-avmin)/15,2)
                    df_plot_out1 = ax[row,col].matshow(arr_out1, cmap=cmap_dr2r,
                                            levels=pplt.arange(-4, vmin_dict[error], 0.4))
                    
                    for (i, j), z in np.ndenumerate(arr):
                        if not np.isnan(z):
                            if z > 0.75:
                                ax[row,col].text(j, i, '{:0.2f}'.format(z), color='y', weight='bold', ha='center', va='center', fontsize=7)
                            elif z in arr1:
                                ax[row,col].text(j, i, '{:0.2f}'.format(z), color='k', weight='bold', ha='center', va='center', fontsize=7)
                            else:
                                ax[row,col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7)
                
                # for (i, j), z in np.ndenumerate(arr):
                #     if not np.isnan(z):
                #         ax[col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7,
                #                 border=True, borderwidth=0.4, bordercolor='k')
                    
                ticks_loc = np.linspace(vmin_dict[error], vmax_dict[error], 3, endpoint=True)
                cbar = fig.colorbar(df_plot, ticks=ticks_loc, loc='b', col=(col+1), shrink=0.8)
                if error == 'pbias':
                    cbar.ax.set_xticklabels([f'<{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'>{vmax_dict[error]}'])
                elif error == 'rsr':
                    cbar.ax.set_xticklabels([f'{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'>{vmax_dict[error]}'])
                else:
                    cbar.ax.set_xticklabels([f'<{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'{vmax_dict[error]}'])
                
                if error == 'pbias':
                    ticks_loc1 = np.linspace(-70, -30, 3, endpoint=True)
                    ticks_loc2 = np.linspace(30, 70, 3, endpoint=True)
                    cbar_out1 = fig.colorbar(df_plot_out1, ticks=ticks_loc1, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out2 = fig.colorbar(df_plot_out2, ticks=ticks_loc2, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out1.ax.set_xticklabels(['<-70', '-50', '-30'])
                    cbar_out2.ax.set_xticklabels(['30', '50', '>70'])
                elif error == 'rsr':
                    ticks_loc1 = np.linspace(vmax_dict[error], 2.4, 3, endpoint=True)
                    cbar_out1 = fig.colorbar(df_plot_out1, ticks=ticks_loc1, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out1.ax.set_xticklabels([f'{vmax_dict[error]}', '1.9', '>2.4'])
                elif error == 'kge':
                    ticks_loc1 = np.linspace(-1.4, vmin_dict[error], 3, endpoint=True)
                    cbar_out1 = fig.colorbar(df_plot_out1, ticks=ticks_loc1, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out1.ax.set_xticklabels(['<-1.4', '-0.9', f'{vmin_dict[error]}'])
                elif error == 'nashsutcliffe':
                    ticks_loc1 = np.linspace(-4, vmin_dict[error], 3, endpoint=True)
                    cbar_out1 = fig.colorbar(df_plot_out1, ticks=ticks_loc1, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out1.ax.set_xticklabels(['<-4', '-2', f'{vmin_dict[error]}'])
                    
                ax[col].set_title(f'{errors_dict[error]}', weight='bold')
                
                # if col == 0:
                ax[col].set_yticks(range(len(criteria[f'{criteria_name}_ID'].values)))
                ax[col].set_yticklabels(list(criteria[f'{criteria_name}_ID'].values))
                # else:
                #     ax[col].set_yticks(range(len(criteria[f'{criteria_name}_ID'].values)))
                
                ax[col].set_xticks(range(7))
                ax[col].set_xticklabels(ds_names_abbr)
                
                if col == 0:
                    ax[col].tick_params(axis='both', which='major', bottom=True, top=True, left=True, right=True,
                                        labelbottom=True, labeltop=True, labelleft=True, labelright=False)
                elif col == 3:
                    ax[col].tick_params(axis='both', which='major', bottom=True, top=True, left=True, right=True,
                                        labelbottom=True, labeltop=True, labelleft=False, labelright=True)
                else:
                    ax[col].tick_params(axis='both', which='major', bottom=True, top=True, left=True, right=True,
                                        labelbottom=True, labeltop=True, labelleft=False, labelright=False)
                    
                ax[col].tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False,
                                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                # if col == 0:
                #     ax[col].format(ytickloc='both', yticklabelloc='left')
                # elif col == 3:
                #     ax[col].format(ytickloc='both', yticklabelloc='right')
                # else:
                #     ax[col].format(ytickloc='both', yticklabelloc='none')
                    
            ax.format(grid=False,
                    # toplabels=('NSE', 'KGE', 'RSR', 'PBIAS'))
                    # leftlabels=f'{criteria_name} ID')
                    xtickloc='both',
                    xminorlocator='none',
                    # xticklabelloc='none',
                    ytickloc='both',
                    yminorlocator='none')
                    # yticklabelloc='none')
            
            ax[0].set_ylabel(f'{criteria_full_name[criteria_name]} ID', size='large', weight='bold')
            # ax[0].format(ytickloc='both', yticklabelloc='left')
            
            if criteria_name == 'Climate':
                plt.figtext(0.01, 0.96, f'{periods_dict[period]}\n{timesteps_names[timestep]}', size='large', weight='bold')
            elif criteria_name == 'Basin':
                plt.figtext(0.01, 0.98, f'{periods_dict[period]}\n{timesteps_names[timestep]}', size='large', weight='bold')
            elif criteria_name == 'State':
                plt.figtext(0.01, 0.97, f'{periods_dict[period]}\n{timesteps_names[timestep]}', size='large', weight='bold')
            
            fig.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/{criteria_name}s/Error_Comparison_{criteria_name}_{periods_dict[period]}_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
            plt.close()
            
# help(ax[col].tick_params)

#############################################################################################
# MATSHOW for all datasets and different climate divided in 2 images
criteria_full_name = {'Basin':'Hydrologic Region', 'Climate':'Climatic Region', 'State':'Administrative Division'}
criteria_names = ['Basin', 'Climate', 'State']
#### Select Filter Criteria
filter_criteria_order = [basins, climates, states]
# for ix in range(3):
# ix = 1
criteria_name = criteria_names[ix]
criteria = filter_criteria_order[ix]
####

for period in periods_dict:
    # period = 2
    print(periods_dict[period])
    # fig, ax = pplt.subplots(ncols=4, nrows=3, share=False, figwidth=11, figheight=13)
    fig, ax = pplt.subplots(ncols=4, nrows=2, share=False, figwidth=11, figheight=9.5)
    # fig, ax = pplt.subplots(ncols=4, nrows=5, share=False, figwidth=11, figheight=21)
    
    # for row, timestep in enumerate(timesteps[:3]):
    for row, timestep in enumerate(timesteps[3:]):
        # timestep = 'd'
        print(row, timesteps_names[timestep])
        
        filter_timestep = (df['timestep'] == timesteps_names[timestep])
        filter_period = (df['period'] == periods_dict[period])
        df_filtered = df[(filter_timestep & filter_period)]
        criteria_errors_array = np.zeros((len(errors_list),len(criteria),len(ds_order)))
        
        for i, ds in enumerate(ds_order):
            # ds = ds_order[0]
            # print(i, ds_names_dict[ds])
            filter_dataset = (df_filtered['dataset'] == ds_names_dict[ds])
            
            for j, filter_criteria in criteria.iterrows():
                filter_crit = (df_filtered[f'{criteria_name}_ID'] == filter_criteria[f'{criteria_name}_ID'])
                
                for k, error in enumerate(errors_list):
                    # k, error = 0, 'nashsutcliffe'
                    # print(k,j,i)
                    filter_outliers = (~df_filtered[f'{error}_outlier'])
                    df_crit = df_filtered[(filter_dataset & filter_outliers & filter_crit)]
                    # df_crit = df_filtered[(filter_dataset & filter_crit)]
                    df_crit = df_crit.replace([-np.inf, np.inf], np.nan)
                    df_crit.dropna(subset=[error], inplace=True)
                    
                    error_crit = df_crit[error].mean()
                    criteria_errors_array[k,j,i] = error_crit
        
        for col, error in enumerate(errors_list):
            # i, error = 0, 'nashsutcliffe'
            # error_name = errors_dict[error]
            # print(col, error)
            
            arr = criteria_errors_array[col]*1
            
            if error == 'pbias':
                # bias_lims = np.ceil(max(abs(np.nanmin(arr)), abs(np.nanmax(arr)))/10)*10
                arr1 = arr*1
                arr1[((arr1<vmin_dict[error])|(arr1>vmax_dict[error]))] = np.nan
                arr_out1 = arr*1
                arr_out1[arr_out1>=vmin_dict[error]] = np.nan
                arr_out2 = arr*1
                arr_out2[arr_out2<=vmax_dict[error]] = np.nan
                
                df_plot = ax[row,col].matshow(arr1, cmap=cmap_rbg,
                                        levels=pplt.arange(vmin_dict[error], vmax_dict[error], 4)) #2*bias_lims/15
                df_plot_out1 = ax[row,col].matshow(arr_out1, cmap=cmap_dr2r,
                                        levels=pplt.arange(-70, vmin_dict[error], 4))
                df_plot_out2 = ax[row,col].matshow(arr_out2, cmap=cmap_g2dg,
                                    levels=pplt.arange(vmax_dict[error], 70, 4))
                
                for (i, j), z in np.ndenumerate(arr):
                    if not np.isnan(z):
                        if abs(z) < 10:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='y', weight='bold', ha='center', va='center', fontsize=7)
                                    # border=True, borderwidth=0.4, bordercolor='k')
                        elif z in arr1:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='k', weight='bold', ha='center', va='center', fontsize=7)
                        else:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7)
                
            elif error == 'rsr':
                # avmin = np.round(np.nanmin(elev_errors_array[col]), 2)
                # avmax = np.round(np.nanmax(elev_errors_array[col]), 2)
                arr1 = arr*1
                arr1[arr1>vmax_dict[error]] = np.nan
                arr_out1 = arr*1
                arr_out1[arr_out1<=vmax_dict[error]] = np.nan
                
                df_plot = ax[row,col].matshow(arr1, cmap=cmap_b2r,
                                        levels=pplt.arange(vmin_dict[error], vmax_dict[error], 0.1)) #np.round((avmax-avmin)/15,2)
                df_plot_out1 = ax[row,col].matshow(arr_out1, cmap=cmap_r2dr, #, #, vmin=0) #,
                                        levels=pplt.arange(vmax_dict[error], 2.4, 0.1))
                
                for (i, j), z in np.ndenumerate(arr):
                    if not np.isnan(z):
                        if z < 0.5:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='y', weight='bold', ha='center', va='center', fontsize=7)
                        elif z in arr1:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='k', weight='bold', ha='center', va='center', fontsize=7)
                        else:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7)
                
            elif error == 'kge':
                # avmin = np.round(np.nanmin(elev_errors_array[col]), 2)
                # avmax = np.round(np.nanmax(elev_errors_array[col]), 2)
                arr1 = arr*1
                arr1[arr1<vmin_dict[error]] = np.nan
                arr_out1 = arr*1
                arr_out1[arr_out1>=vmin_dict[error]] = np.nan
                
                df_plot = ax[row,col].matshow(arr1, cmap=cmap_r2b,
                                        levels=pplt.arange(vmin_dict[error], vmax_dict[error], 0.1)) #np.round((avmax-avmin)/15,2)
                df_plot_out1 = ax[row,col].matshow(arr_out1, cmap=cmap_dr2r,
                                        levels=pplt.arange(-1.4, vmin_dict[error], 0.1))
                
                for (i, j), z in np.ndenumerate(arr):
                    if not np.isnan(z):
                        if z > 0.75:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='y', weight='bold', ha='center', va='center', fontsize=7)
                        elif z in arr1:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='k', weight='bold', ha='center', va='center', fontsize=7)
                        else:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7)
                                    
            elif error == 'nashsutcliffe':
                # avmin = np.round(np.nanmin(elev_errors_array[col]), 2)
                # avmax = np.round(np.nanmax(elev_errors_array[col]), 2)
                arr1 = arr*1
                arr1[arr1<vmin_dict[error]] = np.nan
                arr_out1 = arr*1
                arr_out1[arr_out1>=vmin_dict[error]] = np.nan
                
                df_plot = ax[row,col].matshow(arr1, cmap=cmap_r2b,
                                        levels=pplt.arange(vmin_dict[error], vmax_dict[error], (vmax_dict[error]-vmin_dict[error])/16)) #np.round((avmax-avmin)/15,2)
                df_plot_out1 = ax[row,col].matshow(arr_out1, cmap=cmap_dr2r,
                                        levels=pplt.arange(-4, vmin_dict[error], 0.4))
                
                for (i, j), z in np.ndenumerate(arr):
                    if not np.isnan(z):
                        if z > 0.75:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='y', weight='bold', ha='center', va='center', fontsize=7)
                        elif z in arr1:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='k', weight='bold', ha='center', va='center', fontsize=7)
                        else:
                            ax[row,col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7)
            
            # for (i, j), z in np.ndenumerate(arr):
            #     if not np.isnan(z):
            #         ax[row,col].text(j, i, '{:0.2f}'.format(z), color='w', weight='bold', ha='center', va='center', fontsize=7,
            #                 border=True, borderwidth=0.4, bordercolor='k')
            
            if row == 0:
                ax[row,col].xaxis.set_label_position('top')
                ax[row,col].set_xticks(range(7))
                ax[row,col].set_xticklabels(ds_names_abbr)
            elif row == 1:
                ax[row,col].xaxis.set_label_position('bottom')
                ax[row,col].set_xticks(range(7))
                ax[row,col].set_xticklabels(ds_names_abbr)
                
                ticks_loc = np.linspace(vmin_dict[error], vmax_dict[error], 3, endpoint=True)
                cbar = fig.colorbar(df_plot, ticks=ticks_loc, loc='b', col=(col+1), shrink=0.8)
                if error == 'pbias':
                    cbar.ax.set_xticklabels([f'<{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'>{vmax_dict[error]}'])
                elif error == 'rsr':
                    cbar.ax.set_xticklabels([f'{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'>{vmax_dict[error]}'])
                else:
                    cbar.ax.set_xticklabels([f'<{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'{vmax_dict[error]}'])
                
                if error == 'pbias':
                    ticks_loc1 = np.linspace(-70, -30, 3, endpoint=True)
                    ticks_loc2 = np.linspace(30, 70, 3, endpoint=True)
                    cbar_out1 = fig.colorbar(df_plot_out1, ticks=ticks_loc1, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out2 = fig.colorbar(df_plot_out2, ticks=ticks_loc2, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out1.ax.set_xticklabels(['<-70', '-50', '-30'])
                    cbar_out2.ax.set_xticklabels(['30', '50', '>70'])
                elif error == 'rsr':
                    ticks_loc1 = np.linspace(vmax_dict[error], 2.4, 3, endpoint=True)
                    cbar_out1 = fig.colorbar(df_plot_out1, ticks=ticks_loc1, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out1.ax.set_xticklabels([f'{vmax_dict[error]}', '1.9', '>2.4'])
                elif error == 'kge':
                    ticks_loc1 = np.linspace(-1.4, vmin_dict[error], 3, endpoint=True)
                    cbar_out1 = fig.colorbar(df_plot_out1, ticks=ticks_loc1, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out1.ax.set_xticklabels(['<-1.4', '-0.9', f'{vmin_dict[error]}'])
                elif error == 'nashsutcliffe':
                    ticks_loc1 = np.linspace(-4, vmin_dict[error], 3, endpoint=True)
                    cbar_out1 = fig.colorbar(df_plot_out1, ticks=ticks_loc1, loc='b', col=(col+1), shrink=0.8, pad=0.15)
                    cbar_out1.ax.set_xticklabels(['<-4', '-2', f'{vmin_dict[error]}'])

            if col == 0:
                ax[row,col].yaxis.set_label_position('left')
                ax[row,col].set_yticks(range(len(criteria[f'{criteria_name}_ID'].values)))
                ax[row,col].set_yticklabels(list(criteria[f'{criteria_name}_ID'].values))
            elif col == 3:
                ax[row,col].yaxis.set_label_position('right')
                ax[row,col].set_yticks(range(len(criteria[f'{criteria_name}_ID'].values)))
                ax[row,col].set_yticklabels(list(criteria[f'{criteria_name}_ID'].values))
                
    ax.format(grid=False,
            toplabels=('NSE', 'KGE', 'RSR', 'PBIAS'),
            # leftlabels=[ts_name+'\nClimate ID' for ts_name in ['Daily', 'Monthly', 'Max Monthly', 'Yearly', 'Max Yearly'][:3]],
            leftlabels=[ts_name+'\nClimate ID' for ts_name in ['Daily', 'Monthly', 'Max Monthly', 'Yearly', 'Max Yearly'][3:]],
            leftlabelpad=12,
            leftlabels_kw={'ha':'center'},
            xtickloc='both',
            xminorlocator='none',
            xticklabelloc='none',
            ytickloc='both',
            yminorlocator='none',
            yticklabelloc='none',
            xlabel='',
            ylabel='')
    
    ax[0, :].format(xtickloc='both', xticklabelloc='top')
    ax[-1, :].format(xtickloc='both', xticklabelloc='bottom')
    ax[:, 0].format(ytickloc='both', yticklabelloc='left')
    ax[:, -1].format(ytickloc='both', yticklabelloc='right')
    
    plt.figtext(0.01, 0.985, f'Period:\n{periods_dict[period]}', size='large', weight='bold')
    # plt.show()
    # fig.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/{criteria_name}s/Error_Comparison_{criteria_name}_{periods_dict[period]}_1.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    fig.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/{criteria_name}s/Error_Comparison_{criteria_name}_{periods_dict[period]}_2.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    # fig.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/{criteria_name}s/Error_Comparison_{criteria_name}_{periods_dict[period]}_All.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    plt.close()
    
###########################################################################################
#2D and 3D plots of errors

df_all_ts = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/All/Errors_all.csv', encoding='latin-1')
df_all_ts.drop(['Unnamed: 0'], axis=1, inplace=True)
df_all_ts.replace([np.inf, -np.inf], np.nan, inplace=True)

df = df_all_ts.copy()

# fig, ax = plt.subplots(nrows=3, figsize=(8,18))
for ds in ds_order:
    for timestep in timesteps:
        df = df_all_ts.copy()
        # df = df[df['dataset'] == ds_names_dict[ds]]
        # df = df[df['timestep'] == timesteps_names[timestep]]
    
        # ALL-IN-ONE
        # ['darkblue', 'b', 'c', 'darkred']
        fig, ax = plt.subplots(nrows=3, figsize=(8,18))
        
        sns.scatterplot(x='kge', y='pbias', hue='nashsutcliffe_class', data=df, ax=ax[0], palette=pallete_class, 
                        s=0.75, linewidth=0, zorder=5) #alpha=df['rsr_alpha']
        for l in kge_lims:
            ax[0].axvspan(l, 1.0, facecolor='green', alpha=0.25, zorder=1)
        for l in pbias_lims:
            ax[0].axhspan(l, -l, facecolor='green', alpha=0.25, zorder=1)
        ax[0].set_xlabel('KGE')
        ax[0].set_ylabel('PBIAS')
        ax[0].set_xlim(-0.5,1)
        ax[0].set_ylim(-75,75)
        ax[0].set_xticks([-0.41,0,0.3,0.5,0.75,1])
        ax[0].set_yticks([-75,-50,-25,-10,0,10,25,50,75])
        ax[0].legend(loc='lower left', title='Performance NSE', title_fontsize=10, prop = {'size':9},
                   markerscale=0.5, labelspacing=0.2, handletextpad=0.1, frameon=True, borderpad=0.5, borderaxespad=0.5)
        ax[0].text(-0.48, 68, 'a)')
        
        sns.scatterplot(x='kge', y='nashsutcliffe', hue='pbias_class', data=df, ax=ax[1], palette=pallete_class,
                        s=0.75, linewidth=0, zorder=5)
        for l in kge_lims:
            ax[1].axvspan(l, 1.0, facecolor='green', alpha=0.25, zorder=1)
        for l in nse_lims:
            ax[1].axhspan(l, 1.0, facecolor='green', alpha=0.25, zorder=1)
        ax[1].set_xlabel('KGE')
        ax[1].set_ylabel('NSE')
        ax[1].set_xlim(-0.5,1)
        ax[1].set_ylim(-0.5,1)
        ax[1].set_xticks([-0.41,0,0.3,0.5,0.75,1])
        ax[1].set_yticks([-0.5,0,0.5,0.65,0.75,1])
        ax[1].legend(loc='lower left', title='Performance PBIAS', title_fontsize=10, prop = {'size':9},
                   markerscale=0.5, labelspacing=0.2, handletextpad=0.1, frameon=True, borderpad=0.5, borderaxespad=0.5)
        ax[1].text(-0.48, 0.93, 'b)')
        
        sns.scatterplot(x='pbias', y='nashsutcliffe', hue='kge_class', data=df, ax=ax[2], palette=pallete_class,
                        s=0.75, linewidth=0, zorder=5)
        for l in pbias_lims:
            ax[2].axvspan(l, -l, facecolor='green', alpha=0.25, zorder=1)
        for l in nse_lims:
            ax[2].axhspan(l, 1.0, facecolor='green', alpha=0.25, zorder=1)
        ax[2].set_xlabel('PBIAS')
        ax[2].set_ylabel('NSE')
        ax[2].set_xlim(-75,75)
        ax[2].set_ylim(-0.5,1)
        ax[2].set_xticks([-75,-50,-25,-10,0,10,25,50,75])
        ax[2].set_yticks([-0.5,0,0.5,0.65,0.75,1])
        ax[2].legend(loc='lower left', title='Performance KGE', title_fontsize=10, prop = {'size':9},
                   markerscale=0.5, labelspacing=0.2, handletextpad=0.1, frameon=True, borderpad=0.5, borderaxespad=0.5)
        ax[2].text(-73, 0.93, 'c)')
        
        plt.subplots_adjust(wspace=0.15, hspace=0.15)
        plt.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/3ERRORS_All.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        # plt.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/3ERRORS_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        # plt.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/3ERRORS_{ds}_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        plt.close()

###########################################################################################
#2D and 3D plots of errors All in one

df_all_ts = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/All/Errors_all.csv', encoding='latin-1')
df_all_ts.drop(['Unnamed: 0'], axis=1, inplace=True)
df_all_ts.replace([np.inf, -np.inf], np.nan, inplace=True)

df = df_all_ts.copy()
df['rsr_class_value'] = df['rsr_class'].replace(['Unsatisfactory', 'Satisfactory', 'Good', 'Very Good'], [1, 2, 3, 4])

# fig, ax = plt.subplots(nrows=3, figsize=(8,18))
for ds in ds_order:
    for timestep in timesteps:
        df = df_all_ts.copy()
        # df = df[df['dataset'] == ds_names_dict[ds]]
        # df = df[df['timestep'] == timesteps_names[timestep]]
    
        # ALL-IN-ONE
        # ['darkblue', 'b', 'c', 'darkred']
        # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))
        fig = plt.figure(figsize=(12,12))
        
        ax = fig.add_subplot(2, 2, 1)
        sns.scatterplot(x='kge', y='pbias', hue='nashsutcliffe_class', data=df, ax=ax, palette=pallete_class, 
                        s=0.75, linewidth=0, zorder=5) #alpha=df['rsr_alpha']
        for l in kge_lims:
            ax.axvspan(l, 1.0, facecolor='green', alpha=0.25, zorder=1)
        for l in pbias_lims:
            ax.axhspan(l, -l, facecolor='green', alpha=0.25, zorder=1)
        ax.set_xlabel('KGE')
        ax.set_ylabel('PBIAS')
        ax.set_xlim(-0.5,1)
        ax.set_ylim(-75,75)
        ax.set_xticks([-0.41,0,0.3,0.5,0.75,1])
        ax.set_yticks([-75,-50,-25,-10,0,10,25,50,75])
        ax.legend(loc='lower left', title='Performance NSE', title_fontsize=10, prop = {'size':9},
                   markerscale=0.5, labelspacing=0.2, handletextpad=0.1, frameon=True, borderpad=0.5, borderaxespad=0.5)
        ax.text(-0.48, 68, 'a)')
        
        ax = fig.add_subplot(2, 2, 3)
        sns.scatterplot(x='kge', y='nashsutcliffe', hue='pbias_class', data=df, ax=ax, palette=pallete_class,
                        s=0.75, linewidth=0, zorder=5)
        for l in kge_lims:
            ax.axvspan(l, 1.0, facecolor='green', alpha=0.25, zorder=1)
        for l in nse_lims:
            ax.axhspan(l, 1.0, facecolor='green', alpha=0.25, zorder=1)
        ax.set_xlabel('KGE')
        ax.set_ylabel('NSE')
        ax.set_xlim(-0.5,1)
        ax.set_ylim(-0.5,1)
        ax.set_xticks([-0.41,0,0.3,0.5,0.75,1])
        ax.set_yticks([-0.5,0,0.5,0.65,0.75,1])
        ax.legend(loc='lower left', title='Performance PBIAS', title_fontsize=10, prop = {'size':9},
                   markerscale=0.5, labelspacing=0.2, handletextpad=0.1, frameon=True, borderpad=0.5, borderaxespad=0.5)
        ax.text(-0.48, 0.93, 'b)')
        
        ax = fig.add_subplot(2, 2, 2)
        sns.scatterplot(x='nashsutcliffe', y='pbias', hue='rsr_class', data=df, ax=ax, palette=pallete_class,
                        s=0.75, linewidth=0, zorder=5)
        for l in pbias_lims:
            ax.axhspan(l, -l, facecolor='green', alpha=0.25, zorder=1)
        for l in nse_lims:
            ax.axvspan(l, 1.0, facecolor='green', alpha=0.25, zorder=1)
        ax.set_xlabel('NSE')
        ax.set_ylabel('PBIAS')
        ax.set_xlim(-0.5,1)
        ax.set_ylim(-75,75)
        ax.set_xticks([-0.5,0,0.5,0.65,0.75,1])
        ax.set_yticks([-75,-50,-25,-10,0,10,25,50,75])
        ax.legend(loc='lower left', title='Performance KGE', title_fontsize=10, prop = {'size':9},
                   markerscale=0.5, labelspacing=0.2, handletextpad=0.1, frameon=True, borderpad=0.5, borderaxespad=0.5)
        ax.text(0.93, 68, 'c)')
        
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter3D(df['kge'], df['pbias'], df['nashsutcliffe'], s=0.2, linewidth=0, alpha=0.7, c=df['rsr_class_value'], cmap='jet_r')
        # ax.plot_surface(X1, Y1, Z1, alpha=0.2, color='blue')
        # ax.plot_surface(X2, Y2, Z2, alpha=0.2, color='red')
        ax.set_xlim(-0.5,1)
        ax.set_ylim(-100,100)
        ax.set_zlim(-0.5,1)
        ax.set_xlabel('KGE')
        ax.set_ylabel('PBIAS')
        ax.set_zlabel('NSE')
        ax.view_init(0, 180)
        
        plt.subplots_adjust(wspace=0.15, hspace=0.15)
        plt.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/3ERRORS_All.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        # plt.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/3ERRORS_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        # plt.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/3_ERRORS/3ERRORS_{ds}_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        plt.close()



###########################################################################################
#Make 3D Plot

# vmin_dict = {'kge':-0.4, 'nashsutcliffe':0, 'correlationcoefficient':-1, 'pbias':-30, 'rsr':0, 'rmse':0, 'FB':0, 'POD':0, 'FAR':0, 'TS':0, 'ETS':0}
# vmax_dict = {'kge':1, 'nashsutcliffe':1, 'correlationcoefficient':1, 'pbias':30, 'rsr':1.4, 'rmse':25, 'FB':1, 'POD':1, 'FAR':1, 'TS':1, 'ETS':1}
error_color = 'nashsutcliffe'

limits = mpl.colors.Normalize(vmin=vmin_dict[error_color], vmax=vmax_dict[error_color], clip=False)

# Run command to show interactive plot or default view
# %matplotlib qt
# %matplotlib inline


df.dropna(inplace=True)

Y1, Z1 = np.meshgrid(np.linspace(-100,100,100), np.linspace(0,1,100))
X1 = 0 * np.ones((100, 100)) + 0.3

X2, Z2 = np.meshgrid(np.linspace(-0.41,1,100), np.linspace(0,1,100))
Y2 = 0 * np.ones((100, 100)) + 25

plt.rcParams.update(plt.rcParamsDefault)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(df['kge'], df['pbias'], df['nashsutcliffe'], s=0.2, linewidth=0, alpha=0.7, c=df['rsr_class_value'], cmap='jet_r')
# ax.plot_surface(X1, Y1, Z1, alpha=0.2, color='blue')
# ax.plot_surface(X2, Y2, Z2, alpha=0.2, color='red')
ax.set_xlim(-0.5,1)
ax.set_ylim(-100,100)
ax.set_zlim(-0.5,1)
ax.set_xlabel('KGE')
ax.set_ylabel('PBIAS')
ax.set_zlabel('NSE')
ax.view_init(35, 135)
# fig.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR3D/errors3D.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.close()