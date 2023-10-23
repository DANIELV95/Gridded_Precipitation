# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 13:23:54 2023

@author: HIDRAULICA-Dani
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D
import proplot as pplt
import geopandas as gpd

datasets_all = os.listdir('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/PRECIP/')

periods_dict = {0:'1981-1991', 1:'1991-2001', 2:'2001-2011', 3:'2011-2021'}
timesteps = ['d', 'm', 'mmax', 'y', 'ymax']
timesteps_names = {'d':'Daily', 'm':'Monthly', 'mmax':'Max Monthly', 'y':'Yearly', 'ymax':'Max Yearly'}

errors_list = ['nashsutcliffe', 'kge', 'rsr', 'pbias']
errors_dict = {'nashsutcliffe':'NSE', 'kge':'KGE', 'rsr':'RSR', 'pbias':'PBIAS'}

df_periods = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/len_ests.csv', index_col='Unnamed: 0')
ests25 = df_periods[df_periods['Am']>25].index

ests = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/Catalogo_estaciones.csv', encoding='latin-1')
ests.set_index('Unnamed: 0', inplace=True)
ests.index.name = None
ests.sort_index(inplace=True)
# ests.columns
################################################################################
# Error comparison for all stations in Mexico in one plot
# Compare different datasets and errors for same period and timestep

ests_loc = ests[['LON', 'LAT']][ests.index.isin(ests25)]

shpMx = "D:/DANI/2023/TEMA_TORMENTAS/DATOS/GEE/SHP/Border_Mx.shp"
sMx = gpd.read_file(shpMx, bbox=None, mask=None, rows=None)
sMx.crs = {'init':"epsg:4326"}

ds_order = ['Pchirps', 'Pdaymet', 'Pera', 'Pgldas', 'Pgpm', 'Plivneh', 'Ppersiann']
ds_names_dict = {'Pchirps':'CHIRPS', 'Pdaymet':'Daymet', 'Pera':'ERA5L', 'Pgldas':'GLDAS', 'Pgpm':'IMERG', 'livneh':'Livneh', 'Ppersiann':'PERSIANN'}
ds_names = ['CHIRPS', 'Daymet', 'ERA5L', 'GLDAS', 'IMERG', 'Livneh', 'PERSIANN']

cmap1 = LinearSegmentedColormap.from_list('r2b',["red", "yellow", "cyan", "darkblue"], N=256)
cmap2 = LinearSegmentedColormap.from_list('r2b_r',["darkblue", "cyan", "yellow", "red"], N=256)
cmap3 = LinearSegmentedColormap.from_list('rbg',["red", "magenta", "darkblue", "cyan", "green"], N=256)
cmap4 = LinearSegmentedColormap.from_list('red',["red", "red"], N=256)
# plt.cm.get_cmap("jet_r")

cmaps = {'nashsutcliffe':cmap1, 'kge':cmap1, 'rsr':cmap2, 'pbias':cmap3}
# vmin_dict = {'kge':-0.41, 'nashsutcliffe':0, 'correlationcoefficient':-1, 'pbias':-25, 'rsr':0, 'rmse':0, 'FB':0, 'POD':0, 'FAR':0, 'TS':0, 'ETS':0}
# vmax_dict = {'kge':1, 'nashsutcliffe':1, 'correlationcoefficient':1, 'pbias':25, 'rsr':1, 'rmse':25, 'FB':1, 'POD':1, 'FAR':1, 'TS':1, 'ETS':1}
vmin_dict = {'kge':-0.4, 'nashsutcliffe':0, 'correlationcoefficient':-1, 'pbias':-50, 'rsr':0, 'rmse':0, 'FB':0, 'POD':0, 'FAR':0, 'TS':0, 'ETS':0}
vmax_dict = {'kge':1, 'nashsutcliffe':1, 'correlationcoefficient':1, 'pbias':50, 'rsr':1.4, 'rmse':25, 'FB':1, 'POD':1, 'FAR':1, 'TS':1, 'ETS':1}

model_perf_dict = {1:'Unsatisfactory', 2:'Satisfactory', 3:'Good', 4:'Very Good'}

for period in periods_dict:
    # period = 2
    print(periods_dict[period])
    for timestep in timesteps:
        # timestep = 'd'
        print(timestep)

        fig, ax = pplt.subplots(ncols=4, nrows=7, share=False, figwidth=8, figheight=9.5, wspace=0.3, hspace=0.8)
        
        col = -1
        for error in errors_list:
            print(error)
            # error = 'nashsutcliffe'
            col += 1
            error_df = pd.read_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/P_{periods_dict[period]}_E_{error}_T_{timestep}.csv', index_col=['Unnamed: 0'])
            df1 = pd.concat([ests_loc['LON'], ests_loc['LAT'], error_df], axis=1)
            df1.replace([np.inf, -np.inf], np.nan, inplace=True)
            df1.dropna(subset=error_df.columns, how='all', inplace=True)
            
            Q1 = df1[error_df.columns].quantile(0.25)
            Q3 = df1[error_df.columns].quantile(0.75)
            IQR = Q3 - Q1 #IQR is interquartile range.
            
            filter_lonlat = df1[['LON', 'LAT']].notnull()
            filter_outliers = (df1[error_df.columns] >= Q1 - 1.5 * IQR) & (df1[error_df.columns] <= Q3 + 1.5 *IQR)
            
            df = df1[pd.concat([filter_lonlat, filter_outliers], axis=1)]
            df.dropna(subset=error_df.columns, how='all', inplace=True)

            df_out = df1[pd.concat([filter_lonlat, ~filter_outliers], axis=1)]
            df_out.dropna(subset=error_df.columns, how='all', inplace=True)
            
            row = -1
            for ds in ds_order:
                # ds = 'Pera'
                row += 1
                df_out1 = df_out.dropna(subset=[ds])
                
                # fig, ax = pplt.subplots(ncols=4, nrows=7, share=False, figwidth=8, figheight=10, wspace=0.5, hspace=0.5)
                sMx.boundary.plot(ax=ax[row,col], alpha=1, lw=0.25, color='k', zorder=2)
                limits = mpl.colors.Normalize(vmin=vmin_dict[error], vmax=vmax_dict[error], clip=False)
                
                if error == 'pbias':
                    df_plot = ax[row,col].scatter(df['LON'], df['LAT'], c=df[ds], cmap=cmaps[error], norm=limits, marker='.', mew=0.1,
                                                  s=2.5, alpha=1, vmin=vmin_dict[error], vmax=vmax_dict[error]) #, levels=pplt.arange(vmin_dict[error], vmax_dict[error], 4))
                elif error == 'rsr':
                    df_plot = ax[row,col].scatter(df['LON'], df['LAT'], c=df[ds], cmap=cmaps[error], norm=limits, marker='.', mew=0.1,
                                                  s=2.5, alpha=1, vmin=vmin_dict[error], vmax=vmax_dict[error]) #, levels=pplt.arange(vmin_dict[error], vmax_dict[error], 0.1))
                elif error == 'kge':
                    df_plot = ax[row,col].scatter(df['LON'], df['LAT'], c=df[ds], cmap=cmaps[error], norm=limits, marker='.', mew=0.1,
                                                  s=2.5, alpha=1, vmin=vmin_dict[error], vmax=vmax_dict[error]) #, levels=pplt.arange(vmin_dict[error], vmax_dict[error], 0.1))
                elif error == 'nashsutcliffe':
                    df_plot = ax[row,col].scatter(df['LON'], df['LAT'], c=df[ds], cmap=cmaps[error], norm=limits, marker='.', mew=0.1,
                                                  s=2.5, alpha=1, vmin=vmin_dict[error], vmax=vmax_dict[error]) #, levels=pplt.arange(vmin_dict[error], vmax_dict[error], (vmax_dict[error]-vmin_dict[error])/16))
                
                ax[row,col].scatter(df_out1['LON'], df_out1['LAT'], c='darkred', marker='x', s=0.075, mew=0.1)
                
                if error != 'pbias':
                    mean_value = round(df[ds].mean(), 2) if not pd.isnull(df[ds].mean()) else 'NA'
                else:
                    mean_value = round(abs(df[ds]).mean(), 1) if not pd.isnull(df[ds].mean()) else 'NA'
                
                if mean_value != 'NA':
                    if error == 'nashsutcliffe':
                        performance = (4 if mean_value>=0.75 else
                                       3 if ((mean_value>=0.65)&(mean_value<0.75)) else
                                       2 if ((mean_value>=0.5)&(mean_value<0.65)) else
                                       1)
                        mean_performance = model_perf_dict[performance]
                    elif error == 'kge':
                        performance = (4 if mean_value>=0.65 else
                                       3 if ((mean_value>=0.5)&(mean_value<0.65)) else
                                       2 if ((mean_value>=0.3)&(mean_value<0.5)) else
                                       1)
                        mean_performance = model_perf_dict[performance]
                    elif error == 'rsr':
                        performance = (1 if mean_value>0.7 else
                                       2 if ((mean_value>0.6)&(mean_value<=0.7)) else
                                       3 if ((mean_value>0.5)&(mean_value<=0.6)) else
                                       4)
                        mean_performance = model_perf_dict[performance]
                    elif error == 'pbias':
                        performance = (4 if abs(mean_value)<=10 else
                                       3 if ((abs(mean_value)>10)&(abs(mean_value)<=15)) else
                                       2 if ((abs(mean_value)>15)&(abs(mean_value)<=25)) else
                                       1)
                        mean_performance = model_perf_dict[performance]
                else:
                    mean_performance = 'NA'
                    
                if error != 'pbias':
                    ax[row,col].text(-118, 15, (r'$\overline {x}$ = ' + f'{mean_value}\n'+
                                     f'P: {mean_performance}'),
                                     size='small')
                else:
                    if mean_value != 'NA':
                        ax[row,col].text(-118, 15, (r'|$\overline {x}$| = ' + f'{mean_value}%\n'+
                                         f'P: {mean_performance}'),
                                         size='small')
                    else:
                        ax[row,col].text(-118, 15, (r'|$\overline {x}$| = ' + f'{mean_value}\n'+
                                         f'P: {mean_performance}'),
                                         size='small')
            
            ax.format(grid=False,
                    toplabels=('NSE', 'KGE', 'RSR', 'PBIAS'),
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
            
            ticks_loc = np.linspace(vmin_dict[error], vmax_dict[error], 3, endpoint=True)
            cbar = fig.colorbar(df_plot, ticks=ticks_loc, loc='b', col=(col+1), shrink=0.8)
            if error == 'pbias':
                cbar.ax.set_xticklabels([f'<{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'>{vmax_dict[error]}'])
            elif error == 'rsr':
                cbar.ax.set_xticklabels([f'{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'>{vmax_dict[error]}'])
            else:
                cbar.ax.set_xticklabels([f'<{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'{vmax_dict[error]}'])
        
        # fig, ax = pplt.subplots(ncols=4, nrows=7, share=False, figwidth=8, figheight=9.5, wspace=0.4, hspace=1)
        # ticks_loc = np.linspace(vmin_dict[error], vmax_dict[error], 3, endpoint=True)
        # cbar = fig.colorbar(df_plot, ticks=ticks_loc, loc='b', col=4, shrink=0.8)
        # if ((row==6) & (col==0)):
        legend_elements = [Line2D([0], [0], marker='o', ls='', color='b', label='Station', markerfacecolor='b', markersize=3),
                           Line2D([0], [0], marker='x', ls='', color='darkred', label='Outlier', markerfacecolor='darkred', markersize=3)]
        plt.legend(ncol=1, handles=legend_elements, handletextpad=0, loc='upper right', bbox_to_anchor=(-3.9,1.5), frameon=False,
                   borderpad=-0.5, borderaxespad=1)
        
        plt.figtext(0.01, 0.96, f'{periods_dict[period]}\n{timesteps_names[timestep]}', size='large', weight='bold')
        fig.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Dataset_Error/Error_Comparsion_{periods_dict[period]}_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        plt.close()

##################################################################################
#Plot by model performance

colors_perf = ['red', 'cyan', 'blue', 'darkblue']
model_perf = ['Unsatisfactory', 'Satisfactory', 'Good', 'Very Good']
model_perf_score_dict = {'Unsatisfactory':-2, 'Satisfactory':1, 'Good':2, 'Very Good':4}
thres_perf = {'nashsutcliffe':['', '<0.5', '0.65', '0.75', '1'],
              'kge':['', '<0.3', '0.5', '0.65', '1'],
              'rsr':['', '>0.7', '0.6', '0.5', '0'],
              'pbias':['', '>±25', '±15', '±10', '0']}
thres_perf_float = {'nashsutcliffe':[0, 0.5, 0.65, 0.75, 1],
              'kge':[-0.41, 0.3, 0.5, 0.65, 1],
              'rsr':[1, 0.7, 0.6, 0.5, 0],
              'pbias':[50, 25, 15, 10, 0]}
# model_perf = ['U', 'S', 'G', 'VG']
pd_count = pd.DataFrame(index=[1,2,3,4])

for period in periods_dict:
    # period = 0
    print(periods_dict[period])
    for timestep in timesteps:
        # timestep = 'd'
        print(timestep)
        
        fig, ax = pplt.subplots(ncols=4, nrows=7, share=False, figwidth=8, figheight=9.5, wspace=0.2, hspace=0.5)
        
        col = -1
        for error in errors_list:
            print(error)
            # error = 'kge'
            col += 1
            error_df = pd.read_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/P_{periods_dict[period]}_E_{error}_T_{timestep}.csv', index_col=['Unnamed: 0'])
            
            if error == 'nashsutcliffe':
                error_df[error_df>=0.75] = 4
                error_df[((error_df>=0.65)&(error_df<0.75))] = 3
                error_df[((error_df>=0.5)&(error_df<0.65))] = 2
                error_df[error_df<0.5] = 1
            elif error == 'kge':
                error_df[error_df>=0.65] = 4
                error_df[((error_df>=0.5)&(error_df<0.65))] = 3
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
            
            df = pd.concat([ests_loc['LON'], ests_loc['LAT'], error_df], axis=1)
            df.dropna(subset=error_df.columns, how='all', inplace=True)
            
            row = -1
            for ds in ds_order:
                # ds = 'Pgldas'
                row += 1
                
                # fig, ax = pplt.subplots(ncols=4, nrows=7, share=False, figwidth=8, figheight=10, wspace=0.5, hspace=0.5)
                sMx.boundary.plot(ax=ax[row,col], alpha=1, lw=0.2, color='k', zorder=2)
                # limits = mpl.colors.Normalize(vmin=vmin_dict[error], vmax=vmax_dict[error], clip=False)
                df_plot = ax[row,col].scatter(df['LON'], df['LAT'], c=df[ds], cmap=ListedColormap(colors_perf),
                                              marker='.', mew=0.1, s=2, alpha=1, levels=pplt.arange(1,5,1))
                count_df = pd.concat([pd_count, df.groupby([ds]).count()['LON'].astype(int).to_frame()], axis=1)
                count_df.columns = ['count']
                count_df[count_df.isnull()] = int(0)
                U = int(count_df.loc[1].values[0])
                S = int(count_df.loc[2].values[0])
                G = int(count_df.loc[3].values[0])
                VG = int(count_df.loc[4].values[0])
                Score = round((U*-2 + S*1 + G*2 + VG*4)/(U+S+G+VG),2) if (U+S+G+VG) != 0 else 'NA'
                
                if Score != 'NA':
                        performance = (4 if Score>=3 else
                                       3 if ((Score>=1.5)&(Score<3)) else
                                       2 if ((Score>0.0)&(Score<1.5)) else
                                       1)
                        mean_performance = model_perf_dict[performance]
                else:
                    mean_performance = 'NA'
                
                ax[row,col].text(-96, 25, (f'U = {str(U)}\n'+
                                 f'S = {str(S)}\n'+
                                 f'G = {str(G)}\n'+
                                 f'VG = {str(VG)}'),
                                 size='small')
                ax[row,col].text(-118, 15, (f'Score = {Score}\n'+
                                 f'P: {mean_performance}'),
                                 size='small')
                
            ax.format(grid=False,
                    toplabels=('NSE', 'KGE', 'RSR', 'PBIAS'),
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
            
            cbar = fig.colorbar(df_plot, loc='b', col=(col+1), shrink=0.8)
            loc = np.arange(1,6,1)
            cbar.set_ticks(loc)
            cbar.set_ticklabels(thres_perf[error])
            cbar.set_label('Performance')
            spaces=' '*4
            cbar.ax.text(1.3, 1.6, f'U {spaces} S {spaces} G {spaces}VG', size='large', weight='bold', color='g')
            
        plt.figtext(0.005, 0.96, f'{periods_dict[period]}\n{timesteps_names[timestep]}', size='large', weight='bold')
        
        fig.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Dataset_Error_Class/Performance_Error_Comparsion_{periods_dict[period]}_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        plt.close()

##################################################################################
#Plot all errors for each dataset per image

for error in errors_list:
    # error = 'pbias'
    for ds in ds_order[:3]:
        # ds = 'Pdaymet'
        print(error, ds)

        fig, ax = pplt.subplots(ncols=4, nrows=5, share=False, figwidth=9, figheight=6.5, wspace=0.2, hspace=0.5)
        col = -1
        for period in periods_dict:
            # period = 2
            print(periods_dict[period])
            col += 1
            row = -1
            for timestep in timesteps:
                # timestep = 'd'
                print(timestep)
                row += 1
                
                error_df = pd.read_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/P_{periods_dict[period]}_E_{error}_T_{timestep}.csv', index_col=['Unnamed: 0'])
                
                df1 = pd.concat([ests_loc['LON'], ests_loc['LAT'], error_df], axis=1)
                df1.replace([np.inf, -np.inf], np.nan, inplace=True)
                df1.dropna(subset=error_df.columns, how='all', inplace=True)
                
                Q1 = df1[error_df.columns].quantile(0.25)
                Q3 = df1[error_df.columns].quantile(0.75)
                IQR = Q3 - Q1 #IQR is interquartile range.
                
                filter_lonlat = df1[['LON', 'LAT']].notnull()
                filter_outliers = (df1[error_df.columns] >= Q1 - 1.5 * IQR) & (df1[error_df.columns] <= Q3 + 1.5 *IQR)
                
                df = df1[pd.concat([filter_lonlat, filter_outliers], axis=1)]
                df.dropna(subset=error_df.columns, how='all', inplace=True)

                df_out = df1[pd.concat([filter_lonlat, ~filter_outliers], axis=1)]
                df_out.dropna(subset=error_df.columns, how='all', inplace=True)
                df_out1 = df_out.dropna(subset=[ds])
                
                dsmin = []
                for ds in ds_order:
                    dsmin.append(df_out1.index[df_out1[ds]==df_out1[ds].min()].values[0])
                    
                plt.plot(df_out1[df_out1.index.isin(dsmin)][error_df.columns])
                
                # fig, ax = pplt.subplots(ncols=4, nrows=7, share=False, figwidth=8, figheight=10, wspace=0.5, hspace=0.5)
                sMx.boundary.plot(ax=ax[row,col], alpha=1, lw=0.2, color='k', zorder=2)
                limits = mpl.colors.Normalize(vmin=vmin_dict[error], vmax=vmax_dict[error], clip=False)
                df_plot = ax[row,col].scatter(df['LON'], df['LAT'], c=df[ds], cmap=cmaps[error], norm=limits, marker='.', mew=0.1,
                                              s=2, alpha=1, levels=pplt.arange(vmin_dict[error], vmax_dict[error], (vmax_dict[error]-vmin_dict[error])/11))
                ax[row,col].scatter(df_out1['LON'], df_out1['LAT'], c='darkred', marker='x', s=0.7, mew=0.1)
                
                if error != 'pbias':
                    mean_value = round(df[ds].mean(), 2) if not pd.isnull(df[ds].mean()) else 'NA'
                else:
                    mean_value = round(abs(df[ds]).mean(), 1) if not pd.isnull(df[ds].mean()) else 'NA'
                
                if mean_value != 'NA':
                    if error == 'nashsutcliffe':
                        performance = (4 if mean_value>=0.75 else
                                       3 if ((mean_value>=0.65)&(mean_value<0.75)) else
                                       2 if ((mean_value>=0.5)&(mean_value<0.65)) else
                                       1)
                        mean_performance = model_perf_dict[performance]
                    elif error == 'kge':
                        performance = (4 if mean_value>=0.65 else
                                       3 if ((mean_value>=0.5)&(mean_value<0.65)) else
                                       2 if ((mean_value>=0.3)&(mean_value<0.5)) else
                                       1)
                        mean_performance = model_perf_dict[performance]
                    elif error == 'rsr':
                        performance = (1 if mean_value>0.7 else
                                       2 if ((mean_value>0.6)&(mean_value<=0.7)) else
                                       3 if ((mean_value>0.5)&(mean_value<=0.6)) else
                                       4)
                        mean_performance = model_perf_dict[performance]
                    elif error == 'pbias':
                        performance = (4 if abs(mean_value)<=10 else
                                       3 if ((abs(mean_value)>10)&(abs(mean_value)<=15)) else
                                       2 if ((abs(mean_value)>15)&(abs(mean_value)<=25)) else
                                       1)
                        mean_performance = model_perf_dict[performance]
                else:
                    mean_performance = 'NA'
                
                if error != 'pbias':
                    ax[row,col].text(-118, 15, (r'$\overline {x}$ = ' + f'{mean_value}\n'+
                                     f'P: {mean_performance}'),
                                     size='small')
                else:
                    if mean_value != 'NA':
                        ax[row,col].text(-118, 15, (r'|$\overline {x}$| = ' + f'{mean_value}%\n'+
                                         f'P: {mean_performance}'),
                                         size='small')
                    else:
                        ax[row,col].text(-118, 15, (r'|$\overline {x}$| = ' + f'{mean_value}\n'+
                                         f'P: {mean_performance}'),
                                         size='small')
                
        ax.format(grid=False,
                toplabels=periods_dict.values(),
                leftlabels=timesteps_names.values(),
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
        
        ticks_loc = np.linspace(vmin_dict[error], vmax_dict[error], 3, endpoint=True)
        cbar = fig.colorbar(df_plot, ticks=ticks_loc, loc='r')
        if error == 'pbias':
            cbar.set_ticklabels([f'<{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'>{vmax_dict[error]}'])
        elif error == 'rsr':
            cbar.set_ticklabels([f'{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'>{vmax_dict[error]}'])
        else:
            cbar.set_ticklabels([f'<{vmin_dict[error]}', f'{vmin_dict[error]+(vmax_dict[error]-vmin_dict[error])/2:.1f}', f'{vmax_dict[error]}'])
        # cbar.ax.tick_params(rotation=90)
        cbar.set_label(f'{errors_dict[error]}')
        
        plt.figtext(0.005, 0.95, f'{ds}', size='large', weight='bold')
        
        # cbar = fig.colorbar(df_plot, loc='r')
        # loc = np.arange(1,6,1)
        # cbar.set_ticks(loc)
        # cbar.set_ticklabels(thres_perf[error])
        # cbar.set_label('Performance')
        
        # clb.ax.text(2.5, 0.5, "SECONDLABEL", fontsize=10, rotation=90, va='center')
        
        fig.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Period_Timestep/Error_Comparsion_{ds}_{error}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        plt.close()

# import gc
# gc.collect()
##################################################################################
#Boxplots and histograms
import seaborn as sns

# ds_order = ['Pchirps', 'Pdaymet', 'Pera', 'Pgldas', 'Pgpm', 'Plivneh', 'Ppersiann']

for period in periods_dict:
    # period = 1
    print(periods_dict[period])
    for timestep in timesteps:
        # timestep = 'd'
        print(timestep)
        for error in errors_list:
            print(error)
            # error = 'kge'
            error_df = pd.read_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/P_{periods_dict[period]}_E_{error}_T_{timestep}.csv', index_col=['Unnamed: 0'])
            df = pd.concat([ests_loc['LON'], ests_loc['LAT'], error_df], axis=1)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=error_df.columns, how='all', inplace=True)
            # df.drop([3026, 15378], inplace=True)
            Q1 = df[error_df.columns].quantile(0.25)
            Q3 = df[error_df.columns].quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_top = pd.concat([Q3 + 1.5 * IQR, df[error_df.columns].max()], axis=1)
            outlier_bottom = pd.concat([Q1 - 1.5 * IQR, df[error_df.columns].min()], axis=1)
            outlier_top[3] = outlier_top.T.min()
            outlier_bottom[3] = outlier_bottom.T.max()
            outlier_top_lim = outlier_top[3]
            outlier_bottom_lim = outlier_bottom[3]
            bp_lims = (outlier_bottom_lim.min() - abs(outlier_bottom_lim.min())*0.1, outlier_top_lim.max() + abs(outlier_top_lim.max())*0.1)
            # mask_out_bottom = df[error_df.columns][df.lt(outlier_bottom_lim)].dropna(how='all').count()
            # mask_out_bottom_list = df[error_df.columns][df.lt(outlier_bottom_lim)].dropna(how='all').index
            
            # boxplot = sns.boxplot(df[error_df.columns], orient='horizontal', fliersize=1, showfliers=False)
            # bp_lims = boxplot.get_xlim()
            # plt.close()
            
            boxplot = sns.boxplot(df[error_df.columns], orient='horizontal', fliersize=1, showfliers=True)
            plt.xlim(bp_lims)
            plt.xlabel(f'{errors_dict[error]}')
            plt.figtext(0.02, 0.01, f'{periods_dict[period]}\n{timesteps_names[timestep]}', size='medium', weight='bold')
            plt.tight_layout(pad=0.5)
            plt.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Boxplots/Boxplot_{error}_{periods_dict[period]}_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
            plt.close()
            
        # fig, ax = plt.subplots(nrows=2, ncols=2)
        # row, col = -1, 0
        # for error in errors_list:
        #     print(error)
        #     row += 1
        #     if row > 1:
        #         row = 0
        #         col += 1
        #     # print(row, col)
            

for timestep in timesteps:
    # timestep = 'd'
    print(timestep)
    for period in periods_dict:
        # period = 1
        print(periods_dict[period])
        fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(10, 5))
        col = -1
        for error in errors_list:
            print(error)
            col += 1
            error_df = pd.read_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/P_{periods_dict[period]}_E_{error}_T_{timestep}.csv', index_col=['Unnamed: 0'])
            df = pd.concat([ests_loc['LON'], ests_loc['LAT'], error_df], axis=1)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=error_df.columns, how='all', inplace=True)
            Q1 = df[error_df.columns].quantile(0.25)
            Q3 = df[error_df.columns].quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_top = pd.concat([Q3 + 1.5 * IQR, df[error_df.columns].max()], axis=1)
            outlier_bottom = pd.concat([Q1 - 1.5 * IQR, df[error_df.columns].min()], axis=1)
            outlier_top[3] = outlier_top.T.min()
            outlier_bottom[3] = outlier_bottom.T.max()
            outlier_top_lim = outlier_top[3]
            outlier_bottom_lim = outlier_bottom[3]
            bp_lims = (outlier_bottom_lim.min() - abs(outlier_bottom_lim.min())*0.1, outlier_top_lim.max() + abs(outlier_top_lim.max())*0.1)
            
            boxplot = sns.boxplot(df[error_df.columns], ax=ax[col], orient='horizontal', fliersize=1, showfliers=True, position=0)
            ax[col].set_xlim(bp_lims)
            ax[col].set_xlabel(errors_dict[error])    
            ax[col].tick_params(axis='y', which='minor', left=False)
            
        plt.tight_layout(pad=0.5)
        fig.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Boxplots/Boxplots_{periods_dict[period]}_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        plt.close()
        
#################################################################################
#Get all data in 1 file by error 
cols_order = ['id', 'Pchirps', 'Pdaymet', 'Pera', 'Pgldas', 'Pgpm', 'Plivneh', 'Ppersiann']
# cols_order_all = ['period', 'timestep', 'error', 'id', 'Pchirps', 'Pdaymet', 'Pera', 'Pgldas', 'Pgpm', 'Plivneh', 'Ppersiann']

for timestep in timesteps:
    # timestep = 'd'
    print(timestep)
    for error in errors_list:
        #  error = 'rsr'
        print(error)
        
        df_all = pd.DataFrame()
        
        for period in periods_dict:
            # period = 1
            print(periods_dict[period])
        
        
            cols_order_all = ['dataset', 'period', 'id', error]
            df = pd.DataFrame(columns=cols_order_all)

            error_df = pd.read_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/P_{periods_dict[period]}_E_{error}_T_{timestep}.csv')
            error_df.columns = cols_order
            
            for ds in ds_order:
                df_col = error_df[['id', ds]]
                df_col.columns = ['id', error]
                df_col['dataset'] = ds
                df = pd.concat([df, df_col], axis=0)
            df['period'] = periods_dict[period]
            
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=[error], inplace=True)
            df_all = pd.concat([df_all, df], axis=0)
        
        df_all.sort_values(by=['dataset', 'period', 'id'], inplace=True)
        df_all.reset_index(drop=True, inplace=True)
        df_all.to_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/All/Precip_Datasets_{error}_{timestep}.csv', encoding='latin-1', index=False)
            
#################################################################################
#Boxplots multiple comparisons
ds_names_dict2 = {'Pchirps':'CHIRPS', 'Pdaymet':'Daymet', 'Pera':'ERA5L', 'Pgldas':'GLDAS', 'Pgpm':'IMERG', 'Plivneh':'Livneh', 'Ppersiann':'PERSIANN'}

row = -1
fig, ax = plt.subplots(nrows=5, ncols=4, sharey=True, figsize=(10, 14))
for timestep in timesteps:
    # timestep = 'd'
    print(timestep)
    # fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(8, 5))
    row += 1
    col = -1
    for error in errors_list:
        # error = errors_list[0]
        # print(error)
        col += 1
        df = pd.read_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/All/Precip_Datasets_{error}_{timestep}.csv')
        
        # Change name of datasets
        for ds in ds_names_dict2:
            df.replace(ds, ds_names_dict2[ds], inplace=True)
        
        # fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(8, 5))
        boxplot = sns.boxplot(x=error, y='dataset', hue='period', data=df, ax=ax[row,col], orient='horizontal', 
                              fliersize=0.2, showfliers=False, linewidth=0.4, zorder=10)
        
        sns.boxplot(x=error, y='dataset', hue='period', data=df, orient='horizontal', whis=(1, 99), 
                              fliersize=1, showfliers=True, linewidth=0.4, zorder=10)
        
        if error == 'nashsutcliffe':
            ax[row,col].axvspan(0.5, 1.0, facecolor='g', alpha=0.25, zorder=-1)
            ax[row,col].set_xlim(boxplot.get_xlim()[0],1)
        elif error == 'kge':
            ax[row,col].axvspan(0.3, 1.0, facecolor='g', alpha=0.25, zorder=-1)
            ax[row,col].set_xlim(boxplot.get_xlim()[0],1)
        elif error == 'rsr':
            ax[row,col].axvspan(0, 0.5, facecolor='g', alpha=0.25, zorder=-1)
            ax[row,col].set_xlim(0,boxplot.get_xlim()[1])
        elif error == 'pbias':
            ax[row,col].axvspan(-25, 25, facecolor='g', alpha=0.25, zorder=-1)
            pbias_max = np.max([abs(boxplot.get_xlim()[0]),abs(boxplot.get_xlim()[1])])
            ax[row,col].set_xlim(-pbias_max,pbias_max)
        
        if col == 0:
            ax[row,col].set_ylabel(timesteps_names[timestep])
        else:
            ax[row,col].set_ylabel('')
        ax[row,col].tick_params(axis='y', which='major', left=True, right=True)
        ax[row,col].tick_params(axis='y', which='minor', left=False, right=False)
        
        if row == 0:
            ax[row,col].set_title(errors_dict[error])
        # if row == 4:
        #     ax[row,col].set_xlabel(errors_dict[error])
        ax[row,col].set_xlabel('')
        ax[row,col].tick_params(axis='x', which='both', top=True, bottom=True, labeltop=False, labelbottom=True)
        
        if row + col != 7: #((row != 4) & (col != 3)):
            ax[row,col].legend([],[], frameon=False)
        else:            
            ax[row,col].legend(ncol=4, bbox_to_anchor=(0.02, -0.15))
        
        ax[row,col].grid(visible=None)#axis='x', color='r', linestyle='-', linewidth=10)
            
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# plt.figtext(0.06, 0, f'Time series:\n{timesteps_names[timestep]}', size='medium', weight='bold')
# fig.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Boxplots/Boxplots_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
fig.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Boxplots/Boxplots.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# fig.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Boxplots/Boxplots_fliers.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.close()

#################################################################################
#Boxplots multiple comparisons by region NOT FINISHED

ds_order = ['Pchirps', 'Pdaymet', 'Pera', 'Pgldas', 'Pgpm', 'Plivneh', 'Ppersiann']
ds_names_dict = {'Pchirps':'CHIRPS', 'Pdaymet':'Daymet', 'Pera':'ERA5L', 'Pgldas':'GLDAS', 'Pgpm':'IMERG', 'livneh':'Livneh', 'Ppersiann':'PERSIANN'}
ds_names = ['CHIRPS', 'Daymet', 'ERA5L', 'GLDAS', 'IMERG', 'Livneh', 'PERSIANN']
criteria_names = ['Basin', 'Climate', 'State', 'Elevation']

cmap1 = LinearSegmentedColormap.from_list('r2b',["red", "yellow", "cyan", "darkblue"], N=256)
cmap2 = LinearSegmentedColormap.from_list('r2b_r',["darkblue", "cyan", "yellow", "red"], N=256)
cmap3 = LinearSegmentedColormap.from_list('rbg',["red", "magenta", "darkblue", "cyan", "green"], N=256)
cmap4 = LinearSegmentedColormap.from_list('red',["red", "red"], N=256)
# plt.cm.get_cmap("jet_r")

cmaps = {'nashsutcliffe':cmap1, 'kge':cmap1, 'rsr':cmap2, 'pbias':cmap3}
# vmin_dict = {'kge':-0.41, 'nashsutcliffe':0, 'correlationcoefficient':-1, 'pbias':-25, 'rsr':0, 'rmse':0, 'FB':0, 'POD':0, 'FAR':0, 'TS':0, 'ETS':0}
# vmax_dict = {'kge':1, 'nashsutcliffe':1, 'correlationcoefficient':1, 'pbias':25, 'rsr':1, 'rmse':25, 'FB':1, 'POD':1, 'FAR':1, 'TS':1, 'ETS':1}
vmin_dict = {'kge':-0.4, 'nashsutcliffe':0, 'correlationcoefficient':-1, 'pbias':-30, 'rsr':0, 'rmse':0, 'FB':0, 'POD':0, 'FAR':0, 'TS':0, 'ETS':0}
vmax_dict = {'kge':1, 'nashsutcliffe':1, 'correlationcoefficient':1, 'pbias':30, 'rsr':1.4, 'rmse':25, 'FB':1, 'POD':1, 'FAR':1, 'TS':1, 'ETS':1}

model_perf_dict = {1:'Unsatisfactory', 2:'Satisfactory', 3:'Good', 4:'Very Good'}

dtypes_dict = {'id': int, 'timestep': object, 'period': object, 'dataset': object, 'nashsutcliffe': float,
               'nashsutcliffe_class': object, 'nashsutcliffe_outlier': bool, 'nashsutcliffe_alpha': float,
               'kge': float, 'kge_class': object, 'kge_outlier': bool, 'kge_alpha': float, 'rsr': float,
               'rsr_class': object, 'rsr_outlier': bool, 'rsr_alpha': float, 'pbias': float, 'pbias_class': object,
               'pbias_outlier': bool, 'pbias_alpha': float, 'Elevation': float, 'Z_Class': int,
               'Climate_ID': int, 'Climate_Name': object, 'State_ID': int, 'State_Name': object,
               'Basin_ID': int, 'Basin_Name': object}

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


ests = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/Catalogo_estaciones.csv', encoding='latin-1')
ests.set_index('Unnamed: 0', inplace=True)
ests.index.name = None
ests.sort_index(inplace=True)
ests_loc = ests[['LON', 'LAT']]
ests_loc.index.name = 'id'
df = df.merge(ests_loc.reset_index(), on='id')

df_mask = df[((df['timestep']==timesteps_names[timestep])&(df['period']=='2001-2011')&(df['dataset']=='CHIRPS'))][['kge', 'LON', 'LAT']]
df_mask = df[((df['timestep']==timesteps_names[timestep])&(df['period']=='2001-2011')&(df['dataset']=='CHIRPS'))][['dataset', 'Elevation', 'Z_Range', 'pbias', 'kge']]
df_mask = df[((df['timestep']==timesteps_names[timestep])&(df['period']=='2001-2011'))][['dataset', 'Z_Class', 'Z_Range', 'pbias', 'kge']]

df_mask = df_mask.sort_values(by=['dataset', 'Z_Class'])

df.columns
df_mask.plot(x='pbias', y='Elevation', kind='scatter', s=1)

p = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/PRECIP/livneh/Plivneh_17010.csv', parse_dates=['Unnamed: 0'])
p.set_index('Unnamed: 0', drop=True, inplace=True)
p.index.name = None
pm = p['Plivneh'].mean()

round(sum(p['Plivneh'] - pm)/sum(p['Plivneh']),1)

df_mask.hist('Elevation', bins=100, range=(0,3000))

df_mask.groupby(['dataset', 'Z_Range']).size().plot()

sns.boxplot(x="pbias", y="dataset", hue='Z_Range', data=df_mask, showfliers=False, orient='horizontal')
sns.violinplot(x="dataset", y="kge", hue='Z_Range', data=df_mask)

df_mask.hist(bins=100)

df_mask.plot(x='LON', y='rsr', kind='scatter', s=1)
df_mask.plot(x='kge', y='LAT', kind='scatter', s=1)

df.columns

df_group = df.groupby(['timestep', 'period', 'dataset', 'State_Name']).size()
df_group.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/stations_by_state.csv')

df_group = df.groupby(['timestep', 'period', 'dataset', 'Basin_Name']).size()
df_group.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/stations_by_basin.csv')

df_group = df.groupby(['timestep', 'period', 'dataset', 'Z_Range']).size()
df_group.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/stations_by_elevation.csv')

df_group = df.groupby(['timestep', 'period', 'dataset', 'Climate_Name']).size()
df_group.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/stations_by_climate.csv')


row = -1
fig, ax = plt.subplots(nrows=5, ncols=4, sharey=True, figsize=(10, 14))
for timestep in timesteps:
    # timestep = 'm'
    print(timestep)
    # fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(8, 5))
    row += 1
    col = -1
    for error in errors_list:
        # error = errors_list[1]
        # print(error)
        col += 1
        
        df_mask = df[((df['timestep']==timesteps_names[timestep])&(df['period']=='2001-2011'))]
        df_mask = df[df['period']=='2001-2011']
        
        fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(6, 10))
        boxplot = sns.boxplot(x=error, y='State_Name', hue='dataset', data=df, ax=ax, orient='horizontal', 
                              fliersize=0.5, showfliers=False, linewidth=0.4, zorder=2)
        
        if error == 'nashsutcliffe':
            ax.axvspan(0.5, 1.0, facecolor='g', alpha=0.25, zorder=1)
            ax.set_xlim(boxplot.get_xlim()[0],1)
        elif error == 'kge':
            ax.axvspan(0.3, 1.0, facecolor='g', alpha=0.25, zorder=1)
            ax.set_xlim(boxplot.get_xlim()[0],1)
        elif error == 'rsr':
            ax[row,col].axvspan(0, 0.5, facecolor='g', alpha=0.25, zorder=1)
            ax[row,col].set_xlim(0,boxplot.get_xlim()[1])
        elif error == 'pbias':
            ax[row,col].axvspan(-25, 25, facecolor='g', alpha=0.25, zorder=1)
            pbias_max = np.max([abs(boxplot.get_xlim()[0]),abs(boxplot.get_xlim()[1])])
            ax[row,col].set_xlim(-pbias_max,pbias_max)
        
        if col == 0:
            ax[row,col].set_ylabel(timesteps_names[timestep])
        else:
            ax[row,col].set_ylabel('')
        ax[row,col].tick_params(axis='y', which='major', left=True, right=True)
        ax[row,col].tick_params(axis='y', which='minor', left=False, right=False)
        
        if row == 0:
            ax[row,col].set_title(errors_dict[error])
        # if row == 4:
        #     ax[row,col].set_xlabel(errors_dict[error])
        ax[row,col].set_xlabel('')
        ax[row,col].tick_params(axis='x', which='both', top=True, bottom=True, labeltop=False, labelbottom=True)
        
        if row + col != 7: #((row != 4) & (col != 3)):
            ax[row,col].legend([],[], frameon=False)
        else:            
            ax[row,col].legend(ncol=4, bbox_to_anchor=(0.02, -0.15))
        
        ax[row,col].grid(visible=None)#axis='x', color='r', linestyle='-', linewidth=10)
            
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# plt.figtext(0.06, 0, f'Time series:\n{timesteps_names[timestep]}', size='medium', weight='bold')
# fig.savefig(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Boxplots/Boxplots_{timestep}.jpg', format='jpg', dpi=1000, bbox_inches='tight')
fig.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Boxplots/Boxplots.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.close()

#################################################################################
#Check error between multiple stations and 1 dataset and plot or Check errors with 1 day lag
from datetime import datetime
custom_date_parser = lambda x: datetime.strptime(x, "%d/%m/%Y")
start_date = '2001-01-01'
end_date = '2021-01-01'

ests = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/Catalogo_estaciones.csv', encoding='latin-1')
ests.set_index('Unnamed: 0', inplace=True)
ests.index.name = None
ests = ests.drop([26204, 26207, 26208])
ests.sort_index(inplace=True)
# ests25 = ests[ests['PERIODO']>=25]
ests

errors_df = pd.DataFrame()

ests_sample = ests.sample(100).index
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10,5), gridspec_kw={'height_ratios': [2, 1]})
for est in ests_sample:
    try:
        # est = ests.index[0]
        print(est)
        df1 = pd.read_csv(f'D:/DANI/VARIOS/CNA_EST/OTROS/ESTACIONES/PRECIP/P_{str(est)}.csv', index_col=['Unnamed: 0'], date_parser=custom_date_parser)
        df1.columns = ['Pcna']
        # mask1 = ((df1.index >= start_date) & (df1.index < end_date))
        
        df2 = pd.read_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/PRECIP/daymet/Pdaymet_{str(est)}.csv', parse_dates=['Unnamed: 0'])
        df2.set_index('Unnamed: 0', inplace=True)
        df2.index.name = None
        # mask2 = ((df2.index >= start_date) & (df2.index < end_date))
        
        df = pd.concat([df1['Pcna'], df2['Pdaymet']], axis=1).dropna()
        df2.index = df2.index + pd.Timedelta(days=1000)
        
        df = pd.concat([df, df2['Pdaymet']], axis=1).dropna()
        df.columns = ['Pcna', 'Pdaymet', 'Pdaymet_lag']
        
        df = df[df>0].dropna(how='all')
        df.replace(np.nan, 0, inplace=True)
        
        error = (df['Pcna']-df['Pdaymet'])
        ma_error = abs(error).rolling(7).mean()
        
        ax[0].plot(error, lw=1, alpha=0.8)
        ax[1].plot(ma_error, lw=1, alpha=0.8)
        
        error_lag = (df['Pcna']-df['Pdaymet_lag'])
        ma_error_lag = abs(error_lag).rolling(7).mean()
        # plt.plot(abs(error), lw=1, alpha=0.8)
        
        ax[0].plot(error_lag, lw=1, alpha=0.8)
        ax[1].plot(ma_error_lag, lw=1, alpha=0.8)
        
        error_df = pd.DataFrame([[error.mean(), ma_error.mean(), error_lag.mean(), ma_error_lag.mean()]], index=[est], columns=['error', 'error_ma', 'error_lag', 'error_ma_lag'])
        errors_df = pd.concat([errors_df, error_df], axis=0)
        
    except Exception:
        pass

errors_df.sort_index(inplace=True)

errors_df[['error_ma', 'error_ma_lag']].plot() #ls='', marker='.')
errors_df.plot(kind='box')

errors_df.cumsum()


plt.plot(df1[mask1])
plt.plot(df2[mask2])


###############################################################################
#Get scores with proposed values combining multiple errors
import itertools

model_perf = ['Unsatisfactory', 'Satisfactory', 'Good', 'Very Good']
model_perf_score = [-2, 1, 2, 4]
combs_with_replacement = list(itertools.combinations_with_replacement(model_perf_score, 4))
combs = pd.DataFrame(combs_with_replacement)
combs['Points'] = combs.sum(axis=1)
combs['Score'] = combs['Points']/4

for value in combs['Points']:
    # value = combs['Points'][0]
    print(value)
    if value <= 0:
        combs.loc[combs['Points']==value, 'Performance'] = 'Unsatisfactory'
    elif value < 6:
        combs.loc[combs['Points']==value, 'Performance'] = 'Satisfactory'
    elif value < 12:
        combs.loc[combs['Points']==value, 'Performance'] = 'Good'
    else:
       combs.loc[combs['Points']==value, 'Performance'] = 'Very Good'

combs.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Total_combinations_points.csv', index=False)
# (-8,-1),(0,5),(6,11),(12,16)
