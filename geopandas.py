# -*- coding: utf-8 -*-
"""
Created on Tue May 16 14:09:47 2023

@author: HIDRAULICA-Dani
"""

import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import proplot as pplt
import datetime
from shapely.geometry import Point
import contextily as ctx
import spotpy
from dateutil.relativedelta import relativedelta
from cycler import cycler
from de import de
from de import util
import sklearn
import gc
from itertools import groupby
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

#####################################
import winsound
duration = 1000  # milliseconds
freq = 1000  # Hz
winsound.Beep(freq, duration)
#####################################

os.chdir('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables')
os.listdir()

#################################################################################################################

def skill_scores(evaluation, simulation, thres=0):
    try:
        eva = evaluation.copy()
        eva[evaluation>thres] = 1
        eva[evaluation<=thres] = 0
        
        sim = simulation.copy()
        sim[simulation>thres] = 1
        sim[simulation<=thres] = 0

        tf = pd.concat([eva, sim], axis=1)
        tf.columns = ['eva', 'sim']
        tf = tf.dropna()

        N = len(tf)
        H = len(tf[((tf['eva']==1) & (tf['sim']==1))]) #Hit, observed rain correctly detected
        M = len(tf[((tf['eva']==1) & (tf['sim']==0))]) #Miss, observed rain not detected
        F = len(tf[((tf['eva']==0) & (tf['sim']==1))]) #False alarm, rain detected but not observed
        Nu = len(tf[((tf['eva']==0) & (tf['sim']==0))]) #Null, no rain observed nor detected

        FB = (H+F)/(H+M) # Frequency bias 1
        POD = H/(H+M) # Probability of detection 1
        FAR = F/(H+F) # False alarm ratio 0
        TS = H/(F+M+H) # Critical success index or Threat score 1 
        Hc = (H+M)*(H+F)/N #Chance hits
        ETS = (H-Hc)/(M+F+H-Hc) # Equitable threat score or Gilbert skill score 1
    except Exception:
        FB = np.nan
        POD = np.nan
        FAR = np.nan
        TS = np.nan
        ETS = np.nan
    return [('FB',FB), ('POD',POD), ('FAR',FAR), ('TS',TS), ('ETS',ETS)]

#################################################################################################################

years = range(1951,2023)
datasets_names = ['era', 'gpm', 'gldas', 'daymet', 'persiann', 'trmm', 'chirps']
# datasets = {'era':era, 'gpm':gpm, 'gldas':gldas, 'daymet':daymet, 'persiann':persiann, 'trmm':trmm, 'chirps':chirps}
scales = {'era':11132, 'gpm':11132, 'gldas':27830, 'daymet':1000, 'persiann':27830, 'trmm':27830, 'chirps':5566}
precip_var_names = {'era':'total_precipitation_hourly', 'gpm':'precipitationCal', 'gldas':'Rainf_f_tavg', 'daymet':'prcp', 'persiann':'precipitation', 'trmm':'precipitation', 'chirps':'precipitation'}
conv_factors_mm = {'era':1000, 'gpm':0.5, 'gldas':3600*3, 'daymet':1, 'persiann':1, 'trmm':3, 'chirps':1}
colors = {'era':'b', 'gpm':'y', 'gldas':'c', 'daymet':'r', 'persiann':'g', 'trmm':'m', 'chirps':'orange', 'livneh':'brown'}
start_dates = {'era':'1981-01-01T01:00:00', 'gpm':'2000-06-01T00:00:00', 'gldas':'2000-01-01T03:00:00', 
               'daymet':'1980-01-01T00:00:00', 'persiann':'1983-01-01T00:00:00', 'trmm':'1998-01-01T00:00:00', 
               'chirps':'1981-01-01T00:00:00', 'livneh':'1950-01-01T00:00:00'}
end_dates = {'era':'2023-01-28T23:00:00', 'gpm':'2023-03-30T03:30:00', 'gldas':'2023-03-14T21:00:00', 
             'daymet':'2021-12-31T00:00:00', 'persiann':'2022-09-30T00:00:00', 'trmm':'2019-12-31T21:00:00', 
             'chirps':'2023-02-28T00:00:00', 'livneh':'2013-12-31T00:00:00'}

ds_order = ['Pchirps', 'Pdaymet', 'Pera', 'Pgldas', 'Pgpm', 'Plivneh', 'Ppersiann']
ds_names = ['CHIRPS', 'Daymet', 'ERA5L', 'GLDAS', 'GPM', 'Livneh', 'PERSIANN']
ds_names_dict = {'Pchirps':'CHIRPS', 'Pdaymet':'Daymet', 'Pera':'ERA5L', 'Pgldas':'GLDAS', 'Pgpm':'GPM', 'livneh':'Livneh', 'Ppersiann':'PERSIANN'}
ds_colors = ['orange', 'r', 'b', 'c', 'y', 'm', 'g']

all_ds = ['Pcna', 'Pchirps', 'Pdaymet', 'Pera', 'Pgldas', 'Pgpm', 'Plivneh', 'Ppersiann']
# all_ds_dict = {'Pcna':'Pcna', 'Pchirps':'Pchirps', 'Pdaymet':'Pdaymet', 'Pera':'Pera', 'Pgldas':'Pgldas', 'Pgpm':'Pgpm', 'Plivneh':'Plivneh', 'Ppersiann':'Ppersiann'}
# all_ds_colors = {'Pcna':'k', 'Pchirps':'orange', 'Pdaymet':'r', 'Pera':'b', 'Pgldas':'c', 'Pgpm':'y', 'Plivneh':'m', 'Ppersiann':'g'}
all_ds_colors = ['k', 'orange', 'r', 'b', 'c', 'y', 'm', 'g']
# all_ds_colors_dict = {'Pcna':'k', 'Pchirps':'orange', 'Pdaymet':'r', 'Pera':'b', 'Pgldas':'c', 'Pgpm':'y', 'Plivneh':'m', 'Ppersiann':'g'}
all_ds_ls = ['--', '-', '-', '-', '-', '-', '-', '-']
# all_ds_ls_dict = {'Pcna':'--', 'Pchirps':'-', 'Pdaymet':'-', 'Pera':'-', 'Pgldas':'-', 'Pgpm':'-', 'Plivneh':'-', 'Ppersiann':'-'}

my_cycler = (cycler(color=all_ds_colors) + cycler(ls=all_ds_ls))

# ests40 = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/Catalogo_ests_40.csv', encoding='latin-1')
# ests40.set_index('Unnamed: 0', inplace=True)
# ests40.index.name = None
# ests40 = ests40.drop([26204, 26207, 26208])
# ests40

#List of stations to analyse
files = os.listdir('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_IDW/')
ests = np.sort(np.array([i.replace('P_', '').replace('_idw.csv', '') for i in files], dtype=int))

################################################################################

#Read time series for periods and compute errors by station and create files

errors = spotpy.objectivefunctions.calculate_all_functions(1, 1)
skills = skill_scores(1,1)
errors_skills = errors + skills
errors_skills_name = [errors_skills[i][0] for i in range(len(errors_skills))]
errors_skills_name

start_years = np.arange(1981,2012, 10)
start_dates_list = [datetime.datetime(1991+x*10,1,1) for x in range(3)]
label_dates_list = [datetime.datetime(1981+x*4,1,1) for x in range(11)]
label_years_list = [datetime.datetime(1981+x*4,1,1).year for x in range(11)]

custom_date_parser = lambda x: datetime.datetime.strptime(x, "%d/%m/%Y")
custom_date_parser2 = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")

# Get stations with more than 25 years of data from 1981 to 2021
df_periods = pd.DataFrame(index=ests, columns=['A', 'Am', 'C', 'Cm'])

for est in df_periods.index:
    # est = 1016
    print(est)
    dfA = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/ESTACIONES/PRECIP/P_'+str(est)+'.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
    dfA.set_index('Unnamed: 0', inplace=True)
    dfA.index.name = None
    dfAy = dfA.resample('Y').sum()
    dfA_masked = dfAy[((dfAy.index>'1981-01-01') & (dfAy.index<'2021-01-01'))]
    df_periods.loc[est, 'A'] = len(dfAy[dfAy>0].dropna())
    df_periods.loc[est, 'Am'] = len(dfA_masked[dfA_masked>0].dropna())
    
    dfC = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_IDW/P_'+str(est)+'_idw.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
    dfC.set_index('Unnamed: 0', inplace=True)
    dfC.index.name = None
    dfCy = dfC.resample('Y').sum()
    dfC_masked = dfCy[((dfCy.index>'1981-01-01') & (dfCy.index<'2021-01-01'))]
    df_periods.loc[est, 'C'] = len(dfCy[dfCy>0].dropna())
    df_periods.loc[est, 'Cm'] = len(dfC_masked[dfC_masked>0].dropna())

# df_periods['C'].plot(marker='.')
# df_periods.mean()

# df_periods.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/len_ests.csv')

df_periods = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/len_ests.csv', index_col='Unnamed: 0')
ests25 = df_periods[df_periods['Am']>25].index
# ests25 = df_periods[df_periods['Cm']>25].index

# ests_done = os.listdir('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/PRECIP/chirps/')
# ests_done = [est.replace('Pchirps_', '').replace('.csv', '') for est in ests_done]

# list_dir = 'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/Original/m/'

# for file in glob.glob(list_dir+"*_mmax.csv"):
#     print(file)
#     file = file.replace('\\', '/')
#     dest = file.replace('/m/', '/mmax/')
#     os.rename(file, dest)


#Generation of daily, monthly and yearly time series for alla dataset by station
datasets_all = os.listdir('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/PRECIP/')

error1 = []
error2 = []

# gc.collect()

# for est in ests[ests>=error2[-1]]: #error2[-1]]: #[ests40.index>32057]:
for est in ests25:
#     est = 31094
    
    print(est)
    df_nan_all = pd.DataFrame()
    df_nan_all_m = pd.DataFrame()
    df_nan_all_mmax = pd.DataFrame()
    df_nan_all_y = pd.DataFrame()
    df_nan_all_ymax = pd.DataFrame()
    
    try:
        df_nan = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/ESTACIONES/PRECIP/P_'+str(est)+'.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
        # df_nan = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_IDW/P_'+str(est)+'_idw.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
        df_nan.set_index('Unnamed: 0', inplace=True)
        df_nan.index.name = None
        df_nan.columns = ['Pcna']
        # plt.plot(df_nan['Pcna'].resample('M').sum())
        
        for ds in datasets_all:
            # ds = datasets_all[-4]
            try:
                df_ds = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/PRECIP/'+ds+'/P'+ds+'_'+str(est)+'.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser2)
                df_ds.set_index('Unnamed: 0', inplace=True)
                df_ds.index.name = None
                df_ds.columns = ['P'+ds]
                df_ds[df_ds<0.01] = 0
                
                df_nan = pd.concat([df_nan, df_ds['P'+ds]], axis=1)
            except Exception:
                df_ds = pd.DataFrame([], columns=['P'+ds])
                df_nan = pd.concat([df_nan, df_ds['P'+ds]], axis=1)
            
        df_nan.dropna(subset=['Pcna'], how='any', inplace=True)
        df_dt = pd.date_range(df_nan.index[0].strftime('%Y-%m-%d'), df_nan.index[-1].strftime('%Y-%m-%d'), freq='D')
        df_nan = df_nan.reindex(df_dt, fill_value=np.nan)
        
        # df_nan[df_nan==0] = np.nan
        # df_nan.dropna(subset=['Pcna'], inplace=True)
        # year = 1991
        # start_date = str(year)+'-01-01T00:00'
        # end_date = str(year+10)+'-01-01T00:00'
        # df_nan = df_nan[((df_nan.index>=start_date) & (df_nan.index<end_date))]
        # df_nan = df_nan.dropna()
        # df_nan.resample('Y').count()
        # len(df_nan['Pcna'].values)
        # df_nan[df_nan.index.year==2015].plot(marker='.')
        # np.count_nonzero(~np.isnan(df_nan))
        
        # df_nan['Pcna'].resample('M').sum().plot()
        # df_nan[df_nan_count['Pcna']>10]
        # df_nan_count[df_nan_count<25].plot()
        
        #Drop months with 11 or more days with missing data
        df_nan_count = df_nan['Pcna'].resample('M').count().to_frame()
        df_nan_m = df_nan.resample('M').apply(lambda x: np.nansum(x.values))
        df_nan_m.loc[df_nan_count['Pcna']<10] = np.nan
        
        missing_data = df_nan['Pcna'].isnull()
        
        consecutive_missing = []
        for k, g in groupby(enumerate(missing_data), lambda x: x[1]):
            if k:
                consecutive_missing.append(list(map(lambda x: x[0], list(g))))
        
        #Drop months with 5 or more consecutive days with missing data
        for l in range(len(consecutive_missing)):
            # l = 35
            m = consecutive_missing[l]
            # print(l, m)
            if len(m)>4:
                missing_dates = df_nan.iloc[m].index
                missing_months = df_nan.loc[missing_dates].resample('M').count().index
                df_nan_m.loc[missing_months] = np.nan
        
        # df_nan_m = df_nan.resample('M').sum()
        # df_nan_m = df_nan_m.reindex(columns=df_nan.columns, fill_value=np.nan)
        # df_nan_m[df_nan_m==0] = np.nan
        # df_nan_m[df_nan_m['Pcna'].isnull()] = np.nan
        df_nan_m.dropna(subset=['Pcna'], how='any', inplace=True)
        df_dt_m = pd.date_range(df_nan_m.index[0].strftime('%Y-%m-%d'), df_nan_m.index[-1].strftime('%Y-%m-%d'), freq='M')
        df_nan_m = df_nan_m.reindex(df_dt_m, fill_value=np.nan)
        
        #Mask for start and end dates for each dataset
        for ds1 in datasets_all:
            # ds1 = datasets_all[0]
            df_nan_m[f'P{ds1}'][((df_nan_m.index<start_dates[ds1]) | (df_nan_m.index>end_dates[ds1]))] = np.nan
                        
        # df_nan_mmax = df_nan.resample('M').apply(lambda x: np.max(x.values))# if np.nanmax(x.values)>0 else np.nan)
        df_nan_mmax = df_nan.resample('M').max()
        # df_nan_mmax = df_nan_mmax.reindex(columns=df_nan.columns, fill_value=np.nan)
        df_nan_mmax[df_nan_mmax==0] = np.nan
        # df_nan_mmax[df_nan_mmax['Pcna'].isnull()] = np.nan
        df_nan_mmax.dropna(subset=['Pcna'], how='any', inplace=True)
        df_dt_mmax = pd.date_range(df_nan_mmax.index[0].strftime('%Y-%m-%d'), df_nan_mmax.index[-1].strftime('%Y-%m-%d'), freq='M')
        df_nan_mmax = df_nan_mmax.reindex(df_dt_mmax, fill_value=np.nan)
        
        #Drop years with 1 or more months with missing data
        df_nan_m_count = df_nan_m['Pcna'].resample('Y').count().to_frame()
        df_nan_y = df_nan_m.resample('Y').apply(lambda x: np.nansum(x.values))
        df_nan_y.loc[df_nan_m_count['Pcna']<12] = np.nan
        df_nan_y[df_nan_y<10] = np.nan
        # df_nan_y.plot(marker='.')
        
        # df_nan_y = df_nan_m.resample('Y').apply(lambda x: np.sum(x.values) if np.sum(x.values)>10 else np.nan)
        # df_nan_y = df_nan.resample('Y').apply(lambda x: np.nansum(x.values) if np.nansum(x.values)>0 else np.nan)
        # df_nan_y = df_nan.resample('Y').sum()
        # df_nan_y = df_nan_y.reindex(columns=df_nan.columns, fill_value=np.nan)
        # df_nan_y[df_nan_y<0.01] = np.nan
        # df_nan_y[df_nan_y['Pcna'].isnull()] = np.nan
        df_nan_y.dropna(subset=['Pcna'], how='any', inplace=True)
        df_dt_y = pd.date_range(df_nan_y.index[0].strftime('%Y-%m-%d'), df_nan_y.index[-1].strftime('%Y-%m-%d'), freq='Y')
        df_nan_y = df_nan_y.reindex(df_dt_y, fill_value=np.nan)
        
        # df_nan_ymax = df_nan_mmax.resample('Y').apply(lambda x: np.nanmax(x.values) if np.nanmax(x.values)>10 else np.nan)
        df_nan_ymax = df_nan_mmax.resample('Y').max()
        # df_nan_ymax = df_nan_ymax.reindex(columns=df_nan.columns, fill_value=np.nan)
        df_nan_ymax[df_nan_ymax==0] = np.nan
        # df_nan_ymax[df_nan_ymax['Pcna'].isnull()] = np.nan
        df_nan_ymax.dropna(subset=['Pcna'], how='any', inplace=True)
        df_dt_ymax = pd.date_range(df_nan_ymax.index[0].strftime('%Y-%m-%d'), df_nan_ymax.index[-1].strftime('%Y-%m-%d'), freq='Y')
        df_nan_ymax = df_nan_ymax.reindex(df_dt_ymax, fill_value=np.nan)
        
        df_nan_all = pd.concat([df_nan_all, df_nan], axis=0)
        df_nan_all_m = pd.concat([df_nan_all_m, df_nan_m], axis=0)
        df_nan_all_mmax = pd.concat([df_nan_all_mmax, df_nan_mmax], axis=0)
        df_nan_all_y = pd.concat([df_nan_all_y, df_nan_y], axis=0)
        df_nan_all_ymax = pd.concat([df_nan_all_ymax, df_nan_ymax], axis=0)
        
        df_nan_all.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/d/TimeSeries_'+str(est)+'_d.csv', encoding='latin-1', date_format="%Y-%m-%d")
        df_nan_all_m.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/m/TimeSeries_'+str(est)+'_m.csv', encoding='latin-1', date_format="%Y-%m-%d")
        df_nan_all_mmax.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/mmax/TimeSeries_'+str(est)+'_mmax.csv', encoding='latin-1', date_format="%Y-%m-%d")
        df_nan_all_y.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/y/TimeSeries_'+str(est)+'_y.csv', encoding='latin-1', date_format="%Y-%m-%d")
        df_nan_all_ymax.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/ymax/TimeSeries_'+str(est)+'_ymax.csv', encoding='latin-1', date_format="%Y-%m-%d")
            
    except Exception:
        print(est, 'ERROR1')
        error1.append(est)
        pass


#########################################################################################

    #Plot complete time series
    # if str(est)[:2] == '19':
    # fig, ax = plt.subplots(2,2, figsize=(8,8), gridspec_kw={'wspace':0.22, 'hspace':0.1})
    # ax[0,0].set_prop_cycle(my_cycler)
    # ax[1,0].set_prop_cycle(my_cycler)
    # ax[0,1].set_prop_cycle(my_cycler)
    # ax[1,1].set_prop_cycle(my_cycler)
    
    # ax[0,0].plot(df_nan_all, marker='.', ms=4, label=df_nan_all.columns, alpha=0.8)
    # ax[1,0].plot(df_nan_all_m, marker='.', ms=4, label=df_nan_all.columns, alpha=0.8)
    # ax[0,1].plot(df_nan_all_y, marker='.', ms=4, label=df_nan_all.columns, alpha=0.8)
    # ax[1,1].plot(df_nan_all_ymax, marker='.', ms=4, label=df_nan_all.columns, alpha=0.8)
    # # ax[0,1].plot(df_nan_y.index.year, df_nan_y, marker='.')
    # # ax[1,1].plot(df_nan_ymax.index.year, df_nan_ymax, marker='.')
   
    # ax[0,0].set_xticks(ax[1,0].get_xticks()[::2])
    # ax[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # ax[1,0].set_xticks(ax[1,0].get_xticks()[::2])
    # ax[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # ax[0,1].set_xticks(ax[0,1].get_xticks()[::2])
    # ax[0,1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # ax[1,1].set_xticks(ax[1,1].get_xticks()[::2])
    # ax[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # ax[0,0].set_ylabel('Accumulated Precipitation [mm/day]')
    # ax[1,0].set_ylabel('Accumulated Precipitation [mm/month]')
    # ax[0,1].set_ylabel('Accumulated Precipitation [mm/year]')
    # ax[1,1].set_ylabel('Maximum Anual Precipitation [mm/day]')
    
    # ax[1,1].legend(loc='lower center', ncol=len(df_nan_all.columns)//2, bbox_to_anchor=(-0.15, -0.25))
    
    # fig.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/TimeSeries/TimeSeries_Comparison_'+str(est)+'_'+str(year)+'.jpg', format='jpg', dpi=300, bbox_inches='tight')
    # plt.close()
    try:
        start_date = '1981-01-01T00:00'
        end_date = '2021-01-01T00:00'
        mask = ((df_nan_all_y.index>=start_date) & (df_nan_all_y.index<=end_date))
        
        # plt.plot(df_nan_all_y[mask].dropna(subset=['Pcna']))
        
        if len(df_nan_all_y[mask].dropna(subset=['Pcna'])) >= 25:
            # fig, ax = plt.subplots(4,1, figsize=(14,12), gridspec_kw={'wspace':0.22, 'hspace':0.1})
            
            # for _ax in range(len(ax)):
            #     ax[_ax].set_prop_cycle(my_cycler)
            #     ax[_ax].set_xlim(datetime.datetime(1981, 1, 1, 0, 0), datetime.datetime(2021, 1, 1, 0, 0))
                
            #     for i in range(len(start_dates_list)):
            #         ax[_ax].axvline(start_dates_list[i], lw=1, ls='dotted', color='k')
            
            # ax[0].plot(df_nan_all_m, marker='.', ms=2, lw=1, label=df_nan_all_m.columns, alpha=0.75)
            # ax[1].plot(df_nan_all_mmax, marker='.', ms=2, lw=1, label=df_nan_all_mmax.columns, alpha=0.75)
            # ax[2].plot(df_nan_all_y, marker='.', ms=2, lw=1, label=df_nan_all_y.columns, alpha=0.75)
            # ax[3].plot(df_nan_all_ymax, marker='.', ms=2, lw=1, label=df_nan_all_ymax.columns, alpha=0.75)
            # # ax[0,1].plot(df_nan_y.index.year, df_nan_y, marker='.')
            # # ax[1,1].plot(df_nan_ymax.index.year, df_nan_ymax, marker='.')
        
            # for _ax in range(len(ax)-1):
            #     ax[_ax].set_xticklabels('')
            # ax[3].set_xticks(label_dates_list)
            # ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
            # ax[0].set_ylabel('Precip [mm/month]')
            # ax[1].set_ylabel('Max Precip [mm/day]')
            # ax[2].set_ylabel('Precip [mm/year]')
            # ax[3].set_ylabel('Max Precip [mm/day]')
            
            # ax[3].legend(loc='lower center', ncol=len(df_nan_all.columns), bbox_to_anchor=(0.5, -0.35))
            # fig.align_ylabels(ax[:])
            
            # fig.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/TimeSeries/TimeSeries_Comparison_'+str(est)+'.jpg', format='jpg', dpi=600, bbox_inches='tight')
            # plt.close()
            
            #Save complete time series
            df_nan_all.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/d/TimeSeries_'+str(est)+'_d.csv', encoding='latin-1')
            df_nan_all_m.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/m/TimeSeries_'+str(est)+'_m.csv', encoding='latin-1')
            df_nan_all_mmax.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/mmax/TimeSeries_'+str(est)+'_mmax.csv', encoding='latin-1')
            df_nan_all_y.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/y/TimeSeries_'+str(est)+'_y.csv', encoding='latin-1')
            df_nan_all_ymax.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/ymax/TimeSeries_'+str(est)+'_ymax.csv', encoding='latin-1')
            
            gc.collect()
    except Exception:
        print(est, 'ERROR2')
        plt.close()
        error2.append(est)
        if error1[-1] != error2[-1]:
            winsound.Beep(freq, duration)
            break
        pass

    # plt.plot(df_nan)
    # plt.plot(df_nan_m)
    # plt.plot(df_nan_y)

e1 = np.array(error1)

#Compute errors for all time series
for est in ests:
    # est = 1004
    print(est)
    try:
        df_nan_all = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/d/TimeSeries_'+str(est)+'_d.csv', parse_dates=['Unnamed: 0'])
        df_nan_all.set_index('Unnamed: 0', inplace=True)
        df_nan_all.index.name = None
        
        df_nan_all_m = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/m/TimeSeries_'+str(est)+'_m.csv', parse_dates=['Unnamed: 0'])
        df_nan_all_m.set_index('Unnamed: 0', inplace=True)
        df_nan_all_m.index.name = None
        
        df_nan_all_mmax = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/mmax/TimeSeries_'+str(est)+'_mmax.csv', parse_dates=['Unnamed: 0'])
        df_nan_all_mmax.set_index('Unnamed: 0', inplace=True)
        df_nan_all_mmax.index.name = None
        
        df_nan_all_y = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/y/TimeSeries_'+str(est)+'_y.csv', parse_dates=['Unnamed: 0'])
        df_nan_all_y.set_index('Unnamed: 0', inplace=True)
        df_nan_all_y.index.name = None
        
        df_nan_all_ymax = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/ymax/TimeSeries_'+str(est)+'_ymax.csv', parse_dates=['Unnamed: 0'])
        df_nan_all_ymax.set_index('Unnamed: 0', inplace=True)
        df_nan_all_ymax.index.name = None
    
        es_df_all = pd.DataFrame()
        es_df_all_m = pd.DataFrame()
        es_df_all_mmax = pd.DataFrame()
        es_df_all_y = pd.DataFrame()
        es_df_all_ymax = pd.DataFrame()
        
        for year in start_years:
    #         print(year)
            # year = 2011
            es_df = pd.DataFrame(errors_skills_name, index=errors_skills_name, columns=['errors'])
            es_df_m = pd.DataFrame(errors_skills_name, index=errors_skills_name, columns=['errors'])
            es_df_mmax = pd.DataFrame(errors_skills_name, index=errors_skills_name, columns=['errors'])
            es_df_y = pd.DataFrame(errors_skills_name, index=errors_skills_name, columns=['errors'])
            es_df_ymax = pd.DataFrame(errors_skills_name, index=errors_skills_name, columns=['errors'])
            
            es_df = es_df.drop(['errors'], axis=1)
            es_df_m = es_df_m.drop(['errors'], axis=1)
            es_df_y = es_df_y.drop(['errors'], axis=1)
            es_df_ymax = es_df_ymax.drop(['errors'], axis=1)
            
            start_date = str(year)+'-01-01T00:00'
            end_date = str(year+10)+'-01-01T00:00'
            
            df = df_nan_all[((df_nan_all.index>=start_date) & (df_nan_all.index<end_date))]
            df_m = df_nan_all_m[((df_nan_all_m.index>=start_date) & (df_nan_all_m.index<end_date))]
            df_y = df_nan_all_y[((df_nan_all_y.index>=start_date) & (df_nan_all_y.index<end_date))]
            df_ymax = df_nan_all_ymax[((df_nan_all_ymax.index>=start_date) & (df_nan_all_ymax.index<end_date))]
            
            mask = ((df_ymax.index>=start_date) & (df_ymax.index<=end_date))
            
            # ds_df = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/PRECIP/'+ds+'/P'+ds+'_'+str(est)+'.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser2)
            # ds_df.set_index('Unnamed: 0', inplace=True)
            # ds_df.index.name = None
            
            # Computation of errors
            for ds in datasets_all:
                df_nan = df.dropna(subset=['P'+ds])
                df_nan_m = df_m.dropna(subset=['P'+ds])
                df_nan_y = df_y.dropna(subset=['P'+ds])
                df_nan_ymax = df_ymax.dropna(subset=['P'+ds])
                
                try:
                    if len(df_nan_ymax[mask].dropna(subset=['P'+ds])) >= 7:
                        errors = spotpy.objectivefunctions.calculate_all_functions(df_nan['Pcna'], df_nan['P'+ds])
                        skills = skill_scores(df_nan['Pcna'], df_nan['P'+ds])
                        errors_skills = errors + skills
                        errors_skills_value = [errors_skills[i][1] for i in range(len(errors_skills))]
                        esk_df = pd.DataFrame(errors_skills_value, index=errors_skills_name, columns=['P'+ds])
                        es_df = pd.concat([es_df, esk_df], axis=1)
            
                        errors_m = spotpy.objectivefunctions.calculate_all_functions(df_nan_m['Pcna'], df_nan_m['P'+ds])
                        skills_m = skill_scores(df_nan_m['Pcna'], df_nan_m['P'+ds])
                        errors_skills_m = errors_m + skills_m
                        errors_skills_value_m = [errors_skills_m[i][1] for i in range(len(errors_skills_m))]
                        esk_df_m = pd.DataFrame(errors_skills_value_m, index=errors_skills_name, columns=['P'+ds])
                        es_df_m = pd.concat([es_df_m, esk_df_m], axis=1)
                        
                        errors_y = spotpy.objectivefunctions.calculate_all_functions(df_nan_y['Pcna'], df_nan_y['P'+ds])
                        skills_y = skill_scores(df_nan_y['Pcna'], df_nan_y['P'+ds])
                        errors_skills_y = errors_y + skills_y
                        errors_skills_value_y = [errors_skills_y[i][1] for i in range(len(errors_skills_y))]
                        esk_df_y = pd.DataFrame(errors_skills_value_y, index=errors_skills_name, columns=['P'+ds])
                        es_df_y = pd.concat([es_df_y, esk_df_y], axis=1)
                        
                        errors_ymax = spotpy.objectivefunctions.calculate_all_functions(df_nan_ymax['Pcna'], df_nan_ymax['P'+ds])
                        skills_ymax = skill_scores(df_nan_ymax['Pcna'], df_nan_ymax['P'+ds])
                        errors_skills_ymax = errors_ymax + skills_ymax
                        errors_skills_value_ymax = [errors_skills_ymax[i][1] for i in range(len(errors_skills_ymax))]
                        esk_df_ymax = pd.DataFrame(errors_skills_value_ymax, index=errors_skills_name, columns=['P'+ds])
                        es_df_ymax = pd.concat([es_df_ymax, esk_df_ymax], axis=1)
                    
                    else:
                        errors_skills_value = [np.nan for i in range(len(errors_skills_name))]
                        esk_df = pd.DataFrame(errors_skills_value, index=errors_skills_name, columns=['P'+ds])
                        es_df = pd.concat([es_df, esk_df], axis=1)
            
                        errors_skills_value_m = [np.nan for i in range(len(errors_skills_name))]
                        esk_df_m = pd.DataFrame(errors_skills_value_m, index=errors_skills_name, columns=['P'+ds])
                        es_df_m = pd.concat([es_df_m, esk_df_m], axis=1)
                        
                        errors_skills_value_y = [np.nan for i in range(len(errors_skills_name))]
                        esk_df_y = pd.DataFrame(errors_skills_value_y, index=errors_skills_name, columns=['P'+ds])
                        es_df_y = pd.concat([es_df_y, esk_df_y], axis=1)
                        
                        errors_skills_value_ymax = [np.nan for i in range(len(errors_skills_name))]
                        esk_df_ymax = pd.DataFrame(errors_skills_value_ymax, index=errors_skills_name, columns=['P'+ds])
                        es_df_ymax = pd.concat([es_df_ymax, esk_df_ymax], axis=1)
                        
                except Exception:
        #                 print(ds, 'NA')
                    errors_skills_value = [np.nan for i in range(len(errors_skills_name))]
                    esk_df = pd.DataFrame(errors_skills_value, index=errors_skills_name, columns=['P'+ds])
                    es_df = pd.concat([es_df, esk_df], axis=1)
        
                    errors_skills_value_m = [np.nan for i in range(len(errors_skills_name))]
                    esk_df_m = pd.DataFrame(errors_skills_value_m, index=errors_skills_name, columns=['P'+ds])
                    es_df_m = pd.concat([es_df_m, esk_df_m], axis=1)
                    
                    errors_skills_value_y = [np.nan for i in range(len(errors_skills_name))]
                    esk_df_y = pd.DataFrame(errors_skills_value_y, index=errors_skills_name, columns=['P'+ds])
                    es_df_y = pd.concat([es_df_y, esk_df_y], axis=1)
                    
                    errors_skills_value_ymax = [np.nan for i in range(len(errors_skills_name))]
                    esk_df_ymax = pd.DataFrame(errors_skills_value_ymax, index=errors_skills_name, columns=['P'+ds])
                    es_df_ymax = pd.concat([es_df_ymax, esk_df_ymax], axis=1)
                    pass
                
            es_df_all = pd.concat([es_df_all, es_df], axis=0)
            es_df_all_m = pd.concat([es_df_all_m, es_df_m], axis=0)
            es_df_all_y = pd.concat([es_df_all_y, es_df_y], axis=0)
            es_df_all_ymax = pd.concat([es_df_all_ymax, es_df_ymax], axis=0)
            
    #         plt.plot(df_cna, lw=1, c='k', alpha=0.75, label='Pcna')
    #         plt.xlabel(year)
    #         plt.ylabel('Precipitation [mm]')
    #         plt.title('Comparison of Precipitation Datasets')
    #         plt.legend()
    #         plt.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/VARIOS/GEE_Precip_Datasets_'+str(est)+'_'+str(year)+'.jpg', format='jpg', dpi=300, bbox_inches='tight')
    #         plt.close()
    #         es_df.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/VARIOS/GEE_Precip_Datasets_errors_'+str(est)+'_'+str(year)+'.csv')
        
        #Save error results
        es_df_all.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERRORS/Periods/d/GEE_Precip_Datasets_errors_'+str(est)+'_d.csv', encoding='latin-1')
        es_df_all_m.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERRORS/Periods/m/GEE_Precip_Datasets_errors_'+str(est)+'_m.csv', encoding='latin-1')
        es_df_all_y.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERRORS/Periods/y/GEE_Precip_Datasets_errors_'+str(est)+'_y.csv', encoding='latin-1')
        es_df_all_ymax.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERRORS/Periods/ymax/GEE_Precip_Datasets_errors_'+str(est)+'_ymax.csv', encoding='latin-1')
    
    except Exception:
        print(est, 'NO DATA')
        pass

################################################################################

# Read errors files and arrange in arrays
# Periods of datasets
periods_dict = {0:'1981-1991', 1:'1991-2001', 2:'2001-2011', 3:'2011-2021'}
periods_i = ['1981-1991', '1991-2001', '2001-2011', '2011-2021']
positions_i = [0, 1, 2, 3]
time_steps = ['d', 'm', 'mmax', 'y', 'ymax']
time_steps_names = ['Daily', 'Monthly', 'Max Monthly', 'Yearly', 'Max Yearly']
time_steps_dict = {'d':'Daily', 'm':'Monthly', 'mmax':'Max Monthly', 'y':'Yearly', 'ymax':'Max Yearly'}

#Load shapes

# '6372' # Mexico_ITRF2008_LCC
# '4326' # GCS_WGS_1984

# geometry = [Point(xy) for xy in zip(nses_full['LON'],nses_full['LAT'])]
# geo_df = gpd.GeoDataFrame(geometry=geometry)
# geo_df.crs = {'init':"epsg:4326"}

shpMx = "D:/DANI/2023/TEMA_TORMENTAS/DATOS/GEE/SHP/Border_Mx.shp"
sMx = gpd.read_file(shpMx, bbox=None, mask=None, rows=None)
sMx.crs = {'init':"epsg:4326"}

shpStMx = "D:/DANI/2023/TEMA_TORMENTAS/DATOS/GEE/SHP/Estados_Mx.shp"
shpStMx = gpd.read_file(shpStMx, bbox=None, mask=None, rows=None)
shpStMx.crs = {'init':"epsg:4326"}

shpClim = "D:/DANI/2023/TEMA_TORMENTAS/DATOS/GEE/SHP/Regiones_Clim.shp"
sClim = gpd.read_file(shpClim, bbox=None, mask=None, rows=None)
sClim.crs = {'init':"epsg:4326"}

shpCEM = "D:/DANI/2023/TEMA_TORMENTAS/DATOS/GEE/SHP/cem_class_dis.shp"
shpCEM = gpd.read_file(shpCEM, bbox=None, mask=None, rows=None)
shpCEM.crs = {'init':"epsg:6372"}

sMx.boundary.plot()
shpStMx.boundary.plot()
sClim.boundary.plot()

limits1 = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False) # NSE, KGE, R
limits2 = mpl.colors.Normalize(vmin=-25, vmax=25, clip=False) # PBIAS
limits3 = mpl.colors.Normalize(vmin=0, vmax=1, clip=False) # RSR

# newcmap = LinearSegmentedColormap.from_list('rbg',["r", "m", "b", "c", "g"], N=256)
newcmap = LinearSegmentedColormap.from_list('rby',["r", "m", "b", "c", "y"], N=256)

##########
errors_list = ['kge', 'nashsutcliffe', 'pbias', 'rsr']
# errors_list = ['kge', 'nashsutcliffe', 'correlationcoefficient', 'pbias', 'rsr', 'rmse', 'FB', 'POD', 'FAR', 'TS', 'ETS']
vmin_dict = {'kge':-0.41, 'nashsutcliffe':0, 'correlationcoefficient':-1, 'pbias':-25, 'rsr':0, 'rmse':0, 'FB':0, 'POD':0, 'FAR':0, 'TS':0, 'ETS':0}
vmax_dict = {'kge':1, 'nashsutcliffe':1, 'correlationcoefficient':1, 'pbias':25, 'rsr':1, 'rmse':25, 'FB':1, 'POD':1, 'FAR':1, 'TS':1, 'ETS':1}
# vmin_list = [-0.41, 0, -1, -25, 0, 0, 0, 0, 0, 0, 0]
# vmax_list = [1, 1, 1, 25, 1, 25, 1, 1, 1, 1, 1]

for err_name in errors_list:
    print(err_name)
    limits = mpl.colors.Normalize(vmin=vmin_dict[err_name], vmax=vmax_dict[err_name], clip=False)
    err_df = pd.DataFrame()
##########

est = 1004

df_nan_all = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/d/TimeSeries_'+str(est)+'_d.csv', parse_dates=['Unnamed: 0'])
df_nan_all.set_index('Unnamed: 0', inplace=True)
df_nan_all.index.name = None

for ds in ds_order:
    # ds = 'Pdaymet'
        
    df_x = pd.concat([df_nan_all['Pcna'], df_nan_all[ds]], axis=1)
    df_x.columns = ['obs', 'sim']
    df_x.dropna(inplace=True)
    
    obs_arr = df_x[df_x['obs']>0]['obs'].values
    sim_arr = df_x[df_x['obs']>0]['sim'].values
    
    # calculate diagnostic efficiency
    eff_de = de.calc_de(obs_arr, sim_arr)
    
    # diagnostic polar plot
    # de.diag_polar_plot(obs_arr, sim_arr)
    
    
for ds in ds_order:
    # ds = 'Pdaymet'
    
    plt.plot(np.log(np.sort(obs_arr)))
    plt.plot(np.log(np.sort(sim_arr)))
    plt.plot(np.sort(obs_arr))
    plt.plot(np.sort(sim_arr))
    
####################################################################################

for pos_i in positions_i:
    for time_st in time_steps:
        # pos_i = 0
        # time_st = 'd'
        kges = pd.DataFrame()
        nses = pd.DataFrame()
        pbiass = pd.DataFrame()
        ccs = pd.DataFrame()
        for est in ests.index:
            # print(est)
            try:
                # est = '19004'
                # df = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERRORS/Complete/errors_{}.csv'.format(est), index_col=['Unnamed: 0'])
                df = pd.read_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERRORS/Periods/{time_st}/GEE_Precip_Datasets_errors_{est}_{time_st}.csv', index_col=['Unnamed: 0'])
                kge = df[df.index=='kge'].iloc[pos_i].to_frame().T
                nse = df[df.index=='nashsutcliffe'].iloc[pos_i].to_frame().T
                pbias = df[df.index=='pbias'].iloc[pos_i].to_frame().T
                cc = df[df.index=='correlationcoefficient'].iloc[pos_i].to_frame().T
                
                kge.index = [est]
                kges = pd.concat([kges, kge])
                nse.index = [est]
                nses = pd.concat([nses, nse])
                pbias.index = [est]
                pbiass = pd.concat([pbiass, pbias])
                cc.index = [est]
                ccs = pd.concat([ccs, cc])
            except Exception:
                print(est)
                pass
        
        # kges.drop(['Ptrmm'], axis=1, inplace=True)
        # nses.drop(['Ptrmm'], axis=1, inplace=True)
        # pbiass.drop(['Ptrmm'], axis=1, inplace=True)
        # ccs.drop(['Ptrmm'], axis=1, inplace=True)
        
        kges_full = pd.concat([ests['LON'], ests['LAT'], kges], axis=1)
        kges_full.dropna(how='all', subset=kges.columns, inplace=True)
        nses_full = pd.concat([ests['LON'], ests['LAT'], nses], axis=1)
        nses_full.dropna(how='all', subset=nses.columns, inplace=True)
        pbiass_full = pd.concat([ests['LON'], ests['LAT'], pbiass], axis=1)
        pbiass_full.dropna(how='all', subset=pbiass.columns, inplace=True)
        ccs_full = pd.concat([ests['LON'], ests['LAT'], ccs], axis=1)
        ccs_full.dropna(how='all', subset=ccs.columns, inplace=True)
        
        n = nses_full.copy()
        k = kges_full.copy()
        p = pbiass_full.copy()
        c = ccs_full.copy()
        
        ################################################################################
        
        # Error comparison for all stations in Mexico and plot
        
        fig, ax = pplt.subplots(ncols=4, nrows=7, share=False, figwidth=8, figheight=10, wspace=0.5, hspace=0.5)
        
        i = -1
        for ds in ds_order:
            i += 1
            
            # ncond = n[ds]>=0.5
            ncond = n[ds]<=1
            sClim.boundary.plot(ax=ax[i,0], alpha=1, lw=0.1, color='k', zorder=2)
            nplot = ax[i,0].scatter(n['LON'][ncond], n['LAT'][ncond], c=n[ds][ncond], cmap='jet_r', norm=limits1, s=0.1, alpha=1, levels=pplt.arange(-1, 1, 0.1))
            
            # kcond = k[ds]>=0.3
            kcond = k[ds]<=1
            sClim.boundary.plot(ax=ax[i,1], alpha=1, lw=0.1, color='k', zorder=2)
            kplot = ax[i,1].scatter(k['LON'][kcond], k['LAT'][kcond], c=k[ds][kcond], cmap='jet_r', norm=limits1, s=0.1, alpha=1, levels=pplt.arange(-1, 1, 0.1))
            
            # ccond = c[ds]>=0.25
            ccond = c[ds]<=1
            sClim.boundary.plot(ax=ax[i,2], alpha=1, lw=0.1, color='k', zorder=2)
            cplot = ax[i,2].scatter(c['LON'][ccond], c['LAT'][ccond], c=c[ds][ccond], cmap='jet_r', norm=limits1, s=0.1, alpha=1, levels=pplt.arange(-1, 1, 0.1))
            
            # pcond = abs(p[ds])<=25
            pcond = abs(p[ds])>=0
            sClim.boundary.plot(ax=ax[i,3], alpha=1, lw=0.1, color='k', zorder=2)
            pplot = ax[i,3].scatter(p['LON'][pcond], p['LAT'][pcond], c=p[ds][pcond], cmap=newcmap, norm=limits2, s=0.1, alpha=1, levels=pplt.arange(-25, 25, 2.5))
        
        ax.format(grid=False,
                toplabels=('NSE', 'KGE', 'R', 'PBIAS'),
                leftlabels=ds_names,
                xtickloc='none',
                xticklabelloc='none',
                ytickloc='none',
                yticklabelloc='none',
                xlabel='',
                ylabel='')
        
        # ax[:, 1].format(xtickloc='none', xticklabelloc='none', ytickloc='none', yticklabelloc='none', xlabel='xaxis', ylabel='yaxis')
        # ax[:, 2].format(xtickloc='none', xticklabelloc='none', ytickloc='none', yticklabelloc='none', xlabel='xaxis', ylabel='yaxis')
        ax[0, :].format(xtickloc='top', xticklabelloc='top')
        ax[-1, :].format(xtickloc='bottom', xticklabelloc='bottom')
        ax[:, 0].format(ytickloc='left', yticklabelloc='left')
        ax[:, -1].format(ytickloc='right', yticklabelloc='right')
        
        fig.colorbar(nplot, ticks=0.5, loc='b', col=1, shrink=0.8)
        fig.colorbar(kplot, ticks=0.5, loc='b', col=2, shrink=0.8)
        fig.colorbar(cplot, ticks=0.5, loc='b', col=3, shrink=0.8)
        fig.colorbar(pplot, ticks=10, loc='b', col=4, shrink=0.8)
        
        fig.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/Error_Comparsion_{}_{}.jpg'.format(periods_dict[pos_i], time_st), format='jpg', dpi=1000, bbox_inches='tight')
        plt.close()

################################################################################

#Plot all results for 1 dataset
errors_list = ['kge', 'nashsutcliffe', 'correlationcoefficient', 'pbias'] #, 'rmse', 'rsr', 'FB', 'POD', 'FAR', 'TS', 'ETS']
errors_list = ['pbias'] #, 'rmse', 'rsr', 'FB', 'POD', 'FAR', 'TS', 'ETS']
for err in errors_list:
    for ds in ds_order:
        print(err, ds)
        # ds = 'Pera'
        # err = 'kge'
        # err = 'nashsutcliffe'
        # err = 'correlationcoefficient'
        # err = 'pbias'
        errs1 = pd.DataFrame()
        errs2 = pd.DataFrame()
        errs3 = pd.DataFrame()
        errs4 = pd.DataFrame()
        
        for est in ests40.index:
            # print(est)
            try:
                # est = '19004'
                df1 = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERRORS/Periods/d/GEE_Precip_Datasets_errors_{}_d.csv'.format(est), index_col=['Unnamed: 0'])
                err1 = df1[df1.index==err][ds].to_frame().T
                df2 = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERRORS/Periods/m/GEE_Precip_Datasets_errors_{}_m.csv'.format(est), index_col=['Unnamed: 0'])
                err2 = df2[df2.index==err][ds].to_frame().T
                df3 = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERRORS/Periods/y/GEE_Precip_Datasets_errors_{}_y.csv'.format(est), index_col=['Unnamed: 0'])
                err3 = df3[df3.index==err][ds].to_frame().T
                df4 = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERRORS/Periods/ymax/GEE_Precip_Datasets_errors_{}_ymax.csv'.format(est), index_col=['Unnamed: 0'])
                err4 = df4[df4.index==err][ds].to_frame().T
                
                err1.index = [est]
                err1.columns = periods_i
                errs1 = pd.concat([errs1, err1])
                err2.index = [est]
                err2.columns = periods_i
                errs2 = pd.concat([errs2, err2])
                err3.index = [est]
                err3.columns = periods_i
                errs3 = pd.concat([errs3, err3])
                err4.index = [est]
                err4.columns = periods_i
                errs4 = pd.concat([errs4, err4])
                
            except Exception:
                # print(est)
                pass
        
        # kges.drop(['Ptrmm'], axis=1, inplace=True)
        # nses.drop(['Ptrmm'], axis=1, inplace=True)
        # pbiass.drop(['Ptrmm'], axis=1, inplace=True)
        # ccs.drop(['Ptrmm'], axis=1, inplace=True)
        
        errs1_full = pd.concat([ests40['LON'], ests40['LAT'], errs1], axis=1)
        errs1_full.dropna(how='all', subset=errs1.columns, inplace=True)
        errs2_full = pd.concat([ests40['LON'], ests40['LAT'], errs2], axis=1)
        errs2_full.dropna(how='all', subset=errs2.columns, inplace=True)
        errs3_full = pd.concat([ests40['LON'], ests40['LAT'], errs3], axis=1)
        errs3_full.dropna(how='all', subset=errs3.columns, inplace=True)
        errs4_full = pd.concat([ests40['LON'], ests40['LAT'], errs4], axis=1)
        errs4_full.dropna(how='all', subset=errs4.columns, inplace=True)
        
        d = errs1_full.copy()
        m = errs2_full.copy()
        y = errs3_full.copy()
        ym = errs4_full.copy()
        
        d.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/d/{}/Error_Comparsion_{}_{}_d.csv'.format(err, ds, err), encoding='latin-1')
        m.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/m/{}/Error_Comparsion_{}_{}_m.csv'.format(err, ds, err), encoding='latin-1')
        y.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/y/{}/Error_Comparsion_{}_{}_y.csv'.format(err, ds, err), encoding='latin-1')
        ym.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/ym/{}/Error_Comparsion_{}_{}_ym.csv'.format(err, ds, err), encoding='latin-1')
        ################################################################################
        
        # Error comparison for all stations in Mexico and plot
        
        fig, ax = pplt.subplots(ncols=4, nrows=4, share=False, figwidth=8, figheight=5, wspace=0.5, hspace=0.5)
        
        for pos_i in positions_i:
            # pos_i = 0
            # dcond = d[periods_dict[pos_i]]
            # fig, ax = pplt.subplots(ncols=4, nrows=4, share=False, figwidth=8, figheight=5, wspace=0.5, hspace=0.5)
            if err!='pbias':
                dcond = d[periods_dict[pos_i]]<=1
            else:
                dcond = abs(d[periods_dict[pos_i]])>=0
            sClim.boundary.plot(ax=ax[0, pos_i], alpha=1, lw=0.1, color='k', zorder=2)
            if err!='pbias':
                dplot = ax[0, pos_i].scatter(d['LON'][dcond], d['LAT'][dcond], c=d[periods_dict[pos_i]][dcond], cmap='jet_r', norm=limits1, s=0.1, alpha=1, levels=pplt.arange(-1, 1, 0.1))
            else:
                dplot = ax[0, pos_i].scatter(d['LON'][dcond], d['LAT'][dcond], c=d[periods_dict[pos_i]][dcond], cmap=newcmap, norm=limits2, s=0.1, alpha=1, levels=pplt.arange(-25, 25, 2.5))
            ax[0, pos_i].text(-118, 15, 'Mean = ' + str(round(d[periods_dict[pos_i]][dcond].mean(), 2)))
            
            # mcond = m[periods_dict[pos_i]]
            if err!='pbias':
                mcond = m[periods_dict[pos_i]]<=1
            else:
                mcond = abs(m[periods_dict[pos_i]])>=0
            sClim.boundary.plot(ax=ax[1, pos_i], alpha=1, lw=0.1, color='k', zorder=2)
            if err!='pbias':
                mplot = ax[1, pos_i].scatter(m['LON'][mcond], m['LAT'][mcond], c=m[periods_dict[pos_i]][mcond], cmap='jet_r', norm=limits1, s=0.1, alpha=1, levels=pplt.arange(-1, 1, 0.1))
            else:
                mplot = ax[1, pos_i].scatter(m['LON'][mcond], m['LAT'][mcond], c=m[periods_dict[pos_i]][mcond], cmap=newcmap, norm=limits2, s=0.1, alpha=1, levels=pplt.arange(-25, 25, 2.5))
            ax[1, pos_i].text(-118, 15, 'Mean = ' + str(round(m[periods_dict[pos_i]][mcond].mean(), 2)))
            
            # ycond = y[periods_dict[pos_i]]
            if err!='pbias':
                ycond = y[periods_dict[pos_i]]<=1
            else:
                ycond = abs(y[periods_dict[pos_i]])>=0
            sClim.boundary.plot(ax=ax[2, pos_i], alpha=1, lw=0.1, color='k', zorder=2)
            if err!='pbias':
                yplot = ax[2, pos_i].scatter(y['LON'][ycond], y['LAT'][ycond], c=y[periods_dict[pos_i]][ycond], cmap='jet_r', norm=limits1, s=0.1, alpha=1, levels=pplt.arange(-1, 1, 0.1))
            else:
                yplot = ax[2, pos_i].scatter(y['LON'][ycond], y['LAT'][ycond], c=y[periods_dict[pos_i]][ycond], cmap=newcmap, norm=limits2, s=0.1, alpha=1, levels=pplt.arange(-25, 25, 2.5))
            ax[2, pos_i].text(-118, 15, 'Mean = ' + str(round(y[periods_dict[pos_i]][ycond].mean(), 2)))
            
            # dcond = d[periods_dict[pos_i]]
            if err!='pbias':
                ymcond = ym[periods_dict[pos_i]]<=1
            else:
                ymcond = abs(ym[periods_dict[pos_i]])>=0
            sClim.boundary.plot(ax=ax[3, pos_i], alpha=1, lw=0.1, color='k', zorder=2)
            if err!='pbias':
                ymplot = ax[3, pos_i].scatter(ym['LON'][ymcond], ym['LAT'][ymcond], c=ym[periods_dict[pos_i]][ymcond], cmap='jet_r', norm=limits1, s=0.1, alpha=1, levels=pplt.arange(-1, 1, 0.1))
            else:
                ymplot = ax[3, pos_i].scatter(ym['LON'][ymcond], ym['LAT'][ymcond], c=ym[periods_dict[pos_i]][ymcond], cmap=newcmap, norm=limits2, s=0.1, alpha=1, levels=pplt.arange(-25, 25, 2.5))
            ax[3, pos_i].text(-118, 15, 'Mean = ' + str(round(ym[periods_dict[pos_i]][ymcond].mean(), 2)))
        
        ax.format(grid=False,
                toplabels=periods_i,
                leftlabels=time_steps_names,
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
        
        if err!='pbias':
            fig.colorbar(dplot, ticks=0.5, loc='r') #, shrink=0.8)
        else:
            fig.colorbar(dplot, ticks=10, loc='r') #, shrink=0.8)
        
        fig.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/ERROR_COMPARISON/{}/Error_Comparsion_{}_{}.jpg'.format(err, ds, err), format='jpg', dpi=1000, bbox_inches='tight')
        plt.close()
        


################################################################################
#Create colormap



top = cm.get_cmap('jet_r', 256)
bottom = cm.get_cmap('jet', 256)
newcolors = np.vstack((top(np.linspace(0, 1, 256)),
                       bottom(np.linspace(0, 1, 256))))
newcmp = ListedColormap(newcolors, name='jet_r_jet')

newcmap = LinearSegmentedColormap.from_list('rbg',["r", "m", "b", "c", "g"], N=256)
# newcmap = LinearSegmentedColormap.from_list('rbg',["b", "c", "y", "r"], N=256)
# newcmap = LinearSegmentedColormap.from_list('rbg',["r", "b", "g"], N=256)


def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()

plot_examples([newcmap, newcmp])
################################################################################

# norm=limits2, 


# limits = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False)

# plt.subplot(441)
# plt.scatter(x['LON'], x['LAT'], c=x['livneh'], cmap='jet_r', s=1, norm=limits, alpha=1)
# plt.colorbar(norm=limits)
# plt.subplot(446)
# plt.scatter(x['LON'], x['LAT'], c=x['livneh'], cmap='jet_r', s=1, norm=limits, alpha=1)
# plt.colorbar(norm=limits)



# ax = geo_df.plot(ax=ax, markersize=0.5, c=ests_full['Ppersiann'], cmap='coolwarm', marker=',', zorder=3)
# ctx.add_basemap(ax, crs=geo_df.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
plt.show()

# plt.scatter(ests_full.index, ests_full['livneh'], s=1)
# plt.plot(kges)
# plt.plot(nses)
# plt.plot(ests_full)

ds_names = ['chirps', 'daymet', 'era', 'gldas', 'gpm', 'livneh', 'persiann']
datasets_names = {'chirps':'CHIRPS', 'daymet':'Daymet', 'era':'ERA5L',
                  'gldas':'GLDAS', 'gpm':'IMERG', 'livneh':'Livneh', 'persiann':'PERSIANN'}
start_dates = {'era':'1981-01-01T01:00:00', 'gpm':'2000-06-01T00:00:00', 'gldas':'2000-01-01T03:00:00', 
               'daymet':'1980-01-01T00:00:00', 'persiann':'1983-01-01T00:00:00', 'trmm':'1998-01-01T00:00:00', 
               'chirps':'1981-01-01T00:00:00', 'livneh':'1950-01-01T01:00:00'}
end_dates = {'era':'2023-03-30T23:00:00', 'gpm':'2023-05-17T03:30:00', 'gldas':'2023-04-12T21:00:00',
             'daymet':'2021-12-31T00:00:00', 'persiann':'2022-12-31T00:00:00', 'trmm':'2019-12-31T21:00:00',
             'chirps':'2023-03-31T00:00:00', 'livneh':'2013-12-31T23:00:00'}

names = np.array([])
start = np.array([])
end = np.array([])
for ds in ds_names:
    print(ds)
    names = np.append(names, datasets_names[ds])
    start = np.append(start, datetime.datetime.strptime(start_dates[ds], '%Y-%m-%dT%H:%M:%S'))
    end = np.append(end, datetime.datetime.strptime(end_dates[ds], '%Y-%m-%dT%H:%M:%S'))
    # period_years = int((end - start).days / 365.2425 + (end - start).seconds / (365.25*24*60*60))    
    # plt.barh(ds, period_years)

dates_ticks = [datetime.datetime(1950+x*10,1,1) for x in range(8)]
dates_ticklabes = [datetime.datetime(1950+x*10,1,1).year for x in range(8)]
dates_list = [datetime.datetime(1981+x*10,1,1) for x in range(5)]
dates_listlabels = [datetime.datetime(1981+x*10,1,1).year for x in range(5)]

fig, ax = plt.subplots(1, 1)
ax.barh(range(len(ds_names)), end-start, left=start, color='b', label='GEE')
ax.barh(2, datetime.datetime(1981,1,1)-datetime.datetime(1950,1,1), left=datetime.datetime(1950,1,1), color='r')
ax.barh(5, datetime.datetime(2013,12,31)-datetime.datetime(1950,1,1), left=datetime.datetime(1950,1,1), color='r', label='Provider')
ax.vlines(dates_list, -0.5, 6.5, ls='--', color='k')
ax.set_yticks(range(len(ds_names)), names)
ax.set_xticks(dates_ticks, dates_ticklabes)
ax.set_ylim(6.5, -0.5) # ax.invert_yaxis()
ax.set_xlim(datetime.datetime(1949,12,31), datetime.datetime(2023,5,31))
ax2 = ax.secondary_xaxis('top')
ax2.set_xticks(dates_list, dates_listlabels)
# ax.set_title('Data Availability in Gridded Precipitation Products')
ax.legend()
fig.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Figures/Data_availability.jpg', format='jpg', dpi=300, bbox_inches='tight')


fig, ax = plt.subplots(1, 1)
# Plot eac item as a line
for i, (s, e, l) in enumerate(zip(start, end, datasets_names)):
    print(i, s, e, l)
    ax.plot_date([s, e], [(i + 1)/10] * 2, ls='-', marker=None, lw=2)  # 10 for the line width
    if l == 'era':
        ax.plot_date([datetime.datetime(1950,1,1), datetime.datetime(1980,1,1)], [(i + 1)/10] * 2, ls='-', marker=None, lw=2)  # 10 for the line width
plt.vlines(dates_list, 0.05, 0.75, ls='--', color='k', lw=2)
# Set ticks and labels on y axis
ax.set_yticks(np.arange(0.1, len(datasets_names)/10 + 0.1, 0.1))
ax.set_yticklabels(names)
ax.invert_yaxis()
ax.set_title('Data Availability in Gridded Precipitation Products')





#Scatter plots of daily precipitation Rain Gauge vs Datasets

est = 19049
cna_df = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/ESTACIONES/PRECIP/P_'+str(est)+'.csv', parse_dates=['Unnamed: 0'])
cna_df.set_index('Unnamed: 0', inplace=True)
cna_df.index.name = None
cna_df.columns = ['Pcna']

for ds in ds_names:
    ds_df = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/PRECIP/'+ds+'/P'+ds+'_'+str(est)+'.csv', parse_dates=['Unnamed: 0'])
    ds_df.set_index('Unnamed: 0', inplace=True)
    ds_df.index.name = None
    ds_df.columns = ['P'+ds]
    ds_df[ds_df<0.01] = 0

    df_nan = pd.concat([cna_df['Pcna'], ds_df['P'+ds]], axis=1)
    df_nan = df_nan.dropna(how='any')
    
    
    plt.scatter(df_nan['Pcna'], df_nan['P'+ds])










#################################################################################


# plot the polygon
# ax = sMx.plot(alpha=0.35, color='k', zorder=1)
# plot the boundary only (without fill), just uncomment
# ax = gpd.GeoSeries(sMx.to_crs(epsg=4326)['geometry'].unary_union).boundary.plot(ax=ax, alpha=0.5, color="#ed2518",zorder=2)
# ax = gpd.GeoSeries(sMx['geometry'].unary_union).boundary.plot(ax=ax, alpha=0.5, color="#ed2518",zorder=2)
# plot the marker
# ax = sMx.boundary.plot(alpha=1, lw=0.5, color='k', zorder=1)
# ax = sClim.boundary.plot(alpha=1, lw=0.5, color='k', zorder=1)
# x['geometry'].plot(ax=ax, markersize=0.5, c=x['Ppersiann'], cmap='coolwarm', marker=',', zorder=3)
# limits = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False)
# plt.colorbar(norm=limits)

# x['livneh'][x['livneh']<0.3]

#Plot error for all stations in Mexico

# ax = sMx.boundary.plot(alpha=1, lw=0.5, color='k', zorder=1)
# sClim.plot(color='k', edgecolor='w')

# limits1 = mpl.colors.Normalize(vmin=-1, vmax=1, clip=False)
# limits2 = mpl.colors.Normalize(vmin=-30, vmax=30, clip=False)


# fig, ax = plt.subplots(7,4, figsize=(8,10), gridspec_kw={'wspace':0.05, 'hspace':0.05})

# i = -1
# for ds in ds_order:
#     i += 1
    
#     ncond = n[ds]>=-1
#     sClim.boundary.plot(ax=ax[i,0], alpha=1, lw=0.5, color='k', zorder=1)
#     nplot = ax[i,0].scatter(n['LON'][ncond], n['LAT'][ncond], c=n[ds][ncond], cmap='jet_r', norm=limits1, s=1, alpha=1) #, levels=pplt.arange(-1, 1, 0.1))
#     ax[i,0].set_ylabel(ds_names_dict[ds])
    
#     kcond = k[ds]>=-1
#     sClim.boundary.plot(ax=ax[i,1], alpha=1, lw=0.5, color='k', zorder=1)
#     kplot = ax[i,1].scatter(k['LON'][kcond], k['LAT'][kcond], c=k[ds][kcond], cmap='jet_r', norm=limits1, s=1, alpha=1) #, levels=pplt.arange(-1, 1, 0.1))
    
#     ccond = c[ds]>=0
#     sClim.boundary.plot(ax=ax[i,2], alpha=1, lw=0.5, color='k', zorder=1)
#     cplot = ax[i,2].scatter(c['LON'][ccond], c['LAT'][ccond], c=c[ds][ccond], cmap='jet_r', norm=limits1, s=1, alpha=1) #, levels=pplt.arange(-1, 1, 0.1))
    
#     pcond = abs(p[ds])<=25
#     sClim.boundary.plot(ax=ax[i,3], alpha=1, lw=0.5, color='k', zorder=1)
#     pplot = ax[i,3].scatter(p['LON'][pcond], p['LAT'][pcond], c=p[ds][pcond], cmap='jet_r', norm=limits2, s=1, alpha=1) #, levels=pplt.arange(-25, 25, 1))
    
# ax[0,0].set_title('NSE')
# ax[0,1].set_title('KGE')
# ax[0,2].set_title('R')
# ax[0,3].set_title('PBIAS')

# ax[0,0].tick_params(labelbottom=False, labeltop=True, labelleft=True, labelright=False,
#             bottom=False, top=True, left=True, right=False)
# ax[6,0].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
#             bottom=True, top=False, left=True, right=False)
# ax[0,3].tick_params(labelbottom=False, labeltop=True, labelleft=False, labelright=True,
#             bottom=False, top=True, left=False, right=True)
# ax[6,3].tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=True,
#             bottom=True, top=False, left=False, right=True)
# for sp in range(1,6):
#     ax[sp,0].tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False,
#                 bottom=False, top=False, left=True, right=False)
# for sp in range(1,6):
#     ax[sp,3].tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=True,
#                 bottom=False, top=False, left=False, right=True)
# for sp in range(1,3):
#     ax[0,sp].tick_params(labelbottom=False, labeltop=True, labelleft=False, labelright=False,
#                 bottom=False, top=True, left=False, right=False)
# for sp in range(1,3):
#     ax[6,sp].tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
#                 bottom=True, top=False, left=False, right=False)
# for spx in range(1,3):
#     for sp in range(1,6):
#         ax[sp,spx].tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False,
#                     bottom=False, top=False, left=False, right=False)







