# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:33:21 2023

@author: HIDRAULICA-Dani
"""

import os
import pandas as pd
import numpy as np
import spotpy

#Variables
datasets_all = os.listdir('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/PRECIP/')
df_periods = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/len_ests.csv', index_col='Unnamed: 0')
ests25 = df_periods[df_periods['Am']>25].index

errors = spotpy.objectivefunctions.calculate_all_functions(1, 1)
errors_name = [errors[i][0] for i in range(len(errors))]

start_years = np.arange(1981,2012, 10)
timesteps = ['d', 'm', 'mmax', 'y', 'ymax']

error0 = []

#Compute errors for all time series
for timestep in timesteps:
    print(timestep)
    # timestep = 'd'
    for est in ests25:
        # est = 1001
        # print(timestep, est)
        try:
            df_nan_all = pd.read_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/TimeSeries/{timestep}/TimeSeries_{str(est)}_{timestep}.csv', parse_dates=['Unnamed: 0'])
            df_nan_all.set_index('Unnamed: 0', inplace=True)
            df_nan_all.index.name = None
            
            es_df_all = pd.DataFrame()
            
            for year in start_years:
                # print(year)
                # year = 1991
                es_df = pd.DataFrame(errors, columns=['errors', 'value'])
                es_df.set_index('errors', inplace=True)
                es_df.index.name = None
                es_df.drop(['value'], axis=1, inplace=True)
                
                start_date = str(year)+'-01-01T00:00'
                end_date = str(year+10)+'-01-01T00:00'
                
                df = df_nan_all[((df_nan_all.index>=start_date) & (df_nan_all.index<end_date))]
                
                # Computation of errors
                for ds in datasets_all:
                    # ds = 'gldas'
                    df_nan = df[['Pcna', 'P'+ds]].dropna(how='any')
                    # df_nan.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/df_nan.csv')
                    # df_nan = df_nan[df_nan>0].dropna(how='all').replace(np.nan, 0)
                    
                    if len(df_nan.resample('Y').mean().dropna()) > 4:
                        try:
                            errors = spotpy.objectivefunctions.calculate_all_functions(df_nan['Pcna'], df_nan['P'+ds])
                            
                            # nse = spotpy.objectivefunctions.nashsutcliffe(df_nan['Pcna'], df_nan['P'+ds])
                            # kge = spotpy.objectivefunctions.kge(df_nan['Pcna'], df_nan['P'+ds])
                            # pbias = spotpy.objectivefunctions.pbias(df_nan['Pcna'], df_nan['P'+ds])
                            # rsr = spotpy.objectivefunctions.rsr(df_nan['Pcna'], df_nan['P'+ds])
                            # errors = [('nashsutcliffe', nse), ('kge', kge), ('rsr', rsr), ('pbias', pbias)]
                            
                            esk_df = pd.DataFrame(errors, columns=['error', 'P'+ds])
                            esk_df.set_index('error', inplace=True)
                            esk_df.index.name = None
                            
                        except Exception:
                            esk_df = pd.DataFrame(np.nan, index=errors_name, columns=['P'+ds])
                    else:
                        esk_df = pd.DataFrame(np.nan, index=errors_name, columns=['P'+ds])
                        
                    es_df = pd.concat([es_df, esk_df], axis=1)
                        
                es_df_all = pd.concat([es_df_all, es_df], axis=0)
            
            #Save error results
            es_df_all.to_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERRORS/Periods/{timestep}/GEE_Precip_Datasets_errors_{str(est)}_{timestep}.csv', encoding='latin-1')
        
        except Exception:
            print(est, 'NO DATA')
            error0.append(timestep, est)
            pass

################################################################################       
# Save results comparing datasets and timesteps

errors_list = ['nashsutcliffe', 'kge', 'rsr', 'pbias']
periods_dict = {0:'1981-1991', 1:'1991-2001', 2:'2001-2011', 3:'2011-2021'}

for error in errors_list:
    # error = 'pbias'
    for timestep in timesteps:
        # timestep = 'y'
        for period in periods_dict:
            # period = 0
            error_df = pd.DataFrame()
            for est in ests25:
                # est = 1001
                df = pd.read_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERRORS/Periods/{timestep}/GEE_Precip_Datasets_errors_{est}_{timestep}.csv', index_col=['Unnamed: 0'])
                error_est_df = df[df.index==error].iloc[period].to_frame().T
                error_est_df.index = [est]
                error_df = pd.concat([error_df, error_est_df])
            
            error_df.dropna(how='all', subset=error_df.columns, inplace=True)
            error_df.to_csv(f'D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables/ERROR_COMPARISON/P_{periods_dict[period]}_E_{error}_T_{timestep}.csv')
            