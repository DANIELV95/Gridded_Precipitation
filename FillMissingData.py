# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 13:52:22 2023

@author: HIDRAULICA-Dani
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

custom_date_parser = lambda x: datetime.strptime(x, "%d/%m/%Y")

os.chdir('D:/DANI/2023/TEMA_TORMENTAS/DATOS/VARIOS/Tables')

ests = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/Catalogo_estaciones.csv', encoding='latin-1')
ests.set_index('Unnamed: 0', inplace=True)
ests.index.name = None
ests = ests.drop([26204, 26207, 26208])
ests.sort_index(inplace=True)
# ests25 = ests[ests['PERIODO']>=25]
ests

# dist_km = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/distancias_estaciones.csv', encoding='latin-1', index_col=['Unnamed: 0'])
# dist_km = dist_km[dist_km<50]
# dist_km.replace(0,np.nan, inplace=True)
# dist_km.to_csv('D:/DANI/VARIOS/CNA_EST/OTROS/distancias_estaciones_menor50km.csv', encoding='latin-1')

dist_km = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/distancias_estaciones_menor50km.csv', encoding='latin-1', index_col=['Unnamed: 0'])
dist_km = dist_km[ests.index.astype(str)]
dist_km = dist_km.T[ests.index]
dist_km.index = dist_km.index.astype(int)
dist_km

# Stations without data
x_list = np.array([5017,  5054,  5064, 10175, 10179, 17085, 25136, 25153,
                   30138, 30210, 30217, 30226, 30230, 30238, 30248, 30262,
                   30305, 30349, 30386, 30387, 30388, 30389, 30390, 30391,
                   30392, 30393, 30394, 30395, 30396, 30397, 30398, 32064,
                   32066])

ests.drop(x_list, inplace=True)
dist_km.drop(x_list, inplace=True)

###################################################################################

#Fill na values with Inverse distance weighting
W = 1/dist_km**2
errors0 = []

# errors0_x = np.array(errors0)
# errors0_x_final = np.array([x for x in errors0_x if x not in x_list])
# np.savetxt('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/errors0_x_final.csv', errors0_x_final, delimiter=",")

# ests_none = []
# for est in ests.index:
#     print(est)
#     try:
#         dfA = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/ESTACIONES/PRECIP/P_'+str(est)+'.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
#     except Exception:
#         print(est, 'ERROR')
#         ests_none.append(est)
#         pass

for est in ests.index:
    # print(est)
    try:
        # est = 19068
        dfa = pd.DataFrame()
        dist_ests = dist_km[est].dropna()
        W_ests = W[est].dropna()
        
        # dfA = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_NA/P_'+str(est)+'_na.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
        dfA = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/ESTACIONES/PRECIP/P_'+str(est)+'.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
        dfA.set_index('Unnamed: 0', inplace=True)
        dfA.index.name = None
        dfA.columns = [est]
        
        dfa = pd.concat([dfa, dfA], axis=1)
        
        for est_i in dist_ests.index:
            # est_i = 1004
            # dfB = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_NA/P_'+str(est_i)+'_na.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
            dfB = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/ESTACIONES/PRECIP/P_'+str(est_i)+'.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
            dfB.set_index('Unnamed: 0', inplace=True)
            dfB.index.name = None
            dfB.columns = [est_i]
            
            dfa = pd.concat([dfa, dfB], axis=1)
            
        dfa.sort_index()
        # plt.plot(dfa)
        
        dfA_fill = dfa[est].to_frame()
        
        for row, i in dfa.iterrows():
            # print(row, i)
            if np.isnan(i[est]):
                # print(row)
                j = i.dropna()
                w = W_ests[j.index]
                if len(j)>2:
                    # print(len(j))
                    E1 = (j*w).sum()
                    E2 = w.sum()
                    Px = E1/E2
                    dfA_fill[est][row] = Px
                else:
                    # print(row)
                    dfA_fill[est][row] = np.nan
        
        dfA_fill.dropna(inplace=True)
        df_dt = pd.date_range(dfA_fill.index[0].strftime('%Y-%m-%d'), dfA_fill.index[-1].strftime('%Y-%m-%d'), freq='D')
        dfA_fill = dfA_fill.reindex(df_dt, fill_value=np.nan)
        # plt.plot(dfA_fill)
        # plt.plot(dfa[19068])
        
        dfA_fill.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_IDW/P_'+str(est)+'_idw.csv', date_format='%d/%m/%Y')
    except Exception:
        print(est, 'ERROR0')
        errors0.append(est)

errors0_a = np.array(errors0)
###################################################################################
errors1 = []

#Correct data, remove years with no precipitation and few records
for est in ests.index:
    try:
        # est = 1001
        dfA = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_IDW/P_'+str(est)+'_idw.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
        dfA.set_index('Unnamed: 0', inplace=True)
        dfA.index.name = None
        dfA.columns = [est]
        
        dfA_y = dfA.resample('Y').sum()
        years_na = dfA_y[dfA_y<30].dropna().index.year #Years with annual precipitation less than 50 mm
        dfA_na = dfA[~dfA.index.year.isin(years_na)]
        
        records = dfA.groupby(dfA.index.year).count()
        years_nr = records[records<90].dropna().index #Years with less than 90 daily records
        dfA_nr = dfA_na[~dfA_na.index.year.isin(years_nr)]
        
        df = dfA_nr[dfA_nr.index>'1900-01-01']
        df.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_NA/P_'+str(est)+'_na.csv', date_format='%d/%m/%Y')
    except Exception:
        print(est, 'ERROR1')
        errors1.append(est)

###################################################################################
#Outliers detection
Kn = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/EXCEL/ValoresKn_int.csv', index_col='n')

for est in ests.index:
    # est = 7343
    dfA = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/ESTACIONES/PRECIP/P_'+str(est)+'.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
    dfA.set_index('Unnamed: 0', inplace=True)
    dfA.index.name = None
    dfA.columns = [est]
    dfA_y = dfA.resample('Y').sum()
    # dfA_y = dfA.resample('Y').max()
    dfA_y = dfA_y[dfA_y>0].dropna()
    dfA_y_len = len(dfA_y)
    
    dfB = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_IDW/P_'+str(est)+'_idw.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
    dfB.set_index('Unnamed: 0', inplace=True)
    dfB.index.name = None
    dfB.columns = [est]
    # dfB_y = dfB.resample('Y').sum()
    dfB_y = dfB.resample('Y').max()
    dfB_y = dfB_y[dfB_y>0].dropna()
    dfB_y_len = len(dfB_y)
    
    # dfC = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_NA/P_'+str(est)+'_na.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
    # dfC.set_index('Unnamed: 0', inplace=True)
    # dfC.index.name = None
    # dfC.columns = [est]
    # dfC_y = dfC.resample('Y').sum()
    # # dfC_y = dfC.resample('Y').max()
    # dfC_y = dfC_y[dfC_y>0].dropna()
    # dfC_y_len = len(dfC_y)
    
    # plt.plot(dfA_y, color='k', marker='.', label='Original')
    # plt.plot(dfB_y, color='b', marker='.', label='IDW')
    # plt.plot(dfC_y, color='r', marker='.', label='NA')
    # plt.plot(dfD_y, color='c', marker='.', label='OUT')
    # plt.legend()
    
    # plt.plot(dfA[((dfA.index>'2010-06-01') & (dfA.index<'2010-10-01'))], color='k', marker='.', label='Original')
    # plt.plot(dfB[((dfB.index>'2010-06-01') & (dfB.index<'2010-10-01'))], color='b', marker='.', label='IDW')
    # plt.plot(dfC[((dfC.index>'2010-06-01') & (dfC.index<'2010-10-01'))], color='r', marker='.', label='NA')
    # plt.legend()
    
    # plt.plot(dfA, color='k', marker='.', label='Original')
    # plt.plot(dfB, color='b', marker='.', label='IDW')
    # plt.plot(dfC, color='r', marker='.', label='NA')
    # plt.legend()
    
    #Outliers WDC Kn Value
    # ns = pd.DataFrame(range(10,141), columns=['n'])
    # ns.set_index('n', inplace=True)
    # kn = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/EXCEL/ValoresKn.csv', index_col='n')
    # Kn = pd.concat([ns,kn], axis=1).interpolate()
    # Kn.to_csv('D:/DANI/2023/TEMA_TORMENTAS/EXCEL/ValoresKn_int.csv')
    
    dfD_y = dfB_y.copy()
    dfD_y_len = len(dfD_y)
    
    cond = [0,1]
    outs = pd.DataFrame()
    # outliers = pd.DataFrame()
    # plt.plot(dfD_y, marker='.', alpha=0.75)
    
    while cond[-1] - cond[-2] != 0:
        # print(cond[-1])
        
        #WRC
        n = len(dfD_y)
        y = np.log10(dfD_y)
        my = y.mean().values[0]
        sy = y.std().values[0]
        sky = y.skew().values[0]
        
        if n<10:
            k = Kn.values[0][0]
        elif n>140:
            k = Kn.values[-1][0]
        else:
            k = Kn.loc[n].values[0]

        if sky > 0.4:
            yh = my + k*sy
            outh = 10**y[y>yh].dropna()
            if len(outh) == 0:
                yl = my - k*sy
                outl = 10**y[y<yl].dropna()
            else:
                outl = pd.DataFrame()
        elif sky< -0.4:
            yl = my - k*sy
            outl = 10**y[y<yl].dropna()
            if len(outl) == 0:
                yh = my + k*sy
                outh = 10**y[y>yh].dropna()
            else:
                outh = pd.DataFrame()
        else:
            yh = my + k*sy
            yl = my - k*sy
            outh = 10**y[y>yh].dropna()
            outl = 10**y[y<yl].dropna()
        
        # #Z-score
        # m = dfD_y.mean()
        # s = dfD_y.std()
        # z1 = (dfD_y - m)/s
        # outz = dfD_y[abs(z1)>=3].dropna()
        
        #Modified Z-score
        md = dfD_y.median()
        MAD = abs(dfD_y - md).median()
        z2 = 0.6745*(dfD_y - md)/MAD
        outm = dfD_y[abs(z2)>=3.5].dropna()
        
        m = dfD_y.mean()
        # outliers = pd.concat([outm]).drop_duplicates()
        outliers = pd.concat([outh, outl, outm]).drop_duplicates()
        # outliers = pd.concat([outh, outl, outz, outm]).drop_duplicates()
        outliers_dropindex = outliers.idxmin()
        # outliers = outliers[outliers<m].dropna()
        # dfD_y.drop(outliers.index, inplace=True)
        dfD_y.drop(outliers_dropindex, inplace=True)
        
        if len(outliers) > 0:
            cond.append(cond[-1]+1)
            outs = pd.concat([outs, outliers])
        else:
            cond.append(cond[-1])
        
        # plt.plot(dfD_y, marker='.')
    
    if abs(dfD_y_len - dfA_y_len) >= 10: #len(dfD_y):
        outs.sort_index()
        outs.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_OUT/OUTLIERS/outliers_'+str(est)+'.csv', date_format='%d/%m/%Y')
        
        # plt.plot(dfA_y, label='Original', zorder=2, ls='--')
        # plt.plot(dfD_y, label='Corrected', zorder=1, marker='.')
        plt.plot(dfA_y, color='k', label='Original', ls='--', zorder=2)
        plt.plot(dfB_y, color='b', marker='.', label='IDW', zorder=1)
        # plt.plot(dfC_y, color='r', marker='.', label='NA', zorder=1)
        plt.plot(dfD_y, color='c', marker='.', label='OUT', zorder=1)
        # plt.plot(dfA_y[dfA_y.index>'1980-01-01'], label='Original', zorder=2, ls='--')
        # plt.plot(dfC_y[dfC_y.index>'1980-01-01'], label='Corrected', zorder=1, marker='.')
        plt.ylabel('Annual Precipitation [mm]')
        plt.legend()
        plt.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_OUT/IMAGES/P_'+str(est)+'_out.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        plt.close()
    
    years_out = dfD_y.dropna().index.year #Years with annual precipitation less than 50 mm
    dfC_out = dfD[dfD.index.year.isin(years_out)]
    dfC_out.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_OUT/P_'+str(est)+'_out.csv', date_format='%d/%m/%Y')
    
###################################################################################

#Correct data, remove years with no precipitation and few records with corrected time series
for est in ests.index:
    # est = 2144
    dfA = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_OUT/P_'+str(est)+'_out.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
    dfA.set_index('Unnamed: 0', inplace=True)
    dfA.index.name = None
    dfA.columns = [est]
    
    dfA_y = dfA.resample('Y').sum()
    years_na = dfA_y[dfA_y<30].dropna().index.year #Years with annual precipitation less than 50 mm
    dfA_na = dfA[~dfA.index.year.isin(years_na)]
    
    records = dfA.groupby(dfA.index.year).count()
    years_nr = records[records<90].dropna().index #Years with less than 90 daily records
    dfA_nr = dfA_na[~dfA_na.index.year.isin(years_nr)]
    
    df = dfA_nr[dfA_nr.index>'1900-01-01']
    
    df_y_len = len(df.groupby(df.index.year).sum())
    
    if df_y_len >= 25:
        df.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_MOD/P_'+str(est)+'_mod.csv', date_format='%d/%m/%Y')


###################################################################################

#Outliers detection original data
# Kn = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/EXCEL/ValoresKn_int.csv', index_col='n')

# est = 19004
# dfA = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/ESTACIONES/PRECIP/P_'+str(est)+'.csv', parse_dates=['Unnamed: 0'], date_parser=custom_date_parser)
# dfA.set_index('Unnamed: 0', inplace=True)
# dfA.index.name = None
# dfA.columns = [est]

# dfA_y = dfA.resample('Y').sum()
# years_na = dfA_y[dfA_y<50].dropna().index.year #Years with annual precipitation less than 50 mm
# dfA_na = dfA[~dfA.index.year.isin(years_na)]

# records = dfA.groupby(dfA.index.year).count()
# years_nr = records[records<120].dropna().index #Years with less than 90 daily records
# dfA_nr = dfA_na[~dfA_na.index.year.isin(years_nr)]

# df = dfA_nr[dfA_nr.index>'1900-01-01']

# dfA_y = df.resample('Y').max().dropna()
# dfA_y_len = len(dfA_y)

# cond = [0,1]
# outs = pd.DataFrame()
# # plt.plot(dfA_y, marker='.', alpha=0.75)
# plt.plot(df[df.index.year == 1984], marker='.', alpha=0.75)

# df[df.index.year == 1984].dropna()

# while cond[-1] - cond[-2] != 0:
#     # print(cond[-1])
#     n = len(dfA_y)
#     m = dfA_y.mean()
#     s = dfA_y.std()
#     md = dfA_y.median()
#     sk = dfA_y.skew()
    
#     if n<10:
#         k = Kn.values[0][0]
#     elif n>140:
#         k = Kn.values[-1][0]
#     else:
#         k = Kn.loc[n].values[0]
    
#     #WRC
#     yh = m + k*s
#     yl = m - k*s
#     outh = dfA_y[dfA_y>yh].dropna()
#     outl = dfA_y[dfA_y<yl].dropna()
    
#     #Z-score
#     z1 = (dfA_y - m)/s
#     outz = dfA_y[abs(z1)>=3].dropna()
    
#     #Modified Z-score
    
#     MAD = abs(dfA_y - md).median()
#     z2 = 0.6745*(dfA_y - md)/MAD
#     outm = dfA_y[abs(z2)>=3].dropna()
    
#     outliers = pd.concat([outh, outl, outz, outm]).drop_duplicates()
#     outliers = outliers[outliers<m].dropna()
#     dfA_y.drop(outliers.index, inplace=True)
    
#     if len(outliers) > 0:
#         cond.append(cond[-1]+1)
#         outs = pd.concat([outs, outliers])
#     else:
#         cond.append(cond[-1])
    
#     # plt.plot(dfC_y, marker='.')

# if dfA_y_len > len(dfA_y):
#     outs.sort_index()
#     outs.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_OUT/OUTLIERS/outliers_'+str(est)+'.csv', date_format='%d/%m/%Y')
    
#     # plt.plot(dfA_y[dfA_y.index>'1980-01-01'], label='Original', zorder=2)
#     # plt.plot(dfC_y[dfC_y.index>'1980-01-01'], label='Corrected', zorder=1, marker='.')
#     # plt.ylabel('Annual Precipitation [mm]')
#     # plt.legend()
#     # plt.savefig('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_OUT/IMAGES/P_'+str(est)+'_out.jpg', format='jpg', dpi=1000, bbox_inches='tight')
#     # plt.close()

# years_out = dfA_y.dropna().index.year #Years with annual precipitation less than 50 mm
# dfC_out = dfA[dfA.index.year.isin(years_out)]
# dfC_out.to_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_OUT/P_'+str(est)+'_out.csv', date_format='%d/%m/%Y')

###################################################################################


