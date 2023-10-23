# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:41:45 2023

@author: HIDRAULICA-Dani
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

#19125, 28065
df1 = pd.read_csv('D:/DANI/VARIOS/CNA_EST/OTROS/ESTACIONES/PRECIP/P_28065.csv', parse_dates=['Unnamed: 0'])
df1.set_index('Unnamed: 0', drop=True, inplace=True)
df1.index.name = None
df1.sort_index(inplace=True)

df2 = pd.read_csv('D:/DANI/2023/TEMA_TORMENTAS/DATOS/CNA/PRECIP_IDW/P_28065_idw.csv', parse_dates=['Unnamed: 0'])
df2.set_index('Unnamed: 0', drop=True, inplace=True)
df2.index.name = None
df2.sort_index(inplace=True)


plt.plot(df1, alpha=0.5)
plt.plot(df2, alpha=0.5)

mask = ((df1.index > '2010-01-01') & (df1.index < '2011-01-01'))
df1[mask]

plt.plot(df1[mask], alpha=0.5)
plt.plot(df2, alpha=0.5)

