# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:47:17 2023

@author: HIDRAULICA-Dani
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from osgeo import gdal
from osgeo import osr
import netCDF4 as nc
from scipy import stats
import pyextremes as pyex
import random

os.chdir('D:/DANI/2023/TEMA_TORMENTAS/DATOS/GEE')
os.listdir()

from DistributionFitting import best_fit_distribution, make_pdf

ensembles = range(11)

data = pd.read_csv('../REFORECASTS/P_228228_lt58_wa.csv', parse_dates=['Unnamed: 0'])
data.index = data['Unnamed: 0']
data.index.name = None
data = data.drop(['Unnamed: 0'], axis=1)

data_ens = pd.DataFrame()
for ens in ensembles:
    data_ens = pd.concat([data_ens, data[str(ens)]], axis=0)

delta = pd.Timedelta(days=1)
dates = pd.date_range(start=data.index[-1]-(len(data_ens)-1)*delta, end=data.index[-1])
data_ens.index = dates

data_max = data_ens.resample('Y').max()[1:-1]
plt.plot(data_max)


n = len(data_max)
bins = int(n/5) if n<35 else math.floor(1.88*n**(2/5))
best_distributions = best_fit_distribution(data_max, bins=bins)
best_dist = best_distributions[0]

pdf = make_pdf(best_dist[0], best_dist[1])
plt.plot(pdf)
plt.hist(data_max, bins=bins, density=True)


def prob2tr(x):
    return 1/x
def tr2prob(x):
    return 1/x

TR = np.array([2,5,10,20,50,100,200,500,1000])
ExP = 1/TR
NonExP = 1 - ExP

dist = best_dist[0]
distribution = getattr(stats, dist)
params = best_dist[1]
arg = params[:-2]
loc = params[-2]
scale = params[-1]
P_Tr = distribution.ppf(NonExP, *arg, loc=loc, scale=scale) if arg else distribution.ppf(NonExP, loc=loc, scale=scale)
if P_Tr[0] > P_Tr[-1]:
    P_Tr = distribution.ppf(ExP, *arg, loc=loc, scale=scale) if arg else distribution.ppf(ExP, loc=loc, scale=scale)

data_s = sorted(data_max[0].values, reverse=True)
m = np.arange(0,n)+1
Pr_data = m/(n+1)

boot_n = range(1000)
n = len(data_max)

df_TR = pd.DataFrame(ExP*100, TR, columns=['Prob'])
i = 0

def TRestimate(data_max, TR=[2,5,10,20,50,100,200,500,1000]):
    params = distribution.fit(new_data)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    Tr = np.array(TR)
    ExP = 1/Tr
    NonExP = 1 - ExP
    P_Tr = distribution.ppf(NonExP, *arg, loc=loc, scale=scale) if arg else distribution.ppf(NonExP, loc=loc, scale=scale)
    df_tr = pd.DataFrame(P_Tr, Tr)
    return df_tr

boot_n = range(1000)
df_TR = pd.DataFrame(ExP*100, TR, columns=['Prob'])
for bn in boot_n:
    new_data = np.random.choice(data_max[0].values, n)
    df_tr = TRestimate(new_data)
    if df_tr.max()[0] < 1e10:
        df_TR = pd.concat([df_TR, df_tr], axis=1)
    else:
        new_data = np.random.choice(data_max[0].values, n)
        df_tr = TRestimate(new_data)
        if df_tr.max()[0] < 1e10:
            df_TR = pd.concat([df_TR, df_tr], axis=1)
        else:
            new_data = np.random.choice(data_max[0].values, n)
            df_tr = TRestimate(new_data)
            if df_tr.max()[0] < 1e10:
                df_TR = pd.concat([df_TR, df_tr], axis=1)
df_TR = df_TR.drop(['Prob'], axis=1)
# df_TR.T.describe()
df_TRs = df_TR.T.reset_index(drop=True)
df_TRq = df_TRs.quantile([0.025, 0.5, 0.975]).T

df_TRs.describe()

fig, ax = plt.subplots(1)
plt.scatter(Pr_data, data_s, marker='.', c='c', zorder=10, s=5, label='Data')
plt.plot(ExP, df_TRq, ls='-', c='y', label='Fit')
plt.xscale('log')
plt.title('Return period Plot - Pmax Reforecasts')
plt.xlabel('Probability')
plt.ylabel('Precipitation [mm]')
plt.xticks(ExP)
ax.invert_xaxis()
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
sax = ax.secondary_xaxis(location='top', functions=(prob2tr, tr2prob))
sax.set_xlabel('Return period [years]')
sax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
sax.set_xticks(TR)
plt.legend()
    
    
    # plt.plot(new_data)
    # best_distributions = best_fit_distribution(new_data, bins=bins)
    # best_dist = best_distributions[0]
    # plt.plot(new_data)
    # dist = best_dist[0]
    # distribution = getattr(stats, dist)
    # params = best_dist[1]
    # params = distribution.fit(new_data)
    # arg = params[:-2]
    # loc = params[-2]
    # scale = params[-1]
    
    # P_Tr = distribution.ppf(NonExP, *arg, loc=loc, scale=scale) if arg else distribution.ppf(NonExP, loc=loc, scale=scale)
    # df_tr = pd.DataFrame(P_Tr, TR, columns=[str(bn)])
    
#     if df_tr.max()[0] < 1e10:
#         df_TR = pd.concat([df_TR, df_tr], axis=1)
#         # pdf = make_pdf(dist, params)
#         # plt.plot(pdf)
#         # plt.hist(new_data, bins=bins, density=True, alpha=0.5)
#     else:
#         i += 1
#         # print(df_tr)
#         new_data = np.random.choice(data_max[0].values, n)
#         params = distribution.fit(new_data)
#         arg = params[:-2]
#         loc = params[-2]
#         scale = params[-1]
#         P_Tr = distribution.ppf(NonExP, *arg, loc=loc, scale=scale) if arg else distribution.ppf(NonExP, loc=loc, scale=scale)
#         df_tr = pd.DataFrame(P_Tr, TR, columns=[str(bn)])
#         if df_tr.max()[0] < 1e10:
#             print(df_tr)
#             df_TR = pd.concat([df_TR, df_tr], axis=1)
#         # pdf = make_pdf(dist, params)
#         # plt.plot(pdf)
#         # plt.hist(new_data, bins=bins, density=True, alpha=0.5)
    
# df_TR = df_TR.drop(['Prob'], axis=1)
# df_TR.T.describe()

# df_TRs = df_TR.T.reset_index(drop=True)
# df_TRq = df_TRs.quantile([0.1, 0.5, 0.9]).T

# plt.plot(df_TRs[df_TRs>1e10])
# plt.plot(df_TRq)

# df_TR.loc[1000].max()
# df_TR.T



len(df_TR.T)

df_TR.max().max()

    # plt.plot(P_Tr)
# data_max.max()

