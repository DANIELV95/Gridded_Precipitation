# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:05:15 2023

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

os.chdir('D:/DANI/2023/TEMA_TORMENTAS/DATOS/GEE')
os.listdir()

from DistributionFitting import best_fit_distribution, make_pdf

#Set spatial reference
sr = osr.SpatialReference()
sr.ImportFromEPSG(4326)

#Read tif files and get coordinates of each pixel
ds = gdal.Open("./ERA5_Pmax_Daily/era5_Mx_Pmax_D_1964.tif")
arr = np.array(ds.GetRasterBand(1).ReadAsArray())
width = ds.RasterXSize
height = ds.RasterYSize
gt = ds.GetGeoTransform()
minx = gt[0]
miny = gt[3] + width*gt[4] + height*gt[5]
maxx = gt[0] + width*gt[1] + height*gt[2]
maxy = gt[3]
resx = gt[1]
resy = gt[5]

ds.GetGeoTransform()

lon = np.arange(minx, maxx, resx)
lat = np.arange(maxy, miny, resy)
ilon = np.arange(0,318)
ilat = np.arange(0,183)

#Find closer pixel to specific lat and lon
lon_x = -108.35
lat_y = 27.65
lon_id = (np.abs(lon_x-lon)).argmin()
lat_id = (np.abs(lat_y-lat)).argmin()

#Get info about tiff file
# !gdalinfo ./ERA5_Pmax_Daily/era5_Mx_Pmax_D_1965.tif

def write_geotiff(filename, arr, in_ds):
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    elif arr.dtype == np.float64:
        arr_type = gdal.GDT_Float64
    else:
        arr_type = gdal.GDT_Int32

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)

#Combine all tiff files in a numpy array
# years = np.arange(1964,2022) #Daily Precip ERA5
years = np.arange(1981,2022) #Hourly Precip ERA5
arr = np.zeros((183,317))

for year in years:
    # ds = gdal.Open('./ERA5_Pmax_Daily/era5_Mx_Pmax_D_'+str(year)+'.tif')
    ds = gdal.Open('./ERA5_Pmax_Hourly/era5_Mx_Pmax_'+str(year)+'.tif')
    # ds.SetProjection(sr.ExportToWkt())
    
    arr1 = np.array(ds.GetRasterBand(1).ReadAsArray())
    arr = np.concatenate((arr, arr1), axis=0)

arr_res = np.reshape(arr, (len(years)+1,183,317))
arr_res = arr_res[1:]

# # Write the array to disk
# data = arr_res*1
# with open('ERA5_Pmax_Daily.txt', 'w') as outfile:
# # with open('ERA5_Pmax_Hourly.txt', 'w') as outfile:
#     # I'm writing a header here just for the sake of readability
#     # Any line starting with "#" will be ignored by numpy.loadtxt
#     outfile.write('# Array shape: {0}\n'.format(data.shape))
    
#     # Iterating through a ndimensional array produces slices along
#     # the last axis. This is equivalent to data[i,:,:] in this case
#     for data_slice in data:

#         # The formatting string indicates that I'm writing out
#         # the values in left-justified columns 7 characters in width
#         # with 2 decimal places.  
#         np.savetxt(outfile, data_slice, fmt='%-7.2f')

#         # Writing out a break to indicate different slices...
#         outfile.write('# New slice\n')

# # Read the array from disk
# new_data = np.loadtxt('ERA5_Pmax_Daily.txt')
# # new_data = np.loadtxt('ERA5_Pmax_Hourly.txt')

# # Note that this returned a 2D array!
# print(new_data.shape)

# # However, going back to 3D is easy if we know the 
# # original shape of the array
# new_data = new_data.reshape((58, 183, 317))
# # new_data = new_data.reshape((41, 183, 317))

# # Just to check that they're the same...
# data[np.isnan(data)] = -9999
# new_data[np.isnan(new_data)] = -9999
# np.all(np.round(data,2) == new_data)

#Plot Pmax for each year
for i in range(0,len(years)):
    print(i)
    # i=0
    fig, ax = plt.subplots(1)
    ax.matshow(arr_res[i])
    # divnorm=colors.TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
    c = ax.matshow(arr_res[i], cmap='plasma_r') #, norm=divnorm)
    c.set_clim(0,np.nanmax(arr_res[i]))
    fig.colorbar(c, ax=ax, pad=0.1, shrink=0.8)
    plt.xticks(ilon[np.round(lon,1)%1==0][::3], lon[np.round(lon,1)%1==0].astype(np.int)[::3])
    plt.yticks(ilat[np.round(lat,1)%1==0][::2], lat[np.round(lat,1)%1==0].astype(np.int)[::2])
    plt.tick_params(labelbottom=True, labeltop=True, labelright=True, labelleft=True,
                    bottom=True, top=True, left=True, right=True)
    # plt.title('Pmax Daily - '+str(1964+i))
    plt.title('Pmax Hourly - '+str(1981+i))
    # fig.savefig('../VARIOS/Figures/PmexD/'+str(1964+i)+'.jpg', format='jpg', dpi=300, bbox_inches='tight')
    fig.savefig('../VARIOS/Figures/PmexH/'+str(1981+i)+'.jpg', format='jpg', dpi=300, bbox_inches='tight')
    # write_geotiff('../VARIOS/Figures/PmexD/tiff/'+str(1964+i)+'.tif', arr_res[i], ds)
    write_geotiff('../VARIOS/Figures/PmexH/tiff/'+str(1981+i)+'.tif', arr_res[i], ds)

#Get statistic summary for array
amean = np.mean(arr_res, axis=0)
astd = np.std(arr_res, axis=0)
acv = astd/amean
askew = stats.skew(arr_res, axis=0)


fig, ax = plt.subplots(1)
ax.matshow(arr_res[i])
# divnorm=colors.TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
c = ax.matshow(arr_res[i], cmap='plasma_r') #, norm=divnorm)
c.set_clim(0,np.nanmax(arr_res[i]))
fig.colorbar(c, ax=ax, pad=0.1, shrink=0.8)
plt.xticks(ilon[np.round(lon,1)%1==0][::3], lon[np.round(lon,1)%1==0].astype(np.int)[::3])
plt.yticks(ilat[np.round(lat,1)%1==0][::2], lat[np.round(lat,1)%1==0].astype(np.int)[::2])
plt.tick_params(labelbottom=True, labeltop=True, labelright=True, labelleft=True,
                bottom=True, top=True, left=True, right=True)
# plt.title('Pmax Daily - '+str(1964+i))
plt.title('Pmax Hourly - '+str(1981+i))
# fig.savefig('../VARIOS/Figures/PmexD/'+str(1964+i)+'.jpg', format='jpg', dpi=300, bbox_inches='tight')
fig.savefig('../VARIOS/Figures/PmexH/'+str(1981+i)+'.jpg', format='jpg', dpi=300, bbox_inches='tight')
# write_geotiff('../VARIOS/Figures/PmexD/tiff/Pmean.tif', amean, ds)
write_geotiff('../VARIOS/Figures/PmexH/tiff/PmeanH.tif', amean, ds)

#Plot statistics summary
plt.imshow(amean)
plt.imshow(astd)
plt.imshow(acv)
plt.imshow(askew)

# data = arr_res[:,72,180] #Monterrey Pixel values in array
# x = np.linspace(np.min(data), np.max(data))
# [loc_fit, scale_fit] = stats.gumbel_r.fit(data, method='MM')
# [s_fit, loc_fit, scale_fit] = stats.lognorm.fit(data, method='MM')
# [skew_fit, loc_fit, scale_fit] = stats.pearson3.fit(np.log10(data), method='MM')

#Fit best distribution for each pixel in array and plot results
all_dists_data = np.empty((183,317), dtype=object)
best_dists_data = np.empty((183,317), dtype=object)
best_dists = np.empty((183,317), dtype=object)
best_dists_params = np.empty((183,317), dtype=object)
errors = []

for i in range(height):
    for j in range(width):
        print(i,j)
        try:
            # i = 52
            # j = 101
            data = arr_res[:,i,j]
            if data is not None:
                # data = pd.DataFrame(data, columns=['Data'])
                
                #Get number of bins for histogram and chi square test
                n = len(data)
                bins = int(n/5) if n<35 else math.floor(1.88*n**(2/5))
                
                #Plot histogram of data
                # plt.figure(figsize=(12,8))
                # ax = data.plot(kind='hist', bins=10, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])
                
                # # Save plot limits
                # dataYLim = ax.get_ylim()
                
                # Find best fit distribution
                best_distributions = best_fit_distribution(data, bins=bins)
                best_dist = best_distributions[0]
                # np.savetxt('../VARIOS/Figures/DistFit/txt/dist='+best_dist[0]+'_i='+str(i)+'_j='+str(j)+'.txt', best_distributions, fmt='%s')
                
                # # Update plots
                # ax.set_ylim(dataYLim)
                # ax.set_title(u'Precipitation with \n All Fitted Distributions')
                # ax.set_xlabel(u'Precipitation (mm)')
                # ax.set_ylabel('Frequency')
                # # plt.savefig('../VARIOS/Figures/DistFit/All_dists_i='+str(i)+'_j='+str(j)+'.jpg', format='jpg', dpi=300, bbox_inches='tight')
                
                # # Make PDF with best params
                # pdf = make_pdf(best_dist[0], best_dist[1])
                # plt.plot(pdf)
                # plt.hist(data, bins=bins, density=True)
                
                # # # Display
                # plt.figure(figsize=(12,8))
                # ax = pdf.plot(lw=2, label='PDF')
                # data_y, data_x = np.histogram(data, bins=bins, density=True)
                # ax.hist(data, bins=bins, density=True, alpha=0.5, label='Data')
                # ax.legend()
                # data.plot(kind='hist', bins=bins, density=True, alpha=0.5, label='Data', legend=True, ax=ax)
                
                # dist_shapes = getattr(stats, best_dist[0]).shapes
                # param_names = (dist_shapes + ', loc, scale').split(', ') if dist_shapes else ['loc', 'scale']
                # param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
                # paramchi_names = ['chi2', 'p-value']
                # paramchi_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(paramchi_names, best_dist[4])])
                # dist_str = '{} ({}) ({})'.format(best_dist[0], param_str, paramchi_str)
        
                # ax.set_title(u'Precipitation with best fit distribution i=' + str(i) + ', j=' + str(j) + ' \n' + dist_str)
                # ax.set_xlabel(u'Precipitation (mm)')
                # ax.set_ylabel('Frequency')
                # # plt.savefig('../VARIOS/Figures/DistFit/dist='+best_dist[0]+'_i='+str(i)+'_j='+str(j)+'.jpg', format='jpg', dpi=300, bbox_inches='tight')
                # # plt.close()
                
                # Save results in array
                all_dists_data[i,j] = best_distributions
                best_dists_data[i,j] = best_dist
                best_dists[i,j] = best_dist[0]
                best_dists_params[i,j] = best_dist[1]

        except Exception:
            # errors.append([i,j])
            pass
# np.savetxt('../VARIOS/Figures/DistFit/txt/errors.txt', errors, fmt='%s')


# Check best distributions for Gauge Station Data 19052

data = pd.read_csv('../CNA/19052.csv', encoding='latin-1', index_col='AÑO', usecols=[0,13])
n = len(data)
bins = int(n/5) if n<35 else math.floor(1.88*n**(2/5))
best_distributions = best_fit_distribution(data, bins=bins)
best_dist = best_distributions[0]
pdf = make_pdf(best_dist[0], best_dist[1])
plt.plot(pdf)
plt.hist(data, bins=bins, density=True)

# files = os.listdir('../VARIOS/Figures/DistFit')
# arr_dists = np.empty((183,317), dtype=object)
# for file in files:
#     dist = file[5:-4].split('i=')[0][:-1]
#     i = int(file[5:-4].split('=')[1][:-2])
#     j = int(file[5:-4].split('=')[2])
#     arr_dists[i,j] = dist

#Change the names for code values and plot the results for the best fit
dists_codes = {None:np.nan, 'norm':1, 'expon':2, 'genextreme':3, 'gamma':4, 'gumbel_r':5, 'lognorm':6, 'pearson3':7, 'logpearson3':8}
best_dists_codes = np.copy(best_dists)
for dist in dists_codes:
    print(dist, dists_codes[dist])
    best_dists_codes[best_dists==dist] = dists_codes[dist]
best_dists_codes = best_dists_codes.astype(float)

unique, counts = np.unique(best_dists_codes, return_counts=True)
unique = unique[:-1]
# np.sum(counts[:-1])

#Plot best fitting distributions for each pixel
mat = plt.matshow(best_dists_codes)
colors = [mat.cmap(mat.norm(value)) for value in unique]
patches = [mpatches.Patch(color=colors[i], label=list(dists_codes.keys())[1:][int(i)]) for i in range(len(unique))]
plt.xticks(ilon[np.round(lon,1)%1==0][::3], lon[np.round(lon,1)%1==0].astype(int)[::3])
plt.yticks(ilat[np.round(lat,1)%1==0][::2], lat[np.round(lat,1)%1==0].astype(int)[::2])
plt.tick_params(labelbottom=True, labeltop=True, labelright=True, labelleft=True,
                bottom=True, top=True, left=True, right=True)
# plt.title('Best Fitting Distribution for Pmax Daily')
plt.title('Best Fitting Distribution for Pmax Hourly')
plt.legend(handles=patches, handlelength=0.5, handleheight=0.5, prop={'size':'small'}, title='Distribution', title_fontsize='small')
# plt.savefig('../VARIOS/Figures/DistFit/dists_Pmax_D.jpg', format='jpg', dpi=300, bbox_inches='tight')
plt.savefig('../VARIOS/Figures/DistFit/dists_Pmax_H.jpg', format='jpg', dpi=300, bbox_inches='tight')
# write_geotiff('../VARIOS/Figures/PmexD/tiff/Dists_PmaxD.tif', best_dists_codes, ds)
write_geotiff('../VARIOS/Figures/PmexH/tiff/Dists_PmaxH.tif', best_dists_codes, ds)


#Plot best dist params for each pixel
# best_dists_arg = np.empty((183,317), dtype=object)
# for i in range(height):
#     for j in range(width):
#         print(i,j)
#         bdpa = best_dists_params[i,j]
#         if bdpa is not None:
#             best_dists_arg[i,j] = best_dists_params[i,j][0]
# best_dists_arg = best_dists_arg.astype(float)
# best_dists_arg[best_dists_arg<0] = 0
# best_dists_arg[best_dists_arg>0] = 1
# mat = plt.matshow(best_dists_arg)


# all_dists_data = arr_dists_data_all
# best_dists_data = arr_dists_data
# best_dists = arr_dists
# best_dists_codes = arr_dists_codes

# for i in range(height):
#     for j in range(width):
#         # i = 72
#         # j = 180
#         if best_dists_data[i,j] is not None:
#             best_dists_params[i,j] = best_dists_data[i,j][1]

# Save arrays in numpy format
# np.save('ERA5_Pmax_Daily_dists_data_all.npy', all_dists_data) # save
# np.save('ERA5_Pmax_Daily_dists_data.npy', best_dists_data) # save
# np.save('ERA5_Pmax_Daily_dists.npy', best_dists) # save
# np.save('ERA5_Pmax_Daily_dists_codes.npy', best_dists_codes) # save
# np.save('ERA5_Pmax_Daily_dists_params.npy', best_dists_params) # save

# np.save('ERA5_Pmax_Hourly_dists_data_all.npy', all_dists_data) # save
# np.save('ERA5_Pmax_Hourly_dists_data.npy', best_dists_data) # save
# np.save('ERA5_Pmax_Hourly_dists.npy', best_dists) # save
# np.save('ERA5_Pmax_Hourly_dists_codes.npy', best_dists_codes) # save
# np.save('ERA5_Pmax_Hourly_dists_params.npy', best_dists_params) # save

#Load numpy arrays
all_dists_data = np.load('ERA5_Pmax_Daily_dists_data_all.npy', allow_pickle=True) # load
best_dists_data = np.load('ERA5_Pmax_Daily_dists_data.npy', allow_pickle=True) # load
best_dists = np.load('ERA5_Pmax_Daily_dists.npy', allow_pickle=True) # load
best_dists_codes = np.load('ERA5_Pmax_Daily_dists_codes.npy', allow_pickle=True) # load
best_dists_params = np.load('ERA5_Pmax_Daily_dists_params.npy', allow_pickle=True) # load

all_dists_data = np.load('ERA5_Pmax_Hourly_dists_data_all.npy', allow_pickle=True) # load
best_dists_data = np.load('ERA5_Pmax_Hourly_dists_data.npy', allow_pickle=True) # load
best_dists = np.load('ERA5_Pmax_Hourly_dists.npy', allow_pickle=True) # load
best_dists_codes = np.load('ERA5_Pmax_Hourly_dists_codes.npy', allow_pickle=True) # load
best_dists_params = np.load('ERA5_Pmax_Hourly_dists_params.npy', allow_pickle=True) # load

#Save arrays in txt
# np.savetxt('ERA5_Pmax_Daily_dists_data_all.txt', all_dists_data, fmt='%s')
# np.savetxt('ERA5_Pmax_Daily_dists_data.txt', best_dists_data, fmt='%s')
# np.savetxt('ERA5_Pmax_Daily_dists.txt', best_dists, fmt='%s')
# np.savetxt('ERA5_Pmax_Daily_dists_codes.txt', best_dists_codes, fmt='%s')
# np.savetxt('ERA5_Pmax_Daily_dists_params.txt', best_dists_params, fmt='%s')

# np.savetxt('ERA5_Pmax_Hourly_dists_data_all.txt', all_dists_data, fmt='%s')
# np.savetxt('ERA5_Pmax_Hourly_dists_data.txt', best_dists_data, fmt='%s')
# np.savetxt('ERA5_Pmax_Hourly_dists.txt', best_dists, fmt='%s')
# np.savetxt('ERA5_Pmax_Hourly_dists_codes.txt', best_dists_codes, fmt='%s')
# np.savetxt('ERA5_Pmax_Hourly__dists_params.txt', best_dists_params, fmt='%s')

# best_dists = np.loadtxt('ERA5_Pmax_Daily_dists.txt', dtype=object)
# best_dists = best_dists.reshape((183, 317))
# best_dists[best_dists=='None'] = None

#Get Precipitation for each Return Period and Plot
def prob2tr(x):
    return 1/x
def tr2prob(x):
    return 1/x

TR = np.array([2,5,10,25,50,100,250,500,1000])
ExP = 1/TR
NonExP = 1 - ExP

TR_2 = np.zeros((183,317), dtype=float)
TR_5 = np.zeros((183,317), dtype=float)
TR_10 = np.zeros((183,317), dtype=float)
TR_25 = np.zeros((183,317), dtype=float)
TR_50 = np.zeros((183,317), dtype=float)
TR_100 = np.zeros((183,317), dtype=float)
TR_250 = np.zeros((183,317), dtype=float)
TR_500 = np.zeros((183,317), dtype=float)
TR_1000 = np.zeros((183,317), dtype=float)

TRs_names = {0:'TR 2', 1:'TR 5', 2:'TR 10', 3:'TR 25', 4:'TR 50', 5:'TR 100', 6:'TR 250', 7:'TR 500', 8:'TR 1000'}
TRs = {0:TR_2, 1:TR_5, 2:TR_10, 3:TR_25, 4:TR_50, 5:TR_100, 6:TR_250, 7:TR_500, 8:TR_1000}

# dist_names_hyd = ['norm', 'expon', 'genextreme', 'gamma', 'gumbel_r', 'lognorm', 'pearson3', 'logpearson3']

for i in range(height):
    for j in range(width):
        print(i,j)
        # i, j = [72, 180] # Monterrey
        # i, j = [72, 100] # logpearson3 doubt
        # i, j = [52, 101] # doubt
        # i = 72
        # j = 100
        dist = best_dists[i,j]
        if dist is not None:
            data = arr_res[:,i,j]
            data_s = sorted(data, reverse=True)
            n = len(data)
            m = np.arange(0,n)+1
            Pr_data = m/(n+1)
            # pdf = make_pdf(best_dists_data[i,j][0], best_dists_data[i,j][1])
            
            if dist not in ('logpearson3', 'lognorm'):
                distribution = getattr(stats, dist)
                params = best_dists_params[i,j]
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                P_Tr = distribution.ppf(NonExP, *arg, loc=loc, scale=scale) if arg else distribution.ppf(NonExP, loc=loc, scale=scale)
                if P_Tr[0] > P_Tr[-1]:
                    P_Tr = distribution.ppf(ExP, *arg, loc=loc, scale=scale) if arg else distribution.ppf(ExP, loc=loc, scale=scale)
                
            else:
                if dist == 'logpearson3':
                    distx = 'pearson3'
                elif dist == 'lognorm':
                    distx = 'norm'
                distribution = getattr(stats, distx)

                params = best_dists_params[i,j]
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                P_Trl = distribution.ppf(NonExP, *arg, loc=loc, scale=scale) if arg else distribution.ppf(NonExP, loc=loc, scale=scale)
                if P_Trl[0] > P_Trl[-1]:
                    P_Trl = distribution.ppf(ExP, *arg, loc=loc, scale=scale) if arg else distribution.ppf(ExP, loc=loc, scale=scale)
                P_Tr = 10**P_Trl
            
            # plt.plot(pdf)
            # plt.hist(data, bins=bins, density=True)
            # plt.vlines(P_Tr, ymin=0, ymax=max(pdf), ls='--')
            
            for tr in TRs:
                TRs[tr][i,j] = P_Tr[tr]
            # TR_2[i,j] = P_Tr[0]
            # TR_5[i,j] = P_Tr[1]
            # TR_10[i,j] = P_Tr[2]
            # TR_25[i,j] = P_Tr[3]
            # TR_50[i,j] = P_Tr[4]
            # TR_100[i,j] = P_Tr[5]
            # TR_250[i,j] = P_Tr[6]
            # TR_500[i,j] = P_Tr[7]
            # TR_1000[i,j] = P_Tr[8]
            
            # fig, ax = plt.subplots(1)
            # plt.scatter(Pr_data, data_s, marker='.', c='c', zorder=10, label='Data')
            # plt.plot(ExP, P_Tr, ls='-', c='y', label='Fit')
            # plt.xscale('log')
            # plt.title('Return period Plot - Pmax Daily, i='+str(i)+' j='+str(j))
            # plt.xlabel('Probability')
            # plt.ylabel('Precipitation [mm]')
            # plt.xticks(ExP)
            # ax.invert_xaxis()
            # ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            # sax = ax.secondary_xaxis(location='top', functions=(prob2tr, tr2prob))
            # sax.set_xlabel('Return period [years]')
            # sax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
            # sax.set_xticks(TR)
            # plt.legend()
            # fig.savefig('../VARIOS/Figures/EVA/TRPlot_i='+str(i)+'_j='+str(j)+'.jpg', format='jpg', dpi=300, bbox_inches='tight')

#Write results of Tr for all Mexico
for tr in TRs:
    TRs[tr][TRs[tr]<=1] = np.nan
    # write_geotiff('../VARIOS/Figures/PmexD/tiff/PmaxD_'+TRs_names[tr]+'.tif', TRs[tr], ds)
    # write_geotiff('../VARIOS/Figures/PmexH/tiff/PmaxH_'+TRs_names[tr]+'.tif', TRs[tr], ds)
    # TRs[tr][TRs[tr]>10000] = np.nan

#Read results
for tr in TRs:
    # i, j = [72, 180] # Monterrey
    # tr = 0
    dsD = gdal.Open('../VARIOS/Figures/PmexD/tiff/PmaxD_'+TRs_names[tr]+'.tif')
    arrD = np.array(dsD.GetRasterBand(1).ReadAsArray())
    dsH = gdal.Open('../VARIOS/Figures/PmexH/tiff/PmaxH_'+TRs_names[tr]+'.tif')
    arrH = np.array(dsH.GetRasterBand(1).ReadAsArray())
    R = arrH/arrD
    R[R>1] = np.nan
    write_geotiff('../VARIOS/Figures/Rmex/tiff/R_'+TRs_names[tr]+'.tif', R, ds)
    
    fig, ax = plt.subplots(1)
    ax.matshow(R)
    c = ax.matshow(R, cmap='plasma_r') #, norm=divnorm)
    c.set_clim(0,np.nanmax(R))
    fig.colorbar(c, ax=ax, pad=0.1, shrink=0.8)
    plt.xticks(ilon[np.round(lon,1)%1==0][::3], lon[np.round(lon,1)%1==0].astype(int)[::3])
    plt.yticks(ilat[np.round(lat,1)%1==0][::2], lat[np.round(lat,1)%1==0].astype(int)[::2])
    plt.tick_params(labelbottom=True, labeltop=True, labelright=True, labelleft=True,
                    bottom=True, top=True, left=True, right=True)
    plt.title('R for '+TRs_names[tr])
    plt.savefig('../VARIOS/Figures/Rmex/R_'+TRs_names[tr]+'.jpg', format='jpg', dpi=300, bbox_inches='tight')

arrRs = []
for tr in TRs:
    dsR = gdal.Open('../VARIOS/Figures/Rmex/tiff/R_'+TRs_names[tr]+'.tif')
    arrR = np.array(dsR.GetRasterBand(1).ReadAsArray())
    arrRs.append(arrR)
arrRs = np.reshape(arrRs, (9,183,317))

arrRs[:,i,j]

Rmean = np.nanmean(arrRs, axis=0)
fig, ax = plt.subplots(1)
ax.matshow(Rmean)
c = ax.matshow(Rmean, cmap='plasma_r') #, norm=divnorm)
c.set_clim(0,np.nanmax(Rmean))
fig.colorbar(c, ax=ax, pad=0.1, shrink=0.8)
plt.xticks(ilon[np.round(lon,1)%1==0][::3], lon[np.round(lon,1)%1==0].astype(int)[::3])
plt.yticks(ilat[np.round(lat,1)%1==0][::2], lat[np.round(lat,1)%1==0].astype(int)[::2])
plt.tick_params(labelbottom=True, labeltop=True, labelright=True, labelleft=True,
                bottom=True, top=True, left=True, right=True)
plt.title('R mean')
plt.savefig('../VARIOS/Figures/Rmex/Rmean.jpg', format='jpg', dpi=300, bbox_inches='tight')
write_geotiff('../VARIOS/Figures/Rmex/tiff/Rmean.tif', Rmean, ds)

Rmax = np.nanmax(arrRs, axis=0)
fig, ax = plt.subplots(1)
ax.matshow(Rmax)
c = ax.matshow(Rmax, cmap='plasma_r') #, norm=divnorm)
c.set_clim(0,np.nanmax(Rmax))
fig.colorbar(c, ax=ax, pad=0.1, shrink=0.8)
plt.xticks(ilon[np.round(lon,1)%1==0][::3], lon[np.round(lon,1)%1==0].astype(int)[::3])
plt.yticks(ilat[np.round(lat,1)%1==0][::2], lat[np.round(lat,1)%1==0].astype(int)[::2])
plt.tick_params(labelbottom=True, labeltop=True, labelright=True, labelleft=True,
                bottom=True, top=True, left=True, right=True)
plt.title('R max')
plt.savefig('../VARIOS/Figures/Rmex/Rmax.jpg', format='jpg', dpi=300, bbox_inches='tight')
write_geotiff('../VARIOS/Figures/Rmex/tiff/Rmax.tif', Rmean, ds)


np.mean(arrRs, axis=0)

arrRs[:,i,j]

len(TRs)
arrR.shape



#Plot best fitting distributions for each pixel
fig, ax = plt.subplots(1)
ax.matshow(TR_10)
c = ax.matshow(TR_10, cmap='plasma_r') #, norm=divnorm)
c.set_clim(0,np.nanmax(TR_10))
fig.colorbar(c, ax=ax, pad=0.1, shrink=0.8)
plt.xticks(ilon[np.round(lon,1)%1==0][::3], lon[np.round(lon,1)%1==0].astype(int)[::3])
plt.yticks(ilat[np.round(lat,1)%1==0][::2], lat[np.round(lat,1)%1==0].astype(int)[::2])
plt.tick_params(labelbottom=True, labeltop=True, labelright=True, labelleft=True,
                bottom=True, top=True, left=True, right=True)
plt.title('Pmax Daily for TR 2')
# plt.title('Best Fitting Distribution for Pmax Hourly')
plt.savefig('../VARIOS/Figures/DistFit/dists_Pmax_D.jpg', format='jpg', dpi=300, bbox_inches='tight')
# plt.savefig('../VARIOS/Figures/DistFit/dists_Pmax_H.jpg', format='jpg', dpi=300, bbox_inches='tight')



np.nanmax(TR_2)
np.nanmin(TR_2)

unique_vals, counts_vals = np.unique(TR_2, return_counts=True)

plt.plot(unique_vals, counts_vals)

plt.hist(TR_2)

##############################################################################################

#Trial and error tests

y, x = np.histogram(data, bins=10, density=True)
x = (x + np.roll(x, -1))[:-1] / 2.0

distribution = getattr(stats, 'gumbel_r')

params = distribution.fit(data)

# Separate parts of parameters
arg = params[:-2]
loc = params[-2]
scale = params[-1]
pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
sse = np.sum(np.power(y - pdf, 2.0))
rmse = np.sqrt(np.power(y - pdf, 2.0).mean())

pdfx = pdf*np.sum(y)/np.sum(pdf)
chisq = stats.chisquare(y, f_exp=pdfx)

(chisq[0], chisq[1])






o = np.random.normal(10,1,100)


e = stats.norm.fit(o)

print(stats.lognorm.shapes)
np.mean(o)
np.std(o)
# o = np.array([20.0,25.0,40.0,55.0,60.0,20.0,25.0,40.0,55.0,60.0])
# e = np.array([21.0,23.0,24.0,30.0,61.0])

y, x = np.histogram(o, bins=10) #, range=[round(np.min(o),0),round(np.max(o),0)])
deltab = (x[1]-x[0])/2
binsx = (x+deltab)[:-1]

binsx

plt.bar(binsx, y, width=0.5)
y,x, plotx = plt.hist(o, bins=10, rwidth=0.5)

len(x)

e[0]
np.sum(np.round(stats.norm.pdf(binsx, loc=e[0], scale=e[1])*100, 0))
np.sum(stats.norm.ppf(binsx, loc=e[0], scale=e[1]))


ex = e*np.sum(o)/np.sum(e)
chix = np.sum((o-ex)**2/ex)
p = 1 - stats.chi2.cdf(chix, 4)

stats.chisquare(o, ex, 0)

stats.gumbel_r.name
stats.gumbel_r.name

getattr(stats, stats.gumbel_r.name)

stats.{dist}.rvs(100)


pdf = make_pdf(best_dist[0], best_dist[1])

stats.chi(chix)


mu = np.mean(x)
sigma = np.std(x, ddof=0)

y = np.log(x)
mu_y = np.mean(y)
sigma_y = np.std(y, ddof=0)
scale_y = np.exp(mu_y)

normpdf = 1/(sigma*(2*np.pi)**0.5)*np.exp(-((x-mu)**2)/(2*sigma**2))
[loc_fit, scale_fit] = stats.norm.fit(x, method='MM')
stats.norm.pdf(x, loc=loc_fit, scale=scale_fit)

lognormpdf = 1/(x*sigma_y*(2*np.pi)**0.5)*np.exp(-(y-mu_y)**2/(2*sigma_y**2))
# [s_fit, loc_fit, scale_fit] = stats.lognorm.fit(x, method='MM')
# stats.lognorm.pdf(x, s_fit, loc=loc_fit, scale=scale_fit)

lambdax = 1/mu
exponpdf = lambdax*np.exp(-lambdax*x)
[loc_fit, scale_fit] = stats.expon.fit(x, method='MLE')
stats.expon.pdf(x, loc=loc_fit, scale=scale_fit)


[a_fit, loc_fit, scale_fit] = stats.gamma.fit(x, method='MM')
stats.gamma.pdf(x, a_fit, loc=loc_fit, scale=scale_fit)

y = np.linspace(np.min(np.log10(data)), np.max(np.log10(data)))

plt.plot(data)

plt.hist(np.log10(data), bins=10, density=True)
plt.plot(y, stats.pearson3.pdf(y, skew=skew_fit, loc=loc_fit, scale=scale_fit))

plt.hist(data, bins=15, density=True)
plt.plot(x, stats.norm.pdf(x, mu, st))
plt.plot(x, stats.gumbel_r.pdf(x, loc_fit, scale_fit))

stats.gumbel_r.pdf(x, mu, st)


best_distibutions = best_fit_distribution(x, 20)
best_dist = best_distibutions[0]
pdf = make_pdf(best_dist[0], best_dist[1])

plt.hist(x, bins=20, density=True)
plt.plot(pdf)

data = pd.DataFrame(x)

# Plot for comparison
plt.figure(figsize=(12,8))
ax = data.plot(kind='hist', bins=15, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])

# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_distibutions = best_fit_distribution(x, 15)
best_dist = best_distibutions[0]

# Update plots
ax.set_ylim(dataYLim)
ax.set_title(u'El Niño sea temp.\n All Fitted Distributions')
ax.set_xlabel(u'Temp (°C)')
ax.set_ylabel('Frequency')

# Make PDF with best params 
pdf = make_pdf(best_dist[0], best_dist[1])

# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=15, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
dist_str = '{}({})'.format(best_dist[0].name, param_str)

ax.set_title(u'El Niño sea temp. with best fit distribution \n' + dist_str)
ax.set_xlabel(u'Temp. (°C)')
ax.set_ylabel('Frequency')



mean, var, skew, kurt = stats.gumbel_r.stats(moments='mvsk')


stats.rv_continuous.fit(arr_res[:,72,180])


x = [[[1,1], [2,2]],
     [[3,3], [4,4]]]

np.mean(x, axis=0)


