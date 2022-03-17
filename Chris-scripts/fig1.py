#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:55:02 2019

@author: shofer modified CK

warming CMIP5 vs CMIP6 for Antarctica
"""

import xarray as xr
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
#import cartopy.crs as ccrs
#import xesmf as xe
#import cartopy.feature as feat
from netCDF4 import num2date

sns.set_context('paper')

CM5 = xr.open_dataset('/home/ckittel/Documents/data/proj/GCM-CM5.nc',
                      decode_times=False)
CM5_new = CM5.rename({'TIME':'MODEL', 'ZZ':'YEAR'})

CM6 = xr.open_dataset('/home/ckittel/Documents/data/proj/GCM-CM6-ssp585.nc',
                      decode_times=False)
CM6_new = CM6.rename({'TIME':'MODEL', 'ZZ':'YEAR','TAS_ANN':'TAS'})



#ERA5= xr.open_dataset('/home/ckittel/Documents/data/proj/ERA5_tasy.nc',
#                      decode_times=False)
#
#
#numdates = ERA5['TIME'].data
#dates = [num2date(d,units='month since 1950-01-15 00:00:00',calendar='360_day').year for d in numdates]
#ERA5['TIME'] = dates
#npdates =  pd.date_range('1980', '2017', freq='AS').values

#ERA5['TIME'] = CM6_new['YEAR'].sel(YEAR=slice(1980,2017)).data


ERA5= xr.open_dataset('/home/ckittel/Documents/data/proj/ERA51979-2020y.nc',
                      decode_times=False)
dates=  pd.date_range('1979-01-01', '2020-12-01', freq='AS')
ERA5['time']=dates.year

ERA5before= xr.open_dataset('/home/ckittel/Documents/data/proj/ERA51950-1978.nc',decode_times=False)
dates=  pd.date_range('1950-01-01', '1978-01-01', freq='AS')


ERA5before['time']=dates.year

# =======================================================================================
# AX2 ANALYSIS
#CM5_base_ts = CM5_new.sel(YEAR=slice(1981,2010)).mean(dim=['YEAR','MODEL','LAT','LON'])
#CM6_base_ts = CM6_new.sel(YEAR=slice(1981,2010)).mean(dim=['YEAR','MODEL','LAT','LON'])
#ER5_base_ts= ERA5.sel(TIME=slice(1981,2010)).mean(dim=['TIME','LAT','LON'])
#
#CM5_anom_ts = CM5_new.mean(dim=['MODEL','LAT','LON']) - CM5_base_ts
#CM6_anom_ts = CM6_new.mean(dim=['MODEL','LAT','LON']) - CM6_base_ts
#ER5_anom_ts = ERA5.mean(dim=['LAT','LON']) - ER5_base_ts

CM5_base_ts_polar = CM5_new.sel(YEAR=slice(1981,2010),LAT=slice(-90,-60)).mean(dim=['YEAR','LAT','LON'])
CM6_base_ts_polar = CM6_new.sel(YEAR=slice(1981,2010),LAT=slice(-90,-60)).mean(dim=['YEAR','LAT','LON'])
ER5_base_ts_polar = ERA5.sel(time=slice(1981,2010),latitude=slice(-60,-90)).mean(dim=['time','latitude','longitude'])


CM5_anom_ts_polar = CM5_new.sel(LAT=slice(-90,-60)).mean(dim=['LAT','LON']) - CM5_base_ts_polar
CM6_anom_ts_polar = CM6_new.sel(LAT=slice(-90,-60)).mean(dim=['LAT','LON']) - CM6_base_ts_polar
ER5_anom_ts_polar = ERA5.sel(latitude=slice(-60,-90)).mean(dim=['latitude','longitude']) - ER5_base_ts_polar
ER5before_anom_ts_polar = ERA5before.sel(latitude=slice(-60,-90)).mean(dim=['latitude','longitude'])-ER5_base_ts_polar['t2m'].values


ER5_anom_ts_polar=ER5_anom_ts_polar.rename({'t2m':'T2M'})

datasets = []

    
ds=ER5before_anom_ts_polar 
datasets.append(ds)
ds=ER5_anom_ts_polar

datasets.append(ds)

    
ER5new = xr.concat(datasets, dim='time')


#datasets.append(ER5before_anom_ts_polar)
#ds=CM6_ssp126_new
#
#datasets.append(ds)
#
#    
#CM6_ssp126_new = xr.concat(datasets, dim='YEAR')

#ER5_anom_ts_polar2 = ERA5.sel(LAT=slice(-90,-60)).mean(dim=['LAT','LON'])-273.15 #

CNRM_CM6_1 = CM6_anom_ts_polar.isel(MODEL=9)# -273.15
CESM2=  CM6_anom_ts_polar.isel(MODEL=6)# -273.15
CAN=  CM6_anom_ts_polar.isel(MODEL=5)# -273.15
ACCESS1_3 = CM5_anom_ts_polar.isel(MODEL=0) #-273.15
NorESM = CM5_anom_ts_polar.isel(MODEL=-1)# -273.15


CM5_polar_anom_mean = CM5_anom_ts_polar.mean(dim=['MODEL']) # -273.15
CM6_polar_anom_mean = CM6_anom_ts_polar.mean(dim=['MODEL']) # -273.15



# PLOTTING ###########################
plt.close('all')
#proj = ccrs.PlateCarree(central_longitude=0.0, globe=None)
fig, ax2 = plt.subplots(nrows=1,ncols=1,figsize=(14,6))
# ============ PLOT ON AX2 ============
pal = sns.color_palette("Blues",10)
pal_or = sns.color_palette("Oranges",10)

lw=2

#CM5_polar_anom_mean['TAS'].sel(YEAR=slice(1950,2100)).plot(ax=ax2, color=pal[9], label='CMIP5 mean', lw=4)
#NorESM['TAS'].sel(YEAR=slice(1950,2100)).plot(ax=ax2, color=pal[4], label='NorESM1-M',lw=2)
#ACCESS1_3['TAS'].sel(YEAR=slice(1950,2100)).plot(ax=ax2, color=pal[7], label='ACCESS1-3',lw=2)
#
#
#CM6_polar_anom_mean['TAS'].sel(YEAR=slice(1950,2100)).plot(ax=ax2, color=pal_or[9], label='CMIP6 mean', lw=4)
#CNRM_CM6_1['TAS'].sel(YEAR=slice(1950,2100)).plot(ax=ax2, color=pal_or[7], label='CNRM-CM6-1',lw=2)
#CESM2['TAS'].sel(YEAR=slice(1950,2100)).plot(ax=ax2, color=pal_or[5], label='CESM2',lw=2)
##CAN['TAS'].sel(YEAR=slice(1950,2100)).plot(ax=ax2, color=pal_or[1], label='CAN',lw=1.)
#
##NorESM['TAS'].sel(YEAR=slice(1950,2100)).plot(ax=ax2, color=pal[4],lw=1.)
##ACCESS1_3['TAS'].sel(YEAR=slice(1950,2100)).plot(ax=ax2, color=pal[7],lw=1.)
#
#
##ER5_anom_ts_polar ['t2m'].plot(ax=ax2, color='k', label='ERA5')
##ER5before_anom_ts_polar['T2M'].plot(ax=ax2, color='k')
##               (ax=ax2, color='k', label='ERA5')
#
#CM5_polar_anom_mean['TAS'].sel(YEAR=slice(1950,2100)).plot(ax=ax2, color=pal[9], lw=3)
#CM6_polar_anom_mean['TAS'].sel(YEAR=slice(1950,2100)).plot(ax=ax2, color=pal_or[9], lw=3)

ER5new['T2M'].sel(time=slice(1950,2100)).plot(ax=ax2, color='k', linestyle='dashed', label='ERA5',lw=3)

ax2.legend(frameon=False, fontsize=12)
ax2.set_xlabel('Year',fontsize=16)

ax2.set_ylabel('90°S-60°S near-surface temperature anomaly ($^{\circ}$C)', fontsize=15)
ax2.tick_params(axis="x", labelsize=14) 
ax2.tick_params(axis="y", labelsize=14)



#ax2.yaxis.set_ticks(range(9))
ax2.set_title('')

#
##sns.despine()
fig.tight_layout()
#
#
fig.savefig('./era.pdf',
            format='PDF')
fig.savefig('./era5.png',
            format='PNG', dpi=600)