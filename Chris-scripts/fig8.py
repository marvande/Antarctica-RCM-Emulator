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

CM6 = xr.open_dataset('/home/ckittel/Documents/data/proj/GCM-CM6_test_all.nc',
                      decode_times=False)
CM6_new = CM6.rename({'TIME':'MODEL', 'ZZ':'YEAR','TAS_ANN':'TAS'})



ERA5= xr.open_dataset('/home/ckittel/Documents/data/proj/ERA5_tasy.nc',
                      decode_times=False)

numdates = ERA5['TIME'].data
dates = [num2date(d,units='month since 1950-01-15 00:00:00',calendar='360_day').year for d in numdates]
ERA5['TIME'] = dates
#npdates =  pd.date_range('1980', '2017', freq='AS').values

#ERA5['TIME'] = CM6_new['YEAR'].sel(YEAR=slice(1980,2017)).data


# =======================================================================================
# AX2 ANALYSIS
CM5_base_ts = CM5_new.sel(YEAR=slice(1980,2009)).mean(dim=['YEAR','MODEL','LAT','LON'])
CM6_base_ts = CM6_new.sel(YEAR=slice(1980,2009)).mean(dim=['YEAR','MODEL','LAT','LON'])

CM5_anom_ts = CM5_new.mean(dim=['MODEL','LAT','LON']) - CM5_base_ts
CM6_anom_ts = CM6_new.mean(dim=['MODEL','LAT','LON']) - CM6_base_ts

CM5_base_ts_polar = CM5_new.sel(YEAR=slice(1980,2009),LAT=slice(-90,-60)).mean(dim=['YEAR','LAT','LON'])
CM6_base_ts_polar = CM6_new.sel(YEAR=slice(1980,2009),LAT=slice(-90,-60)).mean(dim=['YEAR','LAT','LON'])

CM5_anom_ts_polar = CM5_new.sel(LAT=slice(-90,-60)).mean(dim=['LAT','LON']) - CM5_base_ts_polar
CM6_anom_ts_polar = CM6_new.sel(LAT=slice(-90,-60)).mean(dim=['LAT','LON']) - CM6_base_ts_polar

ERA5_anom_ts_polar = ERA5.sel(LAT=slice(-90,-60)).mean(dim=['LAT','LON'])#-273.15 #

#CNRM_CM6_1 = CM6_anom_ts_polar.isel(MODEL=9)# -273.15
#CESM2=  CM6_anom_ts_polar.isel(MODEL=6)# -273.15
#CAN=  CM6_anom_ts_polar.isel(MODEL=5)# -273.15
#ACCESS1_3 = CM5_anom_ts_polar.isel(MODEL=0)# -273.15
#NorESM = CM5_anom_ts_polar.isel(MODEL=-1)# -273.15


CM5_polar_anom_mean = CM5_anom_ts_polar.mean(dim=['MODEL'])# -273.15
CM6_polar_anom_mean = CM6_anom_ts_polar.mean(dim=['MODEL'])# -273.15



def ru2(x, a, b,c):
    return (a*x*x)+ (b * x) + c
def sf1(x, a, b):
    return (a * x) + b

# PLOTTING ###########################
plt.close('all')
#proj = ccrs.PlateCarree(central_longitude=0.0, globe=None)
fig, ax2 = plt.subplots(nrows=1,ncols=2,figsize=(14,6))
# ============ PLOT ON AX2 ============
pal = sns.color_palette("Blues",10)
pal_or = sns.color_palette("Oranges",10)



def temp_anom_to_SMB_grd(ds):
    ds['SMB'] = -1.3*(ds.TAS**2) + 115.4*(ds.TAS) -11.1
    ds['SMB'] = ds['SMB'].sel(YEAR=slice(1980,2100))
    ds['STD'] = ds.SMB.std(dim='MODEL')
    ds['STD'] = ds['STD'].sel(YEAR=slice(1980,2100))
    
    return ds
def temp_anom_to_SMB_shf(ds):
    ds['SMB'] = -12.7*(ds.TAS**2) + 32.1*(ds.TAS) -3.1
    ds['STD'] = ds.SMB.std(dim='MODEL')
    ds['SMB'] = ds['SMB'].sel(YEAR=slice(1980,2100))
    ds['STD'] = ds['STD'].sel(YEAR=slice(1980,2100))
    return ds



def plot_and_fill(ds, axs, color, alpha=0.2, label='_nolegend_', lw=2.5):
    ds.SMB.mean(dim='MODEL').plot(ax=axs, color=color, label=label, lw=lw)
    axs.fill_between(ds.YEAR.values, (ds.SMB.mean(dim='MODEL') + 1.64*ds.STD),
                        (ds.SMB.mean(dim='MODEL') - 1.64*ds.STD),
                        color=color, alpha=alpha)

CM6_ano_SMB_CMPI6= temp_anom_to_SMB_grd(CM6_anom_ts_polar)
CM6_ano_SMB_CMPI5= temp_anom_to_SMB_grd(CM5_anom_ts_polar)





new=CM6_ano_SMB_CMPI6.sel(YEAR=slice(2071,2099))['SMB'].mean(dim="YEAR")

print(new.mean(dim="MODEL").values, new.std(dim="MODEL").values)
new=CM6_ano_SMB_CMPI5.sel(YEAR=slice(2071,2099))['SMB'].mean(dim="YEAR")

print(new.mean(dim="MODEL").values, new.std(dim="MODEL").values)




cumCM6= (CM6_ano_SMB_CMPI6.sel(YEAR=slice(1980,2100))['SMB'].cumsum(dim="YEAR")/360)
print (cumCM6.mean(dim="MODEL").sel(YEAR=2099).values,cumCM6.std(dim="MODEL").sel(YEAR=2099).values)

cumCM5= (CM6_ano_SMB_CMPI5.sel(YEAR=slice(1980,2100))['SMB'].cumsum(dim="YEAR")/360)
print (cumCM5.mean(dim="MODEL").sel(YEAR=2099).values,cumCM5.std(dim="MODEL").sel(YEAR=2099).values)

plot_and_fill(CM6_ano_SMB_CMPI5, ax2[0], pal[9], alpha=0.2,label='CMIP5 - rcp8.5')
plot_and_fill(CM6_ano_SMB_CMPI6, ax2[0], pal_or[9], alpha=0.2, label='CMIP6 - ssp585')

CM6_ano_SMB_CMPI6= temp_anom_to_SMB_shf(CM6_anom_ts_polar)
CM6_ano_SMB_CMPI5= temp_anom_to_SMB_shf(CM5_anom_ts_polar)

new=CM6_ano_SMB_CMPI6.sel(YEAR=slice(2071,2099))['SMB'].mean(dim="YEAR")

print(new.mean(dim="MODEL").values, new.std(dim="MODEL").values)
new=CM6_ano_SMB_CMPI5.sel(YEAR=slice(2071,2099))['SMB'].mean(dim="YEAR")

print(new.mean(dim="MODEL").values, new.std(dim="MODEL").values)


plot_and_fill(CM6_ano_SMB_CMPI5, ax2[1], pal[9], alpha=0.2)
plot_and_fill(CM6_ano_SMB_CMPI6, ax2[1], pal_or[9], alpha=0.2)

ymin=-900
ymax=900

ax2[0].set_ylim([ymin,ymax])
#ax2[0].set_ylim(-200,450)
ax2[0].axhline(color='black', alpha=1,lw=1)
ax2[0].tick_params(axis='both', labelsize=14)


ax2[0].set_xlabel('Year', fontsize=16)
ax2[0].set_ylabel('SMB anomaly (Gt yr$^{-1}$)', fontsize=16)
ax2[0].set_title('Grounded ice', fontsize=18)


ax2[0].text(0.05, 0.9, "A",fontsize=18, transform=ax2[0].transAxes,
       fontdict={'weight': 'bold'})
ax2[0].legend(frameon=False,prop={'size': 12},loc='lower right')
ymin=-900
ymax=900

ax2[1].set_ylim([ymin,ymax])
ax2[1].axhline(color='black', alpha=1,lw=1)

ax2[1].tick_params(axis='both', labelsize=14)
ax2[1].set_xlabel('Year', fontsize=16)
ax2[1].set_ylabel('SMB anomaly (Gt yr$^{-1}$)', fontsize=16)
ax2[1].set_title('Ice shelves', fontsize=18)

ax2[1].text(0.05, 0.9, "B",fontsize=18, transform=ax2[1].transAxes,
       fontdict={'weight': 'bold'})

fig.tight_layout()


fig.savefig('./SMBCMIP5and6v2.png',
            format='PNG', dpi=900)


