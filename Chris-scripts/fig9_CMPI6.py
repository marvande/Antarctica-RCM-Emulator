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

CM6_ssp126 = xr.open_dataset('/home/ckittel/Documents/data/proj/GCM-CM6-ssp126.nc',
                      decode_times=False)
CM6_ssp126_new = CM6_ssp126.rename({'TIME':'MODEL', 'ZZ':'YEAR','TAS_ANN':'TAS'})

CM6_ssp245 = xr.open_dataset('/home/ckittel/Documents/data/proj/GCM-CM6-ssp245.nc',
                      decode_times=False)
CM6_ssp245_new = CM6_ssp245.rename({'TIME':'MODEL', 'ZZ':'YEAR','TAS_ANN':'TAS'})


CM6_ssp585 = xr.open_dataset('/home/ckittel/Documents/data/proj/GCM-CM6_test_all.nc',
                      decode_times=False)
CM6_ssp585_new = CM6_ssp585.rename({'TIME':'MODEL', 'ZZ':'YEAR','TAS_ANN':'TAS'})


#CM6_histo=CM6_ssp585_new.sel(YEAR=slice(1960,2014))
#datasets = []
##for example in CM6_histo,CM6_ssp126_new:
##    ds = xr.create_an_xarray_dataset(example)
#    
#ds=CM6_histo
#datasets.append(ds)
#ds=CM6_ssp126_new
#
#datasets.append(ds)
#
#    
#CM6_ssp126_new = xr.concat(datasets, dim='YEAR')






ERA5= xr.open_dataset('/home/ckittel/Documents/data/proj/ERA5_tasy.nc',
                      decode_times=False)

numdates = ERA5['TIME'].data
dates = [num2date(d,units='month since 1950-01-15 00:00:00',calendar='360_day').year for d in numdates]
ERA5['TIME'] = dates
#npdates =  pd.date_range('1980', '2017', freq='AS').values

#ERA5['TIME'] = CM6_ssp585_new['YEAR'].sel(YEAR=slice(1980,2017)).data


# =======================================================================================
# AX2 ANALYSIS
#CM6_ssp126_base_ts = CM6_ssp585_new.sel(YEAR=slice(1980,2009)).mean(dim=['YEAR','LAT','LON'])
#CM6_ssp245_base_ts = CM6_ssp585_new.sel(YEAR=slice(1980,2009)).mean(dim=['YEAR','LAT','LON'])
#CM6_ssp585_base_ts = CM6_ssp585_new.sel(YEAR=slice(1980,2009)).mean(dim=['YEAR','LAT','LON'])
#
#CM6_ssp126_anom_ts = CM6_ssp126_new.mean(dim=['MODEL','LAT','LON']) - CM6_ssp126_base_ts
#CM6_ssp245_anom_ts = CM6_ssp126_new.mean(dim=['MODEL','LAT','LON']) - CM6_ssp245_base_ts
#CM6_ssp585_anom_ts = CM6_ssp585_new.mean(dim=['MODEL','LAT','LON']) - CM6_ssp585_base_ts


CM6_ssp126_base_ts2 = CM6_ssp585_new.sel(YEAR=slice(1986,2005)).mean(dim=['YEAR','LAT','LON'])
CM6_ssp245_base_ts2 = CM6_ssp585_new.sel(YEAR=slice(1986,2005)).mean(dim=['YEAR','LAT','LON'])
CM6_ssp585_base_ts2 = CM6_ssp585_new.sel(YEAR=slice(1986,2005)).mean(dim=['YEAR','LAT','LON'])

CM6_ssp126_anom_ts2 = CM6_ssp126_new.mean(dim=['LAT','LON']) - CM6_ssp126_base_ts2
CM6_ssp245_anom_ts2 = CM6_ssp245_new.mean(dim=['LAT','LON']) - CM6_ssp245_base_ts2
CM6_ssp585_anom_ts2 = CM6_ssp585_new.mean(dim=['LAT','LON']) - CM6_ssp585_base_ts2

CM6_ssp126_anom_ts2.sel(YEAR=slice(2081,2100)).mean(dim='YEAR').mean(dim='MODEL').values
CM6_ssp126_anom_ts2.sel(YEAR=slice(2081,2100)).mean(dim='YEAR').std(dim='MODEL').values

CM6_ssp245_anom_ts2.sel(YEAR=slice(2080,2099)).mean(dim='YEAR').mean(dim='MODEL').values
CM6_ssp245_anom_ts2.sel(YEAR=slice(2080,2099)).mean(dim='YEAR').std(dim='MODEL').values

CM6_ssp585_anom_ts2.sel(YEAR=slice(2081,2100)).mean(dim='YEAR').mean(dim='MODEL').values
CM6_ssp585_anom_ts2.sel(YEAR=slice(2081,2100)).mean(dim='YEAR').std(dim='MODEL').values


CM6_ssp126_base_ts_polar = CM6_ssp585_new.sel(YEAR=slice(1980,2009),LAT=slice(-90,-60)).mean(dim=['YEAR','LAT','LON'])
CM6_ssp245_base_ts_polar = CM6_ssp585_new.sel(YEAR=slice(1980,2009),LAT=slice(-90,-60)).mean(dim=['YEAR','LAT','LON'])
CM6_ssp585_base_ts_polar = CM6_ssp585_new.sel(YEAR=slice(1980,2009),LAT=slice(-90,-60)).mean(dim=['YEAR','LAT','LON'])

CM6_ssp126_anom_ts_polar = CM6_ssp126_new.sel(LAT=slice(-90,-60)).mean(dim=['LAT','LON']) - CM6_ssp126_base_ts_polar
CM6_ssp245_anom_ts_polar = CM6_ssp245_new.sel(LAT=slice(-90,-60)).mean(dim=['LAT','LON']) - CM6_ssp245_base_ts_polar
CM6_ssp585_anom_ts_polar = CM6_ssp585_new.sel(LAT=slice(-90,-60)).mean(dim=['LAT','LON']) - CM6_ssp585_base_ts_polar

ERA5_anom_ts_polar = ERA5.sel(LAT=slice(-90,-60)).mean(dim=['LAT','LON'])#-273.15 #




def ru2(x, a, b,c):
    return (a*x*x)+ (b * x) + c
def sf1(x, a, b):
    return (a * x) + b

# PLOTTING ###########################
plt.close('all')
#proj = ccrs.PlateCarree(central_longitude=0.0, globe=None)
fig, ax2 = plt.subplots(nrows=1,ncols=2,figsize=(14,6))
# ============ PLOT ON AX2 ============
pal   = sns.color_palette("Blues",10)
pal_or = sns.color_palette("Oranges",10)
pal_gr =sns.color_palette("Greens",10)



def temp_anom_to_SMB_grd(ds):
    ds['SMB'] = -1.3*(ds.TAS**2) + 115.4*(ds.TAS) -11.1
    ds['SMB'] = ds['SMB'].sel(YEAR=slice(1980,2100))
    ds['STD'] = ds.SMB.std(dim='MODEL')
    ds['STD'] = ds['STD'].sel(YEAR=slice(1980,2100))
    
    return ds
def temp_anom_to_SMB_shf(ds):
#    ds['SMB'] = -14.1*(ds.TAS**2) + 30.6*(ds.TAS) -1.4 PP_RU
    ds['SMB'] = -12.7*(ds.TAS**2) + 32.1*(ds.TAS) -3
    ds['STD'] = ds.SMB.std(dim='MODEL')
    ds['SMB'] = ds['SMB'].sel(YEAR=slice(1980,2100))
    ds['STD'] = ds['STD'].sel(YEAR=slice(1980,2100))
    return ds




def plot_and_fill(ds, axs, color, alpha=0.2, label='_nolegend_', lw=2.5):
    ds.SMB.mean(dim='MODEL').plot(ax=axs, color=color, label=label, lw=lw)
    axs.fill_between(ds.YEAR.values, (ds.SMB.mean(dim='MODEL') + 1.64*ds.STD),
                        (ds.SMB.mean(dim='MODEL') - 1.64*ds.STD),
                        color=color, alpha=alpha)

CM6_ano_SMB_CMPI6= temp_anom_to_SMB_grd(CM6_ssp585_anom_ts_polar)
CM6_ano_SMB_CMPI62=CM6_ano_SMB_CMPI6.sel(YEAR=slice(2015,2100)) 
CM6_ano_SMB_CMPI6_126= temp_anom_to_SMB_grd(CM6_ssp126_anom_ts_polar)
CM6_ano_SMB_CMPI6_245= temp_anom_to_SMB_grd(CM6_ssp245_anom_ts_polar)

CM6_ano_SMB_CMPI6_histo=CM6_ano_SMB_CMPI6.sel(YEAR=slice(1980,2015)) 

tocon=[CM6_ano_SMB_CMPI6_histo,CM6_ano_SMB_CMPI6_126]
long126=xr.concat(tocon,dim='YEAR')

tocon=[CM6_ano_SMB_CMPI6_histo,CM6_ano_SMB_CMPI6_245]
long245=xr.concat(tocon,dim='YEAR')



long126_SMB= temp_anom_to_SMB_grd(long126)
long245_SMB= temp_anom_to_SMB_grd(long245)

new=long126_SMB.sel(YEAR=slice(2071,2099))['SMB'].mean(dim="YEAR")

print(new.mean(dim="MODEL").values, new.std(dim="MODEL").values)
new=long245_SMB.sel(YEAR=slice(2071,2099))['SMB'].mean(dim="YEAR")

print(new.mean(dim="MODEL").values, new.std(dim="MODEL").values)


cum126= (long126_SMB.sel(YEAR=slice(1980,2100))['SMB'].cumsum(dim="YEAR")/360)
print (cum126.mean(dim="MODEL").sel(YEAR=2099).values,cum126.std(dim="MODEL").sel(YEAR=2099).values)

print(long245_SMB.sel(YEAR=slice(2071,2100))['SMB'].mean(dim="MODEL").values,long245_SMB.sel(YEAR=slice(2071,2100))['SMB'].std(dim="MODEL").values)
cum245= (long245_SMB.sel(YEAR=slice(1980,2100))['SMB'].cumsum(dim="YEAR")/360)
print (cum245.mean(dim="MODEL").sel(YEAR=2099).values,cum245.std(dim="MODEL").sel(YEAR=2099).values)


plot_and_fill(CM6_ano_SMB_CMPI6_126, ax2[0], pal[9], alpha=0.15,label='CMIP6 - ssp126')
plot_and_fill(CM6_ano_SMB_CMPI6_245, ax2[0], pal_gr[9], alpha=0.15,label='CMIP6 - ssp245')
plot_and_fill(CM6_ano_SMB_CMPI62, ax2[0], pal_or[9], alpha=0.15, label='CMIP6 - ssp585')
plot_and_fill(CM6_ano_SMB_CMPI6_histo, ax2[0], 'black', alpha=0.15)

CM6_ano_SMB_CMPI6= temp_anom_to_SMB_shf(CM6_ssp585_anom_ts_polar)
CM6_ano_SMB_CMPI62=CM6_ano_SMB_CMPI6.sel(YEAR=slice(2015,2100)) 
CM6_ano_SMB_CMPI6_126= temp_anom_to_SMB_shf(CM6_ssp126_anom_ts_polar)
CM6_ano_SMB_CMPI6_245= temp_anom_to_SMB_shf(CM6_ssp245_anom_ts_polar)
CM6_ano_SMB_CMPI6_histo=CM6_ano_SMB_CMPI6.sel(YEAR=slice(1980,2015)) 

plot_and_fill(CM6_ano_SMB_CMPI6_126, ax2[1], pal[9], alpha=0.15)
plot_and_fill(CM6_ano_SMB_CMPI6_245, ax2[1], pal_gr[9], alpha=0.15)
plot_and_fill(CM6_ano_SMB_CMPI62, ax2[1], pal_or[9], alpha=0.15)
plot_and_fill(CM6_ano_SMB_CMPI6_histo, ax2[1], 'black', alpha=0.15)


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


fig.savefig('./SMBCMIP6sspv2.png',
            format='PNG', dpi=900)


#
##Grouded coef
#asf=144.9775984757646
#bsf=-29.766894984760455
#
#aru=6.495452019801952
#bru=-0.0467806093036059
#cru=0.9313132086099879
#
#
#
#
#for i in range (0,33):
#    
#    model=CM6_ssp126_anom_ts_polar['TAS'].isel(MODEL=i).sel(YEAR=slice(2015,2100))
#    model_ano= sf1(model,asf,bsf)-ru2(model,aru,bru, cru)
#    model_ano.plot(ax=ax2[0], color=pal[9], lw=0.1)
#    model=CM6_ssp585_anom_ts_polar['TAS'].isel(MODEL=i).sel(YEAR=slice(1980,2014))
#    model_ano= sf1(model,asf,bsf)-ru2(model,aru,bru, cru)
#    model_ano.plot(ax=ax2[0], color=pal[9], lw=0.1)  
#
#
#for i in range (0,33):
#    
#    model=CM6_ssp585_anom_ts_polar['TAS'].isel(MODEL=i).sel(YEAR=slice(1980,2100))
#    model_ano= sf1(model,asf,bsf)-ru2(model,aru,bru, cru)
#
#
#    model_ano.plot(ax=ax2[0], color=pal_or[9], lw=0.1)
#    
#    
#    
#model=CM6_ssp126_polar_anom_mean['TAS'].sel(YEAR=slice(1980,2100))
#model_ano=  sf1(model,asf,bsf)-ru2(model,aru,bru, cru)
#model_ano.plot(ax=ax2[0], color=pal[9], lw=1.5,label="CMIP5-rcp8.5")
##
##
#model=CM6_ssp585_polar_anom_mean['TAS'].sel(YEAR=slice(1980,2100))
#model_ano=  sf1(model,asf,bsf)-ru2(model,aru,bru, cru)
#model_ano.plot(ax=ax2[0], color=pal_or[9], lw=1.5,label="CMIP6-ssp858")
#
#ax2[0].legend(frameon=False,prop={'size': 12})
#
#ax2[0].set_xlabel('Year', fontsize=14)
#ax2[0].set_ylabel('SMB anomaly (Gt yr$^{-1}$)', fontsize=14)
#ax2[0].set_title('Grounded ice', fontsize=18)
#
#ymin=-350
#ymax=750
#
##ax2[0].set_ylim([ymin,ymax])
##ax2[0].set_ylim(-200,450)
#ax2[0].axhline(color='black', alpha=1,lw=1)
#ax2[0].tick_params(axis='both', labelsize=11)
#
##shelves coef
#asf=35.289646069633164
#bsf=-2.9479631284055436
#
#aru=14.055040085683629
#bru=-4.741013603926703
#cru=4.342599756823334
#
#
#
#
#for i in range (0,33):
#    
#    model=CM6_ssp126_anom_ts_polar['TAS'].isel(MODEL=i).sel(YEAR=slice(1980,2100))
#    model_ano= sf1(model,asf,bsf)-ru2(model,aru,bru, cru)
#    model_ano.plot(ax=ax2[1], color=pal[9], lw=0.1)  
#
#
#for i in range (0,33):
#    
#    model=CM6_ssp585_anom_ts_polar['TAS'].isel(MODEL=i).sel(YEAR=slice(1980,2100))
#    model_ano= sf1(model,asf,bsf)-ru2(model,aru,bru, cru)
#
#
#    model_ano.plot(ax=ax2[1], color=pal_or[9], lw=0.1)
#    
#    
#    
#model=CM6_ssp126_polar_anom_mean['TAS'].sel(YEAR=slice(1980,2100))
#model_ano=  sf1(model,asf,bsf)-ru2(model,aru,bru, cru)
#model_ano.plot(ax=ax2[1], color=pal[9], lw=1.5,label="CMIP5-rcp8.5")
##
##
#model=CM6_ssp585_polar_anom_mean['TAS'].sel(YEAR=slice(1980,2100))
#model_ano=  sf1(model,asf,bsf)-ru2(model,aru,bru, cru)
#model_ano.plot(ax=ax2[1], color=pal_or[9], lw=1.5,label="CMIP6-ssp858")
#
#
#ax2[1].set_xlabel('Year', fontsize=14)
#ax2[1].set_ylabel('SMB anomaly (Gt yr$^{-1}$)', fontsize=14)
#ax2[1].set_title('Ice shelves', fontsize=18)
#
#ymin=350
#ymax=-750
#
#ax2[1].set_ylim([ymin,ymax])
#ax2[1].axhline(color='black', alpha=1,lw=1)
#
#ax2[1].tick_params(axis='both', labelsize=11)
#
##sns.despine()
#fig.tight_layout()
#
#
#fig.savefig('./SMBCMIP6ssp.png',
#            format='PNG', dpi=500)
