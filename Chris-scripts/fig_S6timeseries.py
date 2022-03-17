#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:06:07 2020

@author: ckittel
new fig2: Time series of SMB 
"""

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns


# FUNCTIONS ################################
def SMBcomponents_to_gt(SMB_array, variable, data_start=1980, data_end=2100):
    '''This function returns a daily time series of absolute SMB
    component values, expressed as gigatons.
    35 000 m = MAR resolution
    1e12 factor to convert kg/m² in Gt
    '''
    if (variable == 'SMB' or variable == 'RU' or variable == 'ME'):
        data = SMB_array[variable] * mymask.values
    else:
        data = SMB_array[variable] * mymask.values
    # Make sure only wanted time frame is used
    data = data.loc[str(data_start) + '-01-01':str(data_end) + '-12-31']
    # Convert to gigatons and sum up spatially over the AIS
    sum_spatial = data.sum(dim=['X', 'Y']
                           ) * ((35000 * 35000) / (1e12))

    return sum_spatial

def preprocess(ds):
    ds_new = ds[['SMB']]
    try:
        ds_new= ds_new.sel(SECTOR=1)
    except:
        ds_new=ds_new
    return ds_new

# =============================================================================
# # CREATE the ICE MASK 
# =============================================================================
test = xr.open_dataset('/home/ckittel/Documents/repo/my-awesome-projection/data/MARcst-AN35km-176x148.cdf'
                       ,decode_times=False)
test=test.rename_dims({'x':'X', 'y':'Y'}) #change dim from ferret


ais = test['AIS'].where(test['AIS'] >0) #Only AIS=1, other =0
ice= test['ICE'].where(test['ICE']>30)  #Ice where ICE mask >= 30%
mymask = (ais*ice*test['AREA']/ 100)    #Combine ais + ice/100 * factor area projection



# =============================================================================
# # Open CMIP MODELS
# =============================================================================
dates = pd.date_range('1980-01-01', '2100-01-01', freq='AS') 


# open CMIP5 models
AC3 = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_ACCESS1.3-1980-2100.nc'
                           ,decode_times=False,preprocess=preprocess)

NOR = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_NorESM-1980-2100.nc'
                        ,decode_times=False,preprocess=preprocess)
NOR=NOR.rename_dims({'AT':'TIME'})
                 
     

list_SMB_CMIP5 = [AC3,NOR] 
for ds in list_SMB_CMIP5:
    ds['TIME'] = dates
    
    
# open CMIP6 models    
CNRM = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_CNRM-CM6-1980-2100.nc'
                           ,decode_times=False,preprocess=preprocess)

CESM2 = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_CESM2-1980-2100.nc'
                           ,decode_times=False,preprocess=preprocess)
CESM2=CESM2.rename_dims({'AT':'TIME'})

list_SMB_CMIP6 = [CNRM,CESM2]
for ds in list_SMB_CMIP6:
    ds['TIME'] = dates
    


#
#


## =============================================================================
### open ERA5 time series control
## =============================================================================
dates_present = pd.date_range('1979-01-01', '2019-01-01', freq='AS') 
#   
ER5= xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_ERA5-1979-2019.nc'
                          ,decode_times=False,preprocess=preprocess)
ER5['TIME'] = dates_present
#     


# =============================================================================
# #Convert to Gt/yr
# =============================================================================

#CMI5
SMB_AC3 = SMBcomponents_to_gt(AC3, 'SMB', 1980, 2100)
SMB_AC3_ano= SMB_AC3 - SMB_AC3.sel(TIME=slice('1981-01-01', '2010-01-01')).mean(dim=['TIME'])


SMB_NOR = SMBcomponents_to_gt(NOR, 'SMB', 1980, 2100)
SMB_NOR_ano= SMB_NOR - SMB_NOR.sel(TIME=slice('1981-01-01', '2010-01-01')).mean(dim=['TIME'])

#CMIP6
SMB_CNRM= SMBcomponents_to_gt(CNRM, 'SMB', 1980, 2100)
SMB_CNRM_ano=SMB_CNRM - SMB_CNRM.sel(TIME=slice('1981-01-01', '2010-01-01')).mean(dim=['TIME'])

SMB_CESM2= SMBcomponents_to_gt(CESM2, 'SMB', 1980, 2100)
SMB_CESM2_ano=SMB_CESM2 - SMB_CESM2.sel(TIME=slice('1981-01-01', '2010-01-01')).mean(dim=['TIME'])

#ERA5
SMB_ER5= SMBcomponents_to_gt(ER5, 'SMB', 1980, 2018)


#print ('ER5',SMB_ER5.sel(TIME=slice('1980-01-01', '2009-01-01')).mean(dim=['TIME']).values)
print ('AC3',SMB_AC3.sel(TIME=slice('1980-01-01', '2009-01-01')).mean(dim=['TIME']).values)
print ('NOR',SMB_NOR.sel(TIME=slice('1980-01-01', '2009-01-01')).mean(dim=['TIME']).values)
print ('CNRM',SMB_CNRM.sel(TIME=slice('1980-01-01', '2009-01-01')).mean(dim=['TIME']).values)
print ('CESM2',SMB_CESM2.sel(TIME=slice('1980-01-01', '2009-01-01')).mean(dim=['TIME']).values)
# =============================================================================
##Figures
# =============================================================================

pal = sns.color_palette("PuBu",10)
pal_or = sns.color_palette("OrRd",10)

lw2=1.2
lws=1.2
lwa=0.2
#1. Time series with absolute values
fig, ax2= plt.subplots(nrows=1,ncols=1,figsize=(14,6))

SMB_AC3.plot(ax=ax2,color=pal[7],lw=lwa)
SMB_NOR.plot(ax=ax2,color=pal[4],lw=lwa)
SMB_CNRM.plot(ax=ax2,color=pal_or[7],lw=lwa)
SMB_CESM2.plot(ax=ax2,color=pal_or[4],lw=lwa)



#CMIP5
SMB_AC3.rolling(TIME=5, center=True).mean().plot(ax=ax2,color=pal[7], label="MAR(ACCESS1.3)",lw=lws)
SMB_NOR.rolling(TIME=5, center=True).mean().plot(ax=ax2,color=pal[4], label="MAR(NorESM1-M)",lw=lws)

#CMIP6
SMB_CNRM.rolling(TIME=5, center=True).mean().plot(ax=ax2,color=pal_or[7], label="MAR(CNRM-CM6-1)",lw=lws)
SMB_CESM2.rolling(TIME=5, center=True).mean().plot(ax=ax2,color=pal_or[4], label="MAR(CESM2)",lw=lws)

#ERA5
SMB_ER5.plot(ax=ax2,color='limegreen',lw=0.2)
SMB_ER5.rolling(TIME=5, center=True).mean().plot(ax=ax2,color='limegreen', label='ERA5',lw=lw2)


#Customize plots
ax2.set_xlabel('Year', fontsize=12)
ax2.minorticks_off()
ax2.set_ylabel('Annual AIS SMB (Gt yr$^{-1}$)', fontsize=12)
ax2.legend(frameon=False, ncol=2, prop={'size': 10})
ax2.tick_params(axis="x", labelsize=10)
ax2.tick_params(axis="y", labelsize=10)

ax2.legend(frameon=False, fontsize=10)

ax2.get_figure().gca().set_title("") #remove title sector=1


fig.savefig('./SMB_fig2_abso.png',
            format='PNG', dpi=600)

plt.show()
plt.cla()
plt.clf()


#1bis. Time series with absolute values
fig, ax2= plt.subplots(nrows=1,ncols=1,figsize=(14,6))
lw2=1.5
lws=0.2


SMB_AC3.plot(ax=ax2,color=pal[7],lw=lws)
SMB_NOR.plot(ax=ax2,color=pal[4],lw=lws)
SMB_CNRM.plot(ax=ax2,color=pal_or[7],lw=lws)
SMB_CESM2.plot(ax=ax2,color=pal_or[4],lw=lws)


#CMIP5
SMB_AC3.rolling(TIME=5, center=True).mean().plot(ax=ax2,color=pal[7], label="MAR(ACCESS1.3)",lw=lw2)
SMB_NOR.rolling(TIME=5, center=True).mean().plot(ax=ax2,color=pal[4], label="MAR(NorESM1-M)",lw=lw2)

#CMIP6
SMB_CNRM.rolling(TIME=5, center=True).mean().plot(ax=ax2,color=pal_or[7], label="MAR(CNRM-CM6-1)",lw=lw2)
SMB_CESM2.rolling(TIME=5, center=True).mean().plot(ax=ax2,color=pal_or[4], label="MAR(CESM2)",lw=lw2)



#Customize plots
ax2.set_xlabel('Year', fontsize=12)
ax2.minorticks_off()
ax2.set_ylabel('Annual SMB (Gt yr$^{-1}$)', fontsize=12)
ax2.legend(frameon=False, ncol=2, prop={'size': 10})
ax2.tick_params(axis="x", labelsize=10)
ax2.tick_params(axis="y", labelsize=10)

ax2.legend(frameon=False, fontsize=10)

ax2.get_figure().gca().set_title("") #remove title sector=1


fig.savefig('./SMB_ts_abso.png',
            format='PNG', dpi=900)

plt.show()
plt.cla()
plt.clf()



#2. Time series with anomalies
fig, ax2 = plt.subplots(nrows=1,ncols=1,figsize=(14,6))
lw2=2
lws=0.4
#all val
SMB_AC3_ano.plot(ax=ax2,color=pal[7],lw=lws)

SMB_NOR_ano.plot(ax=ax2,color=pal[6],lw=lws)
SMB_CNRM_ano.plot(ax=ax2,color=pal_or[7],lw=lws)
SMB_CESM2_ano.plot(ax=ax2,color=pal_or[4],lw=lws)

#rolling mean
SMB_AC3_ano.rolling(TIME=5, center=True).mean().plot(ax=ax2,color=pal[7], label="MAR(ACCESS1.3)",lw=lw2)
SMB_NOR_ano.rolling(TIME=5, center=True).mean().plot(ax=ax2,color=pal[4], label="MAR(NorESM1-M)",lw=lw2)
SMB_CNRM_ano.rolling(TIME=5, center=True).mean().plot(ax=ax2,color=pal_or[7], label="MAR(CNRM-CM6-1)",lw=lw2)
SMB_CESM2_ano.rolling(TIME=5, center=True).mean().plot(ax=ax2,color=pal_or[4], label="MAR(CESM2)",lw=lw2)


#Customize plots
ax2.set_xlabel('Year', fontsize=16)
ax2.minorticks_off()
ax2.set_ylabel('SMB anomaly (Gt yr$^{-1}$)', fontsize=16)
ax2.legend(frameon=False, ncol=2, prop={'size': 12})
ax2.tick_params(axis="x", labelsize=14)
ax2.tick_params(axis="y", labelsize=14)
ax2.legend(frameon=False, fontsize=12)

ax2.axhline(color='black', alpha=1,lw=1)

ax2.get_figure().gca().set_title("")
##Customize plots
fig.savefig('./SMB_fig2_tmp.png',
            format='PNG', dpi=900)
plt.show()
plt.cla()
plt.clf()
#3. Time series with cum anomalies
fig, ax2 = plt.subplots(nrows=1,ncols=1,figsize=(14,6))

#computing cumsummed
SMB_AC3_ano.cumsum().plot(ax=ax2,color=pal[7], label="MAR(ACCESS1.3)",lw=lw2)
SMB_NOR_ano.cumsum().plot(ax=ax2,color=pal[4], label="MAR(NorESM1-M)",lw=lw2)
SMB_CNRM_ano.cumsum().plot(ax=ax2,color=pal_or[7], label="MAR(CNRM-CM6-1)",lw=lw2)
SMB_CESM2_ano.cumsum().plot(ax=ax2,color=pal_or[4], label="MAR(CESM2)",lw=lw2)

#Customize plots
ax2.set_xlabel('Year', fontsize=12)
ax2.minorticks_off()
ax2.set_ylabel('Cumulated SMB anomaly (Gt)', fontsize=12)
ax2.legend(frameon=False, ncol=2, prop={'size': 10})
ax2.tick_params(axis="x", labelsize=10)
ax2.tick_params(axis="y", labelsize=10)
ax2.legend(frameon=False, fontsize=10)

ax2.get_figure().gca().set_title("")
##Customize plots
fig.savefig('./SMB_fig2_cumsum.png',
           format='PNG', dpi=900)