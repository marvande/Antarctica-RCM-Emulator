#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:55:02 2019

@author: ckittel
ANO SMB vs ANO warming
"""

import xarray as xr
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from netCDF4 import num2date


import function_plot_fig6_7andSup


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  #WARNING WARNINGS ARE IGNORED

def polyfit(x, y, degree):
    """This function computes the determination coefficient between any
    curve fitted with.
    """
    results = {}

    coeffs = np.polyfit(x, y, degree)
    curve = np.polyval(coeffs, x)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssres = np.sum((y - curve)**2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = 1 - ssres / sstot
    return results, curve

def preprocess(ds):
    '''
    Avoid reading/opening the whole ds. => Selection of interesting variable
    Also try to remove SECTOR dimension 
    (Sector=1 corresponds to the ice sheet in MAR (2= tundra or rocks))
    '''
    ds_new = ds[['ME', 'RU', 'RF', 'SF', 'SU','SMB','TT']]
    try:
        ds_new= ds_new.sel(SECTOR=1)
    except:
        ds_new=ds_new
    return ds_new


def preprocess_GCM(ds):
    '''

    '''
    ds_new = ds[['TAS']]
    return ds_new



def SMBcomponents_to_gt(SMB_array, variable,mask, data_start=1980, data_end=2100):
    '''This function returns a daily time series of absolute SMB
    component values, expressed as gigatons.
    35 000 m = MAR resolution
    1e12 factor to convert kg/m² in Gt
    '''
    
    data = SMB_array[variable] * mask.values
    # Make sure only wanted time frame is used
    data = data.loc[str(data_start) + '-01-01':str(data_end) + '-12-31']
    # Convert to gigatons and sum up spatially over the AIS
    sum_spatial = data.sum(dim=['X', 'Y']
                           ) * ((35000 * 35000) / (1e12))

    return sum_spatial


def annual_sum(data):
    '''This function returns the annual sum
    '''
    annual_sum = data.groupby('TIME.year').sum(dim='TIME')
    return annual_sum


def compute_int(list_SMB, variables, list_names,msk, data_start, data_end,season='annual'):
    #df = pd.DataFrame()
    df=list_SMB.copy()
    for var in variables:
        # Extact the radiation time_series as Gt
        SMB_ts = SMBcomponents_to_gt(
            list_SMB, var, msk, data_start, data_end)

#            else:
            # Get the sum for season of each year
#                SMB_JJA_ts = seasonal_sum(SMB_ts, season)
        df[var] = SMB_ts
        #df[var].values
    try:
     df['TT']=list_SMB['TT'].sel(ATMLAY='0.99973').mean(dim=['X','Y'])
    except:
        print()
    #df['TIME'] = pd.date_range(str(data_start)+'-01-01', str(data_end)+'-01-01', freq='AS') #pd.to_dateTIME(df[['year', 'month', 'day']])
    #df = df.set_index('TIME')
    return df
        
def compute_ano(df,refs,refe,period):
    df=df.sel(TIME=slice(str(refs)+'-01-01','2100-01-01'))
    dfo= df - df.sel(TIME=slice(str(refs)+'-01-01',str(refe)+'-01-01')).mean(dim=['TIME'])
    dfor= dfo.rolling(TIME=period, center=True).mean()
    return dfor

# =============================================================================
# 
# START PROG
# 
# =============================================================================





# =============================================================================
# OPEN CMIP5 AND CMIP6
# =============================================================================
def ano_gcm(GCM,refs,refe):
    GCM_base_ts_polar = GCM.sel(TIME=slice(str(refs)+'-01-01',str(refe)+'-01-01'),LAT=slice(-90,-60)).mean(dim=['TIME','LAT','LON'])
    GCM_anom_ts_polar = GCM.sel(LAT=slice(-90,-60)).mean(dim=['LAT','LON']) - GCM_base_ts_polar
    return GCM_anom_ts_polar




#Annual
workdir='/home/ckittel/Documents/data/proj/'
AC3 =  xr.open_mfdataset(workdir+'year_ACCESS1-3.nc2'
                       , decode_times=False,preprocess=preprocess_GCM)
AC3 = AC3.rename({'TIME2':'TIME'})
NOR = xr.open_mfdataset(workdir+'year_NorESM1-M.nc2'
                       , decode_times=False,preprocess=preprocess_GCM)
NOR = NOR.rename({'TIME2':'TIME'})
CNR = xr.open_mfdataset(workdir+'year_CNRM-CM6-1.nc2'
                       , decode_times=False,preprocess=preprocess_GCM)
CNR = CNR.rename({'TIME2':'TIME'})
CSM = xr.open_mfdataset(workdir+'year_CESM2.nc2'
                       , decode_times=False,preprocess=preprocess_GCM)
CSM = CSM.rename({'TIME2':'TIME'})

dates=  pd.date_range('1950-01-01', '2100-01-01', freq='AS')


list_name_CMIP5= ['AC3','NOR']
list_SMB_CMIP5 = [AC3,NOR] 
for ds in list_SMB_CMIP5:
    ds['TIME'] = dates
    
list_name_CMIP6= ['CNR','CSM']
list_SMB_CMIP6 = [CNR, CSM]
for ds in list_SMB_CMIP6:
    ds['TIME'] = dates
    
list_mod = []
list_mod_names = []     

for i in range(len(list_SMB_CMIP5)):
    list_mod.append(list_SMB_CMIP5[i])
    list_mod_names.append(list_name_CMIP5[i])

for i in range(len(list_SMB_CMIP6)):
    list_mod.append(list_SMB_CMIP6[i])
    list_mod_names.append(list_name_CMIP6[i])
    
    
result_tty_dic = {}

for i in range(len(list_mod)):
    print (list_mod_names[i])
    dfi=ano_gcm(list_mod[i],1980,2009)   
    result_tty_dic[list_mod_names[i]]=dfi

#DJA
workdir='/home/ckittel/Documents/data/proj/'
AC3 =  xr.open_mfdataset(workdir+'DJF-ACCESS1-3.nc2'
                       , decode_times=False,preprocess=preprocess_GCM)
AC3 = AC3.rename({'TIME2':'TIME'})
NOR = xr.open_mfdataset(workdir+'DJF-NorESM1-M.nc2'
                       , decode_times=False,preprocess=preprocess_GCM)
NOR = NOR.rename({'TIME2':'TIME'})
CNR = xr.open_mfdataset(workdir+'DJF-CNRM-CM6-1.nc2'
                       , decode_times=False,preprocess=preprocess_GCM)
CNR = CNR.rename({'TIME2':'TIME'})
CSM = xr.open_mfdataset(workdir+'DJF-CESM2.nc2'
                       , decode_times=False,preprocess=preprocess_GCM)
CSM = CSM.rename({'TIME2':'TIME'})

dates=  pd.date_range('1981-01-01', '2100-01-01', freq='AS')   

list_name_CMIP5= ['AC3','NOR']
list_SMB_CMIP5 = [AC3,NOR] 
for ds in list_SMB_CMIP5:
    ds['TIME'] = dates
    
list_name_CMIP6= ['CNR','CSM']
list_SMB_CMIP6 = [CNR, CSM]
for ds in list_SMB_CMIP6:
    ds['TIME'] = dates
    
list_mod = []
list_mod_names = []     

for i in range(len(list_SMB_CMIP5)):
    list_mod.append(list_SMB_CMIP5[i])
    list_mod_names.append(list_name_CMIP5[i])

for i in range(len(list_SMB_CMIP6)):
    list_mod.append(list_SMB_CMIP6[i])
    list_mod_names.append(list_name_CMIP6[i])
    
    
result_tts_dic = {}

for i in range(len(list_mod)):
    print (list_mod_names[i])
    dfi=ano_gcm(list_mod[i],1981,2009)   
    result_tts_dic[list_mod_names[i]]=dfi

result_tt_dic = {}
result_tt_dic['year'] = result_tty_dic
result_tt_dic['DJF']  = result_tts_dic

#dic des dics





 #CREATE the ICE MASK 
test = xr.open_dataset('data/MARcst-AN35km-176x148.cdf'
                       ,decode_times=False)
test=test.rename_dims({'x':'X', 'y':'Y'}) #change dim from ferret


ais = test['AIS'].where(test['AIS'] >0) #Only AIS=1, other islands  =0
ice= test['ICE'].where(test['ICE']>30)  #Ice where ICE mask >= 30% (ICE[0-100%], dividing by 100 in the next ligne)
ice_msk = (ais*ice*test['AREA']/ 100)    #Combine ais + ice/100 * factor area for taking into account the projection

grd=test['GROUND'].where(test['GROUND']>30)
grd_msk = (ais*grd*test['AREA']/ 100)


#OPEN the SF GCM interpolated file
dates = pd.date_range('1960-01-01', '2100-01-01', freq='AS') 
AC3 = xr.open_dataset('/home/ckittel/Documents/data/PSRWN/extract-MAR_universel/ACCES_PRECIP.nc'
                      ,decode_times=False)
NOR =xr.open_dataset('/home/ckittel/Documents/data/PSRWN/extract-MAR_universel/NorESM1-M.nc'
                      ,decode_times=False)
list_name_CMIP5= ['AC3','NOR']
list_SMB_CMIP5 = [AC3,NOR] 
for ds in list_SMB_CMIP5:
    ds['TIME'] = dates
    
CNR =xr.open_dataset('/home/ckittel/Documents/data/PSRWN/extract-MAR_universel/CNRM_PRECIP.nc'
                      ,decode_times=False)
CSM =xr.open_dataset('/home/ckittel/Documents/data/PSRWN/extract-MAR_universel/CESM2_PRECIP.nc'
                      ,decode_times=False)        

list_name_CMIP6= ['CNR','CSM']
list_SMB_CMIP6 = [CNR, CSM]
for ds in list_SMB_CMIP6:
    ds['TIME'] = dates
    


vars_SMB = ['PRSN']   
list_mod = []
list_mod_names = [] 

for i in range(len(list_SMB_CMIP5)):
    list_mod.append(list_SMB_CMIP5[i])
    list_mod_names.append(list_name_CMIP5[i])

for i in range(len(list_SMB_CMIP6)):
    list_mod.append(list_SMB_CMIP6[i])
    list_mod_names.append(list_name_CMIP6[i])



gcm_ice_dic = {}
gcm_grd_dic = {}
gcm_shf_dic = {}

for i in range(len(list_mod)):
    print (list_mod_names[i])
    #ALL AIS
    dfi=compute_int(list_mod[i], vars_SMB, list_mod_names[i],ice_msk, 1981, 2100)
    gcm_ice_dic[list_mod_names[i]]=dfi

    #Grounded only
    dfg=compute_int(list_mod[i], vars_SMB, list_mod_names[i],grd_msk, 1981, 2100)
    gcm_grd_dic[list_mod_names[i]]=dfg

    #Ice shelves
    dfs=dfi-dfg
    gcm_shf_dic[list_mod_names[i]]=dfs

   
gcm_dic_year = {}
gcm_dic_year['ice']=gcm_ice_dic
gcm_dic_year['grd']=gcm_grd_dic
gcm_dic_year['shelf']=gcm_shf_dic


 
# =============================================================================
# #OPEN MAR 
# =============================================================================

###YEAR
dates = pd.date_range('1980-01-01', '2100-01-01', freq='AS') 
# open CMIP5 MAR results
AC3 = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_ACCESS1.3-1980-2100.nc'
                           ,decode_times=False,preprocess=preprocess)

NOR = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_NorESM-1980-2100.nc'
                        ,decode_times=False,preprocess=preprocess) #NOR=   #Warning 2100= 2099

NOR=NOR.rename_dims({'AT':'TIME'})

list_name_CMIP5= ['AC3','NOR']
list_SMB_CMIP5 = [AC3,NOR] 
for ds in list_SMB_CMIP5:
    ds['TIME'] = dates
        
# open CMIP6 MAR results   
CNR = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_CNRM-CM6-1980-2100.nc'
                           ,decode_times=False,preprocess=preprocess)

CSM = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_CESM2-1980-2100.nc'
                           ,decode_times=False,preprocess=preprocess)
CSM=CSM.rename_dims({'AT':'TIME'})

list_name_CMIP6= ['CNR','CSM']
list_SMB_CMIP6 = [CNR, CSM]
for ds in list_SMB_CMIP6:
    ds['TIME'] = dates
    
 

# =============================================================================
# # COMPUTE INTEGRATED VALUES + (rolling) ANOMALIES
# =============================================================================

vars_SMB = ['SMB', 'ME','SF','RU','RF','SU']   
list_mod = []
list_mod_names = [] 

for i in range(len(list_SMB_CMIP5)):
    list_mod.append(list_SMB_CMIP5[i])
    list_mod_names.append(list_name_CMIP5[i])

for i in range(len(list_SMB_CMIP6)):
    list_mod.append(list_SMB_CMIP6[i])
    list_mod_names.append(list_name_CMIP6[i])



result_ice_dic = {}
result_grd_dic = {}
result_shf_dic = {}


for i in range(len(list_mod)):
    print (list_mod_names[i])
    #ALL AIS
    dfi=compute_int(list_mod[i], vars_SMB, list_mod_names[i],ice_msk, 1980, 2100)
    result_ice_dic[list_mod_names[i]]=dfi

    #Grounded only
    dfg=compute_int(list_mod[i], vars_SMB, list_mod_names[i],grd_msk, 1980, 2100)
    result_grd_dic[list_mod_names[i]]=dfg

    #Ice shelves
    dfs=dfi-dfg
    result_shf_dic[list_mod_names[i]]=dfs

# # OUT STORED IN DIC (key= reg => ice, grd, shelf)
 
    
result_dic_year = {}
result_dic_year['ice']=result_ice_dic
result_dic_year['grd']=result_grd_dic
result_dic_year['shelf']=result_shf_dic




#Same WITH DJF
dates = pd.date_range('1981-01-01', '2100-01-01', freq='AS') 
# open CMIP5 MAR results
AC3 = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/DJF-MAR_ACCESS1.3-1981-2100.nc'
                           ,decode_times=False,preprocess=preprocess)

NOR = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/DJF-MAR_NorESM-1981-2100.nc'
                        ,decode_times=False,preprocess=preprocess) #NOR=   #Warning 2100= 2099

#NOR=NOR.rename_dims({'AT':'TIME'})

list_name_CMIP5= ['AC3','NOR']
list_SMB_CMIP5 = [AC3,NOR] 
for ds in list_SMB_CMIP5:
    ds['TIME'] = dates
        
# open CMIP6 MAR results   
CNR = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/DJF-MAR_CNRM-CM6-1981-2100.nc'
                           ,decode_times=False,preprocess=preprocess)

CSM = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/DJF-MAR_CESM2-1981-2100.nc'
                           ,decode_times=False,preprocess=preprocess)
#CSM=CSM.rename_dims({'AT':'TIME'})

list_name_CMIP6= ['CNR','CSM']
list_SMB_CMIP6 = [CNR, CSM]
for ds in list_SMB_CMIP6:
    ds['TIME'] = dates
    
    
list_mod = []
list_mod_names = [] 

for i in range(len(list_SMB_CMIP5)):
    list_mod.append(list_SMB_CMIP5[i])
    list_mod_names.append(list_name_CMIP5[i])

for i in range(len(list_SMB_CMIP6)):
    list_mod.append(list_SMB_CMIP6[i])
    list_mod_names.append(list_name_CMIP6[i])

    
    
result_ice_dic = {}
result_grd_dic = {}
result_shf_dic = {}



for i in range(len(list_mod)):
    print (list_mod_names[i])
    #ALL AIS
    dfi=compute_int(list_mod[i], vars_SMB, list_mod_names[i],ice_msk, 1981, 2100)
    result_ice_dic[list_mod_names[i]]=dfi

    #Grounded only
    dfg=compute_int(list_mod[i], vars_SMB, list_mod_names[i],grd_msk, 1981, 2100)
    result_grd_dic[list_mod_names[i]]=dfg

    #Ice shelves
    dfs=dfi-dfg
    result_shf_dic[list_mod_names[i]]=dfs

   
result_dic_djf = {}
result_dic_djf['ice']=result_ice_dic
result_dic_djf['grd']=result_grd_dic
result_dic_djf['shelf']=result_shf_dic    
    

# =============================================================================
# # OUT STORED IN NESTED DICs 1) key: DJF / year 2) (key= reg => ice, grd, shelf) 3) Mod (Xarray=> var)
# =============================================================================  
    
result_mar_dic = {}    
result_mar_dic['year']= result_dic_year
result_mar_dic['DJF'] = result_dic_djf
    

    
#FIG    

sns.set_context('paper')


reg_list=["ice","grd","shelf"]
#plotvsind()

def relin(mod,var):  
    AC3=result_mar_dic['year']['shelf'][mod][var].sel(TIME=slice('2010-01-01','2100-01-01')).rolling(TIME=30, center=True).mean()
    AC3_MAR=result_mar_dic['year']['shelf'][mod][var].sel(TIME=slice('1981-01-01','2009-01-01')).mean().values
    AC3_GCM=result_tt_dic['year'][mod]['TAS'].sel(TIME=slice('2010-01-01','2100-01-01')).rolling(TIME=30, center=True).mean()
    var=(100*(AC3-AC3_MAR)/AC3_MAR)/AC3_GCM
    var=(100/AC3_GCM)*np.log(AC3/AC3_MAR)
    print((var).sel(TIME='2085-01-01').values)
    return var

  
def pp(mod):
    pp=result_mar_dic['year']['ice'][mod]['SF']+result_mar_dic['year']['ice'][mod]['RF']
    AC3=pp.sel(TIME=slice('2010-01-01','2100-01-01')).rolling(TIME=30, center=True).mean()
    AC3_MAR=pp.sel(TIME=slice('1981-01-01','2009-01-01')).mean().values
    AC3_GCM=result_tt_dic['year'][mod]['TAS'].sel(TIME=slice('2010-01-01','2100-01-01')).rolling(TIME=30, center=True).mean()
    var=(100*(AC3-AC3_MAR)/AC3_MAR)/AC3_GCM
    var=(100/AC3_GCM)*np.log(AC3/AC3_MAR)
    print((var).sel(TIME='2085-01-01').values)
    return var

vars_SMB=['SMB','SF','RU','RF']
#

#
#for var in vars_SMB:
#    print (var)
#    fig, ax2 = plt.subplots(nrows=1,ncols=1,figsize=(7,5))
#    for i in range(len(list_mod)):
#        out=relin(list_mod_names[i],var)
#        (out).plot(ax=ax2,label=list_mod_names[i])
#    ax2.legend()
#    ax2.set_xlabel('Year', fontsize=11)
#    ax2.set_ylabel('Relative increase [%/°C]', fontsize=11)
#    ax2.set_title(var, fontsize=14)
#    fig.savefig('fig/relative_increase'+var+'_ice.png',
#           format='PNG', dpi=300) 
