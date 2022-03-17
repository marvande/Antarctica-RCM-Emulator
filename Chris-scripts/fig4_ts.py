#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 08:57:52 2020

@author: ckittel

Time series of SMB and comps + mean values
"""

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from matplotlib import ticker as mticker


import function_plot_fig4_ts


# =============================================================================
# FUNCTIONS
# =============================================================================


def preprocess(ds):
    '''
    Avoid reading/opening the whole ds. => Selection of interesting variable
    Also try to remove SECTOR dimension 
    (Sector=1 corresponds to the ice sheet in MAR (2= tundra or rocks))
    '''
    ds_new = ds[['ME', 'RU', 'RF', 'SF', 'SU','SMB']]
    try:
        ds_new= ds_new.sel(SECTOR=1)
    except:
        ds_new=ds_new
        
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


def area(mask):
    data=mask
    area= data.sum(dim=['X', 'Y']) * (35 * 35)/100
    return area

def annual_sum(data):
    '''This function returns the annual sum
    '''
    annual_sum = data.groupby('TIME.year').sum(dim='TIME')
    return annual_sum

def annual_mean(data):
    annual_mean = data.groupby('TIME.year').mean(dim='TIME')
    return annual_mean


#def seasonal_sum(data, season):
#    seasonal_sum = data.where(data['TIME.season'] == season).groupby(
#        'TIME.year').sum(dim='TIME')
#    return seasonal_sum
    #Ne fonctionnent car DJF de la même année et non D(Y-1)JF
#def seasonal_mean(data, season):
#    seasonal_mean = data.where(data['TIME.season'] == season).groupby(
#        'TIME.year').mean(dim='TIME')
#    return seasonal_mean
#    

# =============================================================================
# # CREATE the ICE MASK 
# =============================================================================
test = xr.open_dataset('data/MARcst-AN35km-176x148.cdf2'
                       ,decode_times=False,engine="netcdf4")
#test=test.rename_dims({'x':'X', 'y':'Y'}) #change dim from ferret


ais = test['AIS'].where(test['AIS'] >0) #Only AIS=1, other islands  =0
ice= test['ICE'].where(test['ICE']>30)  #Ice where ICE mask >= 30% (ICE[0-100%], dividing by 100 in the next ligne)
ice_msk = (ais*ice*test['AREA']/ 100)    #Combine ais + ice/100 * factor area for taking into account the projection

grd=test['GROUND'].where(test['GROUND']>30)
grd_msk = (ais*grd*test['AREA']/ 100)


# =============================================================================
# # OPEN MAR outputs
# =============================================================================


dates = pd.date_range('1979-01-01', '2019-01-01', freq='AS') 


ER5= xr.open_mfdataset('./data/year-MAR_ERA5-1979-2019.nc' #open_mfdatset can also open multifile
                           ,decode_times=False,preprocess=preprocess)
ER5['TIME'] = dates


dates = pd.date_range('1980-01-01', '2100-01-01', freq='AS') 


# open CMIP5 models
AC3 = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_ACCESS1.3-1980-2100.nc'
                           ,decode_times=False,preprocess=preprocess)

NOR = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_NorESM-1980-2100.nc'
                        ,decode_times=False,preprocess=preprocess) 
NOR=NOR.rename_dims({'AT':'TIME'})

list_name_CMIP5= ['AC3','NOR']
list_SMB_CMIP5 = [AC3,NOR] 
for ds in list_SMB_CMIP5:
    ds['TIME'] = dates
    
    
# open CMIP6 models    
CN6 = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_CNRM-CM6-1980-2100.nc'
                           ,decode_times=False,preprocess=preprocess)

CSM = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_CESM2-1980-2100.nc'
                           ,decode_times=False,preprocess=preprocess)
CSM=CSM.rename_dims({'AT':'TIME'})

list_name_CMIP6= ['CN6','CSM']
list_SMB_CMIP6 = [CN6, CSM]
for ds in list_SMB_CMIP6:
    ds['TIME'] = dates
    


# =============================================================================
# # COMPUTE INTEGRATED VALUES + (rolling) ANOMALIES
# =============================================================================
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

    #df['TIME'] = pd.date_range(str(data_start)+'-01-01', str(data_end)+'-01-01', freq='AS') #pd.to_dateTIME(df[['year', 'month', 'day']])
    #df = df.set_index('TIME')
    return df
        
def compute_ano(df,refs,refe,period):
    dfo= df - df.sel(TIME=slice(str(refs)+'-01-01',str(refe)+'-01-01')).mean(dim=['TIME'])
    dfor= dfo.rolling(TIME=period, center=True).mean()
    return dfor 
    

    
vars_SMB = ['SMB','SF','RU','RF','SU','ME']
  
list_mod = []
list_mod_names = [] 


for i in range(len(list_SMB_CMIP5)):
    list_mod.append(list_SMB_CMIP5[i])
    list_mod_names.append(list_name_CMIP5[i])

for i in range(len(list_SMB_CMIP6)):
    list_mod.append(list_SMB_CMIP6[i])
    list_mod_names.append(list_name_CMIP6[i])


rolling=5
result_ice_dic = {}
result_ice_ano_dic={}

result_grd_dic={}
result_grd_ano_dic={}

result_shf_dic={}
result_shf_ano_dic={}


for i in range(len(list_mod)):
    print (list_mod_names[i])
    #ALL AIS
    dfi=compute_int(list_mod[i], vars_SMB, list_mod_names[i],ice_msk, 1980, 2100)
    result_ice_dic[list_mod_names[i]]=dfi
    
    dfano=compute_ano(dfi,1981,2010,rolling)
    result_ice_ano_dic[list_mod_names[i]]=dfano
        
    #Grounded only
    dfg=compute_int(list_mod[i], vars_SMB, list_mod_names[i],grd_msk, 1980, 2100)
    result_grd_dic[list_mod_names[i]]=dfg
    
    dfano=compute_ano(dfg,1981,2010,rolling)
    result_grd_ano_dic[list_mod_names[i]]=dfano   
    
    dfs=dfi-dfg
    result_shf_dic[list_mod_names[i]]=dfs
    dfano=compute_ano(dfs,1981,2010,rolling)
    result_shf_ano_dic[list_mod_names[i]]=dfano



ERA5=compute_int(ER5,vars_SMB,'ER5',ice_msk, 1981, 2100)
ERAd=compute_int(ER5,vars_SMB,'ER5',grd_msk, 1981, 2100)
ERA5s=ERA5-ERAd
# =============================================================================
# # OUT STORED IN DIC (key= reg => ice, grd, shelf)
# =============================================================================  
    
result_dic = {}
result_dic['ice']=result_ice_dic
result_dic['grd']=result_grd_dic
result_dic['shelf']=result_shf_dic
result_dic_ano = {}
result_dic_ano['ice']=result_ice_ano_dic
result_dic_ano['grd']=result_grd_ano_dic
result_dic_ano['shelf']=result_shf_ano_dic


reg_list=["ice","grd","shelf"]

#    colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple", "tomato"]
#    palette = sns.xkcd_palette(colors)
#    sns.set_palette(palette)    


#for reg in reg_list:  
#   fig_ts(reg) #ice grd shelf
#  fig_ts_maj(reg)
#   fig_ts_sf_pe_ice(reg)
#
#fig_comb_maj()
#
##vars_SMB=['SMB']
#for var in vars_SMB:
#
##    print ('ERA5')
##   print (ERA5[var].sel(TIME=slice('1981-01-01','2010-01-01')).mean(dim=['TIME']).values)
##    print (ERA5[var].sel(TIME=slice('1981-01-01','2010-01-01')).std(dim=['TIME']).values)
#    for i in range(len(list_mod)):
#       std=result_dic_ano['grd'][list_mod_names[i]][var].sel(TIME=slice('1981-01-01','2010-01-01')).std(dim=['TIME']).values
#       ano=result_dic_ano['grd'][list_mod_names[i]][var].sel(TIME=slice('2071-01-01','2100-01-01')).mean(dim=['TIME']).values
##       print (std,ano)
#       if abs(ano) > std:
#           print("sig changes grd for", list_mod_names[i], "in var", var )
#       else:
#           print ("no sig changes grd for", list_mod_names[i], "in var", var )
#           
#           
#       std=result_dic_ano['shelf'][list_mod_names[i]][var].sel(TIME=slice('1981-01-01','2010-01-01')).std(dim=['TIME']).values
#       ano=result_dic_ano['shelf'][list_mod_names[i]][var].sel(TIME=slice('2071-01-01','2100-01-01')).mean(dim=['TIME']).values
##       print (std,ano)
#       if abs(ano) > std:
#           print("sig changes shf for", list_mod_names[i], "in var", var )
#       else:
#           print ("no sig changes shf for", list_mod_names[i], "in var", var )
           
           
#       print(list_mod_names[i])
#       print(result_dic['ice'][list_mod_names[i]][var].sel(TIME=slice('1981-01-01','2010-01-01')).mean(dim=['TIME']).values)
#       print(result_dic_ano['grd'][list_mod_names[i]][var].sel(TIME=slice('2071-01-01','2100-01-01')).mean(dim=['TIME']).values)


#       print(result_dic_ano['grd'][list_mod_names[i]][var].sel(TIME=slice('1981-01-01','2010-01-01')).std(dim=['TIME']).values)

 #      print(result_dic_ano['shelf'][list_mod_names[i]][var].sel(TIME=slice('1981-01-01','2010-01-01')).std(dim=['TIME']).values)


#       print(result_dic_ano['ice'][list_mod_names[i]][var].sel(TIME=slice('2071-01-01','2100-01-01')).mean(dim=['TIME']).values)
#       print(result_dic_ano['shelf'][list_mod_names[i]][var].sel(TIME=slice('2071-01-01','2100-01-01')).mean(dim=['TIME']).values)
#       print(result_dic_ano['shelf'][list_mod_names[i]][var].sel(TIME=slice('2071-01-01','2100-01-01')).std(dim=['TIME']).values)

#        

#def out_ano_present_to_latex():
#    print('\\begin{table}[]')
#    print('\\begin{tabular}{lrrrrrr}')
#    print( '\\tophline')
#    print( 'Mean (Gt\,yr$^{-1}$)   & \multicolumn{1}{c}{SMB} & \multicolumn{1}{c}{SF} & \multicolumn{1}{c}{RF} & \multicolumn{1}{c}{SU} & \multicolumn{1}{c}{RU} & \multicolumn{1}{c}{ME} \\\\')
#    print('\middlehline')
#    
#    smbref=ERA5['SMB'].sel(TIME=slice('1981-01-01','2010-01-01')).mean(dim=['TIME']).values
#    sfref=ERA5['SF'].sel(TIME=slice('1981-01-01','2010-01-01')).mean(dim=['TIME']).values
#    rfref=ERA5['RF'].sel(TIME=slice('1981-01-01','2010-01-01')).mean(dim=['TIME']).values
#    suref=ERA5['SU'].sel(TIME=slice('1981-01-01','2010-01-01')).mean(dim=['TIME']).values
#    ruref=ERA5['RU'].sel(TIME=slice('1981-01-01','2010-01-01')).mean(dim=['TIME']).values
#    meref=ERA5['ME'].sel(TIME=slice('1981-01-01','2010-01-01'),SECTOR1_1='1').mean(dim=['TIME']).values
#    
#    
#    smbstd=ERA5['SMB'].sel(TIME=slice('1981-01-01','2010-01-01')).std(dim=['TIME']).values
#    sfstd=ERA5['SF'].sel(TIME=slice('1981-01-01','2010-01-01')).std(dim=['TIME']).values
#    rfstd=ERA5['RF'].sel(TIME=slice('1981-01-01','2010-01-01')).std(dim=['TIME']).values
#    sustd=ERA5['SU'].sel(TIME=slice('1981-01-01','2010-01-01')).std(dim=['TIME']).values
#    rustd=ERA5['RU'].sel(TIME=slice('1981-01-01','2010-01-01')).std(dim=['TIME']).values
#    mestd=ERA5['ME'].sel(TIME=slice('1981-01-01','2010-01-01'),SECTOR1_1='1').std(dim=['TIME']).values
# 
#    print('MAR(ERA5)& ',int(smbref),'\pm ',int(smbstd),' &  ',int(sfref),'\pm ',int(sfstd),' &  ',int(rfref),'\pm ',int(rfstd), '  &  ',int(suref),'\pm ',int(sustd),'  &  ',int(ruref),'\pm ',int(rustd),'  &  ',int(meref),'\pm ',int(mestd), '\\\\')
#    print('\multicolumn{1}{c}{Anomaly (Gt\,yr$^{-1}$)} & \multicolumn{1}{c}{SMB} & \multicolumn{1}{c}{SF} & \multicolumn{1}{c}{RF} & \multicolumn{1}{c}{SU} & \multicolumn{1}{c}{RU} & \multicolumn{1}{c}{ME} \\\\')
#    print ('\middlehline')
#    for i in range(len(list_mod)):      
#        smb=result_dic['ice'][list_mod_names[i]]['SMB'].sel(TIME=slice('1981-01-01','2010-01-01')).mean(dim=['TIME']).values-smbref
#        sf=result_dic['ice'][list_mod_names[i]]['SF'].sel(TIME=slice('1981-01-01','2010-01-01')).mean(dim=['TIME']).values-sfref
#        rf=result_dic['ice'][list_mod_names[i]]['RF'].sel(TIME=slice('1981-01-01','2010-01-01')).mean(dim=['TIME']).values-rfref
#        su=result_dic['ice'][list_mod_names[i]]['SU'].sel(TIME=slice('1981-01-01','2010-01-01')).mean(dim=['TIME']).values-suref
#        ru=result_dic['ice'][list_mod_names[i]]['RU'].sel(TIME=slice('1981-01-01','2010-01-01')).mean(dim=['TIME']).values-ruref
#        me=result_dic['ice'][list_mod_names[i]]['ME'].sel(TIME=slice('1981-01-01','2010-01-01'),SECTOR1_1='1').mean(dim=['TIME']).values-meref  
#        print('MAR('+list_mod_names[i]+')& ',int(smb),'&  ',int(sf),' &  ',int(rf), '  &  ',int(su),'  &  ',int(ru),'  &  ',int(me), '\\\\')   
#    print ('\\bottomline')
#    print('\end{tabular}')
#    print('\end{table}')



