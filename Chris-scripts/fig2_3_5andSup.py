#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 09:52:33 2020

@author: ckittel
"""


import numpy as np
import numpy.ma as ma # masked array
import xarray as xr
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import matplotlib.gridspec as gridspec
#from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import ck_shade2D as shade


def preprocess(ds):
    '''
    Avoid reading/opening the whole ds. => Selection of interesting variable
    Also try to remove SECTOR dimension 
    (Sector=1 corresponds to the ice sheet in MAR (2= tundra or rocks))
    '''
    ds_new = ds[['ME', 'RU', 'RF', 'SF', 'SU','SMB','SP','UV']]
    try:
        ds_new= ds_new.sel(SECTOR=1) #SMB RU SU
    except:
        ds_new=ds_new
    try:
        ds_new['ME']= ds_new['ME'].sel(SECTOR1_1=1) #ME
    except:
        ds_new['ME']=ds_new['ME'] 
    return ds_new


# =============================================================================
# Definition of ICE MASK ; used color maps, 
# =============================================================================
MSK,lsm,ground,shelf,x2D,y2D,sh,dh = shade.ice_masking()
cmap_seq, cmap_smb, cmap_div, cmap_qual, cmap_BuRd, obs_color, sh_color, cmap_Puor = shade.def_cmaps()

def spcolor(var2D,cmap,bounds,sig):
    return shade.spcolor(x2D,y2D,var2D,cmap,bounds,dh,sig)


# =============================================================================
# #OPEN MAR RESULTS
# =============================================================================


# open present
dates = pd.date_range('1979-01-01', '2019-01-01', freq='AS') 
ER5= xr.open_mfdataset('./data/year-MAR_ERA5-1979-2019.nc' #open_mfdatset can also open multifile
                           ,decode_times=False,preprocess=preprocess,engine="netcdf4")
ER5['TIME'] = dates


dates = pd.date_range('1980-01-01', '2100-01-01', freq='AS') 
# open CMIP5 models
AC3 = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_ACCESS1.3-1980-2100.nc'
                           ,decode_times=False,preprocess=preprocess,engine="netcdf4")
NOR = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_NorESM-1980-2100.nc'
                        ,decode_times=False,preprocess=preprocess,engine="netcdf4") 
NOR=NOR.rename_dims({'AT':'TIME'})
list_name_CMIP5= ['AC3','NOR']
list_SMB_CMIP5 = [AC3,NOR] 
for ds in list_SMB_CMIP5:
    ds['TIME'] = dates    
    
# open CMIP6 models    
CNRM = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_CNRM-CM6-1980-2100.nc'
                           ,decode_times=False,preprocess=preprocess,engine="netcdf4")
CESM = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/year-MAR_CESM2-1980-2100.nc'
                           ,decode_times=False,preprocess=preprocess,engine="netcdf4")
CESM=CESM.rename_dims({'AT':'TIME'})
list_name_CMIP6= ['CN6','CSM']
list_SMB_CMIP6 = [CNRM, CESM]
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


vars_SMB = ['SMB', 'SF','RF','RU','ME','SU','UV']  

MAR_proj_present = {}
MAR_proj_present_std = {}
MAR_proj_futur = {}
MAR_proj_hot = {}

MAR_proj_25 = {}
MAR_proj_40 = {}

for i in range(len(list_mod)):
    print (list_mod_names[i])    
    MAR_proj_present[list_mod_names[i]] =list_mod[i].sel(TIME=slice("1981-01-01","2010-01-01")).mean(dim='TIME')
    MAR_proj_present_std[list_mod_names[i]] =list_mod[i].sel(TIME=slice("1981-01-01","2010-01-01")).std(dim='TIME')
    MAR_proj_futur[list_mod_names[i]] =list_mod[i].sel(TIME=slice("2071-01-01","2100-01-01")).mean(dim='TIME')
    MAR_proj_hot[list_mod_names[i]] =list_mod[i].sel(TIME=slice("2095-01-01","2100-01-01")).mean(dim='TIME')


ref=ER5.sel(TIME=slice("1981-01-01","2010-01-01")).mean(dim='TIME')
refstd=ER5.sel(TIME=slice("1981-01-01","2010-01-01")).std(dim='TIME')

MAR_proj_25['AC3'] =AC3.sel(TIME=slice("2041-01-01","2070-01-01")).mean(dim='TIME') #+2.45
MAR_proj_25['NOR'] =NOR.sel(TIME=slice("2071-01-01","2100-01-01")).mean(dim='TIME') #+2.54
#MAR_proj_25['NOR'] =NOR.sel(TIME=slice("2068-01-01","2097-01-01")).mean(dim='TIME')
MAR_proj_25['CN6'] =CNRM.sel(TIME=slice("2031-01-01","2060-01-01")).mean(dim='TIME') #+2.44
MAR_proj_25['CSM'] =CESM.sel(TIME=slice("2031-01-01","2060-01-01")).mean(dim='TIME') #+2.44

MAR_proj_40['AC3'] =AC3.sel(TIME=slice("2071-01-01","2100-01-01")).mean(dim='TIME') #+2.45
MAR_proj_40['NOR'] =NOR.sel(TIME=slice("2101-01-01","2102-01-01")).mean(dim='TIME') #NAN!
#MAR_proj_40['NOR'] =NOR.sel(TIME=slice("2068-01-01","2097-01-01")).mean(dim='TIME')
MAR_proj_40['CN6'] =CNRM.sel(TIME=slice("2051-01-01","2080-01-01")).mean(dim='TIME') #+2.44
MAR_proj_40['CSM'] =CESM.sel(TIME=slice("2051-01-01","2080-01-01")).mean(dim='TIME') #+2.44

def ERA5(var):
    fig, gs = shade.set_fig() 
    
    dataplot=ma.array(ref[var],mask=lsm)
    cmap=cmap_smb
    
    if var == 'SMB':
        bounds=[-1000,0,20,50,100,200,400,800,1600,3200]
        extend='both'
        #bounds=[-2000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,2000]
        #cmap=cmap_Puor
        
    if var == 'SF':
        bounds=[0,20,50,100,200,400,800,1600,3200]
        extend='max'

    if var == 'RF':
        mskshade=(dataplot<1)
        dataplot=ma.array(dataplot,mask=mskshade)
        bounds=[0,5,10,20,50,100,200,400,800] 
        extend='max'
        
    if var == 'ME':
        mskshade=(dataplot<1)
        dataplot=ma.array(dataplot,mask=mskshade)
        bounds=[0,5,10,20,50,100,200,400,800] 
        extend='max'
        
    if var == 'RU':
        mskshade=(dataplot<1)
        dataplot=ma.array(dataplot,mask=mskshade)
        bounds=[0,5,10,20,50,100,200,400,800] 
        extend='max'
        
    if var == 'SU':
        bounds=[-400,-200,-100,-50,-25,0,25,50,100,200,400] 
        extend='both'
        cmap=cmap_div

    spcolor(dataplot,cmap,bounds,'sig')
    shade.plot_graticules()
    shade.layout(x2D,y2D,sh,lsm,ground)
    plt.text(0,3100,"MAR(ERA5) Mean "+var+"[kg m-2 yr-1]",ha='center',va='top',fontsize='large')
    shade.plot_cbar(gs,cmap,bounds,extend=extend)
    
    fig.savefig('./fig/ERA5'+var+'.png',
                format='PNG', dpi=500)
    

    
    
from sklearn.metrics import mean_squared_error
from math import sqrt

def cb(var,cmap,time):
    fig, ax2 = plt.subplots(nrows=1,ncols=1,figsize=(1.5,9.847))
#    cmap=cmap_BuRd
    extend='both'
    
    if var == 'SMB':
        bounds=[-2000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,2000]
    if var == 'SF':
        bounds=[-2000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,2000]
        
    if time == 'fut':
        if var == 'RU':
            bounds=[-4000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,4000]
        if var == 'ME':
            bounds=[-6000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,6000]
        if var == 'RF':
            bounds=[-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600]
        if var == 'SU':
            bounds=[-400,-200,-100,-50,-25,-10,0,10,25,50,100,200,400]
        
        if var == 'SP':
            bounds=[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4]
 

    if time == 'pre':
        if var == 'RU':
            bounds=[-200,-50,-40,-30,-20,-10,0,10,20,30,40,50,200]  
        if var == 'ME':
            bounds=[-400,-200,-100,-50,-20,-10,0,10,20,50,100,200,400]
        if var == 'RF':
            bounds=[-400,-200,-100,-50,-20,-10,0,10,20,50,100,200,400]
        if var == 'SU':
            bounds=[-200,-50,-40,-30,-20,-10,0,10,20,30,40,50,200]      


    axcb = fig.add_axes([0.2, 0.2, 0.2, 0.6])
    

    if extend=='both':
        ticks = bounds[1:-1]
    elif extend=='max':
        ticks = bounds[:-1]
    elif extend=='min':
        ticks = bounds[1:]
    elif extend=='neither':
        ticks = bounds[:]
    else:
        print ('Error in plot_cbar, extend keyword')
        return
    
    cbar=mpl.colorbar.ColorbarBase(axcb,cmap=cmap,norm=shade.norm_cmap(cmap,bounds),ticks=ticks,extend=extend)
    cbar.ax.tick_params(labelsize=12)


    ax2.set_visible(False)

    
    
    #shade.plot_cbaronly(ax2,cmap,bounds,extend=extend)
    if var != 'SP':
        plt.text(0.075,0.075,"(kg m$^{-2}$ yr$^{-1}$)",fontsize=12,transform=ax2.transAxes)
    elif var == 'SP':
        plt.text(0.075,0.075,"(hPa)",fontsize=12,transform=ax2.transAxes)
        
    fig.savefig('./fig/cb_'+var+'_'+time+'.png',
               format='PNG', dpi=600)
    
    
def score(mod,var):
    testar=MAR_proj_present[mod][var].where(MSK['ICE']>30)
    refar=ref[var].where(MSK['ICE']>30)
    
    bm=np.nanmean(testar)-np.nanmean(refar)
    rms=rmse(mod,var)   
   # print(bm)
    
    return bm,rms
def xymean(mvar2d, marea2d):
    return (mvar2d * marea2d).sum() / (marea2d).sum()

def rmse(mod,var):
    var1=MAR_proj_present[mod][var].values
    ref1=ref[var].values
    mask1=MSK['ICE'].values
    area2d=(MSK['ICE']/MSK['ICE']).values #fausse aire de 1
    
    mod2d=np.ma.masked_where(mask1<30, var1)
    rean2d=np.ma.masked_where(mask1<30, ref1)
    area2dm=np.ma.masked_where(mask1<30, area2d)
    
    diff2 = (mod2d - rean2d)**2.
    rmse = np.sqrt(xymean(diff2, area2dm))
    #print(rmse)
    return rmse


def test_cha(mod,var):
    dimX=176
    dimY=148
    var1=MAR_proj_present['AC3']['SF'].values
    var1b=var1.reshape(dimX*dimY)
    
    ref1=ref['SF'].values
    ref1b=ref1.reshape(dimX*dimY)
    
    mask1=MSK['ICE'].values
    MSK2b=mask1.reshape(dimX*dimY)
    
    
    SD1=0.0
    count=0.0
    for i in range(0,dimX*dimY):
        if MSK2b[i] >= 30.0:
            SD1=SD1+(var1b[i]-ref1b[i])**2
            count=count+1
    SD2=np.sqrt(SD1/count)

    print(SD2)

  
def ano_pre(mod,var):
    plt.close('all')
    fig, gs = shade.set_fig() 
    
    dataplot= MAR_proj_present[mod][var]-ref[var]
    dataplot=ma.array(dataplot,mask=lsm)
    cmap=cmap_div
    extend='both'


    if var == 'SMB':
        bounds=[-2000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,2000]
    if var == 'SF':
        bounds=[-2000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,2000]
    if var == 'SU':
        bounds=[-200,-50,-40,-30,-20,-10,0,10,20,30,40,50,200]
    if var == 'ME':
        bounds=[-400,-200,-100,-50,-20,-10,0,10,20,50,100,200,400]
        mskshade=(ref[var]<1) & (MAR_proj_present[mod][var]<1)
        dataplot=ma.array(dataplot,mask=mskshade)

        
        #cmap=cmap_Puor
    if var == 'RF':
        bounds=[-400,-200,-100,-50,-20,-10,0,10,20,50,100,200,400]
        mskshade=(ref[var]<1) & (MAR_proj_present[mod][var]<1)
        dataplot=ma.array(dataplot,mask=mskshade)        
    if var == 'RU':
        bounds=[-200,-50,-40,-30,-20,-10,0,10,20,30,40,50,200]
        mskshade=(ref[var]<1) & (MAR_proj_present[mod][var]<1)
        dataplot=ma.array(dataplot,mask=mskshade)

        
    anos=ma.masked_where((abs(dataplot)<refstd[var]),dataplot) #que ce qui est sig
    spcolor(anos,cmap,bounds,'sig')
    
    anos=ma.masked_where((abs(dataplot)>refstd[var]),dataplot)
    spcolor(anos,cmap,bounds,'nosig')

    shade.plot_graticules()
    shade.layout(x2D,y2D,sh,lsm,ground)
    #plt.text(0,3100,var+" MAR("+mod+") - MAR(ERA5)",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    #plt.text(2600,-3050,"[kg m-2 yr-1]")
    #shade.plot_cbar(gs,cmap,bounds,extend=extend)
    
    varl=var
    if var == 'SF':
        varl='Snowfall'
    if var == 'RF':
        varl='Rainfall'
    if var == 'SU':
        varl='Sublimation'
    if var == 'ME':
        varl='Melt'
    if var == 'RU':
        varl='Runoff' 
    
    if mod == "AC3":
        plt.text(0,3100,"A. "+varl+" MAR(ACCESS1.3) - MAR(ERA5)",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "NOR":
        plt.text(0,3100,"B. "+varl+" MAR(NorESM1-M) - MAR(ERA5)",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "CSM":
        plt.text(0,3100,"D. "+varl+" MAR(CESM2) - MAR(ERA5)",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "CN6":
        plt.text(0,3100,"C. "+varl+" MAR(CNRM-CM6-1) - MAR(ERA5)",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    #plt.text(2600,-3050,"[kg m-2 yr-1]")
   # shade.plot_cbar(gs,cmap,bounds,extend=extend)
   
   
    bm,rmse=score(mod,var)
    
    #plt.text(-2000,-2300,bm)
    #plt.text(-2000,-2500,rmse)
    
    plt.text(-2000,-2300, r'MB = {:.0f}'.format(bm),fontsize=12)   
    plt.text(-2000,-2500, r'RMSE = {:.0f}'.format(rmse),fontsize=12)


    fig.savefig('./fig/figS1_'+var+'_'+mod+'.png',
                format='PNG', dpi=600)
    
    
    
def changes(mod,var):
    plt.close('all')   
    fig, gs = shade.set_fig() 
    
    dataplot= MAR_proj_futur[mod][var]-MAR_proj_present[mod][var]
    dataplot=ma.array(dataplot,mask=lsm)
    cmap=cmap_BuRd
    extend='both'
    
    varl=var
    if var == 'SF':
        varl='Snowfall'
    if var == 'RF':
        varl='Rainfall'
    if var == 'SU':
        varl='Sublimation'
    if var == 'ME':
        varl='Melt'
    if var == 'RU':
        varl='Runoff' 
        
    ref_std=MAR_proj_present_std[mod][var]
    
    if var == 'SMB':
        bounds=[-2000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,2000]
    if var == 'SF':
        bounds=[-2000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,2000]
    if var == 'RF':
        bounds=[-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600]
        mskshade=(MAR_proj_futur[mod][var]<1) 
        dataplot=ma.array(dataplot,mask=mskshade)
    if var == 'SU':
        bounds=[-400,-200,-100,-50,-25,-10,0,10,25,50,100,200,400]
    if var == 'ME':
        bounds=[-6000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,6000]
        mskshade=(MAR_proj_futur[mod][var]<1) 
        dataplot=ma.array(dataplot,mask=mskshade)
    if var == 'RU':
        bounds=[-4000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,4000]
        mskshade=(MAR_proj_futur[mod][var]<1) 
        dataplot=ma.array(dataplot,mask=mskshade)
        #cmap=cmap_Puor

    if var == 'SP':
        bounds=[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4]
        

    anos=ma.masked_where((abs(dataplot)<ref_std),dataplot) #que ce qui est sig
    spcolor(anos,cmap,bounds,'sig')
    
    anos=ma.masked_where((abs(dataplot)>ref_std),dataplot)
    spcolor(anos,cmap,bounds,'nosig')

    shade.plot_graticules()
    shade.layout(x2D,y2D,sh,lsm,ground)
    if mod == "AC3":
        plt.text(0,3100,"A. "+varl+" MAR(ACCESS1.3) future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "NOR":
        plt.text(0,3100,"B. "+varl+" MAR(NorESM1-M) future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "CSM":
        plt.text(0,3100,"D. "+varl+" MAR(CESM2) future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "CN6":
        plt.text(0,3100,"C. "+varl+" MAR(CNRM-CM6-1) future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    #plt.text(2600,-3050,"[kg m-2 yr-1]")
   # shade.plot_cbar(gs,cmap,bounds,extend=extend)
   
    #plt.contour(x2D,y2D,MAR_proj_futur[mod][var],[0],linewidths=1,colors='g')
    
    fig.savefig('./fig/fig5_'+var+'_'+mod+'.png',
                format='PNG', dpi=600)



def changes25(mod,var):
    plt.close('all')   
    fig, gs = shade.set_fig() 
    #25/40
    dataplot= MAR_proj_25[mod][var]-MAR_proj_present[mod][var]
    dataplot=ma.array(dataplot,mask=lsm)
    cmap=cmap_BuRd
    extend='both'
    
    
    ref_std=MAR_proj_present_std[mod][var]
    
    if var == 'SMB':
        bounds=[-2000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,2000]
    if var == 'SF':
        bounds=[-2000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,2000]
    if var == 'RF':
        bounds=[-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600]
        mskshade=(MAR_proj_futur[mod][var]<1) 
        dataplot=ma.array(dataplot,mask=mskshade)
    if var == 'SU':
        bounds=[-400,-200,-100,-50,-25,-10,0,10,25,50,100,200,400]
    if var == 'ME':
        bounds=[-6000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,6000]
        mskshade=(MAR_proj_futur[mod][var]<1) 
        dataplot=ma.array(dataplot,mask=mskshade)
    if var == 'RU':
        bounds=[-4000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,4000]
        mskshade=(MAR_proj_futur[mod][var]<1) 
        dataplot=ma.array(dataplot,mask=mskshade)
        #cmap=cmap_Puor

    if var == 'SP':
        bounds=[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4]
        

    anos=ma.masked_where((abs(dataplot)<ref_std),dataplot) #que ce qui est sig
    spcolor(anos,cmap,bounds,'sig')
    
    anos=ma.masked_where((abs(dataplot)>ref_std),dataplot)
    spcolor(anos,cmap,bounds,'nosig')

    shade.plot_graticules()
    shade.layout(x2D,y2D,sh,lsm,ground)
    if mod == "AC3":
        plt.text(0,3100,"A. "+var+" MAR(ACCESS1.3) +2.5°C - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "NOR":
        plt.text(0,3100,"B. "+var+" MAR(NorESM1-M) +2.5°C - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "CSM":
        plt.text(0,3100,"D. "+var+" MAR(CESM2) +2.5°C - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "CN6":
        plt.text(0,3100,"C. "+var+" MAR(CNRM-CM6-1) +2.5°C - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    #plt.text(2600,-3050,"[kg m-2 yr-1]")
   # shade.plot_cbar(gs,cmap,bounds,extend=extend)
    
    fig.savefig('./fig/fig25C_'+var+'_'+mod+'.png',
                format='PNG', dpi=600)


def changes_CNRM(mod,var):

    fig, gs = shade.set_fig() 
    
    dataplot= MAR_proj_hot[mod][var]-MAR_proj_present[mod][var]
    dataplot=ma.array(dataplot,mask=lsm)
    
    cmap=cmap_BuRd
    extend='both'
    
    ref_std=MAR_proj_present_std[mod][var]
    
    if var == 'SMB':
        bounds=[-2000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,2000]
    if var == 'SF':
        bounds=[-2000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,2000]
    if var == 'RF':
        bounds=[-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600]
        mskshade=(MAR_proj_hot[mod][var]<1) & (MAR_proj_present[mod][var]<1)
        dataplot=ma.array(dataplot,mask=mskshade)
    if var == 'SU':
        bounds=[-400,-200,-100,-50,-25,-10,0,10,25,50,100,200,400]
    if var == 'ME':
        bounds=[-6000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,6000]
        mskshade=(MAR_proj_hot[mod][var]<1) & (MAR_proj_present[mod][var]<1)
        dataplot=ma.array(dataplot,mask=mskshade)
    if var == 'RU':
        bounds=[-4000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,4000]
        mskshade=(MAR_proj_hot[mod][var]<1) & (MAR_proj_present[mod][var]<1)
        dataplot=ma.array(dataplot,mask=mskshade)
        #cmap=cmap_Puor
      
    anos=ma.masked_where((abs(dataplot)<ref_std),dataplot) #que ce qui est sig
    spcolor(anos,cmap,bounds,'sig')
    
    anos=ma.masked_where((abs(dataplot)>ref_std),dataplot)
    spcolor(anos,cmap,bounds,'nosig')

    shade.plot_graticules()
    shade.layout(x2D,y2D,sh,lsm,ground)
    plt.text(0,3100,var+" MAR("+mod+") (2095-2100) - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    plt.text(2600,-3050,"[kg m-2 yr-1]")
    shade.plot_cbar(gs,cmap,bounds,extend=extend)
    
    fig.savefig('./fig/fighot_'+var+'_'+mod+'.png',
                format='PNG', dpi=900)



def changes_SP(mod):
    #plt.close('all')   
    fig, gs = shade.set_fig() 
    var='SP'
    dataplot= MAR_proj_futur[mod][var]-MAR_proj_present[mod][var]
    #dataplot=ma.array(dataplot,mask=lsm)
    cmap=cmap_BuRd
    extend='both'
    
    
    ref_std=MAR_proj_present_std[mod][var]
    
    if var == 'SP':
        print("test")
        bounds=[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4]
        

    anos=ma.masked_where((abs(dataplot)<ref_std),dataplot) #que ce qui est sig
    spcolor(anos,cmap,bounds,'sig')
    
    anos=ma.masked_where((abs(dataplot)>ref_std),dataplot)
    spcolor(anos,cmap,bounds,'nosig')

    shade.plot_graticules()
    shade.layout(x2D,y2D,sh,lsm,ground)
    if mod == "AC3":
        plt.text(0,3100,"A. "+var+" MAR(ACCESS1.3) future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "NOR":
        plt.text(0,3100,"B. "+var+" MAR(NorESM1-M) future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "CSM":
        plt.text(0,3100,"D. "+var+" MAR(CESM2) future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "CN6":
        plt.text(0,3100,"C. "+var+" MAR(CNRM-CM6-1) future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    #plt.text(2600,-3050,"[kg m-2 yr-1]")
   # shade.plot_cbar(gs,cmap,bounds,extend=extend)
    
    fig.savefig('./fig/figS2_'+var+'_'+mod+'.png',
                format='PNG', dpi=600)


def changes_UV(mod):
    #plt.close('all')   
    fig, gs = shade.set_fig() 
    var='UV'
    dataplot= MAR_proj_futur[mod][var]-MAR_proj_present[mod][var]
    #dataplot=ma.array(dataplot,mask=lsm)
    cmap=cmap_BuRd
    extend='both'
    
    
    ref_std=MAR_proj_present_std[mod][var]
    
    if var == 'UV':
        print("test")
        bounds=[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4]
        

    anos=ma.masked_where((abs(dataplot)<ref_std),dataplot) #que ce qui est sig
    spcolor(anos,cmap,bounds,'sig')
    
    anos=ma.masked_where((abs(dataplot)>ref_std),dataplot)
    spcolor(anos,cmap,bounds,'nosig')

    shade.plot_graticules()
    shade.layout(x2D,y2D,sh,lsm,ground)
    if mod == "AC3":
        plt.text(0,3100,"A. "+var+" MAR(ACCESS1.3) future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "NOR":
        plt.text(0,3100,"B. "+var+" MAR(NorESM1-M) future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "CSM":
        plt.text(0,3100,"D. "+var+" MAR(CESM2) future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "CN6":
        plt.text(0,3100,"C. "+var+" MAR(CNRM-CM6-1) future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    #plt.text(2600,-3050,"[kg m-2 yr-1]")
   # shade.plot_cbar(gs,cmap,bounds,extend=extend)
    
    fig.savefig('./fig/figS2_'+var+'_'+mod+'.png',
                format='PNG', dpi=600)

###FIG REF
vars_SMB=['UV']#['SMB','SU','ME','RF','RU']
for var in vars_SMB: 
    for mod in list_mod_names:    
#       ano_pre(mod,var)
#       cb(var,cmap_div,'pre')
#        changes_SP(mod)
        changes_UV(mod)
#       changes(mod,var)
#       changes25(mod,var)
        cb(var,cmap_BuRd,'fut')




#





#
#
##
