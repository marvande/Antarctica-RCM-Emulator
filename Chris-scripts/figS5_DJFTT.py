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
    ds['TT']= ds['TT'].sel(ATMLAY=0.99973) #ME
    ds_new = ds[['TT']] #'LWD','SWD','SHF','LHF','AL2',
    
    try:
        ds_new= ds_new.sel(SECTOR=1)
    except:
        ds_new=ds_new
        
    #ds_new['SWN']=(1-ds_new['AL2'])*ds_new['SWD']

    return ds_new


# =============================================================================
# Definition of ICE MASK ; used color maps, 
# =============================================================================
MSK,lsm,ground,x2D,y2D,sh,dh = shade.ice_masking()
cmap_seq, cmap_smb, cmap_div, cmap_qual, cmap_BuRd, obs_color, sh_color, cmap_Puor = shade.def_cmaps()

def spcolor(var2D,cmap,bounds,sig):
    return shade.spcolor(x2D,y2D,var2D,cmap,bounds,dh,sig)


# =============================================================================
# #OPEN MAR RESULTS
# =============================================================================


# open present
dates = pd.date_range('1980-01-01', '2019-01-01', freq='AS') 
ER5= xr.open_mfdataset('./data/DJF-MAR_ERA5-1980-2019.nc' #open_mfdatset can also open multifile
                           ,decode_times=False,preprocess=preprocess)
ER5['TIME'] = dates


dates = pd.date_range('1981-01-01', '2100-01-01', freq='AS') 
# open CMIP5 models
AC3 = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/DJF-MAR_ACCESS1.3-1981-2100.nc'
                           ,decode_times=False,preprocess=preprocess)
NOR = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/DJF-MAR_NorESM-1981-2100.nc'
                        ,decode_times=False,preprocess=preprocess) 
NOR=NOR.rename_dims({'AT':'TIME'})
list_name_CMIP5= ['AC3','NOR']
list_SMB_CMIP5 = [AC3,NOR] 
for ds in list_SMB_CMIP5:
    ds['TIME'] = dates    
    
# open CMIP6 models    
CN6 = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/DJF-MAR_CNRM-CM6-1981-2100.nc'
                           ,decode_times=False,preprocess=preprocess)
CSM = xr.open_mfdataset('/home/ckittel/Documents/repo/my-awesome-projection/data/DJF-MAR_CESM2-1981-2100.nc'
                           ,decode_times=False,preprocess=preprocess)
CSM=CSM.rename_dims({'AT':'TIME'})
list_name_CMIP6= ['CN6','CSM']
list_SMB_CMIP6 = [CN6, CSM]
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


vars_SMB = ['TT']# ['SWD','LWD','SWN','AL2','SHF','LHF'] #'TT'

MAR_proj_present = {}
MAR_proj_present_std = {}
MAR_proj_futur = {}
MAR_proj_hot = {}

for i in range(len(list_mod)):
    print (list_mod_names[i])    
    MAR_proj_present[list_mod_names[i]] =list_mod[i].sel(TIME=slice("1981-01-01","2010-01-01")).mean(dim='TIME')
    MAR_proj_present_std[list_mod_names[i]] =list_mod[i].sel(TIME=slice("1981-01-01","2010-01-01")).std(dim='TIME')
    MAR_proj_futur[list_mod_names[i]] =list_mod[i].sel(TIME=slice("2071-01-01","2100-01-01")).mean(dim='TIME')
    MAR_proj_hot[list_mod_names[i]] =list_mod[i].sel(TIME=slice("2095-01-01","2100-01-01")).mean(dim='TIME')


ref=ER5.sel(TIME=slice("1981-01-01","2010-01-01")).mean(dim='TIME')
refstd=ER5.sel(TIME=slice("1981-01-01","2010-01-01")).std(dim='TIME')




def ERA5(var):
    fig, gs = shade.set_fig() 
    
    dataplot=ma.array(ref[var],mask=lsm)
    cmap=cmap_smb
    
    if var == 'TT':
        bounds=[-50,-40,-30,-20,-15,-10,-5,0,5,10]
        extend='both'
        #bounds=[-2000,-1600,-800,-400,-200,-100,-50,-25,0,25,50,100,200,400,800,1600,2000]
        #cmap=cmap_Puor
        

    spcolor(dataplot,cmap,bounds,'sig')
    shade.plot_graticules()
    shade.layout(x2D,y2D,sh,lsm,ground)
    plt.text(0,3100,"MAR(ERA5) Mean "+var+"[°C]",ha='center',va='top',fontsize='large')
    shade.plot_cbar(gs,cmap,bounds,extend=extend)
    
    fig.savefig('./fig/ERA5'+var+'_DJF.png',
                format='PNG', dpi=500)
    
#ERA5('TT')
    
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
    
def ano_pre(mod,var):
    plt.close('all')
    fig, gs = shade.set_fig() 
    
    dataplot= MAR_proj_present[mod][var]-ref[var]
    dataplot=ma.array(dataplot,mask=lsm)
    cmap=cmap_BuRd
    extend='both'
    
    if var == 'TT':
        bounds=[-10,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,10]
        #cmap=cmap_Puor
      
    anos=ma.masked_where((abs(dataplot)<refstd[var]),dataplot) #que ce qui est sig
    spcolor(anos,cmap,bounds,'sig')
    
    anos=ma.masked_where((abs(dataplot)>refstd[var]),dataplot)
    spcolor(anos,cmap,bounds,'nosig')

    shade.plot_graticules()
    shade.layout(x2D,y2D,sh,lsm,ground)
    if mod == "AC3":
        plt.text(0,3100,"A. "+var+" MAR(ACCESS1.3) - MAR(ERA5)",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "NOR":
        plt.text(0,3100,"B. "+var+" MAR(NorESM1-M) - MAR(ERA5)",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "CSM":
        plt.text(0,3100,"D. "+var+" MAR(CESM2) - MAR(ERA5)",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    if mod == "CN6":
        plt.text(0,3100,"C. "+var+" MAR(CNRM-CM6-1) - MAR(ERA5)",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
    #plt.text(2600,-3050,"[kg m-2 yr-1]")
   # shade.plot_cbar(gs,cmap,bounds,extend=extend)
   
   
    bm,rmse=score(mod,var)
    
    #plt.text(-2000,-2300,bm)
    #plt.text(-2000,-2500,rmse)
    
    plt.text(-2000,-2300, r'MB = {:.0f}'.format(bm),fontsize=12)   
    plt.text(-2000,-2500, r'RMSE = {:.0f}'.format(rmse),fontsize=12)
    
    fig.savefig('./fig/figS1_'+var+'_'+mod+'_DJF.png',
                format='PNG', dpi=600)
    


    
def changes(mod,var):

    fig, gs = shade.set_fig() 
    
    dataplot= MAR_proj_futur[mod][var]-MAR_proj_present[mod][var]
    dataplot=ma.array(dataplot,mask=lsm)
    cmap=cmap_BuRd
    extend='both'
    
    ref_std=MAR_proj_present_std[mod][var]
    
    if var == 'TT':
        bounds=[-10,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,10]
        plt.text(0,3100,var+" MAR("+mod+") future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
        plt.text(2600,-3050,"[°C]")
        
    if var == 'LWD' or var == 'SWD' or var == 'SWN' or var == 'LHF' or var == 'SHF':
        bounds=[-50,-40,-30,-20,-10,0,10,20,30,40,50]
        plt.text(0,3100,var+" MAR("+mod+") future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
        plt.text(2600,-3050,"[W m$^{-2}$]")
        
    if var == 'AL2':
        bounds=[-0.5,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.5]
        plt.text(0,3100,"Albedo MAR("+mod+") future - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
        plt.text(2600,-3050,"[-]")
        
        #cmap=cmap_Puor
      
    anos=ma.masked_where((abs(dataplot)<ref_std),dataplot) #que ce qui est sig
    spcolor(anos,cmap,bounds,'sig')
    
    anos=ma.masked_where((abs(dataplot)>ref_std),dataplot)
    spcolor(anos,cmap,bounds,'nosig')

    shade.plot_graticules()
    shade.layout(x2D,y2D,sh,lsm,ground)

    shade.plot_cbar(gs,cmap,bounds,extend=extend)
    
    fig.savefig('./fig/fig5_'+var+'_'+mod+'_DJF.png',
                format='PNG', dpi=500)



def changes_CNRM(mod,var):

    fig, gs = shade.set_fig() 
    
    dataplot= MAR_proj_hot[mod][var]-MAR_proj_present[mod][var]
    dataplot=ma.array(dataplot,mask=lsm)
    cmap=cmap_BuRd
    extend='both'
    
    ref_std=MAR_proj_present_std[mod][var]
    
    if var == 'TT':
        bounds=[-10,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,10]
        plt.text(0,3100,var+" MAR("+mod+") (2095-2100) - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
        plt.text(2600,-3050,"[°C]")
        
    if var == 'LWD' or var == 'SWD' or var == 'SWN' or var == 'LHF' or var == 'SHF':
        #bounds=[-50,-40,-30,-20,-10,0,10,20,30,40,50]
        bounds=np.arange(-50,50,10).tolist()
        plt.text(0,3100,var+" MAR("+mod+") (2095-2100) - present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
        plt.text(2600,-3050,"[W m$^{-2}$]")
        
    if var == 'AL2':
        bounds=[-0.5,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.5]
        plt.text(0,3100,"Albedo MAR("+mod+") (2095-2100)- present",ha='center',va='top',fontsize='large') #[kg m-2 yr-1]
        plt.text(2600,-3050,"[-]")
      
    anos=ma.masked_where((abs(dataplot)<ref_std),dataplot) #que ce qui est sig
    spcolor(anos,cmap,bounds,'sig')
    
    anos=ma.masked_where((abs(dataplot)>ref_std),dataplot)
    spcolor(anos,cmap,bounds,'nosig')

    shade.plot_graticules()
    shade.layout(x2D,y2D,sh,lsm,ground)

    shade.plot_cbar(gs,cmap,bounds,extend=extend)
    
    fig.savefig('./fig/fighot_'+var+'_'+mod+'.png',
                format='PNG', dpi=500)

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
        if var == 'TT':
            bounds=[-10,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,10]
 

    if time == 'pre':
        if var == 'RU':
            bounds=[-200,-50,-40,-30,-20,-10,0,10,20,30,40,50,200]  
        if var == 'ME':
            bounds=[-400,-200,-100,-50,-20,-10,0,10,20,50,100,200,400]
        if var == 'RF':
            bounds=[-400,-200,-100,-50,-20,-10,0,10,20,50,100,200,400]
        if var == 'SU':
            bounds=[-200,-50,-40,-30,-20,-10,0,10,20,30,40,50,200]
        if var == 'TT':
            bounds=[-10,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,10]


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
    plt.text(0.075,0.075,"[°C]",fontsize=12,transform=ax2.transAxes)
    fig.savefig('./fig/cb_'+var+'_'+time+'.png',
               format='PNG', dpi=600)

#for var in vars_SMB: 
#    changes_CNRM('CNRM',var)

#FIG REF
    

  
for var in vars_SMB: 
##    ERA5(var)
    cb(var,cmap_BuRd,'pre')
    for mod in list_mod_names:    
        ano_pre(mod,var)
        








#





#
#
##
