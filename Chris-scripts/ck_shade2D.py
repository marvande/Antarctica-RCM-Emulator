#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 09:52:33 2020

@author: ckittel

general functions for map with MAR
"""


import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import matplotlib.gridspec as gridspec
#from matplotlib.colors import ListedColormap, LinearSegmentedColormap



xmin=-2800
ymin=xmin
xmax=2800
ymax=xmax

#xmin=-2600
#ymin=xmin+50
#xmax=2600
#ymax=xmax-90

def ice_masking():
    '''create the ICE mask + usefull stuffs as x2D,y2D'''
    #CREATE the ICE MASK 
    MSK = xr.open_dataset('data/MARcst-AN35km-176x148.cdf2'
                       ,decode_times=False,engine="netcdf4")
   # MSK=MSK.rename_dims({'x':'X', 'y':'Y'}) #change dim from ferret


    ais = MSK['AIS'].where(MSK['AIS'] >0) #Only AIS=1, other islands  =0
    ice= MSK['ICE'].where(MSK['ICE']>30)  #Ice where ICE mask >= 30% (ICE[0-100%], dividing by 100 in the next ligne)
    ice_msk = (ais*ice*MSK['AREA']/ 100)    #Combine ais + ice/100 * factor area for taking into account the projection

    grd=MSK['GROUND'].where(MSK['GROUND']>30)
    grd_msk = (ais*grd*MSK['AREA']/ 100)

    lsm = (MSK['AIS']<1)
    ground=(MSK['GROUND']*MSK['AIS']>30)
    
    shf=(MSK['ICE']/MSK['ICE']).where((MSK['ICE']>30) & (MSK['GROUND']<50) & (MSK['ROCK']<30) & (ais > 0) )
    shelf=(shf>0)

    x2D, y2D = np.meshgrid(MSK['X'], MSK['Y'])
    sh=MSK['SH']
    
    dh =(MSK['X'].values[0]-MSK['X'].values[1])/2.
    return MSK,lsm,ground,shelf,x2D,y2D,sh,dh

def grat_extend():
    MARbig = xr.open_dataset('data/MARcst-AN35km-201x171.cdf'
                       ,decode_times=False,engine="netcdf4")
    x2Db, y2Db = np.meshgrid(MARbig['x'], MARbig['y'])
    lonbig=MARbig['LON']
    latbig=MARbig['LAT']
    return x2Db, y2Db,lonbig,latbig

def plot_graticules():
    import numpy.ma as ma # masked array
    x2D,y2D,lon2D,lat2D=grat_extend()
    lon2D_out = ma.array(lon2D,mask=np.where((lat2D>=-80.01)&(lon2D>-170),0,1))
    plt.contour(x2D,y2D,lat2D,np.linspace(-80,-50,4),linewidths=0.5,linestyles='--',colors='gray')
    plt.contour(x2D,y2D,lon2D_out,np.linspace(-135,180,8),linewidths=0.5,linestyles='--',colors='gray')


def norm_cmap(cmap,bounds):
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return norm

def def_cmaps():
    cmap_Puor = plt.get_cmap('PuOr')
    cmap_seq = plt.get_cmap ('plasma')#('viridis')
    cmap_smb = plt.get_cmap('YlGnBu')
    cmap_qual = plt.get_cmap('tab10')
    cmap_div = plt.get_cmap('BrBG') # RdBu_r
    cmap_BuRd = plt.get_cmap('RdBu_r')
    obs_color = plt.get_cmap('plasma')(0.45)
    sh_color = plt.get_cmap('gray')(0.5)
    return cmap_seq, cmap_smb, cmap_div, cmap_qual, cmap_BuRd, obs_color, sh_color,cmap_Puor
sh_color = plt.get_cmap('gray')(0.5)


def plot_cbaronly(ax,cmap,bounds,extend='both'):
    axb1 = ax
    if extend=='both':
        ticks = bounds[1:-1]
    elif extend=='max':
        ticks = bounds[0:-1]
    elif extend=='min':
        ticks = bounds[1:]
    elif extend=='neither':
        ticks = bounds[:]
    else:
        print ('Error in plot_cbar, extend keyword')
        return
    cbar= mpl.colorbar.ColorbarBase(axb1,cmap=cmap,norm=norm_cmap(cmap,bounds),ticks=ticks,extend=extend)

    ticklabs = cbar.ax1.get_yticklabels()
    cbar.ax1.set_yticklabels(ticklabs, fontsize=20)

def plot_cbar(gs,cmap,bounds,extend='both'):
    axb1 = plt.subplot(gs[1])
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
    mpl.colorbar.ColorbarBase(axb1,cmap=cmap,norm=norm_cmap(cmap,bounds),ticks=ticks,extend=extend)

def top_label(label):
    text(0,2900,label,ha='center',va='top',fontsize='large')

def set_fig(dr=0):
    fw=4.72441 # 12 cm in inches 
    left, right, top, bottom = 0.1, 0.3+dr, 0.3, 0.3
    wspace = 0#0.1
    wcbar = 0#.15
    w = (fw-wspace-wcbar-left-right)
    fh = w+top+bottom
    fig = plt.figure(figsize=(fw,fh))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.,wcbar/w])
    gs.update(left=left/fw,right=1.-right/fw,bottom=bottom/fh,top=1.-top/fh,wspace=wspace/(fw/2))
    plt.clf()
    plt. subplot(gs[0])
    return fig, gs

def set_fig_or(dr=0.3):
    fw=4.72441 # 12 cm in inches 
    left, right, top, bottom = 0.1, 0.3+dr, 0.3, 0.1
    wspace = 0.1
    wcbar = 0.15
    w = (fw-wspace-wcbar-left-right)
    fh = w+top+bottom
    fig = plt.figure(figsize=(fw,fh))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.,wcbar/w])
    gs.update(left=left/fw,right=1.-right/fw,bottom=bottom/fh,top=1.-top/fh,wspace=wspace/(fw/2))
    plt.clf()
    plt. subplot(gs[0])
    return fig, gs

def set_fig2(dr=0):
    fw=4.72441*2 # 12 cm in inches 
    left, right, top, bottom = 0.1, 0.6+dr, 0.3, 0.3
    wspace = 0.1*2
    wcbar = 0.15*2
    w = (fw-wspace-wcbar-left-right)
    fh = 8.748
    fig = plt.figure(figsize=(fw,fh))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.,wcbar/w])
    gs.update(left=left/fw,right=1.-right/fw,bottom=bottom/fh,top=1.-top/fh,wspace=wspace/(fw/2))
    plt.clf()
    plt. subplot(gs[0])
    return fig, gs


def spcolor(x2D,y2D,var2D,cmap,bounds,dh,sig):

    if sig=='sig':
        plt.pcolormesh(x2D+dh, y2D+dh,var2D,cmap=cmap,norm=norm_cmap(cmap,bounds))
    else:
        grey1='#BEBEBE'
        plt.pcolor(x2D+dh, y2D+dh,var2D,cmap=cmap,hatch='////',edgecolor=grey1,linewidth=0.0,norm=norm_cmap(cmap,bounds))
#    axisminmax(axis_minmax)


def layout(x2D,y2D,sh2D,lsm2D,ground2D,square=False,ground=True,scale=True):
    sh_color = plt.get_cmap('gray')(0.5)
    
    if ground:
        plt.contour(x2D,y2D,ground2D,[0.5],linewidths=0.45,colors='k')
    plt.contour(x2D,y2D,lsm2D,[0.5],linewidths=0.5,colors='k')
    plt.contour(x2D,y2D,sh2D,range(1000,4200,1000),colors=sh_color,linewidths=0.5)

    if square:
        plt.plot([-2500,-1500],[-2500,-2500],lw=1,color='k')
        plt.plot([-2500,-2500],[-2500,-1500],lw=1,color='k')
        plt.plot([-2500,-1500],[-1500,-1500],lw=1,color='k')
        plt.plot([-1500,-1500],[-2500,-1500],lw=1,color='k')
    if scale:
        plt.plot([-1000,0],[-1600,-1600],lw=3,color='k')
        plt.text(-500,-1700,'1000 km',va='top',ha='center')
        
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    
    plt.axis([xmin,xmax,ymin,ymax])




#
#cmap= plt.get_cmap('YlGnBu')
#
#smb=ma.array(AC3SMB,mask=lsm2D)
#bounds=[-20,0,20,50,100,200,400,800,1600,3200]
#plt.pcolormesh(x2D+dh, y2D+dh,smb,cmap=cmap,norm=norm_cmap(cmap,bounds))
#plt.axis([xmin,xmax,ymin,ymax])
#
#
#
##fig.colorbar(c, ax=ax)
#
#
##cbar0.ax.set_xlabel('SMB [kg m$^{-2}$ yr$^{-1}$]',fontsize=8)
#
#
##A fonctionnaliser
#
##Create the fucking lon lat contourplot with non squarred MAR domain
#MARbig = xr.open_dataset('data/MARcst-AN35km-201x171.cdf'
#                       ,decode_times=False)
#x2Db, y2Db = np.meshgrid(MARbig['x'], MARbig['y'])
#
#def layout(x2D,y2D,sh2D,lsm2D,ground2D,square=False,ground=True,scale=True):
#    sh_color = plt.get_cmap('gray')(0.5)
#    
#    if ground:
#        plt.contour(x2D,y2D,ground2D,[0.5],linewidths=0.45,colors='k')
#    plt.contour(x2D,y2D,lsm2D,[0.5],linewidths=0.5,colors='k')
#    plt.contour(x2D,y2D,sh2D,range(1000,4200,1000),colors=sh_color,linewidths=0.5)
#
#    if square:
#        plt.plot([-2500,-1500],[-2500,-2500],lw=1,color='k')
#        plt.plot([-2500,-2500],[-2500,-1500],lw=1,color='k')
#        plt.plot([-2500,-1500],[-1500,-1500],lw=1,color='k')
#        plt.plot([-1500,-1500],[-2500,-1500],lw=1,color='k')
#    if scale:
#        plt.plot([-1000,0],[-1600,-1600],lw=3,color='k')
#        plt.text(-500,-1700,'1000 km',va='top',ha='center')
#        
#    plt.gca().set_xticklabels([])
#    plt.gca().set_yticklabels([])
#    
###    plt.set_xticklabels([])
###    plt.set_yticklabels([])
#
#
#layout(x2D,y2D,sh,lsm2D,ground2D)
#
#
#
##plt.set_ylim(ymin,ymax)
##plt.set_xlim(xmin,xmax)
##
#def plot_graticules(x2D,y2D,lon2D,lat2D):
#    import numpy.ma as ma # masked array
#    lon2D_out = ma.array(lon2D,mask=np.where((lat2D>=-80.01)&(lon2D>-170),0,1))
#    plt.contour(x2D,y2D,lat2D,np.linspace(-80,-50,4),linewidths=0.5,linestyles='--',colors='gray')
#    plt.contour(x2D,y2D,lon2D_out,np.linspace(-135,180,8),linewidths=0.5,linestyles='--',colors='gray')
#
#plot_graticules(x2Db, y2Db,MARbig['LON'],MARbig['LAT'])
#
#plt.text(0,3100,"test",ha='center',va='top',fontsize='large')
#
#plot_cbar(gs,cmap,bounds,extend='both')
#
#fig.savefig('./testpipi.png',
#            format='PNG', dpi=500)
