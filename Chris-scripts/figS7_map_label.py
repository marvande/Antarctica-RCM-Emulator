#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:39:30 2020

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

# =============================================================================
# Definition of ICE MASK ; used color maps, 
# =============================================================================
MSK,lsm,ground,x2D,y2D,sh,dh = shade.ice_masking()
cmap_seq, cmap_smb, cmap_div, cmap_qual, cmap_BuRd, obs_color, sh_color, cmap_Puor = shade.def_cmaps()

def spcolor(var2D,cmap,bounds,sig):
    return shade.spcolor(x2D,y2D,var2D,cmap,bounds,dh,sig)

xmin=-2800
ymin=xmin
xmax=2800
ymax=xmax

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
        plt.plot([-2700,-1700],[-2500,-2500],lw=3,color='k')
        plt.text(-2225,-2600,'1000 km',va='top',ha='center')
        
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    
    plt.axis([xmin,xmax,ymin,ymax])


ice= MSK['ICE'].where(MSK['ICE']>30)
rig=MSK['RIGNOT']

shelf=(ice/ice).where( (ice>30) & (ground < 0.5) )

ais = MSK['AIS'].where(MSK['AIS'] >0)
rig2=rig*ais/rig
shelf=shelf*ais



peninsula = rig2.where( (rig==15) | (rig==16) | (rig==17) | (rig==15) | (rig==25)  | (rig==25.5) | (rig==26) | (rig==26.5) )
east_ais  = rig2.where( (rig==1) | (rig==2) | (rig==9) | (rig==14) | (rig==18)  | (rig==24) | (rig==24.5) | (rig==19) )
west_ais_a= rig2.where( (rig==3) | (rig==4) | (rig==5) | (rig==10) | (rig==12)  | (rig==23)  | (rig==23.5) | (rig==6) | (rig==7) | (rig==8) | (rig==11) | (rig==13)  | (rig==21) | (rig==22) | (rig==21.5) | (rig==22) | (rig==20) )
ross      = rig2.where( (rig==19) | (rig==19.5) )
ronne     = rig2.where( (rig==20) | (rig==20.5 )) 
nonais    = rig2.where( (rig>26) )


rig3=(rig2*1).where(peninsula>0)
rig3=(rig2*2).where(east_ais>0,rig3,0)
rig3=(rig2*3).where(west_ais_a>0,rig3,0)
rig3=(rig2*5).where(ross>0,rig3,0)
rig3=(rig2*6).where(ronne>0,rig3,0)
rig3=(rig2*7).where(nonais>0,rig3,0)


grey1='#BEBEBE'
cmap=cmap_qual
fig, gs = shade.set_fig() 

vmin=-1000
vmax=380



plt.pcolormesh(x2D+dh,y2D+dh,peninsula,cmap=plt.get_cmap('Purples_r'),vmin=vmin,vmax=vmax)
plt.pcolormesh(x2D+dh,y2D+dh,east_ais,cmap=plt.get_cmap('Greens_r'),vmin=vmin,vmax=vmax)
plt.pcolormesh(x2D+dh,y2D+dh,west_ais_a,cmap=plt.get_cmap('Oranges_r'),vmin=vmin,vmax=vmax)
plt.pcolormesh(x2D+dh,y2D+dh,shelf,cmap=plt.get_cmap('Blues_r'),vmin=vmin,vmax=vmax)
plt.pcolormesh(x2D-dh,y2D-dh,nonais,cmap=plt.get_cmap('Reds_r'),vmin=vmin,vmax=vmax)

#plt.pcolormesh(x2D+dh, y2D+dh,rig3,cmap=cmap)
#plt.pcolor(x2D+dh, y2D+dh,rig3*shelf,cmap=cmap,hatch='////',edgecolor=grey1,linewidth=0.2)

shade.plot_graticules()
layout(x2D,y2D,sh,lsm,ground)


plt.text(-300,-1350,"Ross\nice shelf",fontsize=11,ha='center',va='top')
plt.text(-1300,1000,"Filchner-\nRonne\nice shelf",fontsize=11,ha='center',va='bottom')
plt.text(-700,-900,"Marie\nByrd\nLand",fontsize=11,ha='center',va='bottom')
plt.text(-1450,-400,"Ellsworth\nLand",fontsize=11,ha='center',va='bottom')
plt.text(1950,-600,"Queen Mary\nLand",fontsize=11,ha='center',va='bottom')
plt.text(810,-2090,"George V\nLand",fontsize=11,ha='center',va='bottom')
plt.text(1600,-1800,"Adelie\nLand",fontsize=11,ha='center',va='bottom')
plt.text(1950,-1150,"Wilkes\nLand",fontsize=11,ha='center',va='bottom')
plt.text(2450,500,"Amery\nice\nshelf",fontsize=11,ha='center',va='bottom')
plt.text(300,1400,"Queen Maud\nLand",fontsize=11,ha='center',va='bottom')

cmap1 = mpl.cm.get_cmap('Oranges_r')
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
plt.text(1400,2000,"East AIS",fontsize=14,fontdict={'weight': 'bold'},color=cmap1(norm(1)-0.2))


cmap1 = mpl.cm.get_cmap('Greens_r')
plt.text(-2600,-1600,"West AIS",fontsize=14,fontdict={'weight': 'bold'},color=cmap1(norm(1)-0.2))
cmap1 = mpl.cm.get_cmap('Purples_r')
plt.text(-2600,1900,"Peninsula",fontsize=13,fontdict={'weight': 'bold'},color=cmap1(norm(1)-0.2))

fig.savefig('./fig/label_map.png',format='PNG', dpi=500)