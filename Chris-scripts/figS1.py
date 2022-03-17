#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:24:10 2019

@author: ckittel

Anomalies sur le présent et changement 2D
Supplementary?
"""



import os
#import conda

#conda_file_dir = conda.__file__
#conda_dir = conda_file_dir.split('lib')[0]
#proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
#os.environ["PROJ_LIB"] = proj_lib

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

import numpy as np



from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from sklearn.metrics import mean_squared_error
from math import sqrt
from math import exp
from math import pi

from scipy import stats
from matplotlib import ticker as mticker




import matplotlib.colors

import numpy.ma as ma
import matplotlib.patches as mpatches

from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from matplotlib.cm import get_cmap
from operator import add

from matplotlib.collections import LineCollection
import matplotlib.transforms
from matplotlib.patches import Rectangle
from scipy.sparse import diags

import matplotlib.dates as mdates
import numpy as np # array module
import numpy.ma as ma # masked array
from pylab import *
from time import * # module to measure time
import csv
from os.path import exists
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.transforms import Bbox
from matplotlib.path import Path
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import operator
import array
import pandas as pd # Python Data Analysis Library
import scipy

import shade


    
##***** Read datasets *****
xb=0 #equiv j dans ferret
xe=148
yb=0 #equiv i dans ferret
ye=176

dh=17.5*1000

data1=Dataset('MARcst-AN35km-176x148.cdf')
x1D=data1.variables['X']
y1D=data1.variables['Y']
sh=data1.variables['SH'][xb:xe,yb:ye]
lon=data1.variables['LON'][xb:xe,yb:ye]
lat=data1.variables['LAT'][xb:xe,yb:ye]
ais=data1.variables['AIS'][xb:xe,yb:ye]
ground2D=data1.variables['GROUND'][xb:xe,yb:ye]

#data1.close()

x2D, y2D = scipy.meshgrid(x1D,y1D)
lsm2D = (ais<0.5)
lsm = invert(lsm2D)

dh = (x1D[1]-x1D[0])/2.

sh2D = ma.array(sh,mask=lsm2D)
sh=sh2D

data2=Dataset('result.nc')
smbref=data2.variables['SMBREF'][xb:xe,yb:ye]
std=data2.variables['STD'][xb:xe,yb:ye]
anonor=data2.variables['ANONOR'][xb:xe,yb:ye]
anoac3=data2.variables['ANOAC3'][xb:xe,yb:ye]
chanor=data2.variables['CHANOR'][xb:xe,yb:ye]
chaac3=data2.variables['CHAAC3'][xb:xe,yb:ye]
stdnor=data2.variables['STDNOR'][xb:xe,yb:ye]
stdac3=data2.variables['STDAC3'][xb:xe,yb:ye]

data2.close()
del xb,yb,xe,ye

anos=ma.masked_where((abs(anoac3>std)),anoac3)

# shade results #
# ------------- #
'''
define the shade frame for squared Antarctica
'''
def layout(scale=True,axis_minmax=True):
    return shade.layout(x2D,y2D,lon,lat,sh,lsm2D,ground2D,scale=scale,axis_minmax=axis_minmax)
def scontour(var2D,levels,lw=1.,zorder=10,lc='k',ls='-'):
    return shade.scontour(x2D,y2D,var2D,levels,lw=lw,zorder=zorder,lc=lc,ls=ls)
def spcolor(var2D,cmap,bounds,sig):
    return shade.spcolor(x2D,y2D,var2D,cmap,bounds,sig)
def sshade_under(var2D,level,cmap=plt.get_cmap('plasma_r'),lc='fuchsia'):
    return shade.sshade_under(x2D,y2D,var2D,level,cmap=cmap,lc=lc)
def bottom_label(label,lc=None,lw=None,xx=-1700,yy=-2500,area=False,ls='-'):
    return shade.bottom_label(x2D,y2D,label,lc=lc,lw=lw,xx=xx,yy=yy,area=area,ls=ls)
cmap_seq, cmap_smb, cmap_div, cmap_qual, cmap_BuRd, obs_color, sh_color = shade.def_cmaps()


def fig1a(cmap=cmap_smb):
    mask = lsm2D 
    bounds=[-20,0,20,50,100,200,400,800,1600,3200]
    fig, gs = shade.set_fig()
    smb=ma.array(smbref,mask=lsm2D)
    spcolor(smb,cmap,bounds,'sig')
    shade.top_label('Mean of MAR annual SMB (kg m$^{-2}$ yr$^{-1}$)')
#    shade.subfig_label('(a)')
    layout()
    shade.plot_cbar(gs,cmap,bounds,extend='both')
    savefig('mar_ref.png',dpi=600)
    
def fig1b(cmap=cmap_div,ano2D=anonor):
    mask = lsm2D 
    bounds=[-2000,-400,-200,-100,-50,-20,-10,0,10,20,50,100,200,400,2000]
    fig, gs = shade.set_fig()
    
    anos=ma.masked_where((abs(ano2D)<std),ano2D) #que ce qui est sig
    smb=ma.array(anos,mask=lsm2D)
    spcolor(smb,cmap_div,bounds,'sig')
    
    ano=ma.masked_where((abs(ano2D)>std),ano2D)
    smb=ma.array(ano,mask=lsm2D)
    spcolor(smb,cmap_div,bounds,'nosig')
    shade.top_label('Mean of MAR annual SMB (kg m$^{-2}$ yr$^{-1}$)')
    
    shade.top_label('Mean of MAR annual SMB (kg m$^{-2}$ yr$^{-1}$)')

    layout()
    shade.plot_cbar(gs,cmap,bounds,extend='both')
    savefig('anonor.png',dpi=600)
    
def fig1c(cmap=cmap_div,ano2D=anoac3):
    mask = lsm2D 
    bounds=[-2000,-400,-200,-100,-50,-20,-10,0,10,20,50,100,200,400,2000]
    fig, gs = shade.set_fig()
    
    anos=ma.masked_where((abs(ano2D)<std),ano2D) #que ce qui est sig
    smb=ma.array(anos,mask=lsm2D)
    spcolor(smb,cmap_div,bounds,'sig')
    
    
    ano=ma.masked_where((abs(ano2D)>std),ano2D)
    smb=ma.array(ano,mask=lsm2D)
    spcolor(smb,cmap_div,bounds,'nosig')
    shade.top_label('Mean of MAR annual SMB (kg m$^{-2}$ yr$^{-1}$)')

    layout()
    shade.plot_cbar(gs,cmap,bounds,extend='both')
    savefig('anoac3.png',dpi=600)

def fig2a(cmap=cmap_BuRd,ano2D=chanor,std2D=stdnor):
    mask = lsm2D 
    bounds=[-2000,-800,-400,-200,-100,-50,-20,-10,0,10,20,50,100,200,400,800,2000]
    fig, gs = shade.set_fig()
    
    anos=ma.masked_where((abs(ano2D)<std2D),ano2D) #que ce qui est sig
    smb=ma.array(anos,mask=lsm2D)
    spcolor(smb,cmap,bounds,'sig')
    
    
    ano=ma.masked_where((abs(ano2D)>std2D),ano2D)
    smb=ma.array(ano,mask=lsm2D)
    spcolor(smb,cmap,bounds,'nosig')
    shade.top_label('Mean of MAR annual SMB (kg m$^{-2}$ yr$^{-1}$)')

    layout()
    shade.plot_cbar(gs,cmap,bounds,extend='both')
    savefig('chanor.png',dpi=600)

def fig2b(cmap=cmap_BuRd,ano2D=chaac3,std2D=stdac3):
    mask = lsm2D 
    bounds=[-2000,-800,-400,-200,-100,-50,-20,-10,0,10,20,50,100,200,400,800,2000]
    fig, gs = shade.set_fig()
    
    anos=ma.masked_where((abs(ano2D)<std2D),ano2D) #que ce qui est sig
    smb=ma.array(anos,mask=lsm2D)
    spcolor(smb,cmap,bounds,'sig')
    
    
    ano=ma.masked_where((abs(ano2D)>std2D),ano2D)
    smb=ma.array(ano,mask=lsm2D)
    spcolor(smb,cmap,bounds,'nosig')
    shade.top_label('Mean of MAR annual SMB (kg m$^{-2}$ yr$^{-1}$)')

    layout()
    shade.plot_cbar(gs,cmap,bounds,extend='both')
    savefig('chac3.png',dpi=600)

data1.close()
