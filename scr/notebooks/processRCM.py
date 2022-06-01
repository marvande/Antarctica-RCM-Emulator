import os
from matplotlib import pyplot as plt
import matplotlib.path as mpath
import numpy as np
import pandas as pd
import xarray as xr
import cartopy
import cf_units
from datetime import datetime
from datetime import timedelta
import rasterio
import cartopy.crs as ccrs
import gcsfs
from tqdm import tqdm
import pyproj
from pyproj import Transformer
from google.cloud import storage
from re import search
from os import listdir
from os.path import isfile, join


def KmtoM(RCM_xy):
    # Convert RCM from km to m for plots:
    RCM_xy['x'] = RCM_xy.x*1000
    RCM_xy['y'] = RCM_xy.y*1000
    RCM_xy.x.attrs['units'] = 'm'
    RCM_xy.y.attrs['units'] = 'm'
    return RCM_xy

def BasicPreprocRCM(RCMxy, kmToM = True):
    RCMxy = RCMxy.rename({'X':'x', 'Y':'y','TIME':'time'})

    # indicate projection crs
    RCMxy = RCMxy.rio.write_crs("epsg:3031")

    # change km to m
    if kmToM:
        RCMxy = KmtoM(RCMxy)
    return RCMxy

def resampleMonthlyMean(array):
    Monthly = array.resample(time='1M').mean()
    # keep attrs info, ow lost
    Monthly.attrs = array.attrs
    for var in list(Monthly.keys()):
        Monthly[var].attrs = array[var].attrs
    return Monthly


def ProcessRCMVar(var, RCM):
    if var == 'TT':
        # drop atmlay coordinate as singular
        dim = ('TIME', 'Y', 'X')
        RCM['TT'] = xr.Variable(dims = dim, 
                                    data = RCM.TT.mean(dim='ATMLAY'), 
                                    attrs = RCM.TT.attrs)
        RCM = RCM.drop_dims(['ATMLAY'])
        RCM = RCM.drop(['TIME_bnds'])
        return RCM
    if var == 'UUP':
        RCM = RCM.drop(['TIME_bnds','PLEV_bnds'])
        return RCM
    if var == 'VVP':
        RCM = RCM.drop(['TIME_bnds','PLEV_bnds'])
        return RCM
    if var == 'SMB':
        # drop SECTOR coordinate as singular
        dim = ('TIME', 'Y', 'X')
        RCM['SMB'] = xr.Variable(dims = dim, 
                                    data = RCM.SMB.mean(dim='SECTOR'), 
                                    attrs = RCM.SMB.attrs)
        RCM = RCM.drop_dims(['SECTOR'])
        RCM = RCM.drop(['TIME_bnds'])
        return RCM
    else:
        return RCM.drop(['TIME_bnds'])