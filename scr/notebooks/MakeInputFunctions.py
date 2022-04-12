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
from scipy import ndimage
from math import cos,sin,pi
from GC_scripts import *

# Daily spatial means:
# NOTE: Should either do mean and std of unstandardized or standardized files
# 2D inputs on the shape [time,lon,lat, variables]
# in our case: [time,x,y, variables]

def SpatialStdMean(DATASET):
    SpatialMean = DATASET.mean(dim=["x", "y"])
    SpatialSTD = DATASET.std(dim=["x", "y"])
    
    time =  DATASET.dims['time']
    x =  DATASET.dims['x']
    y =  DATASET.dims['y']
    numVar = len(list(DATASET.data_vars))

    SpatialMean = SpatialMean.to_dataframe().drop(['spatial_ref','PLEV'], axis = 1).values
    SpatialSTD = SpatialSTD.to_dataframe().drop(['spatial_ref','PLEV'], axis = 1).values
    
    SpatialMean = np.array(SpatialMean).reshape(time,1,1,numVar)
    SpatialSTD = np.array(SpatialSTD).reshape(time,1,1,numVar)

    return SpatialMean, SpatialSTD

def cosSinEncoding(DATASET):
    time = DATASET.dims['time']
    months = 12
    cosvect = np.tile([cos(2*i*pi/months)  for i in range(months)],int(time/months)) 
    sinvect = np.tile([sin(2*i*pi/months)  for i in range(months)],int(time/months)) 

    cosvect = cosvect.reshape(time,1,1,1)
    sinvect = sinvect.reshape(time,1,1,1)
    
    return cosvect, sinvect

def create1DInput(DATASET, 
                  seas, # put a cos,sin vector to control the season, format : bool
                  means,   # add the mean of the variables raw or stdz, format : r,s,n
                  stds):    # add the std of the variables raw or stdz, format : r,s,n                   
                 
    INPUT_1D = []
    
    # Remove target variable from DATASET:
    DATASET = DATASET.drop('SMB')

    # Create spatial and temporal mean and append to 1D input:
    # for now only one format
    if means and stds:
        SpatialMean, SpatialSTD = SpatialStdMean(DATASET)
        
        print(f'SpatialMean/std shape: {SpatialMean.shape}')
        INPUT_1D.append(SpatialMean)
        INPUT_1D.append(SpatialSTD)

    # seasonal encoding
    if seas :
        costvect, sinvect = cosSinEncoding(DATASET)
        INPUT_1D.append(costvect)
        INPUT_1D.append(sinvect)
        print(f'Cos/sin encoding shape: {costvect.shape}')
        
    INPUT_1D_ARRAY= np.concatenate(INPUT_1D, axis=3)
    print(f'INPUT_1D shape: {INPUT_1D_ARRAY.shape}')
    
    return INPUT_1D_ARRAY


def standardize(data):
    import numpy as np
    mean =  np.nanmean(data,axis=(1,2), keepdims=True)
    sd   =  np.nanstd(data,axis=(1,2), keepdims=True)
    ndata = (data - mean)/sd
    return (ndata)

# Read GCM like RCM:

fileGC = 'MAR(ACCESS1-3)-stereographic_monthly_GCM_like.nc'
pathGC = f'Chris_data/RawData/MAR-ACCESS1.3/monthly/'

def create2DInput(DATASET,
                  stand):              
    ''' 
        MAKE THE 2D INPUT ARRAY
        SHAPE [nbmonths, x, y, nb_vars]
    '''
    
    # Remove target variable from DATASET:
    DATASET = DATASET.drop(['SMB'])
    
    nbmonths = DATASET.dims['time']
    x = DATASET.dims['x']
    y = DATASET.dims['y']
    nb_vars = len(list(DATASET.data_vars))
    var_list = list(DATASET.data_vars)

    print(f'Number of variables: {nb_vars}')
    print(f'Variables: {var_list}')
    
    INPUT_2D=np.transpose(np.asarray([DATASET[i].values for i in var_list]),[1,3,2,0])

    print(f'Dataset shape: {DATASET.dims}')
    
    if stand:
        # Standardize:
        INPUT_2D_SDTZ = standardize(INPUT_2D)

        # in their code with aerosols extra stuff but ignore
        INPUT_2D_ARRAY = INPUT_2D_SDTZ
    else:
        INPUT_2D_ARRAY = INPUT_2D
        
    print(f'INPUT_2D shape: {INPUT_2D_ARRAY.shape}')
    
    return INPUT_2D_ARRAY

def input_maker(fileGC,
                pathGC, 
                stand = True,  # standardization   
                seas = True,   # put a cos,sin vector to control the season, format : bool
                means = True,   # add the mean of the variables raw or stdz, format : r,s,n
                stds = True):   # add the std of the variables raw or stdz, format : r,s,n                   
        
    downloadFileFromGC(pathGC, '', fileGC)
    DATASET = xr.open_dataset(fileGC)
    os.remove(fileGC)
    
    ''' 
        MAKE THE 2D INPUT ARRAY
        SHAPE [nbmonths, x, y, nb_vars]
    '''
    print('Creating 2D input X:\n -------------------')
    INPUT_2D_ARRAY = create2DInput(DATASET, stand)
    
    '''
    MAKE THE 1D INPUT ARRAY
    CONTAINS MEANS, STD SEASON IF ASKED
    '''
    print('Creating 1D input Z:\n -------------------')
    INPUT_1D_ARRAY = create1DInput(DATASET, seas, means, stds)
    
    DATASET.close()
    
    return INPUT_2D_ARRAY, INPUT_1D_ARRAY