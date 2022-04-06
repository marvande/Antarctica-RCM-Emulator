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

from process_pangeo import *
from processRCM import *
from GC_scripts import *

import pyproj
from pyproj import Transformer
from google.cloud import storage

def print_raster(raster):
    print(
        f"shape: {raster.rio.shape}\n"
        f"resolution: {raster.rio.resolution()}\n"
        f"bounds: {raster.rio.bounds()}\n"
        f"CRS: {raster.rio.crs}\n"
    )
    
def printMaxMin_XY(varx, vary, unit = 'km'):
    print('Max and min values of x: {0:.2f}{4} -> {1:.2f}{4} and y: {2:.2f}{4} -> {3:.2f}{4}'.format(varx.min().data, 
                                                                     varx.max().data, 
                                                                     vary.min().data, 
                                                                     vary.max().data, unit))
    
def printMaxMin_LatLon(varlat, varlon, unit = '°'):
    print('Max and min values of lat: {0:.2f}{4} -> {1:.2f}{4} and lon: {2:.2f}{4} -> {3:.2f}{4}'.format(varlat.min().data, 
                                                                     varlat.max().data, 
                                                                     varlon.min().data, 
                                                                     varlon.max().data, unit))
    
def printShape(var, dim1, dim2, dim3, ds = 'GCM'):
    print('Shape of {6} data: {3}: {0}, {4}: {1}, {5}: {2}\n-----------------'.format(var.shape[0],
                                                                 var.shape[1],
                                                                 var.shape[2], dim1, dim2, dim3, ds))

def create_downs_RCMgrid():
    # Load RCM for one variable (geographical coordinates are of importance not time or var)
    
    VAR = 'CC'
    pathGC, fileGC = pathToFiles(VAR, date1='19800101', date2='19801231')
    downloadFileFromGC(pathGC, '', fileGC)
    dsr = BasicPreprocRCM(ProcessRCMVar(VAR, xr.open_dataset(fileGC)), kmToM = False)
    
    CC = dsr.CC
    
    # Get boundaries of RCM grid
    x_lower, x_upper = CC.x.min().data, CC.x.max().data
    y_lower, y_upper = CC.y.min().data, CC.y.max().data
    
    print('Original RCM grid:\n----------------------')
    printMaxMin_XY(CC.x, CC.y, unit = 'km')
    print_raster(CC)
   
    # Downsample to new resolution
    # original resolution 35km between each x-cell, want to go to 68km 
    x_res_new = 68
    y_res_new = 206

    factor_x = x_res_new/35
    factor_y = y_res_new/35

    # Downsample:
    CC['x'] = CC.x*factor_x
    CC['y'] = CC.y*factor_y

    print('Downsample to new resolution:\n----------------------')
    printMaxMin_XY(CC.x, CC.y, unit = 'km')
    print_raster(CC)
    
    print('Cut so that on original x,y bounds:\n--------------------------')
    # Restrict new upsampled grid so that it lies in the original max and min bounds
    cut_X = CC.x[(CC.x<=x_upper)&(CC.x>=x_lower)]
    cut_Y = CC.y[(CC.y<=y_upper)&(CC.y>=y_lower)]

    printMaxMin_XY(cut_X, cut_Y, unit = 'km')
    
    # and change to meters
    gridx = cut_X*1000
    gridy = cut_Y*1000

    print('New grid shape (gridx, gridy):', gridx.shape, gridy.shape)    
    return gridx, gridy


def InterpolateGCM(GCM, RCM_xy, gridx, gridy, left = False):
    GCM_transf = GCM.copy()
    
    if left:
        # ----------- Transform GCM ----------
        # Change all longitude coordinates to map from -180° -> 180°
        GCM_transf['lon'] = GCM_transf.lon-180
        print('After transf')
        printMaxMin_LatLon(GCM_transf.lat, GCM_transf.lon, unit = '°')
       
    # ----------- Set information ----------
    # Define the source and target projections
    target_crs = pyproj.CRS(3031) # Global lat-lon coordinate system
    source_crs = pyproj.CRS(4326) # Coordinate system of the file

    # GCM -epsg:4326
    GCM_latlon = GCM_transf.rio.write_crs(source_crs)
    GCM_latlon.attrs['crs']  = source_crs

    # RCM - epsg:3031
    printShape(GCM_latlon.tas, 'time', 'lat', 'lon', 'GCM')
    printShape(RCM_xy.CC, 'time', 'y', 'x', 'RCM')
    
    # ----------- Create new coordinates ----------
    # Create a grid of x/y values from RCM (onto which we want to project)
    xmesh, ymesh = np.meshgrid(gridx, gridy)
    print('Mesh shapes: {0}, {1}'.format(xmesh.shape, 
                                         ymesh.shape))

    # Create a pyproj.Transformer to transform each point 
    # in xmesh and ymesh into a location in the polar coordinates 
    polar_to_latlon = pyproj.Transformer.from_crs(target_crs, 
                                                  source_crs, 
                                                  always_xy = True)

    # polar coordinates from lat-lon
    lon_om2, lat_om2 = polar_to_latlon.transform(xmesh, ymesh)

    # Create xarray.DataArray for the coordinates with matching dimensions.
    lon_om2 = xr.DataArray(lon_om2, dims=('y','x'))
    lat_om2 = xr.DataArray(lat_om2, dims=('y','x'))
    
    print('Lat-lon mesh:\n-----------------')
    printMaxMin_LatLon(lat_om2, lon_om2, unit = '°')
    
    if left:
        lon_om2 = lon_om2+180
        print('After transf:\n-----------------')
        printMaxMin_LatLon(lat_om2, lon_om2, unit = '°')
        
    # ----------- Interpolate ----------
    # Use the xarray interp method to find the nearest locations 
    # for each transformed point 
    print('\nInterpolate:')
    print('\nGCM before interpolation:\n-----------------')
    printMaxMin_LatLon(GCM_latlon.lat, GCM_latlon.lon, unit = '°')
          
    GCM_latlon_int = GCM_latlon.interp({'lon':lon_om2, 'lat':lat_om2}, method='nearest').load()
    print('\nGCM after interpolation\n-----------------')
    printMaxMin_LatLon(GCM_latlon_int.lat, GCM_latlon_int.lon, unit = '°')
    
    print('\nReplace coordinates by new coordinates:')
    # Replace coordinates by new coordinates
    GCM_xy_right = GCM_latlon_int.assign_coords({'x': (('x'), gridx.data), 
                                       'y': (('y'), gridy.data)}).drop_vars(['lon','lat']).reset_coords()
    return GCM_xy_right