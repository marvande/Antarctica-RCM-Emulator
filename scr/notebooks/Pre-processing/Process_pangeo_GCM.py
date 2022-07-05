# File to process GCM file from the pangeo database

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
import dask
import pyproj
from pyproj import Transformer
from google.cloud import storage

# Connect to Google Cloud Storage
fs = gcsfs.GCSFileSystem(token='anon', access='read_only')

# Read catalogue
df = pd.read_csv("https://cmip6.storage.googleapis.com/pangeo-cmip5.csv")

def createGCM_Pangeo(df,fs):
    # CMIP5
    source_id = 'ACCESS1-3'

    print('Experiments:\n---------------------')
    print(df.query(f"source_id=='{source_id}'").experiment_id.unique())
    print('Member ids:\n---------------------')
    print(df.query(f"source_id=='{source_id}'").member_id.unique())
    ACCESS13 = df.query(f"source_id=='{source_id}'&experiment_id==['historical','rcp85']")

    source_id = 'ACCESS1-3'

    # this experiment has most of the variables
    member_id = 'r1i1p1f1'

    # Historical values
    ACCESS13_hist = df.query(f"source_id=='{source_id}'&experiment_id==['historical'] & member_id == '{member_id}'")
    histVar = ACCESS13_hist.zstore.values

    # RCP85 scenario
    ACCESS13_rcp85 = df.query(f"source_id=='{source_id}'&experiment_id==['rcp85'] & member_id == '{member_id}'")
    rcp85Var = ACCESS13_rcp85.zstore.values

    # Historical values
    # Sea Level Pressure
    psl = xr.open_zarr(fs.get_mapper(histVar[0]), 
                       consolidated=True)
    
    # Outgoing Longwave Radiation
    rlut = xr.open_zarr(fs.get_mapper(histVar[2]), 
                        consolidated=True)
    # Sea Ice Area Fraction
    siconc = xr.open_zarr(fs.get_mapper(histVar[3]), 
                          consolidated=True)
    # Precipitation
    pr = xr.open_zarr(fs.get_mapper(histVar[4]), 
                      consolidated=True)
    # Near-Surface Air Temperature
    tas = xr.open_zarr(fs.get_mapper(histVar[5]), 
                       consolidated=True).drop(['height'])
    # Incident Shortwave Radiation
    rsdt = xr.open_zarr(fs.get_mapper(histVar[6]), 
                        consolidated=True)
        
    # Outgoing Shortwave Radiation
    rsut = xr.open_zarr(fs.get_mapper(histVar[8]), 
                        consolidated=True)
    
    # ---------------
    # these variables have slightly different coordinate system so take lon lat from others
    # Northward Near-Surface Wind
    vas = xr.open_zarr(fs.get_mapper(histVar[7]), 
                       consolidated=True).drop(['height'])
    
    # Eastward Near-Surface Wind
    uas = xr.open_zarr(fs.get_mapper(histVar[1]), 
                       consolidated=True).drop(['height'])
    
    vas = vas.assign_coords({'lat': (('lat'), pr.lat[1:].data), 
                    'lon': (('lon'), pr.lon.data), 
                  'lat_bnds': (('lat', 'bnds'), pr.lat_bnds[1:,].data),
                   'lon_bnds': (('lon', 'bnds'), pr.lon_bnds.data)
                  })
    
    uas = uas.assign_coords({'lat': (('lat'), pr.lat[1:].data), 
                    'lon': (('lon'), pr.lon.data), 
                  'lat_bnds': (('lat', 'bnds'), pr.lat_bnds[1:,].data),
                   'lon_bnds': (('lon', 'bnds'), pr.lon_bnds.data)
                  })

    ACCESS13_hist = xr.merge([psl, uas, pr, rlut, tas, rsdt, vas, rsut], compat='override')

    # RCP 85 scenario 
    # Sea Level Pressure
    psl = xr.open_zarr(fs.get_mapper(rcp85Var[0]), 
                       consolidated=True)
    
    # Outgoing Longwave Radiation
    rlut = xr.open_zarr(fs.get_mapper(rcp85Var[2]), 
                        consolidated=True)
    # Sea Ice Area Fraction
    siconc = xr.open_zarr(fs.get_mapper(rcp85Var[3]), 
                          consolidated=True)
    # Precipitation
    pr = xr.open_zarr(fs.get_mapper(rcp85Var[4]), 
                      consolidated=True)
    # Near-Surface Air Temperature
    tas = xr.open_zarr(fs.get_mapper(rcp85Var[5]), 
                       consolidated=True).drop(['height'])
    # Incident Shortwave Radiation
    rsdt = xr.open_zarr(fs.get_mapper(rcp85Var[6]), 
                        consolidated=True)
    
    # Outgoing Shortwave Radiation
    rsut = xr.open_zarr(fs.get_mapper(rcp85Var[8]), 
                        consolidated=True)

    # ---------------
    # these variables have slightly different coordinate system so take lon lat from others
    # Northward Near-Surface Wind
    vas = xr.open_zarr(fs.get_mapper(rcp85Var[7]), 
                       consolidated=True).drop(['height'])
    
    # Eastward Near-Surface Wind
    uas = xr.open_zarr(fs.get_mapper(rcp85Var[1]), 
                       consolidated=True).drop(['height'])
    
    vas = vas.assign_coords({'lat': (('lat'), pr.lat[1:].data), 
                    'lon': (('lon'), pr.lon.data), 
                  'lat_bnds': (('lat', 'bnds'), pr.lat_bnds[1:,].data),
                   'lon_bnds': (('lon', 'bnds'), pr.lon_bnds.data)
                  })
    
    uas = uas.assign_coords({'lat': (('lat'), pr.lat[1:].data), 
                    'lon': (('lon'), pr.lon.data), 
                  'lat_bnds': (('lat', 'bnds'), pr.lat_bnds[1:,].data),
                   'lon_bnds': (('lon', 'bnds'), pr.lon_bnds.data)
                  })

    ACCESS13_rcp85 = xr.merge([psl, uas, pr, rlut, tas, rsdt, vas, rsut], compat='override')
    ACCESS13 = xr.concat([ACCESS13_rcp85, ACCESS13_hist], dim = 'time').sortby('time')

    return ACCESS13
    
    