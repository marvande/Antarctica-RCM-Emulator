"""
Functions that handle data and masks
"""


import tensorflow as tf 
import xarray as xr
import cartopy.crs as ccrs
from scipy.stats import wasserstein_distance
import torch
import numpy as np 
from matplotlib import pyplot as plt


"""
plotAllVar: plots all variables in a xr.Dataset
"""
def plotAllVar(
    GCM_xy,  # xr.Dataset
    m: int = 3,  # number of rows in plot
    n: int = 3,  # number of columns in plot
    name: str = "GCM",  # name of dataset plotted, for title
    time: int = 0,
):  # time step that should be plotted
    vars_ = sorted(list(GCM_xy.data_vars))
    coords = list(GCM_xy.coords)
    f = plt.figure(figsize=(20, 10))
    map_proj = ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)
    for i in range(len(vars_)):
        var = vars_[i]
        ax = plt.subplot(m, n, i + 1, projection=ccrs.SouthPolarStereo())
        GCM_xy[var].isel(time=time).plot(
            ax=ax, x="x", y="y", transform=ccrs.SouthPolarStereo(), add_colorbar=True
        )
        ax.coastlines("10m", color="black")
        ax.gridlines()
        ax.set_title(f"{GCM_xy[var].long_name} ({var})")
    plt.suptitle(f"First time step {GCM_xy.time[0].values} of {name}")
    

"""
plotAllVar: plots all variables in a xr.Dataset
"""
def plotAllVar2Xr(
    GCM_xy,  # xr.Dataset
    GCM_xy2,
    m: int = 7,  # number of rows in plot
    n: int = 2,  # number of columns in plot
    name: str = "GCM",  # name of dataset plotted, for title
    time: int = 0,
):  # time step that should be plotted
    vars_ = sorted(list(GCM_xy.data_vars))
    coords = list(GCM_xy.coords)
    f = plt.figure(figsize=(20, 20))
    map_proj = ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)
    
    k = 1
    for i in range(len(vars_)):
        var = vars_[i]
        ax = plt.subplot(m, n, k, projection=ccrs.SouthPolarStereo())
        max = np.max([np.nanmax(GCM_xy[var].isel(time=time).values), np.nanmax(GCM_xy2[var].isel(time=time).values)])
        min = np.min([np.nanmin(GCM_xy[var].isel(time=time).values), np.nanmin(GCM_xy2[var].isel(time=time).values)])
        
        GCM_xy[var].isel(time=time).plot(
            ax=ax, x="x", y="y", vmin = min, vmax = max, transform=ccrs.SouthPolarStereo(), add_colorbar=True
        )
        ax.coastlines("10m", color="black")
        ax.gridlines()
        ax.set_title(f"{GCM_xy[var].long_name} GCMLike ({var})")
        k+=1
        ax = plt.subplot(m, n, k, projection=ccrs.SouthPolarStereo())
        GCM_xy2[var].isel(time=time).plot(
            ax=ax, x="x", y="y", vmin = min, vmax = max, transform=ccrs.SouthPolarStereo(), add_colorbar=True
        )
        ax.coastlines("10m", color="black")
        ax.gridlines()
        ax.set_title(f"{GCM_xy2[var].long_name} GCM ({var})")
        k+=1
    plt.suptitle(f"First time step {GCM_xy.time[0].values} of {name}")
    
"""
resize: resizes an image to another size
"""
def resize(
    df,  # image to be resized
    sizex: int,  # new dimension x
    sizey: int,  # new dimension y
    print_: bool = True):
    # resize to input domain size:
    if print_:
        print("Shape before resizing:", df.shape)
    image = tf.constant(df)
    image_resized = tf.image.resize(image, (sizex, sizey))
    df_resized = image_resized.numpy()
    if print_:
        print("Shape after resizing:", df_resized.shape)
    return df_resized

"""
cutBoundaries: cuts boundaries x,y of an xarray dataset
"""
def cutBoundaries(
    df,  # xarray dataset
    max_x,  # max x coordinate to cut to
    max_y,  # max y coordinate to cut to
    lowerHalf=False):
    df = df.where(df.x < max_x, drop=True)
    df = df.where(-max_x <= df.x, drop=True)
    if lowerHalf:
        df = df.where(df.y < 0, drop=True)
    else:
        df = df.where(df.y < max_y, drop=True)
    df = df.where(-max_y <= df.y, drop=True)
    return df

"""
takeRandomSamples: returns a random sample of the input and target datasets
"""
def takeRandomSamples(
    full_input,  # input dataset
    full_target,  # target dataset
    pred: bool = False,  # True if prediction ds as well
    full_prediction=None,
):  # prediction dataset if included
    # take random time
    randTime = rn.randint(0, len(full_input[0] - 1))
    sample2dtrain = full_input[0][randTime]
    sample1dtrain = full_input[1][randTime]
    sampletarget = full_target[randTime]
    
    if pred:
        samplepred = full_prediction[randTime]
        
        return sample2dtrain, sample1dtrain, sampletarget, samplepred, randTime
    else:
        return sample2dtrain, sample1dtrain, sampletarget, randTime

"""
plotTrain: Plot a training sample for a certain time step
"""
def plotTrain(
    GCMLike,  # GCM dataset
    sample2dtrain,  # 2d training sample
    numVar,  # number of variable to plot (because 7 channels)
    ax,
    time,  # timestep of plot
    list_var,  # list of all variables in GCM
    region="Whole Antarctica", # region
):  
    if region != "Whole Antarctica":
        ds = createLowerInput(GCMLike, region=region, Nx=48, Ny=25, print_=False)
    else:
        ds = GCMLike
        
    VAR = list_var[numVar]
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain[VAR] = xr.Variable(
        dims=("y", "x"), data=sample2dtrain[:, :, numVar], attrs=ds[VAR].attrs
    )
    dftrain[VAR].plot(
        ax=ax,
        x="x",
        transform=ccrs.SouthPolarStereo(),
        add_colorbar=False,
        cmap="RdYlBu_r",
    )
    
    ax.coastlines("10m", color="black")
    ax.gridlines()
    ax.set_title(f"{time} Input: {VAR}")
    

"""
plotTarget: Plot a target sample for a certain time step
"""
def plotTarget(
    target_dataset,  # target RCM dataset
    sampletarget,  # 2d target sample
    ax,
    vmin,  # min value of prediction and target, for shared colorbar
    vmax,  # max value of prediction and target, for shared colorbar
    region="Whole Antarctica",  # region
):
    if region != "Whole Antarctica":
        ds = createLowerTarget(
            target_dataset, region=region, Nx=64, Ny=64, print_=False
        )
    else:
        ds = target_dataset
        
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain["SMB"] = xr.Variable(
        dims=("y", "x"), data=sampletarget[:, :, 0], attrs=ds["SMB"].attrs
    )
    pl = dftrain.SMB.plot(
        ax=ax,
        x="x",
        transform=ccrs.SouthPolarStereo(),
        add_colorbar=False,
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax.coastlines("10m", color="black")
    ax.gridlines()
    ax.set_title(f"Target: SMB, {region}")
    
    return pl
    
    
"""
plotInterp: Plot a interpolated SMB sample for a certain time step
"""
def plotInterp(
    target_dataset,  # target RCM dataset
    sampletarget,  # 2d target sample
    ax,
    vmin,  # min value of prediction and target, for shared colorbar
    vmax,  # max value of prediction and target, for shared colorbar
    region="Whole Antarctica",  # region
):
    if region != "Whole Antarctica":
        ds = createLowerTarget(
            target_dataset, region=region, Nx=64, Ny=64, print_=False
        )
    else:
        ds = target_dataset
        
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain["SMB"] = xr.Variable(
        dims=("y", "x"), data=sampletarget[:, :, 0], attrs=ds["SMB"].attrs
    )
    pl = dftrain.SMB.plot(
        ax=ax,
        x="x",
        transform=ccrs.SouthPolarStereo(),
        add_colorbar=False,
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax.coastlines("10m", color="black")
    ax.gridlines()
    ax.set_title(f"Interpolated SMB, {region}")
    

"""
plotPred: Plot a prediction sample for a certain time step
"""
def plotPred(
    target_dataset,  # target RCM dataset
    samplepred,  # 2d prediction sample
    ax,
    vmin,  # min value of prediction and target, for shared colorbar
    vmax,  # max value of prediction and target, for shared colorbar
    region="Whole Antarctica",  # region
):
    if region != "Whole Antarctica":
        ds = createLowerTarget(
            target_dataset, region=region, Nx=64, Ny=64, print_=False
        )
    else:
        ds = target_dataset
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain["SMB"] = xr.Variable(
        dims=("y", "x"), data=samplepred[:, :, 0], attrs=ds["SMB"].attrs
    )
    dftrain.SMB.plot(
        ax=ax,
        x="x",
        transform=ccrs.SouthPolarStereo(),
        add_colorbar=False,
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax.coastlines("10m", color="black")
    ax.gridlines()
    ax.set_title(f"Prediction: SMB, {region}")
    

"""createLowerTarget: creates a subset of the target domain, cut to new dimensions x and y
"""
def createLowerTarget(
    target_dataset,  # target dataset
    region: str = "Larsen",  # region of interest
    Nx: int = 64,  # new size of x dimension
    Ny: int = 64,  # new size of y dimension
    print_=True,
):
    max_y = Nx * 35 * 1000
    max_x = Ny * 35 * 1000
    
    min_x_res = target_dataset.x.min().values
    min_y_res = target_dataset.y.min().values
    
    max_x_res = target_dataset.x.max().values
    max_y_res = target_dataset.y.max().values
    
    if region == "Larsen":
        df = target_dataset.where(target_dataset.y >= 0, drop=True)
        df = df.where(df.y < max_y, drop=True)
        df = df.where(df.x < min_x_res + max_x, drop=True)
        
    if region == "Lower peninsula":
        max_x = (Nx / 2) * 35 * 1000
        max_y = Ny * 35 * 1000
        df = cutBoundaries(target_dataset, max_x, max_y, lowerHalf=True)
        
    if region == "Amundsen":
        df = target_dataset.where(
            target_dataset.y <= min_y_res + 1.5 * max_y, drop=True
        )
        df = df.where(df.y > min_y_res + 0.5 * max_y, drop=True)
        df = df.where(df.x < min_x_res + max_x, drop=True)
        
    if region == "Wilkes":
        df = target_dataset.where(target_dataset.x > max_x_res - max_x, drop=True)
        df = df.where(df.y <= min_y_res + 1.5 * max_y, drop=True)
        df = df.where(df.y > min_y_res + 0.5 * max_y, drop=True)
        
    if region == "Maud":
        df = target_dataset.where(target_dataset.y > max_y_res - max_y, drop=True)
        df = df.where(df.x >= max_x_res - 1.5 * max_x, drop=True)
        df = df.where(df.x < max_x_res - 0.5 * max_x, drop=True)
    if print_:
        print("New target dimensions:", df.dims)
    return df


"""createLowerInput: creates a subset of the input domain, cut to new dimensions x and y
"""
def createLowerInput(
    GCMLike,
    region: str = "Larsen",  # region of interest
    Nx: int = 48,  # new size of x dimension
    Ny: int = 25,  # new size of y dimension
    print_=True,
):
    if region == "Lower peninsula":
        max_x = (Nx / 2) * 68 * 1000
        max_y = (Ny) * 206 * 1000
        
        df = GCMLike.where(GCMLike.x < max_x, drop=True)
        df = df.where(-max_x <= df.x, drop=True)
        
    if region == "Larsen":
        min_x = GCMLike.x.min().values
        df = GCMLike.where(GCMLike.x < min_x + (Nx * 68 * 1000), drop=True)
        
    if region == "Amundsen":  # same region as Larsen
        min_x = GCMLike.x.min().values
        df = GCMLike.where(GCMLike.x < min_x + (Nx * 68 * 1000), drop=True)
    
    if region == "Wilkes":  # same region as Larsen
        max_x = GCMLike.x.max().values
        df = GCMLike.where(GCMLike.x > max_x - (Nx * 68 * 1000), drop=True)
    
    if region == "Maud":  # same region as lower peninsula
        max_x = (Nx / 2) * 68 * 1000
        max_y = (Ny) * 206 * 1000
    
        df = GCMLike.where(GCMLike.x < max_x, drop=True)
        df = df.where(-max_x <= df.x, drop=True)
    
    if print_:
        print("New dimensions:", df.dims)
        
    return df

# create mask
def createMask(true_smb, onechannel = True):
    arraySMB = np.array(true_smb)
    if onechannel:
        dimx = true_smb.shape[1]
        dimy = true_smb.shape[0]
    mask = np.empty((dimy,dimx))
    for y in range(dimy):
        for x in range(dimx):
            pixelSMB = arraySMB[y,x,0]
            if not np.any(pixelSMB): # check if all zeros (then on sea)
                mask[y,x] = 0
            else:
                mask[y,x] = 1
    return mask