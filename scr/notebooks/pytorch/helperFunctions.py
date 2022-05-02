import tensorflow as tf 
import xarray as xr
import cartopy.crs as ccrs
from scipy.stats import wasserstein_distance
import torch
import numpy as np 

def plotAllVar(GCM_xy, m=3, n=3, name="GCM", time=0):
    vars_ = list(GCM_xy.data_vars)
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
    
    
def resize(df, sizex, sizey, print_=True):
    # resize to input domain size:
    if print_:
        print("Shape before resizing:", df.shape)
    image = tf.constant(df)
    image_resized = tf.image.resize(image, (sizex, sizey))
    df_resized = image_resized.numpy()
    if print_:
        print("Shape after resizing:", df_resized.shape)

    return df_resized

def cutBoundaries(df, max_x, max_y, lowerHalf = False):
    df = df.where(df.x < max_x, drop=True)
    df = df.where(-max_x <= df.x, drop=True)
    if lowerHalf:
        df = df.where(df.y < 0, drop=True)
    else:
        df = df.where(df.y < max_y, drop=True)
    df = df.where(-max_y <= df.y, drop=True)
    return df

def takeRandomSamples(full_input, full_target, pred=False, full_prediction=None):
    randTime = rn.randint(0, len(full_input[0]-1))
    sample2dtrain = full_input[0][randTime]
    sample1dtrain = full_input[1][randTime]

    sampletarget = full_target[randTime]

    if pred:
        samplepred = full_prediction[randTime]

        return sample2dtrain, sample1dtrain, sampletarget, samplepred, randTime
    else:
        return sample2dtrain, sample1dtrain, sampletarget, randTime


def plotTrain(GCMLike, sample2dtrain, numVar, ax, time, list_var, region="Whole antarctica"):
    if region != "Whole antarctica":
        ds = createLowerInput(GCMLike, region = region, Nx=48, Ny=25, print_=False)
    else:
        ds = GCMLike

    VAR = list_var[numVar]
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain[VAR] = xr.Variable(
        dims=("y", "x"), data=sample2dtrain[:, :, numVar], attrs=ds[VAR].attrs
    )
    dftrain[VAR].plot(
        ax=ax, x="x", transform=ccrs.SouthPolarStereo(), add_colorbar=True, cmap='RdYlBu_r'
    )

    ax.coastlines("10m", color="black")
    ax.gridlines()

    ax.set_title(f"{time} Input: {VAR}")


def plotTarget(target_dataset, sampletarget, ax, vmin, vmax, region="Whole antarctica"):
    if region != "Whole antarctica":
        ds = createLowerTarget(target_dataset, region = region, Nx=64, Ny = 64, print_=False)
    else:
        ds = target_dataset

    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain["SMB"] = xr.Variable(
        dims=("y", "x"), data=sampletarget[:, :, 0], attrs=ds["SMB"].attrs
    )
    pl = dftrain.SMB.plot(ax=ax, x="x", transform=ccrs.SouthPolarStereo(), add_colorbar=True, 
                          cmap='RdYlBu_r', vmin = vmin, vmax = vmax)
    ax.coastlines("10m", color="black")
    ax.gridlines()
    ax.set_title(f"Target: SMB, {region}")


def plotPred(target_dataset, samplepred, ax, vmin, vmax, region="Whole antarctica"):
    if region != "Whole antarctica":
        ds = createLowerTarget(target_dataset, region = region, Nx=64, Ny = 64, print_=False)
    else:
        ds = target_dataset
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain["SMB"] = xr.Variable(
        dims=("y", "x"), data=samplepred[:, :, 0], attrs=ds["SMB"].attrs
    )
    dftrain.SMB.plot(ax=ax, x="x", transform=ccrs.SouthPolarStereo(), add_colorbar=True, 
                          cmap='RdYlBu_r', vmin = vmin, vmax = vmax)
    ax.coastlines("10m", color="black")
    ax.gridlines()
    ax.set_title(f"Prediction: SMB, {region}")


def plotPearsonCorr(target_dataset, samplepred, ax, vmin, vmax, region="Whole antarctica"):
    if region != "Whole antarctica":
        ds = createLowerTarget(target_dataset, region = region, Nx=64, Ny = 64, print_=False)
    else:
        ds = target_dataset
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain["Correlation"] = xr.Variable(
        dims=("y", "x"), data=samplepred[:, :, 0]
    )
    dftrain.Correlation.plot(ax=ax, x="x", transform=ccrs.SouthPolarStereo(), add_colorbar=True, 
                          cmap='RdYlBu_r', vmin = vmin, vmax = vmax)
    ax.coastlines("10m", color="black")
    ax.gridlines()
    ax.set_title(f"Pearson correlation between prediction and target, {region}")
    
def plotWasserstein(target_dataset, samplepred, ax, vmin, vmax, region="Whole antarctica"):
    if region != "Whole antarctica":
        ds = createLowerTarget(target_dataset, region = region, Nx=64, Ny = 64, print_=False)
    else:
        ds = target_dataset
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain["Wasserstein"] = xr.Variable(
        dims=("y", "x"), data=samplepred[:, :, 0]
    )
    dftrain.Wasserstein.plot(ax=ax, x="x", transform=ccrs.SouthPolarStereo(), add_colorbar=True, 
                          cmap='RdYlBu_r', vmin = vmin, vmax = vmax)
    ax.coastlines("10m", color="black")
    ax.gridlines()
    ax.set_title(f"Wasserstein distance between prediction and target, {region}")
    
    
def createLowerTarget(target_dataset, region, Nx=64, Ny = 64, print_=True):
    max_y = Nx * 35 * 1000
    max_x = Ny * 35 * 1000
    
    min_x_res = target_dataset.x.min().values
    min_y_res = target_dataset.y.min().values
    
    max_x_res = target_dataset.x.max().values
    max_y_res = target_dataset.y.max().values
    
    if region == 'Larsen':
        df = target_dataset.where(target_dataset.y>=0, drop = True)
        df = df.where(df.y < max_y, drop = True)
        df = df.where(df.x < min_x_res+max_x, drop=True)
        
    if region == 'Lower peninsula':
        max_x = (Nx / 2) * 35 * 1000
        max_y = Ny * 35 * 1000
        df = cutBoundaries(target_dataset, max_x, max_y, lowerHalf=True)
        
    if region == 'Amundsen':
        df = target_dataset.where(target_dataset.y<=min_y_res+1.5*max_y, drop = True)
        df = df.where(df.y>min_y_res+0.5*max_y, drop = True)
        df = df.where(df.x < min_x_res+max_x, drop=True)
        
    if region == 'Wilkes':
        df = target_dataset.where(target_dataset.x>max_x_res-max_x, drop = True)
        df = df.where(df.y<=min_y_res+1.5*max_y, drop = True)
        df = df.where(df.y>min_y_res+0.5*max_y, drop = True)  
        
    if region == 'Maud':
        df = target_dataset.where(target_dataset.y>max_y_res-max_y, drop = True)
        df = df.where(df.x>=max_x_res-1.5*max_x, drop = True)
        df = df.where(df.x<max_x_res-0.5*max_x, drop = True)  
    if print_:
        print("New target dimensions:", df.dims)
    return df

def createLowerInput(GCMLike, region, Nx=48, Ny=25, print_=True):
    if region == 'Lower peninsula':
        max_x = (Nx/2) * 68 * 1000
        max_y = (Ny) * 206 * 1000
        
        df = GCMLike.where(GCMLike.x < max_x, drop=True)
        df = df.where(-max_x <= df.x, drop=True)
        
    if region == 'Larsen':
        min_x = GCMLike.x.min().values
        df = GCMLike.where(GCMLike.x < min_x+ (Nx * 68 * 1000), drop=True)
        
    if region == 'Amundsen': # same region as Larsen
        min_x = GCMLike.x.min().values
        df = GCMLike.where(GCMLike.x < min_x+ (Nx * 68 * 1000), drop=True)
    
    if region == 'Wilkes': # same region as Larsen
        max_x = GCMLike.x.max().values
        df = GCMLike.where(GCMLike.x > max_x- (Nx * 68 * 1000), drop=True)
    
    if region == 'Maud': # same region as lower peninsula
        max_x = (Nx/2) * 68 * 1000
        max_y = (Ny) * 206 * 1000
    
        df = GCMLike.where(GCMLike.x < max_x, drop=True)
        df = df.where(-max_x <= df.x, drop=True)
    
    if print_:
        print("New dimensions:", df.dims)
        
    return df

def createLowerInput(GCMLike, region, Nx=48, Ny=25, print_=True):
    if region == 'Lower peninsula':
        max_x = (Nx/2) * 68 * 1000
        max_y = (Ny) * 206 * 1000
        
        df = GCMLike.where(GCMLike.x < max_x, drop=True)
        df = df.where(-max_x <= df.x, drop=True)
        
    if region == 'Larsen':
        min_x = GCMLike.x.min().values
        df = GCMLike.where(GCMLike.x < min_x+ (Nx * 68 * 1000), drop=True)
        
    if region == 'Amundsen': # same region as Larsen
        min_x = GCMLike.x.min().values
        df = GCMLike.where(GCMLike.x < min_x+ (Nx * 68 * 1000), drop=True)
    
    if region == 'Wilkes': # same region as Larsen
        max_x = GCMLike.x.max().values
        df = GCMLike.where(GCMLike.x > max_x- (Nx * 68 * 1000), drop=True)
    
    if region == 'Maud': # same region as lower peninsula
        max_x = (Nx/2) * 68 * 1000
        max_y = (Ny) * 206 * 1000
    
        df = GCMLike.where(GCMLike.x < max_x, drop=True)
        df = df.where(-max_x <= df.x, drop=True)
    
    if print_:
        print("New dimensions:", df.dims)
        
    return df

def standardize(data):
    import numpy as np
    mean =  np.nanmean(data,axis=(1,2), keepdims=True)
    sd   =  np.nanstd(data,axis=(1,2), keepdims=True)
    ndata = (data - mean)/sd
    return (ndata)

# A quick function to get the highest power of two close to n.
def highestPowerof2(n):
    res = 0;
    for i in range(n, 0, -1):
        # If i is a power of 2
        if ((i & (i - 1)) == 0):
            res = i;
            break;
    return res;


def calculatePearson(preds,true_smb):
  predictions = torch.tensor(preds)
  target = torch.tensor(true_smb)

  PearsonCorr = np.empty((predictions.shape[1], predictions.shape[2],1))
  for i in range(predictions.shape[1]): #x
    for j in range(predictions.shape[2]): #y
      pixelPred = predictions[:,i,j,0].numpy()
      pixelTarg = target[:,i,j,0].numpy()
      PearsonCorr[i,j] = np.corrcoef(pixelPred,pixelTarg)[0,1]

  # Fill NaN with 0 (uncorrelated)
  PearsonCorr = np.nan_to_num(PearsonCorr)
  return PearsonCorr

def calculateWasserstein(preds,true_smb):
  predictions = torch.tensor(preds)
  target = torch.tensor(true_smb)

  Wasserstein = np.empty((predictions.shape[1], predictions.shape[2],1))
  for i in range(predictions.shape[1]): #x
    for j in range(predictions.shape[2]): #y
      pixelPred = predictions[:,i,j,0].numpy()
      pixelTarg = target[:,i,j,0].numpy()
      Wasserstein[i,j] = wasserstein_distance(pixelPred,pixelTarg)

  return Wasserstein  