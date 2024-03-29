"""
Functions to calculate evaluation metrics
"""

#!/usr/bin/env python3
import tensorflow as tf
import xarray as xr
import cartopy.crs as ccrs
from scipy.stats import wasserstein_distance
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import random as rn
import pandas as pd 
import math
from datetime import datetime
import seaborn as sns
import matplotlib
from matplotlib.ticker import LogFormatter 

# Helper scripts
from dataFunctions import *
from config import *
from reproducibility import *

# Set seed:
seed_all(SEED)

"""
calculatePearson: 
calculates pearson correlation between the timeseries of each pixel (i,j) of prediction and target
@input:
- np.array preds: predictions of shape (t, x, y)
- np.array true_smb: truth values of shape (t, x, y)
@output: 2D matrix where each (i,j) coordinate is the correlation of the timeseries at this location
"""


def calculatePearson(preds, true_smb, ignoreSea=True):
    predictions = torch.tensor(preds).clone().detach()
    target = torch.tensor(true_smb).clone().detach()
    PearsonCorr = np.empty((predictions.shape[1], predictions.shape[2], 1))
    for i in range(predictions.shape[1]):  # x
        for j in range(predictions.shape[2]):  # y
            pixelPred = predictions[:, i, j, 0].numpy()
            pixelTarg = target[:, i, j, 0].numpy()
            PearsonCorr[i, j] = np.corrcoef(pixelPred, pixelTarg)[0, 1]

    if ignoreSea:
        return PearsonCorr
    else:
        # Fill NaN with 0 (uncorrelated)
        PearsonCorr = np.nan_to_num(PearsonCorr)
        return PearsonCorr
    
# Function that removes the mean over time
def remove_time_mean(x):
    return x - x.mean(dim='time')

"""
calculatePearsonAnom: 
calculates pearson correlation between the timeseries of each pixel (i,j) of prediction and target 
with seasonal cycle removed
@input:
- np.array preds: predictions of shape (t, x, y)
- np.array true_smb: truth values of shape (t, x, y)
- xarray target_dataset 
@output: 2D matrix where each (i,j) coordinate is the correlation of anomalies of the timeseries at this location
"""

def calculatePearsonAnom(preds, true_smb, target_dataset, train_set, REGION, ignoreSea=True):
    
    # create xarray dataset for target and predictions
    ds = createLowerTarget(target_dataset, region=REGION, Nx=64, Ny=64, print_=False)
    coords = {'time':ds.coords['time'][len(train_set):], "y": ds.coords["y"], "x": ds.coords["x"]}
    dfPred = xr.Dataset(coords=coords, attrs=ds.attrs)
    dfPred["SMB"] = xr.Variable(
            dims=("time", "y", "x"), data=np.array(preds)[:,:,:,0], attrs=ds["SMB"].attrs
        )
    dfTarg = xr.Dataset(coords=coords, attrs=ds.attrs)
    dfTarg["SMB"] = xr.Variable(
            dims=("time", "y", "x"), data=np.array(true_smb)[:,:,:,0], attrs=ds["SMB"].attrs
        )
        
    # Create tensors of size [time, y, x, 1]
    # Remove monthly mean
    ds_anom_pred = torch.tensor(dfPred.groupby('time.month').apply(remove_time_mean).SMB.values).unsqueeze(3)
    ds_anom_targ = torch.tensor(dfTarg.groupby('time.month').apply(remove_time_mean).SMB.values).unsqueeze(3)
    
    # Calculate pearson coefficient as usual 
    return calculatePearson(ds_anom_pred, ds_anom_targ, ignoreSea)
    

"""
calculateWasserstein: 
calculates wasserstein distance between the timeseries of each pixel (i,j) of prediction and target
@input:
- np.array preds: predictions of shape (t, x, y)
- np.array true_smb: truth values of shape (t, x, y)
@output: 2D matrix where each (i,j) coordinate is the wasserstein distance of the timeseries at this location
"""


def calculateWasserstein(preds, true_smb, ignoreSea=True):
    predictions = torch.tensor(preds).clone().detach()
    target = torch.tensor(true_smb).clone().detach()
    Wasserstein = np.empty((predictions.shape[1], predictions.shape[2], 1))
    for i in range(predictions.shape[1]):  # x
        for j in range(predictions.shape[2]):  # y
            pixelPred = predictions[:, i, j, 0].numpy()
            pixelTarg = target[:, i, j, 0].numpy()

            if ignoreSea:
                if not np.any(pixelTarg):  # check if all zeros (then on sea)
                    Wasserstein[i, j] = np.nan
                else:
                    Wasserstein[i, j] = wasserstein_distance(pixelPred, pixelTarg)
            else:
                Wasserstein[i, j] = wasserstein_distance(pixelPred, pixelTarg)

    return Wasserstein


"""
calculateROV: 
calculates ratio of variance between the timeseries of each pixel (i,j) of prediction and target
@input:
- np.array preds: predictions of shape (t, x, y)
- np.array true_smb: truth values of shape (t, x, y)
@output: 2D matrix where each (i,j) coordinate is the ROV of the timeseries at this location
"""


def ROVTwoPixels(pixelPred, pixelTarg):
    # ratio of variance
    varPred = np.var(pixelPred)
    varTar = np.var(pixelTarg)
    if varPred != 0 and varTar != 0:
        return (varPred / varTar) * 100
    else:
        return 0


def calculateROV(preds, true_smb, ignoreSea=True):
    predictions = torch.tensor(preds).clone().detach()
    target = torch.tensor(true_smb).clone().detach()
    ROV = np.empty((predictions.shape[1], predictions.shape[2], 1))
    for i in range(predictions.shape[1]):  # x
        for j in range(predictions.shape[2]):  # y
            pixelPred = predictions[:, i, j, 0].numpy()
            pixelTarg = target[:, i, j, 0].numpy()

            if ignoreSea:
                if not np.any(pixelTarg):  # check if all zeros (then on sea)
                    ROV[i, j] = np.nan
                else:
                    ROV[i, j] = ROVTwoPixels(pixelPred, pixelTarg)
            else:
                ROV[i, j] = ROVTwoPixels(pixelPred, pixelTarg)
    return ROV


"""
calculateRMSE: 
calculates RMSE between the timeseries of each pixel (i,j) of prediction and target
@input:
- np.array preds: predictions of shape (t, x, y)
- np.array true_smb: truth values of shape (t, x, y)
@output: 2D matrix where each (i,j) coordinate is the RMSE of the timeseries at this location
"""
def calculateRMSE(preds, true_smb, ignoreSea=True, normalised = True, squared = False):
    predictions = torch.tensor(preds).clone().detach()
    target = torch.tensor(true_smb).clone().detach()
    max_ = np.nanmax(np.array(true_smb))
    min_ = np.nanmin(np.array(true_smb))
        
    if normalised: 
        divident = (max_-min_)
    else:
        divident = 1
    RMSE = np.empty((predictions.shape[1], predictions.shape[2], 1))
    for i in range(predictions.shape[1]):  # x
        for j in range(predictions.shape[2]):  # y
            pixelPred = predictions[:, i, j, 0].numpy()
            pixelTarg = target[:, i, j, 0].numpy()

            if ignoreSea:
                if not np.any(pixelTarg):  # check if all zeros (then on sea)
                    RMSE[i, j] = np.nan
                else:
                    RMSE[i, j] =  mean_squared_error(y_true = pixelTarg, y_pred = pixelPred, squared=squared)/divident # RMSE if squared False
            else:
                RMSE[i, j] = mean_squared_error(y_true = pixelTarg, y_pred = pixelPred, squared=squared)/divident
    return RMSE



def calculateMetrics(preds, true_smb, target_dataset, train_set, REGION, ignoreSea):
    metrics = {}
    metrics["PearsonCorr"] = calculatePearson(preds, true_smb, ignoreSea)
    metrics["Wasserstein"] = calculateWasserstein(preds, true_smb, ignoreSea)
    metrics["ROV"] = calculateROV(preds, true_smb, ignoreSea)
    metrics["MSE"] = calculateRMSE(preds, true_smb, ignoreSea, normalised = False, squared = True)
    metrics["RMSE"] = calculateRMSE(preds, true_smb, ignoreSea, normalised = False, squared = False)
    metrics["NRMSE"] = calculateRMSE(preds, true_smb, ignoreSea, normalised = True, squared = False)
    metrics["PearsonCorrAn"] = calculatePearsonAnom(preds, true_smb, target_dataset, train_set, REGION, ignoreSea)
    metrics["ROD"] = calculateROD(true_smb, preds)
    
    return metrics


def metricsData(data):
    dic = dict(
        min=np.nanmin(data),
        #p05=np.nanpercentile(data, 5),
        mean=np.nanmean(data),
        median=np.nanmedian(data),
        #p95=np.nanpercentile(data, 95),
        max=np.nanmax(data),
    )
    return dic


"""
plotPearsonCorr: Plot a 2D plot whit its correlation value for each pixel (i,j)
@input: 
- xr.Dataset target_dataset
- samplecorr: correlation matrix to plot
- ax
- float vmin, vmax: min and max of colorbar
- str region: region of interest
"""


def plotPearsonCorr(
    target_dataset, samplecorr, mean, ax, region="Whole Antarctica", cmap = 'GnBu', type  = 'RCM', colorbar = True, vmin = -1
):
    if type == 'RCM':
        ds = createLowerTarget(
                target_dataset, region=region, Nx=64, Ny=64, print_=False
            )
    else:
        ds = createLowerInput(target_dataset, region="Larsen", Nx=48, Ny=25, print_=False)
        #ds = ds.where(ds.y>0, drop=True)

    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain["Correlation"] = xr.Variable(dims=("y", "x"), data=samplecorr[:, :, 0])
    im = dftrain.Correlation.plot(
        ax=ax,
        x="x",
        transform=ccrs.SouthPolarStereo(),
        add_colorbar=colorbar,
        cmap=cmap,
        vmin=vmin,
        vmax=1,
    )
    ax.coastlines("10m", color="black", linewidth = 1)
    #ax.gridlines()
    ax.set_title("Correlation, mean:{:.2f}".format(mean))
    return im


"""
plotWasserstein: Plot a 2D plot whit its wasserstein distance for each pixel (i,j)
@input: 
- xr.Dataset target_dataset
- samplewass: wasserstein distance matrix to plot
- ax
- float vmin, vmax: min and max of colorbar
- str region: region of interest
"""


def plotWasserstein(
    target_dataset, samplewass, mean, ax, vmin, vmax, region="Whole Antarctica", cmap = 'GnBu', colorbar = True
):
    if region != "Whole Antarctica":
        ds = createLowerTarget(
            target_dataset, region=region, Nx=64, Ny=64, print_=False
        )
    else:
        ds = target_dataset
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain["Wasserstein"] = xr.Variable(dims=("y", "x"), data=samplewass[:, :, 0])
    im = dftrain.Wasserstein.plot(
        ax=ax,
        x="x",
        transform=ccrs.SouthPolarStereo(),
        add_colorbar=colorbar,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=matplotlib.colors.LogNorm()
    )    
    ax.coastlines("10m", color="black", linewidth = 1)
    ax.gridlines()
    ax.set_title("Wasserstein distance, mean:{:.2f}".format(mean))
    return im


"""
plotROV: Plot a 2D plot whit its ratio of variance for each pixel (i,j)
@input: 
- xr.Dataset target_dataset
- samplerov: ratio of variance matrix to plot
- ax
- float vmin, vmax: min and max of colorbar
- str region: region of interest
"""


def plotROV(target_dataset, samplerov, mean, ax, vmin, vmax, region="Whole Antarctica", cmap = 'GnBu'):
    if region != "Whole Antarctica":
        ds = createLowerTarget(
            target_dataset, region=region, Nx=64, Ny=64, print_=False
        )
    else:
        ds = target_dataset
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain["ROV"] = xr.Variable(dims=("y", "x"), data=samplerov[:, :, 0])
    im = dftrain.ROV.plot(
        ax=ax,
        x="x",
        transform=ccrs.SouthPolarStereo(),
        add_colorbar=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.coastlines("10m", color="black", linewidth = 1)
    ax.gridlines()
    ax.set_title("ROV, mean:{:.2f}".format(mean))

def calculateROD(true_smb_Larsen, preds_Larsen):
    targetArray = np.array(true_smb_Larsen)[:,:,:]
    predArray = np.array(preds_Larsen)[:,:,:]    
    ROD = mean_relative_percent(targetArray, predArray)
    
    return ROD

def plotROD(ROD, target_dataset, ax, REGION, cmap = 'GnBu'):    
    ROD_Mean = np.nanmean(ROD, axis = 0)
    vmin = np.nanmin(ROD_Mean)
    vmax = np.nanmax(ROD_Mean)
    
    if REGION != "Whole Antarctica":
        ds = createLowerTarget(
            target_dataset, region=REGION, Nx=64, Ny=64, print_=False
        )
    else:
        ds = target_dataset
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain["ROD"] = xr.Variable(dims=("y", "x"), data=ROD_Mean[:, :, 0])
    cmap = cmap
    im = dftrain.ROD.plot(
        ax=ax,
        x="x",
        transform=ccrs.SouthPolarStereo(),
        add_colorbar=True,
        cmap=cmap,
        vmin=2.1,
        vmax=-2.1,
    )
    ax.coastlines("10m", color="black", linewidth = 1)
    ax.gridlines()
    ax.set_title("Mean relative percent diff, mean:{:.2f}".format(np.nanmean(ROD_Mean)))
    
    

"""
plotRMSE: Plot a 2D plot whit its RMSE for each pixel (i,j)
@input: 
- xr.Dataset target_dataset
- samplerov: ratio of variance matrix to plot
- ax
- float vmin, vmax: min and max of colorbar
- str region: region of interest
"""


def plotNRMSE(
    target_dataset, samplermse, mean, ax, vmin, vmax, region="Whole Antarctica", cmap = "GnBu", normalised = False, colorbar = True
):
    if region != "Whole Antarctica":
        ds = createLowerTarget(
            target_dataset, region=region, Nx=64, Ny=64, print_=False
        )
    else:
        ds = target_dataset
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    # cmap = 'RdYlBu_r'
    if normalised:
        dftrain["NRMSE"] = xr.Variable(dims=("y", "x"), data=samplermse[:, :, 0])
        im = dftrain.NRMSE.plot(
            ax=ax,
            x="x",
            transform=ccrs.SouthPolarStereo(),
            add_colorbar=colorbar,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=matplotlib.colors.LogNorm()
        )
    else:
        dftrain["RMSE"] = xr.Variable(dims=("y", "x"), data=samplermse[:, :, 0])
        im = dftrain.RMSE.plot(
            ax=ax,
            x="x",
            transform=ccrs.SouthPolarStereo(),
            add_colorbar=colorbar,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=matplotlib.colors.LogNorm()
        )
    ax.coastlines("10m", color="black", linewidth = 1)
    ax.gridlines()
    ax.set_title("NRMSE, mean:{:.2f}".format(mean))
    
    return im

"""
plotMetrics: Plot all metrics for each pixel (i,j)
@input: 
- xr.Dataset target_dataset: target dataset
- PearsonCorr: pearson correlation at each pixel
- Wasserstein: wasserstein distance at each pixel
- ROV: ratio of variance at each pixel
- RMSE: RMSE of variance at each pixel
- str today: date of creation of plot (for save fig)
- int num_epochs: number of epochs used to train model
- int batch_size: batch size used to train model
- str region: region of interest (e.g. Larsen)
"""
    
    
def plotMetrics(
    PearsonCorr,
    Wasserstein,
    ROV,
    RMSE,
    PearsonCorrAn, 
    ROD, 
    target_dataset,
    region: str,
    today: str,
    num_epochs: int,
    batch_size: int,
    cmap = 'GnBu',
    figsize = (20, 10)
):
    fig = plt.figure(figsize=figsize)
    vmin, vmax = np.nanmin(PearsonCorr), np.nanmax(PearsonCorr)
    ax = plt.subplot(2, 3, 4, projection=ccrs.SouthPolarStereo())
    meanPearson = np.nanmean(PearsonCorr)
    
    plotPearsonCorr(
        target_dataset, PearsonCorr, meanPearson, ax, vmin, vmax, region=region, cmap = cmap
    )
    
    ax = plt.subplot(2, 3, 3, projection=ccrs.SouthPolarStereo())
    vmin, vmax = np.nanmin(Wasserstein), np.nanmax(Wasserstein)
    meanWass = np.nanmean(Wasserstein)
    plotWasserstein(
        target_dataset, Wasserstein, meanWass, ax, vmin, vmax, region=region, cmap = cmap
    )
    
    ax = plt.subplot(2, 3, 2, projection=ccrs.SouthPolarStereo())
    vmin, vmax = np.nanmin(ROV), np.nanmax(ROV)
    meanROV = np.nanmean(ROV)
    plotROV(target_dataset, ROV, meanROV, ax, vmin, vmax, region=region, cmap = cmap)
    
    ax = plt.subplot(2, 3, 1, projection=ccrs.SouthPolarStereo())
    vmin, vmax = np.nanmin(RMSE), np.nanmax(RMSE)
    meanRMSE = np.nanmean(RMSE)
    plotNRMSE(target_dataset, RMSE, meanRMSE, ax, vmin, vmax, region=region, cmap = cmap)
    ax.set_title('RMSE mean: {:.2f}'.format(meanRMSE))
    
    ax = plt.subplot(2, 3, 5, projection=ccrs.SouthPolarStereo())
    vmin, vmax = np.nanmin(PearsonCorrAn), np.nanmax(PearsonCorrAn)
    meanPearsonAn = np.nanmean(PearsonCorrAn)
    plotPearsonCorr(target_dataset, PearsonCorrAn, meanPearsonAn, ax, vmin, vmax, region=region, cmap = cmap)
    ax.set_title('Correlation w/o seasonality, mean: {:.2f}'.format(meanPearsonAn))
    
    ax = plt.subplot(2, 3, 6, projection=ccrs.SouthPolarStereo())
    plotROD(ROD, target_dataset, ax, region, cmap = cmap)
    nameFig = f"{today}_metrics_{region}_{num_epochs}_{batch_size}.png"
    plt.savefig(nameFig)
    # files.download(nameFig)
    

    
# Bad because division by 0
def mean_absolute_error(ytrue, ypred):
    return (np.abs(ytrue-ypred))/np.abs(ytrue)

# Relative Percent Difference
"""This is a signed expression, 
positive when ytrue exceeds ypred and negative when ypred exceeds ytrue. 
Its value always lies between −2 and 2
"""
def mean_relative_percent(ytrue, ypred):
    ytrue[ytrue == 0.0] = np.nan # fill ocean values with NaN
    return 2*(ytrue-ypred)/(np.abs(ytrue)+np.abs(ypred))


def plotGCMTimeseries(true_smb_Larsen, preds_Larsen, UPRCM, PearsonCorr, target_dataset, train_set, points_GCM, points_RCM, REGION):
    f = plt.figure(figsize=(25, 30))
    xticks = pd.date_range(datetime(2088, 1, 1), datetime(2100, 1, 1), freq="YS")
    
    M = 7
    ax1 = plt.subplot(M, 4, 1, projection=ccrs.SouthPolarStereo())
    randTime = rn.randint(0, len(true_smb_Larsen) - 1)
    
    ## PLOT GCM: 
    dsGCM = createLowerInput(UPRCM, region=REGION, Nx=48, Ny=25, print_=False)
    dsGCM.RF.isel(time=randTime).plot(x="x", ax=ax1, transform=ccrs.SouthPolarStereo())
    ax1.coastlines("10m", color="white")
    ax1.gridlines()
    
    # {'x':30, 'y':13}

    for p in points_GCM:
        ax1.scatter(
            dsGCM.isel(x=p["x"]).x.values,
            dsGCM.isel(y=p["y"]).y.values,
            marker="x",
            s=100,
            color="red",
        )
        
    ## PEARSON:
    ax2 = plt.subplot(M, 4, 2, projection=ccrs.SouthPolarStereo())
    meanPearson = np.nanmean(PearsonCorr)
    plotPearsonCorr(
        target_dataset,
        PearsonCorr,
        meanPearson,
        ax2,
        np.nanmin(PearsonCorr),
        np.nanmax(PearsonCorr),
        region=REGION,
    )
    ## TARGETS:
    target_sample = true_smb_Larsen[randTime]
    target_pred = preds_Larsen[randTime]
    vmin = np.nanmin([target_pred, target_sample])
    vmax = np.nanmax([target_pred, target_sample])
    
    ax3 = plt.subplot(M, 4, 3, projection=ccrs.SouthPolarStereo())
    plotTarget(target_dataset, target_sample, ax3, vmin, vmax, region=REGION)
    ax3.coastlines("10m", color="black")
    ax3.gridlines()
    
    ax4 = plt.subplot(M, 4, 4, projection=ccrs.SouthPolarStereo())
    plotPred(target_dataset, target_pred, ax4, vmin, vmax, region=REGION)
    ax4.coastlines("10m", color="black")
    ax4.gridlines()
    
    axs = [ax2, ax3, ax4]
    ds = createLowerTarget(target_dataset, region=REGION, Nx=64, Ny=64, print_=False)
    for ax in axs:
        for p in points_RCM:
            ax.scatter(
                ds.isel(x=p["x"]).x.values,
                ds.isel(y=p["y"]).y.values,
                marker="x",
                s=100,
                color="red",
            )
            
    # TARGET AND PREDICTION
    i, j = 5, 1
    for m in range(len(points_GCM)):
        # Add predictions and target
        p_RCM = points_RCM[m]
        randomPixel_pred = np.array(preds_Larsen)[:, p_RCM["y"], p_RCM["x"], 0]
        randomPixel_targ = np.array(true_smb_Larsen)[:, p_RCM["y"], p_RCM["x"], 0]
        df = pd.DataFrame(
            data={"pred": randomPixel_pred, "target": randomPixel_targ},
            index=target_dataset.time.values[len(train_set) :],
        )
        ax2 = plt.subplot(M, 4, (i, i + 1))
        df["target"].plot(
            label="target", color="blue", alpha=0.5, ax=ax2, xticks=xticks.to_pydatetime()
        )
        df["pred"].plot(
            label="prediction",
            color="red",
            ax=ax2,
            xticks=xticks.to_pydatetime(),
        )
        ax2.axhline(np.mean(df["target"]), color = 'blue', alpha=0.3, linestyle="--")
        ax2.axhline(np.mean(df["pred"]), color = 'red', alpha=0.3, linestyle="--")
        pearson = np.corrcoef(df["pred"], df["target"])[0, 1]
        rmse = mean_squared_error(y_true = df["target"], y_pred = df["pred"], squared = False)
        nrmse = rmse/(df["target"].max()- df["target"].min())
        ax2.set_title("Point:{}, pearson:{:.2f}, rmse:{:.2f}, nrmse:{:.2f}".format(p_RCM, pearson, rmse, nrmse))
        ax2.legend()
        ax2.set_xticklabels([x.strftime("%Y") for x in xticks])
        if j == 1:
            i += 2
            j += 1
        else:
            i += 10
            j = 1
            
    # GCM variables
    i, j = 9, 1
    for m in range(len(points_GCM)):
        p_GCM = points_GCM[m]
        randomPixel_tt = dsGCM.isel(x=p_GCM["x"], y=p_GCM["y"]).TT.values[len(train_set):]
        randomPixel_smb = dsGCM.isel(x=p_GCM["x"], y=p_GCM["y"]).SMB.values[len(train_set):]
        randomPixel_vp = dsGCM.isel(x=p_GCM["x"], y=p_GCM["y"]).VVP.values[len(train_set):]
        randomPixel_uup = dsGCM.isel(x=p_GCM["x"], y=p_GCM["y"]).UUP.values[len(train_set):]
        randomPixel_rf = dsGCM.isel(x=p_GCM["x"], y=p_GCM["y"]).RF.values[len(train_set):]
        df = pd.DataFrame(
            data={"TT": randomPixel_tt, "VVP": randomPixel_vp, "UUP": randomPixel_uup, "RF": randomPixel_rf, 'SMB':randomPixel_smb},
            index=dsGCM.time.values[len(train_set) :],
        )
        ax = plt.subplot(M, 4, (i, i + 1))
        
        df["TT"].plot(
            label="TT", alpha=0.5, color = 'red', ax=ax, xticks=xticks.to_pydatetime()
        )
        df["VVP"].plot(
            label="VVP", alpha=0.5, color = 'darkblue', ax=ax, linestyle="--", xticks=xticks.to_pydatetime()
        )
        df["UUP"].plot(
            label="UUP", alpha=0.5, color = 'lightblue', ax=ax, linestyle="--", xticks=xticks.to_pydatetime()
        )
        ax2 = ax.twinx()
        df["RF"].plot(
            label="RF", color = 'black', alpha=0.8, ax=ax2, linestyle="-.", xticks=xticks.to_pydatetime()
        )
        df["SMB"].plot(
            label="SMB", color = 'blue', alpha=0.5, ax=ax2, linestyle="-", xticks=xticks.to_pydatetime()
        )
        ax.set_title("Point:{}".format(points_RCM[m]))
        ax.set_xticklabels([x.strftime("%Y") for x in xticks])
        ax.legend()
        ax2.legend()
        ax2.set_ylabel('RF: mmWe/day')
        
        if j == 1:
            i += 2
            j += 1
        else:
            i += 10
            j = 1
            
    i, j = 13, 1
    for m in range(len(points_GCM)):
        p_GCM = points_GCM[m]
        randomPixel_SWD = dsGCM.isel(x=p_GCM["x"], y=p_GCM["y"]).SWD.values[len(train_set):]
        randomPixel_LWD = dsGCM.isel(x=p_GCM["x"], y=p_GCM["y"]).LWD.values[len(train_set):]
        randomPixel_SP = dsGCM.isel(x=p_GCM["x"], y=p_GCM["y"]).SP.values[len(train_set):]
        df = pd.DataFrame(
            data={"SWD": randomPixel_SWD, "LWD": randomPixel_LWD, "SP": randomPixel_SP},
            index=dsGCM.time.values[len(train_set) :],
        )
        ax = plt.subplot(M, 4, (i, i + 1))
        
        df["SWD"].plot(
            label="SWD", color = 'orange', alpha=0.5, ax=ax, xticks=xticks.to_pydatetime()
        )
        df["LWD"].plot(
            label="LWD", color = 'yellow', alpha=0.5, ax=ax, xticks=xticks.to_pydatetime()
        )
        ax2 = ax.twinx()
        df["SP"].plot(
            label="SP", color = 'black', alpha=0.5, linestyle="-.", ax=ax2, xticks=xticks.to_pydatetime()
        )
        
        ax.set_title("Point:{}".format(points_RCM[m]))
        ax.set_xticklabels([x.strftime("%Y") for x in xticks])
        ax.legend()
        ax2.legend()
        
        if j == 1:
            i += 2
            j += 1
        else:
            i += 10
            j = 1
            
    
def randomPoints_Losses(
    points,
    PearsonCorr,
    true_smb_Larsen,
    preds_Larsen_MSE,
    preds_Larsen_RMSE,
    preds_Larsen_NRMSE,
    target_dataset,
    UPRCM,
    train_set,
    region: str,
    N:int = 4,
):
    f = plt.figure(figsize=(20, 15))
    M = (N/2)+1
    ax1 = plt.subplot(M, 4, 1, projection=ccrs.SouthPolarStereo())
    meanPearson = np.nanmean(PearsonCorr)
    plotPearsonCorr(
        target_dataset,
        PearsonCorr,
        meanPearson,
        ax1,
        np.nanmin(PearsonCorr),
        np.nanmax(PearsonCorr),
        region=region,
    )
     
    ax1.set_title(f"Correlation with MSE loss")
    ds = createLowerTarget(target_dataset, region=region, Nx=64, Ny=64, print_=False)
    
    randTime = rn.randint(0, len(true_smb_Larsen) - 1)
    dt = pd.to_datetime([UPRCM.time.isel(time=randTime).values])
    time = str(dt.date[0])
    meanTarget = np.nanmean(np.array(true_smb_Larsen), axis=0)
    
    vmin = np.min(
        [
            meanTarget,
            true_smb_Larsen[randTime],
            preds_Larsen_MSE[randTime],
            preds_Larsen_RMSE[randTime],
            preds_Larsen_NRMSE[randTime],
        ]
    )
    vmax = np.max(
        [
            meanTarget,
            true_smb_Larsen[randTime],
            preds_Larsen_MSE[randTime],
            preds_Larsen_RMSE[randTime],
            preds_Larsen_NRMSE[randTime],
        ]
    )
    
    ax2 = plt.subplot(M, 4, 2, projection=ccrs.SouthPolarStereo())
    plotTarget(target_dataset, meanTarget, ax2, vmin, vmax, region=region)
    ax2.set_title(f"Target: mean SMB, {region}")
    
    ax3 = plt.subplot(M, 4, 3, projection=ccrs.SouthPolarStereo())
    plotTarget(
        target_dataset, true_smb_Larsen[randTime], ax3, vmin, vmax, region=region
    )
    
    ax4 = plt.subplot(M, 4, 4, projection=ccrs.SouthPolarStereo())
    plotPred(target_dataset, preds_Larsen_MSE[randTime], ax4, vmin, vmax, region=region)
    ax4.set_title(f"Predictions with MSE loss")
    
    axs = [ax1, ax2, ax3, ax4]
    for ax in axs:
        for p in points:
            ax.scatter(
                ds.isel(x=p["x"]).x.values,
                ds.isel(y=p["y"]).y.values,
                marker="x",
                s=100,
                color="red",
            )
    plt.suptitle(time)
    
    # Plot timeseries
    p = points[0]
    randomPixel_pred_MSE = np.array(preds_Larsen_MSE)[:, p["y"], p["x"], 0]
    randomPixel_pred_RMSE = np.array(preds_Larsen_RMSE)[:, p["y"], p["x"], 0]
    randomPixel_pred_NRMSE = np.array(preds_Larsen_NRMSE)[:, p["y"], p["x"], 0]
    randomPixel_targ = np.array(true_smb_Larsen)[:, p["y"], p["x"], 0]
    df = pd.DataFrame(
        data={
            "pred_MSE": randomPixel_pred_MSE,
            "pred_RMSE": randomPixel_pred_RMSE,
            "pred_NRMSE": randomPixel_pred_NRMSE,
            "target": randomPixel_targ,
        },
        index=target_dataset.time.values[len(train_set) :],
    )
    ax5 = plt.subplot(M, 4, (5, 6))
    ax5.plot(df["target"], label="target", color="grey", alpha=0.8)
    df["pred_MSE"].plot(label="MSE", linestyle="--", ax=ax5)
    df["pred_RMSE"].plot(label="RMSE", linestyle="--", ax=ax5)
    df["pred_NRMSE"].plot(label="NRMSE",  linestyle="--", ax=ax5)
    ax5.legend()
    pearson = np.corrcoef(df["pred_MSE"], df["target"])[0, 1]
    rmse = mean_squared_error(y_true = df["target"], y_pred = df["pred_MSE"], squared = False)
    nrmse = rmse / (df["target"].max() - df["target"].min())
    ax5.set_title(
        "Point:{}, pearson:{:.2f}, rmse:{:.2f}, nrmse:{:.2f}".format(
            p, pearson, rmse, nrmse
        )
    )
    
    i = 7
    for p in points[1:]:
        randomPixel_pred_MSE = np.array(preds_Larsen_MSE)[:, p["y"], p["x"], 0]
        randomPixel_pred_RMSE = np.array(preds_Larsen_RMSE)[:, p["y"], p["x"], 0]
        randomPixel_pred_NRMSE = np.array(preds_Larsen_NRMSE)[:, p["y"], p["x"], 0]
        randomPixel_targ = np.array(true_smb_Larsen)[:, p["y"], p["x"], 0]
        df = pd.DataFrame(
            data={
                "pred_MSE": randomPixel_pred_MSE,
                "pred_RMSE": randomPixel_pred_RMSE,
                "pred_NRMSE": randomPixel_pred_NRMSE,
                "target": randomPixel_targ,
            },
            index=target_dataset.time.values[len(train_set) :],
        )
        # ax = plt.subplot(2, 3, i, sharey=ax5)
        ax = plt.subplot(M, 4, (i, i + 1))
        df["target"].plot(label="target", color="grey", alpha=0.8, ax=ax)
        df["pred_MSE"].plot(label="MSE", linestyle="--", ax=ax)
        df["pred_RMSE"].plot(label="RMSE",  linestyle="--", ax=ax)
        df["pred_NRMSE"].plot(label="NRMSE", linestyle="--", ax=ax)
        pearson = np.corrcoef(df["pred_MSE"], df["target"])[0, 1]
        rmse = mean_squared_error(y_true = df["target"], y_pred = df["pred_MSE"], squared = False)
        nrmse = rmse / (df["target"].max() - df["target"].min())
        ax.set_title(
            "Point:{}, pearson:{:.2f}, rmse:{:.2f}, nrmse:{:.2f}".format(
                p, pearson, rmse, nrmse
            )
        )
        ax.legend()
        i += 2
    plt.suptitle(f"Three time series at different coordinates {time}")
    
    