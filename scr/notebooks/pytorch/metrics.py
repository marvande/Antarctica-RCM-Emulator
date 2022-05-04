#!/usr/bin/env python3
import tensorflow as tf
import xarray as xr
import cartopy.crs as ccrs
from scipy.stats import wasserstein_distance
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from dataFunctions import *

"""
calculatePearson: 
calculates pearson correlation between the timeseries of each pixel (i,j) of prediction and target
@input:
- np.array preds: predictions of shape (t, x, y)
- np.array true_smb: truth values of shape (t, x, y)
@output: 2D matrix where each (i,j) coordinate is the correlation of the timeseries at this location
"""


def calculatePearson(preds, true_smb, ignoreSea=True):
    predictions = torch.tensor(preds)
    target = torch.tensor(true_smb)

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


"""
calculateWasserstein: 
calculates wasserstein distance between the timeseries of each pixel (i,j) of prediction and target
@input:
- np.array preds: predictions of shape (t, x, y)
- np.array true_smb: truth values of shape (t, x, y)
@output: 2D matrix where each (i,j) coordinate is the wasserstein distance of the timeseries at this location
"""


def calculateWasserstein(preds, true_smb, ignoreSea=True):
    predictions = torch.tensor(preds)
    target = torch.tensor(true_smb)
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
    predictions = torch.tensor(preds)
    target = torch.tensor(true_smb)
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


def calculateRMSE(preds, true_smb, ignoreSea=True):
    predictions = torch.tensor(preds)
    target = torch.tensor(true_smb)
    RMSE = np.empty((predictions.shape[1], predictions.shape[2], 1))
    for i in range(predictions.shape[1]):  # x
        for j in range(predictions.shape[2]):  # y
            pixelPred = predictions[:, i, j, 0].numpy()
            pixelTarg = target[:, i, j, 0].numpy()

            if ignoreSea:
                if not np.any(pixelTarg):  # check if all zeros (then on sea)
                    RMSE[i, j] = np.nan
                else:
                    RMSE[i, j] = mean_squared_error(pixelTarg, pixelPred, squared=False)
            else:
                RMSE[i, j] = mean_squared_error(pixelTarg, pixelPred, squared=False)
    return RMSE


def calculateMetrics(preds, true_smb, ignoreSea):
    PearsonCorr = calculatePearson(preds, true_smb, ignoreSea)
    Wasserstein = calculateWasserstein(preds, true_smb, ignoreSea)
    ROV = calculateROV(preds, true_smb, ignoreSea)
    RMSE = calculateRMSE(preds, true_smb, ignoreSea)
    return PearsonCorr, Wasserstein, ROV, RMSE


def metricsData(data):
    dic = dict(
        min=np.nanmin(data),
        p05=np.nanpercentile(data, 5),
        mean=np.nanmean(data),
        median=np.nanmedian(data),
        p95=np.nanpercentile(data, 95),
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
    target_dataset, samplecorr, mean, ax, vmin, vmax, region="Whole Antarctica"
):
    if region != "Whole Antarctica":
        ds = createLowerTarget(
            target_dataset, region=region, Nx=64, Ny=64, print_=False
        )
    else:
        ds = target_dataset
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain["Correlation"] = xr.Variable(dims=("y", "x"), data=samplecorr[:, :, 0])
    dftrain.Correlation.plot(
        ax=ax,
        x="x",
        transform=ccrs.SouthPolarStereo(),
        add_colorbar=True,
        cmap="GnBu",
        vmin=vmin,
        vmax=1,
    )
    ax.coastlines("10m", color="black")
    ax.gridlines()
    ax.set_title("[{}] Pearson correlation, mean:{:.2f}".format(region, mean))


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
    target_dataset, samplewass, mean, ax, vmin, vmax, region="Whole Antarctica"
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
    dftrain.Wasserstein.plot(
        ax=ax,
        x="x",
        transform=ccrs.SouthPolarStereo(),
        add_colorbar=True,
        cmap="GnBu",
        vmin=vmin,
        vmax=vmax,
    )
    ax.coastlines("10m", color="black")
    ax.gridlines()
    ax.set_title("[{}] Wasserstein dist, mean:{:.2f}".format(region, mean))


"""
plotROV: Plot a 2D plot whit its ratio of variance for each pixel (i,j)
@input: 
- xr.Dataset target_dataset
- samplerov: ratio of variance matrix to plot
- ax
- float vmin, vmax: min and max of colorbar
- str region: region of interest
"""


def plotROV(target_dataset, samplerov, mean, ax, vmin, vmax, region="Whole Antarctica"):
    if region != "Whole Antarctica":
        ds = createLowerTarget(
            target_dataset, region=region, Nx=64, Ny=64, print_=False
        )
    else:
        ds = target_dataset
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain["ROV"] = xr.Variable(dims=("y", "x"), data=samplerov[:, :, 0])
    dftrain.ROV.plot(
        ax=ax,
        x="x",
        transform=ccrs.SouthPolarStereo(),
        add_colorbar=True,
        cmap="GnBu",
        vmin=vmin,
        vmax=vmax,
    )
    ax.coastlines("10m", color="black")
    ax.gridlines()
    ax.set_title("[{}] ROV, mean:{:.2f}".format(region, mean))


"""
plotRMSE: Plot a 2D plot whit its RMSE for each pixel (i,j)
@input: 
- xr.Dataset target_dataset
- samplerov: ratio of variance matrix to plot
- ax
- float vmin, vmax: min and max of colorbar
- str region: region of interest
"""


def plotRMSE(
    target_dataset, samplermse, mean, ax, vmin, vmax, region="Whole Antarctica"
):
    if region != "Whole Antarctica":
        ds = createLowerTarget(
            target_dataset, region=region, Nx=64, Ny=64, print_=False
        )
    else:
        ds = target_dataset
    coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
    dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
    dftrain["RMSE"] = xr.Variable(dims=("y", "x"), data=samplermse[:, :, 0])
    # cmap = 'RdYlBu_r'
    cmap = "GnBu"
    dftrain.RMSE.plot(
        ax=ax,
        x="x",
        transform=ccrs.SouthPolarStereo(),
        add_colorbar=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.coastlines("10m", color="black")
    ax.gridlines()
    ax.set_title("[{}] RMSE, mean:{:.2f}".format(region, mean))


def plotMetrics(PearsonCorr, Wasserstein, ROV, RMSE, target_dataset, region, today, num_epochs, batch_size):
    fig = plt.figure(figsize=(25, 5))
    vmin, vmax = np.nanmin(PearsonCorr), np.nanmax(PearsonCorr)
    ax = plt.subplot(1, 4, 3, projection=ccrs.SouthPolarStereo())
    meanPearson = np.nanmean(PearsonCorr)
    plotPearsonCorr(
        target_dataset, PearsonCorr, meanPearson, ax, vmin, vmax, region=region
    )

    ax = plt.subplot(1, 4, 4, projection=ccrs.SouthPolarStereo())
    vmin, vmax = np.nanmin(Wasserstein), np.nanmax(Wasserstein)
    meanWass = np.nanmean(Wasserstein)
    plotWasserstein(
        target_dataset, Wasserstein, meanWass, ax, vmin, vmax, region=region
    )

    ax = plt.subplot(1, 4, 2, projection=ccrs.SouthPolarStereo())
    vmin, vmax = np.nanmin(ROV), np.nanmax(ROV)
    meanROV = np.nanmean(ROV)
    plotROV(target_dataset, ROV, meanROV, ax, vmin, vmax, region=region)

    ax = plt.subplot(1, 4, 1, projection=ccrs.SouthPolarStereo())
    vmin, vmax = np.nanmin(RMSE), np.nanmax(RMSE)
    meanRMSE = np.nanmean(RMSE)
    plotRMSE(target_dataset, RMSE, meanRMSE, ax, vmin, vmax, region=region)

    nameFig = f"{today}_metrics_{region}_{num_epochs}_{batch_size}.png"
    plt.savefig(nameFig)
    # files.download(nameFig)
