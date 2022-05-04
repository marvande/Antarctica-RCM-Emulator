#!/usr/bin/env python3
import tensorflow as tf 
import xarray as xr
import cartopy.crs as ccrs
from scipy.stats import wasserstein_distance
import torch
import numpy as np 

from dataFunctions import * 

"""
calculatePearson: 
calculates pearson correlation between the timeseries of each pixel (i,j) of prediction and target
@input:
- np.array preds: predictions of shape (t, x, y)
- np.array true_smb: truth values of shape (t, x, y)
@output: 2D matrix where each (i,j) coordinate is the correlation of the timeseries at this location
"""
def calculatePearson(preds, true_smb, ignoreSea = True):
	predictions = torch.tensor(preds)
	target = torch.tensor(true_smb)
	
	PearsonCorr = np.empty((predictions.shape[1], predictions.shape[2], 1))
	for i in range(predictions.shape[1]):  # x
		for j in range(predictions.shape[2]):  # y
			pixelPred = predictions[:, i, j, 0].numpy()
			pixelTarg = target[:, i, j, 0].numpy()
			PearsonCorr[i, j] = np.corrcoef(pixelPred, pixelTarg)[0, 1]
	
	mean = np.nanmean(PearsonCorr) # mean of values, ignoring NaN

	if ignoreSea:
		return PearsonCorr, mean
	else:
		# Fill NaN with 0 (uncorrelated)
		PearsonCorr = np.nan_to_num(PearsonCorr)
		return PearsonCorr, mean
"""
calculateWasserstein: 
calculates wasserstein distance between the timeseries of each pixel (i,j) of prediction and target
@input:
- np.array preds: predictions of shape (t, x, y)
- np.array true_smb: truth values of shape (t, x, y)
@output: 2D matrix where each (i,j) coordinate is the wasserstein distance of the timeseries at this location
"""
def calculateWasserstein(preds, true_smb, ignoreSea = True):
	predictions = torch.tensor(preds)
	target = torch.tensor(true_smb)
	Wasserstein = np.empty((predictions.shape[1], predictions.shape[2], 1))
	for i in range(predictions.shape[1]):  # x
		for j in range(predictions.shape[2]):  # y
			pixelPred = predictions[:, i, j, 0].numpy()
			pixelTarg = target[:, i, j, 0].numpy()
			
			if ignoreSea:
				if not np.any(pixelTarg): # check if all zeros (then on sea)
					Wasserstein[i, j] = np.nan
				else:
					Wasserstein[i, j] = wasserstein_distance(pixelPred, pixelTarg)
			else:
				Wasserstein[i, j] = wasserstein_distance(pixelPred, pixelTarg)
	mean = np.nanmean(Wasserstein) # mean of values, ignoring NaN
	
	return Wasserstein, mean

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
		return  (varPred/varTar)
	else: 
		return  0
	
def calculateROV(preds, true_smb, ignoreSea = True):
	predictions = torch.tensor(preds)
	target = torch.tensor(true_smb)
	ROV = np.empty((predictions.shape[1], predictions.shape[2], 1))
	for i in range(predictions.shape[1]):  # x
		for j in range(predictions.shape[2]):  # y
			pixelPred = predictions[:, i, j, 0].numpy()
			pixelTarg = target[:, i, j, 0].numpy()
			
			if ignoreSea:
				if not np.any(pixelTarg): # check if all zeros (then on sea)
					ROV[i, j] = np.nan
				else:
					ROV[i, j] = ROVTwoPixels(pixelPred, pixelTarg)
			else:
				ROV[i, j] = ROVTwoPixels(pixelPred, pixelTarg)
	mean = np.nanmean(ROV) # mean of values, ignoring NaN
	return ROV, mean

"""
plotPearsonCorr: Plot a 2D plot whit its correlation value for each pixel (i,j)
@input: 
- xr.Dataset target_dataset
- samplecorr: correlation matrix to plot
- ax
- float vmin, vmax: min and max of colorbar
- str region: region of interest
"""
def plotPearsonCorr(target_dataset, samplecorr, mean, ax, vmin, vmax, region="Whole Antarctica"):
	if region != "Whole Antarctica":
		ds = createLowerTarget(target_dataset, region = region, Nx=64, Ny = 64, print_=False)
	else:
		ds = target_dataset
	coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
	dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
	dftrain["Correlation"] = xr.Variable(
		dims=("y", "x"), data=samplecorr[:, :, 0]
	)
	dftrain.Correlation.plot(ax=ax, x="x", transform=ccrs.SouthPolarStereo(), add_colorbar=True, 
							cmap='RdYlBu_r', vmin = vmin, vmax = vmax)
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
def plotWasserstein(target_dataset, samplewass, mean, ax, vmin, vmax, region="Whole Antarctica"):
	if region != "Whole Antarctica":
		ds = createLowerTarget(target_dataset, region = region, Nx=64, Ny = 64, print_=False)
	else:
		ds = target_dataset
	coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
	dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
	dftrain["Wasserstein"] = xr.Variable(
		dims=("y", "x"), data=samplewass[:, :, 0]
	)
	dftrain.Wasserstein.plot(ax=ax, x="x", transform=ccrs.SouthPolarStereo(), add_colorbar=True, 
							cmap='RdYlBu_r', vmin = vmin, vmax = vmax)
	ax.coastlines("10m", color="black")
	ax.gridlines()
	ax.set_title("[{}] Wasserstein distance, mean:{:.2f}".format(region, mean))
	
	
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
		ds = createLowerTarget(target_dataset, region = region, Nx=64, Ny = 64, print_=False)
	else:
		ds = target_dataset
	coords = {"y": ds.coords["y"], "x": ds.coords["x"]}
	dftrain = xr.Dataset(coords=coords, attrs=ds.attrs)
	dftrain["ROV"] = xr.Variable(
		dims=("y", "x"), data=samplerov[:, :, 0]
	)
	dftrain.ROV.plot(ax=ax, x="x", transform=ccrs.SouthPolarStereo(), add_colorbar=True, 
							cmap='RdYlBu_r', vmin = vmin, vmax = vmax)
	ax.coastlines("10m", color="black")
	ax.gridlines()
	ax.set_title("[{}] ROV, mean:{:.2f}".format(region, mean))
	