#!/usr/bin/env python3
from dataFunctions import *
from math import cos,sin,pi

def input_maker(
	GCMLike,
	size_input_domain:int=16,  # size of domain, format: 8,16,32, must be defined in advance
	stand:bool=True,  # standardization
	seas:bool=True,  # put a cos, sin vector to control the season, format : bool
	means:bool=True,  # add the mean of the variables raw or stdz, format : bool
	stds:bool=True,  # add the std of the variables raw or stdz, format : bool
	resize_input:bool=True,  # resize input to size_input_domain
	region:str="Larsen",  # region of interest
	regionNbr:int=0, # number of region of interest
	print_:bool=True,
	reg:bool = True # encoding of region
):
	
	if region != "Whole Antarctica":
		DATASET = createLowerInput(GCMLike, region=region, Nx=48, Ny=25, print_=False)
	else:
		DATASET = GCMLike
	"""
	MAKE THE 2D INPUT ARRAY
	SHAPE [nbmonths, x, y, nb_vars]
	"""
		
	# Remove target variable from DATASET:
	DATASET = DATASET.drop(["SMB"])
	
	nbmonths = DATASET.dims["time"]
	x = DATASET.dims["x"]
	y = DATASET.dims["y"]
	nb_vars = len(list(DATASET.data_vars))
	VAR_LIST = list(DATASET.data_vars)
	
	INPUT_2D_bf = np.transpose(
		np.asarray([DATASET[i].values for i in VAR_LIST]), [1, 2, 3, 0]
	)
	
	# if no size is given, take smallest power of 2
	if size_input_domain == None:
		size_input_domain = np.max(
			[
				highestPowerof2(INPUT_2D_bf.shape[1]),
				highestPowerof2(INPUT_2D_bf.shape[2]),
			]
		)
		
	if resize_input:
		# resize to size_input_domain
		INPUT_2D = resize(INPUT_2D_bf, size_input_domain, size_input_domain, print_)
	else:
		INPUT_2D = INPUT_2D_bf
		
	if stand:
		# Standardize:
		INPUT_2D_SDTZ = standardize(INPUT_2D)
		# in their code with aerosols extra stuff but ignore
		INPUT_2D_ARRAY = INPUT_2D_SDTZ
	else:
		INPUT_2D_ARRAY = INPUT_2D
		
	if print_:
		print("Parameters:\n -------------------")
		print("Size of input domain:", size_input_domain)
		print("Region:", region)
		print("\nCreating 2D input X:\n -------------------")
		print(f"Number of variables: {nb_vars}")
		print(f"Variables: {VAR_LIST}")
		print(f"INPUT_2D shape: {INPUT_2D_ARRAY.shape}")
		print("\nCreating 1D input Z:\n -------------------")
		
	"""
	MAKE THE 1D INPUT ARRAY
	CONTAINS MEANS, STD SEASON IF ASKED
	"""
		
	INPUT_1D = []
	if means and stds:
		vect_std = INPUT_2D.std(axis=(1, 2))
		vect_means = INPUT_2D.mean(axis=(1, 2))
		SpatialMean = vect_means.reshape(INPUT_2D.shape[0], 1, 1, INPUT_2D.shape[3])
		SpatialSTD = vect_std.reshape(INPUT_2D.shape[0], 1, 1, INPUT_2D.shape[3])
		
		INPUT_1D.append(SpatialMean)
		INPUT_1D.append(SpatialSTD)
		if print_:
			print(f"SpatialMean/std shape: {SpatialMean.shape}")
			
	if seas:
		months = 12
		cosvect = np.tile(
			[cos(2 * i * pi / months) for i in range(months)],
			int(INPUT_2D.shape[0] / months),
		)
		sinvect = np.tile(
			[sin(2 * i * pi / months) for i in range(months)],
			int(INPUT_2D.shape[0] / months),
		)
		cosvect = cosvect.reshape(INPUT_2D.shape[0], 1, 1, 1)
		sinvect = sinvect.reshape(INPUT_2D.shape[0], 1, 1, 1)
		
		INPUT_1D.append(cosvect)
		INPUT_1D.append(sinvect)
		if print_:
			print(f"Cos/sin encoding shape: {cosvect.shape}")
	# add a en encoding of the current region number to Z
	if reg: 
		regvect = np.array([regionNbr for i in range(INPUT_2D.shape[0])])
		regvect = cosvect.reshape(INPUT_2D.shape[0], 1, 1, 1)
		INPUT_1D.append(regvect)
		
	INPUT_1D_ARRAY = np.concatenate(INPUT_1D, axis=3)
	if print_:
		print(f"INPUT_1D shape: {INPUT_1D_ARRAY.shape}")
		
	DATASET.close()
	return INPUT_2D_ARRAY, INPUT_1D_ARRAY, VAR_LIST


def make_inputs(GCMLike, 
				size_input_domain:int, 
				Region:str, 
				regionNbr:int=0): # for combined regions, each sample gets a number so that we know to which region it corresponds
	# Make input
	i2D, i1D, VAR_LIST = input_maker(
		GCMLike=GCMLike,
		size_input_domain=size_input_domain,
		stand=True,  # standardization
		seas=True,  # put a cos,sin vector to control the season, format : bool
		means=True,  # add the mean of the variables raw or stdz, format : r,s,n
		stds=True,
		resize_input=True,
		region=Region,
		regionNbr=regionNbr,
		print_=False,
		reg = True
	)
	
	# Make a non standardised version for plots:
	i2D_ns, i1D_ns, var_list = input_maker(
		GCMLike=GCMLike,
		size_input_domain=size_input_domain,
		stand=False,  # standardization
		seas=True,  # put a cos,sin vector to control the season, format : bool
		means=True,  # add the mean of the variables raw or stdz, format : r,s,n
		stds=True,
		resize_input=False,
		region=Region,
		regionNbr=regionNbr,
		print_=False,
		reg = True
	)
	return i1D, i2D, i1D_ns, i2D_ns, VAR_LIST


def target_maker(
	target_dataset,
	region="Larsen",  # region of interest
	resize=True,  # resize to target_size
	target_size=None,  # if none provided and resize true, set to min highest power of 2
):
	target_times = []
	targets = []
	
	if region != "Whole antarctica":
		lowerTarget = createLowerTarget(
			target_dataset, region=region, Nx=64, Ny=64, print_=False
		)
		targetArray = lowerTarget.SMB.values
	else:
		targetArray = target_dataset.SMB.values
		
	targetArray = targetArray.reshape(
		targetArray.shape[0], targetArray.shape[1], targetArray.shape[2], 1
	)
	
	if target_size == None:
		# resize to highest power of 2:
		target_size = np.min(
			[
				highestPowerof2(targetArray.shape[1]),
				highestPowerof2(targetArray.shape[2]),
			]
		)
		
	if resize:
		target_SMB = resize(targetArray, target_size, target_size)
	else:
		target_SMB = targetArray
		
	targets.append(target_SMB)
	target_times.append(target_dataset.time.values)
	
	return targets, target_times


""""Create encoding of regions so that each sample has a number 
that indicates which region of Antarctica it corresponds to."""
def regionEncoder(X, REGION):
	# Indicator of regions and their order if combined dataset
	# Encoding 0-> Num regions
	regions = []
	if REGION == "Combined":
			N = int(len(X) / len(REGIONS))
			for j in range(len(REGIONS)):
					for i in range(N):
							regions.append(j)
	else:
			N = len(X)
			regions = [0 for i in range(N)]
	R = torch.tensor(regions) # Change to tensor
	return R



"""standardize: standardises data array by substracting mean and dividing by std
"""
def standardize(data):
	import numpy as np
	
	mean = np.nanmean(data, axis=(1, 2), keepdims=True)
	sd = np.nanstd(data, axis=(1, 2), keepdims=True)
	ndata = (data - mean) / sd
	return ndata


"""highestPowerof2: quick function to get the highest power of two close to n.
"""
def highestPowerof2(n):
	res = 0
	for i in range(n, 0, -1):
		# If i is a power of 2
		if (i & (i - 1)) == 0:
			res = i
			break
	return res