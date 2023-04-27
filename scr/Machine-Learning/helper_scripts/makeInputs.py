"""
Functions that create inputs for model
"""
from dataFunctions import *
from math import cos, sin, pi


def input_maker(
    UPRCM,
    GCM=None,
    size_input_domain: int = 16,  # size of domain, format: 8,16,32, must be defined in advance
    stand: bool = True,  # standardization of X
    standZ: bool = True,  # standardization of Z
    seas: bool = True,  # put a cos, sin vector to control the season, format : bool
    means: bool = True,  # add the mean of the variables raw or stdz, format : bool
    stds: bool = True,  # add the std of the variables raw or stdz, format : bool
    resize_input: bool = True,  # resize input to size_input_domain
    region: str = "Larsen",  # region of interest
    print_: bool = True,
    dropvarRCM=None,
    dropvarGCM=None,
):

    DATASET = createLowerInput(
        UPRCM, region=region, Nx=48, Ny=25, print_=False
    )  # UPRCM
    if GCM != None:
        DATASETGCM = createLowerInput(
            GCM, region=region, Nx=48, Ny=25, print_=False
        )  # GCM

    """
	MAKE THE 2D INPUT ARRAY
	SHAPE [nbmonths, x, y, nb_vars]
	"""
    # Remove target variable from DATASET for UPRCM:
    DATASET = DATASET.drop(["SMB"])

    if dropvarRCM != None:  # drop additional variables if needed
        DATASET = DATASET.drop(dropvarRCM)

    if GCM != None and dropvarGCM != None:
        DATASETGCM = DATASETGCM.drop(dropvarGCM)

    nbmonths = DATASET.dims["time"]
    x = DATASET.dims["x"]
    y = DATASET.dims["y"]
    nb_vars = len(list(DATASET.data_vars))
    VAR_LIST = sorted(list(DATASET.data_vars))  # sort variables alphabetically
    if print_:
        print("Variables:", VAR_LIST)

    INPUT_2D_bf = np.transpose(
        np.asarray([DATASET[i].values for i in VAR_LIST]), [1, 2, 3, 0]
    )
    if GCM != None:
        INPUT_2D_bf_GCM = np.transpose(
            np.asarray([DATASETGCM[i].values for i in VAR_LIST]), [1, 2, 3, 0]
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
        if GCM != None:
            INPUT_2D_GCM = resize(
                INPUT_2D_bf_GCM, size_input_domain, size_input_domain, print_
            )
    else:
        INPUT_2D = INPUT_2D_bf
        if GCM != None:
            INPUT_2D_GCM = INPUT_2D_bf_GCM

    if stand:
        # Standardize UPRCM:
        INPUT_2D_SDTZ = standardize(INPUT_2D, mean=None, std=None, ownstd=True)
        INPUT_2D_ARRAY = INPUT_2D_SDTZ
        if GCM != None:
            # standardize GCM according to RCM values
            # mean = np.nanmean(INPUT_2D, axis=(1, 2), keepdims=True)
            # std = np.nanstd(INPUT_2D, axis=(1, 2), keepdims=True)
            INPUT_2D_SDTZ_GCM = standardize(
                INPUT_2D_GCM, mean=None, std=None, ownstd=True
            )
            INPUT_2D_ARRAY_GCM = INPUT_2D_SDTZ_GCM
    else:
        INPUT_2D_ARRAY = INPUT_2D
        if GCM != None:
            INPUT_2D_ARRAY_GCM = INPUT_2D_GCM

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
        if GCM != None:
            INPUT_2D = INPUT_2D_GCM
            
        vect_std = INPUT_2D.std(axis=(1, 2))
        vect_means = INPUT_2D.mean(axis=(1, 2))

        SpatialMean = vect_means.reshape(INPUT_2D.shape[0], 1, 1, INPUT_2D.shape[3])
        SpatialSTD = vect_std.reshape(INPUT_2D.shape[0], 1, 1, INPUT_2D.shape[3])

        if standZ:
            refmean = INPUT_2D.mean(axis=0)  # mean over time
            refstd = INPUT_2D.std(axis=0)  # std over time

            REF_DATASET = DATASET.sel(time=slice("1980", "2000"))
            REF_ARRAY = np.transpose(
                np.asarray([REF_DATASET[i].values for i in VAR_LIST]), [1, 2, 3, 0]
            )

            # standardize mean
            ref_means = REF_ARRAY.mean(axis=(1, 2))
            ref_means_mean = ref_means.mean(axis=0)
            ref_means_std = ref_means.std(axis=0)
            vect_means_stdz = (vect_means - ref_means_mean) / ref_means_std

            # standardize std
            ref_stds = REF_ARRAY.std(axis=(1, 2))
            ref_stds_mean = ref_stds.mean(axis=0)
            ref_stds_std = ref_stds.std(axis=0)
            vect_std_stdz = (vect_std - ref_stds_mean) / ref_stds_std

            SpatialMean = vect_means_stdz.reshape(
                INPUT_2D.shape[0], 1, 1, INPUT_2D.shape[3]
            )
            SpatialSTD = vect_std_stdz.reshape(
                INPUT_2D.shape[0], 1, 1, INPUT_2D.shape[3]
            )

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

    INPUT_1D_ARRAY = np.concatenate(INPUT_1D, axis=3)
    if print_:
        print(f"INPUT_1D shape: {INPUT_1D_ARRAY.shape}")

    DATASET.close()
    if GCM != None:
        # print('Return input of GCM')
        DATASETGCM.close()
        return INPUT_2D_ARRAY_GCM, INPUT_1D_ARRAY, VAR_LIST
    else:
        # print('Return input of UPRCM')
        return INPUT_2D_ARRAY, INPUT_1D_ARRAY, VAR_LIST


def make_inputs(
    UPRCM,
    GCM,
    size_input_domain: int,
    Region: str,
    dropvarRCM,
    dropvarGCM,
):  # for combined regions, each sample gets a number so that we know to which region it corresponds
    # Make input
    i2D, i1D, VAR_LIST = input_maker(
        UPRCM=UPRCM,
        GCM=GCM,
        size_input_domain=size_input_domain,
        stand=True,  # standardization
        standZ=True,
        seas=True,  # put a cos,sin vector to control the season, format : bool
        means=True,  # add the mean of the variables raw or stdz, format : r,s,n
        stds=True,
        resize_input=True,
        region=Region,
        print_=False,
        dropvarRCM=dropvarRCM,
        dropvarGCM=dropvarGCM,  # variables to drop if necessary
    )

    # Make a non standardised version for plots:
    i2D_ns, i1D_ns, var_list = input_maker(
        UPRCM=UPRCM,
        GCM=GCM,
        size_input_domain=size_input_domain,
        stand=False,  # standardization
        standZ=True,
        seas=True,  # put a cos,sin vector to control the season, format : bool
        means=True,  # add the mean of the variables raw or stdz, format : r,s,n
        stds=True,
        resize_input=False,
        region=Region,
        print_=False,
        dropvarRCM=dropvarRCM,
        dropvarGCM=dropvarGCM,
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

    lowerTarget = createLowerTarget(
        target_dataset, region=region, Nx=64, Ny=64, print_=False
    )
    targetArray = lowerTarget.SMB.values

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

def regionEncoder(X, region, regions):
    # Indicator of regions and their order if combined dataset
    # Encoding 0-> Num regions
    r_ = []
    if region == "Combined":
        N = int(len(X) / len(regions))
        for j in range(len(regions)):
            for i in range(N):
                r_.append(j)
    else:
        N = len(X)
        r_ = [0 for i in range(N)]
    R = torch.tensor(r_)  # Change to tensor
    return R


"""standardize: standardizes data array by substracting mean and dividing by std
"""


def standardize(data, mean=None, std=None, ownstd=True):
    import numpy as np

    if ownstd:
        mean = np.nanmean(data, axis=(1, 2), keepdims=True)
        std = np.nanstd(data, axis=(1, 2), keepdims=True)
    ndata = (data - mean) / std
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
