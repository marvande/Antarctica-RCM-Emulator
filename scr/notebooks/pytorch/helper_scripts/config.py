"""
Config file with all big constants needed
"""

# If files are local or on GC
DOWNLOAD_FROM_GC = False

# Paths to data
# paths to data and modelson google cloud (not used,  load from google drive)
pathGC = f"Chris_data/RawData/MAR-ACCESS1.3/monthly/RCM/"
pathModel = "Chris_data/RawData/MAR-ACCESS1.3/monthly/SavedModels/checkpoints/"

# path to data on ic cluster (not used, load from google drive)
pathCluster = "../../../../../../mlodata1/marvande/data/"


# FileNames
fileTarget = "MAR(ACCESS1-3)_monthly.nc"
fileTargetSmoothed = "MAR(ACCESS1-3)_monthly_smoothed.nc"
fileGCMLike = "MAR(ACCESS1-3)-stereographic_monthly_GCM_like.nc"
fileGCM = "ACCESS1-3-stereographic_monthly_cleaned.nc"

# Choose either on of the regions below, or Combined
# which will use all of them
REGION = "Larsen"
# all possible regions
REGIONS = ["Larsen", "Wilkes", "Maud", "Amundsen"]

# Parameters for model:
SIZE_INPUT_DOMAIN = 32 # size of input to model (GCM) batch x SIZE_INPUT_DOMAIN x SIZE_INPUT_DOMAIN x num_channels 
BATCH_SIZE = 32 
NUM_EPOCHS = 100
LR = 0.005
TEST_PERCENT = 0.1 # size of test set and training set
VAL_PERCENT = 0.2 # size of validation set and training set
SAVE_CHECKPOINT = True
AMP = False
SEED = 0
TYPENET = "Attention" # type of model
LOSS_ = "NRMSE" # loss model uses
