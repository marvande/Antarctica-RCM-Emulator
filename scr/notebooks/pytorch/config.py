# Paths to data
pathGC = f'Chris_data/RawData/MAR-ACCESS1.3/monthly/RCM/'
pathCluster = '../../../../../../mlodata1/marvande/data/'

# If files are local or on GC
DOWNLOAD_FROM_GC = True

# FileNames
fileTarget = 'MAR(ACCESS1-3)_monthly.nc'
fileGCMLike = 'MAR(ACCESS1-3)-stereographic_monthly_GCM_like.nc'


# Choose either on of the regions below, or Combined
# which will use all of them
REGION ="Combined"
REGIONS = ['Larsen', 'Wilkes', 'Maud', 'Amundsen']

# Training param
BATCH_SIZE = 32
NUM_EPOCHS = 100
LR = 0.005
VAL_PERCENT = 0.2
TEST_PERCENT = 0.1
SAVE_CHECKPOINT = True
AMP = False
SEED = 123
