# Constants:
DOWNLOAD_FROM_GC = False

fileUPRCM = "MAR(ACCESS1-3)-stereographic_monthly_GCM_like.nc"
fileTarget = "MAR(ACCESS1-3)_monthly.nc"
fileGCM = "ACCESS1-3-stereographic_monthly_cleaned.nc"

REGION = 'Larsen' # Antarctic Peninsula
SIZE_INPUT_DOMAIN = 32  # size of input to model (GCM) batch x SIZE_INPUT_DOMAIN x SIZE_INPUT_DOMAIN x num_channels

# Training constants:
BATCH_SIZE = 100
SEED = 0
TEST_PERCENT = 0.1  # size of test set and training set
VAL_PERCENT = 0.2  # size of validation set and training set
NUM_EPOCHS = 50
LR = 0.005