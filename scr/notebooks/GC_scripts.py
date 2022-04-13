from google.cloud import storage
from re import search
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

# Google cloud info
PROJECT = 'ee-iceshelf-gee4geo'
BUCKET = "ee-downscalingclimatemodels"

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET)


def listFilesGC(pathGC, VAR):
    # Get all files already on GC in a bucket
    filesGC = []
    for blob in storage_client.list_blobs(bucket, prefix=pathGC):
        file_ = str(blob)
        if search(f'{VAR}_', file_):
            span = search(f"{VAR}_ant(.*?).nc", file_).span(0)
            filesGC.append(file_[span[0]:span[1]])
    return filesGC

#pathGC = f'Chris_data/RawData/MAR-ACCESS1.3/{VAR}/'
# Download single file
def downloadFileFromGC(pathGC, pathLocal, fileGC):
    # Download from GC locally
    blob = bucket.blob(pathGC + fileGC)
    blob.download_to_filename(pathLocal+fileGC)

# Download multiple files    
def downloadFilesFromGC(pathGC, pathLocal, filesGC):
    N = len(filesGC)
    for i in tqdm(range(N)):
        file_name = filesGC[i]
        downloadFileFromGC(pathGC, pathLocal, file_name)
        
# Upload single files  
def uploadFileToGC(pathGC, fileGC):
    # upload to google cloud:
    blob = bucket.blob(pathGC+fileGC)
    blob.upload_from_filename(fileGC)
    
# Upload multiple files  
def uploadFilesToGC(pathGC, filesGC):
    N = len(filesGC)
    for i in tqdm(range(N)):
        # upload to google cloud:
        file_name = filesGC[i]
        blob = bucket.blob(pathGC+file_name)
        blob.upload_from_filename(file_name)
        
def pathToFiles(VAR, date1='19800101', date2='19801231'):
    pathGC = f'Chris_data/RawData/MAR-ACCESS1.3/{VAR}/'
    fileGC = f'{VAR}_ant-35km_ACCESS1.3_rcp8.5_r1i1p1_ULg-MAR311_v1_day_{date1}-{date2}.nc'
    return pathGC, fileGC

def filesInDir(pathLocal):
    return sorted([f for f in listdir(pathLocal) if isfile(join(pathLocal, f))])

def empty_dir(pathLocal):
  # delete all files as precaution
  for file_name in os.listdir(pathLocal):
    # construct full file path
    file = pathLocal + file_name
    if os.path.isfile(file):
        os.remove(file)
        
def filesInDirWithVar(pathLocal, VAR):
    files = filesInDir(pathLocal)
    filesWithVar = []
    for f in files:
        if search(f'{VAR}_', f):
            filesWithVar.append(f)
    return filesWithVar