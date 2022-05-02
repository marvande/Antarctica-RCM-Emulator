#!/usr/bin/env python3

from datetime import date
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from config import *
import logging
import wandb
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm 
from pathlib import Path
from matplotlib import pyplot as plt
import random as rn
import pandas as pd
import numpy as np
from google.colab import files

from GC_scripts import * # Google cloud scripts
from helperFunctions import *


def train_net(
	net,
	dataset,
	device,
	epochs: int = 5,
	batch_size: int = 1,
	learning_rate: float = 1e-5,
	val_percent: float = 0.1,
	save_checkpoint: bool = True,
	amp: bool = False,
	dir_checkpoint: str = Path("./checkpoints/"), 
  region: str = 'Larsen'
):
	# 2. Split into train / validation partitions
	n_val = int(len(dataset) * val_percent)
	n_train = len(dataset) - n_val
	train_set, val_set = random_split(
		dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
	)
	logging.info(f"Train set size: {n_val}\n" f"Validation set size: {n_train}\n")
	# 3. Create data loaders
	loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
	train_loader = DataLoader(train_set, shuffle=True, **loader_args)
	val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
	
	# (Initialize logging)
	experiment = wandb.init(project="U-Net", resume="allow", anonymous="must")
	experiment.config.update(
		dict(
			epochs=epochs,
			batch_size=batch_size,
			learning_rate=learning_rate,
			val_percent=val_percent,
			save_checkpoint=save_checkpoint,
			amp=amp,
		)
	)
	logging.info(
		f"""Starting training:
		Epochs:          {epochs}
		Batch size:      {batch_size}
		Learning rate:   {learning_rate}
		Training size:   {n_train}
		Validation size: {n_val}
		Checkpoints:     {save_checkpoint}
		Device:          {device.type}
		Mixed Precision: {amp}
	"""
	)
	
	# 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
	optimizer = optim.Adam(net.parameters(), lr=learning_rate)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, "min", patience=4, verbose=1
	)  # goal: minimize loss
	grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
	criterion = nn.MSELoss()
	global_step = 0
	
	# 5. Begin training
	train_loss_e, val_loss_e = [], [] # MSE per epoch
	train_rmse_e, val_rmse_e = [], [] # RMSE per epoch
	for epoch in range(1, epochs + 1):
		net.train()
		epoch_loss = 0
		with tqdm(
			total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="timestep"
		) as pbar:
			train_loss, val_loss = 0, 0 # MSE
			train_rmse, val_rmse = 0, 0 # RMSE
			
			for batch in train_loader:
				X_train, Z_train, Y_train, R_train = (
					batch[0],
					batch[1],
					batch[2],
					batch[3],
				)
				X_train_cuda = X_train.to(device=device, dtype=torch.float32)
				Z_train_cuda = Z_train.to(device=device, dtype=torch.float32)
				true_smb = Y_train.to(device=device, dtype=torch.float32)
				
				with torch.cuda.amp.autocast(enabled=amp):
					smb_pred = net(X_train_cuda, Z_train_cuda)
					loss = criterion(smb_pred, true_smb)
					train_loss += loss.item()  # mse for plots
					
					# evaluation metrics:
					train_rmse += torch.sqrt(loss).item() # rmse
					
					
				optimizer.zero_grad(set_to_none=True)
				grad_scaler.scale(loss).backward()
				grad_scaler.step(optimizer)
				grad_scaler.update()
				
				pbar.update(X_train.shape[0])
				global_step += 1
				epoch_loss += loss.item()
				experiment.log(
					{"train loss": loss.item(), "step": global_step, "epoch": epoch}
				)
				pbar.set_postfix(**{"loss (batch)": loss.item()})
				
				# Evaluation round
				division_step = n_train // (10 * batch_size)
				if division_step > 0:
					if global_step % division_step == 0:
						histograms = {}
						val_score, val_rmse_score = evaluate(net, val_loader, device, criterion)
						val_loss += val_score.item() # mse
						
						val_rmse += val_rmse_score # rmse
						
						scheduler.step(val_score)
						
						# logging.info('Validation MSE loss: {}'.format(val_score))
						experiment.log(
							{
								"learning rate": optimizer.param_groups[0]["lr"],
								"validation mse": val_score,
								"images": wandb.Image(X_train_cuda[0, 0, :, :].cpu()),
								"masks": {
									"true": wandb.Image(true_smb[0].cpu()),
									"pred": wandb.Image(smb_pred[0].cpu()),
								},
								"step": global_step,
								"epoch": epoch,
								**histograms,
							}
						)
						
		if save_checkpoint:
			Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
			torch.save(
				net.state_dict(),
				str(dir_checkpoint / "checkpoint_epoch{}.pth".format(epoch)),
			)
			logging.info(f"Checkpoint {epoch} saved!")
			
		train_loss_e.append(
			train_loss / len(train_loader)
		)  # return train loss divided by num batches
		
		train_rmse_e.append(train_rmse / len(train_loader))		
		val_loss_e.append(val_loss / len(val_loader))
		val_rmse_e.append(val_rmse / len(val_loader)) 
		
	# Save final model:
	Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
	today = str(date.today())
	nameSave = f"MODEL_{today}_{region}_{epochs}_{batch_size}.pth"
	torch.save(
		net.state_dict(),
		str(dir_checkpoint / nameSave),
	)
	
	# Upload final model to GC:
	pathGC = 'Chris_data/RawData/MAR-ACCESS1.3/monthly/SavedModels/'
	uploadFileToGC(pathGC, str(dir_checkpoint / nameSave))
	
	# Outputs of Losses for plots:
	train_loss_out = {'MSE':train_loss_e, 'RMSE':train_rmse_e}
	val_loss_out = {'MSE':val_loss_e, 'RMSE':val_rmse_e}
	return train_loss_out, val_loss_out 


def evaluate(net, dataloader, device, criterion):
	net.eval()
	num_val_batches = len(dataloader)
	mse_loss = 0
	rmse = 0
	
	# iterate over the validation set
	# for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
	for batch in dataloader:
		X_val, Z_val, Y_val, R_val = batch[0], batch[1], batch[2], batch[3]
		# move images and labels to correct device and type
		X_val_cuda = X_val.to(device=device, dtype=torch.float32)
		Z_val_cuda = Z_val.to(device=device, dtype=torch.float32)
		true_smb = Y_val.to(device=device, dtype=torch.float32)
		
		with torch.no_grad():
			# predict the mask
			smb_pred = net(X_val_cuda, Z_val_cuda)
			mse_loss += criterion(smb_pred, true_smb)
			
			# rmse:
			rmse += torch.sqrt(mse_loss).item()
	net.train()
	# Fixes a potential division by zero error
	if num_val_batches == 0:
		return mse_loss
	
	mse_e  = mse_loss / num_val_batches
	rmse_e  = rmse / num_val_batches
	return mse_e, rmse_e


def predict(net, device, test_loader, model, 
	dir_checkpoint: str = Path("./checkpoints/")):
	logging.info(f"Loading saved model {model}")
	logging.info(f"Using device {device}")
	
	net.to(device=device)
	net.load_state_dict(torch.load(str(dir_checkpoint / model), map_location=device))
	logging.info("Saved model loaded!")
	
	preds = []
	x = []
	z = []
	y = []
	r = []
	
	for batch in tqdm(
		test_loader,
		total=len(test_loader),
		desc="Testing round",
		unit="batch",
		leave=False,
	):
		X_test, Z_test, Y_test, R_test = batch[0], batch[1], batch[2], batch[3]
		X_test_cuda = X_test.to(device=device, dtype=torch.float32)
		Z_test_cuda = Z_test.to(device=device, dtype=torch.float32)
		true_smb = Y_test.to(device=device, dtype=torch.float32)
	
		net.eval()
		prediction = net(X_test_cuda, Z_test_cuda)
		prediction = prediction.cpu().detach().numpy()
		preds.append(prediction.transpose(0, 2, 3, 1)[0])  # change to numpy
		x.append(X_test.numpy().transpose(0, 2, 3, 1)[0])
		z.append(Z_test.numpy().transpose(0, 2, 3, 1)[0])
		y.append(Y_test.numpy().transpose(0, 2, 3, 1)[0])
		r.append(R_test.numpy()[0])
	
	return preds, x, z, y, r


def plotRandomPredictions(preds, x, z, true_smb, r, GCMLike, VAR_LIST, target_dataset, N = 10):
	today = str(date.today())
	
	f = plt.figure(figsize=(20, 60))
	map_proj = ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)
	
	for i in range(N):
			randTime = rn.randint(0, len(preds)-1)
			sample2dtest_, sample_z, sampletarget_, samplepred_  = x[randTime], z[randTime], true_smb[randTime], preds[randTime]
			region = REGIONS[r[randTime]] # region of sample
			dt = pd.to_datetime([GCMLike.time.isel(time=randTime).values])
			time = str(dt.date[0])
		
			if region != "Whole Antarctica":
					sample2dtest_ = resize(sample2dtest_, 25, 48, print_=False)
			else:
					sample2dtest_ = resize(sample2dtest_, 25, 90, print_=False)
				
			vmin = np.min([sampletarget_, samplepred_])
			vmax = np.max([sampletarget_, samplepred_])
		
			M = 3
			for m in range(M):
					if m == 0:
							ax = plt.subplot(N, M, (i * M) + m + 1, projection=ccrs.SouthPolarStereo())
							plotTrain(GCMLike, sample2dtest_, 4, ax, time, VAR_LIST, region=region)
					if m == 1:
							ax = plt.subplot(N, M, (i * M) + m + 1, projection=ccrs.SouthPolarStereo())
							plotTarget(target_dataset, sampletarget_, ax, vmin, vmax, region=region)
					if m == 2:
							ax = plt.subplot(N, M, (i * M) + m + 1, projection=ccrs.SouthPolarStereo())
							plotPred(target_dataset, samplepred_, ax, vmin, vmax, region=region)						
	nameFig = f'{today}_pred_{REGION}_{NUM_EPOCHS}_{BATCH_SIZE}.png'
	plt.savefig(nameFig)
	files.download(nameFig)

def plotLoss(train_loss_e, val_loss_e, metric = 'MSE'):
    f = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 2, 1)
    ax.plot(train_loss_e)
    ax.set_title(f'Training {metric} for {NUM_EPOCHS} epochs')
    ax.set_xlabel('Num epochs')
    ax = plt.subplot(1, 2, 2)
    ax.plot(val_loss_e)
    ax.set_title(f'Validation {metric} for {NUM_EPOCHS} epochs')
    ax.set_xlabel('Num epochs')