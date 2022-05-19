""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


	
class LinearUpsamplingZ(nn.Module):
	# Linear upsampling of z to same number of channels as downsampled X
	# so that can concatenate
	def __init__(self, in_channels=16, out_channels=1024):
		super().__init__()
		self.linearUp = nn.Sequential(
				nn.Linear(in_channels, 128), 
				nn.Linear(128, 256),
				nn.Linear(256, 512),
				nn.Linear(512, 1024)
		)
	def forward(self, z):
		# transform to remove last dimensions
		z1 = z.view(z.shape[0], -1) # (batch, 16)
		z5 = self.linearUp(z1.float()) # (batch, 1024)
		# transform back to original shape
		z6 = z5.view(z5.shape[0], 1,1, -1).transpose(3,1) # (batch, 1024, 1, 1)
		return z6
	
	
