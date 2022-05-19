#!/usr/bin/env python3

from torch import nn
import torch
from unet_parts_attention import LinearUpsamplingZ
from unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS, UpLastConv, OutConv, InitialConv
from layers import CBAM


class SmaAt_UNet(nn.Module):
	def __init__(self, n_channels_x, n_channels_z, kernels_per_layer=2, bilinear=False, reduction_ratio=16):
		super(SmaAt_UNet, self).__init__()
		self.n_channels_x = n_channels_x
		self.n_channels_z = n_channels_z
		self.size = 32
		#self.n_classes = n_classes
		kernels_per_layer = kernels_per_layer
		self.bilinear = bilinear
		reduction_ratio = reduction_ratio
		
		# input  (batch, 7, 32, 32)
		# first conv:
		self.inc = InitialConv(self.n_channels_x, 32, 32, self.size)
		
		# linewar downsampling of z:
		self.linz = LinearUpsamplingZ(self.n_channels_z , 1024)
		
		#self.inc = DoubleConvDS(self.n_channels_x, 32, kernels_per_layer=kernels_per_layer)
		self.cbam1 = CBAM(32, reduction_ratio=reduction_ratio)
		self.down1 = DownDS(32, 64, kernels_per_layer=kernels_per_layer)
		self.cbam2 = CBAM(64, reduction_ratio=reduction_ratio)
		self.down2 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
		self.cbam3 = CBAM(128, reduction_ratio=reduction_ratio)
		self.down3 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
		self.cbam4 = CBAM(256, reduction_ratio=reduction_ratio)
		self.down4 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
		self.cbam5 = CBAM(512, reduction_ratio=reduction_ratio)
		
		factor = 2 if self.bilinear else 1
		self.down5 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
		self.cbam6 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
		
		self.up1 = UpDS(2048, 1024 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
		self.up2 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
		self.up3 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
		self.up4 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
		self.up5 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
		
		self.up6 = UpDS(64, 32, self.bilinear, kernels_per_layer=kernels_per_layer)
		self.up7 = UpLastConv(32, 32, 16, self.bilinear)
		
		# self last layer:
		self.outconv = OutConv(32, 1)
		
	def forward(self, x, z):
		# input  (batch, 7, 32, 32)
		x1 = self.inc(x) # (batch, 32, 32, 32)
		x1Att = self.cbam1(x1) # (batch, 32, 32, 32)
		
		# downsampling
		x2 = self.down1(x1) 	# (batch, 64, 16, 16)
		x2Att = self.cbam2(x2)  
		x3 = self.down2(x2) 	# (batch, 128, 8, 8)
		x3Att = self.cbam3(x3) 	
		x4 = self.down3(x3) 	# (batch, 256, 4, 4)
		x4Att = self.cbam4(x4) 	
		x5 = self.down4(x4) 	# (batch, 512, 2, 2)
		x5Att = self.cbam5(x5) 	
		x6 = self.down5(x5) 	# (batch, 1024, 1, 1)
		x6Att = self.cbam6(x6)
		
		# Upsample second input to have same shape as last_conv [1024, 1, 1]
		z1 = self.linz(z)
		# attention: 
		z1Att = self.cbam6(z1) # (batch, 1024, 1, 1)
		
		# concatenate both:
		x8 = torch.cat([x6Att, z1Att], dim=1)  # (batch, 2048, 1, 1)
		
		x = self.up1(x8, x6Att) # (batch, 1024, 1, 1)
		x = self.up2(x, x5Att) # (batch, 512, 2, 2)
		x = self.up3(x, x4Att) # (batch, 256, 4, 4)
		x = self.up4(x, x3Att) # (batch, 128, 8, 8)
		x = self.up5(x, x2Att) # (batch, 64, 16, 16)
		x = self.up6(x, x1Att) # (batch, 32, 32, 32)
		
		# Last layer: change from 32x32 to 64x64
		x = self.up7(x)  # (batch, 32, 64, 64) 
		
		
		# Change num channels from 32 to 1
		# (None, 64, 64, 1)
		x = self.outconv(x)
		
		return x