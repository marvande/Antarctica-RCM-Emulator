#!/usr/bin/env python3

""" Parts of the U-Net model """
# Base model taken from: https://github.com/milesial/Pytorch-UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import DepthwiseSeparableConv


class DoubleConvDS(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""
	
	def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)
		
	def forward(self, x):
		return self.double_conv(x)
	
	
class UpLastConv(nn.Module):
		"""Upscaling then double conv without concatenating with skip connection"""
		def __init__(self, in_channels, out_channels, mid_channels, bilinear=True, kernels_per_layer=1):
				super().__init__()
			
				# if bilinear, use the normal convolutions to reduce the number of channels
				if bilinear:
						self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
						#self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
						self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
				else:
						self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2)
						#self.conv = DoubleConv(mid_channels, out_channels)
						self.conv = DoubleConvDS(mid_channels, out_channels, kernels_per_layer=kernels_per_layer)
					
		def forward(self, x, zsize=64):
				x1 = self.up(x)
				# input is CHW
				diffY = zsize - x1.size()[2]
				diffX = zsize - x1.size()[3]
			
				x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
												diffY // 2, diffY - diffY // 2])
				return self.conv(x1)
	
	
class DownDS(nn.Module):
	"""Downscaling with maxpool then double conv"""
	
	def __init__(self, in_channels, out_channels, kernels_per_layer=1):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
		)
		
	def forward(self, x):
		return self.maxpool_conv(x)
	
	
class UpDS(nn.Module):
	"""Upscaling then double conv"""
	
	def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=1):
		super().__init__()
		
		# if bilinear, use the normal convolutions to reduce the number of channels
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
		else:
			self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2)
			self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
			
	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]
		
		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		# if you have padding issues, see
		# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
		# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)
	
	
class OutConv(nn.Module):
		def __init__(self, in_channels, out_channels):
				super().__init__()
				self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
			
		def forward(self, x):
				return self.conv(x)
		
		
class InitialConv(nn.Module):
		def __init__(self, in_channels, out_channels, sizex, size):
				super().__init__()
				self.diff = sizex-size+1
				self.n_channels_in = in_channels
				self.n_channels_out = out_channels
				self.firstconv = nn.Sequential(
						nn.Conv2d(self.n_channels_in, self.n_channels_out, kernel_size=self.diff), 
						nn.BatchNorm2d(out_channels),
						nn.ReLU(inplace=True)
				)
		def forward(self, x):
				x1 = self.firstconv(x)
				return x1