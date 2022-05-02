""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
		"""(convolution => [BN] => ReLU) * 2"""
	
		def __init__(self, in_channels, out_channels, mid_channels=None):
				super().__init__()
				if not mid_channels:
						mid_channels = out_channels
				self.double_conv = nn.Sequential(
						nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
						nn.BatchNorm2d(mid_channels),
						nn.ReLU(inplace=True),
						nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
						nn.BatchNorm2d(out_channels),
						nn.ReLU(inplace=True)
				)
			
		def forward(self, x):
				return self.double_conv(x)
	
	
class Down(nn.Module):
		"""Downscaling with double conv then maxpool"""
	
		def __init__(self, in_channels, out_channels):
				super().__init__()
				self.maxpool_conv = nn.Sequential(
						nn.MaxPool2d(2),
						DoubleConv(in_channels, out_channels)
				)
				self.maxpool = nn.MaxPool2d(2)
				self.doubleconv = DoubleConv(in_channels, out_channels)
			
		def forward(self, x):
				x1 = self.doubleconv(x)
				x2 = self.maxpool(x1)
				return x2, x1
	
class Up(nn.Module):
		"""Upscaling then double conv"""
		def __init__(self, in_channels, out_channels, bilinear=True):
				super().__init__()
			
				# if bilinear, use the normal convolutions to reduce the number of channels
				if bilinear:
						self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
						self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
				else:
						self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2)
						self.conv = DoubleConv(in_channels, out_channels)
					
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
	
	
class UpLastConv(nn.Module):
		"""Upscaling then double conv without concatenating with skip connection"""
		def __init__(self, in_channels, out_channels, mid_channels, bilinear=True):
				super().__init__()
			
				# if bilinear, use the normal convolutions to reduce the number of channels
				if bilinear:
						self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
						self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
				else:
						self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2)
						self.conv = DoubleConv(mid_channels, out_channels)
					
		def forward(self, x, zsize=64):
				x1 = self.up(x)
				# input is CHW
				diffY = zsize - x1.size()[2]
				diffX = zsize - x1.size()[3]
			
				x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
												diffY // 2, diffY - diffY // 2])
				return self.conv(x1)
	
class InitialConv(nn.Module):
		def __init__(self, in_channels, out_channels, sizex, size):
				super(InitialConv, self).__init__()
				self.diff = sizex-size+1
				self.firstconv = nn.Sequential(
						nn.Conv2d(in_channels, out_channels, kernel_size=self.diff), 
						nn.BatchNorm2d(out_channels),
						nn.ReLU(inplace=True)
				)
		def forward(self, x):
				x1 = self.firstconv(x)
				return x1
	
class OutConv(nn.Module):
		def __init__(self, in_channels, out_channels):
				super(OutConv, self).__init__()
				self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
			
		def forward(self, x):
				return self.conv(x)
	
class LinearUpsamplingZ(nn.Module):
	# Linear upsampling of z to same number of channels as downsampled X
	# so that can concatenate
	def __init__(self, in_channels=16, out_channels=1024):
		super(LinearUpsamplingZ, self).__init__()
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
	