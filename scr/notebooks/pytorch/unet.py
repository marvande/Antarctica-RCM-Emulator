""" Full assembly of the parts to form the complete network """
from unet_parts import *

class UNetMarijn(nn.Module):
	def __init__(self, n_channels_x=7, n_channels_z=16, size=32, filter=64, bilinear=False):
		super(UNetMarijn, self).__init__()
		self.n_channels_x = n_channels_x
		self.n_channels_z = n_channels_z
		self.size = size
		self.filter = filter
		self.bilinear = bilinear
		
		# input  (batch, 7, 32, 32)
		# first conv:
		self.initialconv = InitialConv(n_channels_x, 32, 32, self.size) 
		
		# downsampling:
		self.d1 = Down(32, 64)
		self.d2 = Down(64, 128)
		self.d3 = Down(128, 256)
		self.d4 = Down(256, 512)
		self.d5 = Down(512, 1024)
		
		# last conv of downsampling
		self.last_conv_down = DoubleConv(1024, 1024)
		
		# linewar downsampling of z:
		self.linz = LinearUpsamplingZ(n_channels_z, 1024)
		
		# upsampling:
		self.up1 = Up(2048, 1024, bilinear)
		self.up2 = Up(1024, 512, bilinear)
		self.up3 = Up(512, 256, bilinear)
		self.up4 = Up(256, 128, bilinear)
		self.up5 = Up(128, 64, bilinear)
		self.up6 = UpLastConv(64, 64, 32, bilinear)
		
		# self last layer:
		self.up7 = OutConv(64, 1)
		
	def forward(self, x,z):
		# input  (batch, 7, 32, 32)
		x1 = self.initialconv(x) # (batch, 32, 32, 32)
		
		# Downsampling 
		x2, x2_bm = self.d1(x1)  # (batch, 64, 16, 16)
		x3, x3_bm = self.d2(x2)  # (batch, 128, 8, 8)
		x4, x4_bm = self.d3(x3)  # (batch, 256, 4, 4)
		x5, x5_bm = self.d4(x4)  # (batch, 512, 2, 2)
		x6, x6_bm = self.d5(x5)  # (batch, 1024, 1, 1)
		x7 = self.last_conv_down(x6) # (batch, 1024, 1, 1)
		
		# Upsample second input to have same shape as last_conv [1024, 1, 1]
		z1 = self.linz(z)
		
		# concatenate both:
		x8 = torch.cat([x7, z1], dim = 1) # (batch, 2048, 1, 1)
		
		# Upsampling
		x = self.up1(x8, x6_bm) # (batch, 1024, 2, 2)
		x = self.up2(x, x5_bm)  # (batch, 512, 4, 4)
		x = self.up3(x, x4_bm)  # (batch, 256, 8, 8)
		x = self.up4(x, x3_bm)  # (batch, 128, 16, 16)
		x = self.up5(x, x2_bm)  # (batch, 64, 32, 32)
		
		# Last layer: change from 32x32 to 64x64
		x = self.up6(x) # (batch, 64, 64, 64)
		
		# Change num channels from 64 to 1
		# (None, 64, 64, 1)
		x = self.up7(x)
		return x