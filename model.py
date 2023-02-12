import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm
from pickle import load, dump

import os
import cv2
import random
import datetime
import argparse


class Mish(nn.Module):
    @staticmethod
    def mish(x):
        return x * torch.tanh(F.softplus(x))
    
    def forward(self, x):
        return Mish.mish(x)

class GLU(nn.Module):
    def forward(self, x):
        channel = x.size(1)
        assert channel % 2 == 0, 'must divide by 2.'
        return x[:, :channel//2] * torch.sigmoid(x[:, channel//2:])

class PixelwiseNormalization(nn.Module):
    def pixel_norm(self, x):
        eps = 1e-8
        return x * torch.rsqrt(torch.mean(x * x, 1, keepdim=True) + eps)
    
    def forward(self, x):
        return self.pixel_norm(x)

class GeneratorBlock(nn.Module):
    def __init__(self, input_nc, output_nc, n_channel):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(input_nc, output_nc * 2, kernel_size=3, stride=1, padding=1)
        self.normalize = PixelwiseNormalization()
        self.activate = GLU()
        
    def forward(self, image):
        image = self.upsample(image)
        image = self.conv(image)
        image = self.normalize(image)
        image = self.activate(image)
        
        return image

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1),  # downsample
            PixelwiseNormalization(),
            Mish(),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1)
        )
        
        self.conv = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)
        )
           
        self.activation = nn.Sequential(
            PixelwiseNormalization(),
            Mish()
        )

    def forward(self, x):
        out = self.model(x)
        skip = self.conv(x)
        out = out + skip
        out = self.activation(out)
        
        return out

class Generator(nn.Module):
    def __init__(self, num_depth, num_fmap, n_channel=1):
        super().__init__()
        
        self.num_depth = num_depth
        self.blocks = nn.ModuleList([GeneratorBlock(num_fmap(i), num_fmap(i + 1), n_channel) for i in range(num_depth)])
        self.toRGB = nn.Sequential(
            nn.Conv2d(num_fmap(num_depth), n_channel, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        x = self.toRGB(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, num_depth, num_fmap, n_channel=1):
        super().__init__()
        
        self.fromRGB = nn.Conv2d(n_channel, num_fmap(num_depth), kernel_size=3, stride=1, padding=1)
        self.blocks = nn.ModuleList([DiscriminatorBlock(num_fmap(i+1), num_fmap(i)) for i in range(num_depth)][::-1])
        
        self.z_decoder = nn.Conv2d(num_fmap(0), num_fmap(0), kernel_size=4, stride=2, padding=1)
        
        self.conv_feature = nn.Conv2d(num_fmap(0) + num_fmap(0), num_fmap(0), kernel_size=3, stride=1, padding=1)
        self.conv_patch = nn.Conv2d(num_fmap(0), 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, z):
        x = self.fromRGB(x)
            
        for block in self.blocks:
            x = block(x)
        
        z = self.z_decoder(z).expand(x.shape)
        feature = self.conv_feature(torch.cat([x, z], dim=1))
        out = self.conv_patch(feature)
        
        return out, feature

class Encoder(nn.Module):
    class BasicBlock(nn.Module):
        def __init__(self, dim_in, dim_out):
            super().__init__()
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim_in, dim_out * 2, kernel_size=3, stride=1, padding=1),
                PixelwiseNormalization(),
                GLU()
            )
        def forward(self, x):
            return self.block(x)
    
    def __init__(self, num_depth, num_fmap, n_channel=1):
        super().__init__()
        
        self.fromRGB = nn.Conv2d(n_channel, num_fmap(num_depth), kernel_size=3, stride=1, padding=1)
        self.blocks = nn.ModuleList([DiscriminatorBlock(num_fmap(i+1), num_fmap(i)) for i in range(num_depth)][::-1])
        self.to_z = Encoder.BasicBlock(num_fmap(0), num_fmap(0))
    
    def forward(self, x):
        x = self.fromRGB(x)
        
        for block in self.blocks:
            x = block(x)
        
        z = F.adaptive_avg_pool2d(x, (1, 1))
        z = self.to_z(z)
        
        return z