"""PatchGAN Discriminator model for the pix2pix architecture"""

import torch
import torch.nn as nn

class PatchGAN(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,ndf:int=64):
        super().__init__()
        self.leaky_relu1 = nn.LeakyReLU(0.2,inplace=True)
        self.leaky_relu2 = nn.LeakyReLU(0.2,inplace=True)
        self.leaky_relu3 = nn.LeakyReLU(0.2,inplace=True)
        self.leaky_relu4 = nn.LeakyReLU(0.2,inplace=True)
        self.leaky_relu5 = nn.LeakyReLU(0.2,inplace=True)
        self.batch_norm1 = nn.BatchNorm2d(ndf * 2)
        self.batch_norm2 = nn.BatchNorm2d(ndf * 4)
        self.batch_norm3 = nn.BatchNorm2d(ndf * 8)
        self.batch_norm4 = nn.BatchNorm2d(ndf * 8)
        self.conv1 = nn.Conv2d(in_channels * 2,ndf,kernel_size=4,stride=2,padding=1)
        self.conv2 = nn.Conv2d(ndf,ndf * 2,kernel_size=4,stride=2,padding=1)
        self.conv3 = nn.Conv2d(ndf * 2,ndf * 4,kernel_size=4,stride=2,padding=1)
        self.conv4 = nn.Conv2d(ndf * 4,ndf * 8,kernel_size=4,stride=2,padding=1)
        self.conv5 = nn.Conv2d(ndf * 8,ndf * 8,kernel_size=3,stride=1,padding=1)
        self.conv6 = nn.Conv2d(ndf * 8,out_channels,kernel_size=3,stride=1,padding=1)
        self.final_activation = nn.Sigmoid() # we no longer use this only use with BCE Loss in the final architecture

    def forward(self,x:torch.Tensor,y:torch.Tensor) -> torch.Tensor:
        x = torch.cat([x,y],dim=1)
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu3(x)
        x = self.conv4(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu4(x)
        x = self.conv5(x)
        x = self.batch_norm4(x)
        x = self.leaky_relu5(x)
        x = self.conv6(x) 
        return x
