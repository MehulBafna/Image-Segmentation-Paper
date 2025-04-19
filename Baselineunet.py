import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
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

class SimpleUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=5, base_channels=64):
        super(SimpleUNet, self).__init__()
        
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels, base_channels*2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels*2, base_channels*4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels*4, base_channels*8)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels*8, base_channels*16)
        )
        
        self.bottleneck = DoubleConv(base_channels*16, base_channels*16)
        
        self.up1 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(base_channels*16, base_channels*8)
        
        self.up2 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(base_channels*8, base_channels*4)
        
        self.up3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(base_channels*4, base_channels*2)
        
        self.up4 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(base_channels*2, base_channels)
        
        self.final_conv = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x6 = self.bottleneck(x5)
        
        x = self.up1(x6)
        x = torch.cat([x, x4], dim=1)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv4(x)
        

        return self.final_conv(x)