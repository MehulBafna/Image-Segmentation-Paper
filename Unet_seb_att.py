import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = torch.cat([avg_y, max_y], dim=1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, g, x):
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * (psi * self.scale)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(DoubleConv, self).__init__()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob) if dropout_prob > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob/2) if dropout_prob > 0 else nn.Identity()
        )
        
        self.se = SEBlock(out_channels)

    def forward(self, x):
        identity = self.residual(x)
        out = self.conv(x)
        out = self.se(out)
        return F.relu(out + identity)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=5):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 32, dropout_prob=0.0)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64, dropout_prob=0.1))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128, dropout_prob=0.1))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256, dropout_prob=0.2))
        self.bottleneck = DoubleConv(256, 512, dropout_prob=0.2)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        self.conv1 = DoubleConv(512, 256, dropout_prob=0.2)
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.conv2 = DoubleConv(256, 128, dropout_prob=0.1)
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.conv3 = DoubleConv(128, 64, dropout_prob=0.1)
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = DoubleConv(64, 32, dropout_prob=0.0)

        self.att1 = AttentionBlock(256, 256, 128)
        self.att2 = AttentionBlock(128, 128, 64)
        self.att3 = AttentionBlock(64, 64, 32)
        self.att4 = AttentionBlock(32, 32, 16)

        self.outc = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)


        x = self.up1(x5)
        if x.size()[2:] != x4.size()[2:]:
            x = F.interpolate(x, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x4_att = self.att1(x, x4)
        x = torch.cat([x, x4_att], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        if x.size()[2:] != x3.size()[2:]:
            x = F.interpolate(x, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x3_att = self.att2(x, x3)
        x = torch.cat([x, x3_att], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        if x.size()[2:] != x2.size()[2:]:
            x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x2_att = self.att3(x, x2)
        x = torch.cat([x, x2_att], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        if x.size()[2:] != x1.size()[2:]:
            x = F.interpolate(x, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x1_att = self.att4(x, x1)
        x = torch.cat([x, x1_att], dim=1)
        x = self.conv4(x)

        return self.outc(x)