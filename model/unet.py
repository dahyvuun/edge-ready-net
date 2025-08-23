# model/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

def center_crop(enc_feat, target_tensor):
    _, _, h, w = enc_feat.shape
    _, _, th, tw = target_tensor.shape
    dh = (h - th) // 2
    dw = (w - tw) // 2
    # 만약 enc_feat가 더 작으면 패딩 (예외 방어)
    if dh < 0 or dw < 0:
        pad = (max(0, -dw), max(0, -dh), max(0, -dw), max(0, -dh))
        enc_feat = F.pad(enc_feat, pad)
        _, _, h, w = enc_feat.shape
        dh = (h - th) // 2
        dw = (w - tw) // 2
    return enc_feat[:, :, dh:dh+th, dw:dw+tw]

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        enc4c = center_crop(enc4, dec4)
        dec4 = self.dec4(torch.cat([dec4, enc4c], dim=1))

        dec3 = self.upconv3(dec4)
        enc3c = center_crop(enc3, dec3)
        dec3 = self.dec3(torch.cat([dec3, enc3c], dim=1))

        dec2 = self.upconv2(dec3)
        enc2c = center_crop(enc2, dec2)
        dec2 = self.dec2(torch.cat([dec2, enc2c], dim=1))

        dec1 = self.upconv1(dec2)
        enc1c = center_crop(enc1, dec1)
        dec1 = self.dec1(torch.cat([dec1, enc1c], dim=1))

        return torch.sigmoid(self.final(dec1))
