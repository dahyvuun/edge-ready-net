import torch, torch.nn as nn, torch.nn.functional as F

class DWConv(nn.Module):
    """Depthwise separable conv (선택적으로 사용)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.bn(x)
        return F.relu_(x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, depthwise=False):
        super().__init__()
        block = DWConv if depthwise else nn.Conv2d
        self.c1 = (DWConv(in_ch, out_ch) if depthwise
                   else nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False),
                                      nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)))
        self.c2 = (DWConv(out_ch, out_ch) if depthwise
                   else nn.Sequential(nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False),
                                      nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)))
    def forward(self, x): return self.c2(self.c1(x))

def center_crop(a, b):
    _,_,h,w = a.shape; _,_,th,tw = b.shape
    dh, dw = (h-th)//2, (w-tw)//2
    if dh<0 or dw<0:
        a = F.pad(a, (max(0,-dw),max(0,-dh),max(0,-dw),max(0,-dh)))
        _,_,h,w = a.shape; dh, dw = (h-th)//2, (w-tw)//2
    return a[:, :, dh:dh+th, dw:dw+tw]

class UNetLight(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32, depthwise=False):
        super().__init__()
        ch1, ch2, ch3, ch4, bott = base, base*2, base*4, base*8, base*16
        self.enc1 = DoubleConv(in_ch, ch1, depthwise)
        self.enc2 = DoubleConv(ch1, ch2, depthwise)
        self.enc3 = DoubleConv(ch2, ch3, depthwise)
        self.enc4 = DoubleConv(ch3, ch4, depthwise)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(ch4, bott, depthwise)
        self.up4 = nn.ConvTranspose2d(bott, ch4, 2, 2)
        self.dec4 = DoubleConv(ch4+ch4, ch4, depthwise)
        self.up3 = nn.ConvTranspose2d(ch4, ch3, 2, 2)
        self.dec3 = DoubleConv(ch3+ch3, ch3, depthwise)
        self.up2 = nn.ConvTranspose2d(ch3, ch2, 2, 2)
        self.dec2 = DoubleConv(ch2+ch2, ch2, depthwise)
        self.up1 = nn.ConvTranspose2d(ch2, ch1, 2, 2)
        self.dec1 = DoubleConv(ch1+ch1, ch1, depthwise)
        self.final = nn.Conv2d(ch1, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.up4(b); d4 = self.dec4(torch.cat([d4, center_crop(e4, d4)], 1))
        d3 = self.up3(d4); d3 = self.dec3(torch.cat([d3, center_crop(e3, d3)], 1))
        d2 = self.up2(d3); d2 = self.dec2(torch.cat([d2, center_crop(e2, d2)], 1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1, center_crop(e1, d1)], 1))
        return torch.sigmoid(self.final(d1))
