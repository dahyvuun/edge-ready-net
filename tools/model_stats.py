# tools/model_stats.py
import os, sys, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.unet import UNet
from model.unet_light import UNetLight

def count_params(model): return sum(p.numel() for p in model.parameters())

def size_mb(path): return os.path.getsize(path)/1024/1024

if __name__=="__main__":
    b=UNet(1,1)
    l=UNetLight(1,1,base=32,depthwise=True)
    print("UNet params:", count_params(b))
    print("UNet-Light params:", count_params(l))
    for f in ["checkpoints/unet_drive.onnx","checkpoints/unet_light.onnx"]:
        if os.path.exists(f): print(f, "≈", f"{size_mb(f):.2f} MB")
