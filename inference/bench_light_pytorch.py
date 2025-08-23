# inference/bench_light_pytorch.py
import os, sys, time, numpy as np
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from model.unet_light import UNetLight

SIZE=(256,256); IMG_DIR="data/images"
files=[f for f in sorted(os.listdir(IMG_DIR)) if f.lower().endswith(".png")][:20]

def load1(p):
    a=(np.array(Image.open(p).convert("L").resize(SIZE))/255.).astype(np.float32)
    return torch.from_numpy(a)[None,None]

@torch.no_grad()
def main():
    m=UNetLight(1,1,base=32,depthwise=True).eval()
    m.load_state_dict(torch.load("checkpoints/unet_light_cpu.pth", map_location="cpu"))
    # warmup
    _=m(load1(os.path.join(IMG_DIR, files[0])))
    iters=5; t0=time.perf_counter()
    for _ in range(iters):
        for f in files: _=m(load1(os.path.join(IMG_DIR,f)))
    dt=time.perf_counter()-t0
    print(f"UNet-Light PyTorch ≈ {(dt/(iters*len(files))*1000):.2f} ms/img")
if __name__=="__main__": main()
