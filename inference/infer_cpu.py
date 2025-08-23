# inference/infer_cpu.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from PIL import Image
from model.unet import UNet

DEVICE = "cpu"
IMG_DIR = "data/images"
OUT_DIR = "outputs"
CKPT = "checkpoints/unet_drive_cpu.pth"
SIZE = (256, 256)

os.makedirs(OUT_DIR, exist_ok=True)

def load_img(path):
    img = Image.open(path).convert("L").resize(SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return img, ten

def load_gt(path):  # mask가 있으면 불러서 리사이즈
    if os.path.exists(path):
        m = Image.open(path).convert("L").resize(SIZE, Image.NEAREST)
        return m
    return None

def guess_mask_name(img_name):
    stem = os.path.splitext(img_name)[0]
    num  = stem.split("_")[0]
    cands = [f"{stem}.png", f"{stem.replace('_training','_manual1')}.png", f"{num}_manual1.png", f"{num}.png"]
    for c in cands:
        p = os.path.join("data/masks", c)
        if os.path.exists(p):
            return p
    return None

@torch.no_grad()
def main():
    model = UNet(1,1).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    imgs = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png")])[:10]
    for name in imgs:
        pil_in, ten = load_img(os.path.join(IMG_DIR, name))
        pred = model(ten.to(DEVICE)).cpu().squeeze(0).squeeze(0).numpy()
        pred_mask = (pred > 0.5).astype(np.uint8)*255
        pil_pred = Image.fromarray(pred_mask)

        # GT 있으면 같이
        gt_path = guess_mask_name(name)
        pil_gt = load_gt(gt_path) if gt_path else None

        # 3-컬럼 합치기 (입력/GT/Pred)
        cols = [pil_in.convert("RGB")]
        if pil_gt is not None:
            cols.append(pil_gt.convert("RGB"))
        cols.append(pil_pred.convert("RGB"))
        w, h = SIZE
        canvas = Image.new("RGB", (w*len(cols), h))
        for i, c in enumerate(cols):
            canvas.paste(c, (i*w, 0))
        canvas.save(os.path.join(OUT_DIR, f"viz_{name}"))

        # 단독 예측 마스크도 저장
        pil_pred.save(os.path.join(OUT_DIR, f"pred_{name}"))

    print(f"✅ saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
