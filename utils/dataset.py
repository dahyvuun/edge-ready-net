# utils/dataset.py
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

TARGET_SIZE = (256, 256)  # 16의 배수 (256/512 중 택1)

class SimpleMedicalDataset(Dataset):
    def __init__(self, root):
        self.image_dir = os.path.join(root, "images")
        self.mask_dir  = os.path.join(root, "masks")
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(".png")])

    def _guess_mask_path(self, img_name: str):
        stem = os.path.splitext(img_name)[0]
        num  = stem.split("_")[0]
        candidates = [
            f"{stem}.png", f"{stem}.gif",
            f"{stem.replace('_training','_manual1')}.png",
            f"{stem.replace('_training','_manual1')}.gif",
            f"{num}_manual1.png", f"{num}_manual1.gif",
            f"{num}.png", f"{num}.gif",
        ]
        for c in candidates:
            p = os.path.join(self.mask_dir, c)
            if os.path.exists(p):
                return p
        return None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = self._guess_mask_path(img_name)
        if mask_path is None:
            raise FileNotFoundError(f"No mask for {img_name}")

        # 1) 로드 & 리사이즈 (다르게!)
        img  = Image.open(img_path).convert("L").resize(TARGET_SIZE, Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize(TARGET_SIZE, Image.NEAREST)

        # 2) 텐서화
        img  = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)   # [1,H,W]
        mask = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)   # [1,H,W], 0/1

        return img, mask
