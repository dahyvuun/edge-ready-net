# convert_existing_tif_gif.py
import os
from PIL import Image

def convert_images_to_png(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".tif") or filename.endswith(".gif"):
            filepath = os.path.join(folder, filename)
            img = Image.open(filepath).convert("L")
            newname = os.path.splitext(filename)[0] + ".png"
            save_path = os.path.join(folder, newname)
            img.save(save_path)
            print(f"✅ {filename} → {newname} 저장됨")

# 실행E:/Edge-Ready-UNet/data/images
convert_images_to_png("E:/project/edge-ready-unet/data/images")
convert_images_to_png("E:/project/edge-ready-unet/data/mask")
