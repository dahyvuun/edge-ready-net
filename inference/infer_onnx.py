# inference/infer_onnx.py
import onnxruntime as ort, numpy as np
from PIL import Image
import os, time

SIZE=(256,256)
def load(path):
    im = Image.open(path).convert("L").resize(SIZE, Image.BILINEAR)
    arr = (np.array(im, dtype=np.float32)/255.0)[None,None,:,:]  # [1,1,H,W]
    return arr

sess = ort.InferenceSession("checkpoints/unet_drive.onnx",
                            providers=["CPUExecutionProvider"])
img = sorted([f for f in os.listdir("data/images") if f.endswith(".png")])[0]
x = load(os.path.join("data/images", img))
out = sess.run(["mask"], {"input": x})[0]
print("ONNX output shape:", out.shape)
