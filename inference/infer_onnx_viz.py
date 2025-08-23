# inference/infer_onnx_viz.py
import os, numpy as np
from PIL import Image
import onnxruntime as ort

SIZE=(256,256)
IMG_DIR="data/images"
MASK_DIR="data/masks"
OUT_DIR="outputs_onnx"
os.makedirs(OUT_DIR, exist_ok=True)

def guess_mask(img_name):
    stem=img_name.split(".")[0]; num=stem.split("_")[0]
    for c in [f"{stem}.png", f"{stem.replace('_training','_manual1')}.png",
              f"{num}_manual1.png", f"{num}.png"]:
        p=os.path.join(MASK_DIR,c)
        if os.path.exists(p): return p
    return None

def to_tensor(img_path):
    im = Image.open(img_path).convert("L").resize(SIZE, Image.BILINEAR)
    arr = (np.array(im,dtype=np.float32)/255.0)[None,None,:,:]
    return im, arr

sess = ort.InferenceSession("checkpoints/unet_drive.onnx",
                            providers=["CPUExecutionProvider"])

files=[f for f in sorted(os.listdir(IMG_DIR)) if f.lower().endswith(".png")][:10]
for name in files:
    pil_in, x = to_tensor(os.path.join(IMG_DIR,name))
    y = sess.run(["mask"], {"input": x})[0][0,0]          # (H,W)
    pred = (y>0.5).astype(np.uint8)*255
    pil_pred = Image.fromarray(pred)

    gt_path = guess_mask(name)
    cols=[pil_in.convert("RGB")]
    if gt_path:
        pil_gt = Image.open(gt_path).convert("L").resize(SIZE, Image.NEAREST)
        cols.append(pil_gt.convert("RGB"))
    cols.append(pil_pred.convert("RGB"))

    w,h=SIZE; canvas=Image.new("RGB",(w*len(cols),h))
    for i,c in enumerate(cols): canvas.paste(c,(i*w,0))
    canvas.save(os.path.join(OUT_DIR, f"viz_{name}"))
    pil_pred.save(os.path.join(OUT_DIR, f"pred_{name}"))
print(f"✅ saved to {OUT_DIR}")
