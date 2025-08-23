# inference/bench_compare.py
import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from PIL import Image
import torch
import onnxruntime as ort
from model.unet import UNet

IMG_DIR = "data/images"
CKPT_PT = "checkpoints/unet_drive_cpu.pth"
CKPT_ONNX = "checkpoints/unet_drive.onnx"
SIZE = (256, 256)

def list_pngs(limit=20):
    files = [f for f in sorted(os.listdir(IMG_DIR)) if f.lower().endswith(".png")]
    files = files[:limit]
    print(f"📁 Using {len(files)} PNG images")
    return files

def load_np(path):
    im = Image.open(path).convert("L").resize(SIZE, Image.BILINEAR)
    a = (np.array(im, dtype=np.float32) / 255.0)[None, None, :, :]  # [1,1,H,W]
    return a

def bench_pytorch(files):
    print("🧪 PyTorch (per-image)")
    model = UNet(1,1).eval()
    model.load_state_dict(torch.load(CKPT_PT, map_location="cpu"))

    # 워밍업
    _ = model(torch.from_numpy(load_np(os.path.join(IMG_DIR, files[0]))))

    iters = 5
    t0 = time.perf_counter()
    for _ in range(iters):
        for f in files:
            x = torch.from_numpy(load_np(os.path.join(IMG_DIR, f)))
            _ = model(x)
    dt = time.perf_counter() - t0
    per_img_ms = (dt / (iters * len(files))) * 1000
    return per_img_ms

def make_low_mem_session(model_path):
    so = ort.SessionOptions()
    # 메모리 절약
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # 스레드 보수적
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

def bench_onnx(files):
    print("🧪 ONNX (per-image, low-mem)")
    sess = make_low_mem_session(CKPT_ONNX)

    # 워밍업
    _ = sess.run(["mask"], {"input": load_np(os.path.join(IMG_DIR, files[0]))})

    iters = 5
    t0 = time.perf_counter()
    for _ in range(iters):
        for f in files:
            x = load_np(os.path.join(IMG_DIR, f))
            _ = sess.run(["mask"], {"input": x})
    dt = time.perf_counter() - t0
    per_img_ms = (dt / (iters * len(files))) * 1000
    return per_img_ms

def main():
    files = list_pngs(limit=20)
    if not files:
        print("🚫 No images found under data/images")
        return
    pt = bench_pytorch(files)
    ox = bench_onnx(files)

    print("\n===== RESULT =====")
    print(f"PyTorch: ~{pt:.2f} ms/img")
    print(f"ONNX:    ~{ox:.2f} ms/img")

if __name__ == "__main__":
    main()
