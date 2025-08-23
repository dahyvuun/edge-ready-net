# inference/bench_onnx_variant.py
import sys, time, glob
import numpy as np
import onnx
import onnxruntime as ort

def has_convinteger(path: str) -> bool:
    try:
        m = onnx.load(path)
        return any(n.op_type == "ConvInteger" for n in m.graph.node)
    except Exception:
        return False

def _np_dtype(ort_type: str):
    # common mappings; default to float32
    return {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
        "tensor(bool)": np.bool_,
    }.get(ort_type, np.float32)

def _concrete_shape(shape):
    # Replace symbolic/None dims with typical values
    out = []
    for i, d in enumerate(shape):
        if isinstance(d, int) and d > 0:
            out.append(d)
        else:
            # batch -> 1, channels -> 1, spatial -> 256
            out.append(1 if i < 2 else 256)
    # ensure at least 4D (NCHW). If model is 3D (N,H,W), pad channel=1
    if len(out) == 3:
        out.insert(1, 1)
    return tuple(out)

def bench(model_path: str, iters: int = 30, warmup: int = 5) -> float:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    imeta = sess.get_inputs()[0]
    inp_name = imeta.name
    shape = _concrete_shape(imeta.shape)
    dtype = _np_dtype(imeta.type)

    # Deterministic dummy data
    if np.issubdtype(dtype, np.floating):
        x = np.ones(shape, dtype=dtype)
    elif np.issubdtype(dtype, np.integer):
        x = np.zeros(shape, dtype=dtype)
    else:
        x = np.ones(shape, dtype=np.float32)

    # Warmup
    for _ in range(warmup):
        _ = sess.run(None, {inp_name: x})

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = sess.run(None, {inp_name: x})
    t1 = time.perf_counter()

    ms_per_img = (t1 - t0) * 1000.0 / iters
    return ms_per_img

def main():
    models = sys.argv[1:]
    if not models:
        models = [
            r"checkpoints/unet_drive.onnx",
            *glob.glob(r"checkpoints/*.onnx"),
        ]

    for m in models:
        if has_convinteger(m):
            print(f"{m} SKIPPED: contains ConvInteger (unsupported by CPU EP)")
            continue
        try:
            t = bench(m)
            print(f"{m} ≈ {t:.2f} ms/img")
        except Exception as e:
            print(f"{m} SKIPPED: {e}")

if __name__ == "__main__":
    main()
