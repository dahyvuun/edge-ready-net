# optimize/export_onnx_light.py
import sys, os, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.unet_light import UNetLight

CKPT = "checkpoints/unet_light_cpu.pth"
OUT  = "checkpoints/unet_light.onnx"

def main():
    model = UNetLight(in_ch=1, out_ch=1, base=32, depthwise=True)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    x = torch.randn(1,1,256,256)
    torch.onnx.export(
        model, x, OUT, opset_version=13, do_constant_folding=True,
        input_names=["input"], output_names=["mask"],
        dynamic_axes={"input":{0:"batch"}, "mask":{0:"batch"}}
    )
    print(f"✅ exported: {OUT}")

if __name__ == "__main__":
    main()
