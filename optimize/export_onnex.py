# optimize/export_onnx.py
import sys, os, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.unet import UNet

CKPT = "checkpoints/unet_drive_cpu.pth"
OUT = "checkpoints/unet_drive.onnx"

def main():
    model = UNet(1,1)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    x = torch.randn(1,1,256,256)
    torch.onnx.export(
        model, x, OUT, input_names=["input"], output_names=["mask"],
        opset_version=13, do_constant_folding=True,
        dynamic_axes={"input":{0:"batch"}, "mask":{0:"batch"}}
    )
    print(f"✅ exported: {OUT}")

if __name__ == "__main__":
    main()
