# optimize/prune.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, torch.nn.utils.prune as prune
from model.unet import UNet

IN_CKPT = "checkpoints/unet_drive_cpu.pth"
OUT_CKPT = "checkpoints/unet_pruned.pth"

def apply_global_pruning(model, amount=0.4):
    params = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            params.append((m, "weight"))
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    # 프루닝 마스크 제거하고 가중치 고정
    for m, _ in params:
        prune.remove(m, "weight")

def main():
    model = UNet(1,1)
    model.load_state_dict(torch.load(IN_CKPT, map_location="cpu"))
    apply_global_pruning(model, amount=0.4)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), OUT_CKPT)
    print(f"✅ saved pruned model: {OUT_CKPT}")

if __name__ == "__main__":
    main()
