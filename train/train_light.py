# train/train_light.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from model.unet_light import UNetLight
from utils.dataset import SimpleMedicalDataset
from utils.loss import dice_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = SimpleMedicalDataset(root="data")
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)

    model = UNetLight(in_ch=1, out_ch=1, base=32, depthwise=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    epochs = 10  # 가볍게 시작해도 됨
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            out = model(images)
            loss = criterion(out, masks) + dice_loss(out, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        print(f"[Light] Epoch {epoch+1}/{epochs} | Loss: {total:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/unet_light_cpu.pth")
    print("✅ Saved: checkpoints/unet_light_cpu.pth")

if __name__ == "__main__":
    main()
