# train/train.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.unet import UNet
from utils.dataset import SimpleMedicalDataset   # 리사이즈/텐서화 내장
from utils.loss import dice_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 데이터셋/로더 (transform 필요 X)
    train_set = SimpleMedicalDataset(root="data")
    train_loader = DataLoader(
        train_set, batch_size=4, shuffle=True,
        num_workers=0,        # Windows 권장
        pin_memory=False      # CPU-only면 False
    )

    # 2) 모델/옵티/로스
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()  # 모델이 sigmoid 출력이므로 BCELoss 사용

    # 3) 학습 루프
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)   # [B,1,256,256]
            masks  = masks.to(device)    # [B,1,256,256], {0,1}

            outputs = model(images)      # sigmoid된 [B,1,256,256]
            loss = criterion(outputs, masks) + dice_loss(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    # (선택) 가중치 저장
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/unet_drive_cpu.pth")
    print("✅ Saved: checkpoints/unet_drive_cpu.pth")

if __name__ == "__main__":
    main()
