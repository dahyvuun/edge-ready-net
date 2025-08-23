# 🩺 Edge-Ready U-Net (CPU-Only Medical Image Segmentation)

> Lightweight, CPU-deployable U-Net pipeline for medical image segmentation  
> ✔️ Full PyTorch training → ✔️ ONNX export → ✔️ CPU inference & benchmark

---

## 📌 Overview

This project implements a **U-Net-based segmentation model** optimized for **CPU-only environments** (edge devices, low-spec machines).  
It uses the [DRIVE dataset](http://drive.grand-challenge.org/) (retinal vessel segmentation), but the code is easily adaptable to any binary medical segmentation task.

✅ Features:
- Custom U-Net with center-crop skip-connections
- BCE + Dice loss
- CPU-only training & inference
- ONNX export + ONNXRuntime inference
- PyTorch vs ONNX **CPU latency benchmarks**
- Lightweight UNet-Light (depthwise, fewer params)

---


---

## ⚙️ Installation

```bash
conda create -n unet-env python=3.10 -y
conda activate unet-env
conda install pytorch torchvision cpuonly -c pytorch -y
pip install -r requirements.txt


requirements.txt

torch==2.3.1
torchvision==0.18.1
numpy==1.26.4
pillow==10.4.0
onnx==1.16.1
onnxruntime==1.18.0

##Data

Download: http://drive.grand-challenge.org/

Convert .tif/.gif → .png

## 🏋️ Training

The model trains on DRIVE images (256×256).  
Loss decreases steadily over epochs, confirming convergence:

Epoch 1/20 | Loss: 7.16
Epoch 10/20 | Loss: 5.37
Epoch 20/20 | Loss: 4.82


✔️ Model weights are saved automatically to:
- `checkpoints/unet_drive_cpu.pth` (baseline U-Net)
- `checkpoints/unet_light_cpu.pth` (lightweight U-Net)


My CPU Results (256x256, per image):
| Variant    | Backend |    ms/img |
| ---------- | ------- | --------: |
| UNet       | PyTorch |   \~750.6 |
| UNet       | ONNX    | \~514–611 |
| UNet-Light | PyTorch |   \~127.0 |
| UNet-Light | ONNX    |   \~32.78 |

Model Size & Params
| Model      |     Params | ONNX Size |
| ---------- | ---------: | --------: |
| UNet       | 31,042,369 | 118.39 MB |
| UNet-Light |  1,513,354 |   5.79 MB |


## Optional Optimizations
Pruning

python optimize/prune.py

Quantization(dynamic INT8)

python optimize/quantize_dynamic.py

OpenVINO Execution Provider (Intel CPU)

pip install onnxruntime-openvino

## 🔮 Future Work

To further optimize for edge devices, the following directions are possible:

- **Pruning**: Remove redundant connections/filters to reduce computation while maintaining accuracy.  
- **Quantization (INT8)**: Convert model weights to 8-bit precision for faster inference and smaller footprint.  
- **OpenVINO Execution Provider**: Leverage Intel CPU acceleration backend for additional runtime speedup.  
- **Mobile Deployment**: Package the ONNX model for mobile (Android/iOS) apps or web inference.  

*(These steps are planned but not yet included in this repository. The current pipeline already supports full training, inference, ONNX export, and CPU benchmarking.)*


##Citation
U-Net: Ronneberger et al., 2015
DRIVE: Digital Retinal Images for Vessel Extraction
