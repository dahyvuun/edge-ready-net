# optimize/quantize_dynamic.py
from onnxruntime.quantization import quantize_dynamic, QuantType
inp="checkpoints/unet_drive.onnx"
out="checkpoints/unet_drive_int8_dyn.onnx"
quantize_dynamic(inp, out, weight_type=QuantType.QInt8)
print("✅ dynamic INT8 saved:", out)
