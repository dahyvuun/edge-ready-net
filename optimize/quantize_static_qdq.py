# E:\project\edge-ready-unet\optimize\quantize_static_qdq.py
import glob, os, numpy as np, onnx
from PIL import Image
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader,
    QuantType, QuantFormat,
)

MODEL_FP32 = r"checkpoints\unet_drive.onnx"
MODEL_INT8 = r"checkpoints\unet_drive.int8.qdq.onnx"
IMAGES_DIR = r"data\images"   # use a handful (10–50) of typical inputs

class UNetCalibReader(CalibrationDataReader):
    def __init__(self, model_path, images_glob):
        self.enum_data_dicts = None
        self.input_name = self._get_input_name(model_path)
        self.images = sorted(glob.glob(images_glob))[:50]

    def _get_input_name(self, model_path):
        m = onnx.load(model_path)
        return m.graph.input[0].name

    def get_next(self):
        if self.enum_data_dicts is None:
            # lazy-load to a generator of dicts
            def gen():
                for p in self.images:
                    # adapt preprocessing to your export (NCHW, 1x256x256, float32 0..1)
                    arr = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
                    arr = arr[None, None, :, :]  # (1,1,H,W)
                    yield {self.input_name: arr}
            self.enum_data_dicts = gen()
        return next(self.enum_data_dicts, None)

calib = UNetCalibReader(MODEL_FP32, os.path.join(IMAGES_DIR, "*.png"))

quantize_static(
    model_input=MODEL_FP32,
    model_output=MODEL_INT8,
    calibration_data_reader=calib,
    weight_type=QuantType.QInt8,         # per-tensor by default
    activation_type=QuantType.QInt8,     # activations int8
    quant_format=QuantFormat.QDQ,        # <<< avoids ConvInteger
    per_channel=True,                    # better for conv weights
    reduce_range=False
)

print("Saved:", MODEL_INT8)
