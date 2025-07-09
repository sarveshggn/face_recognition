import sys
import openvino as ov
try:
    import nncf
except ImportError:
    print("Error: NNCF module not found. Please install openvino-dev:")
    print("Run: pip install openvino-dev==2024.6.0")
    sys.exit(1)
import numpy as np
import cv2
import os
from typing import List

# Custom Dataset for NNCF
class CalibrationDataset:
    def __init__(self, dataset_path: str, input_shape=(360, 640)):
        self.dataset_path = dataset_path
        self.input_shape = input_shape
        self.image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]
        if not self.image_files:
            raise ValueError(f"No images found in {dataset_path}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> List[np.ndarray]:
        img_path = os.path.join(self.dataset_path, self.image_files[index])
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.resize(img, self.input_shape[::-1])  # Resize to 640x360
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        img = img.transpose(2, 0, 1)  # To CHW format
        return [img[np.newaxis, ...]]  # Add batch dimension

# Paths
model_path = '/home/sr/ov_fr/buffalo_l/det_10g.onnx'
calibration_dataset_path = '/home/sr/calib-data-fd'
output_dir = '/home/sr/ov_fr/buffalo_l/FD'

# Verify paths
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)
if not os.path.exists(calibration_dataset_path):
    print(f"Error: Calibration dataset directory not found at {calibration_dataset_path}")
    sys.exit(1)
os.makedirs(output_dir, exist_ok=True)

# Convert ONNX to OpenVINO FP32 model
try:
    ov_model = ov.convert_model(model_path, input=[1, 3, 360, 640])
    ov.save_model(ov_model, f'{output_dir}/model_fp32.xml')
    print("FP32 model saved successfully")
except Exception as e:
    print(f"Error converting ONNX to FP32: {e}")
    sys.exit(1)

# Configure NNCF quantization
try:
    calibration_dataset = CalibrationDataset(calibration_dataset_path)
    quantized_model = nncf.quantize(
        model=ov_model,
        calibration_dataset=nncf.Dataset(calibration_dataset),
        preset=nncf.QuantizationPreset.PERFORMANCE,  # Use MIXED for higher accuracy
        subset_size=2000  # Use all 2000 images
    )
    ov.save_model(quantized_model, f'{output_dir}/model_int8.xml')
    print("INT8 model saved successfully")
except Exception as e:
    print(f"Error during quantization: {e}")
    sys.exit(1)
