import onnx
import nncf
import openvino as ov
from onnx import helper
from model_conversion.create_calibration_dataset import create_calibration_dataset

# Load the ONNX model
onnx_model_path = "/home/sr/ov_fr/buffalo_l/adaface_ir_50_model.onnx"  # Replace with your model path
try:
    onnx_model = onnx.load(onnx_model_path)
except Exception as e:
    raise ValueError(f"Failed to load ONNX model: {e}")

# Verify model input name
model_inputs = [input.name for input in onnx_model.graph.input]
print(f"Model input names: {model_inputs}")  # Should include 'input'

# Create calibration dataset
image_dir = "/home/sr/calib-data"  # Path to your folder with 4000 images
try:
    calibration_dataset = create_calibration_dataset(image_dir, max_images=2000)
except Exception as e:
    raise ValueError(f"Failed to create calibration dataset: {e}")

# Verify calibration dataset input and data type
sample_data = next(iter(calibration_dataset.get_inference_data()))
print(f"Calibration dataset input keys: {list(sample_data.keys())}")  # Should include 'input'
print(f"Calibration dataset input data type: {sample_data['input'].dtype}")  # Should be float32

# Quantize the model to INT-8
try:
    quantized_model = nncf.quantize(
        onnx_model,
        calibration_dataset,
        fast_bias_correction=False,  # To improve accuracy use False
        subset_size=2000,  # Use 2000 images for calibration
        target_device=nncf.TargetDevice.CPU 
    )
except Exception as e:
    raise ValueError(f"Quantization failed: {e}")

# Save the quantized ONNX model
quantized_model_path = "quantized_adaface_r50_bias_coorected.onnx"
onnx.save(quantized_model, quantized_model_path)

# Convert to OpenVINO IR
try:
    ov_model = ov.convert_model(quantized_model_path)
except Exception as e:
    raise ValueError(f"Conversion to OpenVINO IR failed: {e}")

# Save the OpenVINO IR model
ov.save_model(ov_model, "adaface_r50_int8_bias_corrected.xml")
print("Quantization and conversion completed successfully!")
