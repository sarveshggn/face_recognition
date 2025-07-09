import openvino as ov

model_path = '/home/sr/ov_fr/buffalo_l/det_10g.onnx'
ov_model = ov.convert_model(model_path, input=[1, 3, 360, 640])
# ov_model = ov.convert_model(model_path, input=[1, 3, 112, 112])
# ov_model = ov.convert_model(model_path, input=[1, 3, 640, 640])
ov.save_model(ov_model, '/home/sr/ov_fr/buffalo_l/FD/model.xml', compress_to_fp16=False)

# compiled_model = ov.compile_model(ov_model)

# prepare input_data
# import numpy as np
# input_data = np.random.rand(1, 3, 640, 640)

# run inference
# result = compiled_model(input_data)
