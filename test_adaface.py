# import os
# import cv2
# import numpy as np
# from adaface_onnx import AdaFaceONNX
# from adaface_openvino import AdaFaceOpenVINO
# from scrfd import SCRFD
# import onnxruntime as ort

# ort.set_default_logger_severity(4)  # suppress verbose onnxruntime logs

# # Load detector
# assets_dir = os.path.expanduser('buffalo_l')
# detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
# detector.prepare(0)

# # Load AdaFace model
# # adaface_model_path = os.path.join(assets_dir, 'adaface_ir_101_model.onnx')
# # adaface_rec = AdaFaceONNX(adaface_model_path)
# # adaface_rec.prepare(0)

# adaface_model_path = os.path.join(assets_dir, 'adaface_ir_50_model.xml')
# adaface_rec = AdaFaceOpenVINO(adaface_model_path)
# adaface_rec.prepare(0)

# def get_embedding(img_path):
#     image = cv2.imread(img_path)
#     if image is None:
#         print("Could not read image.")
#         return None

#     bboxes, kpss = detector.autodetect(image, max_num=1)
#     if bboxes.shape[0] == 0:
#         print("No face detected.")
#         return None

#     kps = kpss[0]
#     feat = adaface_rec.get(image, kps)
#     return feat.tolist()  # convert numpy array to list of floats

# if __name__ == "__main__":
#     test_image_path = "/home/sr/ov_fr/videos/input/VID_39837_001.jpg"  # <-- replace this with your test image path

#     embedding = get_embedding(test_image_path)
#     if embedding is not None:
#         print(f"Embedding (len={len(embedding)}):")
#         print(embedding)
#     else:
#         print("Failed to generate embedding.")

import os
import cv2
import json
import numpy as np
from adaface_onnx import AdaFaceONNX
from adaface_openvino import AdaFaceOpenVINO
from scrfd import SCRFD
import onnxruntime as ort

ort.set_default_logger_severity(4)  # suppress verbose onnxruntime logs

# ---------------------------------------------------------------------
# Config –‑‑ set these two paths before running
# ---------------------------------------------------------------------
assets_dir          = os.path.expanduser('buffalo_l')     # model folder
embeddings_jsonl    = "/home/sr/ov_fr/final_embeddings_adaface_ov_fp16.jsonl"         # existing JSONL
test_image_path     = "/home/sr/ov_fr/videos/input/VID_39837_001.jpg"
# ---------------------------------------------------------------------

# Face detector
detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
detector.prepare(0)

# AdaFace model (OpenVINO here; comment out if you prefer the ONNX path)
adaface_model_path = os.path.join(assets_dir, 'adaface_ir_50_model_fp16.xml')
adaface_rec = AdaFaceOpenVINO(adaface_model_path)
adaface_rec.prepare(0)


def get_embedding(img_path: str):
    """Return 512‑D AdaFace embedding as a Python list or None if no face."""
    image = cv2.imread(img_path)
    if image is None:
        print(f"[!] Could not read image: {img_path}")
        return None

    bboxes, kpss = detector.autodetect(image, max_num=1)
    if bboxes.shape[0] == 0:
        print(f"[!] No face detected in {img_path}")
        return None

    feat = adaface_rec.get(image, kpss[0])  # (512,)
    return feat.astype(float).tolist()       # ensure JSON‑serialisable


def update_jsonl(jsonl_path: str, entry: dict, prepend: bool = True):
    """
    Add `entry` to a JSONL file.
    • If prepend=True (default) – write entry at the *top* of the file.
    • If prepend=False            – append entry at the *end* of the file.
    The file is created automatically if it does not exist.
    """
    line = json.dumps(entry, ensure_ascii=False)
    if not os.path.exists(jsonl_path):
        # Fresh file – just write the first line
        with open(jsonl_path, "w", encoding="utf‑8") as f:
            f.write(line + "\n")
        return

    if prepend:
        # Read all, then rewrite with new line first
        with open(jsonl_path, "r+", encoding="utf‑8") as f:
            existing = f.readlines()
            f.seek(0)
            f.write(line + "\n")
            f.writelines(existing)
    else:  # append
        with open(jsonl_path, "a", encoding="utf‑8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    feat_list = get_embedding(test_image_path)
    if feat_list is None:
        exit(1)

    entry = {"path": test_image_path, "feat": feat_list}
    update_jsonl(embeddings_jsonl, entry, prepend=True)
    print(f"[✓] Added embedding for {test_image_path} → {embeddings_jsonl}")
