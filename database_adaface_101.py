import sqlite3
import os
import cv2
import numpy as np
from adaface_onnx import AdaFaceONNX
from adaface_openvino import AdaFaceOpenVINO
from scrfd import SCRFD
# from scrfd_openvino import SCRFDOpenVINO
import onnxruntime as ort
ort.set_default_logger_severity(4) 

# Initialize DB connection
conn = sqlite3.connect('photos-test.db')
c = conn.cursor()

# Load detector
assets_dir = os.path.expanduser('buffalo_l')
detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
# detector = SCRFDOpenVINO(os.path.join(assets_dir, 'det_10g.xml'))
detector.prepare(0)

# Load AdaFace model
# adaface_model_path = os.path.join(assets_dir, 'adaface_ir_101_model.onnx')
adaface_model_path = os.path.join(assets_dir, 'adaface_ir_50_model_fp16.xml')
# adaface_rec = AdaFaceONNX(adaface_model_path)
adaface_rec = AdaFaceOpenVINO(adaface_model_path)
adaface_rec.prepare(0)

# Embedding output folder
adaface_path = "Embeddings/Adaface_ov_fp16"
os.makedirs(adaface_path, exist_ok=True)  

def embed_image_adaface(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None
    bboxes, kpss = detector.autodetect(image, max_num=1)
    if bboxes.shape[0] == 0:
        return None
    kps = kpss[0]
    feat = adaface_rec.get(image, kps)
    return feat

def go_through_adaface():
    src_folder = "/home/sr/Photos"
    photo_dirs = sorted([d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d))])
    i = 0
    skip = 0
    remove = 0

    for photo_dir in photo_dirs:
        photo_folder = os.path.join(src_folder, photo_dir)
        images = sorted([f for f in os.listdir(photo_folder)])

        for image in images:
            parts = image.split('_')
            if len(parts) < 3:
                continue

            img_path = os.path.join(photo_folder, image)

            # Skip if already embedded
            c.execute('''SELECT adaface_embedding_ov_ffp16 FROM photos WHERE path = ?''', (img_path,))
            result = c.fetchone()
            if result is not None and result[0] is not None:
                skip += 1
                continue

            try:
                if cv2.imread(img_path) is None:
                    remove += 1
                    os.remove(img_path)
                    continue
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                remove += 1
                os.remove(img_path)
                continue

            feat = embed_image_adaface(img_path)
            if feat is None:
                with open('failed_adaface_images.txt', 'a') as f:
                    f.write(img_path + '\n')
                continue

            emb_part = image.split('.')[0]
            adaface_emb_path = os.path.join(adaface_path, f"{emb_part}.bin")

            # Ensure directory for embedding file exists
            os.makedirs(os.path.dirname(adaface_emb_path), exist_ok=True)

            with open(adaface_emb_path, 'wb') as f:
                f.write(feat.astype(np.float32).tobytes())

            # Update SQLite database
            c.execute('''UPDATE photos SET adaface_embedding_ov_ffp16 = ? WHERE path = ?''',
                      (adaface_emb_path, img_path))
            conn.commit()

            i += 1
            if i % 100 == 0:
                print(f"{i} embeddings generated.")

    print(f"\nSkipped {skip} existing embeddings.")
    print(f"Removed {remove} unreadable images.")
    print(f"Total processed images: {i}")
    c.execute('''SELECT COUNT(*) FROM photos''')
    count = c.fetchone()[0]
    print(f"Total images in database: {count}")

# Run it
if __name__ == "__main__":
    go_through_adaface()
