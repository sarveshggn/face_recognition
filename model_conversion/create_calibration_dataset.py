import os
import cv2
import numpy as np
from nncf import Dataset
from typing import List

def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess a single image for AdaFace (112x112, normalized)."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Resize to 112x112 (AdaFace input size)
    img = cv2.resize(img, (112, 112))
    
    # Convert to float32 and normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Apply mean and std normalization (adjust as per AdaFace requirements)
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    img = (img - mean) / std
    
    # Convert to model input format (1, C, H, W)
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Ensure output is float32
    img = img.astype(np.float32)
    return img

def load_images(image_dir: str, max_images: int = 2000) -> List[np.ndarray]:
    """Load and preprocess images from directory."""
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    if not image_paths:
        raise ValueError(f"No images found in directory: {image_dir}")
    image_paths_first = image_paths[:max_images-1000]  # First 1000 images
    images = []
    for path in image_paths_first:
        try:
            images.append(preprocess_image(path))
        except Exception as e:
            print(f"Skipping image {path}: {e}")

    image_paths_last = image_paths[max_images+1000:]  # Last 1000 images
    for path in image_paths_last:
        try:
            images.append(preprocess_image(path))
        except Exception as e:
            print(f"Skipping image {path}: {e}")
    return images


def create_calibration_dataset(image_dir: str, max_images: int = 2000) -> Dataset:
    """Create NNCF calibration dataset."""
    images = load_images(image_dir, max_images)
    if not images:
        raise ValueError("No valid images loaded for calibration dataset")
    def transform_fn(image):
        return {"input": image}  # Use 'input' to match the model's input name
    return Dataset(images, transform_fn)

# Example usage
if __name__ == "__main__":
    image_dir = "/home/sr/calib-data"  # Path to your folder with 4000 images
    images = load_images(image_dir, max_images=2000)
    calibration_dataset = create_calibration_dataset(image_dir, max_images=2000)
    print(f"Calibration dataset created with {len(images)} samples")
    # Debug input keys and data type
    sample_data = next(iter(calibration_dataset.get_inference_data()))
    print(f"Calibration dataset input keys: {list(sample_data.keys())}")
    print(f"Sample data type: {sample_data['input'].dtype}")