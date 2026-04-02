"""检查不同检测器的 face crop 尺寸和视觉差异。"""
from PIL import Image
import os

for d in ["data/processed/exp_001_faces_224/train", "data/processed/exp_008_mtcnn_faces_112/train"]:
    if os.path.isdir(d):
        img = Image.open(os.path.join(d, "0.png"))
        print(f"{d}: {img.size}")
