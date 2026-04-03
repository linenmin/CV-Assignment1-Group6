"""
export_images.py
----------------
将训练数据按类别导出为 PNG 文件，方便查看。

用法
----
cd Part1
python export_images.py

输出目录结构
-----------
exported_images/
  class_0/
    train_0.png
    train_3.png
    ...
  class_1/
    train_1.png
    ...
  class_2/
    ...
"""

import os
import cv2
import numpy as np
from data_loader import load_data


OUTPUT_DIR = 'exported_images'


def export_images(output_dir: str = OUTPUT_DIR) -> None:
    train, _ = load_data()

    classes = sorted(train['class'].unique())
    for cls in classes:
        cls_dir = os.path.join(output_dir, f'class_{cls}')
        os.makedirs(cls_dir, exist_ok=True)

    count = 0
    for idx, row in train.iterrows():
        cls = row['class']
        img_rgb = row['img']  # RGB ndarray
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(output_dir, f'class_{cls}', f'train_{idx}.png')
        cv2.imwrite(out_path, img_bgr)
        count += 1

    print(f"已导出 {count} 张图片到 '{output_dir}/'")
    for cls in classes:
        n = len(train[train['class'] == cls])
        print(f"  class_{cls}: {n} 张")


if __name__ == '__main__':
    export_images()
