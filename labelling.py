import numpy as np
import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "dataset")

for root, dirs, files in os.walk(image_dir):
    for f in files:
        if f.endswith("png") or f.endswith('jpg'):
            path = os.path.join(root,f)
            label = os.path.basename(root).replace(' ',"-").lower()
            image_data=Image.open(path).convert("L")
            image_data =cv2.resize(image_data, (64,64))
            image_array = np.array(image_data)