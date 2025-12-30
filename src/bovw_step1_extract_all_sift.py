import os
import numpy as np
import cv2
import joblib
from config import *

# Configuration
# We just need the paths from any config

MAX_PER_IMAGE = 500 # Limit to 500 features per image
OUT_FILE = os.path.join(DESCRIPTOR_FOLDER, 'bovw_all_sift_descriptors_{MAX_PER_IMAGE}.pkl')

print(f"Step 1: Extracting all SIFT descriptors")
print(f"Reading from: {IMAGE_FOLDER}")
# 1. Initialize SIFT detector (from demokeypoint.py)
sift = cv2.SIFT_create(nfeatures=MAX_PER_IMAGE)
all_descriptors_list = []
image_count = 0

# 2. Loop through all images in IMAGE_FOLDER
for filename in os.listdir(IMAGE_FOLDER):
    if not filename.endswith(".bmp"):
        continue

    img_path = os.path.join(IMAGE_FOLDER, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # SIFT needs grayscale
    if img is None: continue

    # 3. Detect keypoints and compute descriptors
    kp, des = sift.detectAndCompute(img, None)

    if des is not None:
        all_descriptors_list.append(des)
        image_count += 1

    if (image_count + 1) % 50 == 0:
        print(f"Processed {image_count + 1} images...")

# 4. Combine all descriptors into one giant array
all_descriptors = np.vstack(all_descriptors_list)
print(f"Total descriptors found: {all_descriptors.shape}") # e.g., (150000, 128)

# 5. Save to disk
print(f"Saving all descriptors to {OUT_FILE}...")
joblib.dump(all_descriptors, OUT_FILE)
print("Step 1 Complete")