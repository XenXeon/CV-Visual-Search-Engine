import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans
from config import *

# --- Configuration ---
# Paths are built from the imported config variable
MAX_PER_IMAGE = 500
DESC_IN_FILE = os.path.join(DESCRIPTOR_FOLDER, 'bovw_all_sift_descriptors_{MAX_PER_IMAGE}.pkl')
CODEBOOK_OUT_FILE = os.path.join(DESCRIPTOR_FOLDER, 'bovw_codebook.pkl')
K_WORDS = 200 # The number of "visual words" (bins in our histogram)
print(f"--- Step 2: Creating Codebook with k={K_WORDS} ---")

# 1. Load all descriptors
print(f"Loading descriptors from {DESC_IN_FILE}...")

try:
    all_descriptors = joblib.load(DESC_IN_FILE)

except FileNotFoundError:
    print(f"Error: File not found at {DESC_IN_FILE}")
    print("Please run 'bovw_step1_extract_all_sift.py' first.")
    exit()

# 2. Use MiniBatchKMeans (faster than standard KMeans)
# This is the same logic as kmeanstest_plus.py but much faster
print("Clustering with MiniBatchKMeans...")
kmeans = MiniBatchKMeans(n_clusters=K_WORDS,
                         verbose=True,
                         batch_size=1024, # Optimized for speed
                         n_init=10,
                         max_iter=500,
                         random_state=42) # For reproducible results
kmeans.fit(all_descriptors)

# 3. The codebook is the cluster centers
codebook = kmeans.cluster_centers_

# 4. Save the codebook
print(f"Saving codebook (shape: {codebook.shape}) to {CODEBOOK_OUT_FILE}...")
joblib.dump(codebook, CODEBOOK_OUT_FILE)
print("--- Step 2 Complete ---")