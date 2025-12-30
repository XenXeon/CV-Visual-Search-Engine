"""
COMPUTE PCA (Principal Component Analysis) 

This script performs Principal Component Analysis (PCA) on a set of
existing, high-dimensional descriptors.

"""

import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib  # For saving/loading scikit-learn models (scaler, pca)
from config import * # Import all settings (DESCRIPTOR_PATH, PCA_DESCRIPTOR_PATH, etc.)

def main():
    # This script reads from DESCRIPTOR_PATH and saves models and
    # new data to the paths defined in config.py (PCA_*)

    ## 1. LOAD ALL ORIGINAL DESCRIPTORS
    print(f"--- Computing PCA ---")
    print(f"Loading descriptors from: {DESCRIPTOR_PATH}")
    
    ALLFEAT = []  # This list will hold all (e.g., 512-dim) feature vectors
    ALLFILES = [] # This list will hold their corresponding filenames
    
    for filename in os.listdir(DESCRIPTOR_PATH):
        if filename.endswith('.mat'):
            file_path = os.path.join(DESCRIPTOR_PATH, filename)
            img_data = sio.loadmat(file_path)
            ALLFILES.append(filename)      # Store filename (e.g., '1_1_s.mat')
            ALLFEAT.append(img_data['F'][0]) # Store the 1D feature vector

    # Convert the list of 1D vectors into a single 2D NumPy array
    # Shape will be (N_images x D_features)
    ALLFEAT = np.array(ALLFEAT)
    NIMG = ALLFEAT.shape[0]
    
    if NIMG == 0:
        print(f"Error: No .mat files found in {DESCRIPTOR_PATH}. Exiting.")
        return

    print(f"Loaded {NIMG} descriptors.")

    ## 2. SCALE THE DATA
    # PCA is highly sensitive to the variance of features.
    # We must scale the data first so all features have
    # zero mean and unit variance (a "Z-score" normalization).
    print("Scaling data (zero mean, unit variance)...")
    scaler = StandardScaler()
    
    # 'fit_transform' learns the mean and std dev of the data,
    # and then applies the scaling.
    ALLFEAT_scaled = scaler.fit_transform(ALLFEAT)

    ## 3. COMPUTE PCA
    
    # Set the number of components to keep.
    N_COMPONENTS = 20
    pca = PCA(n_components=N_COMPONENTS)

    print(f"Fitting PCA (reducing to {N_COMPONENTS} components)...")
    
    # 'fit' computes the principal components (eigenvectors)
    # from the scaled data.
    pca.fit(ALLFEAT_scaled)

    # The eigenvalues (pca.explained_variance_) are the variance
    # of the data along each principal component. This is
    # crucial for calculating the Mahalanobis distance in PCA space.
    eigenvalues = pca.explained_variance_

    print(f"PCA complete.")
    print(f"Original dimensions: {ALLFEAT.shape[1]}")
    # pca.n_components_ is the actual number of components kept
    print(f"New dimensions: {pca.n_components_}")

    ## 4. SAVE THE MODELS AND EIGENVALUES
    # We save the 'scaler' and 'pca' models so we can
    # project *new* data in the future without re-fitting.
    
    print(f"Saving PCA model to: {PCA_MODEL_FILE}")
    joblib.dump(pca, PCA_MODEL_FILE)

    print(f"Saving Scaler to: {SCALER_MODEL_FILE}")
    joblib.dump(scaler, SCALER_MODEL_FILE)

    print(f"Saving eigenvalues to: {PCA_EIGENVALUES_FILE}")
    np.save(PCA_EIGENVALUES_FILE, eigenvalues)

    ## 5. PROJECT AND SAVE NEW DESCRIPTORS
    print(f"Projecting descriptors and saving to: {PCA_DESCRIPTOR_PATH}")
    # Ensure the output (e.g., '.../lbp_pca') directory exists
    os.makedirs(PCA_DESCRIPTOR_PATH, exist_ok=True)

    # 'transform' projects the scaled data onto the new PCA components
    # This converts the (N, 512) array into an (N, 20) array.
    ALLFEAT_PCA = pca.transform(ALLFEAT_scaled)

    # Loop through all the new, low-dimensional PCA vectors
    for i in range(NIMG):
        filename = ALLFILES[i] # Get the original filename (e.g., '1_1_s.mat')
        F_pca = ALLFEAT_PCA[i] # Get the new (e.g., 20-dim) vector
        
        # Define the full path for the new file
        fout_path = os.path.join(PCA_DESCRIPTOR_PATH, filename)
        
        # Save in the same .mat format for consistency with the rest
        # of the pipeline.
        sio.savemat(fout_path, {'F': F_pca})

    print("All PCA descriptors saved.")


# Standard Python entry point
if __name__ == "__main__":
    main()