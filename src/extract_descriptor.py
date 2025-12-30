"""
FEATURE DESCRIPTOR EXTRACTION FUNCTIONS 

This file contains all the core functions used to extract
different types of feature descriptors from images, including:
- 3D Color Histogram (Global)
- Bag of Visual Words (BoVW)
- Edge Orientation Histogram (EOH)
- Local Binary Patterns (LBP)
- Spatial Grid (which combines the above methods)
"""

import numpy as np
import cv2
import joblib  # For loading the pickled BoVW codebook
from sklearn.neighbors import NearestNeighbors  # For BoVW word lookup
from skimage import feature  # For calculating LBP
from config import * # Imports global configuration variables

# --- 1. BOVW CODEBOOK INITIALIZATION ---
# This block runs once when the module is imported.
# It loads the pre-computed BoVW k-means cluster centers (the "codebook")
# and pre-fits a NearestNeighbors model to it. This allows for
# very fast "quantization" (finding the-nearest-word) in the
# extract_bovw_histogram function.

try:
    # This path must be correct for BoVW to work.
    CODEBOOK_PATH = r'D:\Surrey\Computer Vision\Skeleton_Python-R1\Skeleton_Python\MSRC_ObjCategImageDatabase_v2\descriptors\bovw_codebook.pkl'
    CODEBOOK = joblib.load(CODEBOOK_PATH)
    K_WORDS = CODEBOOK.shape[0]  # Get K (number of words)
    
    # Pre-fit a 1-Nearest-Neighbor model for fast lookups
    NN_MODEL = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(CODEBOOK)
    print(f"Successfully loaded BoVW Codebook with k={K_WORDS} words.")
except FileNotFoundError:
    print("Warning: BoVW Codebook not found. BoVW descriptor will not work.")
    CODEBOOK = None

# --- 2. DESCRIPTOR FUNCTIONS ---

def extract_global_histogram(img, q=16):
    """
    Computes a 3D color histogram for the entire image.
    
    Args:
        img (np.array): The input image (in BGR format).
        q (int): The number of bins *per channel*.
    
    Returns:
        np.array: A 1D, L1-normalized feature vector of size (q*q*q).
    """
    
    # 1. Calculate the 3D histogram
    hist = cv2.calcHist(
        [img],
        [2, 1, 0],  # Channels: R(2), G(1), B(0)
        None,       # No mask
        [q, q, q],  # Number of bins for R, G, B
        [0, 256, 0, 256, 0, 256] # Ranges for R, G, B
    )
    
    # 2. Flatten the 3D (q, q, q) histogram into a 1D (q^3) vector
    H = hist.flatten()
    
    # 3. L1 Normalize the histogram (so all bins sum to 1)
    sum_H = np.sum(H)
    if sum_H > 0:
        F = H / sum_H
    else:
        F = H  # It's already all zeros
        
    return F

def extract_spatial_grid(img, grid_size=(4, 4), descriptor_type='color_and_texture', angular_bins=8):
    """
    Computes a spatial grid descriptor.
    
    This function divides the image into a grid (e.g., 4x4) and
    computes a descriptor for each cell. It then concatenates
    all cell descriptors into one long feature vector.
    
    Args:
        img (np.array): The input image (BGR format).
        grid_size (tuple): A tuple (rows, cols) for the grid.
        descriptor_type (str): Type of descriptor to compute per cell.
            Valid options: 'color', 'texture' (EOH), 'lbp',
                           'color_and_texture' (ColorHist + EOH),
                           'color_and_lbp' (AvgColor + LBP).
        angular_bins (int): Number of bins for EOH or LBP.
        
    Returns:
        np.array: A 1D, L2-normalized feature vector.
    """
    h, w, _ = img.shape
    rows, cols = grid_size
    
    # Calculate cell height and width
    cell_h = h // rows
    cell_w = w // cols
    
    final_descriptor = []  # This list will hold all cell descriptors
    
    # Iterate over each cell in the grid
    for i in range(rows):
        for j in range(cols):
            # 1. Extract the cell region
            y_start = i * cell_h
            y_end = (i + 1) * cell_h
            x_start = j * cell_w
            x_end = (j + 1) * cell_w
            
            # Ensure we get the last pixels if not perfectly divisible
            if i == rows - 1: y_end = h
            if j == cols - 1: x_end = w
                
            cell = img[y_start:y_end, x_start:x_end]
            cell_descriptor_parts = []

            # 2. Compute the descriptor(s) for the cell
            if descriptor_type == 'color':
                # --- Color Feature (3D Histogram, 4 bins per channel) ---
                color_part = extract_global_histogram(cell, q=4)
                cell_descriptor_parts = [color_part]
            
            elif descriptor_type == 'texture':
                # --- Texture Feature (Edge Orientation Histogram) ---
                cell_descriptor_parts = [compute_eoh(cell, angular_bins)]

            elif descriptor_type == 'lbp':
                # --- Texture Feature (Local Binary Patterns) ---
                cell_descriptor_parts = [compute_lbp(cell, angular_bins)]
            
            elif descriptor_type == 'color_and_texture':
                # --- Combined Feature (ColorHist + EOH) ---
                # First, get color (4x4x4 histogram)
                color_part = extract_global_histogram(cell, q=4)
                # Second, get texture (EOH)
                texture_part = compute_eoh(cell, angular_bins)
                
                cell_descriptor_parts.append(color_part)
                cell_descriptor_parts.append(texture_part)

            elif descriptor_type == 'color_and_lbp':
                # --- Combined Feature (AvgColor + LBP) ---
                # Get average R, G, B
                avg_bgr = np.mean(cell, axis=(0, 1))
                color_part = [avg_bgr[2], avg_bgr[1], avg_bgr[0]] # R, G, B
                # Get LBP
                texture_part = compute_lbp(cell, angular_bins)
                
                cell_descriptor_parts.append(color_part)
                cell_descriptor_parts.append(texture_part)

            else:
                raise ValueError("Unknown descriptor_type. Use 'color', 'texture', 'lbp', 'color_and_texture', or 'color_and_lbp'.")
                
            # 3. Concatenate and append the cell's descriptor
            # np.hstack flattens and joins all parts into a 1D vector
            final_descriptor.extend(np.hstack(cell_descriptor_parts))
    
    # 4. Normalize the final concatenated descriptor
    # L2 norm is generally good for concatenated histograms.
    F = np.array(final_descriptor)
    norm = np.linalg.norm(F)
    if norm > 0:
        F = F / norm
        
    return F

def compute_eoh(cell, angular_bins):
    """
    Computes an Edge Orientation Histogram (EOH) for a single image cell.
    
    Args:
        cell (np.array): The input image cell (BGR).
        angular_bins (int): The number of orientation bins (e.g., 8).
    
    Returns:
        np.array: A 1D, L1-normalized EOH of size (angular_bins).
    """
    # Handle empty cells
    if cell.shape[0] == 0 or cell.shape[1] == 0:
        return np.zeros(angular_bins)
        
    # 1. Convert to grayscale
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    
    # 2. Calculate gradients using Sobel operators
    sobel_x = cv2.Sobel(gray_cell, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_cell, cv2.CV_64F, 0, 1, ksize=3)
    
    # 3. Calculate magnitude and orientation (in radians)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    orientation = cv2.phase(sobel_x, sobel_y, angleInDegrees=False) # Range [0, 2pi]

    # 4. Create the histogram
    # We use the magnitude as a weight, so strong edges
    # contribute more to the histogram.
    hist, _ = np.histogram(
        orientation.flatten(),
        bins=angular_bins,
        range=(0, 2 * np.pi),
        weights=magnitude.flatten()
    )
    
    # 5. L1 Normalize the cell histogram
    sum_hist = np.sum(hist)
    if sum_hist > 0:
        hist = hist / sum_hist
        
    return hist

def extract_bovw_histogram(img):
    """
    Computes a Bag of Visual Words (BoVW) histogram for a single image.
    
    Args:
        img (np.array): The input image (BGR).
    
    Returns:
        np.array: A 1D, L2-normalized histogram of visual word counts.
    """
    if CODEBOOK is None:
        raise FileNotFoundError("BoVW Codebook is not loaded. Run the BoVW codebook generation script first.")

    # 1. Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Initialize SIFT (or another feature detector)
    sift = cv2.SIFT_create(nfeatures=500)

    # 3. Detect keypoints and compute SIFT descriptors
    kp, des = sift.detectAndCompute(gray_img, None)

    # 4. Create an empty histogram with K_WORDS bins
    hist = np.zeros(K_WORDS)

    if des is not None:
        # 5. Quantize: Find the nearest "word" in the codebook for each descriptor
        # We use the pre-fitted NN_MODEL for fast lookup.
        distances, indices = NN_MODEL.kneighbors(des)
        
        # 6. Build the histogram: Count the occurrences of each word
        # indices is a (N_descriptors x 1) array, so we just iterate it.
        for i in indices:
            hist[i[0]] += 1

    # 7. Normalize the histogram (L2 norm is common for BoVW)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm
        
    return hist

def compute_lbp(cell, P=10, R=1):
    """
    Computes a Local Binary Pattern (LBP) histogram for a single image cell
    using the scikit-image library.
    
    Args:
        cell (np.array): The input image cell (BGR).
        P (int): Number of sampling points for LBP.
        R (int): Radius for LBP.
    
    Returns:
        np.array: A 1D, L1-normalized histogram of LBP codes.
    """
    # Calculate the number of bins, which is 2^P
    n_bins = 2**P
    
    # Handle empty cells
    if cell.shape[0] == 0 or cell.shape[1] == 0:
        return np.zeros(n_bins)
        
    # 1. Convert to grayscale
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    
    # 2. Calculate LBP features using scikit-image
    # 'default' method gives 2^P bins (e.g., 256 for P=8)
    lbp = feature.local_binary_pattern(gray_cell, P, R, method='default')
    
    # 3. Create histogram of LBP codes
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )
    
    # 4. L1 Normalize the cell histogram
    sum_hist = np.sum(hist)
    if sum_hist > 0:
        hist = hist / sum_hist
        
    return hist