"""
COMPUTE DESCRIPTORS 

This script is the main executor for the feature extraction pipeline.

It reads the experiment settings (like DESCRIPTOR_MODE) from the
'config.py' file. It then iterates through all '.bmp' images in the
'IMAGE_FOLDER', computes the specified feature descriptor for each
image, and saves the resulting feature vector as a '.mat' file in
the 'DESCRIPTOR_PATH'.

It also times the entire extraction process and saves the total
and average time to 'extraction_time.txt' in the same folder.
"""

import os
import numpy as np
import cv2  # For reading images
import scipy.io as sio  # For saving descriptors as .mat files
import time

# --- Custom Imports ---
# Imports all extraction functions (e.g., extract_global_histogram)
from extract_descriptor import * # Imports all global paths and settings (e.g., DESCRIPTOR_MODE)
from config import * 

def main():
    """
    Main function to run the descriptor computation loop.
    """
    print(f"Starting Descriptor Computation")
    # These settings are loaded from config.py
    print(f"Mode: {DESCRIPTOR_MODE}")
    print(f"Outputting to: {DESCRIPTOR_PATH}")

    ## 1. SETUP
    # Ensure the output directory (e.g., '.../descriptors/lbp_4x4_bins10')
    # exists before we try to save files to it.
    os.makedirs(DESCRIPTOR_PATH, exist_ok=True)

    # Start a high-precision timer
    t_start = time.perf_counter()
    image_count = 0

    ## 2. MAIN IMAGE LOOP
    # Loop through every file in the source image folder
    for filename in os.listdir(IMAGE_FOLDER):
        # Process only the .bmp image files
        if filename.endswith(".bmp"):
            # Construct the full, absolute path to the image
            img_path = os.path.join(IMAGE_FOLDER, filename)
            # Read the image using OpenCV
            img = cv2.imread(img_path) 
            
            # Safety check: If cv2.imread fails, it returns None
            if img is None:
                print(f"Warning: Could not read {img_path}. Skipping.")
                continue 
            
            image_count += 1
            # Define the output path for the descriptor .mat file
            # e.g., '.../descriptors/lbp/1_1_s.mat'
            fout = os.path.join(DESCRIPTOR_PATH, filename.replace('.bmp', '.mat'))
            
            # --- FEATURE EXTRACTION ROUTING ---
            # This block reads the DESCRIPTOR_MODE from the config
            # and calls the correct extraction function.
            
            # 3D Color Histogram
            if DESCRIPTOR_MODE == 'global':
                F = extract_global_histogram(img, q=GLOBAL_Q)
                
            # Grid-based Color (3D histogram per cell)
            elif DESCRIPTOR_MODE == 'grid_color':
                F = extract_spatial_grid(img, 
                                         grid_size=(GRID_ROWS, GRID_COLS), 
                                         descriptor_type='color')
                
            # Grid-based Texture (Edge Orientation Histogram per cell)
            elif DESCRIPTOR_MODE == 'grid_texture':
                F = extract_spatial_grid(img, 
                                         grid_size=(GRID_ROWS, GRID_COLS), 
                                         descriptor_type='texture',
                                         angular_bins=ANGULAR_BINS)
                
            # Grid-based Color + Texture (ColorHist + EOH per cell)
            elif DESCRIPTOR_MODE == 'grid_col_tex':
                F = extract_spatial_grid(img, 
                                         grid_size=(GRID_ROWS, GRID_COLS), 
                                         descriptor_type='color_and_texture',
                                         angular_bins=ANGULAR_BINS)
                
            # Grid-based LBP (LBP histogram per cell)
            elif DESCRIPTOR_MODE == 'lbp':
                F = extract_spatial_grid(img, 
                                         grid_size=(GRID_ROWS, GRID_COLS), 
                                         descriptor_type='lbp',
                                         angular_bins=ANGULAR_BINS) # angular_bins is used as P (points)

            # Grid-based Color + LBP (Avg Color + LBP hist per cell)
            elif DESCRIPTOR_MODE == 'color_and_lbp':
                F = extract_spatial_grid(img, 
                                         grid_size=(GRID_ROWS, GRID_COLS), 
                                         descriptor_type='color_and_lbp',
                                         angular_bins=ANGULAR_BINS) # angular_bins is used as P (points)
            
            # Bag of Visual Words (Global SIFT + Codebook)
            elif DESCRIPTOR_MODE == 'bovw':
                F = extract_bovw_histogram(img)
            
            # --- Save the descriptor ---
            # We save the feature vector 'F' in a dictionary {'F': F}
            # This makes the .mat file compatible with MATLAB.
            sio.savemat(fout, {'F': F})

    ## 3. TIMING AND FINALIZATION
    t_end = time.perf_counter()
    total_time = t_end - t_start
    
    # Avoid division by zero if no images were found
    avg_time_per_image = total_time / image_count if image_count > 0 else 0 

    print("\nDescriptor Computation Complete")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total images: {image_count}")
    print(f"Average time: {avg_time_per_image*1000:.2f} ms per image")

    # Save timing info to a file in the same descriptor folder
    # The 'TIME_FILE' path is loaded from config.py
    with open(TIME_FILE, 'w') as f:
        f.write(f"total_time: {total_time}\n")
        f.write(f"image_count: {image_count}\n")
        f.write(f"avg_time_ms: {avg_time_per_image*1000}\n")

    print(f"Saved timing info to {TIME_FILE}")


# Standard Python entry point
if __name__ == "__main__":
    main()