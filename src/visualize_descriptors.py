"""
===================================================================
--- DESCRIPTOR VISUALIZATION TOOL ---
===================================================================

This script is a standalone, interactive tool for visualizing the
output of different feature descriptors on a random image.

It provides a command-line menu to:
1.  Load a random image from the dataset.
2.  Visualize intermediate steps for descriptors like EOH, LBP, and BoVW.
3.  Visualize the final 1D histogram for each descriptor.
4.  Visualize grid-based descriptors on a single, highlighted cell.

This file is intended for debugging and for generating figures for
reports, not for the main batch-processing pipeline.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from skimage import feature

# --- User's Code Imports ---
try:
    import config as cfg
    # Import all the necessary extraction functions from your project
    from extract_descriptor import *
except ImportError as e:
    print(f"Error: Could not import necessary files.")
    print(f"Please make sure 'config.py' and 'extract_descriptor.py' are in the same directory.")
    print(f"Details: {e}")
    exit()
except Exception as e:
    print(f"An error occurred during import: {e}")
    exit()

# --- 1. HELPER FUNCTIONS ---

def select_random_image(image_folder):
    """
    Loads a random .bmp image from the specified folder.
    
    Args:
        image_folder (str): Path to the folder containing .bmp images.
    
    Returns:
        tuple: (img, random_image_name)
               (None, None) if loading fails.
    """
    try:
        all_images = [f for f in os.listdir(image_folder) if f.endswith('.bmp')]
        if not all_images:
            print(f"Error: No .bmp images found in {image_folder}")
            return None, None
            
        # Select a random image and load it
        random_image_name = random.choice(all_images)
        image_path = os.path.join(image_folder, random_image_name)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return None, None
            
        print(f"Loaded random image: {random_image_name}")
        return img, random_image_name
    except FileNotFoundError:
        print(f"Error: Image folder not found at {image_folder}")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading a random image: {e}")
        return None, None

def plot_image_grid(image_dict, main_title, grid_shape=None, fig_size=(15, 7)):
    """
    Helper function to plot a grid of images using matplotlib.
    
    Args:
        image_dict (dict): A dictionary of {title: image_data}.
        main_title (str): The main title for the entire figure.
        grid_shape (tuple, optional): (rows, cols). Auto-calculated if None.
        fig_size (tuple, optional): The (width, height) of the figure.
    """
    num_images = len(image_dict)
    if num_images == 0:
        return

    # Automatically determine grid shape if not provided
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    else:
        rows, cols = grid_shape
        
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    fig.suptitle(main_title, fontsize=16)
    
    # Ensure 'axes' is always iterable, even for a single plot
    if num_images > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Iterate through the images and plot them
    for i, (title, img) in enumerate(image_dict.items()):
        ax = axes[i]
        
        if len(img.shape) == 2:
            # Handle grayscale images (like LBP or magnitude)
            # Use 'gray' colormap
            if img.dtype != np.uint8:
                # For non-uint8 (e.g., float LBP), scale vmin/vmax
                ax.imshow(img, cmap='gray', vmin=np.min(img), vmax=np.max(img))
            else:
                ax.imshow(img, cmap='gray')
        else:
            # Handle BGR color images
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
        ax.set_title(title)
        ax.axis('off')

    # Hide any unused subplots in the grid
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_histogram(hist_data, title, xlabel, ylabel="Frequency"):
    """
    Helper function to plot a 1D bar histogram (the final descriptor).
    
    Args:
        hist_data (np.array): The 1D feature vector.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str, optional): Y-axis label.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(hist_data)), hist_data)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- 2. DESCRIPTOR-SPECIFIC VISUALIZATION FUNCTIONS ---

def visualize_global_histogram(img, filename):
    """Visualizes the 3D color histogram and the final 1D vector."""
    print("Visualizing: Global Color Histogram")
    Q = cfg.GLOBAL_Q
    
    # 1. Calculate the 3D histogram (un-flattened)
    hist_3d = cv2.calcHist(
        [img], [2, 1, 0], None, 
        [Q, Q, Q], 
        [0, 256, 0, 256, 0, 256]
    )
    # Normalize for visualization
    sum_H = np.sum(hist_3d)
    if sum_H > 0: hist_3d_norm = hist_3d / sum_H
    else: hist_3d_norm = hist_3d

    # 2. Prepare data for the 3D scatter plot
    r_coords, g_coords, b_coords, sizes, colors = [], [], [], [], []
    SIZE_SCALE = 8000  # Scale factor to make points visible
    for r in range(Q):
        for g in range(Q):
            for b in range(Q):
                val = hist_3d_norm[r, g, b]
                if val > 0:  # Only plot bins that have pixels
                    r_coords.append(r)
                    g_coords.append(g)
                    b_coords.append(b)
                    sizes.append(val * SIZE_SCALE)
                    # Color the dot based on the bin's color
                    colors.append(((r + 0.5) / Q, (g + 0.5) / Q, (b + 0.5) / Q))

    # 3. Plot the 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r_coords, g_coords, b_coords, s=sizes, c=colors, alpha=0.7, edgecolors='w', linewidth=0.5)
    ax.set_xlabel('Red Bin'); ax.set_ylabel('Green Bin'); ax.set_zlabel('Blue Bin')
    ax.set_xticks(range(Q)); ax.set_yticks(range(Q)); ax.set_zticks(range(Q))
    ax.set_title(f"3D Color Histogram (q={Q})\n{filename}", fontsize=16)
    
    # 4. Show the original image and the 3D plot
    plot_image_grid(
        {"Original Image": img},
        "Global Histogram Input",
        grid_shape=(1, 1),
        fig_size=(6, 6)
    )
    print("Displaying 3D histogram plot...")
    plt.show()

    # 5. Show the final 1D descriptor (what's saved)
    F = extract_global_histogram(img, q=Q)
    plot_histogram(
        F,
        f"Final 1D Descriptor (Flattened, Length={len(F)})",
        "Descriptor Dimension"
    )

def visualize_spatial_grid(img, filename):
    """Draws the grid on the image and highlights/extracts one cell."""
    print("Visualizing: Spatial Grid")
    
    rows, cols = cfg.GRID_ROWS, cfg.GRID_COLS
    h, w, _ = img.shape
    cell_h, cell_w = h // rows, w // cols
    
    # 1. Draw the grid lines on a copy of the image
    img_with_grid = img.copy()
    for i in range(1, rows):
        y = i * cell_h
        cv2.line(img_with_grid, (0, y), (w, y), (0, 255, 0), 2)  # Green horizontal lines
    for j in range(1, cols):
        x = j * cell_w
        cv2.line(img_with_grid, (x, 0), (x, h), (0, 255, 0), 2)  # Green vertical lines

    # 2. Highlight and extract a sample cell (e.g., one from the middle)
    hl_row, hl_col = rows // 2, cols // 2
    y1, y2 = hl_row * cell_h, (hl_row + 1) * cell_h
    x1, x2 = hl_col * cell_w, (hl_col + 1) * cell_w
    cv2.rectangle(img_with_grid, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red rectangle
    
    cell = img[y1:y2, x1:x2]

    # 3. Plot the original, the grid, and the extracted cell
    plot_image_grid(
        {
            "Original Image": img,
            f"{rows}x{cols} Grid (Cell {hl_row},{hl_col} Highlighted)": img_with_grid,
            f"Extracted Cell ({hl_row},{hl_col})": cell
        },
        f"Spatial Grid Visualization\n{filename}",
        grid_shape=(1, 3),
        fig_size=(18, 6)
    )
    
    # Return the cell for further processing (e.g., running EOH on it)
    return cell, (hl_row, hl_col)

def visualize_eoh(cell, cell_name="Image"):
    """Visualizes the intermediate steps of EOH (Sobel, Mag, Orient)."""
    print(f"Visualizing: Edge Orientation Histogram (EOH) on {cell_name}")
    
    if cell.shape[0] < 3 or cell.shape[1] < 3:
        print(f"Warning: Cell is too small ({cell.shape}) to compute Sobel. Skipping EOH.")
        return

    # 1. Compute intermediate steps
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    sobel_x_64f = cv2.Sobel(gray_cell, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_64f = cv2.Sobel(gray_cell, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x_64f, sobel_y_64f)
    orientation_deg = cv2.phase(sobel_x_64f, sobel_y_64f, angleInDegrees=True) 

    # 2. Convert to 8-bit images for visualization
    sobel_x_viz = cv2.convertScaleAbs(sobel_x_64f)
    sobel_y_viz = cv2.convertScaleAbs(sobel_y_64f)
    magnitude_viz = cv2.convertScaleAbs(magnitude)
    
    # 3. Create HSV-based orientation visualization
    orientation_hsv = np.zeros_like(cell, dtype=np.uint8)
    orientation_hsv[..., 0] = orientation_deg / 2  # Hue = Angle (OpenCV uses 0-179 for Hue)
    orientation_hsv[..., 1] = 255                  # Saturation = 255
    orientation_hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) # Value = Magnitude
    orientation_viz = cv2.cvtColor(orientation_hsv, cv2.COLOR_HSV2BGR)
    
    # 4. Plot the grid of intermediate steps
    plot_image_grid(
        {
            f"Original {cell_name}": cell,
            "Grayscale": gray_cell,
            "Sobel X": sobel_x_viz,
            "Sobel Y": sobel_y_viz,
            "Magnitude": magnitude_viz,
            "Orientation (Hue=Angle)": orientation_viz
        },
        f"EOH Intermediate Steps for {cell_name}",
        grid_shape=(2, 3)
    )

    # 5. Plot the final 1D descriptor
    bins = cfg.ANGULAR_BINS
    eoh_hist = compute_eoh(cell, angular_bins=bins)
    plot_histogram(
        eoh_hist,
        f"Final EOH Descriptor ({bins} Bins)\n(Normalized)",
        "Angle Bin Index (0 to 2*pi)"
    )

def visualize_lbp(cell, cell_name="Image"):
    """Visualizes the scikit-image LBP image and its histogram."""
    print(f"Visualizing: scikit-image LBP on {cell_name}")
    
    P, R = 8, 1  # 8 points, radius 1
    if cell.shape[0] < (2*R + 1) or cell.shape[1] < (2*R + 1):
        print(f"Warning: Cell is too small ({cell.shape}) for LBP. Skipping.")
        return

    # 1. Compute intermediate steps
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    # Use scikit-image's function
    lbp_image = feature.local_binary_pattern(gray_cell, P, R, method='default')
    
    # 2. Plot the steps
    plot_image_grid(
        {
            f"Original {cell_name}": cell,
            "Grayscale": gray_cell,
            "LBP Image (skimage)": lbp_image.astype(np.uint8) # Convert to 8-bit for vis
        },
        f"LBP (skimage) Intermediate Steps for {cell_name}",
        grid_shape=(1, 3),
        fig_size=(18, 6)
    )
    
    # 3. Plot the final 1D descriptor
    lbp_hist = compute_lbp(cell, P=P, R=R)
    plot_histogram(
        lbp_hist,
        f"Final LBP Descriptor ({len(lbp_hist)} Bins)\n(Normalized)",
        f"LBP Code (0 to {2**P - 1})"
    )

def visualize_bovw(img, filename):
    """Visualizes SIFT keypoints and the final BoVW histogram."""
    print("Visualizing: Bag of Visual Words (BoVW)")
    
    if CODEBOOK is None:
        print("Error: BoVW Codebook is not loaded. Cannot visualize.")
        return

    # 1. Detect SIFT keypoints
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=500)
    kp, des = sift.detectAndCompute(gray_img, None)
    
    if des is None:
        print("No SIFT features found in this image.")
        return

    # 2. Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(
        gray_img, kp, img.copy(), 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    print(f"Found {len(kp)} SIFT keypoints.")
    
    # 3. Plot the original and the keypoint image
    plot_image_grid(
        {
            "Original Image": img,
            f"Image with {len(kp)} SIFT Keypoints": img_with_keypoints
        },
        f"BoVW Keypoint Detection\n{filename}",
        grid_shape=(1, 2),
        fig_size=(15, 7)
    )

    # 4. Plot the final 1D descriptor
    bovw_hist = extract_bovw_histogram(img)
    plot_histogram(
        bovw_hist,
        f"Final BoVW Descriptor ({len(bovw_hist)} Visual Words)\n(Normalized)",
        "Visual Word Index"
    )

def visualize_block_lbp(cell, cell_name="Image"):
    """Visualizes the steps of the *manual* block-based LBP."""
    print(f"Visualizing: Manual Block LBP on {cell_name}")

    block_size = 3 # This is hardcoded in the compute_block_lbp_hist function
    
    # --- 1. Re-compute intermediate images (for visualization) ---
    
    # --- 1a. Block-Averaged Image ---
    new_h = cell.shape[0] // block_size
    new_w = cell.shape[1] // block_size
    
    if new_h < 3 or new_w < 3: # Need at least 3x3 blocks for LBP
        print(f"Warning: Cell is too small ({cell.shape}) for 3x3 Block LBP. Skipping.")
        return

    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    # This resize with INTER_AREA is a fast way to get block averages
    block_mean_img = cv2.resize(gray_cell, (new_w, new_h), 
                                interpolation=cv2.INTER_AREA)

    # --- 1b. Manual LBP on the block-averaged image ---
    # This code is duplicated from compute_block_lbp_hist
    gray_cell_padded = np.pad(block_mean_img, 1, mode='constant', constant_values=0)
    center = gray_cell_padded[1:-1, 1:-1]
    n0 = gray_cell_padded[0:-2, 0:-2]; n1 = gray_cell_padded[0:-2, 1:-1]; n2 = gray_cell_padded[0:-2, 2:  ]
    n3 = gray_cell_padded[1:-1, 2:  ]; n4 = gray_cell_padded[2:  , 2:  ]; n5 = gray_cell_padded[2:  , 1:-1]
    n6 = gray_cell_padded[2:  , 0:-2]; n7 = gray_cell_padded[1:-1, 0:-2]
    
    b0 = (n0 >= center).astype(np.uint8); b1 = (n1 >= center).astype(np.uint8)
    b2 = (n2 >= center).astype(np.uint8); b3 = (n3 >= center).astype(np.uint8)
    b4 = (n4 >= center).astype(np.uint8); b5 = (n5 >= center).astype(np.uint8)
    b6 = (n6 >= center).astype(np.uint8); b7 = (n7 >= center).astype(np.uint8)

    # Combine bits to get the final LBP code image
    lbp_image = (b0 * 1) + (b1 * 2) + (b2 * 4) + (b3 * 8) + \
                (b4 * 16) + (b5 * 32) + (b6 * 64) + (b7 * 128)
    
    # --- 2. Plot the intermediate steps ---
    plot_image_grid(
        {
            f"Original {cell_name}": cell,
            f"Grayscale Cell ({gray_cell.shape})": gray_cell,
            f"Block-Averaged Image ({block_mean_img.shape})": block_mean_img,
            f"Final LBP Image ({lbp_image.shape})": lbp_image.astype(np.uint8)
        },
        f"Manual Block LBP Intermediate Steps for {cell_name}",
        grid_shape=(2, 2)
    )


# --- 3. MAIN EXECUTION (MENU) ---

def main():
    """Runs the interactive command-line menu."""
    img, filename = select_random_image(cfg.IMAGE_FOLDER)
    if img is None:
        return  # Exit if image loading failed

    while True:
        # Display the menu
        print("\n--- Descriptor Visualization Menu ---")
        print(f"Image: {filename}")
        print("1. Global Color Histogram (3D Plot + 1D Vector)")
        print("2. Spatial Grid (Draws the grid)")
        print("3. EOH (on full image)")
        print("4. LBP (scikit-image, on full image)")
        print("5. Bag of Visual Words (BoVW)")
        print("--- Combinations (on one cell) ---")
        print("6. Spatial Grid + EOH")
        print("7. Spatial Grid + LBP (scikit-image)")
        print("---------------------------------")
        print("L. Load new random image")
        print("q. Quit")
        
        choice = input("Enter your choice: ").strip().lower()
        
        # --- Handle Menu Choices ---
        
        if choice == '1':
            # 1. Global Color Histogram
            visualize_global_histogram(img, filename)
            
        elif choice == '2':
            # 2. Spatial Grid
            visualize_spatial_grid(img, filename)
            
        elif choice == '3':
            # 3. EOH on Full Image
            visualize_eoh(img, cell_name=f"Full Image ({filename})")
            
        elif choice == '4':
            # 4. LBP on Full Image
            visualize_lbp(img, cell_name=f"Full Image ({filename})")
            
        elif choice == '5':
            # 5. BoVW
            visualize_bovw(img, filename)
            
        elif choice == '6':
            # 6. Grid + EOH
            print("First, showing the grid...")
            cell, (r, c) = visualize_spatial_grid(img, filename)
            print("\nNext, running EOH on the highlighted cell...")
            visualize_eoh(cell, cell_name=f"Cell ({r},{c})")
            
        elif choice == '7':
            # 7. Grid + LBP (skimage)
            print("First, showing the grid...")
            cell, (r, c) = visualize_spatial_grid(img, filename)
            print("\nNext, running LBP (skimage) on the highlighted cell...")
            visualize_lbp(cell, cell_name=f"Cell ({r},{c})")
            
        elif choice == 'L':
            # l. Load new image
            img, filename = select_random_image(cfg.IMAGE_FOLDER)
            if img is None:
                return # Exit if loading failed
                
        elif choice == 'q':
            # q. Quit
            print("Exiting.")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()