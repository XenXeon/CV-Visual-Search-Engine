# --- 0. IMPORTS ---
import os
import numpy as np
import scipy.io as sio  # For loading .mat descriptor files
import cv2  # For loading and manipulating images
from random import randint  # For picking a random query
import matplotlib.pyplot as plt  # For displaying the final results grid

# --- Custom/Local Imports ---
# cvpr_compare: Assumed to contain the distance metric functions (e.g., L1, L2, Mahalanobis)
from cvpr_compare import cvpr_compare
# config.py: Assumed to contain all global configuration variables
# (e.g., paths, metric, and visualization settings)
from config import *

# --- 1. INITIALIZATION & DATA LOADING ---
print(f"--- Starting Visual Search ---")
print(f"Experiment ID: {EXPERIMENT_ID_STRING}")

eigenvalues = None  # Will hold eigenvalues if METRIC_TO_TEST is 'Mahalanobis'
current_descriptor_path = ""  # The folder we will read descriptors from

# Check config to see if we should use PCA-processed data or original data
if USE_PCA_DATA:
    print(f"Loading PCA descriptors from: {PCA_DESCRIPTOR_PATH}")
    current_descriptor_path = PCA_DESCRIPTOR_PATH
    
    # If using Mahalanobis distance, we MUST also load the eigenvalues
    if METRIC_TO_TEST == 'Mahalanobis':
        try:
            print(f"Loading eigenvalues from: {PCA_EIGENVALUES_FILE}")
            eigenvalues = np.load(PCA_EIGENVALUES_FILE)
        except FileNotFoundError:
            print(f"Error: Eigenvalues file not found... Please run 'compute_pca.py' first.")
            exit()
else:
    # Using original, non-PCA descriptors
    print(f"Loading original descriptors from: {DESCRIPTOR_PATH}")
    current_descriptor_path = DESCRIPTOR_PATH

# --- Load all descriptor files from the chosen path ---
ALLFEAT = []  # This list will hold all feature vectors (the data)
ALLFILES = []  # This list will hold the corresponding file paths (the labels)

print("Loading all descriptors into memory...")
for filename in os.listdir(current_descriptor_path):
    if filename.endswith('.mat'):
        # Construct the full path to the .mat file
        mat_path = os.path.join(current_descriptor_path, filename)
        
        # Load the .mat file
        img_data = sio.loadmat(mat_path)
        
        # Store the file path
        ALLFILES.append(mat_path)
        # Store the feature vector. Assumes data is in key 'F' and is a 1D array.
        ALLFEAT.append(img_data['F'][0])

# Convert the list of feature vectors into a single N x D NumPy array
# (N = number of images, D = descriptor dimensions)
ALLFEAT = np.array(ALLFEAT)

# Get the total number of images loaded
NIMG = ALLFEAT.shape[0]

# Error check: Exit if no descriptors were found
if NIMG == 0:
    print(f"Error: No .mat files found in {current_descriptor_path}")
    exit()
    
print(f"Successfully loaded {NIMG} descriptors.")

# --- 2. SELECT QUERY IMAGE ---
# Create a lookup dictionary to find an image's index (0 to NIMG-1)
# from its base filename (e.g., '1_1_s')
file_basename_to_index = {}
for i, f_path in enumerate(ALLFILES):
    base = os.path.basename(f_path)  # e.g., '1_1_s.mat'
    name = os.path.splitext(base)[0]  # e.g., '1_1_s'
    file_basename_to_index[name] = i
    file_basename_to_index[name + ".bmp"] = i  # Also allow searching with .bmp

print("\n--- 2. Select Query Image ---")
query_name = input("Enter image name (e.g., '1_1_s') or press Enter for random: ").strip()

queryimg = -1  # This will hold the index of the query image

if query_name == "":
    # User pressed Enter, so pick a random image index
    queryimg = randint(0, NIMG - 1)
    print("Selected random image.")
else:
    # User entered a name, try to find it in our lookup dictionary
    if query_name in file_basename_to_index:
        queryimg = file_basename_to_index[query_name]
    elif query_name + ".bmp" in file_basename_to_index:
        queryimg = file_basename_to_index[query_name + ".bmp"]
    elif os.path.splitext(query_name)[0] in file_basename_to_index:
        queryimg = file_basename_to_index[os.path.splitext(query_name)[0]]

# Fallback: If the name wasn't found, pick a random image
if queryimg == -1:
    print(f"Error: Could not find image '{query_name}'. Picking a random image instead.")
    queryimg = randint(0, NIMG - 1)

# --- Extract the query vector and its name ---
query_vector = ALLFEAT[queryimg]  # The actual descriptor (e.g., a 512-dim vector)
query_descriptor_path = ALLFILES[queryimg]  # Full path to its .mat file
query_base_filename = os.path.basename(query_descriptor_path)  # '1_1_s.mat'
query_image_name = os.path.splitext(query_base_filename)[0]  # '1_1_s'

print(f"Query image selected: {query_image_name}")

# --- 3. COMPUTE DISTANCES ---
print(f"--- Computing Distances using {METRIC_TO_TEST} ---")

# 'dst' will be a list of tuples: (distance, index)
dst = []
for i in range(NIMG):
    # Get the feature vector for the i-th image
    candidate = ALLFEAT[i]
    
    # Compare the query vector to the candidate vector
    distance = cvpr_compare(query_vector, candidate, 
                            metric=METRIC_TO_TEST, 
                            eigenvalues=eigenvalues)  # eigenvalues is None unless using Mahalanobis
    
    # Store the result
    dst.append((distance, i))

# Sort the list based on the distance (the first element of the tuple)
# This brings the closest matches to the top.
dst.sort(key=lambda x: x[0])
print("Distance computation complete.")

# --- 4. BUILD VISUAL RESULTS CANVAS ---
print("Building results grid...")
# Initialize a blank (black) canvas to tile our results onto
# Dimensions are defined in config.py (e.g., 3 rows * 150px, 5 cols * 200px)
canvas_height = VISUAL_GRID_ROWS * THUMB_SIZE[1]
canvas_width = VISUAL_GRID_COLS * THUMB_SIZE[0]
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Loop through the top 'SHOW' results (e.g., top 15)
# 'SHOW' is defined in config.py
for i in range(SHOW):
    
    # 1. Get the index of the i-th best match from our sorted list
    match_index = dst[i][1]
    
    # 2. Get the .mat descriptor path for that match
    descriptor_path = ALLFILES[match_index]
    
    # 3. Derive the image file path from the descriptor file path
    base_filename = os.path.basename(descriptor_path)  # e.g., '1_1_s.mat'
    # Assumes image extension is .bmp, defined in config.py
    image_filename = os.path.splitext(base_filename)[0] + '.bmp' 
    
    # 4. Create the full path to the image
    # IMAGE_FOLDER is defined in config.py
    image_path = os.path.join(IMAGE_FOLDER, image_filename)
    
    # 5. Load the image
    img = cv2.imread(image_path)
    
    # 6. Check, resize, and place the image on the canvas
    if img is not None:
        # Resize to the standard thumbnail size
        thumb = cv2.resize(img, THUMB_SIZE, interpolation=cv2.INTER_AREA)
        
        # --- Highlight the query image (which is always the 1st result) ---
        if i == 0:
            # Draw a 5px thick green border
            cv2.rectangle(thumb, (0, 0), (THUMB_SIZE[0]-1, THUMB_SIZE[1]-1), 
                          (0, 255, 0), 5) # (B, G, R) color

        # --- Calculate grid position ---
        row = i // VISUAL_GRID_COLS  # Integer division gives the row index (0, 1, 2, ...)
        col = i % VISUAL_GRID_COLS   # Modulo gives the column index (0, 1, 2, 3, 4)
        
        # Calculate the pixel coordinates to place this thumbnail
        y_start = row * THUMB_SIZE[1]
        y_end = (row + 1) * THUMB_SIZE[1]
        x_start = col * THUMB_SIZE[0]
        x_end = (col + 1) * THUMB_SIZE[0]
        
        # Place the thumbnail onto the main canvas
        canvas[y_start:y_end, x_start:x_end] = thumb
    else:
        print(f"Warning: Could not load image at {image_path}")

# --- 5. DISPLAY & SAVE THE FINAL RESULTS ---

# Create a readable title for the plot
plot_title = f"Query: {query_image_name}\n{EXPERIMENT_ID_STRING}"

# Convert the canvas from OpenCV's BGR format to Matplotlib's RGB format
canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

# Create a new figure to display the results
plt.figure(figsize=(12, 8))  # Set figure size
plt.imshow(canvas_rgb)       # Display the image
plt.title(plot_title)        # Set the title
plt.axis('off')              # Hide the x/y axes

# Save the figure as a PDF (if enabled in config.py)
if SAVE_VISUAL_SEARCH_PDF:
    # Create a descriptive filename
    pdf_filename = f"search__{query_image_name}__{EXPERIMENT_ID_STRING}.pdf"
    # RESULTS_PDF_FOLDER is defined in config.py
    pdf_save_path = os.path.join(RESULTS_PDF_FOLDER, pdf_filename)
    
    # Save the figure
    plt.savefig(pdf_save_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved search result to {pdf_save_path}")

# Show the plot window to the user
print("Displaying results. Close the plot window to exit.")
plt.show()

print("--- Visual Search Complete ---")