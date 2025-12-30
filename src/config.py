"""
===================================================================
--- CONFIGURATION FILE (config.py) ---
===================================================================
This file defines all static and dynamic settings for the project.

- Default Settings: Initial values for experiments.
- Static Paths: Base folders that rarely change.
- Dynamic Paths: File/folder paths that are derived from the
  default settings (e.g., the path to descriptors for 'lbp').
- update_config(): A crucial function that recalculates all
  dynamic paths whenever a setting is changed.

This file is imported by all other scripts (e.g., cvpr_visualsearch.py,
compute_evaluation.py) to ensure they all use the same set of
parameters for a given experiment.
"""

import os
import sys

# 1. DEFAULT EXPERIMENT SETTINGS

# These variables control *what* experiment to run.
# They are intended to be overwritten by update_config() at the
# start of an experiment script.

# 'global', 'grid_col', 'grid_tex', 'grid_col_tex', 'lbp', 'color_and_lbp', 'bovw'.
DESCRIPTOR_MODE = 'lbp'
# 'L1', 'L2', 'L3', 'Mahalanobis'
METRIC_TO_TEST = 'L2'
# If True, use descriptors from the '_pca' folder.
# If False, use original descriptors.
USE_PCA_DATA = True

# 2. DESCRIPTOR PARAMETERS 

# These variables control *how* descriptors are generated.
# They are used by 'compute_descriptors.py'.

GLOBAL_Q = 4       # Bins per channel for 'global' 3D color histogram
GRID_ROWS = 4      # Number of rows for 'grid' descriptors
GRID_COLS = 4      # Number of columns for 'grid' descriptors
ANGULAR_BINS = 10  # Number of bins for 'grid_texture' or 'lbp'

# 3. EVALUATION & VISUALIZATION SETTINGS 

# These control the behavior of 'visual_search.py'
# and 'compute_evaluation.py'.

K_RESULTS = 15           # K for Precision@K (how many results to check)
SHOW = 15                # Total number of images to show in the visual search grid
VISUAL_GRID_ROWS = 3     # Rows in the visual search output grid
VISUAL_GRID_COLS = 5     # Columns in the visual search output grid
THUMB_SIZE = (200, 150)  # (width, height) to resize result images
SAVE_VISUAL_SEARCH_PDF = True # Save the results grid to a PDF?

# Optional Plot Toggles
SHOW_3D_HISTOGRAM = False      # Show query's 3D color plot (global only)
SHOW_1D_DESCRIPTOR_PLOT = False  # Show query's 1D feature vector
SHOW_PCA_SCATTER_PLOT = False    # Show 2D/3D scatter of all PCA features

# 4. STATIC PATHS (Set these once) 

# These are the main folder paths for the project.
# All other paths are built from these.

# Get the absolute path of the directory containing this config file (src/)
# and go one level up to get the Project Root (CV-Visual-Search-Engine/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define the dataset root inside the 'data' folder
# Structure: data/MSRC_ObjCategImageDatabase_v2/Images
DATASET_NAME = 'MSRC_ObjCategImageDatabase_v2'
DATASET_FOLDER = os.path.join(PROJECT_ROOT, 'data', DATASET_NAME)

# Input folder (where original .bmp images are)
IMAGE_FOLDER = os.path.join(DATASET_FOLDER, 'Images')

# Output folder (where .mat descriptors will be saved)
DESCRIPTOR_FOLDER = os.path.join(DATASET_FOLDER, 'descriptors')

# Output folder (for visual reports like PR curves)
RESULTS_PDF_FOLDER = os.path.join(PROJECT_ROOT, 'report_visuals')

# Output folder (base for all .mat descriptors)
os.makedirs(DESCRIPTOR_FOLDER, exist_ok=True)
os.makedirs(RESULTS_PDF_FOLDER, exist_ok=True)

# 5. DYNAMIC / DERIVED PATHS 

# These are initialized to None and will be *calculated* by
# update_config() based on the settings in sections 1 & 2.
# DO NOT SET THESE MANUALLY.

DESCRIPTOR_SUBFOLDER = None  # e.g., "lbp_4x4_bins10"
DESCRIPTOR_PATH = None       # Full path to the descriptor subfolder
TIME_FILE = None             # Path to 'extraction_time.txt'
PCA_DESCRIPTOR_PATH = None   # Path to PCA-transformed descriptors
PCA_MODEL_FILE = None        # Path to 'pca_model.joblib'
SCALER_MODEL_FILE = None     # Path to 'scaler_model.joblib'
PCA_EIGENVALUES_FILE = None  # Path to 'pca_eigenvalues.npy'
EXPERIMENT_ID_STRING = None  # A unique name for this experiment run
NPZ_FILE = None              # Output .npz file for evaluation stats
PR_PLOT_FILE = None          # Output PDF file for PR curve
CM_PLOT_FILE = None          # Output PDF file for confusion matrix

# 6. CONFIGURATION UPDATE FUNCTION 

def update_config(**kwargs):
    """
    Updates the global config variables based on keyword arguments
    and then recalculates all derived paths.

    This is the core function of the config file. It is called by
    other scripts to set up a specific experiment.

    Example:
        import config
        config.update_config(DESCRIPTOR_MODE='global', METRIC_TO_TEST='L2')
        # Now all global path variables in config are set for this experiment
    """
    # Get the dictionary of all global variables defined in this file
    globs = globals()

    # --- A. Update global settings from kwargs ---
    # Loop through all key/value pairs provided to the function
    for key, value in kwargs.items():
        if key in globs:
            # If the key exists as a global variable, update it
            globs[key] = value
        else:
            # Warn if a non-existent config key is provided
            print(f"Config Warning: '{key}' not a valid config setting.")

    # Re-calculate all derived paths
    # Get the (potentially new) settings to build paths
    mode = globs['DESCRIPTOR_MODE']
    g_q = globs['GLOBAL_Q']
    g_r = globs['GRID_ROWS']
    g_c = globs['GRID_COLS']
    a_b = globs['ANGULAR_BINS']
    metric = globs['METRIC_TO_TEST']
    use_pca = globs['USE_PCA_DATA']

    # 1. Create the unique subfolder name based on descriptor mode and params
    if mode == 'global':
        subfolder = f"global_q{g_q}"
    elif mode == 'grid_color':
        subfolder = f"grid_color_{g_r}x{g_c}"
    elif mode == 'grid_texture':
        subfolder = f"grid_texture_{g_r}x{g_c}_bins{a_b}"
    elif mode == 'lbp':
        subfolder = f"lbp{g_r}x{g_c}_bins{a_b}"
    elif mode == 'color_and_lbp':
        subfolder = f"color_and_lbp{g_r}x{g_c}_bins{a_b}"
    elif mode == 'grid_col_tex':
        subfolder = f"grid_col_tex_{g_r}x{g_c}_bins{a_b}"
    elif mode == 'bovw':
        subfolder = 'bovw_k500'  # Example: BoVW folder name
    else:
        raise ValueError(f"Unknown DESCRIPTOR_MODE: {mode}")

    # Set the global DESCRIPTOR_SUBFOLDER
    globs['DESCRIPTOR_SUBFOLDER'] = subfolder

    # Set the full path to the original descriptors and time file
    desc_path = os.path.join(globs['DESCRIPTOR_FOLDER'], subfolder)
    globs['DESCRIPTOR_PATH'] = desc_path
    globs['TIME_FILE'] = os.path.join(desc_path, 'extraction_time.txt')

    # Set the paths for PCA models, eigenvalues, and PCA-descriptors
    # Note: PCA models are saved in the *original* descriptor folder
    globs['PCA_MODEL_FILE'] = os.path.join(desc_path, "pca_model.joblib")
    globs['SCALER_MODEL_FILE'] = os.path.join(desc_path, "scaler_model.joblib")
    globs['PCA_EIGENVALUES_FILE'] = os.path.join(desc_path, "pca_eigenvalues.npy")
    # PCA-transformed descriptors get their own folder
    globs['PCA_DESCRIPTOR_PATH'] = os.path.join(globs['DESCRIPTOR_FOLDER'], subfolder + "_pca")

    # Create the unique Experiment ID string
    # This is used for naming output files (plots, stats)
    id_str = f"{subfolder}__{metric}"
    if use_pca:
        # Append '_pca' if we're using PCA data
        id_str = f"{subfolder}_pca__{metric}"
    globs['EXPERIMENT_ID_STRING'] = id_str

    # 6. Set the final output file paths for evaluation results
    globs['NPZ_FILE'] = os.path.join(globs['RESULTS_PDF_FOLDER'], f"stats__{id_str}.npz")
    globs['PR_PLOT_FILE'] = os.path.join(globs['RESULTS_PDF_FOLDER'], f"pr_curve__{id_str}.pdf")
    globs['CM_PLOT_FILE'] = os.path.join(globs['RESULTS_PDF_FOLDER'], f"cm_plot__{id_str}.pdf")


# INITIALIZATION

# Run the function once on import.
# This ensures that all derived paths (DESCRIPTOR_PATH, NPZ_FILE, etc.)
# are populated based on the *default* settings listed at the top
# of this file.
update_config()