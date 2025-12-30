"""
EXPERIMENT BATCH RUNNER (run_all_experiments.py)

This is the main "controller" script for running a large batch of
experiments. It does the following:

1.  Defines Parameter Spaces: Specifies all descriptor types,
    metrics, PCA options, and parameters (like 'Q' or 'grid_size')
    you want to test.
2.  *Generates Experiments: Automatically creates a "to-do list"
    of all valid combinations of these parameters.
3.  Runs the Pipeline: For each experiment in the list, it runs
    the full 3-step pipeline:
        a. Compute Descriptors (if not already done)
        b. Compute PCA (if required and not already done)
        c. Compute Evaluation Stats
4.  Skips Redundant Work: It's smart enough to not re-compute
    descriptors or PCA models if they've already been generated
    by a previous experiment in the batch.
"""

import importlib  # Used to 'reload' modules
import time
import os

# Import your global configuration and callable scripts
import config
import cvpr_computedescriptors
import compute_pca
import compute_evaluation_stats

# 1. DEFINE YOUR PARAMETER SPACES
# This is the "control panel" for the experiments.
# Add any parameters you want to test in these lists.

# Core Parameters 
DESCRIPTOR_MODES = [
    'global'
]

PCA_OPTIONS = [False, True]
METRICS = ['L1', 'L2', 'Mahalanobis'] 

# Descriptor-Specific Parameters 
GLOBAL_Q_VALUES = [4, 8]      # e.g., Test q=4 and q=8
GRID_SIZES = [(4, 4), (8, 8)] # e.g., Test 4x4 and 8x8 grids
ANGULAR_BIN_VALUES = [8, 10]  # e.g., Test 8 and 10 bins

# 2. AUTOMATICALLY GENERATE EXPERIMENT LIST

experiments = []  # This list will hold all experiments
print("Generating all experiment combinations...")

# Loop through the "Cartesian product" of all core parameters
for mode in DESCRIPTOR_MODES:
    for use_pca in PCA_OPTIONS:
        for metric in METRICS:

            # Logic to skip invalid or redundant combinations

            # Rule 1: Mahalanobis distance REQUIRES PCA.
            # Skip any experiment that tries to use Mahalanobis *without* PCA.
            if metric == 'Mahalanobis' and not use_pca:
                continue  # Skip this combination

            # Handle mode-specific parameters
            # Now we create the specific job dictionaries.

            if mode == 'global':
                # 'global' mode only cares about GLOBAL_Q_VALUES
                for q in GLOBAL_Q_VALUES:
                    exp = {
                        'DESCRIPTOR_MODE': mode,
                        'GLOBAL_Q': q,
                        'METRIC_TO_TEST': metric,
                        'USE_PCA_DATA': use_pca
                    }
                    experiments.append(exp)

            elif mode == 'bovw':
                # 'bovw' has no special parameters in this setup
                exp = {
                    'DESCRIPTOR_MODE': mode,
                    'METRIC_TO_TEST': metric,
                    'USE_PCA_DATA': use_pca
                }
                experiments.append(exp)

            elif mode in ['grid_col', 'grid_tex', 'grid_col_tex', 'lbp', 'color_and_lbp']:
                # These modes all use GRID_SIZES.
                # Some also use ANGULAR_BIN_VALUES.
                for (rows, cols) in GRID_SIZES:
                    # For grid_color, the 'bins' value is irrelevant but harmless
                    # For lbp/texture, it is relevant.
                    for bins in ANGULAR_BIN_VALUES:
                        exp = {
                            'DESCRIPTOR_MODE': mode,
                            'GRID_ROWS': rows,
                            'GRID_COLS': cols,
                            'ANGULAR_BINS': bins,
                            'METRIC_TO_TEST': metric,
                            'USE_PCA_DATA': use_pca
                        }
                        experiments.append(exp)
            
            # (Add other 'elif' blocks here if you create new modes)

# These sets will act as a "cache" to track what we've already computed.
# This prevents re-computing LBP descriptors 10 times
# if we're just testing 10 different metrics.
computed_descriptors = set()
computed_pca = set()

# 3. PREVIEW GENERATED JOBS
print(f"Generated {len(experiments)} total experiments to run.")
for i, exp in enumerate(experiments):
    # We must call update_config briefly just to get the *name*
    # (EXPERIMENT_ID_STRING) that this experiment *will* have.
    config.update_config(**exp)
    print(f"  Job {i+1}: {config.EXPERIMENT_ID_STRING}")

# 4. MAIN EXPERIMENT RUNNER LOOP
print(f"\n Starting Experiment Batch")
total_start_time = time.time()

for i, exp_settings in enumerate(experiments):
    exp_start_time = time.time()
    
    # 4.1: Update Configuration
    # This call tells config.py to update all its global variables
    # (like DESCRIPTOR_PATH, NPZ_FILE, etc.) for THIS experiment.
    print(f"\nRunning Experiment {i+1}/{len(experiments)}")
    config.update_config(**exp_settings)
    
    # 4.2: Reload Modules 
    # CRITICAL: This forces Python to re-import the script modules.
    # When they are re-imported, they will load the *new* variables
    # that were just set inside the config.py file.
    importlib.reload(cvpr_computedescriptors)
    importlib.reload(compute_pca)
    importlib.reload(compute_evaluation_stats)

    # Get the unique name for the base descriptor (e.g., "lbp_4x4_bins10")
    # This is *before* PCA is applied.
    base_descriptor_id = config.DESCRIPTOR_SUBFOLDER

    # 4.3: Step 1: Compute Descriptors (if needed)
    if base_descriptor_id not in computed_descriptors:
        print(f"[{config.DESCRIPTOR_SUBFOLDER}] Computing base descriptors...")
        # Check if descriptor files already exist
        if os.path.exists(config.DESCRIPTOR_PATH) and len(os.listdir(config.DESCRIPTOR_PATH)) > 0:
            print(f"[{config.DESCRIPTOR_SUBFOLDER}] Files already exist. Skipping computation.")
        else:
            cvpr_computedescriptors.main()
        # Add to cache so we don't run it again
        computed_descriptors.add(base_descriptor_id) 
    else:
        print(f"[{config.DESCRIPTOR_SUBFOLDER}] Base descriptors already cached. Skipping.")

    # 4.4: Step 2: Compute PCA (if needed)
    pca_id = base_descriptor_id + "_pca" # e.g., "lbp_4x4_bins10_pca"
    
    # Check if this experiment *requires* PCA
    if config.USE_PCA_DATA:
        # If it does, check if we've *already computed* it
        if pca_id not in computed_pca:
            print(f"[{pca_id}] Computing PCA...")
            # Check if PCA files already exist
            if os.path.exists(config.PCA_DESCRIPTOR_PATH) and len(os.listdir(config.PCA_DESCRIPTOR_PATH)) > 0:
                 print(f"[{pca_id}] PCA files already exist. Skipping computation.")
            else:
                compute_pca.main()
            # Add to cache so we don't run it again
            computed_pca.add(pca_id)
        else:
            print(f"[{pca_id}] PCA already cached. Skipping.")
    
    # 4.5: Step 3: Run Evaluation
    # We *always* run the evaluation step, as this is unique
    # for every metric and PCA/non-PCA combination.
    print(f"[{config.EXPERIMENT_ID_STRING}] Running evaluation...")
    # Check if final results .npz file already exists
    if os.path.exists(config.NPZ_FILE):
        print(f"[{config.EXPERIMENT_ID_STRING}] Stats file already exists. Skipping evaluation.")
    else:
        compute_evaluation_stats.main()
    
    exp_end_time = time.time()
    print(f"Experiment {i+1} Finished ({config.EXPERIMENT_ID_STRING})")
    print(f"Time taken: {exp_end_time - exp_start_time:.2f} seconds")

total_end_time = time.time()
print("\nExperiment Batch Complete")
print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
print("Run 'aggregate_results.py' to summarize.")