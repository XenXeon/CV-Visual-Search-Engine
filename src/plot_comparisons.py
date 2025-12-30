import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn is imported but not used, but kept as is
from config import RESULTS_PDF_FOLDER

print("Generating Comparison PR Plots")

## 1. LOAD THE SUMMARY CSV TO FIND DESCRIPTOR MODES
# We load the main summary file *first* just to get the list of
# unique descriptor types we need to generate plots for.
summary_csv_path = os.path.join(RESULTS_PDF_FOLDER, "results_summary.csv")
try:
    df = pd.read_csv(summary_csv_path)
except FileNotFoundError:
    print(f"Error: {summary_csv_path} not found.")
    print("Please run 'aggregate_results.py' first.")
    exit()

# Get a list of all unique descriptor modes (e.g., 'lbp', 'global', etc.)
descriptor_modes = df['descriptor'].unique()

## 2. FIND ALL THE .NPZ DATA FILES
# Now, get a list of all the individual result files. We will
# filter this list later inside the loop.
search_path = os.path.join(RESULTS_PDF_FOLDER, "stats__*.npz")
all_npz_files = glob.glob(search_path)

if not all_npz_files:
    print("Error: No 'stats__*.npz' files found. Cannot plot.")
    exit()

print(f"Found {len(descriptor_modes)} descriptor types to plot.")

# --- This list will store the *best* PR data from each mode ---
best_results_list = []

## 3. GENERATE COMPARISON PR CURVES
# This is the main loop. It runs once for each descriptor type
# (e.g., once for 'lbp', once for 'global', etc.).
for mode in descriptor_modes:
    
    # Create one new figure for this descriptor mode
    plt.figure(figsize=(11, 8))
    ax = plt.gca()  # Get current axes
    
    #  Filter all .npz files to find ones for this mode
    # This logic checks for files like:
    # 'stats__lbp_4x4...' AND 'stats__lbp_4x4_pca...'
    # by checking if the filename *starts with* 'stats__<mode_name>'
    files_for_this_mode = [
        f for f in all_npz_files 
        if os.path.basename(f).startswith(f"stats__{mode}")
    ]
    
    if not files_for_this_mode:
        print(f"\nWarning: No .npz files found for descriptor '{mode}'. Skipping.")
        plt.close()  # Close the unused figure
        continue

    print(f"\n Plotting {len(files_for_this_mode)} PR curves for '{mode}' ")
    
    # --- Start: Create a unique color for each line ---
    # Get N_lines and generate N unique colors from the 'nipy_spectral' colormap
    num_lines = len(files_for_this_mode)
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_lines))
    # Set the plot's color cycle to use these new unique colors
    ax.set_prop_cycle(color=colors)
    # --- End: Create a unique color for each line ---

    # --- Variables to track the best result for *this mode* ---
    best_map_for_this_mode = -1.0
    best_data_for_this_mode = None

    #  Inner loop: Plot all curves for this mode 
    # Now we loop through all the files we just found (e.g., all 'lbp' files)
    # and plot them on the *same* figure.
    for f_path in files_for_this_mode:
        try:
            data = np.load(f_path, allow_pickle=True)
            
            # --- Re-create the experiment label from the file data ---
            metric = str(data['metric'].item())
            use_pca = bool(data['USE_PCA_DATA'].item())
            
            # --- Start: Extract bins_info from filename ---
            basename_no_ext = os.path.splitext(os.path.basename(f_path))[0]
            prefix = f"stats__{mode}"
            pca_suffix = f"_pca_{use_pca}"
            metric_suffix = f"__{metric}"
            
            bins_info = basename_no_ext
            if bins_info.startswith(prefix):
                bins_info = bins_info[len(prefix):]
            if bins_info.endswith(pca_suffix):
                bins_info = bins_info[:-len(pca_suffix)]
            if bins_info.endswith(metric_suffix):
                bins_info = bins_info[:-len(metric_suffix)]
            bins_info = bins_info.strip('_')
            # --- End: Extract bins_info from filename ---
            
            # Create a clean label, e.g., "L1 (PCA)" or "L2 (Original)"
            pca_label = "PCA" if use_pca else "Original"
            
            # --- Start: Updated label creation ---
            if bins_info:
                label = f"{bins_info} - {metric} ({pca_label})"
            else:
                label = f"{metric} ({pca_label})"
            # --- End: Updated label creation ---
            
            # --- Calculate mean P and R and get mAP ---
            # data['all_precisions'] has shape (N_images, K_results)
            # np.mean(..., axis=0) averages across all images
            mean_precision = np.mean(data['all_precisions'], axis=0)
            mean_recall = np.mean(data['all_recalls'], axis=0)
            map_score = data['mean_average_precision'].item()

            # --- Plot the PR curve ---
            # We add the mAP score to the label for easy reference
            ax.plot(mean_recall, mean_precision, marker='.', linestyle='-', 
                    label=f"{label} - MAP={map_score:.3f}")
            print(f"    > Plotted {label}")

            # --- Check if this is the best result for this mode ---
            if map_score > best_map_for_this_mode:
                best_map_for_this_mode = map_score
                # Store all the data we need for the final plot
                best_data_for_this_mode = {
                    'label_part': label, # This is "q16 - L1 (PCA)"
                    'map': map_score,
                    'recall': mean_recall,
                    'precision': mean_precision,
                    'mode': mode # Store the mode name ("global")
                }

        except Exception as e:
            print(f"    > Warning: Could not load/plot {f_path}. Error: {e}")

    # --- After inner loop, store the best result for this mode ---
    if best_data_for_this_mode:
        best_results_list.append(best_data_for_this_mode)
    else:
        print(f"    > Warning: No valid data found for mode '{mode}' to add to best-of plot.")

    # --- 4. FINALIZE THE PLOT (for this mode) ---
    # This code runs after all curves for the current mode are plotted
    ax.set_title(f"Precision-Recall Comparison\nDescriptor: '{mode}'")
    ax.set_xlabel('Mean Recall')
    ax.set_ylabel('Mean Precision')
    ax.grid(True)  # Add a grid for readability
    
    # Set fixed axis limits to make plots comparable
    # You may need to adjust these based on your results
    ax.set_xlim([0.0, 0.225]) 
    ax.set_ylim([0.0, 0.65])
    
    # --- Sort the legend alphabetically ---
    # This makes the legend much cleaner and easier to read
    handles, labels = ax.get_legend_handles_labels()
    # Sort by the label text (x[1])
    sorted_legend = sorted(zip(handles, labels), key=lambda x: x[1])
    handles_sorted = [h for h, l in sorted_legend]
    labels_sorted = [l for h, l in sorted_legend]
    ax.legend(handles_sorted, labels_sorted, loc='upper right', fontsize='small')
    
    # --- Save the plot ---
    plot_filename = os.path.join(RESULTS_PDF_FOLDER, f"compare_PR__{mode}.pdf")
    plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
    print(f"    > Saved plot to {plot_filename}")
    
    # Show the plot to the user (optional, can be commented out)
    plt.show() 

print("\nComparison plotting complete")


## 5. GENERATE FINAL "BEST OF" PLOT
print("\nGenerating final 'Best of Each Descriptor' PR plot...")
if not best_results_list:
    print("Warning: No results were collected. Skipping final plot.")
else:
    plt.figure(figsize=(11, 8))
    ax_final = plt.gca()
    
    # --- Create a unique color for each line ---
    num_lines = len(best_results_list)
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_lines))
    ax_final.set_prop_cycle(color=colors)
    
    # --- Loop and plot the best curve from each mode ---
    for result in best_results_list:
        recall = result['recall']
        precision = result['precision']
        map_score = result['map']
        # Create a label like: "global (q16 - L1 (PCA)) - MAP=0.543"
        label = f"{result['mode']} ({result['label_part']}) - MAP={map_score:.3f}"
        
        ax_final.plot(recall, precision, marker='.', linestyle='-', label=label)
        print(f"    > Plotting best for '{result['mode']}'")

    # --- Finalize the plot ---
    ax_final.set_title("Precision-Recall Comparison (Best of Each Descriptor)")
    ax_final.set_xlabel('Mean Recall')
    ax_final.set_ylabel('Mean Precision')
    ax_final.grid(True)
    
    # Use the same limits for consistency
    ax_final.set_xlim([0.0, 0.225]) 
    ax_final.set_ylim([0.0, 0.65])
    
    # --- Sort the legend alphabetically ---
    handles, labels = ax_final.get_legend_handles_labels()
    sorted_legend = sorted(zip(handles, labels), key=lambda x: x[1])
    handles_sorted = [h for h, l in sorted_legend]
    labels_sorted = [l for h, l in sorted_legend]
    ax_final.legend(handles_sorted, labels_sorted, loc='upper right', fontsize='small')
    
    # --- Save the plot ---
    plot_filename = os.path.join(RESULTS_PDF_FOLDER, "compare_PR__BEST_OF_ALL.pdf")
    plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
    print(f"    > Saved final comparison plot to {plot_filename}")
    
    plt.show()

print("\nAll plotting complete.")