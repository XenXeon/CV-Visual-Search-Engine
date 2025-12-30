import os
import numpy as np
import pandas as pd
import glob  # Used to find files matching a pattern

# Import the results folder path from the central config file
from config import RESULTS_PDF_FOLDER

print("Aggregating All Experiment Results...")

## 1. FIND ALL NPZ RESULT FILES
# Construct a search pattern to find all files that
# start with 'stats__' and end with '.npz'
search_path = os.path.join(RESULTS_PDF_FOLDER, "stats__*.npz")
all_npz_files = glob.glob(search_path)

# Check if any files were found
if not all_npz_files:
    print(f"Error: No 'stats_*.npz' files found in {RESULTS_PDF_FOLDER}")
    print("Please run 'run_all_experiments.py' or 'compute_evaluation_stats.py' first.")
    exit()

print(f"Found {len(all_npz_files)} result files. Loading...")

all_results = []  # A list to hold one dictionary per experiment

## 2. LOAD DATA FROM EACH FILE
# Loop through each .npz file path found
for f_path in all_npz_files:
    try:
        # Load the .npz file. allow_pickle=True is needed to load
        # object arrays, which is how NumPy saves strings.
        data = np.load(f_path, allow_pickle=True)
        
        # --- Extract Key Metrics ---
        
        # .item() extracts the single scalar value from a 0-d array
        map_score = data['mean_average_precision'].item()
        
        # Load the arrays of precision and recall values
        # Shape is (N_images, K_results)
        all_p = data['all_precisions']
        all_r = data['all_recalls']
        
        # 1. np.mean(..., axis=0) averages across all images,
        #    giving an array of shape (K_results,)
        # 2. [-1] gets the *last* value from that array, which is
        #    the mean precision/recall at the maximum K.
        mean_p_at_k = np.mean(all_p, axis=0)[-1]
        mean_r_at_k = np.mean(all_r, axis=0)[-1]
        
        # Get the K value (e.g., 15)
        k_val = data['k_results'].item()
        
        # --- Re-build Experiment ID ---
        # We load the config settings *from the file* to ensure
        # we are logging the correct parameters for this result.
        desc_mode = str(data['descriptor_mode'].item())
        metric = str(data['metric'].item())
        use_pca = bool(data['USE_PCA_DATA'].item())
        
        # --- Store in a Dictionary ---
        # This structure makes it easy to create a DataFrame later.
        result = {
            "map": map_score,  # Mean Average Precision
            "descriptor": desc_mode,
            "metric": metric,
            "use_pca": use_pca,
            # Create dynamic column names like 'precision_at_k15'
            f"precision_at_k{k_val}": mean_p_at_k,
            f"recall_at_k{k_val}": mean_r_at_k,
            "source_file": os.path.basename(f_path)
        }
        all_results.append(result)
        
    except Exception as e:
        # Catch any errors during loading (e.g., corrupted file)
        print(f"Warning: Could not load or parse {f_path}. Error: {e}")

# Check if we successfully loaded any results
if not all_results:
    print("Error: No valid result files could be loaded.")
    exit()

## 3. CONVERT TO PANDAS DATAFRAME
# This converts the list of dictionaries into a table-like structure.
df = pd.DataFrame(all_results)

## 4. SORT DATAFRAME BY MAP SCORE
# Sort the entire table by the "map" column, from highest to lowest.
df = df.sort_values(by="map", ascending=False)

## 5. SAVE TO CSV
# Define the output file path and save the sorted DataFrame as a CSV.
output_csv = os.path.join(RESULTS_PDF_FOLDER, "results_summary.csv")
df.to_csv(output_csv, index=False)

## 6. PRINT SUMMARY TO CONSOLE
print("\nResults Summary (Sorted by Best MAP)")
# Set display options to ensure all columns are visible in the terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# .to_string(index=False) prints the DataFrame nicely without the row numbers
print(df.to_string(index=False))

print(f"\nSuccessfully saved summary to {output_csv}")