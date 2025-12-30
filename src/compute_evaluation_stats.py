"""
COMPUTE EVALUATION STATISTICS

This is the main evaluation script for a single experiment.

For each of the N images in the dataset, it does the following:
1.  Treats the image as a query.
2.  Computes its distance to all N images.
3.  Sorts the results.
4.  Calculates Precision, Recall, and Average Precision (AP) for this query.
5.  Updates a confusion matrix based on the top-ranked result.

After all N queries are processed, it calculates:
-   Mean Average Precision (mAP)
-   Mean Precision and Recall curves
-   Per-class accuracy

It then saves all of these statistics to a single compressed
'.npz' file in the 'RESULTS_PDF_FOLDER' for later aggregation.

"""

import os
import numpy as np
import scipy.io as sio
import collections
from cvpr_compare import cvpr_compare  # Imports your distance functions (L1, L2, etc.)
from config import * # Imports all global settings (paths, metric, etc.)

def main():
    
    # --- All config is now in config.py ---
    print(f"--- Starting Evaluation ---")
    print(f"Experiment ID: {EXPERIMENT_ID_STRING}")

    ## 1. HELPER FUNCTION
    def get_category_from_path(filepath):
        """
        Helper function to extract the category label from a file path.
        Example: '.../1_1_s.mat' -> '1'
        """
        base = os.path.basename(filepath)
        name = os.path.splitext(base)[0]
        category = name.split('_')[0]
        return category

    ## 2. LOAD ALL DESCRIPTORS
    # This block determines *which* descriptor folder to read from
    # based on the settings in config.py
    
    eigenvalues = None  # Will only be set if METRIC_TO_TEST == 'Mahalanobis'
    
    if USE_PCA_DATA:
        print(f"Loading PCA descriptors from: {PCA_DESCRIPTOR_PATH}")
        current_descriptor_path = PCA_DESCRIPTOR_PATH
        
        # We only need the eigenvalues if we are using the Mahalanobis
        # distance, which is defined by the PCA space.
        if METRIC_TO_TEST == 'Mahalanobis':
            try:
                print(f"Loading eigenvalues from: {PCA_EIGENVALUES_FILE}")
                eigenvalues = np.load(PCA_EIGENVALUES_FILE)
            except FileNotFoundError:
                print(f"Error: Eigenvalues file not found at {PCA_EIGENVALUES_FILE}")
                print("Please run 'compute_pca.py' first.")
                exit()
    else:
        print(f"Loading original descriptors from: {DESCRIPTOR_PATH}")
        current_descriptor_path = DESCRIPTOR_PATH

    print("Loading all descriptors into memory...")
    ALLFEAT = []   # List to hold all feature vectors
    ALLFILES = []  # List to hold all corresponding file paths
    
    for filename in os.listdir(current_descriptor_path):
        if filename.endswith('.mat'):
            file_path = os.path.join(current_descriptor_path, filename)
            img_data = sio.loadmat(file_path)
            ALLFILES.append(file_path)
            ALLFEAT.append(img_data['F'][0])  # Add the 1D feature vector

    # Convert lists to a single N x D NumPy array
    ALLFEAT = np.array(ALLFEAT)
    NIMG = ALLFEAT.shape[0]  # Total number of images
    print(f"Loaded {NIMG} descriptors.")

    if NIMG == 0:
        print(f"Error: No .mat files found in {current_descriptor_path}")
        exit()

    ## 3. LOAD EXTRACTION TIME
    # This reads the 'extraction_time.txt' file created by
    # 'cvpr_computedescriptors.py' so we can include it in our stats.
    extraction_time = 0.0
    # TIME_FILE is built from DESCRIPTOR_SUBFOLDER, not PCA_DESCRIPTOR_PATH,
    # so this correctly gets the *original* descriptor extraction time.
    try:
        with open(TIME_FILE, 'r') as f:
            line = f.readline() # Reads "total_time: 123.45"
            extraction_time = float(line.split(':')[1].strip())
        print(f"Loaded extraction time: {extraction_time:.2f}s")
    except FileNotFoundError:
        print(f"Warning: Could not find {TIME_FILE}. Setting extraction time to 0.")
    except Exception as e:
        print(f"Warning: Could not read time file. Error: {e}")

    ## 4. PREPARE GROUND TRUTH
    # This block builds the "answer key" for the evaluation.
    # It maps every image to its correct category.
    print("Preparing ground truth...")
    
    # Get the category (e.g., '1') for each file path
    image_categories = [get_category_from_path(f) for f in ALLFILES]
    # Get a sorted, unique list of all category labels (e.g., ['1', '2', ..., '20'])
    category_labels = sorted(list(set(image_categories)), key=int)
    num_categories = len(category_labels)
    
    # Create a lookup map: {'1': 0, '2': 1, ...}
    cat_to_index = {label: i for i, label in enumerate(category_labels)}
    # Create a list of the category *index* for every image
    image_cat_indices = [cat_to_index[cat] for cat in image_categories]
    # Count how many images are in each category
    # e.g., {0: 30, 1: 25, ...}
    category_counts = collections.Counter(image_cat_indices)

    ## 5. MAIN EVALUATION LOOP (O(N^2) COMPLEXITY)
    # These lists will store the results for all N queries
    all_precisions = []          # List of N lists, each of size K
    all_recalls = []             # List of N lists, each of size K
    all_average_precisions = []  # List of N single AP scores
    
    # A 20x20 matrix to store confusion
    confusion_matrix = np.zeros((num_categories, num_categories), dtype=int)

    print(f"Starting evaluation loop for {NIMG} queries using {METRIC_TO_TEST} distance...")
    
    # --- THE QUERY LOOP (Outer loop, runs N times) ---
    for query_idx in range(NIMG):
        if (query_idx + 1) % 50 == 0:
            print(f"   Processing query {query_idx + 1}/{NIMG}...")

        # Get the query's data
        query_vector = ALLFEAT[query_idx]
        query_cat_index = image_cat_indices[query_idx]
        
        # Get the total number of "correct" images in the dataset for this query
        # We subtract 1 (the query itself)
        total_relevant = category_counts[query_cat_index]
        total_relevant_docs = max(1, total_relevant - 1) # Avoid division by zero

        # --- THE DISTANCE LOOP (Inner loop, runs N times) ---
        # This is the N*N part. We compare one query to all N images.
        dst = []
        for i in range(NIMG):
            candidate_vector = ALLFEAT[i]
            distance = cvpr_compare(query_vector, candidate_vector, 
                                    metric=METRIC_TO_TEST, 
                                    eigenvalues=eigenvalues) # eigenvalues is None if not Mahalanobis
            
            dst.append((distance, i)) # Store (distance, index)

        # Sort the results by distance (smallest distance first)
        dst.sort(key=lambda x: x[0])

        ## 5a. Update Confusion Matrix
        # The "top-1" prediction is the *second* item in the list
        # (the first is the query itself with distance 0.0)
        predicted_img_idx = dst[1][1]
        predicted_cat_index = image_cat_indices[predicted_img_idx]
        
        # Increment the cell (true_label, predicted_label)
        # Rows = True Class, Columns = Predicted Class
        confusion_matrix[query_cat_index, predicted_cat_index] += 1

        ## 5b. Calculate P/R and Average Precision (AP) for this query
        precisions_at_k = []
        recalls_at_k = []
        relevant_found = 0
        average_precision_sum = 0.0

        # Loop from k=1 up to K_RESULTS (e.g., k=1, 2, ..., 15)
        # We skip dst[0] (the query) so k=1 corresponds to dst[1]
        for k in range(1, K_RESULTS + 1):
            # Get the k-th result's info
            result_img_idx = dst[k][1]
            result_cat_index = image_cat_indices[result_img_idx]
            
            is_relevant = 0 # 0 = not relevant, 1 = relevant
            if result_cat_index == query_cat_index:
                relevant_found += 1
                is_relevant = 1

            # P@k = (num_relevant_found_so_far) / k
            precision = relevant_found / k
            # R@k = (num_relevant_found_so_far) / (total_possible_relevant)
            recall = relevant_found / total_relevant_docs
            
            # Add to AP sum: (P(k) * rel(k))
            # This only adds the precision if the k-th item was relevant.
            average_precision_sum += (precision * is_relevant)

            precisions_at_k.append(precision)
            recalls_at_k.append(recall)
        
        # --- Finalize AP for this query ---
        # AP = (Sum of P(k)*rel(k)) / (Total Relevant Docs)
        if total_relevant_docs > 0:
            average_precision = average_precision_sum / total_relevant_docs
        else:
            average_precision = 0.0 # No relevant docs, so AP is 0
        
        # Store the results for *this query*
        all_average_precisions.append(average_precision)
        all_precisions.append(precisions_at_k)
        all_recalls.append(recalls_at_k)
    
    # --- End of Main Loop ---
    print("Evaluation complete.")

    ## 6. CALCULATE SUMMARY STATISTICS
    # Now we average the results from all N queries
    
    # 1. Mean Average Precision (MAP)
    # The average of all N query-specific AP scores.
    mean_average_precision = np.mean(all_average_precisions)

    # 2. Mean Precision and Recall @ K
    # We get the last P/R value (at K=15) from each query's list
    # and then average those N values.
    mean_precision_at_k = np.mean([p_list[-1] for p_list in all_precisions])
    mean_recall_at_k = np.mean([r_list[-1] for r_list in all_recalls])

    # 3. Per-Class Accuracy (from the Confusion Matrix)
    # Get the total number of items per class
    class_totals = [category_counts[cat_to_index[label]] for label in category_labels]
    # Get the number of correct predictions (the diagonal of the CM)
    class_correct = confusion_matrix.diagonal()
    # Calculate accuracy
    per_class_accuracy = class_correct / class_totals

    # 4. Sorted Class List (for display)
    class_perf = sorted(zip(category_labels, per_class_accuracy), 
                        key=lambda x: x[1], 
                        reverse=True)

    ## 7. PRINT SUMMARY TO CONSOLE
    print("\nEvaluation Summary")
    print(f"\nExperiment: {EXPERIMENT_ID_STRING}")
    print(f"Mean Average Precision (MAP): {mean_average_precision:.4f}")
    print(f"Mean Precision @ K={K_RESULTS}:   {mean_precision_at_k:.4f}")
    print(f"Mean Recall @ K={K_RESULTS}:     {mean_recall_at_k:.4f}")
    print("\n--- Per-Class Accuracy (Top 5) ---")
    for label, acc in class_perf[:5]:
        print(f"   {label} (Class {label}): {acc:.2%}")
    print("\n--- Per-Class Accuracy (Bottom 5) ---")
    for label, acc in class_perf[-5:]:
        print(f"   {label} (Class {label}): {acc:.2%}")
    print("---------------------------------")

    ## 8. SAVE AGGREGATED RESULTS
    # This saves all the raw data from this experiment into a
    # single compressed .npz file. This file is what
    # 'aggregate_results.py' and 'plot_comparison_pr.py' will load.
    print(f"Saving aggregated data to {NPZ_FILE}...")
    np.savez(NPZ_FILE, 
             # --- Raw P/R data ---
             all_precisions=np.array(all_precisions),
             all_recalls=np.array(all_recalls),
             # --- Confusion Matrix data ---
             confusion_matrix=confusion_matrix,
             per_class_accuracy=per_class_accuracy,
             class_performance_sorted=np.array(class_perf),
             category_labels=np.array(category_labels),
             # --- Key metrics ---
             mean_average_precision=mean_average_precision,
             # --- Experiment config (for logging) ---
             k_results=K_RESULTS,
             extraction_time_seconds=extraction_time,
             descriptor_mode=DESCRIPTOR_MODE,
             metric=METRIC_TO_TEST,
             USE_PCA_DATA=USE_PCA_DATA,
             )
    print("Done.")

## 9. SCRIPT ENTRY POINT
if __name__ == "__main__":
    main()