import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import *

# --- All config is now in config.py ---
print(f"--- Plotting Results ---")
print(f"Loading stats: {NPZ_FILE}")
print(f"Saving PR plot to: {PR_PLOT_FILE}")
print(f"Saving CM plot to: {CM_PLOT_FILE}")

# --- 2. Load the Pre-computed Data ---
try:
    data = np.load(NPZ_FILE, allow_pickle=True)
except FileNotFoundError:
    print(f"Error: Data file '{NPZ_FILE}' not found.")
    print("Please run 'compute_evaluation_stats.py' first.")
    exit()

all_precisions = data['all_precisions']
all_recalls = data['all_recalls']
confusion_matrix = data['confusion_matrix']
category_labels = data['category_labels']
k_results = data['k_results'].item() # .item() extracts the scalar value

# --- Load the summary stats ---
# Use .item() to extract scalar values if they exist
map_score = data.get('mean_average_precision', 0.0).item()
class_perf_sorted = data.get('class_performance_sorted', [])
experiment_id_loaded = f"{data['descriptor_mode'].item()}__{data['metric'].item()}"
if str(data.get('USE_PCA_DATA', False).item()) == 'True': # Check if USE_PCA_DATA was saved
    experiment_id_loaded = f"{data['descriptor_mode'].item()}_pca__{data['metric'].item()}"


# ---Load experiment info from the file ---
descriptor_mode_loaded = data['descriptor_mode'].item()
metric_loaded = data['metric'].item()
experiment_id_loaded = f"{descriptor_mode_loaded}__{metric_loaded}"

print("Data loaded.")

# --- 3. Plot Mean Precision-Recall Curve ---
print("Plotting PR Curve...")
mean_precision = np.mean(all_precisions, axis=0)
mean_recall = np.mean(all_recalls, axis=0)

plt.figure(figsize=(8, 6))
plt.plot(mean_recall, mean_precision, marker='o', linestyle='-')
plt.xlabel('Mean Recall')
plt.ylabel('Mean Precision')

plt.title(f'Mean PR Curve (MAP = {map_score:.4f})\nTop {k_results})\n{experiment_id_loaded}')

plt.grid(True)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.savefig(PR_PLOT_FILE, format='pdf', bbox_inches='tight')
print(f"Saved PR curve to {PR_PLOT_FILE}")
plt.show()

# --- 4. Plot Confusion Matrix ---
print("Plotting Confusion Matrix...")
# Normalize the matrix to show percentages (from 0 to 1)
cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, 
            annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=category_labels, yticklabels=category_labels)
plt.title(f'Confusion Matrix (Row-Normalized)\n{experiment_id_loaded}')
plt.ylabel('True Category')
plt.xlabel('Predicted Category (Top 1 Result)')

plt.savefig(CM_PLOT_FILE, format='pdf', bbox_inches='tight')
print(f"Saved Confusion Matrix to {CM_PLOT_FILE}")
plt.show()

# --- Print Class Performance Summary ---
if class_perf_sorted.any(): # Check if the array is not empty
    print("\n--- Per-Class Accuracy Summary ---")
    print("--- Best Performing Classes ---")
    for label, acc in class_perf_sorted[:5]:
        print(f"  {label} (Class {label}): {float(acc):.2%}")
    print("\n--- Worst Performing Classes ---")
    for label, acc in class_perf_sorted[-5:]:
        print(f"  {label} (Class {label}): {float(acc):.2%}")
    print("---------------------------------")

print("Done plotting.")