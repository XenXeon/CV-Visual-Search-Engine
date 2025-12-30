import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import * # Make sure this points to DESCRIPTOR_PATH

def plot_pca_elbow(descriptors):
    """
    Scales data and plots the Scree Plot (individual variance) 
    to find the "elbow".
    
    It also prints the number of components needed to
    explain 90% and 95% of the variance.
    
    Args:
        descriptors (np.array): Your raw feature descriptors.
    """
    
    # 1. MUST scale data first
    print("Scaling data for analysis...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(descriptors)
    
    # 2. Fit PCA with all components to analyze them
    print("Fitting PCA (this may take a moment)...")
    # n_components=None keeps all components
    pca = PCA(n_components=None).fit(data_scaled)
    
    # 3. Get the variance data
    individual_variance = pca.explained_variance_ratio_
    # We still calculate cumulative variance for the printout, even if not plotted
    cumulative_variance = np.cumsum(individual_variance)
    
    # 4. Create the plots
    print("Plotting results...")
    # --- Create a single plot ---
    plt.figure(figsize=(12, 6))
    ax = plt.gca() # Get current axes

    # --- PLOT 1: The "Scree Plot" (for the Elbow) ---
    ax.plot(range(1, len(individual_variance) + 1), individual_variance, 
             'o-', label='Individual Component Variance')
    ax.set_title('Scree Plot (Look for the "Elbow" Here)')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_xlabel('Number of Components') # Added x-label
    ax.legend()
    ax.grid(True)

    # --- PLOT 2: Cumulative Variance (Removed as requested) ---

    # Limit x-axis to zoom in on the important part (e.g., first 100 components)
    # Adjust this 'plot_limit' if your elbow is further out
    plot_limit = min(len(individual_variance), 100)
    ax.set_xlim(0, plot_limit)
    
    plt.tight_layout()
    plt.show()

    # Print recommendations
    try:
        components_for_90 = np.argmax(cumulative_variance >= 0.90) + 1
        components_for_95 = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"\n--- Analysis ---")
        print(f"To keep 90% variance, you need: {components_for_90} components")
        print(f"To keep 95% variance, you need: {components_for_95} components")
        print("Now, look at the 'Scree Plot' to find the elbow.")
    except ValueError:
        print("Could not calculate variance thresholds.")

# --- Main part of the script ---

# 1. Load all descriptors (same as your main script)
print(f"Loading descriptors from: {DESCRIPTOR_PATH}")
ALLFEAT = []
for filename in os.listdir(DESCRIPTOR_PATH):
    if filename.endswith('.mat'):
        file_path = os.path.join(DESCRIPTOR_PATH, filename)
        try:
            img_data = sio.loadmat(file_path)
            ALLFEAT.append(img_data['F'][0])
        except Exception as e:
            print(f"Could not load {filename}: {e}")

if not ALLFEAT:
    print("No descriptors found. Check DESCRIPTOR_PATH in config.py")
else:
    ALLFEAT = np.array(ALLFEAT)
    print(f"Loaded {ALLFEAT.shape[0]} descriptors.")
    print(f"Original dimensions: {ALLFEAT.shape[1]}")
    
    # 2. Run the analysis
    plot_pca_elbow(ALLFEAT)