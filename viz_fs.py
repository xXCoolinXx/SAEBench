import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# 1. Setup file search
file_pattern = "eval_results/sparse_probing/*_gemma-scope-2b-pt-res-canonical_layer_12_width_16k_canonical_0_16383_eval_results.json"
files = glob.glob(file_pattern)

if not files:
    print("No files found!")
else:
    print(f"Found {len(files)} files.")

# Data Stores
results = {} 
baselines = {} # {dataset_name: {'llm': val, 'sae': val}}

# 2. Parse Data
for file_path in files:
    filename = os.path.basename(file_path)
    
    # Identify method
    if "topk" in filename.lower():
        method = "TopK"
    elif "rda" in filename.lower():
        method = "RDA"
    elif "omp" in filename.lower():
        method = "OMP"
    else:
        continue 
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    for entry in data.get("eval_result_details", []):
        dataset = entry.get("dataset_name", "Unknown Dataset")
        
        # Init dataset in dicts if new
        if dataset not in results:
            results[dataset] = {}
            # Default baselines to None so we can track if we found them
            baselines[dataset] = {'llm': None, 'sae': None}
        
        if method not in results[dataset]:
            results[dataset][method] = {}

        # --- EXTRACT BASELINES (STRICT PRIORITY) ---
        # Only trust TopK file for baselines first. 
        # If we already found a baseline from TopK, don't overwrite it with OMP.
        llm_val = entry.get("llm_test_accuracy", 0)
        sae_val = entry.get("sae_test_accuracy", 0)

        # Logic: 
        # 1. If method is TopK, ALWAYS update (authoritative source)
        # 2. If method is OMP, only update if we haven't found a value yet
        if method == "TopK":
            baselines[dataset]['llm'] = llm_val
            baselines[dataset]['sae'] = sae_val
        elif method == "OMP":
            if baselines[dataset]['llm'] is None:
                baselines[dataset]['llm'] = llm_val
            if baselines[dataset]['sae'] is None:
                baselines[dataset]['sae'] = sae_val

        # --- EXTRACT CURVES ---
        for key, val in entry.items():
            if key.startswith("sae_top_") and key.endswith("_test_accuracy"):
                try:
                    parts = key.split("_")
                    if len(parts) >= 3 and parts[2].isdigit():
                        k = int(parts[2])
                        if val is not None:
                            results[dataset][method][k] = val
                except ValueError:
                    continue

# 3. Plotting
if not results:
    print("No valid data parsed.")
else:
    num_datasets = len(results)
    cols = 2
    rows = (num_datasets + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
    if num_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    print("\n--- DEBUG: BASELINE VALUES FOUND ---")
    
    for i, (dataset_name, methods_data) in enumerate(sorted(results.items())):
        ax = axes[i]
        
        # Print what we are plotting to verify
        llm_b = baselines[dataset_name]['llm']
        sae_b = baselines[dataset_name]['sae']
        print(f"Dataset: {dataset_name} | LLM: {llm_b} | SAE: {sae_b}")

        # Styles
        styles = {
            "RDA": {"color": "#d62728", "marker": "o", "label": "RDA"},
            "OMP": {"color": "#2ca02c", "marker": "^", "label": "OMP"},
            "TopK": {"color": "#1f77b4", "marker": "s", "label": "TopK"}
        }
        
        # Plot Curves
        for method in sorted(methods_data.keys()):
            data_points = sorted(methods_data[method].items())
            if not data_points: continue
            ks, accs = zip(*data_points)
            style = styles.get(method, {"marker": "x"})
            ax.plot(ks, accs, linewidth=2, **style)
            
        # Plot Baselines
        if llm_b is not None and llm_b > 0 and llm_b <= 1.0:
            ax.axhline(y=llm_b, color='black', linestyle='--', alpha=0.7, label=f"LLM Probe ({llm_b:.2f})")
        
        if sae_b is not None and sae_b > 0 and sae_b <= 1.0:
            ax.axhline(y=sae_b, color='gray', linestyle=':', alpha=0.7, label=f"SAE Probe ({sae_b:.2f})")

        ax.set_title(dataset_name, fontsize=10, fontweight='bold')
        ax.set_xlabel("k Features", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("feature_selection_comparison_fixed.png", dpi=300)
    plt.show()