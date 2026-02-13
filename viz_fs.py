import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIGURATION - Specify all inputs here
# ============================================================

inputs = [
    {
        "name": "Gemma 3 4b with Gemma Scope 2 65k l0 medium",  # Display name for legend/title
        "file_pattern": "eval_results/sparse_probing/*_gemma-scope-2-4b-pt-res_layer_17_width_65k_l0_medium_0_65535_eval_results.json",
        "linestyle": "-",       # Line style for this input's curves
        "alpha": 1.0,           # Opacity
        "baseline_priority": ["TopK", "Naive L1", "Lasso Path", "OFP"],  # Order of trust for baselines
        "methods_to_find": {
            # keyword (lowercase) in filename -> method label
            "topk": "TopK",
            "ofp": "OFP",
            "l1_path": "Lasso Path",
            "l1_no_path": "Naive L1",
        },
        "styles": {
            "OFP":  {"color": "#d62728", "marker": "o"},
            "Naive L1":  {"color": "#2ca02c", "marker": "^"},
            "Lasso Path":  {"color": "#FFA500", "marker": "+"},
            "TopK": {"color": "#1f77b4", "marker": "s"},
        },
        "baseline_colors": {
            "llm": "black",
            "sae": "gray",
        },
    },
    # --- Add more inputs below by copying the block above ---
    # {
    #     "name": "gemma-scope-2b L12 W32k",
    #     "file_pattern": "eval_results/sparse_probing/*_gemma-scope-2b-pt-res-canonical_layer_12_width_32k_canonical_*_eval_results.json",
    #     "linestyle": "--",
    #     "alpha": 0.8,
    #     "baseline_priority": ["TopK", "OMP", "RDA"],
    #     "methods_to_find": {
    #         "topk": "TopK",
    #         "rda": "RDA",
    #         "omp": "OMP",
    #     },
    #     "styles": {
    #         "RDA":  {"color": "#ff7f0e", "marker": "D"},
    #         "OMP":  {"color": "#9467bd", "marker": "v"},
    #         "TopK": {"color": "#17becf", "marker": "P"},
    #     },
    #     "baseline_colors": {
    #         "llm": "darkblue",
    #         "sae": "darkgray",
    #     },
    # },
]

# Plot output settings
output_filename = "feature_selection_comparison_fixed.png"
output_dpi = 300
cols = 2
fig_width = 15
fig_row_height = 6

# ============================================================
# PARSING
# ============================================================

# Structure: all_results[input_idx][dataset][method] = {k: accuracy}
# Structure: all_baselines[input_idx][dataset] = {'llm': val, 'sae': val}
all_results = []
all_baselines = []

for input_idx, inp in enumerate(inputs):
    file_pattern = inp["file_pattern"]
    methods_to_find = inp["methods_to_find"]
    baseline_priority = inp.get("baseline_priority", list(methods_to_find.values()))

    files = glob.glob(file_pattern)
    if not files:
        print(f"[Input {input_idx} '{inp['name']}'] No files found for pattern: {file_pattern}")
    else:
        print(f"[Input {input_idx} '{inp['name']}'] Found {len(files)} files.")

    results = {}
    baselines = {}
    # Track which method set each baseline so we can respect priority
    baseline_source = {}  # {dataset: {'llm': method_name, 'sae': method_name}}

    for file_path in files:
        filename = os.path.basename(file_path)

        # Identify method
        method = None
        for keyword, method_label in methods_to_find.items():
            if keyword in filename.lower():
                method = method_label
                break
        if method is None:
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        for entry in data.get("eval_result_details", []):
            dataset = entry.get("dataset_name", "Unknown Dataset")

            # Init
            if dataset not in results:
                results[dataset] = {}
                baselines[dataset] = {'llm': None, 'sae': None}
                baseline_source[dataset] = {'llm': None, 'sae': None}

            if method not in results[dataset]:
                results[dataset][method] = {}

            # --- EXTRACT BASELINES (PRIORITY-BASED) ---
            llm_val = entry.get("llm_test_accuracy", 0)
            sae_val = entry.get("sae_test_accuracy", 0)

            def _priority(m):
                """Lower index = higher priority."""
                try:
                    return baseline_priority.index(m)
                except ValueError:
                    return len(baseline_priority)

            for bkey, bval in [('llm', llm_val), ('sae', sae_val)]:
                current_src = baseline_source[dataset][bkey]
                if current_src is None or _priority(method) < _priority(current_src):
                    baselines[dataset][bkey] = bval
                    baseline_source[dataset][bkey] = method

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

    all_results.append(results)
    all_baselines.append(baselines)

# ============================================================
# COLLECT ALL DATASETS ACROSS INPUTS
# ============================================================

all_datasets = set()
for results in all_results:
    all_datasets.update(results.keys())
all_datasets = sorted(all_datasets)

if not all_datasets:
    print("No valid data parsed from any input.")
else:
    num_datasets = len(all_datasets)
    rows = (num_datasets + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_row_height * rows))
    if num_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    print("\n--- DEBUG: BASELINE VALUES FOUND ---")

    for i, dataset_name in enumerate(all_datasets):
        ax = axes[i]

        for input_idx, inp in enumerate(inputs):
            results = all_results[input_idx]
            baselines = all_baselines[input_idx]
            input_name = inp["name"]
            linestyle = inp.get("linestyle", "-")
            alpha = inp.get("alpha", 1.0)
            styles = inp.get("styles", {})
            baseline_colors = inp.get("baseline_colors", {"llm": "black", "sae": "gray"})

            if dataset_name not in results:
                continue

            methods_data = results[dataset_name]
            llm_b = baselines[dataset_name]['llm']
            sae_b = baselines[dataset_name]['sae']

            print(f"  [{input_name}] Dataset: {dataset_name} | LLM: {llm_b} | SAE: {sae_b}")

            # Plot Curves
            for method in sorted(methods_data.keys()):
                data_points = sorted(methods_data[method].items())
                if not data_points:
                    continue
                ks, accs = zip(*data_points)
                style = styles.get(method, {"color": "black", "marker": "x"})
                # Build label: "InputName - Method" when multiple inputs, else just "Method"
                if len(inputs) > 1:
                    label = f"{input_name} - {method}"
                else:
                    label = method
                ax.plot(
                    ks, accs,
                    linewidth=2,
                    linestyle=linestyle,
                    alpha=alpha,
                    label=label,
                    **style,
                )

            # Plot Baselines
            if llm_b is not None and 0 < llm_b <= 1.0:
                llm_label = (
                    f"{input_name} LLM Probe ({llm_b:.2f})"
                    if len(inputs) > 1
                    else f"LLM Probe ({llm_b:.2f})"
                )
                ax.axhline(
                    y=llm_b,
                    color=baseline_colors.get("llm", "black"),
                    linestyle='--',
                    alpha=0.7 * alpha,
                    label=llm_label,
                )

            if sae_b is not None and 0 < sae_b <= 1.0:
                sae_label = (
                    f"{input_name} SAE Probe ({sae_b:.2f})"
                    if len(inputs) > 1
                    else f"SAE Probe ({sae_b:.2f})"
                )
                ax.axhline(
                    y=sae_b,
                    color=baseline_colors.get("sae", "gray"),
                    linestyle=':',
                    alpha=0.7 * alpha,
                    label=sae_label,
                )

        ax.set_title(dataset_name, fontsize=10, fontweight='bold')
        ax.set_xlabel("k Features", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for j in range(len(all_datasets), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=output_dpi)
    plt.show()
    print(f"\nSaved to {output_filename}")