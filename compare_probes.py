import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def plot_comparison(baseline_path, omp_path, save_dir=None):
    baseline = load_results(baseline_path)
    omp = load_results(omp_path)

    k_values = [1, 2, 5, 10, 20, 50, 100]

    bl_details = {d["dataset_name"]: d for d in baseline["eval_result_details"]}
    omp_details = {d["dataset_name"]: d for d in omp["eval_result_details"]}
    all_datasets = sorted(set(bl_details.keys()) & set(omp_details.keys()))

    for dataset in all_datasets:
        short_name = dataset.split("/")[-1].replace("_results", "")

        bl_d = bl_details[dataset]
        omp_d = omp_details[dataset]

        bl_scores = []
        omp_scores = []
        bl_k = []
        omp_k = []

        for k in k_values:
            key = f"sae_top_{k}_test_accuracy"
            bl_val = bl_d.get(key)
            omp_val = omp_d.get(key)
            if bl_val is not None:
                bl_scores.append(bl_val)
                bl_k.append(k)
            if omp_val is not None:
                omp_scores.append(omp_val)
                omp_k.append(k)

        sae_dense = bl_d["sae_test_accuracy"]
        llm_dense = bl_d["llm_test_accuracy"]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(bl_k, bl_scores, "o-", color="tab:red", label="Mean-Diff", linewidth=2, markersize=7)
        ax.plot(omp_k, omp_scores, "s-", color="tab:blue", label="OMP", linewidth=2, markersize=7)

        ax.axhline(y=sae_dense, color="gray", linestyle="--", alpha=0.7, label=f"SAE Dense ({sae_dense:.3f})")
        ax.axhline(y=llm_dense, color="black", linestyle="--", alpha=0.7, label=f"LLM Dense ({llm_dense:.3f})")

        ax.set_xscale("log")
        all_k = sorted(set(bl_k + omp_k))
        ax.set_xticks(all_k)
        ax.set_xticklabels([str(k) for k in all_k])

        ax.set_xlabel("k (number of selected latents)", fontsize=12)
        ax.set_ylabel("Test Accuracy", fontsize=12)
        ax.set_title(short_name, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        all_vals = bl_scores + omp_scores + [sae_dense, llm_dense]
        y_min = min(all_vals) - 0.02
        y_max = max(all_vals) + 0.02
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(save_dir) / f"{short_name}.png", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_comparison.py <baseline_json> <omp_json> [save_dir]")
        sys.exit(1)

    save_dir = sys.argv[3] if len(sys.argv) > 3 else None
    plot_comparison(sys.argv[1], sys.argv[2], save_dir)