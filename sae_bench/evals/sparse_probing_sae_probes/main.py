import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import sae_bench.sae_bench_utils.general_utils as general_utils
import torch
from sae_bench.evals.sparse_probing_sae_probes.eval_config import (
    SparseProbingSaeProbesEvalConfig,
)
from sae_bench.evals.sparse_probing_sae_probes.eval_output import (
    SaeProbesLlmMetrics,
    SaeProbesMetricCategories,
    SaeProbesResultDetail,
    SaeProbesSaeMetrics,
    SparseProbingSaeProbesEvalOutput,
)
from sae_bench.sae_bench_utils import (
    get_eval_uuid,
    get_sae_bench_version,
    get_sae_lens_version,
)
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex
from sae_lens import SAE
from sae_probes import run_baseline_evals
from tqdm import tqdm


def _sae_probes_results_glob(
    results_root: str, model_name: str, setting: str, prefix: str = "sae_probes"
) -> list[Path]:
    root = Path(results_root) / f"{prefix}_{model_name}" / f"{setting}_setting"
    return sorted(list(root.glob("*.json")))


def _parse_dataset_from_filename(path: Path) -> str:
    # Filenames look like: "119_us_state_TX_blocks.4.hook_resid_post_l1.json"
    # Dataset short name is the prefix until the first occurrence of "_blocks."
    stem = path.stem
    if "_blocks." in stem:
        return stem.split("_blocks.")[0]
    return stem


def _aggregate_metrics_from_sae_probes_json(
    file_path: Path,
) -> dict[str, float]:
    with open(file_path) as f:
        data: list[dict[str, Any]] = json.load(f)
    # sae-probes saves a list of entries, one per K (and possibly metadata entries)
    # Each entry contains keys: {"k", "test_acc", "test_auc", "test_f1", ...}
    k_to_metrics: dict[int, dict[str, float]] = {}
    for entry in data:
        if "k" in entry and "test_acc" in entry:
            try:
                k = int(entry["k"])  # type: ignore[arg-type]
                metrics = {
                    "test_accuracy": float(entry["test_acc"]),  # type: ignore[arg-type]
                }
                if "test_auc" in entry:
                    metrics["test_auc"] = float(entry["test_auc"])  # type: ignore[arg-type]
                if "test_f1" in entry:
                    metrics["test_f1"] = float(entry["test_f1"])  # type: ignore[arg-type]
                k_to_metrics[k] = metrics
            except Exception:
                continue
    out: dict[str, float] = {}
    for k, metrics in k_to_metrics.items():
        for metric_name, value in metrics.items():
            out[f"sae_top_{k}_{metric_name}"] = value
    return out


def _mean_of_keys(dicts: list[dict[str, float]], key: str) -> float | None:
    vals = [d[key] for d in dicts if key in d]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


# Unholy monkey patch
# --- 1. Imports from your library structure ---
# Import the specific module we are patching to access its local functions
import importlib
from unittest.mock import patch

from sae_probes.constants import DEFAULT_RESULTS_PATH  # noqa: E402
from sae_probes.generate_sae_activations import generate_sae_activations  # noqa: E402
from sae_probes.utils_training import find_best_reg  # noqa: E402
from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np

# Global switch: "topk" or "omp"
FEATURE_SELECTION_METHOD = "topk"

def create_latent_subset_patch(sae_feature_indices: torch.Tensor, sae_evals_mod):
    """
    Returns a patched version of run_sae_eval that restricts analysis to
    specific SAE latent indices provided in the tensor.
    """

    def patched_run_sae_eval(
        sae,
        dataset,
        hook_name,
        reg_type,
        setting,
        model_name,
        model_cache_path,
        binarize=False,
        num_train=None,
        corrupt_frac=None,
        frac=None,
        device="cuda",
        batch_size=128,
        ks=None,
        results_path=DEFAULT_RESULTS_PATH,
        mean_diff_normalization="mean",
    ):
        print("PATCHED run_sae_eval", dataset, "subset_size", len(sae_feature_indices))
        print(f"Feature selection method: {FEATURE_SELECTION_METHOD}")

        activations = generate_sae_activations(
            sae=sae,
            setting=setting,
            dataset=dataset,
            hook_name=hook_name,
            model_name=model_name,
            device=device,
            num_train=num_train,
            frac=frac,
            batch_size=batch_size,
            model_cache_path=model_cache_path,
        )

        X_train_sae = activations.X_train
        X_test_sae = activations.X_test
        y_train = activations.y_train
        y_test = activations.y_test

        idxs = sae_feature_indices.to(X_train_sae.device)

        X_train_sae = X_train_sae[:, idxs]
        X_test_sae = X_test_sae[:, idxs]

        subset_size = X_train_sae.shape[1]

        if ks is None:
            if setting == "normal":
                ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            else:
                ks = [16, 128]

        ks = [k for k in ks if k <= subset_size]
        if not ks:
            ks = [subset_size]

        all_metrics = []

        mean_act_normalization = getattr(sae_evals_mod, "mean_act_normalization")
        get_sorted_indices = getattr(sae_evals_mod, "get_sorted_indices")
        get_save_metrics_path = getattr(sae_evals_mod, "get_save_metrics_path")

        normalization = None
        if mean_diff_normalization == "mean":
            normalization = mean_act_normalization
        elif mean_diff_normalization == "none":
            normalization = None
        elif callable(mean_diff_normalization):
            normalization = mean_diff_normalization
        else:
            raise ValueError(f"Invalid mean_diff_normalization: {mean_diff_normalization}")

        # Precompute for topk
        if FEATURE_SELECTION_METHOD == "topk":
            sorted_indices = get_sorted_indices(X_train_sae, y_train, normalization)

        # Precompute numpy arrays for OMP
        if FEATURE_SELECTION_METHOD == "omp":
            X_np = X_train_sae.cpu().float().numpy()
            y_np = y_train.cpu().float().numpy()

        for k in tqdm(ks, desc="Evaluating k values"):
            if FEATURE_SELECTION_METHOD == "omp":
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k)
                omp.fit(X_np, y_np)
                top_subset_indices = np.where(omp.coef_ != 0)[0]
            else:
                top_subset_indices = sorted_indices[:k]

            X_train_filtered = X_train_sae[:, top_subset_indices]
            X_test_filtered = X_test_sae[:, top_subset_indices]

            if binarize and setting == "normal":
                X_train_filtered = X_train_filtered > 1
                X_test_filtered = X_test_filtered > 1

            results = find_best_reg(
                X_train=X_train_filtered,
                y_train=y_train,
                X_test=X_test_filtered,
                y_test=y_test,
                n_jobs=-1,
                parallel=False,
                penalty=reg_type,
            )
            metrics = asdict(results.metrics)

            global_indices = idxs[top_subset_indices].tolist()

            metrics.update(
                {
                    "k": k,
                    "dataset": dataset,
                    "hook_name": hook_name,
                    "reg_type": reg_type,
                    "binarize": binarize,
                    "indices": global_indices,
                    "feature_selection": FEATURE_SELECTION_METHOD,
                }
            )

            if setting == "scarcity":
                metrics["num_train"] = num_train
            elif setting == "imbalance":
                metrics["frac"] = frac

            all_metrics.append(metrics)

        save_path = get_save_metrics_path(
            dataset=dataset,
            hook_name=hook_name,
            reg_type=reg_type,
            binarize=binarize,
            model_name=model_name,
            setting=setting,
            num_train=num_train,
            corrupt_frac=corrupt_frac,
            frac=frac,
            sae_results_path=results_path,
        )

        print(f"Saving results to {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(all_metrics, f, indent=4, ensure_ascii=False)

        return True

    return patched_run_sae_eval

def run_eval(
    config: SparseProbingSaeProbesEvalConfig,
    selected_saes: list[tuple[str, SAE]] | list[tuple[str, str]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
) -> dict[str, dict[str, Any]]:
    if config.setting != "normal":
        raise NotImplementedError(
            "Only 'normal' setting is supported for sparse_probing_sae_probes aggregation currently."
        )

    if len(selected_saes) > 1 and config.sae_feature_indices is not None:
        raise NotImplementedError("Bruh!")

    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(config.results_path, exist_ok=True)

    results_dict: dict[str, dict[str, Any]] = {}

    for sae_release, sae_object_or_id in tqdm(
        selected_saes, desc="Running sae-probes on selected SAEs"
    ):
        sae_id, sae, sparsity = general_utils.load_and_format_sae(  # type: ignore
            sae_release, sae_object_or_id, device
        )

        if config.sae_feature_indices is None:
            d_sae = sae.W_dec.shape[0]
            config.sae_feature_indices = torch.arange(0, d_sae, device=device)

        r_min = config.sae_feature_indices.min().item()
        r_max = config.sae_feature_indices.max().item()
        sae_result_path = general_utils.get_results_filepath(
            output_path,
            sae_release,
            sae_id,
            extra_str=f"{r_min}_{r_max}",
        )

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Skipping {sae_release}_{sae_id} as results already exist")
            continue

        sae_results_path = os.path.join(
            config.results_path, f"{sae_release}_{sae_id}_{r_min}_{r_max}"
        )

        os.makedirs(sae_results_path, exist_ok=True)

        # Perform the unholy monkey patch
        sae_evals_mod = importlib.import_module("sae_probes.run_sae_evals")
        patched = create_latent_subset_patch(config.sae_feature_indices, sae_evals_mod)

        with patch.object(sae_evals_mod, "run_sae_eval", new=patched):
            sae_evals_mod.run_sae_evals(
                sae=sae,
                model_name=config.model_name,
                hook_name=sae.cfg.hook_name,
                reg_type=config.reg_type,
                setting=config.setting,
                ks=config.ks,
                binarize=config.binarize,
                results_path=sae_results_path,
                model_cache_path=config.model_cache_path,
                datasets=config.dataset_names,
                device=device,
            )

        # Collect per-dataset JSONs and collate (filter by hook/reg to avoid stale files)
        expected_suffix = f"_{sae.cfg.hook_name}_{config.reg_type}.json"
        json_files = [
            f
            for f in _sae_probes_results_glob(
                sae_results_path, config.model_name, config.setting
            )
            if f.name.endswith(expected_suffix)
        ]
        per_dataset_details: list[SaeProbesResultDetail] = []
        per_dataset_metric_dicts: list[dict[str, float]] = []
        for jf in json_files:
            ds_name = _parse_dataset_from_filename(jf)
            ds_metrics = _aggregate_metrics_from_sae_probes_json(jf)
            per_dataset_metric_dicts.append(
                {k: v for k, v in ds_metrics.items() if k.startswith("sae_top_")}
            )

            # Build sae_metrics_by_k dictionary for this dataset
            ds_metrics_by_k: dict[int, dict[str, float]] = {}
            for k in config.ks:
                k_metrics = {}
                for metric in ["test_accuracy", "test_auc", "test_f1"]:
                    key = f"sae_top_{k}_{metric}"
                    if key in ds_metrics:
                        k_metrics[metric] = ds_metrics[key]
                if k_metrics:
                    ds_metrics_by_k[k] = k_metrics

            per_dataset_details.append(
                SaeProbesResultDetail(
                    dataset_name=ds_name,
                    sae_metrics_by_k=ds_metrics_by_k if ds_metrics_by_k else None,
                    **ds_metrics,
                )
            )

        # Aggregate across datasets (mean per-k)
        agg_metrics_dict: dict[str, float | None] = {}
        agg_metrics_by_k: dict[int, dict[str, float]] = {}
        for k in config.ks:
            k_metrics: dict[str, float] = {}
            for metric in ["test_accuracy", "test_auc", "test_f1"]:
                key = f"sae_top_{k}_{metric}"
                mean_val = _mean_of_keys(per_dataset_metric_dicts, key)
                agg_metrics_dict[key] = mean_val
                if mean_val is not None:
                    k_metrics[metric] = mean_val
            if k_metrics:
                agg_metrics_by_k[k] = k_metrics

        llm_metrics = SaeProbesLlmMetrics()
        if config.include_llm_baseline:
            # Run baseline evals (idempotent) and parse results to populate llm metrics
            run_baseline_evals(
                model_name=config.model_name,
                hook_name=sae.cfg.hook_name,
                setting=config.setting,  # type: ignore[arg-type]
                method=config.baseline_method,  # type: ignore[arg-type]
                results_path=config.results_path,
                model_cache_path=config.model_cache_path,
                datasets=config.dataset_names,
                device=device,
            )
            # Baseline JSON pattern: baseline_results_{model_name}/{setting}_setting/{dataset}_{hook}_{method}.json
            baseline_suffix = f"_{sae.cfg.hook_name}_{config.baseline_method}.json"
            baseline_files = [
                f
                for f in _sae_probes_results_glob(
                    config.results_path,
                    config.model_name,
                    config.setting,
                    prefix="baseline_results",
                )
                if f.name.endswith(baseline_suffix)
            ]
            # compute overall mean test_acc, test_auc, test_f1 across datasets
            llm_accs: list[float] = []
            llm_aucs: list[float] = []
            llm_f1s: list[float] = []
            per_ds_metrics: dict[str, dict[str, float]] = {}
            for bf in baseline_files:
                ds_name = _parse_dataset_from_filename(bf)
                with open(bf) as f:
                    entries: list[dict[str, Any]] = json.load(f)
                # baselines save a single-element list
                if entries and "test_acc" in entries[0]:
                    metrics = {}
                    if "test_acc" in entries[0]:
                        acc = float(entries[0]["test_acc"])  # type: ignore[arg-type]
                        metrics["test_accuracy"] = acc
                        llm_accs.append(acc)
                    if "test_auc" in entries[0]:
                        auc = float(entries[0]["test_auc"])  # type: ignore[arg-type]
                        metrics["test_auc"] = auc
                        llm_aucs.append(auc)
                    if "test_f1" in entries[0]:
                        f1 = float(entries[0]["test_f1"])  # type: ignore[arg-type]
                        metrics["test_f1"] = f1
                        llm_f1s.append(f1)
                    per_ds_metrics[ds_name] = metrics
            if llm_accs:
                llm_metrics.llm_test_accuracy = float(sum(llm_accs) / len(llm_accs))
            if llm_aucs:
                llm_metrics.llm_test_auc = float(sum(llm_aucs) / len(llm_aucs))
            if llm_f1s:
                llm_metrics.llm_test_f1 = float(sum(llm_f1s) / len(llm_f1s))
            # attach per-dataset baseline to details
            for detail in per_dataset_details:
                if detail.dataset_name in per_ds_metrics:
                    metrics = per_ds_metrics[detail.dataset_name]
                    if "test_accuracy" in metrics:
                        detail.llm_test_accuracy = metrics["test_accuracy"]
                    if "test_auc" in metrics:
                        detail.llm_test_auc = metrics["test_auc"]
                    if "test_f1" in metrics:
                        detail.llm_test_f1 = metrics["test_f1"]

        eval_output = SparseProbingSaeProbesEvalOutput(
            eval_config=config,
            eval_id=eval_instance_id,
            datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
            eval_result_metrics=SaeProbesMetricCategories(
                llm=llm_metrics,
                sae=SaeProbesSaeMetrics(**agg_metrics_dict),  # type: ignore[arg-type]
            ),
            eval_result_details=per_dataset_details,
            sae_metrics_by_k=agg_metrics_by_k if agg_metrics_by_k else None,
            eval_result_unstructured=None,
            sae_bench_commit_hash=sae_bench_commit_hash,
            sae_lens_id=sae_id,
            sae_lens_release_id=sae_release,
            sae_lens_version=sae_lens_version,
            sae_cfg_dict=asdict(sae.cfg),
        )

        results_dict[f"{sae_release}_{sae_id}"] = asdict(eval_output)
        eval_output.to_json_file(sae_result_path, indent=2)

        # Reset
        config.sae_feature_indices = None

    return results_dict


def create_config_and_selected_saes(
    args,
) -> tuple[SparseProbingSaeProbesEvalConfig, list[tuple[str, str]]]:
    config = SparseProbingSaeProbesEvalConfig(
        model_name=args.model_name,
        reg_type=args.reg_type,
        setting=args.setting,
        ks=args.ks,
        binarize=args.binarize,
        results_path=args.results_path,
        model_cache_path=args.model_cache_path,
    )

    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    releases = set([release for release, _ in selected_saes])
    print(f"Selected SAEs from releases: {releases}")
    for release, sae in selected_saes:
        print(f"Sample SAEs: {release}, {sae}")

    return config, selected_saes


def arg_parser():
    parser = argparse.ArgumentParser(description="Run sae-probes sparse probing eval")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--sae_regex_pattern", type=str, required=True)
    parser.add_argument("--sae_block_pattern", type=str, required=True)
    parser.add_argument(
        "--reg_type",
        type=str,
        default="l1",
        choices=["l1", "l2"],
        help="sae-probes regularization type",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="normal",
        choices=["normal", "scarcity", "imbalance"],
        help="sae-probes data-balance setting",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50, 100],
        help="List of K values",
    )
    parser.add_argument("--binarize", action="store_true")
    parser.add_argument(
        "--results_path",
        type=str,
        default="artifacts/sparse_probing_sae_probes",
        help="Directory where sae-probes writes JSONs",
    )
    parser.add_argument(
        "--model_cache_path",
        type=str,
        default=None,
        help="Optional directory to persist model activations",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="eval_results/sparse_probing_sae_probes",
        help="SAEBench output folder",
    )
    parser.add_argument("--force_rerun", action="store_true")
    return parser


if __name__ == "__main__":
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()

    start_time = time.time()
    config, selected_saes = create_config_and_selected_saes(args)
    os.makedirs(args.output_folder, exist_ok=True)
    run_eval(
        config,
        selected_saes,
        device,
        args.output_folder,
        force_rerun=args.force_rerun,
    )
    end_time = time.time()
    print(f"Finished evaluation in {end_time - start_time} seconds")
