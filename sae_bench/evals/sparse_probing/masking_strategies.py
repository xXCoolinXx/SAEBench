from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from joblib import Parallel, delayed
from scipy.sparse import csc_matrix
from sklearn.linear_model import (
    LogisticRegression,
    OrthogonalMatchingPursuit,
)
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

import sae_bench.sae_bench_utils.dataset_info as dataset_info


# Abstract Functor for definining a new mask. This is for the purpose of feature selection
class MaskingStrategy(ABC):
    def __init__(self, max_k : int = 100):
        self.cache = {} # Define a cache in case the method needs to do multiple sweeps (e.g. Lasso Path). Greedy/Top K methods work well without this, but would be sped up slightly
        self.max_k = max_k

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def __call__(self, 
                 acts : Float[torch.Tensor, "batch_size d_model"], 
                 labels : Int[torch.Tensor, "batch_size"], 
                 k : int, 
                 cache_key : str | None = None
                 ) -> Bool[torch.Tensor, "d_model"]: # Other required parameters should be specified in the derived class init function
        pass

    def clear_cache(self):
        # Free up space. Best to call once finished with the given probing task
        self.cache.clear()

def acts_helper(
    acts : Float[torch.Tensor, "batch_size d_model"], 
    labels : Int[torch.Tensor, "batch_size"]
    ) -> tuple[Float[torch.Tensor, "batch_size d_model"], Float[torch.Tensor, "batch_size d_model"]]:

    positive_mask = labels == dataset_info.POSITIVE_CLASS_LABEL
    negative_mask = labels == dataset_info.NEGATIVE_CLASS_LABEL

    positive_acts = acts[positive_mask]
    negative_acts = acts[negative_mask]

    return positive_acts, negative_acts

def abs_mean_diff_bc(
    acts : Float[torch.Tensor, "batch_size d_model"], 
    labels : Int[torch.Tensor, "batch_size"]
    ) -> Float[torch.Tensor, "d_model"]:
    
    positive_acts, negative_acts = acts_helper(acts, labels)

    positive_means = positive_acts.mean(dim=0)
    negative_means = negative_acts.mean(dim=0)

    # Return absolute mean differences
    return (positive_means - negative_means).abs()

def fisher_score_bc(
        acts : Float[torch.Tensor, "batch_size d_model"], 
        labels : Int[torch.Tensor, "batch_size"], 
        epsilon=1e-3) -> Float[torch.Tensor, "d_model"]:
    """Helper function that computes the Fisher Score between positive and negative labels"""
    # Get abs mean differences
    abs_mean_differences = abs_mean_diff_bc(acts, labels)

    # Compute variances
    positive_acts, negative_acts = acts_helper(acts, labels)
    positive_vars = positive_acts.var(dim=0)
    negative_vars = negative_acts.var(dim=0)

    # Return Fisher Score
    return abs_mean_differences ** 2 / (positive_vars + negative_vars + epsilon)

class TopK_SFS_Mask(MaskingStrategy):
    def __init__(self, filter_count : int = 200, max_k : int = 100, target_metric : Literal["accuracy", "roc_auc", "f1"] = "accuracy", n_jobs=-1):
        """filter_count : int - Number of latents to filter to. By default uses the top K latents by Fisher Score"""
        super().__init__(max_k=max_k)

        self.filter_count = filter_count
        self.target_metric = target_metric
        self.n_jobs = n_jobs

        if max_k > filter_count:
            raise ValueError(f"Maximum K value ({max_k}) higher than total available after filtering process {filter_count}")

    @jaxtyped(typechecker=beartype)
    def __call__(self, 
                 acts : Float[torch.Tensor, "batch_size d_model"], 
                 labels : Int[torch.Tensor, "batch_size"], 
                 k : int, 
                 cache_key : str | None = None
                 ) -> Bool[torch.Tensor, "d_model"]: # Other required parameters should be specified in the derived class init function
        if cache_key not in self.cache:
            print("No cache hit. Generating:")
            fisher_scores = fisher_score_bc(acts, labels) 

            _, filtered_indices = torch.topk(fisher_scores, self.filter_count)

            filtered_acts = acts[:, filtered_indices].cpu().float().numpy()
            labels_np = labels.cpu().numpy()

            remaining = set(range(filtered_acts.shape[1]))
            selected_order = []

            def evaluate_candidate(candidate, selected_order):
                feat_indices = selected_order + [candidate]
                X_sub = filtered_acts[:, feat_indices]
                scores = cross_val_score(
                    LogisticRegression(max_iter=200, solver="liblinear", penalty="l2"),
                    X_sub, labels_np, cv=10, scoring=self.target_metric,
                )
                return candidate, scores.mean()

            for step in tqdm(range(self.max_k)):
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(evaluate_candidate)(c, selected_order)
                    for c in remaining
                )
                best_feat = max(results, key=lambda x: x[1])[0] # type: ignore
                selected_order.append(best_feat)
                remaining.remove(best_feat)

            self.cache[cache_key] = (filtered_indices, selected_order)

        filtered_indices, selected_order = self.cache[cache_key]
        selected_in_filtered = selected_order[:k]
        selected_original_indices = filtered_indices[selected_in_filtered]

        mask = torch.ones(acts.shape[1], dtype=torch.bool, device=acts.device)
        mask[selected_original_indices] = False

        return mask

class TopK_L1_Mask(MaskingStrategy):
    def __init__(self, max_k : int = 100, C : float = 1e-3):
        """C defines inverse regularization strength
        Selects the Top K weights of an L1 probe for training"""
        super().__init__(max_k=max_k)
        self.C = C

    @jaxtyped(typechecker=beartype)
    def __call__(self, 
                 acts : Float[torch.Tensor, "batch_size d_model"], 
                 labels : Int[torch.Tensor, "batch_size"], 
                 k : int, 
                 cache_key : str | None = None
                 ) -> Bool[torch.Tensor, "d_model"]:
        if cache_key not in self.cache:
            clf = LogisticRegression(
                penalty="l1",
                C=self.C,
                solver="saga",
                max_iter=10000,
            )
            clf.fit(acts.cpu().float().numpy(), labels.cpu().float().numpy())

            # coef_ is (1, d_model) for binary classification
            weights = torch.tensor(clf.coef_[0], device=acts.device, dtype=acts.dtype).abs()

            _, top_k_indices = torch.topk(weights, self.max_k)
            self.cache[cache_key] = top_k_indices

        top_k_indices = self.cache[cache_key][:k]
        mask = torch.ones(acts.shape[1], dtype=torch.bool, device=acts.device)
        mask[top_k_indices] = False

        return mask

class OMP_Mask(MaskingStrategy):
    @jaxtyped(typechecker=beartype)
    def __call__(self, 
                 acts : Float[torch.Tensor, "batch_size d_model"], 
                 labels : Int[torch.Tensor, "batch_size"], 
                 k : int, 
                 cache_key : str | None = None  # noqa: ARG002
                 ) -> Bool[torch.Tensor, "d_model"]:
        """Use Orthogonal Matching Pursuit to greedily select k features
        that best predict the binary label (treated as a regression target)."""
        acts_np = acts.cpu().float().numpy()
        labels_np = labels.cpu().float().numpy()

        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k)
        omp.fit(acts_np, labels_np)

        selected = omp.coef_ != 0  # boolean numpy array of shape (d_model,)

        # Existing convention: True = EXCLUDE, False = KEEP
        mask = torch.ones(acts.shape[1], dtype=torch.bool, device=acts.device)
        mask[torch.from_numpy(selected).to(acts.device)] = False

        return mask
        
class TopK_Metric_Mask(MaskingStrategy):
    def __init__(self, max_k : int = 100, metric : Literal["abs_mean_diff", "fisher"] = "abs_mean_diff"):
        super().__init__(max_k=max_k)

        match metric:
            case "abs_mean_diff":
                self.score_function = abs_mean_diff_bc
            case "fisher":
                self.score_function = fisher_score_bc
            case _:
                raise NotImplementedError(f"The selected metric for heuristic filtering ({metric}) is not yet implemented.")   

    @jaxtyped(typechecker=beartype)
    def __call__(self, 
        acts : Float[torch.Tensor, "batch_size d_model"], 
        labels : Int[torch.Tensor, "batch_size"], 
        k : int, 
        cache_key : str | None = None  # noqa: ARG002
        ) -> Bool[torch.Tensor, "d_model"]:
        # TODO: Add caching to avoid repeated computation
        scores = self.score_function(acts, labels)

        _, top_k_indices = torch.topk(scores, k)

        mask = torch.ones(acts.shape[1], dtype=torch.bool, device=acts.device)
        mask[top_k_indices] = False

        return mask
    
class TopK_AbsMeanDiff_Mask(TopK_Metric_Mask):
    def __init__(self, max_k : int = 100):
        super().__init__(max_k=max_k, metric="abs_mean_diff")

class TopK_Fisher_Mask(TopK_Metric_Mask):
    def __init__(self, max_k : int = 100):
        super().__init__(max_k=max_k, metric="fisher")

class OrthogonalFisherPursuit_Mask(MaskingStrategy):
    def __init__(self, max_k : int = 100, epsilon = 1e-2):
        super().__init__(max_k=max_k)

        self.epsilon = epsilon

    @jaxtyped(typechecker=beartype)
    def __call__(self, 
        acts : Float[torch.Tensor, "batch_size d_model"], 
        labels : Int[torch.Tensor, "batch_size"], 
        k : int, 
        cache_key : str | None = None  # noqa: ARG002
        ) -> Bool[torch.Tensor, "d_model"]:
        residual_acts = acts.clone().float()
        pos_mask = labels == dataset_info.POSITIVE_CLASS_LABEL
        neg_mask = labels == dataset_info.NEGATIVE_CLASS_LABEL
        
        selected_indices = []
        d_model = acts.shape[1]
        
        # Safety check for empty classes
        if not pos_mask.any() or not neg_mask.any():
            return torch.ones(d_model, dtype=torch.bool, device=acts.device)
        
        for _ in range(k):
            fisher_scores = fisher_score_bc(residual_acts, labels, epsilon=self.epsilon)

            # Mask previously selected
            if selected_indices:
                fisher_scores[selected_indices] = -1.0

            # Select best feature
            best_idx = torch.argmax(fisher_scores).item()
            selected_indices.append(best_idx)

            # Orthogonalize
            v = residual_acts[:, best_idx] # type: ignore
            v_norm = torch.linalg.vector_norm(v)
            
            if v_norm > 1e-6:
                v_hat = v / v_norm
                projections = v_hat @ residual_acts
                residual_acts -= torch.outer(v_hat, projections)

        # Return mask (True = EXCLUDE)
        mask = torch.ones(d_model, dtype=torch.bool, device=acts.device)
        mask[selected_indices] = False
        
        return mask
    
class LassoPath_Mask(MaskingStrategy):
    def __init__(self, max_k : int = 100, n_C : int = 100):
        super().__init__(max_k=max_k)

        self.n_C = n_C

    @jaxtyped(typechecker=beartype)
    def __call__(self, 
        acts : Float[torch.Tensor, "batch_size d_model"], 
        labels : Int[torch.Tensor, "batch_size"], 
        k : int, 
        cache_key : str | None = None 
        ) -> Bool[torch.Tensor, "d_model"]:
        d_model = acts.shape[1]

        if cache_key is not None and cache_key in self.cache:
            entry_order = self.cache[cache_key]
        else:
            X = acts.cpu().float().numpy()
            y = labels.cpu().numpy()

            # Standardize to avoid issues
            var0 = X[y == dataset_info.POSITIVE_CLASS_LABEL].var(axis=0)
            var1 = X[y == dataset_info.NEGATIVE_CLASS_LABEL].var(axis=0)
            within_std = np.sqrt((var0 + var1) / 2) + 1e-10
            X = X / within_std

            if np.mean(X == 0) > 0.3:
                X = csc_matrix(X)

            Cs = np.logspace(-4, 4, self.n_C)
            entry_order = []
            seen = set()

            clf = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                warm_start=True,
                max_iter=10000,
                tol=1e-5,
            )

            for C in tqdm(Cs):
                clf.set_params(C=C)
                clf.fit(X, y)
                active = np.where(clf.coef_[0] != 0)[0]
                new = [f for f in active if f not in seen]
                if new:
                    new.sort(key=lambda f: abs(clf.coef_[0, f]), reverse=True)
                    entry_order.extend(new)
                    seen.update(new)
                if len(entry_order) >= self.max_k:
                    break

            entry_order = entry_order[:self.max_k]
            if cache_key is not None:
                self.cache[cache_key] = entry_order

        top_k_indices = entry_order[:k]

        mask = torch.ones(d_model, dtype=torch.bool, device=acts.device)
        mask[top_k_indices] = False

        return mask
    
mask_registry: dict[str, type[MaskingStrategy]] = {
    "top_k_abs_mean_diff": TopK_AbsMeanDiff_Mask,
    "top_k_fisher": TopK_Fisher_Mask,
    "top_k_l1": TopK_L1_Mask,
    "omp": OMP_Mask,
    "ofp": OrthogonalFisherPursuit_Mask,
    "lasso_path": LassoPath_Mask,
    "sfs": TopK_SFS_Mask,
}
