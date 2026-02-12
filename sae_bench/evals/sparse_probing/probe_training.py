import copy
import math

import torch
import torch.nn as nn
from tqdm import tqdm
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from sklearn.linear_model import LogisticRegression, OrthogonalMatchingPursuit
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector

import sae_bench.sae_bench_utils.dataset_info as dataset_info

from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score

class Probe(nn.Module):
    def __init__(self, activation_dim: int, dtype: torch.dtype):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True, dtype=dtype)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@jaxtyped(typechecker=beartype)
def prepare_probe_data(
    all_activations: dict[
        str, Float[torch.Tensor, "num_datapoints_per_class ... d_model"]
    ],
    class_name: str,
    perform_scr: bool = False,
) -> tuple[
    Float[torch.Tensor, "num_datapoints_per_class_x_2 ... d_model"],
    Int[torch.Tensor, "num_datapoints_per_class_x_2"],
]:
    """perform_scr is for the SCR metric. In this case, all_activations has 3 pairs of keys, or 6 total.
    It's a bit unfortunate to introduce coupling between the metrics, but most of the code is reused between them.
    The ... means we can have an optional seq_len dimension between num_datapoints_per_class and d_model.
    """
    positive_acts_BD = all_activations[class_name]
    device = positive_acts_BD.device

    num_positive = len(positive_acts_BD)

    if perform_scr:
        if class_name in dataset_info.PAIRED_CLASS_KEYS.keys():
            selected_negative_acts_BD = all_activations[
                dataset_info.PAIRED_CLASS_KEYS[class_name]
            ]
        elif class_name in dataset_info.PAIRED_CLASS_KEYS.values():
            reversed_dict = {v: k for k, v in dataset_info.PAIRED_CLASS_KEYS.items()}
            selected_negative_acts_BD = all_activations[reversed_dict[class_name]]
        else:
            raise ValueError(f"Class {class_name} not found in paired class keys.")
    else:
        # Collect all negative class activations and labels
        selected_negative_acts_BD = []
        negative_keys = [k for k in all_activations.keys() if k != class_name]
        num_neg_classes = len(negative_keys)
        samples_per_class = math.ceil(num_positive / num_neg_classes)

        for negative_class_name in negative_keys:
            sample_indices = torch.randperm(len(all_activations[negative_class_name]))[
                :samples_per_class
            ]
            selected_negative_acts_BD.append(
                all_activations[negative_class_name][sample_indices]
            )

        selected_negative_acts_BD = torch.cat(selected_negative_acts_BD)

    # Randomly select num_positive samples from negative class
    indices = torch.randperm(len(selected_negative_acts_BD))[:num_positive]
    selected_negative_acts_BD = selected_negative_acts_BD[indices]

    assert selected_negative_acts_BD.shape == positive_acts_BD.shape

    # Combine positive and negative samples
    combined_acts = torch.cat([positive_acts_BD, selected_negative_acts_BD])

    combined_labels = torch.empty(len(combined_acts), dtype=torch.int, device=device)
    combined_labels[:num_positive] = dataset_info.POSITIVE_CLASS_LABEL
    combined_labels[num_positive:] = dataset_info.NEGATIVE_CLASS_LABEL

    # Shuffle the combined data
    shuffle_indices = torch.randperm(len(combined_acts))
    shuffled_acts = combined_acts[shuffle_indices]
    shuffled_labels = combined_labels[shuffle_indices]

    return shuffled_acts, shuffled_labels


@jaxtyped(typechecker=beartype)
def get_top_k_mean_diff_mask(
    acts_BD: Float[torch.Tensor, "batch_size d_model"],
    labels_B: Int[torch.Tensor, "batch_size"],
    k: int,
    cache_key : str | None = None,
) -> Bool[torch.Tensor, "k"]:
    return get_fisher_mask(acts_BD, labels_B, k)
    # return get_omp_mask(acts_BD, labels_B, k)
    # return get_topk_filter_sfs_mask(acts_BD, labels_B, k, cache_key)
    # positive_mask_B = labels_B == dataset_info.POSITIVE_CLASS_LABEL
    # negative_mask_B = labels_B == dataset_info.NEGATIVE_CLASS_LABEL

    # positive_distribution_D = acts_BD[positive_mask_B].mean(dim=0)
    # negative_distribution_D = acts_BD[negative_mask_B].mean(dim=0)
    # distribution_diff_D = (positive_distribution_D - negative_distribution_D).abs()
    # top_k_indices_D = torch.argsort(distribution_diff_D, descending=True)[:k]

    # mask_D = torch.ones(acts_BD.shape[1], dtype=torch.bool, device=acts_BD.device)
    # mask_D[top_k_indices_D] = False

    # return mask_D

_sfs_cache = {}

def get_topk_filter_sfs_mask(acts_BD, labels_B, k,cache_key : str | None = None, filter_count=200, ):
    if cache_key not in _sfs_cache:
        positive_mask_B = labels_B == dataset_info.POSITIVE_CLASS_LABEL
        negative_mask_B = labels_B == dataset_info.NEGATIVE_CLASS_LABEL

        positive_acts = acts_BD[positive_mask_B]
        negative_acts = acts_BD[negative_mask_B]

        positive_distribution_D = positive_acts.mean(dim=0)
        negative_distribution_D = negative_acts.mean(dim=0)

        pvar = positive_acts.var(dim = 0)
        nvar = negative_acts.var(dim = 0)

        distribution_diff_D = (positive_distribution_D - negative_distribution_D).abs()
        fisher_D = distribution_diff_D ** 2 / (pvar + nvar + 1e-3)

        top_filter_indices_F = torch.argsort(fisher_D, descending=True)[:filter_count]

        filtered_acts_BF = acts_BD[:, top_filter_indices_F].cpu().float().numpy()
        labels_np = labels_B.cpu().numpy()

        max_k = min(100, filter_count)
        remaining = set(range(filtered_acts_BF.shape[1]))
        selected_order = []

        def evaluate_candidate(candidate, selected_order):
            feat_indices = selected_order + [candidate]
            X_sub = filtered_acts_BF[:, feat_indices]
            scores = cross_val_score(
                LogisticRegression(max_iter=200, solver="liblinear", penalty="l2"),
                X_sub, labels_np, cv=10, scoring="accuracy",
            )
            return candidate, scores.mean()

        print("No cache hit. Generating:")
        for step in tqdm(range(max_k)):
            results = Parallel(n_jobs=16)(
                delayed(evaluate_candidate)(c, selected_order)
                for c in remaining
            )
            best_feat = max(results, key=lambda x: x[1])[0]
            selected_order.append(best_feat)
            remaining.remove(best_feat)

        _sfs_cache[cache_key] = (top_filter_indices_F, selected_order)

    top_filter_indices_F, selected_order = _sfs_cache[cache_key]
    selected_in_filtered = selected_order[:k]
    selected_original_indices = top_filter_indices_F[selected_in_filtered]

    mask_D = torch.ones(acts_BD.shape[1], dtype=torch.bool, device=acts_BD.device)
    mask_D[selected_original_indices] = False
    return mask_D


# Test with orthogonal matching pursuit feature selection - maybe this helps!
@jaxtyped(typechecker=beartype)
def get_omp_mask(
    acts_BD: Float[torch.Tensor, "batch_size d_model"],
    labels_B: Int[torch.Tensor, "batch_size"],
    k: int,
    cache_key : str | None = None,
) -> Bool[torch.Tensor, "d_model"]:
    """Use Orthogonal Matching Pursuit to greedily select k features
    that best predict the binary label (treated as a regression target)."""
    acts_np = acts_BD.cpu().float().numpy()
    labels_np = labels_B.cpu().float().numpy()

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k)
    omp.fit(acts_np, labels_np)

    selected = omp.coef_ != 0  # boolean numpy array of shape (d_model,)

    # Existing convention: True = EXCLUDE, False = KEEP
    mask_D = torch.ones(acts_BD.shape[1], dtype=torch.bool, device=acts_BD.device)
    mask_D[torch.from_numpy(selected).to(acts_BD.device)] = False

    return mask_D

@beartype
def get_omd_mask(
    acts_BD: Float[torch.Tensor, "batch_size d_model"],
    labels_B: Int[torch.Tensor, "batch_size"],
    k: int,
    cache_key : str | None = None,
) -> Bool[torch.Tensor, "d_model"]:
    """
    Orthogonal Mean Difference (OMD) for classification.
    Greedily selects k features based on class mean difference, orthogonalizing 
    remaining activations w.r.t the selected feature at each step.
    """
    residual_acts = acts_BD.clone().float()
    pos_mask = labels_B == 1
    neg_mask = labels_B == 0
    
    selected_indices = []
    d_model = acts_BD.shape[1]
    
    # Ensure we don't crash on empty classes
    if not pos_mask.any() or not neg_mask.any():
        return torch.ones(d_model, dtype=torch.bool, device=acts_BD.device)

    for _ in range(k):
        # 1. Compute mean difference on current residuals
        mu_pos = residual_acts[pos_mask].mean(dim=0)
        mu_neg = residual_acts[neg_mask].mean(dim=0)
        diff = torch.abs(mu_pos - mu_neg)

        # 2. Mask previously selected
        if selected_indices:
            diff[selected_indices] = -1.0

        # 3. Select best feature
        best_idx = torch.argmax(diff).item()
        selected_indices.append(best_idx)

        # 4. Orthogonalize: Project selected feature out of all activations
        v = residual_acts[:, best_idx]
        v_norm = torch.linalg.vector_norm(v)
        
        if v_norm > 1e-6:
            v_hat = v / v_norm
            projections = v_hat @ residual_acts  # (D,)
            residual_acts -= torch.outer(v_hat, projections)

    # Return mask (True = EXCLUDE)
    mask_D = torch.ones(d_model, dtype=torch.bool, device=acts_BD.device)
    mask_D[selected_indices] = False
    
    return mask_D

@beartype
def get_fisher_mask(
    acts_BD: Float[torch.Tensor, "batch_size d_model"],
    labels_B: Int[torch.Tensor, "batch_size"],
    k: int,
    cache_key : str | None = None,
    gamma : float = 1e-2,
) -> Bool[torch.Tensor, "d_model"]:
    """
    Orthogonal Fisher Analysis.
    Selects features that maximize the Fisher Score (Signal-to-Noise Ratio),
    then orthogonalizes residuals.
    
    Score = (Mean_Diff)^2 / (Var_Pos + Var_Neg + gamma)
    Gamma term is added to avoid issues with very low variance
    """
    residual_acts = acts_BD.clone().float()
    pos_mask = labels_B == 1
    neg_mask = labels_B == 0
    
    selected_indices = []
    d_model = acts_BD.shape[1]
    
    # Safety check for empty classes
    if not pos_mask.any() or not neg_mask.any():
        return torch.ones(d_model, dtype=torch.bool, device=acts_BD.device)
    
    for _ in range(k):
        # 1. Compute Means and Variances for each class on RESIDUALS
        # Note: We use var(dim=0) to get variance per feature
        
        # Positive Class Stats
        acts_pos = residual_acts[pos_mask]
        mu_pos = acts_pos.mean(dim=0)
        var_pos = acts_pos.var(dim=0)
        
        # Negative Class Stats
        acts_neg = residual_acts[neg_mask]
        mu_neg = acts_neg.mean(dim=0)
        var_neg = acts_neg.var(dim=0)

        # 2. Compute Fisher Score (Signal / Noise)
        # We add epsilon to denominator to prevent division by zero for constant features
        numerator = (mu_pos - mu_neg) ** 2
        denominator = var_pos + var_neg + gamma
        #variance_term = torch.log((var_pos + var_neg) / (2 * torch.sqrt(var_pos * var_neg) + gamma))
        fisher_scores = numerator / denominator# + variance_term

        # 3. Mask previously selected
        if selected_indices:
            fisher_scores[selected_indices] = -1.0

        # 4. Select best feature
        best_idx = torch.argmax(fisher_scores).item()
        selected_indices.append(best_idx)

        # 5. Orthogonalize (Same as OMD)
        v = residual_acts[:, best_idx]
        v_norm = torch.linalg.vector_norm(v)
        
        if v_norm > 1e-6:
            v_hat = v / v_norm
            projections = v_hat @ residual_acts
            residual_acts -= torch.outer(v_hat, projections)

    # Return mask (True = EXCLUDE)
    mask_D = torch.ones(d_model, dtype=torch.bool, device=acts_BD.device)
    mask_D[selected_indices] = False
    
    return mask_D

@jaxtyped(typechecker=beartype)
def apply_topk_mask_zero_dims(
    acts_BD: Float[torch.Tensor, "batch_size d_model"],
    mask_D: Bool[torch.Tensor, "d_model"],
) -> Float[torch.Tensor, "batch_size k"]:
    masked_acts_BD = acts_BD.clone()
    masked_acts_BD[:, mask_D] = 0.0

    return masked_acts_BD


@jaxtyped(typechecker=beartype)
def apply_topk_mask_reduce_dim(
    acts_BD: Float[torch.Tensor, "batch_size d_model"],
    mask_D: Bool[torch.Tensor, "d_model"],
) -> Float[torch.Tensor, "batch_size k"]:
    masked_acts_BD = acts_BD.clone()

    masked_acts_BD = masked_acts_BD[:, ~mask_D]

    return masked_acts_BD


@beartype
def train_sklearn_probe(
    train_inputs: Float[torch.Tensor, "train_dataset_size d_model"],
    train_labels: Int[torch.Tensor, "train_dataset_size"],
    test_inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    test_labels: Int[torch.Tensor, "test_dataset_size"],
    max_iter: int = 1000,  # non-default sklearn value, increased due to convergence warnings
    C: float = 1.0,  # default sklearn value
    verbose: bool = False,
    l1_ratio: float | None = None,
) -> tuple[LogisticRegression, float]:
    train_inputs = train_inputs.to(dtype=torch.float32)
    test_inputs = test_inputs.to(dtype=torch.float32)

    # Convert torch tensors to numpy arrays
    train_inputs_np = train_inputs.cpu().numpy()
    train_labels_np = train_labels.cpu().numpy()
    test_inputs_np = test_inputs.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()

    # Initialize the LogisticRegression model
    if l1_ratio is not None:
        # Use Elastic Net regularization
        probe = LogisticRegression(
            penalty="elasticnet" if l1_ratio != 1.0 else "l1",
            solver="saga" if l1_ratio != 1.0 else "liblinear",
            C=C,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            verbose=int(verbose),
        )
    else:
        # Use L2 regularization
        probe = LogisticRegression(
            penalty="l2", C=C, max_iter=max_iter, verbose=int(verbose)
        )

    # Train the model
    probe.fit(train_inputs_np, train_labels_np)

    # Compute accuracies
    train_accuracy = accuracy_score(train_labels_np, probe.predict(train_inputs_np))
    test_accuracy = accuracy_score(test_labels_np, probe.predict(test_inputs_np))

    if verbose:
        print("\nTraining completed.")
        print(f"Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy:.6f}\n")

    return probe, test_accuracy


# Helper function to test the probe
@beartype
def test_sklearn_probe(
    inputs: Float[torch.Tensor, "dataset_size d_model"],
    labels: Int[torch.Tensor, "dataset_size"],
    probe: LogisticRegression,
) -> float:
    inputs = inputs.to(dtype=torch.float32)
    inputs_np = inputs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    predictions = probe.predict(inputs_np)
    return accuracy_score(labels_np, predictions)  # type: ignore


@jaxtyped(typechecker=beartype)
@torch.no_grad
def test_probe_gpu(
    inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    labels: Int[torch.Tensor, "test_dataset_size"],
    batch_size: int,
    probe: Probe,
) -> float:
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        corrects_0 = []
        corrects_1 = []
        all_corrects = []
        losses = []

        for i in range(0, len(labels), batch_size):
            acts_BD = inputs[i : i + batch_size]
            labels_B = labels[i : i + batch_size]
            logits_B = probe(acts_BD)
            preds_B = (logits_B > 0.0).long()
            correct_B = (preds_B == labels_B).float()

            all_corrects.append(correct_B)
            corrects_0.append(correct_B[labels_B == 0])
            corrects_1.append(correct_B[labels_B == 1])

            loss = criterion(logits_B, labels_B.to(dtype=probe.net.weight.dtype))
            losses.append(loss)

        accuracy_all = torch.cat(all_corrects).mean().item()

    return accuracy_all


@jaxtyped(typechecker=beartype)
def train_probe_gpu(
    train_inputs: Float[torch.Tensor, "train_dataset_size d_model"],
    train_labels: Int[torch.Tensor, "train_dataset_size"],
    test_inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    test_labels: Int[torch.Tensor, "test_dataset_size"],
    dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    verbose: bool = False,
    l1_penalty: float | None = None,
    early_stopping_patience: int = 10,
) -> tuple[Probe, float]:
    """We have a GPU training function for training on all SAE features, which was very slow (1 minute+) on CPU.
    This is also used for SCR / TPP, which require probe weights."""
    device = train_inputs.device
    model_dtype = train_inputs.dtype

    print(f"Training probe with dim: {dim}, device: {device}, dtype: {model_dtype}")

    probe = Probe(dim, model_dtype).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)  # type: ignore
    criterion = nn.BCEWithLogitsLoss()

    best_test_accuracy = 0.0
    best_probe = None
    patience_counter = 0
    for epoch in range(epochs):
        indices = torch.randperm(len(train_inputs))

        for i in range(0, len(train_inputs), batch_size):
            batch_indices = indices[i : i + batch_size]
            acts_BD = train_inputs[batch_indices]
            labels_B = train_labels[batch_indices]
            logits_B = probe(acts_BD)
            loss = criterion(
                logits_B, labels_B.clone().detach().to(device=device, dtype=model_dtype)
            )

            if l1_penalty is not None:
                l1_loss = l1_penalty * torch.sum(torch.abs(probe.net.weight))
                loss += l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy = test_probe_gpu(train_inputs, train_labels, batch_size, probe)
        test_accuracy = test_probe_gpu(test_inputs, test_labels, batch_size, probe)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_probe = copy.deepcopy(probe)
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} Loss: {loss.item()}, train accuracy: {train_accuracy}, test accuracy: {test_accuracy}"  # type: ignore
            )

        if patience_counter >= early_stopping_patience:
            print(
                f"GPU probe training early stopping triggered after {epoch + 1} epochs"
            )
            break

    assert best_probe is not None
    return best_probe, best_test_accuracy


@jaxtyped(typechecker=beartype)
def train_probe_on_activations(
    train_activations: dict[str, Float[torch.Tensor, "train_dataset_size d_model"]],
    test_activations: dict[str, Float[torch.Tensor, "test_dataset_size d_model"]],
    select_top_k: int | None = None,
    use_sklearn: bool = True,
    batch_size: int = 16,
    epochs: int = 5,
    lr: float = 1e-3,
    verbose: bool = False,
    early_stopping_patience: int = 10,
    perform_scr: bool = False,
    l1_penalty: float | None = None,
    dataset_name : str | None = None
) -> tuple[dict[str, LogisticRegression | Probe], dict[str, float]]:
    """Train a probe on the given activations and return the probe and test accuracies for each profession.
    use_sklearn is a flag to use sklearn's LogisticRegression model instead of a custom PyTorch model.
    We use sklearn by default. probe training on GPU is only for training a probe on all SAE features.
    """
    torch.set_grad_enabled(True)

    probes, test_accuracies = {}, {}
    penalty_str = l1_penalty if l1_penalty is not None else "None"


    for profession in train_activations.keys():
        train_acts, train_labels = prepare_probe_data(
            train_activations, profession, perform_scr
        )
        test_acts, test_labels = prepare_probe_data(
            test_activations, profession, perform_scr
        )

        if select_top_k is not None:
            activation_mask_D = get_top_k_mean_diff_mask(
                train_acts, train_labels, select_top_k, cache_key=f"{dataset_name}_{profession}"
            )
            train_acts = apply_topk_mask_reduce_dim(train_acts, activation_mask_D)
            test_acts = apply_topk_mask_reduce_dim(test_acts, activation_mask_D)

        activation_dim = train_acts.shape[1]

        print(f"Num non-zero elements: {activation_dim}, L1 Penalty: {penalty_str}")
        l1_ratio = 1.0 if l1_penalty is not None else None
        
        if use_sklearn:
            c_val = l1_penalty if l1_penalty else 1.0
            probe, test_accuracy = train_sklearn_probe(
                train_acts,
                train_labels,
                test_acts,
                test_labels,
                verbose=False,
                l1_ratio=l1_ratio,
                C=c_val,
            )
        else:
            probe, test_accuracy = train_probe_gpu(
                train_acts,
                train_labels,
                test_acts,
                test_labels,
                dim=activation_dim,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                verbose=verbose,
                early_stopping_patience=early_stopping_patience,
                l1_penalty=l1_penalty,
            )

        print(f"Test accuracy for {profession}: {test_accuracy}")

        probes[profession] = probe
        test_accuracies[profession] = test_accuracy

    return probes, test_accuracies
