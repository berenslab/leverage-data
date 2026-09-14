import os
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sksurv.metrics import concordance_index_censored
from itertools import product
def set_seed(seed: int = 12345, silent=False) -> None:
    """Set seed for reproducibility."""
    try:
        import numpy as np

        np.random.seed(seed)
        if not silent:
            print(f"Numpy random seed was set as {seed}")
    except ImportError:
        print("Could not set seed for numpy: Numpy not found.")

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if not silent:
            print(f"Torch random seed was set as {seed}")
    except ImportError:
        print("Could not set seed for torch: Torch not found.")

    try:
        import random

        random.seed(seed)
        if not silent:
            print(f"Random seed was set as {seed}")
    except ImportError:
        if not silent:
            print("Could not set seed for random: Random not found.")
        else:
            print()

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    if not silent:
        print(f"PYTHONHASHSEED was set as {seed}")
        print()

def get_areds_data_dir():

    yaml_path = os.path.join(
        os.path.dirname(__file__), "../../configs/image_dirs.yml"
    )

    with open(yaml_path) as f:
        yml = yaml.safe_load(f)

    paths = yml["image_dirs"]

    for path in paths:
        if "$SCRATCH" in path:
            path = os.path.join(os.environ["SCRATCH"], path.split("$SCRATCH/")[-1])
        if os.path.exists(path):
            return Path(path)
        
    raise Exception("No data path found")

def isattr(obj, attr, value=True) -> bool:
    """Check if an attribute has the passed value. Combines hasattr and getattr in one function.
    Returns False if it does not exist.

    Args:
        obj: Object
        attr: Attribute name (str, optional)
        value: Value to check

    Returns:
        bool: True if attribute exists
    """

    if attr is None:
        raise ValueError("Attribute name cannot be None. Pass a string.")

    return hasattr(obj, attr) and getattr(obj, attr) == value

def get_risk_scores(_events, _durations, survival_curves, evaluation_visits_from_now):
    """Extract risk scores from survival curves."""
    risks = 1.0 - np.array(survival_curves)
    sum_risk_c = (np.sum(risks, axis=1) if len(evaluation_visits_from_now) > 1 else risks.flatten())
    return sum_risk_c, np.array(_events).astype(bool), np.array(_durations)



def benchmark_cindex_pvalues(
    durations, events, model_scores: dict, reference_models: list,
    n_bootstrap=5000, seed=42
):
    """
    Benchmark p-values: all non-reference models compared against each reference.

    Parameters
    ----------
    durations        : array-like (N,)
    events           : array-like (N,)  1=event, 0=censored
    model_scores     : dict  {model_name: risk_score_array}
    reference_models : list  of model names (must be keys in model_scores)
    n_bootstrap      : int

    Returns
    -------
    results   : DataFrame with C-index, difference, CI, raw and corrected p-values
    cindices  : dict of observed C-index per model
    """
    rng       = np.random.default_rng(seed)
    durations = np.array(durations)
    events    = np.array(events)
    n         = len(durations)

    # --- Observed C-indices ---
    cindices = {
        name: concordance_index_censored(events.astype(bool), durations, scores)[0]
        for name, scores in model_scores.items()
    }

    # --- Bootstrap ---
    boot_cindices = {name: [] for name in model_scores}
    for _ in range(n_bootstrap):
        idx  = rng.integers(0, n, size=n)
        d, e = durations[idx], events[idx]
        for name, scores in model_scores.items():
            boot_cindices[name].append(
                concordance_index_censored(e.astype(bool), d, scores[idx])[0]
            )
    boot_cindices = {k: np.array(v) for k, v in boot_cindices.items()}

    # --- Pairwise comparisons: each non-ref vs each reference ---
    non_refs = [m for m in model_scores if m not in reference_models]
    rows = []

    for ref, model in product(reference_models, non_refs):
        obs_diff   = cindices[model] - cindices[ref]
        boot_diffs = boot_cindices[model] - boot_cindices[ref]

        ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])

        # Two-sided p-value
        p_value = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))
        print('reference model is0', ref)
        rows.append({
            "reference":  ref,
            "model":      model,
            "c_ref":      cindices[ref],
            "c_model":    cindices[model],
            "difference": obs_diff,
            "ci_lower":   ci_lower,
            "ci_upper":   ci_upper,
            "p_raw":      p_value,
        })

    results = pd.DataFrame(rows)

    # --- Benjamini-Hochberg correction ---
    results = results.sort_values("p_raw").reset_index(drop=True)
    m = len(results)
    results["bh_rank"]     = np.arange(1, m + 1)
    results["p_corrected"] = (results["p_raw"] * m / results["bh_rank"]).clip(upper=1.0)
    results["p_corrected"] = results["p_corrected"][::-1].cummin()[::-1]  # monotonicity
    results["significant"] = results["p_corrected"] < 0.05

    results = results.drop(columns="bh_rank").sort_values(["reference", "difference"])

    return results, cindices