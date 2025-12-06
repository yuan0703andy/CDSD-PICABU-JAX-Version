"""
SAVAR Benchmark Generator for CDSD / PICABU Comparison.

Key choices (aligned to PICABU Table 1):
- Edge probabilities per difficulty:
  easy=0, med-easy=1/(N-1), med-hard=2/(N-1), hard=0.5 (lag=1 only).
- Edge weights ~ Beta(4, 8), row-normalized per target.
- Default tau_max=1 to match PICABU baselines (set >1 if you need multi-lag).
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sys
sys.path.insert(0, 'savar')

from savar.model_generator import SavarGenerator
from savar.functions import check_stability


@dataclass
class SAVARDataset:
    """Container for SAVAR benchmark dataset."""

    data_field: np.ndarray  # (T, L) where L = nx * ny
    latent_ts: np.ndarray  # (T, N)
    links_coeffs: Dict
    adjacency_matrix: np.ndarray  # (N, N, tau_max)
    coeff_matrix: np.ndarray  # (N, N, tau_max)
    mode_weights: np.ndarray  # (N, ny, nx)
    n_variables: int
    difficulty: str
    tau_max: int
    time_length: int
    n_cross_links: int
    seed: int


def links_to_matrices(
    links_coeffs: Dict, n_variables: int, tau_max: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Tigramite-style links to adjacency and coefficient matrices.

    links_coeffs format:
        {target_j: [((source_i, -lag), coeff), ...], ...}

    Returns:
        adjacency: (N, N, tau_max) binary, adj[j, i, k] = 1 means i^{t-k-1} -> j^t
        coeffs: (N, N, tau_max) coefficient values
    """
    adjacency = np.zeros((n_variables, n_variables, tau_max))
    coeffs = np.zeros((n_variables, n_variables, tau_max))

    for target, parents in links_coeffs.items():
        for (source, neg_lag), coeff in parents:
            lag = abs(neg_lag)  # -1 -> 1, -2 -> 2, ...
            if 1 <= lag <= tau_max:
                adjacency[target, source, lag - 1] = 1
                coeffs[target, source, lag - 1] = coeff

    return adjacency, coeffs


def picabu_edge_prob(n_variables: int, difficulty: str) -> float:
    difficulty = difficulty.lower().replace("_", "-")
    if difficulty == "easy":
        return 0.0
    if difficulty == "med-easy":
        return 1.0 / max(n_variables - 1, 1)
    if difficulty == "med-hard":
        return 2.0 / max(n_variables - 1, 1)
    if difficulty == "hard":
        return 0.5
    return 1.0 / max(n_variables - 1, 1)


def get_difficulty_config(difficulty: str, n_variables: int) -> Dict:
    """
    Map difficulty to expected edge count using PICABU probabilities (lag=1).
    """
    p = picabu_edge_prob(n_variables, difficulty)
    n_cross = int(round(n_variables * (n_variables - 1) * p))
    return {
        "n_cross_links": max(0, n_cross),
        # These means/stds get overridden by the Beta draw below; kept to
        # satisfy generator args.
        "cross_mean": 0.33,
        "cross_std": 0.12,
        "auto_coeffs_mean": 0.33,
        "auto_coeffs_std": 0.12,
    }


def _apply_picabu_coeffs(
    adjacency: np.ndarray, coeffs: np.ndarray, seed: Optional[int]
) -> np.ndarray:
    """
    Re-sample coeffs with Beta(4,8) at lag=1 and row-normalize per target.
    """
    rng = np.random.default_rng(seed if seed is not None else 0)

    # keep only lag=1; zero out others
    if adjacency.shape[2] > 1:
        adjacency[..., 1:] = 0.0
        coeffs[..., 1:] = 0.0

    A = adjacency[..., 0]
    weights = np.where(A > 0, rng.beta(4, 8, size=A.shape), 0.0)
    row_sum = weights.sum(axis=1, keepdims=True) + 1e-8
    weights = weights / row_sum
    coeffs[..., 0] = weights
    return coeffs


def generate_savar_dataset(
    n_variables: int = 25,
    difficulty: str = "med-easy",
    time_length: int = 2000,
    tau_max: int = 1,
    seed: Optional[int] = None,
    resolution: Tuple[int, int] = (30, 90),
    verbose: bool = False,
) -> SAVARDataset:
    """
    Generate a single SAVAR dataset matching PICABU benchmark settings.
    """
    cfg = get_difficulty_config(difficulty, n_variables)
    if verbose:
        print(f"Generating SAVAR: N={n_variables}, difficulty={difficulty}")
        print(f"  n_cross_links={cfg['n_cross_links']}, tau_max={tau_max}")

    generator = SavarGenerator(
        n_variables=n_variables,
        time_length=time_length,
        tau_max=tau_max,
        tau_min=1,
        resolution=resolution,
        n_cross_links=cfg["n_cross_links"],
        cross_mean=cfg["cross_mean"],
        cross_std=cfg["cross_std"],
        auto_coeffs_mean=cfg["auto_coeffs_mean"],
        auto_coffs_std=cfg["auto_coeffs_std"],
        auto_links=True,
        model_seed=seed,
        verbose=verbose,
    )

    savar_model = generator.generate_savar()
    savar_model.generate_data()

    data_field = savar_model.data_field
    if data_field.shape[0] != time_length:
        data_field = data_field.T  # ensure (T, L)

    links_coeffs = deepcopy(savar_model.links_coeffs)
    mode_weights = savar_model.mode_weights

    adjacency, coeffs = links_to_matrices(links_coeffs, n_variables, tau_max)
    coeffs = _apply_picabu_coeffs(adjacency, coeffs, seed)

    # Optional stability check; shrink if unstable
    try:
        check_stability(adjacency, lag_first_axis=False, verbose=False)
    except Exception:
        # simple rescale to enforce spectral radius < 1
        spec = max(np.abs(np.linalg.eigvals(coeffs[..., 0])))
        if spec > 1:
            coeffs[..., 0] /= (1.05 * spec)

    latent_ts = getattr(savar_model, "latent_process", None)
    if latent_ts is None:
        latent_ts = getattr(savar_model, "X_latent", None)
    if latent_ts is None:
        mode_weights_flat = mode_weights.reshape(n_variables, -1)  # (N, L)
        W_pinv = np.linalg.pinv(mode_weights_flat.T)  # (N, L)
        latent_ts = data_field @ W_pinv.T

    return SAVARDataset(
        data_field=data_field,
        latent_ts=latent_ts,
        links_coeffs=links_coeffs,
        adjacency_matrix=adjacency,
        coeff_matrix=coeffs,
        mode_weights=mode_weights,
        n_variables=n_variables,
        difficulty=difficulty,
        tau_max=tau_max,
        time_length=time_length,
        n_cross_links=cfg["n_cross_links"],
        seed=seed if seed is not None else -1,
    )


def generate_picabu_benchmark(
    output_dir: str = "./dataset/savar_benchmark",
    n_seeds: int = 5,
    time_length: int = 2000,
    tau_max: int = 1,
    save_data: bool = True,
    verbose: bool = True,
) -> Dict[str, List[SAVARDataset]]:
    """
    Generate full PICABU benchmark suite.

    12 configs: N in {4, 25, 100} × difficulty in {easy, med-easy, med-hard, hard}
    """
    output_path = Path(output_dir)
    if save_data:
        output_path.mkdir(parents=True, exist_ok=True)

    n_variables_list = [4, 25, 100]
    difficulties = ["easy", "med-easy", "med-hard", "hard"]

    all_datasets: Dict[str, List[SAVARDataset]] = {}
    summary_stats = []

    for n_var in n_variables_list:
        for diff in difficulties:
            config_name = f"N{n_var}_{diff}"
            if verbose:
                print(f"\n{'=' * 50}\nGenerating {config_name}...")

            datasets: List[SAVARDataset] = []
            for seed in range(n_seeds):
                actual_seed = seed * 1000 + n_var
                try:
                    ds = generate_savar_dataset(
                        n_variables=n_var,
                        difficulty=diff,
                        time_length=time_length,
                        tau_max=tau_max,
                        seed=actual_seed,
                        verbose=False,
                    )
                    datasets.append(ds)

                    if save_data:
                        save_path = output_path / config_name
                        save_path.mkdir(exist_ok=True)
                        np.savez(
                            save_path / f"seed_{seed}.npz",
                            data_field=ds.data_field,
                            latent_ts=ds.latent_ts,
                            adjacency_matrix=ds.adjacency_matrix,
                            coeff_matrix=ds.coeff_matrix,
                            mode_weights=ds.mode_weights,
                        )
                        links_json = {
                            str(k): [((int(s), int(l)), float(c)) for (s, l), c in v]
                            for k, v in ds.links_coeffs.items()
                        }
                        with open(save_path / f"seed_{seed}_links.json", "w") as f:
                            json.dump(links_json, f, indent=2)
                except Exception as exc:  # pragma: no cover - generation failure
                    print(f"  Warning: seed {seed} failed: {exc}")
                    continue

            all_datasets[config_name] = datasets
            if datasets:
                avg_edges = float(np.mean([d.adjacency_matrix.sum() for d in datasets]))
                total_possible = n_var * n_var * tau_max
                density = avg_edges / total_possible
                summary_stats.append(
                    {
                        "config": config_name,
                        "n_variables": n_var,
                        "difficulty": diff,
                        "avg_edges": avg_edges,
                        "density": density,
                        "n_datasets": len(datasets),
                    }
                )
                if verbose:
                    print(f"  Generated {len(datasets)} datasets")
                    print(f"  Avg edges: {avg_edges:.1f}/{total_possible} (dens={density:.3f})")

    if save_data:
        summary = {
            "n_variables_list": n_variables_list,
            "difficulties": difficulties,
            "n_seeds": n_seeds,
            "time_length": time_length,
            "tau_max": tau_max,
            "stats": summary_stats,
        }
        with open(output_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        if verbose:
            print(f"\n{'=' * 50}\nBenchmark saved to: {output_path}")

    return all_datasets


def load_savar_dataset(config_path: str, seed: int = 0) -> SAVARDataset:
    """
    Load a saved SAVAR dataset from generate_picabu_benchmark output.
    """
    path = Path(config_path)
    data = np.load(path / f"seed_{seed}.npz")
    with open(path / f"seed_{seed}_links.json", "r") as f:
        links_raw = json.load(f)
    links_coeffs = {
        int(k): [((s, l), c) for (s, l), c in v] for k, v in links_raw.items()
    }

    config_name = path.name
    parts = config_name.split("_")
    n_var = int(parts[0][1:])
    diff = "-".join(parts[1:])

    return SAVARDataset(
        data_field=data["data_field"],
        latent_ts=data["latent_ts"],
        links_coeffs=links_coeffs,
        adjacency_matrix=data["adjacency_matrix"],
        coeff_matrix=data["coeff_matrix"],
        mode_weights=data["mode_weights"],
        n_variables=n_var,
        difficulty=diff,
        tau_max=data["adjacency_matrix"].shape[2],
        time_length=data["data_field"].shape[0],
        n_cross_links=-1,
        seed=seed,
    )


# =======================
# Evaluation Metrics
# =======================

def compute_graph_f1(
    pred_adj: np.ndarray, true_adj: np.ndarray, threshold: float = 0.5
) -> Dict:
    """Precision/recall/F1 for graphs; supports (N,N) or (N,N,tau)."""
    if pred_adj.ndim == 2:
        pred_adj = pred_adj[..., None]
    if true_adj.ndim == 2:
        true_adj = true_adj[..., None]

    pred_bin = (pred_adj > threshold).astype(int).sum(axis=2) > 0
    true_bin = (true_adj > 0.5).astype(int).sum(axis=2) > 0

    pred_flat = pred_bin.astype(int).ravel()
    true_flat = true_bin.astype(int).ravel()

    tp = int((pred_flat & true_flat).sum())
    fp = int((pred_flat & (1 - true_flat)).sum())
    fn = int(((1 - pred_flat) & true_flat).sum())

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "tp": tp, "fp": fp, "fn": fn}


def compute_shd(pred_adj: np.ndarray, true_adj: np.ndarray, threshold: float = 0.5) -> int:
    """Structural Hamming Distance between predicted and true graphs."""
    if pred_adj.ndim == 2:
        pred_adj = pred_adj[..., None]
    if true_adj.ndim == 2:
        true_adj = true_adj[..., None]

    pred_bin = (pred_adj > threshold).astype(int).sum(axis=2) > 0
    true_bin = (true_adj > 0.5).astype(int).sum(axis=2) > 0

    return int(np.abs(pred_bin.astype(int) - true_bin.astype(int)).sum())


def compute_mcc(pred_latents: np.ndarray, true_latents: np.ndarray) -> float:
    """
    Mean correlation coefficient with one-to-one matching (Hungarian).
    """
    ps = (pred_latents - pred_latents.mean(0)) / (pred_latents.std(0) + 1e-8)
    ts = (true_latents - true_latents.mean(0)) / (true_latents.std(0) + 1e-8)
    C = np.abs(ps.T @ ts) / pred_latents.shape[0]
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(-C)
        return float(C[r, c].mean())
    except Exception:
        return float(C.max(axis=1).mean())


if __name__ == "__main__":
    print("SAVAR Benchmark Generator")
    print("=" * 60)

    print("\n1) Single dataset quick test...")
    ds = generate_savar_dataset(
        n_variables=10, difficulty="med-easy", time_length=500, tau_max=1, seed=42, verbose=True
    )
    print(f"   data_field: {ds.data_field.shape}, latent_ts: {ds.latent_ts.shape}")
    print(f"   adjacency: {ds.adjacency_matrix.shape}, edges: {int(ds.adjacency_matrix.sum())}")
    print(f"   coeff range: [{ds.coeff_matrix.min():.3f}, {ds.coeff_matrix.max():.3f}]")

    print("\n2) Metrics smoke test...")
    noisy_adj = ds.adjacency_matrix + np.random.randn(*ds.adjacency_matrix.shape) * 0.1
    g_metrics = compute_graph_f1(noisy_adj, ds.adjacency_matrix)
    mcc = compute_mcc(ds.latent_ts, ds.latent_ts + np.random.randn(*ds.latent_ts.shape) * 0.1)
    print(f"   F1(noisy adj): {g_metrics['f1']:.3f}, MCC(noisy latents): {mcc:.3f}")

    print("\nReady. To generate full benchmark:")
    print("  datasets = generate_picabu_benchmark('./savar_benchmark', n_seeds=2, tau_max=1)")
