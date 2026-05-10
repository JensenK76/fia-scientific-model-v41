#!/usr/bin/env python3
"""
FIA Scientific Model V4.1
========================

A stricter null-controlled testbench for FIA-inspired mass-diverse
relational networks.

Main V4.1 upgrades over V4
------------------------
1. Common random numbers:
   Paired controls share graph, initial phases, quenched vectors, and dynamic
   phase-noise paths for each seed/sigma/beta condition.

2. Fixed permutations over beta:
   Shuffled controls use seed/sigma-fixed permutations, not beta-dependent
   permutations. This makes beta sweeps and monotonicity tests meaningful.

3. True equal-identical null:
   equal_identical has m_i=1, uniform omega_i=omega0, and no quenched
   omega heterogeneity. It still has dynamic phase noise unless phase_noise=0.

4. Factorial diversity channels:
   kernel, capacity, and omega channels can be activated separately:
   equal_identical, kernel_only, capacity_only, omega_only, kernel_capacity,
   kernel_omega, capacity_omega, full.

5. Burn-in:
   Persistent hob tracking starts only after burn_in_steps.

6. Threshold robustness:
   Final-frame candidate robustness is measured across a grid of score, R,
   and C thresholds.

7. Bootstrap confidence intervals:
   Paired full-minus-control differences include bootstrap CIs for primary
   endpoints.

8. Graph nulls:
   random_graph_full and degree_preserving_rewire_full are both included.

9. Signed stiffness diagnostics:
   The in-phase metric uses max(cos,0), but V4.1 also reports the fraction of
   negative stiffness edges and a signed Laplacian minimum eigenvalue.

10. Deterministic inference layer:
    Python hash randomization is removed from bootstrap seeds. V4.1 uses stable
    label hashing, paired sign-flip tests, and bootstrap CIs.

11. Factorial channel decomposition:
    The 2^3 kernel/capacity/omega controls are converted into per-seed
    channel effects and interaction contrasts.

12. Decision report:
    V4.1 writes an explicit scientific verdict report summarizing which claims
    are supported, inconclusive, or contradicted by the control comparisons.

Primary endpoint
----------------
The preregistered primary endpoint in this testbench is:

    C_metric_full

where

    W_metric_ij = (K/d_ref) A_ij max(cos(theta_i-theta_j), 0)

and

    C_metric_full = lambda_2(L_W_metric, M_capacity)

Secondary endpoints include persistent_n, persistent_units_max, C_A_full,
C_metric_over_C_A_full, threshold robustness, and signed-stiffness diagnostics.

Run
---
    python fia_scientific_v41.py
    python fia_scientific_v41.py --quick
    python fia_scientific_v41.py --targeted --no-plots
    python fia_scientific_v41.py --out FIA_V41_output --no-plots
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class FiaV4Config:
    # Size and integration
    N: int = 140
    steps: int = 220
    dt: float = 0.035
    seed0: int = 41
    n_reps: int = 6

    # Graph
    spatial_dim: int = 3
    k_nn: int = 8
    sigma_space: float = 0.22
    random_edges_per_node: int = 1
    rewire_swaps_per_edge: int = 6
    baseline_weight_floor: float = 1e-12
    score_ref_quantile: float = 0.95

    # Masses and mismatch kernel
    mass_sigmas: Tuple[float, ...] = (0.0, 0.15, 0.30)
    betas: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)

    # Dynamics
    K: float = 1.4
    omega0: float = 0.0
    omega_mass_slope: float = 0.50
    omega_quenched_noise: float = 0.00  # default 0 to avoid hidden diversity
    phase_noise: float = 0.005          # dynamic SDE-like phase noise, scaled by sqrt(dt)
    normalization: str = "global"       # "global" or "local_degree"

    # Diagnostics and hob detection: primary thresholds
    detect_every: int = 10
    burn_in_steps: int = 80
    score_threshold: float = 0.58
    min_hob_size: int = 4
    R_threshold: float = 0.72
    C_threshold: float = 0.010

    # Threshold-robustness grid for final snapshots
    score_thresholds: Tuple[float, ...] = (0.52, 0.58, 0.64)
    R_thresholds: Tuple[float, ...] = (0.65, 0.72, 0.80)
    C_thresholds: Tuple[float, ...] = (0.005, 0.010, 0.020)

    # Persistence tracking
    jaccard_threshold: float = 0.55
    min_lifetime_frames: int = 3

    # Controls
    controls: Tuple[str, ...] = (
        "full",
        "equal_identical",
        "beta0_full",
        "kernel_only",
        "capacity_only",
        "omega_only",
        "kernel_capacity",
        "kernel_omega",
        "capacity_omega",
        "shuffle_kernel_fixed",
        "shuffle_capacity_fixed",
        "shuffle_omega_fixed",
        "static_phase",
        "random_graph_full",
        "degree_preserving_rewire_full",
    )

    # Bootstrap
    bootstrap_n: int = 1000
    bootstrap_seed: int = 12345

    # Outputs
    output_dir: str = "FIA_Scientific_V41_output"
    make_plots: bool = True


KEY_METRICS = [
    "C_metric_full",
    "C_A_full",
    "C_score_full",
    "C_old_full",
    "C_metric_over_C_A_full",
    "R_global_final",
    "candidate_n_final",
    "candidate_units_final",
    "persistent_n",
    "persistent_units_max",
    "persistent_lifetime_max",
    "persistent_mean_C",
    "negative_stiffness_fraction",
    "signed_laplacian_min_eig",
]


# ============================================================
# Utilities
# ============================================================

def stable_label_int(label: object, modulus: int = 10_000_000) -> int:
    """Deterministic integer hash independent of Python's randomized hash()."""
    digest = hashlib.blake2b(str(label).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulus



def safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    return float(num / (den + eps))


def mass_diversity(m: np.ndarray) -> float:
    return float(np.var(m) / (np.mean(m) ** 2 + 1e-12))


def vector_diversity(x: np.ndarray) -> float:
    scale = np.mean(np.abs(x)) + 1e-12
    return float(np.var(x) / (scale * scale))


def order_parameter(theta: np.ndarray, idx: Optional[Sequence[int]] = None) -> float:
    vals = theta if idx is None else theta[np.asarray(idx, dtype=int)]
    if len(vals) == 0:
        return 0.0
    return float(np.abs(np.mean(np.exp(1j * vals))))


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def bootstrap_ci_mean(x: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = len(x)
    for b in range(n_boot):
        means[b] = np.mean(x[rng.integers(0, n, size=n)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# ============================================================
# Graph construction
# ============================================================

def dense_edges_from_matrix(G: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    iu, ju = np.where(np.triu(G, k=1) > 0)
    wu = G[iu, ju]
    return iu.astype(int), ju.astype(int), wu.astype(float)


def build_spatial_baseline_graph(
    rng: np.random.Generator,
    N: int,
    spatial_dim: int,
    k_nn: int,
    sigma_space: float,
    random_edges_per_node: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = rng.random((N, spatial_dim))
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(diff * diff, axis=2)

    G0 = np.zeros((N, N), dtype=float)
    for i in range(N):
        idx = np.argsort(r2[i])[1:k_nn + 1]
        G0[i, idx] = np.exp(-r2[i, idx] / (2.0 * sigma_space ** 2))

    G0 = np.maximum(G0, G0.T)

    # Add sparse long-range edges independent of mass.
    n_extra = int(max(0, random_edges_per_node) * N // 2)
    attempts = 0
    added = 0
    while added < n_extra and attempts < 50 * max(1, n_extra):
        attempts += 1
        i = int(rng.integers(0, N))
        j = int(rng.integers(0, N))
        if i == j or G0[i, j] > 0:
            continue
        G0[i, j] = G0[j, i] = 0.5
        added += 1

    np.fill_diagonal(G0, 0.0)
    iu, ju, g0u = dense_edges_from_matrix(G0)
    return iu, ju, g0u, G0, pos


def randomize_graph_edges(
    rng: np.random.Generator,
    N: int,
    edge_count: int,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = set()
    attempts = 0
    max_attempts = 250 * edge_count + 1000
    while len(pairs) < edge_count and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, N))
        j = int(rng.integers(0, N))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))
    if len(pairs) < edge_count:
        raise RuntimeError("Could not generate enough random edges.")
    pair_arr = np.array(sorted(pairs), dtype=int)
    iu = pair_arr[:, 0]
    ju = pair_arr[:, 1]
    g0u = rng.permutation(weights.copy())
    return iu, ju, g0u


def degree_preserving_rewire(
    rng: np.random.Generator,
    N: int,
    iu: np.ndarray,
    ju: np.ndarray,
    g0u: np.ndarray,
    swaps_per_edge: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unweighted degree-preserving double-edge swap.

    Weight multiset is preserved and randomly assigned after rewiring.
    """
    edges = set()
    for a, b in zip(iu, ju):
        i, j = (int(a), int(b))
        if i > j:
            i, j = j, i
        edges.add((i, j))

    edge_list = list(edges)
    n_swaps = swaps_per_edge * len(edge_list)

    for _ in range(n_swaps):
        if len(edge_list) < 2:
            break
        eidx = rng.choice(len(edge_list), size=2, replace=False)
        (a, b) = edge_list[eidx[0]]
        (c, d) = edge_list[eidx[1]]
        if len({a, b, c, d}) < 4:
            continue

        # Swap endpoints: (a,b),(c,d) -> (a,d),(c,b)
        new1 = tuple(sorted((a, d)))
        new2 = tuple(sorted((c, b)))
        if new1[0] == new1[1] or new2[0] == new2[1]:
            continue
        if new1 in edges or new2 in edges:
            continue

        edges.remove((a, b))
        edges.remove((c, d))
        edges.add(new1)
        edges.add(new2)
        edge_list[eidx[0]] = new1
        edge_list[eidx[1]] = new2

    edge_arr = np.array(sorted(edges), dtype=int)
    new_iu = edge_arr[:, 0]
    new_ju = edge_arr[:, 1]
    new_g0u = rng.permutation(g0u.copy())
    return new_iu, new_ju, new_g0u


def directed_edges(iu: np.ndarray, ju: np.ndarray, wu: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.concatenate([iu, ju]).astype(int)
    dst = np.concatenate([ju, iu]).astype(int)
    w = np.concatenate([wu, wu]).astype(float)
    return src, dst, w


# ============================================================
# Masses, frequencies, and edge weights
# ============================================================

def sample_masses(rng: np.random.Generator, N: int, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.ones(N, dtype=float)
    z = rng.normal(0.0, 1.0, size=N)
    m = np.exp(sigma * z)
    return m / np.mean(m)


def edge_weights_from_kernel(
    iu: np.ndarray,
    ju: np.ndarray,
    g0u: np.ndarray,
    kernel_mass: np.ndarray,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    delta = np.abs(np.log(kernel_mass[iu] / kernel_mass[ju]))
    wu = g0u * np.exp(-beta * delta)
    return wu.astype(float), delta.astype(float)


def edge_mismatch_exposure(g0u: np.ndarray, delta: np.ndarray) -> float:
    return float(np.sum(g0u * delta) / (np.sum(g0u) + 1e-12))


def natural_frequencies(
    omega_mass: np.ndarray,
    omega_active: bool,
    cfg: FiaV4Config,
    omega_noise_vec: np.ndarray,
) -> np.ndarray:
    if not omega_active:
        return np.full(len(omega_mass), cfg.omega0)
    return (
        cfg.omega0
        + cfg.omega_mass_slope * (omega_mass - 1.0)
        + cfg.omega_quenched_noise * omega_noise_vec
    )


# ============================================================
# Spectral quantities
# ============================================================

def matrix_from_edges(N: int, iu: np.ndarray, ju: np.ndarray, wu: np.ndarray) -> np.ndarray:
    W = np.zeros((N, N), dtype=float)
    W[iu, ju] = wu
    W[ju, iu] = wu
    np.fill_diagonal(W, 0.0)
    return W


def spectral_gap_weighted_matrix(W: np.ndarray, mass: np.ndarray) -> float:
    n = len(mass)
    if n < 2 or W.size == 0:
        return 0.0
    W = 0.5 * (W + W.T)
    degree = np.sum(W, axis=1)
    L = np.diag(degree) - W
    inv_sqrt = 1.0 / np.sqrt(mass)
    S = (inv_sqrt[:, None] * L) * inv_sqrt[None, :]
    S = 0.5 * (S + S.T)
    eigvals = np.linalg.eigvalsh(S)
    eigvals = np.sort(np.maximum(eigvals, 0.0))
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0


def ordinary_gap_matrix(W: np.ndarray) -> float:
    n = W.shape[0]
    if n < 2:
        return 0.0
    W = 0.5 * (W + W.T)
    L = np.diag(np.sum(W, axis=1)) - W
    eigvals = np.linalg.eigvalsh(0.5 * (L + L.T))
    eigvals = np.sort(np.maximum(eigvals, 0.0))
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0


def mass_range_bound_check(W: np.ndarray, mass: np.ndarray) -> Dict[str, float]:
    C_M = spectral_gap_weighted_matrix(W, mass)
    lam2 = ordinary_gap_matrix(W)
    lower = lam2 / (np.max(mass) + 1e-12)
    upper = lam2 / (np.min(mass) + 1e-12)
    return {
        "C_M": C_M,
        "lambda2_L": lam2,
        "bound_lower": float(lower),
        "bound_upper": float(upper),
        "bound_passed": float(lower - 1e-9 <= C_M <= upper + 1e-9),
    }


def signed_laplacian_min_eig(N: int, iu: np.ndarray, ju: np.ndarray, signed_edges: np.ndarray) -> float:
    W = matrix_from_edges(N, iu, ju, signed_edges)
    L = np.diag(np.sum(W, axis=1)) - W
    vals = np.linalg.eigvalsh(0.5 * (L + L.T))
    return float(np.min(vals)) if len(vals) else 0.0


# ============================================================
# Dynamics
# ============================================================

def phase_update(
    theta: np.ndarray,
    omega: np.ndarray,
    capacity_mass: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    wdir: np.ndarray,
    cfg: FiaV4Config,
    d_ref: float,
    degree_A: np.ndarray,
    noise_vec: np.ndarray,
) -> np.ndarray:
    diff = theta[dst] - theta[src]
    raw = np.bincount(src, weights=wdir * np.sin(diff), minlength=len(theta))

    if cfg.normalization == "local_degree":
        denom = capacity_mass * (degree_A + 1e-12)
    elif cfg.normalization == "global":
        denom = capacity_mass * (d_ref + 1e-12)
    else:
        raise ValueError(f"Unknown normalization: {cfg.normalization}")

    deterministic = omega + cfg.K * raw / denom
    noise = math.sqrt(cfg.dt) * cfg.phase_noise * noise_vec
    return np.mod(theta + cfg.dt * deterministic + noise, 2.0 * np.pi)


# ============================================================
# Diagnostics: score, metric, hobs, persistence
# ============================================================

def final_edge_diagnostics(
    theta: np.ndarray,
    iu: np.ndarray,
    ju: np.ndarray,
    wu: np.ndarray,
    A_ref: float,
    cfg: FiaV4Config,
    d_ref: float,
) -> Dict[str, np.ndarray]:
    phase_diff = theta[iu] - theta[ju]
    cosv = np.cos(phase_diff)
    coherence_score = 0.5 * (1.0 + cosv)
    positive_stiffness = np.maximum(cosv, 0.0)

    score_edges = np.clip(wu / (A_ref + 1e-12), 0.0, 1.0) * coherence_score
    metric_edges = (cfg.K / (d_ref + 1e-12)) * wu * positive_stiffness
    signed_edges = (cfg.K / (d_ref + 1e-12)) * wu * cosv
    A_edges = (cfg.K / (d_ref + 1e-12)) * wu
    old_edges = np.clip(wu / (np.max(wu) + 1e-12), 0.0, 1.0) * coherence_score

    return {
        "score_edges": score_edges,
        "metric_edges": metric_edges,
        "signed_edges": signed_edges,
        "A_edges": A_edges,
        "old_edges": old_edges,
        "coherence_score": coherence_score,
        "positive_stiffness": positive_stiffness,
        "cosv": cosv,
    }


def components_from_edges(
    N: int,
    iu: np.ndarray,
    ju: np.ndarray,
    score_edges: np.ndarray,
    threshold: float,
) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(N)]
    active = score_edges > threshold
    for i, j in zip(iu[active], ju[active]):
        ii = int(i)
        jj = int(j)
        adj[ii].append(jj)
        adj[jj].append(ii)

    visited = np.zeros(N, dtype=bool)
    comps: List[List[int]] = []
    for start in range(N):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp: List[int] = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        comps.append(comp)
    return comps


def candidate_hobs_with_thresholds(
    theta: np.ndarray,
    mass: np.ndarray,
    iu: np.ndarray,
    ju: np.ndarray,
    score_edges: np.ndarray,
    metric_edges: np.ndarray,
    min_hob_size: int,
    score_threshold: float,
    R_threshold: float,
    C_threshold: float,
) -> List[Dict[str, object]]:
    N = len(theta)
    comps = components_from_edges(N, iu, ju, score_edges, score_threshold)
    W_metric_full = matrix_from_edges(N, iu, ju, metric_edges)

    candidates: List[Dict[str, object]] = []
    for comp in comps:
        if len(comp) < min_hob_size:
            continue
        idx = np.array(comp, dtype=int)
        R_H = order_parameter(theta, idx)
        if R_H < R_threshold:
            continue
        W_sub = W_metric_full[np.ix_(idx, idx)]
        C_H = spectral_gap_weighted_matrix(W_sub, mass[idx])
        if C_H < C_threshold:
            continue
        candidates.append({
            "nodes": tuple(int(x) for x in idx.tolist()),
            "size": int(len(idx)),
            "R": float(R_H),
            "C": float(C_H),
        })
    return candidates


def candidate_hobs(theta, mass, iu, ju, score_edges, metric_edges, cfg: FiaV4Config):
    return candidate_hobs_with_thresholds(
        theta, mass, iu, ju, score_edges, metric_edges,
        cfg.min_hob_size, cfg.score_threshold, cfg.R_threshold, cfg.C_threshold
    )


class HobTracker:
    def __init__(self, jaccard_threshold: float):
        self.jaccard_threshold = jaccard_threshold
        self.tracks: List[Dict[str, object]] = []
        self.next_id = 0

    def update(self, frame: int, candidates: List[Dict[str, object]]) -> None:
        active_indices = [i for i, tr in enumerate(self.tracks) if tr["last_frame"] == frame - 1]
        assigned_tracks = set()
        assigned_candidates = set()

        pairs: List[Tuple[float, int, int]] = []
        for ti in active_indices:
            old_nodes = self.tracks[ti]["nodes"]
            for ci, cand in enumerate(candidates):
                score = jaccard(old_nodes, cand["nodes"])
                if score >= self.jaccard_threshold:
                    pairs.append((score, ti, ci))
        pairs.sort(reverse=True, key=lambda x: x[0])

        for score, ti, ci in pairs:
            if ti in assigned_tracks or ci in assigned_candidates:
                continue
            cand = candidates[ci]
            tr = self.tracks[ti]
            tr["nodes"] = cand["nodes"]
            tr["last_frame"] = frame
            tr["lifetime_frames"] = int(tr["lifetime_frames"]) + 1
            tr["max_size"] = max(int(tr["max_size"]), int(cand["size"]))
            tr["R_values"].append(float(cand["R"]))
            tr["C_values"].append(float(cand["C"]))
            tr["jaccard_values"].append(float(score))
            assigned_tracks.add(ti)
            assigned_candidates.add(ci)

        for ci, cand in enumerate(candidates):
            if ci in assigned_candidates:
                continue
            self.tracks.append({
                "id": self.next_id,
                "nodes": cand["nodes"],
                "birth_frame": frame,
                "last_frame": frame,
                "lifetime_frames": 1,
                "max_size": int(cand["size"]),
                "R_values": [float(cand["R"])],
                "C_values": [float(cand["C"])],
                "jaccard_values": [],
            })
            self.next_id += 1

    def persistent_tracks(self, min_lifetime_frames: int) -> List[Dict[str, object]]:
        return [tr for tr in self.tracks if int(tr["lifetime_frames"]) >= min_lifetime_frames]

    def summary(self, min_lifetime_frames: int) -> Dict[str, float]:
        pts = self.persistent_tracks(min_lifetime_frames)
        if not pts:
            return {
                "persistent_n": 0,
                "persistent_units_max": 0,
                "persistent_lifetime_max": 0,
                "persistent_mean_C": 0.0,
                "persistent_mean_R": 0.0,
                "persistent_mean_jaccard": 0.0,
            }
        mean_C, mean_R, mean_J = [], [], []
        for tr in pts:
            mean_C.append(float(np.mean(tr["C_values"])))
            mean_R.append(float(np.mean(tr["R_values"])))
            js = tr["jaccard_values"]
            mean_J.append(float(np.mean(js)) if js else 0.0)
        return {
            "persistent_n": int(len(pts)),
            "persistent_units_max": int(max(int(tr["max_size"]) for tr in pts)),
            "persistent_lifetime_max": int(max(int(tr["lifetime_frames"]) for tr in pts)),
            "persistent_mean_C": float(np.mean(mean_C)),
            "persistent_mean_R": float(np.mean(mean_R)),
            "persistent_mean_jaccard": float(np.mean(mean_J)),
        }


# ============================================================
# Controls
# ============================================================

CONTROL_CHANNELS = {
    # kernel, capacity, omega, beta_zero, shuffle_kernel, shuffle_capacity, shuffle_omega,
    # random_graph, degree_rewire, static_phase
    "equal_identical": (False, False, False, False, False, False, False, False, False, False),
    "kernel_only": (True, False, False, False, False, False, False, False, False, False),
    "capacity_only": (False, True, False, False, False, False, False, False, False, False),
    "omega_only": (False, False, True, False, False, False, False, False, False, False),
    "kernel_capacity": (True, True, False, False, False, False, False, False, False, False),
    "kernel_omega": (True, False, True, False, False, False, False, False, False, False),
    "capacity_omega": (False, True, True, False, False, False, False, False, False, False),
    "full": (True, True, True, False, False, False, False, False, False, False),
    "beta0_full": (True, True, True, True, False, False, False, False, False, False),
    "shuffle_kernel_fixed": (True, True, True, False, True, False, False, False, False, False),
    "shuffle_capacity_fixed": (True, True, True, False, False, True, False, False, False, False),
    "shuffle_omega_fixed": (True, True, True, False, False, False, True, False, False, False),
    "static_phase": (True, True, True, False, False, False, False, False, False, True),
    "random_graph_full": (True, True, True, False, False, False, False, True, False, False),
    "degree_preserving_rewire_full": (True, True, True, False, False, False, False, False, True, False),
}


def prepare_control(control: str, base: Dict[str, object], beta: float) -> Dict[str, object]:
    if control not in CONTROL_CHANNELS:
        raise ValueError(f"Unknown control: {control}")

    (
        use_kernel, use_capacity, use_omega, beta_zero,
        shuffle_kernel, shuffle_capacity, shuffle_omega,
        random_graph, degree_rewire, static_phase
    ) = CONTROL_CHANNELS[control]

    base_mass = np.asarray(base["mass"], dtype=float)
    ones = np.ones_like(base_mass)

    kernel_mass = base_mass.copy() if use_kernel else ones.copy()
    capacity_mass = base_mass.copy() if use_capacity else ones.copy()
    omega_mass = base_mass.copy() if use_omega else ones.copy()

    if shuffle_kernel:
        kernel_mass = kernel_mass[np.asarray(base["perm_kernel"], dtype=int)]
    if shuffle_capacity:
        capacity_mass = capacity_mass[np.asarray(base["perm_capacity"], dtype=int)]
    if shuffle_omega:
        omega_mass = omega_mass[np.asarray(base["perm_omega"], dtype=int)]

    return {
        "kernel_mass": kernel_mass,
        "capacity_mass": capacity_mass,
        "omega_mass": omega_mass,
        "omega_active": bool(use_omega),
        "beta_eff": 0.0 if beta_zero else float(beta),
        "use_random_graph": bool(random_graph),
        "use_degree_rewire": bool(degree_rewire),
        "is_static_phase": bool(static_phase),
    }


# ============================================================
# Simulation of one condition
# ============================================================

def compute_snapshot_metrics(
    theta: np.ndarray,
    mass: np.ndarray,
    iu: np.ndarray,
    ju: np.ndarray,
    g0u: np.ndarray,
    wu: np.ndarray,
    A_ref: float,
    cfg: FiaV4Config,
    d_ref: float,
) -> Tuple[Dict[str, float], List[Dict[str, object]], Dict[str, np.ndarray]]:
    diag = final_edge_diagnostics(theta, iu, ju, wu, A_ref, cfg, d_ref)
    N = len(theta)

    W_A = matrix_from_edges(N, iu, ju, diag["A_edges"])
    W_metric = matrix_from_edges(N, iu, ju, diag["metric_edges"])
    W_score = matrix_from_edges(N, iu, ju, diag["score_edges"])
    W_old = matrix_from_edges(N, iu, ju, diag["old_edges"])

    C_A = spectral_gap_weighted_matrix(W_A, mass)
    C_metric = spectral_gap_weighted_matrix(W_metric, mass)
    C_score = spectral_gap_weighted_matrix(W_score, mass)
    C_old = spectral_gap_weighted_matrix(W_old, mass)

    candidates = candidate_hobs(theta, mass, iu, ju, diag["score_edges"], diag["metric_edges"], cfg)

    if candidates:
        sizes = [int(c["size"]) for c in candidates]
        Rs = [float(c["R"]) for c in candidates]
        Cs = [float(c["C"]) for c in candidates]
        candidate_summary = {
            "candidate_n": int(len(candidates)),
            "candidate_units": int(np.sum(sizes)),
            "candidate_max_size": int(np.max(sizes)),
            "candidate_mean_R": float(np.mean(Rs)),
            "candidate_mean_C": float(np.mean(Cs)),
        }
    else:
        candidate_summary = {
            "candidate_n": 0,
            "candidate_units": 0,
            "candidate_max_size": 0,
            "candidate_mean_R": 0.0,
            "candidate_mean_C": 0.0,
        }

    bound = mass_range_bound_check(W_A, mass)

    neg_frac = float(np.mean(diag["cosv"] < 0.0)) if len(diag["cosv"]) else 0.0
    signed_min = signed_laplacian_min_eig(N, iu, ju, diag["signed_edges"])

    metrics = {
        "R_global": order_parameter(theta),
        "C_A_full": C_A,
        "C_metric_full": C_metric,
        "C_score_full": C_score,
        "C_old_full": C_old,
        "C_metric_over_C_A_full": safe_ratio(C_metric, C_A),
        "C_score_over_C_A_full": safe_ratio(C_score, C_A),
        "mean_score_edge": float(np.mean(diag["score_edges"])) if len(diag["score_edges"]) else 0.0,
        "mean_metric_edge": float(np.mean(diag["metric_edges"])) if len(diag["metric_edges"]) else 0.0,
        "mean_positive_stiffness": float(np.mean(diag["positive_stiffness"])) if len(diag["positive_stiffness"]) else 0.0,
        "negative_stiffness_fraction": neg_frac,
        "signed_laplacian_min_eig": signed_min,
        **candidate_summary,
        "bound_lower_A": bound["bound_lower"],
        "bound_upper_A": bound["bound_upper"],
        "bound_passed_A": bound["bound_passed"],
    }
    return metrics, candidates, diag


def threshold_robustness_rows(
    cfg: FiaV4Config,
    seed: int,
    sigma: float,
    beta: float,
    control: str,
    theta: np.ndarray,
    mass: np.ndarray,
    iu: np.ndarray,
    ju: np.ndarray,
    score_edges: np.ndarray,
    metric_edges: np.ndarray,
) -> List[Dict[str, float]]:
    rows = []
    for score_thr in cfg.score_thresholds:
        for R_thr in cfg.R_thresholds:
            for C_thr in cfg.C_thresholds:
                cands = candidate_hobs_with_thresholds(
                    theta=theta, mass=mass, iu=iu, ju=ju,
                    score_edges=score_edges, metric_edges=metric_edges,
                    min_hob_size=cfg.min_hob_size,
                    score_threshold=score_thr,
                    R_threshold=R_thr,
                    C_threshold=C_thr,
                )
                rows.append({
                    "seed": seed,
                    "sigma": sigma,
                    "beta": beta,
                    "control": control,
                    "score_threshold": score_thr,
                    "R_threshold": R_thr,
                    "C_threshold": C_thr,
                    "candidate_n": len(cands),
                    "candidate_units": int(sum(int(c["size"]) for c in cands)) if cands else 0,
                    "candidate_max_size": int(max([int(c["size"]) for c in cands], default=0)),
                    "candidate_mean_C": float(np.mean([float(c["C"]) for c in cands])) if cands else 0.0,
                })
    return rows


def run_one_condition(
    cfg: FiaV4Config,
    seed: int,
    sigma: float,
    beta: float,
    control: str,
    base: Dict[str, object],
) -> Tuple[Dict[str, float], List[Dict[str, float]], List[Dict[str, object]], List[Dict[str, float]]]:
    ctrl = prepare_control(control, base, beta)

    if ctrl["use_random_graph"]:
        iu, ju, g0u = base["random_graph"]
    elif ctrl["use_degree_rewire"]:
        iu, ju, g0u = base["rewired_graph"]
    else:
        iu, ju, g0u = base["iu"], base["ju"], base["g0u"]

    capacity_mass = np.asarray(ctrl["capacity_mass"], dtype=float)
    kernel_mass = np.asarray(ctrl["kernel_mass"], dtype=float)
    omega_mass = np.asarray(ctrl["omega_mass"], dtype=float)
    beta_eff = float(ctrl["beta_eff"])

    wu, delta = edge_weights_from_kernel(iu, ju, g0u, kernel_mass, beta_eff)
    src, dst, wdir = directed_edges(iu, ju, wu)

    d_ref = max(float(2.0 * np.sum(g0u) / cfg.N), cfg.baseline_weight_floor)
    A_ref = max(float(np.quantile(g0u, cfg.score_ref_quantile)), cfg.baseline_weight_floor)
    degree_A = np.bincount(src, weights=wdir, minlength=cfg.N)

    theta = np.array(base["theta0"], dtype=float).copy()
    omega = natural_frequencies(
        omega_mass=omega_mass,
        omega_active=bool(ctrl["omega_active"]),
        cfg=cfg,
        omega_noise_vec=base["omega_noise_vec"],
    )

    tracker = HobTracker(cfg.jaccard_threshold)
    timeseries_rows: List[Dict[str, float]] = []
    threshold_rows: List[Dict[str, float]] = []
    track_rows: List[Dict[str, object]] = []

    obs_frame = 0

    def observe(step: int, update_persistence: bool) -> Dict[str, np.ndarray]:
        nonlocal obs_frame
        metrics, cands, diag = compute_snapshot_metrics(theta, capacity_mass, iu, ju, g0u, wu, A_ref, cfg, d_ref)
        if update_persistence:
            tracker.update(obs_frame, cands)
            obs_frame += 1

        timeseries_rows.append({
            "seed": seed,
            "sigma": sigma,
            "beta": beta,
            "beta_eff": beta_eff,
            "control": control,
            "step": step,
            "time": step * cfg.dt,
            "obs_frame": obs_frame if update_persistence else -1,
            "post_burn_in": int(update_persistence),
            **metrics,
        })
        return diag

    last_diag = None

    if bool(ctrl["is_static_phase"]):
        last_diag = observe(0, update_persistence=False)
    else:
        for step in range(cfg.steps + 1):
            if step % cfg.detect_every == 0 or step == cfg.steps:
                update_persistence = step >= cfg.burn_in_steps
                last_diag = observe(step, update_persistence=update_persistence)
            if step < cfg.steps:
                theta = phase_update(
                    theta=theta,
                    omega=omega,
                    capacity_mass=capacity_mass,
                    src=src,
                    dst=dst,
                    wdir=wdir,
                    cfg=cfg,
                    d_ref=d_ref,
                    degree_A=degree_A,
                    noise_vec=base["noise_path"][step],
                )

    # Final threshold robustness.
    if last_diag is not None:
        threshold_rows.extend(threshold_robustness_rows(
            cfg, seed, sigma, beta, control, theta, capacity_mass, iu, ju,
            last_diag["score_edges"], last_diag["metric_edges"],
        ))

    final_metrics = timeseries_rows[-1].copy()
    persistent_summary = tracker.summary(cfg.min_lifetime_frames)

    row = {
        "seed": seed,
        "sigma": sigma,
        "beta": beta,
        "beta_eff": beta_eff,
        "control": control,
        "N": cfg.N,
        "steps": cfg.steps,
        "dt": cfg.dt,
        "normalization": cfg.normalization,
        "persistence_applicable": int(not bool(ctrl["is_static_phase"])),
        "D_m_capacity": mass_diversity(capacity_mass),
        "D_m_kernel": mass_diversity(kernel_mass),
        "D_m_omega_mass": mass_diversity(omega_mass),
        "D_omega": vector_diversity(omega),
        "edge_mismatch_exposure": edge_mismatch_exposure(g0u, delta),
        "mean_A_over_G0": float(np.sum(wu) / (np.sum(g0u) + 1e-12)),
        "R_global_final": final_metrics["R_global"],
        "C_A_full": final_metrics["C_A_full"],
        "C_metric_full": final_metrics["C_metric_full"],
        "C_score_full": final_metrics["C_score_full"],
        "C_old_full": final_metrics["C_old_full"],
        "C_metric_over_C_A_full": final_metrics["C_metric_over_C_A_full"],
        "C_score_over_C_A_full": final_metrics["C_score_over_C_A_full"],
        "mean_score_edge_final": final_metrics["mean_score_edge"],
        "mean_metric_edge_final": final_metrics["mean_metric_edge"],
        "mean_positive_stiffness_final": final_metrics["mean_positive_stiffness"],
        "negative_stiffness_fraction": final_metrics["negative_stiffness_fraction"],
        "signed_laplacian_min_eig": final_metrics["signed_laplacian_min_eig"],
        "candidate_n_final": final_metrics["candidate_n"],
        "candidate_units_final": final_metrics["candidate_units"],
        "candidate_max_size_final": final_metrics["candidate_max_size"],
        "candidate_mean_R_final": final_metrics["candidate_mean_R"],
        "candidate_mean_C_final": final_metrics["candidate_mean_C"],
        "bound_lower_A": final_metrics["bound_lower_A"],
        "bound_upper_A": final_metrics["bound_upper_A"],
        "bound_passed_A": final_metrics["bound_passed_A"],
        **persistent_summary,
    }

    for tr in tracker.tracks:
        track_rows.append({
            "seed": seed,
            "sigma": sigma,
            "beta": beta,
            "beta_eff": beta_eff,
            "control": control,
            "track_id": int(tr["id"]),
            "birth_frame": int(tr["birth_frame"]),
            "last_frame": int(tr["last_frame"]),
            "lifetime_frames": int(tr["lifetime_frames"]),
            "max_size": int(tr["max_size"]),
            "mean_R": float(np.mean(tr["R_values"])),
            "mean_C": float(np.mean(tr["C_values"])),
            "mean_jaccard": float(np.mean(tr["jaccard_values"])) if tr["jaccard_values"] else 0.0,
            "is_persistent": int(int(tr["lifetime_frames"]) >= cfg.min_lifetime_frames),
        })

    return row, timeseries_rows, track_rows, threshold_rows


# ============================================================
# Experiment helpers
# ============================================================

def make_base_for_seed(cfg: FiaV4Config, seed: int, sigma: float) -> Dict[str, object]:
    rng = np.random.default_rng(cfg.seed0 + seed)
    iu, ju, g0u, G0, pos = build_spatial_baseline_graph(
        rng=rng,
        N=cfg.N,
        spatial_dim=cfg.spatial_dim,
        k_nn=cfg.k_nn,
        sigma_space=cfg.sigma_space,
        random_edges_per_node=cfg.random_edges_per_node,
    )

    mass_rng = np.random.default_rng(cfg.seed0 + 10000 + seed * 100 + int(1000 * sigma))
    theta_rng = np.random.default_rng(cfg.seed0 + 20000 + seed)
    omega_rng = np.random.default_rng(cfg.seed0 + 30000 + seed)
    noise_rng = np.random.default_rng(cfg.seed0 + 40000 + seed)

    perm_rng = np.random.default_rng(cfg.seed0 + 50000 + seed * 100 + int(1000 * sigma))
    graph_rng = np.random.default_rng(cfg.seed0 + 60000 + seed)
    rewire_rng = np.random.default_rng(cfg.seed0 + 70000 + seed)

    random_graph = randomize_graph_edges(graph_rng, cfg.N, edge_count=len(iu), weights=g0u)
    rewired_graph = degree_preserving_rewire(
        rewire_rng, cfg.N, iu, ju, g0u,
        swaps_per_edge=cfg.rewire_swaps_per_edge,
    )

    return {
        "iu": iu,
        "ju": ju,
        "g0u": g0u,
        "G0": G0,
        "pos": pos,
        "mass": sample_masses(mass_rng, cfg.N, sigma),
        "theta0": theta_rng.uniform(0.0, 2.0 * np.pi, size=cfg.N),
        "omega_noise_vec": omega_rng.normal(0.0, 1.0, size=cfg.N),
        "noise_path": noise_rng.normal(0.0, 1.0, size=(cfg.steps, cfg.N)),
        "perm_kernel": perm_rng.permutation(cfg.N),
        "perm_capacity": perm_rng.permutation(cfg.N),
        "perm_omega": perm_rng.permutation(cfg.N),
        "random_graph": random_graph,
        "rewired_graph": rewired_graph,
    }


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "D_m_capacity": ["mean"],
        "D_m_kernel": ["mean"],
        "D_m_omega_mass": ["mean"],
        "D_omega": ["mean"],
        "edge_mismatch_exposure": ["mean"],
        "mean_A_over_G0": ["mean", "std"],
        "R_global_final": ["mean", "std"],
        "C_A_full": ["mean", "std"],
        "C_metric_full": ["mean", "std"],
        "C_metric_over_C_A_full": ["mean", "std"],
        "negative_stiffness_fraction": ["mean", "std"],
        "signed_laplacian_min_eig": ["mean", "std"],
        "candidate_n_final": ["mean", "std"],
        "candidate_units_final": ["mean", "std"],
        "candidate_mean_R_final": ["mean"],
        "candidate_mean_C_final": ["mean"],
        "persistent_n": ["mean", "std"],
        "persistent_units_max": ["mean", "std"],
        "persistent_lifetime_max": ["mean", "std"],
        "persistent_mean_C": ["mean"],
        "persistent_mean_R": ["mean"],
        "bound_passed_A": ["min", "mean"],
    }
    summary = df.groupby(["sigma", "beta", "control"]).agg(agg_dict)
    summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    counts = df.groupby(["sigma", "beta", "control"]).size().reset_index(name="n")
    summary = summary.merge(counts, on=["sigma", "beta", "control"], how="left")
    for metric in ["C_metric_full", "C_A_full", "persistent_n", "candidate_units_final"]:
        std_col = f"{metric}_std"
        if std_col in summary.columns:
            summary[f"{metric}_se"] = summary[std_col] / np.sqrt(summary["n"].clip(lower=1))
    return summary


def paired_differences(df: pd.DataFrame, baseline_control: str = "full") -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for (seed, sigma, beta), sub in df.groupby(["seed", "sigma", "beta"]):
        base = sub[sub["control"] == baseline_control]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for _, ctrl_row in sub.iterrows():
            control = ctrl_row["control"]
            if control == baseline_control:
                continue
            row = {"seed": seed, "sigma": sigma, "beta": beta, "baseline": baseline_control, "control": control}
            for metric in KEY_METRICS:
                if metric not in df.columns:
                    continue
                b = float(base_row[metric])
                c = float(ctrl_row[metric])
                row[f"diff_{metric}_full_minus_control"] = b - c
                row[f"ratio_{metric}_full_over_control"] = safe_ratio(b, c)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_paired_differences(diff_df: pd.DataFrame, cfg: FiaV4Config) -> pd.DataFrame:
    if diff_df.empty:
        return pd.DataFrame()

    agg_cols = [c for c in diff_df.columns if c.startswith("diff_") or c.startswith("ratio_")]
    summary = diff_df.groupby(["sigma", "beta", "control"])[agg_cols].agg(["mean", "std", "median"])
    summary.columns = ["_".join([str(part) for part in col if part]) for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    counts = diff_df.groupby(["sigma", "beta", "control"]).size().reset_index(name="n")
    summary = summary.merge(counts, on=["sigma", "beta", "control"], how="left")

    # Bootstrap CI for primary endpoint and two secondary persistence endpoints.
    ci_rows = []
    ci_cols = [
        "diff_C_metric_full_full_minus_control",
        "diff_persistent_n_full_minus_control",
        "diff_persistent_units_max_full_minus_control",
    ]
    for (sigma, beta, control), sub in diff_df.groupby(["sigma", "beta", "control"]):
        row = {"sigma": sigma, "beta": beta, "control": control}
        for col in ci_cols:
            if col not in sub.columns:
                continue
            lo, hi = bootstrap_ci_mean(
                sub[col].to_numpy(),
                n_boot=cfg.bootstrap_n,
                seed=cfg.bootstrap_seed + int(1000 * sigma) + int(100 * beta) + stable_label_int(control, 10000),
            )
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
        ci_rows.append(row)
    ci_df = pd.DataFrame(ci_rows)
    summary = summary.merge(ci_df, on=["sigma", "beta", "control"], how="left")

    # Paired sign-flip p-values for the same core endpoints.
    p_rows = []
    for (sigma, beta, control), sub in diff_df.groupby(["sigma", "beta", "control"]):
        row = {"sigma": sigma, "beta": beta, "control": control}
        for col in ci_cols:
            if col not in sub.columns:
                continue
            row[f"{col}_p_signflip"] = sign_flip_pvalue_mean(
                sub[col].to_numpy(),
                n_perm=max(200, cfg.bootstrap_n),
                seed=cfg.bootstrap_seed + 900_000 + int(1000 * sigma) + int(100 * beta) + stable_label_int(control, 10000),
            )
        p_rows.append(row)
    p_df = pd.DataFrame(p_rows)
    summary = summary.merge(p_df, on=["sigma", "beta", "control"], how="left")
    return summary


def threshold_robustness_summary(threshold_df: pd.DataFrame) -> pd.DataFrame:
    if threshold_df.empty:
        return pd.DataFrame()
    return (
        threshold_df.groupby(["sigma", "beta", "control", "score_threshold", "R_threshold", "C_threshold"])
        .agg(
            candidate_n_mean=("candidate_n", "mean"),
            candidate_units_mean=("candidate_units", "mean"),
            candidate_max_size_mean=("candidate_max_size", "mean"),
            candidate_mean_C_mean=("candidate_mean_C", "mean"),
        )
        .reset_index()
    )


def theory_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for (sigma, beta, control), sub in df.groupby(["sigma", "beta", "control"]):
        rows.append({
            "check": "mass_range_bound_A",
            "sigma": sigma,
            "beta": beta,
            "control": control,
            "passed_fraction": float(np.mean(sub["bound_passed_A"])),
            "failures": int(np.sum(sub["bound_passed_A"] < 0.5)),
        })

    # Monotonicity check, valid where graph/permutation is fixed over beta.
    for control in ["full", "static_phase", "shuffle_kernel_fixed", "shuffle_capacity_fixed",
                    "shuffle_omega_fixed", "degree_preserving_rewire_full", "random_graph_full"]:
        for (seed, sigma), sub in df[df["control"] == control].groupby(["seed", "sigma"]):
            ss = sub.sort_values("beta")
            vals = ss["C_A_full"].to_numpy(dtype=float)
            betas = ss["beta"].to_numpy(dtype=float)
            failures = int(np.sum(np.diff(vals) > 1e-8))
            rows.append({
                "check": "C_A_nonincreasing_in_beta",
                "sigma": sigma,
                "beta": float("nan"),
                "control": control,
                "seed": seed,
                "passed_fraction": float(1.0 if failures == 0 else 0.0),
                "failures": failures,
                "beta_min": float(np.min(betas)),
                "beta_max": float(np.max(betas)),
            })
    return pd.DataFrame(rows)


def sign_flip_pvalue_mean(x: np.ndarray, n_perm: int, seed: int) -> float:
    """Two-sided paired randomization p-value for a mean difference.

    This does not assume normality. It tests whether paired signs could be
    arbitrarily flipped under the null of no directional effect.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan")
    observed = abs(float(np.mean(x)))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(max(1, n_perm)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n, replace=True)
        if abs(float(np.mean(signs * x))) >= observed - 1e-15:
            extreme += 1
    # Add-one correction.
    return float((extreme + 1) / (n_perm + 1))


FACTORIAL_CHANNELS = {
    "equal_identical": (0, 0, 0),
    "kernel_only": (1, 0, 0),
    "capacity_only": (0, 1, 0),
    "omega_only": (0, 0, 1),
    "kernel_capacity": (1, 1, 0),
    "kernel_omega": (1, 0, 1),
    "capacity_omega": (0, 1, 1),
    "full": (1, 1, 1),
}


FACTORIAL_METRICS = [
    "C_metric_full",
    "C_A_full",
    "C_metric_over_C_A_full",
    "persistent_n",
    "persistent_units_max",
    "negative_stiffness_fraction",
]


def factorial_channel_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate 2^3 kernel/capacity/omega channel effects per paired block.

    Effects are computed within each seed/sigma/beta block using the eight
    factorial controls. Main effects are mean(active)-mean(inactive). Two-way
    interactions are difference-in-differences averaged over the third channel.
    The three-way interaction is the standard 2x2x2 contrast.
    """
    rows: List[Dict[str, float]] = []
    required = set(FACTORIAL_CHANNELS)
    for (seed, sigma, beta), sub in df.groupby(["seed", "sigma", "beta"]):
        controls = set(sub["control"])
        if not required.issubset(controls):
            continue
        lookup = {row["control"]: row for _, row in sub.iterrows() if row["control"] in required}
        for metric in FACTORIAL_METRICS:
            if metric not in df.columns:
                continue
            y: Dict[Tuple[int, int, int], float] = {}
            for control, channels in FACTORIAL_CHANNELS.items():
                y[channels] = float(lookup[control][metric])
            vals = np.array(list(y.values()), dtype=float)
            row = {
                "seed": seed,
                "sigma": sigma,
                "beta": beta,
                "metric": metric,
                "baseline_equal": y[(0, 0, 0)],
                "full": y[(1, 1, 1)],
                "full_minus_equal": y[(1, 1, 1)] - y[(0, 0, 0)],
            }

            def mean_where(axis: int, value: int) -> float:
                return float(np.mean([v for k, v in y.items() if k[axis] == value]))

            row["kernel_main"] = mean_where(0, 1) - mean_where(0, 0)
            row["capacity_main"] = mean_where(1, 1) - mean_where(1, 0)
            row["omega_main"] = mean_where(2, 1) - mean_where(2, 0)

            # Difference-in-differences averaged over the remaining channel.
            kc = []
            ko = []
            co = []
            for o in (0, 1):
                kc.append(y[(1, 1, o)] - y[(1, 0, o)] - y[(0, 1, o)] + y[(0, 0, o)])
            for c in (0, 1):
                ko.append(y[(1, c, 1)] - y[(1, c, 0)] - y[(0, c, 1)] + y[(0, c, 0)])
            for k in (0, 1):
                co.append(y[(k, 1, 1)] - y[(k, 1, 0)] - y[(k, 0, 1)] + y[(k, 0, 0)])
            row["kernel_capacity_interaction"] = float(np.mean(kc))
            row["kernel_omega_interaction"] = float(np.mean(ko))
            row["capacity_omega_interaction"] = float(np.mean(co))
            row["three_way_interaction"] = (
                y[(1, 1, 1)] - y[(1, 1, 0)] - y[(1, 0, 1)] - y[(0, 1, 1)]
                + y[(1, 0, 0)] + y[(0, 1, 0)] + y[(0, 0, 1)] - y[(0, 0, 0)]
            )
            row["factorial_value_range"] = float(np.max(vals) - np.min(vals))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_factorial_effects(fx: pd.DataFrame) -> pd.DataFrame:
    if fx.empty:
        return pd.DataFrame()
    effect_cols = [
        "baseline_equal",
        "full",
        "full_minus_equal",
        "kernel_main",
        "capacity_main",
        "omega_main",
        "kernel_capacity_interaction",
        "kernel_omega_interaction",
        "capacity_omega_interaction",
        "three_way_interaction",
        "factorial_value_range",
    ]
    summary = fx.groupby(["sigma", "beta", "metric"])[effect_cols].agg(["mean", "std", "median"])
    summary.columns = ["_".join([str(part) for part in col if part]) for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    counts = fx.groupby(["sigma", "beta", "metric"]).size().reset_index(name="n")
    return summary.merge(counts, on=["sigma", "beta", "metric"], how="left")


def threshold_robustness_endpoint(threshold_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse threshold grid to per-run robustness endpoints."""
    if threshold_df.empty:
        return pd.DataFrame()
    grouped = threshold_df.groupby(["seed", "sigma", "beta", "control"])
    return grouped.agg(
        robustness_any_hob_fraction=("candidate_n", lambda s: float(np.mean(np.asarray(s) > 0))),
        robustness_units_mean=("candidate_units", "mean"),
        robustness_units_max=("candidate_units", "max"),
        robustness_hmax_mean=("candidate_max_size", "mean"),
        robustness_C_mean=("candidate_mean_C", "mean"),
    ).reset_index()


def summarize_threshold_robustness_endpoint(robust_df: pd.DataFrame) -> pd.DataFrame:
    if robust_df.empty:
        return pd.DataFrame()
    return robust_df.groupby(["sigma", "beta", "control"]).agg(
        robustness_any_hob_fraction_mean=("robustness_any_hob_fraction", "mean"),
        robustness_any_hob_fraction_std=("robustness_any_hob_fraction", "std"),
        robustness_units_mean_mean=("robustness_units_mean", "mean"),
        robustness_units_mean_std=("robustness_units_mean", "std"),
        robustness_units_max_mean=("robustness_units_max", "mean"),
        robustness_C_mean_mean=("robustness_C_mean", "mean"),
        n=("seed", "count"),
    ).reset_index()


def write_v41_decision_report(
    out: Path,
    cfg: FiaV4Config,
    summary: pd.DataFrame,
    diff_summary: pd.DataFrame,
    factorial_summary: pd.DataFrame,
    robustness_endpoint_summary: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    """Write a compact scientific decision report."""
    lines: List[str] = []
    lines.append("# FIA Scientific Model V4.1 — decision report\n")
    lines.append("## Configuration\n")
    lines.append(f"- N: {cfg.N}")
    lines.append(f"- steps: {cfg.steps}")
    lines.append(f"- reps: {cfg.n_reps}")
    lines.append(f"- mass_sigmas: {cfg.mass_sigmas}")
    lines.append(f"- betas: {cfg.betas}")
    lines.append(f"- controls: {len(cfg.controls)} controls")
    lines.append(f"- primary endpoint: `C_metric_full`\n")

    if not checks.empty:
        fail_total = int(checks.get("failures", pd.Series(dtype=float)).fillna(0).sum())
        lines.append("## Theory checks\n")
        lines.append(f"- Total theory-check failures: {fail_total}\n")

    if not summary.empty:
        max_sigma = float(max(summary["sigma"]))
        lines.append(f"## Compact full-model trend at max sigma = {max_sigma}\n")
        full = summary[(summary["sigma"] == max_sigma) & (summary["control"] == "full")].sort_values("beta")
        if not full.empty:
            cols = ["beta", "C_A_full_mean", "C_metric_full_mean", "C_metric_over_C_A_full_mean", "persistent_n_mean", "persistent_units_max_mean", "mean_A_over_G0_mean"]
            cols = [c for c in cols if c in full.columns]
            lines.append(full[cols].to_markdown(index=False))
            lines.append("")

    primary_col = "diff_C_metric_full_full_minus_control_mean"
    ci_lo_col = "diff_C_metric_full_full_minus_control_ci_low"
    ci_hi_col = "diff_C_metric_full_full_minus_control_ci_high"
    p_col = "diff_C_metric_full_full_minus_control_p_signflip"
    if not diff_summary.empty and primary_col in diff_summary.columns:
        max_sigma = float(max(diff_summary["sigma"]))
        max_beta = float(max(diff_summary["beta"]))
        sub = diff_summary[(diff_summary["sigma"] == max_sigma) & (diff_summary["beta"] == max_beta)].copy()
        lines.append(f"## Primary paired contrasts at sigma={max_sigma}, beta={max_beta}\n")
        keep = ["control", primary_col]
        for c in [ci_lo_col, ci_hi_col, p_col, "n"]:
            if c in sub.columns:
                keep.append(c)
        lines.append(sub[keep].sort_values(primary_col, ascending=False).to_markdown(index=False))
        lines.append("")

        lines.append("## Automatic interpretation of primary endpoint\n")
        for _, row in sub.iterrows():
            control = row["control"]
            mean = float(row[primary_col])
            lo = float(row[ci_lo_col]) if ci_lo_col in row and pd.notna(row[ci_lo_col]) else float("nan")
            hi = float(row[ci_hi_col]) if ci_hi_col in row and pd.notna(row[ci_hi_col]) else float("nan")
            if pd.notna(lo) and lo > 0:
                verdict = "full > control supported"
            elif pd.notna(hi) and hi < 0:
                verdict = "full < control supported"
            else:
                verdict = "inconclusive"
            lines.append(f"- `{control}`: mean diff={mean:.6g}, CI=[{lo:.6g}, {hi:.6g}] → **{verdict}**")
        lines.append("")

    if not factorial_summary.empty:
        lines.append("## Factorial decomposition: primary endpoint\n")
        fx = factorial_summary[factorial_summary["metric"] == "C_metric_full"].copy()
        if not fx.empty:
            max_sigma = float(max(fx["sigma"]))
            fx = fx[fx["sigma"] == max_sigma].sort_values("beta")
            cols = [
                "beta",
                "kernel_main_mean",
                "capacity_main_mean",
                "omega_main_mean",
                "kernel_capacity_interaction_mean",
                "kernel_omega_interaction_mean",
                "three_way_interaction_mean",
            ]
            cols = [c for c in cols if c in fx.columns]
            lines.append(fx[cols].to_markdown(index=False))
            lines.append("")

    if not robustness_endpoint_summary.empty:
        lines.append("## Threshold robustness endpoint\n")
        max_sigma = float(max(robustness_endpoint_summary["sigma"]))
        max_beta = float(max(robustness_endpoint_summary["beta"]))
        rb = robustness_endpoint_summary[(robustness_endpoint_summary["sigma"] == max_sigma) & (robustness_endpoint_summary["beta"] == max_beta)]
        cols = ["control", "robustness_any_hob_fraction_mean", "robustness_units_mean_mean", "robustness_units_max_mean"]
        cols = [c for c in cols if c in rb.columns]
        lines.append(f"Threshold-grid robustness at sigma={max_sigma}, beta={max_beta}:")
        lines.append(rb[cols].sort_values("robustness_any_hob_fraction_mean", ascending=False).to_markdown(index=False))
        lines.append("")

    lines.append("## Conservative scientific conclusion\n")
    lines.append("V4.1 should be read as a falsifiable testbench, not as proof of FIA. The strong claim is supported only when the `full` condition beats equal, beta0, shuffle, and graph null controls on the preregistered primary endpoint and persistent-hob endpoints with paired uncertainty estimates. If those contrasts are inconclusive, the supported conclusion is limited to mismatch-driven structural/dynamic fragmentation.\n")
    (out / "v41_decision_report.md").write_text("\n".join(lines), encoding="utf-8")



# ============================================================
# Plots
# ============================================================

def make_plots(out: Path, summary: pd.DataFrame, diff_summary: pd.DataFrame) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not available; skipping plots.")
        return
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    def plot_metric_vs_beta(metric_mean: str, ylabel: str, filename: str, sigma: Optional[float] = None):
        plt.figure(figsize=(8, 5))
        sig = sigma if sigma is not None else float(sorted(summary["sigma"].unique())[-1])
        sub = summary[summary["sigma"] == sig]
        controls = ["full", "equal_identical", "beta0_full", "kernel_only", "capacity_only",
                    "omega_only", "static_phase", "degree_preserving_rewire_full"]
        for control in controls:
            ss = sub[sub["control"] == control].sort_values("beta")
            if ss.empty or metric_mean not in ss.columns:
                continue
            plt.plot(ss["beta"], ss[metric_mean], marker="o", label=control)
        plt.xlabel(r"mismatch sensitivity $\beta$")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} (sigma={sig})")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()

    plot_metric_vs_beta("C_A_full_mean", r"structural gap $C_A$", "01_structural_gap_CA.png")
    plot_metric_vs_beta("C_metric_full_mean", r"stiffness-weighted gap $C_{metric}$", "02_metric_gap.png")
    plot_metric_vs_beta("C_metric_over_C_A_full_mean", r"dynamic realization $C_{metric}/C_A$", "03_metric_over_CA.png")
    plot_metric_vs_beta("persistent_n_mean", "persistent hob count", "04_persistent_hob_count.png")
    plot_metric_vs_beta("persistent_units_max_mean", "max persistent hob units", "05_persistent_hob_units.png")
    plot_metric_vs_beta("negative_stiffness_fraction_mean", "negative stiffness fraction", "06_negative_stiffness_fraction.png")

    if not diff_summary.empty:
        metric_col = "diff_C_metric_full_full_minus_control_mean"
        if metric_col in diff_summary.columns:
            plt.figure(figsize=(8, 5))
            sig = float(sorted(diff_summary["sigma"].unique())[-1])
            sub = diff_summary[diff_summary["sigma"] == sig]
            for control in sorted(sub["control"].unique()):
                ss = sub[sub["control"] == control].sort_values("beta")
                plt.plot(ss["beta"], ss[metric_col], marker="o", label=control)
            plt.axhline(0.0, linestyle="--")
            plt.xlabel(r"mismatch sensitivity $\beta$")
            plt.ylabel(r"$C^{full}_{metric}-C^{control}_{metric}$")
            plt.title(f"Paired full-minus-control metric gap (sigma={sig})")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(fig_dir / "07_full_minus_control_metric_gap.png", dpi=180)
            plt.close()


# ============================================================
# Runner
# ============================================================

def run_experiment(cfg: FiaV4Config) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "v41_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    t0 = time.time()
    result_rows: List[Dict[str, float]] = []
    timeseries_rows: List[Dict[str, float]] = []
    track_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, float]] = []

    total = cfg.n_reps * len(cfg.mass_sigmas) * len(cfg.betas) * len(cfg.controls)
    done = 0
    print("=== FIA Scientific Model V4.1 ===")
    print(f"Output: {out}")
    print(f"Conditions: {total}")

    for rep in range(1, cfg.n_reps + 1):
        seed = cfg.seed0 + rep
        for sigma in cfg.mass_sigmas:
            base = make_base_for_seed(cfg, seed, sigma)
            for beta in cfg.betas:
                for control in cfg.controls:
                    row, ts, tracks, thresh = run_one_condition(cfg, seed, sigma, beta, control, base)
                    result_rows.append(row)
                    timeseries_rows.extend(ts)
                    track_rows.extend(tracks)
                    threshold_rows.extend(thresh)
                    done += 1
                    if done % max(1, total // 10) == 0 or done == total:
                        print(f"  completed {done}/{total} conditions")

    df = pd.DataFrame(result_rows)
    ts_df = pd.DataFrame(timeseries_rows)
    tracks_df = pd.DataFrame(track_rows)
    thresh_df = pd.DataFrame(threshold_rows)

    df.to_csv(out / "v41_run_results.csv", index=False)
    ts_df.to_csv(out / "v41_timeseries.csv", index=False)
    tracks_df.to_csv(out / "v41_hob_tracks.csv", index=False)
    thresh_df.to_csv(out / "v41_threshold_robustness.csv", index=False)

    summary = summarize_results(df)
    summary.to_csv(out / "v41_summary.csv", index=False)

    diff_df = paired_differences(df)
    diff_df.to_csv(out / "v41_paired_differences.csv", index=False)
    diff_summary = summarize_paired_differences(diff_df, cfg)
    diff_summary.to_csv(out / "v41_paired_difference_summary.csv", index=False)

    factorial_df = factorial_channel_effects(df)
    factorial_df.to_csv(out / "v41_factorial_effects.csv", index=False)
    factorial_summary = summarize_factorial_effects(factorial_df)
    factorial_summary.to_csv(out / "v41_factorial_effect_summary.csv", index=False)

    thresh_summary = threshold_robustness_summary(thresh_df)
    thresh_summary.to_csv(out / "v41_threshold_robustness_summary.csv", index=False)

    robustness_endpoint = threshold_robustness_endpoint(thresh_df)
    robustness_endpoint.to_csv(out / "v41_threshold_robustness_endpoint.csv", index=False)
    robustness_endpoint_summary = summarize_threshold_robustness_endpoint(robustness_endpoint)
    robustness_endpoint_summary.to_csv(out / "v41_threshold_robustness_endpoint_summary.csv", index=False)

    checks = theory_checks(df)
    checks.to_csv(out / "v41_theory_checks.csv", index=False)

    write_v41_decision_report(out, cfg, summary, diff_summary, factorial_summary, robustness_endpoint_summary, checks)

    if cfg.make_plots:
        make_plots(out, summary, diff_summary)

    elapsed = time.time() - t0
    print("Done.")
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Wrote: {out / 'v41_run_results.csv'}")
    print(f"Wrote: {out / 'v41_summary.csv'}")
    print(f"Wrote: {out / 'v41_paired_difference_summary.csv'}")
    print(f"Wrote: {out / 'v41_factorial_effect_summary.csv'}")
    print(f"Wrote: {out / 'v41_decision_report.md'}")

    max_sigma = max(cfg.mass_sigmas)
    compact = summary[
        (summary["sigma"] == max_sigma)
        & (summary["control"].isin(["full", "equal_identical", "beta0_full", "kernel_only", "capacity_only", "omega_only"]))
    ]
    cols = [
        "sigma", "beta", "control", "C_A_full_mean", "C_metric_full_mean",
        "C_metric_over_C_A_full_mean", "persistent_n_mean", "persistent_units_max_mean",
        "mean_A_over_G0_mean", "negative_stiffness_fraction_mean",
    ]
    cols = [c for c in cols if c in compact.columns]
    print("\nCompact summary at max sigma:")
    print(compact[cols].to_string(index=False))


def build_config_from_args() -> FiaV4Config:
    parser = argparse.ArgumentParser(description="Run FIA Scientific Model V4.1.")
    parser.add_argument("--out", type=str, default=None, help="Output directory.")
    parser.add_argument("--quick", action="store_true", help="Run a small quick test.")
    parser.add_argument("--targeted", action="store_true", help="Run the N=80, sigma=0.30 targeted control test used for rapid evaluation.")
    parser.add_argument("--medium", action="store_true", help="Run a medium scientific test: N=120, reps=12, sigmas=(0.15,0.30).")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    parser.add_argument("--reps", type=int, default=None, help="Override number of repetitions.")
    parser.add_argument("--N", type=int, default=None, help="Override system size.")
    parser.add_argument("--steps", type=int, default=None, help="Override integration steps.")
    parser.add_argument("--normalization", choices=["global", "local_degree"], default=None, help="Override coupling normalization.")
    parser.add_argument("--phase-noise", type=float, default=None, help="Override dynamic phase noise.")
    parser.add_argument("--bootstrap", type=int, default=None, help="Override bootstrap/sign-flip resamples.")
    args = parser.parse_args()

    cfg = FiaV4Config(output_dir="FIA_Scientific_V41_output")
    if args.quick:
        cfg = replace(
            cfg,
            N=60,
            steps=70,
            n_reps=2,
            mass_sigmas=(0.0, 0.25),
            betas=(0.0, 1.0, 3.0),
            controls=("full", "equal_identical", "beta0_full", "kernel_only", "capacity_only", "omega_only", "static_phase"),
            detect_every=10,
            burn_in_steps=20,
            output_dir="FIA_Scientific_V41_quick_output",
            bootstrap_n=200,
        )
    if args.targeted:
        cfg = replace(
            cfg,
            N=80,
            steps=120,
            n_reps=8,
            mass_sigmas=(0.30,),
            betas=(0.0, 0.5, 1.0, 2.0, 4.0),
            output_dir="FIA_Scientific_V41_targeted_output",
            bootstrap_n=500,
        )
    if args.medium:
        cfg = replace(
            cfg,
            N=120,
            steps=180,
            n_reps=12,
            mass_sigmas=(0.15, 0.30),
            betas=(0.0, 0.5, 1.0, 2.0, 4.0),
            output_dir="FIA_Scientific_V41_medium_output",
            bootstrap_n=750,
        )
    if args.out is not None:
        cfg = replace(cfg, output_dir=args.out)
    if args.no_plots:
        cfg = replace(cfg, make_plots=False)
    if args.reps is not None:
        cfg = replace(cfg, n_reps=args.reps)
    if args.N is not None:
        cfg = replace(cfg, N=args.N)
    if args.steps is not None:
        cfg = replace(cfg, steps=args.steps)
    if args.normalization is not None:
        cfg = replace(cfg, normalization=args.normalization)
    if args.phase_noise is not None:
        cfg = replace(cfg, phase_noise=args.phase_noise)
    if args.bootstrap is not None:
        cfg = replace(cfg, bootstrap_n=args.bootstrap)
    return cfg


if __name__ == "__main__":
    run_experiment(build_config_from_args())
