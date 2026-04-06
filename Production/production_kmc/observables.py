from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .constants import SPECIES_NAME_TO_TYPE, TYPE_TO_SPECIES_NAME


@dataclass
class ObservableRecord:
    step: int
    time_s: float
    vacancy_x: float
    vacancy_y: float
    vacancy_z: float
    msd_a2: float
    hop_entropy: float
    hop_randomness_R: float
    sro_proxy: float
    vacancy_1nn_Fe: float
    vacancy_1nn_Ni: float
    vacancy_1nn_Cr: float
    cluster_all: int
    cluster_Fe: int
    cluster_Ni: int
    cluster_Cr: int


PAIR_TYPE_KEYS: tuple[tuple[int, int], ...] = (
    (SPECIES_NAME_TO_TYPE["Fe"], SPECIES_NAME_TO_TYPE["Fe"]),
    (SPECIES_NAME_TO_TYPE["Fe"], SPECIES_NAME_TO_TYPE["Ni"]),
    (SPECIES_NAME_TO_TYPE["Fe"], SPECIES_NAME_TO_TYPE["Cr"]),
    (SPECIES_NAME_TO_TYPE["Ni"], SPECIES_NAME_TO_TYPE["Ni"]),
    (SPECIES_NAME_TO_TYPE["Ni"], SPECIES_NAME_TO_TYPE["Cr"]),
    (SPECIES_NAME_TO_TYPE["Cr"], SPECIES_NAME_TO_TYPE["Cr"]),
)


def pair_label(pair: tuple[int, int]) -> str:
    i, j = pair
    return f"{TYPE_TO_SPECIES_NAME[int(i)]}-{TYPE_TO_SPECIES_NAME[int(j)]}"


def atomic_msd_a2(atom_displacements: np.ndarray, removed_atom_id: int) -> float:
    sq = np.sum(atom_displacements * atom_displacements, axis=1)
    mask = np.ones(atom_displacements.shape[0], dtype=bool)
    mask[0] = False
    if 0 <= removed_atom_id < mask.shape[0]:
        mask[removed_atom_id] = False
    if not np.any(mask):
        return 0.0
    return float(np.mean(sq[mask]))


def estimate_diffusion_coefficient(
    times_s: np.ndarray,
    msd_a2: np.ndarray,
    dimensions: int = 3,
    fit_fraction: float = 0.5,
) -> Optional[float]:
    t = np.asarray(times_s, dtype=np.float64)
    y = np.asarray(msd_a2, dtype=np.float64)

    if t.size < 3 or y.size < 3:
        return None

    start = int(max(1, np.floor((1.0 - fit_fraction) * t.size)))
    xfit = t[start:]
    yfit = y[start:]

    if xfit.size < 2 or np.allclose(xfit, xfit[0]):
        return None

    slope, _intercept = np.polyfit(xfit, yfit, 1)
    return float(slope / (2.0 * dimensions))


def tracer_diffusion_vs_time(
    times_s: np.ndarray,
    msd_a2: np.ndarray,
    dimensions: int = 3,
) -> np.ndarray:
    """Return time-resolved tracer diffusion via Einstein ratio D(t)=MSD(t)/(2*d*t).

    For default d=3, this is D(t)=MSD(t)/(6*t). Entries with t<=0 (or non-finite
    input) are returned as NaN to avoid invalid division.
    """
    t = np.asarray(times_s, dtype=np.float64)
    y = np.asarray(msd_a2, dtype=np.float64)
    if t.shape != y.shape:
        raise ValueError(f"times_s and msd_a2 must have matching shapes, got {t.shape} vs {y.shape}")
    if dimensions <= 0:
        raise ValueError("dimensions must be > 0")

    out = np.full(t.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(y) & (t > 0.0)
    if np.any(valid):
        out[valid] = y[valid] / (2.0 * float(dimensions) * t[valid])
    return out


def event_entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    s = float(np.sum(p))
    if s <= 0.0:
        return 0.0
    p = np.clip(p / s, 1e-16, 1.0)
    return float(-np.sum(p * np.log(p)))


def jump_randomness_R(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    s = float(np.sum(p))
    if s <= 0.0 or p.size <= 1:
        return 0.0
    p = p / s

    sigma = float(np.std(p))
    z = int(p.size)
    sigma_max = np.sqrt(z - 1.0) / z
    if sigma_max <= 0.0:
        return 0.0

    r = 1.0 - sigma / sigma_max
    return float(np.clip(r, 0.0, 1.0))


def vacancy_shell_composition(types: np.ndarray, vacancy_index: int, nn1_neighbors: np.ndarray) -> Dict[str, float]:
    neigh_types = types[nn1_neighbors[vacancy_index]]
    valid = neigh_types[neigh_types > 0]
    if valid.size == 0:
        return {"Fe": 0.0, "Ni": 0.0, "Cr": 0.0}

    total = float(valid.size)
    return {
        "Fe": float(np.sum(valid == SPECIES_NAME_TO_TYPE["Fe"]) / total),
        "Ni": float(np.sum(valid == SPECIES_NAME_TO_TYPE["Ni"]) / total),
        "Cr": float(np.sum(valid == SPECIES_NAME_TO_TYPE["Cr"]) / total),
    }


def short_range_order_proxy(types: np.ndarray, vacancy_index: int, nn1_neighbors: np.ndarray) -> float:
    nonvac = types[types > 0]
    if nonvac.size == 0:
        return 0.0

    c_fe = float(np.sum(nonvac == SPECIES_NAME_TO_TYPE["Fe"]) / nonvac.size)
    c_ni = float(np.sum(nonvac == SPECIES_NAME_TO_TYPE["Ni"]) / nonvac.size)
    c_cr = float(np.sum(nonvac == SPECIES_NAME_TO_TYPE["Cr"]) / nonvac.size)

    local = vacancy_shell_composition(types, vacancy_index, nn1_neighbors)
    comps = {"Fe": c_fe, "Ni": c_ni, "Cr": c_cr}

    alphas = []
    for key in ("Fe", "Ni", "Cr"):
        if comps[key] > 1e-16:
            alphas.append(1.0 - local[key] / comps[key])
    if not alphas:
        return 0.0
    return float(np.mean(alphas))


def _largest_component_size(nodes: np.ndarray, adjacency: np.ndarray, allowed_mask: np.ndarray) -> int:
    if nodes.size == 0:
        return 0
    visited = np.zeros(allowed_mask.shape[0], dtype=bool)
    best = 0

    for node in nodes:
        if visited[node]:
            continue
        stack = [int(node)]
        visited[node] = True
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in adjacency[u]:
                if not allowed_mask[v] or visited[v]:
                    continue
                visited[v] = True
                stack.append(int(v))
        best = max(best, size)
    return best


def largest_cluster_sizes(types: np.ndarray, nn1_neighbors: np.ndarray) -> Dict[str, int]:
    adjacency = nn1_neighbors

    nonvac_mask = types > 0
    nonvac_nodes = np.where(nonvac_mask)[0]

    out = {
        "all": _largest_component_size(nonvac_nodes, adjacency, nonvac_mask),
    }

    for name in ("Fe", "Ni", "Cr"):
        t = SPECIES_NAME_TO_TYPE[name]
        mask = types == t
        nodes = np.where(mask)[0]
        out[name] = _largest_component_size(nodes, adjacency, mask)

    return out


def local_order_pair_statistics(types: np.ndarray, nn1_neighbors: np.ndarray) -> Dict[str, float]:
    """Compute legacy-style local order statistics on undirected 1NN pairs.

    Returns columns matching legacy analysis style:
    - X_<pair>, N_<pair>, N0_<pair>, XN_<pair> for each pair in PAIR_TYPE_KEYS.
    """
    t = np.asarray(types, dtype=np.int64)
    nn = np.asarray(nn1_neighbors, dtype=np.int64)

    if t.ndim != 1:
        raise ValueError("types must be 1D")
    if nn.ndim != 2 or nn.shape[0] != t.shape[0]:
        raise ValueError("nn1_neighbors must have shape (N, z) matching types length")

    pair_counts = {k: 0 for k in PAIR_TYPE_KEYS}
    total_pairs = 0
    n_sites = int(t.shape[0])

    for i in range(n_sites):
        ti = int(t[i])
        if ti <= 0:
            continue
        for j_raw in nn[i]:
            j = int(j_raw)
            if j <= i:
                continue
            tj = int(t[j])
            if tj <= 0:
                continue
            key = (ti, tj) if ti <= tj else (tj, ti)
            if key in pair_counts:
                pair_counts[key] += 1
                total_pairs += 1

    nonvac = t[t > 0]
    conc = {
        SPECIES_NAME_TO_TYPE["Fe"]: 0.0,
        SPECIES_NAME_TO_TYPE["Ni"]: 0.0,
        SPECIES_NAME_TO_TYPE["Cr"]: 0.0,
    }
    if nonvac.size > 0:
        denom = float(nonvac.size)
        for species in conc:
            conc[species] = float(np.sum(nonvac == species) / denom)

    out: Dict[str, float] = {}
    total_pairs_f = float(total_pairs)
    for key in PAIR_TYPE_KEYS:
        i, j = key
        label = pair_label(key)
        n_ij = float(pair_counts[key])
        if i == j:
            p_rand = conc[i] * conc[j]
        else:
            p_rand = 2.0 * conc[i] * conc[j]
        n0_ij = total_pairs_f * p_rand
        x_ij = n_ij - n0_ij
        xn_ij = x_ij / n0_ij if n0_ij > 1e-12 else float("nan")

        out[f"X_{label}"] = float(x_ij)
        out[f"N_{label}"] = float(n_ij)
        out[f"N0_{label}"] = float(n0_ij)
        out[f"XN_{label}"] = float(xn_ij)

    return out


def build_observable_record(
    step: int,
    time_s: float,
    positions: np.ndarray,
    atom_displacements: np.ndarray,
    removed_atom_id: int,
    event_probabilities: np.ndarray,
    types: np.ndarray,
    vacancy_index: int,
    nn1_neighbors: np.ndarray,
    include_clusters: bool,
) -> ObservableRecord:
    msd = atomic_msd_a2(atom_displacements, removed_atom_id)
    entropy = event_entropy(event_probabilities)
    randomness = jump_randomness_R(event_probabilities)
    sro = short_range_order_proxy(types, vacancy_index, nn1_neighbors)
    shell_comp = vacancy_shell_composition(types, vacancy_index, nn1_neighbors)
    vacancy_pos = positions[vacancy_index]

    if include_clusters:
        clusters = largest_cluster_sizes(types, nn1_neighbors)
    else:
        clusters = {"all": -1, "Fe": -1, "Ni": -1, "Cr": -1}

    return ObservableRecord(
        step=int(step),
        time_s=float(time_s),
        vacancy_x=float(vacancy_pos[0]),
        vacancy_y=float(vacancy_pos[1]),
        vacancy_z=float(vacancy_pos[2]),
        msd_a2=float(msd),
        hop_entropy=float(entropy),
        hop_randomness_R=float(randomness),
        sro_proxy=float(sro),
        vacancy_1nn_Fe=float(shell_comp["Fe"]),
        vacancy_1nn_Ni=float(shell_comp["Ni"]),
        vacancy_1nn_Cr=float(shell_comp["Cr"]),
        cluster_all=int(clusters["all"]),
        cluster_Fe=int(clusters["Fe"]),
        cluster_Ni=int(clusters["Ni"]),
        cluster_Cr=int(clusters["Cr"]),
    )
