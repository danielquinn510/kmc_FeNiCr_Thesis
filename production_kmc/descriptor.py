from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple
import json
import re

import numpy as np

from .constants import CANONICAL_SHELL_COUNTS, DEFAULT_BARRIER_COLUMNS
from .structure import minimum_image_vectors

_FEATURE_RE = re.compile(r"^(\d+)NN_(\d+)_type_(\d+)$")


@dataclass(frozen=True)
class DescriptorLayout:
    feature_columns: List[str]
    shell_counts: Dict[int, int]
    barrier_columns: List[str]
    target_max_values: np.ndarray
    hidden_sizes: List[int]
    input_dim: int
    output_dim: int
    index_by_shell_slot_type: Dict[Tuple[int, int, int], int]

    @property
    def max_shell(self) -> int:
        return max(self.shell_counts)

    @classmethod
    def from_metadata_file(cls, path: str | Path) -> "DescriptorLayout":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

        shell_counts_payload = payload.get("shell_counts")
        if shell_counts_payload is None:
            shell_counts = None
        else:
            shell_counts = {int(k): int(v) for k, v in shell_counts_payload.items()}

        feature_columns_payload = payload.get("feature_columns")
        feature_columns = resolve_feature_columns(
            feature_columns_payload=feature_columns_payload,
            shell_counts=shell_counts,
        )

        inferred_shell_counts = infer_shell_counts_from_feature_columns(feature_columns)
        if shell_counts is None:
            shell_counts = inferred_shell_counts

        hidden_sizes = list(payload.get("hidden_sizes", []))
        input_dim = int(payload.get("input_dim", len(feature_columns)))
        output_dim = int(payload.get("output_dim", len(DEFAULT_BARRIER_COLUMNS)))

        target_max_values = np.asarray(payload.get("target_max_values", [1.0] * output_dim), dtype=np.float32)
        if target_max_values.shape[0] != output_dim:
            raise ValueError(
                f"target_max_values length ({target_max_values.shape[0]}) does not match output_dim ({output_dim})"
            )

        barrier_columns = list(payload.get("barrier_columns", DEFAULT_BARRIER_COLUMNS[:output_dim]))
        if len(barrier_columns) != output_dim:
            barrier_columns = [f"1NN_{i}" for i in range(1, output_dim + 1)]

        idx_map = build_feature_index_map(feature_columns)

        return cls(
            feature_columns=feature_columns,
            shell_counts=shell_counts,
            barrier_columns=barrier_columns,
            target_max_values=target_max_values,
            hidden_sizes=hidden_sizes,
            input_dim=input_dim,
            output_dim=output_dim,
            index_by_shell_slot_type=idx_map,
        )


def canonical_feature_columns(shell_counts: Mapping[int, int]) -> List[str]:
    cols: List[str] = []
    for shell in sorted(shell_counts):
        for slot in range(1, int(shell_counts[shell]) + 1):
            for species in (1, 2, 3):
                cols.append(f"{shell}NN_{slot}_type_{species}")
    return cols


def resolve_feature_columns(
    feature_columns_payload: Iterable[str] | None,
    shell_counts: Mapping[int, int] | None,
) -> List[str]:
    if feature_columns_payload is not None:
        cols = list(feature_columns_payload)
        if not cols:
            raise ValueError("feature_columns in metadata is empty")
        return cols

    use_shell_counts = shell_counts if shell_counts is not None else CANONICAL_SHELL_COUNTS
    return canonical_feature_columns(use_shell_counts)


def build_feature_index_map(feature_columns: Iterable[str]) -> Dict[Tuple[int, int, int], int]:
    out: Dict[Tuple[int, int, int], int] = {}
    for idx, col in enumerate(feature_columns):
        m = _FEATURE_RE.match(col)
        if not m:
            continue
        shell = int(m.group(1))
        slot = int(m.group(2))
        species = int(m.group(3))
        out[(shell, slot, species)] = idx
    if not out:
        raise ValueError("No descriptor columns matched '<shell>NN_<slot>_type_<species>' format")
    return out


def infer_shell_counts_from_feature_columns(feature_columns: Iterable[str]) -> Dict[int, int]:
    shell_counts: Dict[int, int] = {}
    for col in feature_columns:
        m = _FEATURE_RE.match(col)
        if not m:
            continue
        shell = int(m.group(1))
        slot = int(m.group(2))
        shell_counts[shell] = max(shell_counts.get(shell, 0), slot)

    if not shell_counts:
        raise ValueError("Could not infer shell counts from feature columns")
    return dict(sorted(shell_counts.items()))


@dataclass
class NeighborShells:
    shell_radii: Dict[int, float]
    shell_neighbors: Dict[int, np.ndarray]  # shell -> [N, count(shell)] site indices


def build_neighbor_shells(
    positions: np.ndarray,
    bounds: np.ndarray,
    shell_counts: Mapping[int, int],
    tolerance: float = 1e-6,
) -> NeighborShells:
    positions = np.asarray(positions, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    lengths = bounds[:, 1] - bounds[:, 0]

    n_sites = positions.shape[0]
    max_shell = int(max(shell_counts))

    # Determine shell radii from site 0 under minimum-image convention.
    dr0 = positions - positions[0]
    dr0 = minimum_image_vectors(dr0, lengths)
    d0 = np.linalg.norm(dr0, axis=1)
    unique = np.unique(np.round(d0[d0 > tolerance], 8))
    if unique.size < max_shell:
        raise ValueError(
            f"Structure has only {unique.size} resolvable shells, need {max_shell} for descriptor parity"
        )

    shell_radii = {shell: float(unique[shell - 1]) for shell in range(1, max_shell + 1)}

    shell_neighbors: Dict[int, np.ndarray] = {}
    for shell, count in shell_counts.items():
        shell_neighbors[int(shell)] = np.zeros((n_sites, int(count)), dtype=np.int64)

    for center in range(n_sites):
        delta = positions - positions[center]
        delta = minimum_image_vectors(delta, lengths)
        dist = np.linalg.norm(delta, axis=1)

        for shell, count in shell_counts.items():
            radius = shell_radii[int(shell)]
            mask = np.abs(dist - radius) <= tolerance
            idx = np.where(mask)[0]

            if idx.size < count:
                # fallback for slight numeric noise
                idx = np.argsort(np.abs(dist - radius))[: int(count)]

            # deterministic ordering by relative vector then by index
            vec = delta[idx]
            order = np.lexsort(
                (
                    idx,
                    np.round(vec[:, 2], 8),
                    np.round(vec[:, 1], 8),
                    np.round(vec[:, 0], 8),
                )
            )
            idx = idx[order][: int(count)]

            if idx.size != count:
                raise ValueError(
                    f"Shell {shell} at site {center} expected {count} neighbors, found {idx.size}"
                )

            shell_neighbors[int(shell)][center, :] = idx

    return NeighborShells(shell_radii=shell_radii, shell_neighbors=shell_neighbors)


@dataclass
class DescriptorEncoder:
    layout: DescriptorLayout
    neighbors: NeighborShells

    def encode_vacancy_environment(self, types: np.ndarray, vacancy_index: int) -> np.ndarray:
        features = np.zeros(self.layout.input_dim, dtype=np.float32)

        for shell, count in self.layout.shell_counts.items():
            local_indices = self.neighbors.shell_neighbors[shell][vacancy_index]
            local_types = types[local_indices]
            for slot0, species in enumerate(local_types):
                slot = slot0 + 1
                key = (shell, slot, int(species))
                idx = self.layout.index_by_shell_slot_type.get(key)
                if idx is not None:
                    features[idx] = 1.0

        return features
