from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Dict, Optional, Tuple

import numpy as np

from .constants import SPECIES_NAME_TO_TYPE


@dataclass
class AtomicState:
    """In-memory atomic lattice state for KMC.

    - `atom_ids` are immutable site labels from input (1-based if loaded from LAMMPS).
    - `positions` are static lattice coordinates under PBC.
    - `types` uses 0 for vacancy and 1..3 for Fe/Ni/Cr.
    """

    atom_ids: np.ndarray
    positions: np.ndarray
    types: np.ndarray
    bounds: np.ndarray  # shape (3, 2): [[xlo,xhi],[ylo,yhi],[zlo,zhi]]

    def __post_init__(self) -> None:
        self.atom_ids = np.asarray(self.atom_ids, dtype=np.int64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.types = np.asarray(self.types, dtype=np.int64)
        self.bounds = np.asarray(self.bounds, dtype=np.float64)

        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.types.ndim != 1 or self.types.shape[0] != self.positions.shape[0]:
            raise ValueError("types must have shape (N,) and match positions")
        if self.atom_ids.ndim != 1 or self.atom_ids.shape[0] != self.positions.shape[0]:
            raise ValueError("atom_ids must have shape (N,) and match positions")
        if self.bounds.shape != (3, 2):
            raise ValueError("bounds must have shape (3, 2)")

        lengths = self.lengths
        if np.any(lengths <= 0):
            raise ValueError(f"Invalid bounds lengths: {lengths}")

    @property
    def n_sites(self) -> int:
        return int(self.positions.shape[0])

    @property
    def lengths(self) -> np.ndarray:
        return self.bounds[:, 1] - self.bounds[:, 0]

    @property
    def vacancy_indices(self) -> np.ndarray:
        return np.where(self.types == 0)[0]

    @property
    def vacancy_index(self) -> int:
        vac = self.vacancy_indices
        if vac.size != 1:
            raise ValueError(f"State must have exactly one vacancy (type 0), found {vac.size}")
        return int(vac[0])

    def copy(self) -> "AtomicState":
        return AtomicState(
            atom_ids=self.atom_ids.copy(),
            positions=self.positions.copy(),
            types=self.types.copy(),
            bounds=self.bounds.copy(),
        )

    def enforce_single_vacancy(self, vacancy_atom_id: Optional[int] = None) -> None:
        if vacancy_atom_id is not None:
            matches = np.where(self.atom_ids == int(vacancy_atom_id))[0]
            if matches.size != 1:
                raise ValueError(f"Vacancy atom id {vacancy_atom_id} not found in structure.")
            self.types[self.types == 0] = 1
            self.types[matches[0]] = 0
            return

        vac = np.where(self.types == 0)[0]
        if vac.size == 1:
            return

        if vac.size == 0:
            # deterministic fallback: pick the lowest atom id
            idx = int(np.argmin(self.atom_ids))
            self.types[idx] = 0
            return

        # more than one vacancy -> keep first by atom id, fill others with Fe (type 1)
        keep = int(vac[np.argmin(self.atom_ids[vac])])
        for idx in vac:
            if idx != keep:
                self.types[idx] = 1

    def species_counts(self) -> Dict[str, int]:
        out = {name: 0 for name in SPECIES_NAME_TO_TYPE}
        for name, t in SPECIES_NAME_TO_TYPE.items():
            out[name] = int(np.sum(self.types == t))
        return out


def _closest_triplet_factors(n: int) -> Tuple[int, int, int]:
    if n < 1:
        raise ValueError("n must be >= 1")
    best = (1, 1, n)
    best_score = (n - 1, n)
    a_max = int(round(n ** (1.0 / 3.0))) + 3
    for a in range(1, a_max + 1):
        if n % a != 0:
            continue
        rem = n // a
        b_max = int(rem ** 0.5) + 2
        for b in range(a, b_max + 1):
            if rem % b != 0:
                continue
            c = rem // b
            trip = tuple(sorted((a, b, c)))
            score = (trip[2] - trip[0], trip[2])
            if score < best_score:
                best = trip
                best_score = score
    return best


def _counts_from_composition(total_sites: int, composition_percent: Dict[str, float]) -> Dict[str, int]:
    names = ["Fe", "Ni", "Cr"]
    frac = np.array([composition_percent[name] for name in names], dtype=np.float64) / 100.0
    raw = frac * float(total_sites)
    base = np.floor(raw).astype(np.int64)
    remainder = int(total_sites - int(base.sum()))

    if remainder > 0:
        order = np.argsort(-(raw - base))
        for idx in order[:remainder]:
            base[idx] += 1

    return {names[i]: int(base[i]) for i in range(3)}


def generate_random_fcc_state(
    num_atoms: int,
    composition_percent: Dict[str, float],
    vacancy_atom_id: Optional[int],
    random_seed: int,
) -> AtomicState:
    if num_atoms <= 0:
        raise ValueError("num_atoms must be > 0")
    if num_atoms % 4 != 0:
        raise ValueError("num_atoms must be a multiple of 4 for FCC")

    total_pct = sum(float(v) for v in composition_percent.values())
    if not isclose(total_pct, 100.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Composition must sum to 100%. Got {total_pct:.8f}%")

    n_cells = num_atoms // 4
    cx, cy, cz = _closest_triplet_factors(n_cells)
    lx, ly, lz = 2 * cx, 2 * cy, 2 * cz

    coords = []
    for x in range(lx):
        for y in range(ly):
            z0 = (x + y) % 2
            for z in range(z0, lz, 2):
                coords.append((float(x), float(y), float(z)))

    positions = np.asarray(coords, dtype=np.float64)
    if positions.shape[0] != num_atoms:
        raise RuntimeError(
            f"Internal FCC generation mismatch: requested {num_atoms}, generated {positions.shape[0]}"
        )

    counts = _counts_from_composition(num_atoms, composition_percent)
    types = np.concatenate(
        [
            np.full(counts["Fe"], SPECIES_NAME_TO_TYPE["Fe"], dtype=np.int64),
            np.full(counts["Ni"], SPECIES_NAME_TO_TYPE["Ni"], dtype=np.int64),
            np.full(counts["Cr"], SPECIES_NAME_TO_TYPE["Cr"], dtype=np.int64),
        ]
    )
    rng = np.random.default_rng(int(random_seed))
    rng.shuffle(types)

    atom_ids = np.arange(1, num_atoms + 1, dtype=np.int64)
    bounds = np.array([[0.0, float(lx)], [0.0, float(ly)], [0.0, float(lz)]], dtype=np.float64)

    state = AtomicState(atom_ids=atom_ids, positions=positions, types=types, bounds=bounds)
    state.enforce_single_vacancy(vacancy_atom_id=vacancy_atom_id)
    return state


def minimum_image_vectors(delta: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    out = np.array(delta, dtype=np.float64, copy=True)
    out -= np.rint(out / lengths[None, :]) * lengths[None, :]
    return out
