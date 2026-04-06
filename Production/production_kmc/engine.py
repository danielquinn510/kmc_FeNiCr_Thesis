from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ann import CachedBarrierPredictor, PredictionResult
from .constants import K_B_EV_PER_K
from .descriptor import DescriptorEncoder
from .structure import AtomicState, minimum_image_vectors


@dataclass
class KMCState:
    types: np.ndarray
    vacancy_index: int
    atom_id_at_site: np.ndarray
    removed_atom_id: int
    atom_displacements: np.ndarray  # [atom_id, xyz], Angstrom
    time_s: float = 0.0
    steps_done: int = 0


@dataclass
class KMCStepResult:
    step: int
    time_s: float
    dt_s: float
    vacancy_site_before: int
    vacancy_site_after: int
    candidate_sites: np.ndarray
    candidate_types: np.ndarray
    candidate_barriers_eV: np.ndarray
    candidate_rates_s_inv: np.ndarray
    candidate_probabilities: np.ndarray
    chosen_candidate_idx: int
    chosen_site: int
    chosen_barrier_eV: float
    chosen_rate_s_inv: float
    total_rate_s_inv: float
    cache_hit: bool


class StopSimulation(RuntimeError):
    pass


def initialize_kmc_state(structure: AtomicState) -> KMCState:
    types = structure.types.copy()
    vacancy_index = int(np.where(types == 0)[0][0])

    atom_id_at_site = structure.atom_ids.copy().astype(np.int64)
    removed_atom_id = int(atom_id_at_site[vacancy_index])
    atom_id_at_site[vacancy_index] = 0

    n_atoms = int(np.max(structure.atom_ids))
    atom_displacements = np.zeros((n_atoms + 1, 3), dtype=np.float64)

    return KMCState(
        types=types,
        vacancy_index=vacancy_index,
        atom_id_at_site=atom_id_at_site,
        removed_atom_id=removed_atom_id,
        atom_displacements=atom_displacements,
    )


def _compute_rates(barriers_eV: np.ndarray, temperature_K: float, attempt_frequency_s_inv: float) -> np.ndarray:
    barriers = np.asarray(barriers_eV, dtype=np.float64)
    rates = attempt_frequency_s_inv * np.exp(-barriers / (K_B_EV_PER_K * temperature_K))
    return rates


def _select_event(rates: np.ndarray, rng: np.random.Generator) -> tuple[int, float, float, np.ndarray]:
    total = float(np.sum(rates))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"Total transition rate is not positive: {total}")

    probabilities = rates / total
    # cumulative roulette selection for speed and deterministic behavior parity
    threshold = float(rng.random()) * total
    cumulative = np.cumsum(rates)
    event_idx = int(np.searchsorted(cumulative, threshold, side="left"))
    if event_idx >= rates.shape[0]:
        event_idx = rates.shape[0] - 1

    u = float(rng.random())
    if u <= 0.0:
        u = np.nextafter(0.0, 1.0)
    dt = -np.log(u) / total

    return event_idx, dt, total, probabilities


def _local_energy_of_species_at_site(
    types: np.ndarray,
    species_type: int,
    site_index: int,
    nn1_neighbors: np.ndarray,
    interaction_matrix_eV: np.ndarray,
) -> float:
    neighbors = nn1_neighbors[site_index]
    neigh_types = types[neighbors]
    valid = neigh_types > 0
    if not np.any(valid):
        return 0.0
    return float(np.sum(interaction_matrix_eV[species_type - 1, neigh_types[valid] - 1]))


def _detailed_balance_adjustment(
    barriers_eV: np.ndarray,
    types: np.ndarray,
    vacancy_index: int,
    candidate_sites: np.ndarray,
    nn1_neighbors: np.ndarray,
    interaction_matrix_eV: np.ndarray,
) -> np.ndarray:
    adjusted = np.asarray(barriers_eV, dtype=np.float64).copy()
    for j, target_site in enumerate(candidate_sites):
        species_type = int(types[target_site])
        if species_type <= 0:
            continue

        e_before = _local_energy_of_species_at_site(
            types=types,
            species_type=species_type,
            site_index=int(target_site),
            nn1_neighbors=nn1_neighbors,
            interaction_matrix_eV=interaction_matrix_eV,
        )

        # virtual post-swap local environment energy at vacancy site
        e_after = _local_energy_of_species_at_site(
            types=types,
            species_type=species_type,
            site_index=int(vacancy_index),
            nn1_neighbors=nn1_neighbors,
            interaction_matrix_eV=interaction_matrix_eV,
        )

        delta_e = e_after - e_before
        adjusted[j] = max(0.0, adjusted[j] + max(0.0, 0.5 * delta_e))

    return adjusted


def _apply_swap_and_update_displacements(
    state: KMCState,
    positions: np.ndarray,
    bounds: np.ndarray,
    target_site: int,
) -> None:
    old_vac = int(state.vacancy_index)
    new_vac = int(target_site)

    # atom displacement bookkeeping for atomic MSD (not vacancy MSD)
    moved_atom_id = int(state.atom_id_at_site[new_vac])
    if moved_atom_id != 0 and moved_atom_id != state.removed_atom_id:
        dr = positions[old_vac] - positions[new_vac]
        lengths = bounds[:, 1] - bounds[:, 0]
        dr = minimum_image_vectors(dr[None, :], lengths)[0]
        state.atom_displacements[moved_atom_id] += dr

    # swap atom identities and atom types
    state.atom_id_at_site[old_vac], state.atom_id_at_site[new_vac] = (
        state.atom_id_at_site[new_vac],
        state.atom_id_at_site[old_vac],
    )
    state.types[old_vac], state.types[new_vac] = state.types[new_vac], state.types[old_vac]
    state.vacancy_index = new_vac


def run_kmc_step(
    state: KMCState,
    encoder: DescriptorEncoder,
    predictor: CachedBarrierPredictor,
    positions: np.ndarray,
    bounds: np.ndarray,
    temperature_K: float,
    attempt_frequency_s_inv: float,
    rng: np.random.Generator,
    interaction_matrix_eV: Optional[np.ndarray] = None,
) -> KMCStepResult:
    vacancy_before = int(state.vacancy_index)
    descriptor = encoder.encode_vacancy_environment(state.types, vacancy_before)
    prediction: PredictionResult = predictor.predict(descriptor)

    nn1_neighbors = encoder.neighbors.shell_neighbors[1]
    candidate_sites = nn1_neighbors[vacancy_before].copy()
    candidate_types = state.types[candidate_sites].copy()

    barriers = np.asarray(prediction.barriers_eV[: candidate_sites.shape[0]], dtype=np.float64)
    if barriers.shape[0] != candidate_sites.shape[0]:
        raise ValueError(
            f"Model output ({barriers.shape[0]}) does not match candidate event count ({candidate_sites.shape[0]})."
        )

    if interaction_matrix_eV is not None:
        barriers = _detailed_balance_adjustment(
            barriers_eV=barriers,
            types=state.types,
            vacancy_index=vacancy_before,
            candidate_sites=candidate_sites,
            nn1_neighbors=nn1_neighbors,
            interaction_matrix_eV=np.asarray(interaction_matrix_eV, dtype=np.float64),
        )

    rates = _compute_rates(barriers, temperature_K, attempt_frequency_s_inv)
    chosen_idx, dt, total_rate, probs = _select_event(rates, rng)
    chosen_site = int(candidate_sites[chosen_idx])

    _apply_swap_and_update_displacements(
        state=state,
        positions=positions,
        bounds=bounds,
        target_site=chosen_site,
    )

    state.time_s += dt
    state.steps_done += 1

    return KMCStepResult(
        step=state.steps_done,
        time_s=state.time_s,
        dt_s=dt,
        vacancy_site_before=vacancy_before,
        vacancy_site_after=int(state.vacancy_index),
        candidate_sites=candidate_sites,
        candidate_types=candidate_types,
        candidate_barriers_eV=barriers,
        candidate_rates_s_inv=rates,
        candidate_probabilities=probs,
        chosen_candidate_idx=chosen_idx,
        chosen_site=chosen_site,
        chosen_barrier_eV=float(barriers[chosen_idx]),
        chosen_rate_s_inv=float(rates[chosen_idx]),
        total_rate_s_inv=total_rate,
        cache_hit=prediction.cache_hit,
    )
