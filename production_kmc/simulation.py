from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Optional
import json

import numpy as np
import pandas as pd

from .ann import CachedBarrierPredictor
from .config import RunSettings
from .descriptor import DescriptorEncoder, DescriptorLayout, build_neighbor_shells
from .engine import KMCState, KMCStepResult, initialize_kmc_state, run_kmc_step
from .io_lammps import read_lammps_data, write_lammps_atomic
from .media import render_frames_2d_3d
from .observables import (
    ObservableRecord,
    build_observable_record,
    estimate_diffusion_coefficient,
)
from .plotting import render_standard_plots
from .structure import AtomicState, generate_random_fcc_state


@dataclass
class SimulationCallbacks:
    on_log: Optional[Callable[[str], None]] = None
    on_progress: Optional[Callable[[int, int], None]] = None
    on_observable: Optional[Callable[[ObservableRecord], None]] = None


@dataclass
class RunResult:
    output_dir: Path
    steps_completed: int
    total_sim_time_s: float
    diffusion_estimate_a2_per_s: Optional[float]
    stopped_early: bool
    frames_2d_path: Optional[Path] = None
    frames_3d_path: Optional[Path] = None


def _emit(cb: Optional[Callable], *args) -> None:
    if cb is not None:
        cb(*args)


def _load_or_generate_structure(settings: RunSettings) -> AtomicState:
    s = settings.structure
    if s.mode == "file":
        lmp = read_lammps_data(s.data_path)
        state = lmp.state.copy()
        state.enforce_single_vacancy(vacancy_atom_id=s.vacancy_atom_id)
        return state

    return generate_random_fcc_state(
        num_atoms=s.num_atoms,
        composition_percent=s.composition_percent,
        vacancy_atom_id=s.vacancy_atom_id,
        random_seed=s.random_seed,
    )


def _state_from_kmc_state(template: AtomicState, kmc_state: KMCState) -> AtomicState:
    return AtomicState(
        atom_ids=template.atom_ids.copy(),
        positions=template.positions.copy(),
        types=kmc_state.types.copy(),
        bounds=template.bounds.copy(),
    )


def _write_step_event_table(
    out_dir: Path,
    step_result: KMCStepResult,
    positions: np.ndarray,
) -> None:
    df = pd.DataFrame(
        {
            "event_idx": np.arange(step_result.candidate_sites.shape[0], dtype=int),
            "site_index": step_result.candidate_sites,
            "site_type": step_result.candidate_types,
            "x": positions[step_result.candidate_sites, 0],
            "y": positions[step_result.candidate_sites, 1],
            "z": positions[step_result.candidate_sites, 2],
            "barrier_eV": step_result.candidate_barriers_eV,
            "rate_s_inv": step_result.candidate_rates_s_inv,
            "probability": step_result.candidate_probabilities,
        }
    )
    df.to_csv(out_dir / f"events_step{step_result.step:06d}.csv", index=False)


def run_simulation(
    settings: RunSettings,
    callbacks: Optional[SimulationCallbacks] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> RunResult:
    callbacks = callbacks or SimulationCallbacks()
    settings.validate()

    model_cfg = settings.model
    sim_cfg = settings.simulation

    out_dir = Path(sim_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save normalized config for reproducibility.
    run_config = {
        "model": {
            "checkpoint_path": str(model_cfg.checkpoint_path),
            "metadata_path": str(model_cfg.metadata_path),
            "device": model_cfg.device,
            "cache_size": model_cfg.cache_size,
        },
        "structure": {
            "mode": settings.structure.mode,
            "data_path": None if settings.structure.data_path is None else str(settings.structure.data_path),
            "num_atoms": settings.structure.num_atoms,
            "composition_percent": settings.structure.composition_percent,
            "vacancy_atom_id": settings.structure.vacancy_atom_id,
            "random_seed": settings.structure.random_seed,
        },
        "simulation": {
            "temperature_K": sim_cfg.temperature_K,
            "attempt_frequency_s_inv": sim_cfg.attempt_frequency_s_inv,
            "num_steps": sim_cfg.num_steps,
            "steps_per_save": sim_cfg.steps_per_save,
            "generate_frame_videos": sim_cfg.generate_frame_videos,
            "video_max_frames": sim_cfg.video_max_frames,
            "video_fps": sim_cfg.video_fps,
            "random_seed": sim_cfg.random_seed,
            "enable_detailed_balance": sim_cfg.enable_detailed_balance,
            "interaction_matrix_eV": sim_cfg.interaction_matrix_eV,
        },
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    _emit(callbacks.on_log, "Preparing structure...")
    structure = _load_or_generate_structure(settings)
    write_lammps_atomic(out_dir / "structure_initial.data", structure)

    _emit(callbacks.on_log, "Loading model metadata and building descriptor layout...")
    layout = DescriptorLayout.from_metadata_file(model_cfg.metadata_path)

    _emit(callbacks.on_log, "Precomputing neighbor shells (one-time setup)...")
    neighbors = build_neighbor_shells(
        positions=structure.positions,
        bounds=structure.bounds,
        shell_counts=layout.shell_counts,
    )
    encoder = DescriptorEncoder(layout=layout, neighbors=neighbors)

    _emit(callbacks.on_log, "Loading ANN checkpoint...")
    predictor = CachedBarrierPredictor(
        checkpoint_path=model_cfg.checkpoint_path,
        descriptor_layout=layout,
        device=model_cfg.device,
        cache_size=model_cfg.cache_size,
    )

    kmc_state = initialize_kmc_state(structure)
    rng = np.random.default_rng(sim_cfg.random_seed)

    interaction_matrix = None
    if sim_cfg.enable_detailed_balance and sim_cfg.interaction_matrix_eV is not None:
        interaction_matrix = np.asarray(sim_cfg.interaction_matrix_eV, dtype=np.float64)

    step_rows: list[Dict[str, float | int | bool]] = []
    observables: list[ObservableRecord] = []
    frame_types: list[np.ndarray] = [kmc_state.types.copy()]
    frame_steps: list[int] = [0]
    frame_times_s: list[float] = [0.0]

    total_steps = int(sim_cfg.num_steps)
    save_every = int(sim_cfg.steps_per_save)
    stopped_early = False

    _emit(
        callbacks.on_log,
        f"Starting KMC loop: T={sim_cfg.temperature_K:g} K, nu={sim_cfg.attempt_frequency_s_inv:.3e} s^-1, steps={total_steps}",
    )

    for _ in range(total_steps):
        if stop_requested is not None and stop_requested():
            stopped_early = True
            _emit(callbacks.on_log, "Stop requested. Finalizing current outputs...")
            break

        step_result = run_kmc_step(
            state=kmc_state,
            encoder=encoder,
            predictor=predictor,
            positions=structure.positions,
            bounds=structure.bounds,
            temperature_K=sim_cfg.temperature_K,
            attempt_frequency_s_inv=sim_cfg.attempt_frequency_s_inv,
            rng=rng,
            interaction_matrix_eV=interaction_matrix,
        )

        include_clusters = (step_result.step % save_every == 0) or (step_result.step == total_steps)
        observable = build_observable_record(
            step=step_result.step,
            time_s=step_result.time_s,
            positions=structure.positions,
            atom_displacements=kmc_state.atom_displacements,
            removed_atom_id=kmc_state.removed_atom_id,
            event_probabilities=step_result.candidate_probabilities,
            types=kmc_state.types,
            vacancy_index=kmc_state.vacancy_index,
            nn1_neighbors=neighbors.shell_neighbors[1],
            include_clusters=include_clusters,
        )

        observables.append(observable)
        _emit(callbacks.on_observable, observable)

        step_rows.append(
            {
                "step": step_result.step,
                "time_s": step_result.time_s,
                "dt_s": step_result.dt_s,
                "vacancy_site_before": step_result.vacancy_site_before,
                "vacancy_site_after": step_result.vacancy_site_after,
                "chosen_event_idx": step_result.chosen_candidate_idx,
                "chosen_site": step_result.chosen_site,
                "chosen_barrier_eV": step_result.chosen_barrier_eV,
                "chosen_rate_s_inv": step_result.chosen_rate_s_inv,
                "total_rate_s_inv": step_result.total_rate_s_inv,
                "cache_hit": int(step_result.cache_hit),
            }
        )

        if include_clusters:
            _write_step_event_table(out_dir=out_dir, step_result=step_result, positions=structure.positions)
            frame_types.append(kmc_state.types.copy())
            frame_steps.append(int(step_result.step))
            frame_times_s.append(float(step_result.time_s))

        if step_result.step % max(1, total_steps // 20) == 0:
            _emit(
                callbacks.on_log,
                (
                    f"step {step_result.step}/{total_steps} | "
                    f"t={step_result.time_s:.3e} s | "
                    f"dt={step_result.dt_s:.3e} s | "
                    f"MSD={observable.msd_a2:.3e} A^2"
                ),
            )

        _emit(callbacks.on_progress, step_result.step, total_steps)

    step_df = pd.DataFrame(step_rows)
    if not step_df.empty:
        step_df.to_csv(out_dir / "time_log.csv", index=False)

    obs_df = pd.DataFrame([asdict(record) for record in observables])
    if not obs_df.empty:
        obs_df.to_csv(out_dir / "observables.csv", index=False)

    final_state = _state_from_kmc_state(structure, kmc_state)
    write_lammps_atomic(out_dir / "structure_final.data", final_state)

    if kmc_state.steps_done > 0 and frame_steps[-1] != kmc_state.steps_done:
        frame_types.append(kmc_state.types.copy())
        frame_steps.append(int(kmc_state.steps_done))
        frame_times_s.append(float(kmc_state.time_s))

    types_frames = np.stack(frame_types, axis=0).astype(np.int8, copy=False)
    frame_steps_arr = np.asarray(frame_steps, dtype=np.int64)
    frame_times_arr = np.asarray(frame_times_s, dtype=np.float64)
    trajectory_frames_path = out_dir / "trajectory_frames.npz"
    np.savez_compressed(
        trajectory_frames_path,
        types_frames=types_frames,
        frame_steps=frame_steps_arr,
        frame_times_s=frame_times_arr,
    )

    diffusion = None
    if not obs_df.empty:
        diffusion = estimate_diffusion_coefficient(
            times_s=obs_df["time_s"].to_numpy(),
            msd_a2=obs_df["msd_a2"].to_numpy(),
            dimensions=3,
            fit_fraction=0.5,
        )

    frames_2d_path: Optional[Path] = None
    frames_3d_path: Optional[Path] = None
    if sim_cfg.generate_frame_videos:
        try:
            media_artifacts = render_frames_2d_3d(
                positions=structure.positions,
                bounds=structure.bounds,
                types_frames=types_frames,
                frame_steps=frame_steps_arr,
                out_dir=out_dir,
                max_frames=sim_cfg.video_max_frames,
                fps=sim_cfg.video_fps,
            )
            frames_2d_path = media_artifacts.frames_2d
            frames_3d_path = media_artifacts.frames_3d
            if frames_2d_path is not None:
                _emit(callbacks.on_log, f"Saved observable: {frames_2d_path.name}")
            if frames_3d_path is not None:
                _emit(callbacks.on_log, f"Saved observable: {frames_3d_path.name}")
        except Exception as exc:
            _emit(callbacks.on_log, f"[warn] Frame video generation failed: {exc}")

    summary = {
        "steps_completed": int(kmc_state.steps_done),
        "stopped_early": bool(stopped_early),
        "total_sim_time_s": float(kmc_state.time_s),
        "diffusion_estimate_a2_per_s": None if diffusion is None else float(diffusion),
        "vacancy_final_site": int(kmc_state.vacancy_index),
        "trajectory_frames_npz": str(trajectory_frames_path),
        "frames_2D": None if frames_2d_path is None else str(frames_2d_path),
        "frames_3D": None if frames_3d_path is None else str(frames_3d_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        if (out_dir / "observables.csv").exists():
            render_standard_plots(
                out_dir / "observables.csv",
                out_dir,
                time_log_csv=(out_dir / "time_log.csv"),
            )
    except Exception as exc:
        _emit(callbacks.on_log, f"[warn] Plot generation failed: {exc}")

    _emit(callbacks.on_log, f"Run complete. Outputs written to: {out_dir}")

    return RunResult(
        output_dir=out_dir,
        steps_completed=int(kmc_state.steps_done),
        total_sim_time_s=float(kmc_state.time_s),
        diffusion_estimate_a2_per_s=diffusion,
        stopped_early=stopped_early,
        frames_2d_path=frames_2d_path,
        frames_3d_path=frames_3d_path,
    )
