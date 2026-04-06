#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
PRODUCTION_ROOT = THIS_FILE.parents[1]
if str(PRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_ROOT))

from production_kmc import ModelSettings, RunSettings, SimulationCallbacks, SimulationSettings, StructureSettings, run_simulation
from production_kmc.descriptor import build_neighbor_shells
from production_kmc.io_lammps import read_lammps_data
from production_kmc.observables import local_order_pair_statistics
from production_kmc.plotting import render_local_order_vs_temperature


def _parser() -> argparse.ArgumentParser:
    repo_root = PRODUCTION_ROOT.parents[0]
    p = argparse.ArgumentParser(description="Run temperature sweep for Production KMC-ANN")

    p.add_argument("--mode", choices=["file", "generated"], default="file")
    p.add_argument("--data", default=str(repo_root / "Production/data/df_atoms_fcc_FeNiCr.data"))
    p.add_argument("--num_atoms", type=int, default=500)
    p.add_argument("--fe_pct", type=float, default=33.34)
    p.add_argument("--ni_pct", type=float, default=33.33)
    p.add_argument("--cr_pct", type=float, default=33.33)
    p.add_argument("--vacancy_atom_id", type=int, default=-1)
    p.add_argument("--structure_seed", type=int, default=123)

    p.add_argument("--ckpt", default=str(repo_root / "Production/models/example/best_model_10TypeFeatures_best.pth"))
    p.add_argument("--meta", default=str(repo_root / "Production/models/example/example_model.meta.json"))
    p.add_argument("--cache_size", type=int, default=200_000)

    p.add_argument("--Tmin", type=float, default=600.0)
    p.add_argument("--Tmax", type=float, default=1400.0)
    p.add_argument("--Tstep", type=float, default=200.0)

    p.add_argument("--nu", type=float, default=1.0e13)
    p.add_argument("--nu_log10", type=float, default=None, help="If set, nu = 10^X (s^-1)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--frame_videos", action="store_true", help="Generate frames_2D / frames_3D videos for each run")
    p.add_argument("--video_max_frames", type=int, default=300)
    p.add_argument("--video_fps", type=int, default=12)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out_root", default=str(repo_root / "Production/runs/sweep"))

    p.add_argument("--detailed_balance", action="store_true")
    p.add_argument("--interaction_matrix", default="")
    return p


def main() -> int:
    args = _parser().parse_args()

    interaction_matrix = None
    if args.detailed_balance:
        if args.interaction_matrix.strip():
            interaction_matrix = json.loads(args.interaction_matrix)
        else:
            interaction_matrix = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    temps = np.arange(args.Tmin, args.Tmax + 1e-12, args.Tstep)
    if args.nu_log10 is not None:
        nu_value = float(10.0 ** float(args.nu_log10))
    else:
        nu_value = float(args.nu)

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    xij_rows = []
    for T in temps:
        run_dir = out_root / f"T_{T:g}K"
        print(f"\n=== Running T = {T:g} K ===", flush=True)

        settings = RunSettings(
            model=ModelSettings(
                checkpoint_path=Path(args.ckpt).expanduser().resolve(),
                metadata_path=Path(args.meta).expanduser().resolve(),
                device="cpu",
                cache_size=int(args.cache_size),
            ),
            structure=StructureSettings(
                mode=args.mode,
                data_path=Path(args.data).expanduser().resolve() if args.mode == "file" else None,
                num_atoms=int(args.num_atoms),
                composition_percent={"Fe": args.fe_pct, "Ni": args.ni_pct, "Cr": args.cr_pct},
                vacancy_atom_id=None if int(args.vacancy_atom_id) < 0 else int(args.vacancy_atom_id),
                random_seed=int(args.structure_seed),
            ),
            simulation=SimulationSettings(
                temperature_K=float(T),
                attempt_frequency_s_inv=nu_value,
                num_steps=int(args.steps),
                steps_per_save=int(args.save_every),
                generate_frame_videos=bool(args.frame_videos),
                video_max_frames=int(args.video_max_frames),
                video_fps=int(args.video_fps),
                output_dir=run_dir,
                random_seed=int(args.seed),
                enable_detailed_balance=bool(args.detailed_balance),
                interaction_matrix_eV=interaction_matrix,
            ),
        )

        result = run_simulation(
            settings,
            callbacks=SimulationCallbacks(on_log=lambda msg: print(msg, flush=True)),
        )

        obs_path = run_dir / "observables.csv"
        final_row = {}
        if obs_path.exists():
            obs = pd.read_csv(obs_path)
            if not obs.empty:
                final_row = obs.iloc[-1].to_dict()

        xij_row = {"T": float(T)}
        try:
            final_state = read_lammps_data(run_dir / "structure_final.data").state
            nn1 = build_neighbor_shells(
                positions=final_state.positions,
                bounds=final_state.bounds,
                shell_counts={1: 12},
            ).shell_neighbors[1]
            xij_row.update(local_order_pair_statistics(final_state.types, nn1))
            xij_rows.append(xij_row)
        except Exception as exc:
            print(f"[warn] Failed local-order analysis at T={T:g}K: {exc}", flush=True)

        rows.append(
            {
                "T": float(T),
                "steps_completed": result.steps_completed,
                "total_sim_time_s": result.total_sim_time_s,
                "diffusion_estimate_a2_per_s": result.diffusion_estimate_a2_per_s,
                **{f"final_{k}": v for k, v in final_row.items()},
            }
        )

    summary = pd.DataFrame(rows).sort_values("T")
    summary_path = out_root / "sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSweep complete: {summary_path}")

    if xij_rows:
        xij_df = pd.DataFrame(xij_rows).sort_values("T")
        xij_csv_path = out_root / "xij_by_T.csv"
        xij_df.to_csv(xij_csv_path, index=False)
        xij_plot_path = out_root / "xij_norm_vs_T.png"
        render_local_order_vs_temperature(
            xij_df=xij_df,
            out_path=xij_plot_path,
            normalized=True,
        )
        print(f"Local-order table: {xij_csv_path}")
        print(f"Local-order plot:  {xij_plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
