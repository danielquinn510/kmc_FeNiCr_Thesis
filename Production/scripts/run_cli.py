#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PRODUCTION_ROOT = THIS_FILE.parents[1]
if str(PRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_ROOT))

from production_kmc import ModelSettings, RunSettings, SimulationCallbacks, SimulationSettings, StructureSettings, run_simulation


def _build_parser() -> argparse.ArgumentParser:
    repo_root = PRODUCTION_ROOT.parents[0]

    p = argparse.ArgumentParser(description="Production ANN+KMC runner")
    p.add_argument("--mode", choices=["file", "generated"], default="file")
    p.add_argument("--data", default=str(repo_root / "Production/data/df_atoms_fcc_FeNiCr.data"))
    p.add_argument("--num_atoms", type=int, default=500)
    p.add_argument("--fe_pct", type=float, default=33.34)
    p.add_argument("--ni_pct", type=float, default=33.33)
    p.add_argument("--cr_pct", type=float, default=33.33)
    p.add_argument("--vacancy_atom_id", type=int, default=-1)
    p.add_argument("--structure_seed", type=int, default=123)

    p.add_argument(
        "--ckpt",
        default=str(repo_root / "Production/models/example/best_model_10TypeFeatures_best.pth"),
    )
    p.add_argument(
        "--meta",
        default=str(repo_root / "Production/models/example/example_model.meta.json"),
    )
    p.add_argument("--cache_size", type=int, default=200_000)

    p.add_argument("--T", type=float, default=1000.0)
    p.add_argument("--nu", type=float, default=1.0e13)
    p.add_argument("--nu_log10", type=float, default=None, help="If set, nu = 10^X (s^-1)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--no_frame_videos", action="store_true", help="Disable frames_2D / frames_3D output videos")
    p.add_argument("--video_max_frames", type=int, default=300)
    p.add_argument("--video_fps", type=int, default=12)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out_dir", default=str(repo_root / "Production/runs/cli_run"))

    p.add_argument("--detailed_balance", action="store_true")
    p.add_argument(
        "--interaction_matrix",
        default="",
        help="3x3 interaction matrix JSON, e.g. [[0,0,0],[0,0,0],[0,0,0]]",
    )

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    interaction_matrix = None
    if args.detailed_balance:
        if args.interaction_matrix.strip():
            interaction_matrix = json.loads(args.interaction_matrix)
        else:
            interaction_matrix = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    if args.nu_log10 is not None:
        nu_value = float(10.0 ** float(args.nu_log10))
    else:
        nu_value = float(args.nu)

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
            temperature_K=float(args.T),
            attempt_frequency_s_inv=nu_value,
            num_steps=int(args.steps),
            steps_per_save=int(args.save_every),
            generate_frame_videos=not bool(args.no_frame_videos),
            video_max_frames=int(args.video_max_frames),
            video_fps=int(args.video_fps),
            output_dir=Path(args.out_dir).expanduser().resolve(),
            random_seed=int(args.seed),
            enable_detailed_balance=bool(args.detailed_balance),
            interaction_matrix_eV=interaction_matrix,
        ),
    )

    callbacks = SimulationCallbacks(
        on_log=lambda msg: print(msg, flush=True),
        on_progress=lambda done, total: None,
        on_observable=None,
    )

    result = run_simulation(settings, callbacks=callbacks)
    print("\nCompleted.")
    print(f"Output directory: {result.output_dir}")
    print(f"Steps completed: {result.steps_completed}")
    print(f"Total simulated time: {result.total_sim_time_s:.6e} s")
    if result.diffusion_estimate_a2_per_s is not None:
        print(f"Estimated diffusion coefficient: {result.diffusion_estimate_a2_per_s:.6e} A^2/s")
    if result.frames_2d_path is not None:
        print(f"frames_2D: {result.frames_2d_path}")
    if result.frames_3d_path is not None:
        print(f"frames_3D: {result.frames_3d_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
