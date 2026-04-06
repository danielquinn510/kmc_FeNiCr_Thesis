# Production KMC-ANN

Production-grade ANN-driven Kinetic Monte Carlo (KMC) simulator and GUI for vacancy diffusion in fcc Fe-Ni-Cr.

This folder is a full implementation aligned with the project description requirements and user stories, with a focus on:
- Descriptor parity with ANN training schema.
- Efficient KMC execution via one-time neighbor-shell precomputation.
- Persistent ANN inference caching.
- Reproducible outputs and observables suitable for analysis against Nature Communications (2024) 15:3879 analog trends.

## 1) What This Delivers

### User story coverage
1. **Select a LAMMPS `.data` file**:
- GUI supports file selection for initial structure loading.

2. **Create random custom atomic configuration**:
- GUI supports generated FCC structures with:
  - total atom/site count,
  - Fe/Ni/Cr composition percentages,
  - explicit vacancy atom id (or auto selection).

3. **Select ANN checkpoint, metadata, output directory**:
- GUI has dedicated file/directory selectors.
- CLI supports `--ckpt`, `--meta`, and `--out_dir`.

4. **Control parameters (`T`, attempt frequency, steps, save cadence)**:
- Full controls are exposed in GUI and CLI.
- GUI uses attempt-frequency exponent input `X` with `nu = 10^X`.
- GUI supports single-`T` runs and temperature sweeps (`Tmin`, `Tmax`, `Tstep`).

5. **Run complete KMC simulation**:
- Worker-threaded GUI execution + CLI execution path.

6. **View log updates during run**:
- Live log tab in GUI with progress updates.

7. **Toggle produced observables in GUI**:
- Observable tab with selectable image-backed plots (non-video):
  - MSD,
  - tracer diffusion vs time (`D(t) = MSD(t)/(6t)`),
  - histogram of chosen migration barriers (eV),
  - local order vs T (sweep mode),
  - jump randomness `R`,
  - cluster/order evolution (`Fe`, `Ni`, `Cr`),
  - `2D Vacancy Migraation`,
  - `3D Vacancy Migration`.
- After a sweep run, non-`T` observables include a temperature selector so you can choose which sweep temperature to display.
- Optional rendered media observables written per run:
  - `frames_2D` (XY vacancy trajectory),
  - `frames_3D` (XYZ vacancy trajectory).

### Project objective alignment
This implementation targets the observables described in the project document:
- Diffusion metrics (MSD, tracer diffusion vs time `D(t)=MSD/(6t)`, diffusion estimate from MSD slope).
- Local chemical order proxy (vacancy-shell composition + SRO-like scalar).
- Local chemical-order sweep plot analogous to legacy `xij_norm_vs_T` (`XN_ij` vs `T` for Fe-Fe, Fe-Ni, Fe-Cr, Ni-Ni, Ni-Cr, Cr-Cr).
- Jump randomness metrics (entropy and normalized randomness index `R`).
- Jump randomness visualization with a vacancy-site `R` map and `R` probability histogram.
- Cluster/order evolution metrics over KMC steps (`Fe`, `Ni`, `Cr` largest connected-component sizes).

## 2) Directory Layout

```text
Production/
  README.md
  requirements.txt
  data/
    df_atoms_fcc_FeNiCr.data
  models/
    example/
      best_model_10TypeFeatures_best.pth
      example_model.meta.json
      encoder_model.joblib
    legacy/
      mlp_best.pt
      mlp_best.meta.json
  production_kmc/
    __init__.py
    constants.py
    config.py
    structure.py
    io_lammps.py
    descriptor.py
    ann.py
    engine.py
    observables.py
    media.py
    plotting.py
    simulation.py
    gui/
      __init__.py
      app.py
  scripts/
    run_cli.py
    run_gui.py
    run_sweep.py

```

## 3) Core Architecture

### `structure.py`
- Defines `AtomicState` (positions/types/ids/bounds).
- Generates random FCC structures with composition control.
- Enforces exactly one vacancy (`type == 0`).

### `io_lammps.py`
- Robust parser for LAMMPS data files (`atomic` and common `charge` style atoms section).
- Writer for `Atoms # atomic` output snapshots.

### `descriptor.py`
- Loads descriptor layout from metadata.
- Supports canonical fallback feature schema from shell counts.
- Builds one-time neighbor shell maps per site under PBC.
- Encodes vacancy environment into ANN input vector (600 features for canonical 10-shell schema).

### `ann.py`
- Loads checkpoint + architecture from metadata (or infers hidden sizes if missing).
- Performs ANN inference and scales normalized outputs via `target_max_values`.
- Uses LRU-like descriptor-byte cache to avoid repeated ANN calls for recurring environments.

### `engine.py`
- Residence-time KMC event selection (`p_i = k_i/sum(k)`; `dt = -ln(u)/sum(k)`).
- Vacancy-atom swap execution.
- Atom-identity displacement bookkeeping for **atomic MSD** (not vacancy MSD).
- Optional detailed-balance surrogate barrier adjustment using user-provided 3x3 interaction matrix.

### `observables.py`
- Computes per-step observables:
  - atomic MSD,
  - tracer diffusion vs time via Einstein ratio `D(t)=MSD(t)/(6t)` (computed from existing MSD/time arrays),
  - hop entropy,
  - jump randomness index `R`,
  - vacancy-shell composition (Fe/Ni/Cr fractions),
  - SRO-like proxy,
  - largest connected-component sizes (all and by species) on 1NN graph.
- Estimates diffusion coefficient from MSD-time slope on late trajectory segment.

### Cluster / Order Evolution
- `cluster_sizes_vs_steps.png` is a line plot of largest connected-component sizes over KMC steps for `Fe`, `Ni`, and `Cr`.

### `simulation.py`
- Orchestrates complete run:
  - validation,
  - structure load/generation,
  - descriptor/neighbor setup,
  - ANN predictor setup,
  - KMC loop,
  - logging/output writing,
  - plotting.

### `media.py`
- Renders structure-evolution videos from saved type snapshots:
  - `frames_2D.mp4` (or `.gif` fallback) in XY projection,
  - `frames_3D.mp4` (or `.gif` fallback) in XYZ.
- Subsamples long runs to `video_max_frames` for consistent rendering cost.

### `gui/app.py`
- Full PyQt GUI with worker thread and live updates.
- Uses the same `run_simulation` backend as CLI for consistency.
- Non-video observables are image-backed from run outputs (final-only refresh policy).

## 4) Efficiency Improvements vs. Prior Code

The production implementation is optimized in several key ways:

1. **Neighbor shells precomputed once**:
- No repeated shell reconstruction from DataFrames every KMC step.

2. **Descriptor encoding is array-based**:
- Avoids repeated pandas one-hot operations in the hot loop.

3. **Persistent ANN cache**:
- Repeated local environments are served from cache instead of re-running inference.

4. **Array-based state updates**:
- Vacancy swaps and event-rate logic run on numpy arrays, reducing overhead.

5. **Single backend for GUI and CLI**:
- Removes duplicated simulation codepaths and drift.

## 5) Installation

From repository root:

```bash
python3 -m pip install -r Production/requirements.txt
```

Notes:
- `run_gui.py` now configures local writable matplotlib/font caches under `Production/.cache` to avoid cache-permission issues.
- If you previously installed dependencies with `pyparsing>=3.3`, reinstall from `requirements.txt` before running GUI on Python 3.14.

## 6) Running

### GUI

```bash
python3 Production/scripts/run_gui.py
```

### CLI

Example using file input:

```bash
python3 Production/scripts/run_cli.py \
  --mode file \
  --data Production/data/df_atoms_fcc_FeNiCr.data \
  --ckpt Production/models/example/best_model_10TypeFeatures_best.pth \
  --meta Production/models/example/example_model.meta.json \
  --T 1000 \
  --nu_log10 13 \
  --steps 2000 \
  --save_every 25 \
  --video_max_frames 300 \
  --video_fps 12 \
  --out_dir Production/runs/cli_run
```

Example using generated structure:

```bash
python3 Production/scripts/run_cli.py \
  --mode generated \
  --num_atoms 500 \
  --fe_pct 33.34 --ni_pct 33.33 --cr_pct 33.33 \
  --vacancy_atom_id -1 \
  --steps 1000 \
  --out_dir Production/runs/generated_run
```

Disable media observables for a run:

```bash
python3 Production/scripts/run_cli.py --no_frame_videos --out_dir Production/runs/no_media_run
```

### Temperature Sweep

```bash
python3 Production/scripts/run_sweep.py \
  --mode file \
  --data Production/data/df_atoms_fcc_FeNiCr.data \
  --Tmin 600 --Tmax 1400 --Tstep 200 \
  --nu_log10 13 \
  --steps 2000 \
  --frame_videos \
  --out_root Production/runs/sweep
```

## 7) Outputs

Each run directory contains:
- `run_config.json`: normalized full configuration.
- `structure_initial.data`: initial state snapshot.
- `structure_final.data`: final state snapshot.
- `time_log.csv`: per-step timing and chosen-event records.
- `observables.csv`: per-step observables.
- `trajectory_frames.npz`: saved type snapshots (`types_frames`, `frame_steps`, `frame_times_s`) used for video observables.
- `frames_2D.mp4` or `frames_2D.gif`: 2D vacancy-only trajectory observable.
- `frames_3D.mp4` or `frames_3D.gif`: 3D vacancy-only trajectory observable.
- `xij_by_T.csv` (sweep): local-order pair statistics per temperature (`X_ij`, `N_ij`, `N0_ij`, `XN_ij`).
- `xij_norm_vs_T.png` (sweep): legacy-style local chemical order plot (`XN_ij` vs `T`).
- `tracer_diffusion_vs_time.png`: tracer diffusion observable using `D(t)=MSD(t)/(6t)`.
- `migration_barrier_histogram.png`: histogram of chosen migration barriers from `time_log.csv`.
- `jump_randomness_map_and_distribution.png`: jump-randomness map + probability histogram.
- `cluster_sizes_vs_steps.png`: cluster/order evolution (`Fe`, `Ni`, `Cr` largest cluster sizes vs KMC steps).
- `events_stepXXXXXX.csv`: saved candidate events at save cadence.
- `summary.json`: final summary, including diffusion estimate.
- PNG plots for standard observables.

## 8) Metadata Format Expectations

Metadata (`.json`) should contain at least:
- `input_dim`
- `output_dim`
- `target_max_values`
- `hidden_sizes` (recommended)
- either `feature_columns` or `shell_counts`
- optional `barrier_columns`

`example_model.meta.json` is included for the Example checkpoint.

## 9) Detailed-Balance Surrogate (Optional)

If enabled, the engine applies a local barrier adjustment using a user-provided 3x3 interaction matrix (eV):
- estimate local energy difference for moving species between hop site and vacancy site,
- adjust forward barrier conservatively.

This is an optional surrogate for consistency checks and sensitivity studies.

## 10) Tests

Run:

```bash
PYTHONPATH=Production python3 -m unittest discover Production/tests
```

## 11) Notes

- ANN inference requires PyTorch.
- GUI requires PyQt5.
- `frames_2D/frames_3D` rendering prefers Pillow GIF output for direct GUI preview and falls back to `ffmpeg` MP4 when needed.
- The included default model artifacts are set to the Example checkpoint/metadata as requested.
