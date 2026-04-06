from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .observables import (
    PAIR_TYPE_KEYS,
    pair_label,
    tracer_diffusion_vs_time,
)


def _save_fig(fig, path: Path, tight: bool = True, dpi: float = 180.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=dpi)


def render_standard_plots(
    observables_csv: Path,
    out_dir: Path,
    time_log_csv: Path | None = None,
) -> None:
    df = pd.read_csv(observables_csv)
    if df.empty:
        return

    # MSD
    fig = Figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(df["time_s"], df["msd_a2"], lw=1.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("MSD (A^2)")
    ax.set_title("Atomic Mean Squared Displacement")
    ax.grid(alpha=0.3)
    _save_fig(fig, out_dir / "msd_vs_time.png")

    # Tracer diffusion D(t) from Einstein ratio
    fig = Figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    d_t = tracer_diffusion_vs_time(
        times_s=df["time_s"].to_numpy(),
        msd_a2=df["msd_a2"].to_numpy(),
        dimensions=3,
    )
    valid = pd.notna(d_t)
    if valid.any():
        ax.plot(df.loc[valid, "time_s"], d_t[valid], lw=1.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Tracer diffusion D(t) (A^2/s)")
    ax.set_title("Tracer Diffusion vs Time")
    ax.grid(alpha=0.3)
    _save_fig(fig, out_dir / "tracer_diffusion_vs_time.png")

    # Migration barrier histogram
    barrier_vals = np.array([], dtype=np.float64)
    barrier_source = time_log_csv if time_log_csv is not None else (out_dir / "time_log.csv")
    if barrier_source.exists():
        try:
            tdf = pd.read_csv(barrier_source, usecols=["chosen_barrier_eV"])
            if "chosen_barrier_eV" in tdf.columns:
                raw = pd.to_numeric(tdf["chosen_barrier_eV"], errors="coerce").to_numpy(dtype=float)
                barrier_vals = raw[np.isfinite(raw)]
        except Exception:
            barrier_vals = np.array([], dtype=np.float64)

    fig = Figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    if barrier_vals.size > 0:
        bins = int(np.clip(np.sqrt(barrier_vals.size) * 2.0, 10, 60))
        ax.hist(
            barrier_vals,
            bins=bins,
            color="#1f77b4",
            alpha=0.85,
            edgecolor="white",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No migration-barrier data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    ax.set_xlabel("Migration barrier Ea (eV)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Chosen Migration Barriers")
    ax.grid(alpha=0.3)
    _save_fig(fig, out_dir / "migration_barrier_histogram.png")

    # Jump randomness: spatial map + statistical distribution
    fig = Figure(figsize=(7, 7))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 0.05],
        height_ratios=[2.2, 1.2],
        hspace=0.35,
        wspace=0.08,
    )
    ax_map = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_hist_pad = fig.add_subplot(gs[1, 1])
    ax_hist_pad.axis("off")

    cols = {"vacancy_x", "vacancy_y", "hop_randomness_R"}
    if cols.issubset(df.columns):
        valid = (
            pd.notna(df["vacancy_x"])
            & pd.notna(df["vacancy_y"])
            & pd.notna(df["hop_randomness_R"])
        )
        if valid.any():
            x = df.loc[valid, "vacancy_x"].to_numpy(dtype=float)
            y = df.loc[valid, "vacancy_y"].to_numpy(dtype=float)
            r = df.loc[valid, "hop_randomness_R"].to_numpy(dtype=float)

            # Aggregate repeated visits to the same lattice site for a cleaner map.
            xy = pd.DataFrame({"x": x, "y": y, "R": r})
            grouped = (
                xy.assign(x=xy["x"].round(8), y=xy["y"].round(8))
                .groupby(["x", "y"], as_index=False)
                .agg(R=("R", "mean"))
            )
            sc = ax_map.scatter(
                grouped["x"],
                grouped["y"],
                c=grouped["R"],
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=20,
                marker="s",
                linewidths=0.0,
            )
            cb = fig.colorbar(sc, cax=ax_cbar)
            cb.set_label("R")
            ax_map.set_xlabel("x")
            ax_map.set_ylabel("y")
            ax_map.set_title("Jump Randomness Map (vacancy-site averaged)")
            ax_map.set_aspect("auto")
            ax_map.grid(alpha=0.2)

            bins = np.linspace(0.0, 1.0, 21)
            weights = np.ones_like(r, dtype=float) / float(r.size)
            ax_hist.hist(
                r,
                bins=bins,
                weights=weights,
                color="#d8b4c6",
                edgecolor="#555555",
                alpha=0.9,
            )
            ax_hist.set_xlim(0.0, 1.0)
            ax_hist.set_xlabel("Lattice jump randomness R")
            ax_hist.set_ylabel("Probability")
            ax_hist.set_title("Statistical Distribution")
            ax_hist.grid(alpha=0.25, axis="y")
        else:
            ax_cbar.axis("off")
            ax_map.text(0.5, 0.5, "No valid jump-randomness map data", transform=ax_map.transAxes, ha="center", va="center")
            ax_map.set_title("Jump Randomness Map")
            ax_hist.text(0.5, 0.5, "No valid jump-randomness histogram data", transform=ax_hist.transAxes, ha="center", va="center")
            ax_hist.set_title("Statistical Distribution")
    else:
        ax_cbar.axis("off")
        ax_map.text(0.5, 0.5, "Jump-randomness map columns not found", transform=ax_map.transAxes, ha="center", va="center")
        ax_map.set_title("Jump Randomness Map")
        ax_hist.text(0.5, 0.5, "Jump-randomness histogram columns not found", transform=ax_hist.transAxes, ha="center", va="center")
        ax_hist.set_title("Statistical Distribution")

    _save_fig(fig, out_dir / "jump_randomness_map_and_distribution.png")

    # Clusters: filter out unsampled rows (cluster=-1)
    sampled = df[df["cluster_all"] >= 0]
    if not sampled.empty:
        fig = Figure(figsize=(7, 4.5))
        ax = fig.add_subplot(111)
        ax.plot(sampled["step"], sampled["cluster_Fe"], label="Fe", lw=1.2)
        ax.plot(sampled["step"], sampled["cluster_Ni"], label="Ni", lw=1.2)
        ax.plot(sampled["step"], sampled["cluster_Cr"], label="Cr", lw=1.2)
        ax.set_xlabel("KMC steps")
        ax.set_ylabel("Largest cluster size")
        ax.set_title("Cluster / Order Evolution")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        _save_fig(fig, out_dir / "cluster_sizes_vs_steps.png")


def render_local_order_vs_temperature(
    xij_df: pd.DataFrame,
    out_path: Path,
    normalized: bool = True,
) -> None:
    if xij_df.empty or "T" not in xij_df.columns:
        return

    df = xij_df.sort_values("T")
    prefix = "XN_" if normalized else "X_"

    fig = Figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    for pair in PAIR_TYPE_KEYS:
        label = pair_label(pair)
        col = f"{prefix}{label}"
        if col in df.columns:
            ax.plot(df["T"], df[col], marker="o", lw=1.8, label=label)

    ax.axhline(0.0, linestyle="--", alpha=0.55, color="#4f8abf")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Order parameter δij")
    ax.set_title("Local chemical order")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    _save_fig(fig, out_path)
