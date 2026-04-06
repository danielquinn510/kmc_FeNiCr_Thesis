from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from matplotlib import animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


@dataclass
class FrameArtifacts:
    frames_2d: Optional[Path]
    frames_3d: Optional[Path]


def _subsample_indices(n_frames: int, max_frames: int) -> np.ndarray:
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=int)
    idx = np.linspace(0, n_frames - 1, num=max_frames, dtype=int)
    return np.unique(idx)


def _writer_factory_and_ext(fps: int) -> tuple[Callable[[], animation.AbstractMovieWriter], str]:
    if animation.writers.is_available("pillow"):
        return lambda: animation.PillowWriter(fps=max(1, min(fps, 20))), ".gif"
    if animation.writers.is_available("ffmpeg"):
        return lambda: animation.FFMpegWriter(fps=fps, bitrate=2000), ".mp4"
    raise RuntimeError("No animation writer available (ffmpeg/pillow)")


def _vacancy_positions_per_frame(types_frames: np.ndarray, positions: np.ndarray) -> np.ndarray:
    vacancy_mask = types_frames == 0
    vacancy_counts = vacancy_mask.sum(axis=1)
    if np.any(vacancy_counts != 1):
        raise ValueError("Each frame must contain exactly one vacancy site (type == 0).")
    vacancy_indices = np.argmax(vacancy_mask, axis=1)
    return positions[vacancy_indices]


def render_frames_2d_3d(
    positions: np.ndarray,
    bounds: np.ndarray,
    types_frames: np.ndarray,
    frame_steps: np.ndarray,
    out_dir: Path,
    max_frames: int = 300,
    fps: int = 12,
) -> FrameArtifacts:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if types_frames.ndim != 2:
        raise ValueError("types_frames must have shape (F, N)")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if types_frames.shape[1] != positions.shape[0]:
        raise ValueError("types_frames second dimension must match number of positions")

    n_frames = int(types_frames.shape[0])
    if n_frames < 2:
        return FrameArtifacts(frames_2d=None, frames_3d=None)

    max_frames = max(2, int(max_frames))
    fps = max(1, int(fps))
    keep = _subsample_indices(n_frames, max_frames)

    types_sel = types_frames[keep]
    steps_sel = frame_steps[keep]
    vacancy_positions = _vacancy_positions_per_frame(types_sel, positions)

    writer_factory, ext = _writer_factory_and_ext(fps)

    xlo, xhi = bounds[0]
    ylo, yhi = bounds[1]
    zlo, zhi = bounds[2]
    x_v = vacancy_positions[:, 0]
    y_v = vacancy_positions[:, 1]
    z_v = vacancy_positions[:, 2]

    # 2D animation (xy vacancy position only)
    fig2 = Figure(figsize=(7.2, 6.0))
    FigureCanvasAgg(fig2)
    ax2 = fig2.add_subplot(111)
    marker2 = ax2.scatter([x_v[0]], [y_v[0]], c="#d62728", s=48, edgecolors="none")
    ax2.set_xlim(float(xlo), float(xhi))
    ax2.set_ylim(float(ylo), float(yhi))
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(alpha=0.2)

    def _update_2d(i: int):
        marker2.set_offsets(np.array([[x_v[i], y_v[i]]], dtype=float))
        ax2.set_title(f"2D Vacancy Migration | step={int(steps_sel[i])}")
        return (marker2,)

    anim2 = animation.FuncAnimation(fig2, _update_2d, frames=len(steps_sel), interval=1000 / fps, blit=False)
    out2 = out_dir / f"frames_2D{ext}"
    anim2.save(str(out2), writer=writer_factory())

    # 3D animation (vacancy position only)
    fig3 = Figure(figsize=(7.2, 6.0))
    FigureCanvasAgg(fig3)
    ax3 = fig3.add_subplot(111, projection="3d")
    marker3 = ax3.scatter([x_v[0]], [y_v[0]], [z_v[0]], c="#d62728", s=52, depthshade=False)
    ax3.set_xlim(float(xlo), float(xhi))
    ax3.set_ylim(float(ylo), float(yhi))
    ax3.set_zlim(float(zlo), float(zhi))
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")

    def _update_3d(i: int):
        marker3._offsets3d = (  # Matplotlib 3D scatter update API
            np.array([x_v[i]], dtype=float),
            np.array([y_v[i]], dtype=float),
            np.array([z_v[i]], dtype=float),
        )
        ax3.set_title(f"3D Vacancy Migration | step={int(steps_sel[i])}")
        return (marker3,)

    anim3 = animation.FuncAnimation(fig3, _update_3d, frames=len(steps_sel), interval=1000 / fps, blit=False)
    out3 = out_dir / f"frames_3D{ext}"
    anim3.save(str(out3), writer=writer_factory())

    return FrameArtifacts(frames_2d=out2, frames_3d=out3)
