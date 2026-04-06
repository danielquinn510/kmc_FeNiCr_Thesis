from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Optional
import re

import numpy as np
import pandas as pd
import matplotlib as mpl

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover - GUI runtime dependency
    raise RuntimeError("PyQt5 is required to run the Production GUI.") from exc

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import image as mpimg

from ..config import ModelSettings, RunSettings, SimulationSettings, StructureSettings
from ..descriptor import build_neighbor_shells
from ..io_lammps import read_lammps_data
from ..observables import (
    PAIR_TYPE_KEYS,
    local_order_pair_statistics,
    pair_label,
)
from ..plotting import render_local_order_vs_temperature
from ..simulation import RunResult, SimulationCallbacks, run_simulation

APP_TITLE = "FeNiCr KMC Simulation"

# Defensive workaround for Python 3.14 + pyparsing recursion crashes seen in some
# matplotlib mathtext paths on startup. GUI labels do not rely on mathtext.
mpl.rcParams["text.parse_math"] = False
mpl.rcParams["axes.formatter.use_mathtext"] = False
mpl.rcParams["text.usetex"] = False

GUI_DEFAULT_PATHS = {
    "data_path": Path("Production/data/df_atoms_fcc_FeNiCr.data"),
    "checkpoint_path": Path("Production/models/example/best_model_10TypeFeatures_best.pth"),
    "metadata_path": Path("Production/models/example/example_model.meta.json"),
    "output_dir": Path("Production/runs/gui_run"),
}

GUI_DEFAULTS = {
    "window_size": (980, 700),
    "splitter_sizes": (400, 580),
    "left_panel_min_width": 380,
    "left_panel_max_width": 940,
    "structure_source": "file",  # file | generated
    "generated_num_atoms": 500,
    "composition_percent": {"Fe": 33.34, "Ni": 33.33, "Cr": 33.33},
    "vacancy_atom_id": -1,  # -1 means auto
    "structure_seed": 123,
    "cache_size": 200_000,
    "temperature_K": 1000.0,
    "attempt_frequency_log10": 13.0,
    "enable_temp_sweep": False,
    "sweep_t_min": 600.0,
    "sweep_t_max": 1400.0,
    "sweep_t_step": 200.0,
    "num_steps": 1000,
    "steps_per_save": 1,
    "kmc_seed": 123,
    "generate_frame_videos": True,
    "video_max_frames": 300,
    "video_fps": 12,
    "enable_detailed_balance": False,
    "interaction_matrix_text": "[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]",
}

IMAGE_BACKED_METRIC_FILES: dict[str, str] = {
    "MSD (A^2)": "msd_vs_time.png",
    "Tracer Diffusion D(t)": "tracer_diffusion_vs_time.png",
    "Migration Barrier Histogram (eV)": "migration_barrier_histogram.png",
    "Jump Randomness R": "jump_randomness_map_and_distribution.png",
    "Cluster / Order Evolution": "cluster_sizes_vs_steps.png",
}

SUPPRESSED_LOG_SUBSTRINGS = (
    "local-order",
    "xij_by_t",
    "xij_norm_vs_t",
)


def _initial_window_rect(default_size: tuple[int, int]) -> Optional[QtCore.QRect]:
    app = QtWidgets.QApplication.instance()
    screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
    if screen is None and app is not None:
        screen = app.primaryScreen()
    if screen is None:
        return None

    available = screen.availableGeometry()
    default_w, default_h = int(default_size[0]), int(default_size[1])
    width = min(default_w, int(round(available.width() * 0.95)))
    height = min(default_h, int(round(available.height() * 0.95)))
    width = max(1, width)
    height = max(1, height)
    x = int(available.x() + (available.width() - width) / 2)
    y = int(available.y() + (available.height() - height) / 2)
    return QtCore.QRect(x, y, width, height)


def _preload_torch_main_thread() -> str:
    """Import torch on the main thread to avoid thread-import crashes on some builds."""
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "PyTorch import failed in GUI process. Install/repair torch before running simulations."
        ) from exc
    return str(torch.__version__)


class ObservableCanvas(FigureCanvasQTAgg):
    _BASE_WIDTH_PX = 420
    _BASE_HEIGHT_PX = 320

    def __init__(self, parent=None):
        fig = Figure(figsize=(8.0, 6.0), dpi=120)
        super().__init__(fig)
        self.setParent(parent)
        self._native_image_size: Optional[tuple[int, int]] = None
        self._clear_native_image_mode()

    def sizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        if self._native_image_size is not None:
            return QtCore.QSize(int(self._native_image_size[0]), int(self._native_image_size[1]))
        # Keep the default observables area compact at startup; large PNGs are handled
        # by native-image mode + scrollbars when loaded.
        return QtCore.QSize(self._BASE_WIDTH_PX, self._BASE_HEIGHT_PX)

    def minimumSizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        if self._native_image_size is not None:
            return QtCore.QSize(int(self._native_image_size[0]), int(self._native_image_size[1]))
        return QtCore.QSize(320, 240)

    def _reset_subplot_layout_defaults(self) -> None:
        # Matplotlib keeps subplotpars across figure.clear(); reset explicitly so
        # full-bleed image settings do not leak into standard plots.
        self.figure.subplots_adjust(
            left=0.125,
            right=0.9,
            bottom=0.11,
            top=0.88,
            wspace=0.2,
            hspace=0.2,
        )

    def _sync_figure_size_to_widget(self) -> None:
        w_px = max(1, int(self.width()))
        h_px = max(1, int(self.height()))
        dpi = float(self.figure.dpi) if float(self.figure.dpi) > 0.0 else 120.0
        w_in = max(4.0, float(w_px) / dpi)
        h_in = max(3.0, float(h_px) / dpi)
        cur_w, cur_h = self.figure.get_size_inches()
        if abs(cur_w - w_in) > 0.05 or abs(cur_h - h_in) > 0.05:
            self.figure.set_size_inches(w_in, h_in, forward=False)

    def _set_native_image_mode(self, size_px: tuple[int, int]) -> None:
        w, h = max(1, int(size_px[0])), max(1, int(size_px[1]))
        self._native_image_size = (w, h)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumSize(w, h)
        self.setMaximumSize(w, h)
        # Keep a subtle border tightly around the rendered image instead of
        # around the full (potentially much taller) scroll viewport.
        self.setStyleSheet("background: #ffffff; border: 1px solid #666;")
        self.resize(w, h)
        self.updateGeometry()

    def _clear_native_image_mode(self) -> None:
        self._native_image_size = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setMinimumSize(320, 240)
        self.setMaximumSize(QtWidgets.QWIDGETSIZE_MAX, QtWidgets.QWIDGETSIZE_MAX)
        self.setStyleSheet("")
        dpi = float(self.figure.dpi) if float(self.figure.dpi) > 0.0 else 120.0
        self.figure.set_size_inches(
            float(self._BASE_WIDTH_PX) / dpi,
            float(self._BASE_HEIGHT_PX) / dpi,
            forward=False,
        )
        self.resize(self._BASE_WIDTH_PX, self._BASE_HEIGHT_PX)
        self.updateGeometry()

    def plot_metric(self, records: list[dict], metric: str) -> None:
        self.plot_message(
            "This observable is image-backed and is rendered after run completion.",
            title=f"{metric}",
        )

    def plot_message(self, message: str, title: str | None = None) -> None:
        self._clear_native_image_mode()
        self._sync_figure_size_to_widget()
        self.figure.clear()
        self._reset_subplot_layout_defaults()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        if title:
            ax.set_title(title)
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
        self.draw_idle()

    def plot_image(
        self,
        image_path: Path,
        title: str | None = None,
        fit_within_px: Optional[tuple[int, int]] = None,
    ) -> None:
        try:
            img = mpimg.imread(str(image_path))
            # Trim uniform whitespace borders (common for pre-rendered figure images)
            # so static observables fill the canvas similarly to live matplotlib plots.
            if img.ndim >= 2:
                rgb = img[..., :3] if img.ndim == 3 else img
                if rgb.ndim == 2:
                    content_mask = rgb < 0.995
                else:
                    content_mask = np.any(rgb < 0.995, axis=-1)
                ys, xs = np.where(content_mask)
                if ys.size > 0 and xs.size > 0:
                    y0 = max(0, int(np.min(ys)) - 2)
                    y1 = min(int(img.shape[0]), int(np.max(ys)) + 3)
                    x0 = max(0, int(np.min(xs)) - 2)
                    x1 = min(int(img.shape[1]), int(np.max(xs)) + 3)
                    img = img[y0:y1, x0:x1]

            h_px = int(img.shape[0])
            w_px = int(img.shape[1])
            target_w = w_px
            target_h = h_px
            if fit_within_px is not None:
                fit_w = max(1, int(fit_within_px[0]))
                fit_h = max(1, int(fit_within_px[1]))
                scale = min(1.0, float(fit_w) / float(w_px), float(fit_h) / float(h_px))
                target_w = max(1, int(round(float(w_px) * scale)))
                target_h = max(1, int(round(float(h_px) * scale)))

            dpi = float(self.figure.dpi) if float(self.figure.dpi) > 0.0 else 120.0
            self.figure.set_size_inches(float(target_w) / dpi, float(target_h) / dpi, forward=False)
            self._set_native_image_mode((target_w, target_h))
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            ax.set_position([0.0, 0.0, 1.0, 1.0])
            ax.imshow(img, aspect="equal")
            ax.axis("off")
            if title:
                ax.set_title(title)
        except Exception:
            self._clear_native_image_mode()
            self._sync_figure_size_to_widget()
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_title("Failed to load image observable")
            ax.text(0.5, 0.5, str(image_path), transform=ax.transAxes, ha="center", va="center")
            ax.axis("off")
        self.draw_idle()

    def plot_local_order_vs_temperature(self, xij_df: Optional[pd.DataFrame]) -> None:
        self._sync_figure_size_to_widget()
        self.figure.clear()
        self._reset_subplot_layout_defaults()
        ax = self.figure.add_subplot(111)

        if xij_df is None or xij_df.empty or "T" not in xij_df.columns:
            ax.set_title("No sweep local-order data yet")
            self.draw_idle()
            return

        df = xij_df.sort_values("T")
        for pair in PAIR_TYPE_KEYS:
            label = pair_label(pair)
            col = f"XN_{label}"
            if col in df.columns:
                ax.plot(df["T"], df[col], marker="o", lw=1.8, label=label)

        ax.axhline(0.0, linestyle="--", alpha=0.55, color="#4f8abf")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Order parameter δij")
        ax.set_title("Local chemical order")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        self.draw_idle()


class SimulationWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int)
    log = QtCore.pyqtSignal(str)
    observable = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self._stop_event = threading.Event()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self._run_impl()
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")

    def stop(self):
        self._stop_event.set()

    def _make_settings(
        self,
        temperature_override: Optional[float] = None,
        output_dir_override: Optional[Path] = None,
    ) -> RunSettings:
        structure_mode = "generated" if self.cfg["use_generated"] else "file"
        structure = StructureSettings(
            mode=structure_mode,
            data_path=None if self.cfg["use_generated"] else Path(self.cfg["data_path"]),
            num_atoms=int(self.cfg["num_atoms"]),
            composition_percent={
                "Fe": float(self.cfg["fe_pct"]),
                "Ni": float(self.cfg["ni_pct"]),
                "Cr": float(self.cfg["cr_pct"]),
            },
            vacancy_atom_id=self.cfg["vacancy_atom_id"],
            random_seed=int(self.cfg["structure_seed"]),
        )

        interaction_matrix = None
        if self.cfg["enable_detailed_balance"]:
            raw = self.cfg["interaction_matrix_text"].strip()
            if raw:
                interaction_matrix = json.loads(raw)
            else:
                interaction_matrix = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        return RunSettings(
            model=ModelSettings(
                checkpoint_path=Path(self.cfg["checkpoint_path"]),
                metadata_path=Path(self.cfg["metadata_path"]),
                device="cpu",
                cache_size=int(self.cfg["cache_size"]),
            ),
            structure=structure,
            simulation=SimulationSettings(
                temperature_K=float(
                    self.cfg["temperature_K"] if temperature_override is None else temperature_override
                ),
                attempt_frequency_s_inv=float(self.cfg["attempt_frequency"]),
                num_steps=int(self.cfg["num_steps"]),
                steps_per_save=int(self.cfg["steps_per_save"]),
                generate_frame_videos=bool(self.cfg["generate_frame_videos"]),
                video_max_frames=int(self.cfg["video_max_frames"]),
                video_fps=int(self.cfg["video_fps"]),
                output_dir=Path(self.cfg["output_dir"]) if output_dir_override is None else Path(output_dir_override),
                random_seed=int(self.cfg["kmc_seed"]),
                enable_detailed_balance=bool(self.cfg["enable_detailed_balance"]),
                interaction_matrix_eV=interaction_matrix,
            ),
        )

    def _temperature_grid(self) -> np.ndarray:
        tmin = float(self.cfg["sweep_t_min"])
        tmax = float(self.cfg["sweep_t_max"])
        tstep = float(self.cfg["sweep_t_step"])
        if tstep <= 0.0:
            raise ValueError("Sweep temperature step must be > 0")
        if tmax < tmin:
            raise ValueError("Sweep Tmax must be >= Tmin")

        temps = np.arange(tmin, tmax + 1e-12, tstep, dtype=float)
        if temps.size == 0:
            raise ValueError("Sweep temperature grid is empty")
        return temps

    def _local_order_row_from_run_dir(self, run_dir: Path) -> dict:
        state = read_lammps_data(run_dir / "structure_final.data").state
        nn1 = build_neighbor_shells(
            positions=state.positions,
            bounds=state.bounds,
            shell_counts={1: 12},
        ).shell_neighbors[1]
        return local_order_pair_statistics(state.types, nn1)

    def _run_single(self) -> None:
        settings = self._make_settings()

        callbacks = SimulationCallbacks(
            on_log=self.log.emit,
            on_progress=lambda done, total: self.progress.emit(int(done), int(total)),
            on_observable=lambda rec: self.observable.emit(rec.__dict__),
        )

        result: RunResult = run_simulation(
            settings=settings,
            callbacks=callbacks,
            stop_requested=self._stop_event.is_set,
        )

        self.finished.emit(
            {
                "output_dir": str(result.output_dir),
                "steps_completed": result.steps_completed,
                "total_sim_time_s": result.total_sim_time_s,
                "diffusion_estimate": result.diffusion_estimate_a2_per_s,
                "stopped_early": result.stopped_early,
                "frames_2d_path": None if result.frames_2d_path is None else str(result.frames_2d_path),
                "frames_3d_path": None if result.frames_3d_path is None else str(result.frames_3d_path),
            }
        )

    def _run_sweep(self) -> None:
        temps = self._temperature_grid()
        out_root = Path(self.cfg["output_dir"])
        out_root.mkdir(parents=True, exist_ok=True)

        num_steps = int(self.cfg["num_steps"])
        total_progress_steps = int(max(1, num_steps * len(temps)))
        completed_steps = 0
        total_sim_time_s = 0.0

        sweep_rows: list[dict] = []
        xij_rows: list[dict] = []
        stopped = False

        for idx, T in enumerate(temps):
            if self._stop_event.is_set():
                stopped = True
                self.log.emit("Stop requested. Ending sweep.")
                break

            run_dir = out_root / f"T_{T:g}K"
            self.log.emit(f"=== Sweep {idx + 1}/{len(temps)}: T={T:g} K ===")
            settings = self._make_settings(temperature_override=float(T), output_dir_override=run_dir)

            progress_offset = int(completed_steps)
            callbacks = SimulationCallbacks(
                on_log=lambda msg, t=float(T): self.log.emit(f"[T={t:g}K] {msg}"),
                on_progress=lambda done, total, offset=progress_offset: self.progress.emit(
                    int(offset + done), total_progress_steps
                ),
                on_observable=None,
            )

            result = run_simulation(
                settings=settings,
                callbacks=callbacks,
                stop_requested=self._stop_event.is_set,
            )
            completed_steps += int(result.steps_completed)
            total_sim_time_s += float(result.total_sim_time_s)
            self.progress.emit(int(completed_steps), total_progress_steps)

            final_row = {}
            obs_path = run_dir / "observables.csv"
            if obs_path.exists():
                obs = pd.read_csv(obs_path)
                if not obs.empty:
                    final_row = obs.iloc[-1].to_dict()

            sweep_rows.append(
                {
                    "T": float(T),
                    "steps_completed": result.steps_completed,
                    "total_sim_time_s": result.total_sim_time_s,
                    "diffusion_estimate_a2_per_s": result.diffusion_estimate_a2_per_s,
                    **{f"final_{k}": v for k, v in final_row.items()},
                }
            )

            try:
                xij_rows.append({"T": float(T), **self._local_order_row_from_run_dir(run_dir)})
            except Exception as exc:
                self.log.emit(f"[warn] Failed local-order analysis at T={T:g} K: {exc}")

            if result.stopped_early:
                stopped = True
                break

        summary_path = out_root / "sweep_summary.csv"
        xij_csv_path = out_root / "xij_by_T.csv"
        xij_plot_path = out_root / "xij_norm_vs_T.png"

        if sweep_rows:
            pd.DataFrame(sweep_rows).sort_values("T").to_csv(summary_path, index=False)

        xij_written = False
        if xij_rows:
            xij_df = pd.DataFrame(xij_rows).sort_values("T")
            xij_df.to_csv(xij_csv_path, index=False)
            render_local_order_vs_temperature(
                xij_df=xij_df,
                out_path=xij_plot_path,
                normalized=True,
            )
            xij_written = True

        self.finished.emit(
            {
                "output_dir": str(out_root),
                "steps_completed": int(completed_steps),
                "total_sim_time_s": float(total_sim_time_s),
                "diffusion_estimate": None,
                "stopped_early": bool(stopped),
                "frames_2d_path": None,
                "frames_3d_path": None,
                "sweep_mode": True,
                "sweep_summary_path": str(summary_path) if summary_path.exists() else None,
                "xij_csv_path": str(xij_csv_path) if xij_written else None,
                "xij_plot_path": str(xij_plot_path) if xij_written else None,
            }
        )

    def _run_impl(self):
        if bool(self.cfg.get("enable_temp_sweep", False)):
            self._run_sweep()
            return
        self._run_single()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        rect = _initial_window_rect(tuple(GUI_DEFAULTS["window_size"]))
        if rect is None:
            self.resize(*GUI_DEFAULTS["window_size"])
        else:
            self.setGeometry(rect)

        self._worker: Optional[SimulationWorker] = None
        self._thread: Optional[QtCore.QThread] = None
        self._observable_records: list[dict] = []
        self._media_movie: Optional[QtGui.QMovie] = None
        self._last_run_output_dir: Optional[Path] = None
        self._last_xij_csv_path: Optional[Path] = None
        self._last_sweep_mode: bool = False
        self._sweep_temp_dirs: dict[str, Path] = {}

        root = QtWidgets.QWidget(self)
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        self.left_panel = self._build_left_panel()
        self.right_panel = self._build_right_panel()
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes(list(GUI_DEFAULTS["splitter_sizes"]))

        layout.addWidget(splitter)

        self._sync_structure_mode()
        self._sync_sweep_mode()
        self._sync_frame_export()
        self._sync_temperature_selector()

    def _build_left_panel(self) -> QtWidgets.QWidget:
        defaults = GUI_DEFAULTS
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(int(defaults["left_panel_min_width"]))
        panel.setMaximumWidth(int(defaults["left_panel_max_width"]))
        panel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        outer = QtWidgets.QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(10, 12, 12, 12)
        content_layout.setSpacing(22)

        repo_root = Path(__file__).resolve().parents[3]

        # Structure source
        self.rb_from_file = QtWidgets.QRadioButton("Load existing LAMMPS .data")
        self.rb_generated = QtWidgets.QRadioButton("Generate random FCC structure")
        self.rb_from_file.setChecked(defaults["structure_source"] == "file")
        self.rb_generated.setChecked(defaults["structure_source"] == "generated")
        self.rb_from_file.toggled.connect(self._sync_structure_mode)
        src_box = QtWidgets.QWidget()
        src_lay = QtWidgets.QHBoxLayout(src_box)
        src_lay.setContentsMargins(0, 0, 0, 0)
        src_lay.setSpacing(14)
        src_lay.addWidget(self.rb_from_file)
        src_lay.addWidget(self.rb_generated)
        src_lay.addStretch(1)

        self.ed_data = QtWidgets.QLineEdit(str(repo_root / GUI_DEFAULT_PATHS["data_path"]))
        self.btn_data = self._file_button(self.ed_data, "Select LAMMPS data", "LAMMPS data (*.data);;All (*)")

        self.sp_atoms = QtWidgets.QSpinBox()
        self.sp_atoms.setRange(4, 2_000_000)
        self.sp_atoms.setSingleStep(4)
        self.sp_atoms.setValue(int(defaults["generated_num_atoms"]))

        self.sp_fe = QtWidgets.QDoubleSpinBox(); self._configure_pct_spin(self.sp_fe, defaults["composition_percent"]["Fe"])
        self.sp_ni = QtWidgets.QDoubleSpinBox(); self._configure_pct_spin(self.sp_ni, defaults["composition_percent"]["Ni"])
        self.sp_cr = QtWidgets.QDoubleSpinBox(); self._configure_pct_spin(self.sp_cr, defaults["composition_percent"]["Cr"])

        self.sp_vacancy = QtWidgets.QSpinBox()
        self.sp_vacancy.setRange(-1, 10_000_000)
        self.sp_vacancy.setValue(int(defaults["vacancy_atom_id"]))
        self.sp_vacancy.setToolTip("-1 = auto. Otherwise use atom id.")

        self.sp_structure_seed = QtWidgets.QSpinBox()
        self.sp_structure_seed.setRange(0, 2_147_483_647)
        self.sp_structure_seed.setValue(int(defaults["structure_seed"]))

        # Model + output
        self.ed_ckpt = QtWidgets.QLineEdit(str(repo_root / GUI_DEFAULT_PATHS["checkpoint_path"]))
        self.btn_ckpt = self._file_button(self.ed_ckpt, "Select checkpoint", "Checkpoint (*.pt *.pth);;All (*)")

        self.ed_meta = QtWidgets.QLineEdit(str(repo_root / GUI_DEFAULT_PATHS["metadata_path"]))
        self.btn_meta = self._file_button(self.ed_meta, "Select metadata", "Metadata (*.json);;All (*)")

        self.ed_out = QtWidgets.QLineEdit(str(repo_root / GUI_DEFAULT_PATHS["output_dir"]))
        self.btn_out = self._dir_button(self.ed_out, "Select output directory")

        self.sp_cache = QtWidgets.QSpinBox()
        self.sp_cache.setRange(0, 5_000_000)
        self.sp_cache.setValue(int(defaults["cache_size"]))

        # Simulation controls
        self.sp_temperature = QtWidgets.QDoubleSpinBox()
        self.sp_temperature.setRange(1.0, 5000.0)
        self.sp_temperature.setDecimals(2)
        self.sp_temperature.setValue(float(defaults["temperature_K"]))
        self.sp_temperature.setSuffix(" K")

        self.sp_nu_log10 = QtWidgets.QDoubleSpinBox()
        self.sp_nu_log10.setRange(0.0, 30.0)
        self.sp_nu_log10.setDecimals(2)
        self.sp_nu_log10.setSingleStep(0.25)
        self.sp_nu_log10.setValue(float(defaults["attempt_frequency_log10"]))
        self.sp_nu_log10.setToolTip("Attempt frequency is set as 10^X Hz.")

        self.chk_sweep = QtWidgets.QCheckBox("Enable temperature sweep")
        self.chk_sweep.setChecked(bool(defaults["enable_temp_sweep"]))
        self.chk_sweep.toggled.connect(self._sync_sweep_mode)

        self.sp_sweep_tmin = QtWidgets.QDoubleSpinBox()
        self.sp_sweep_tmin.setRange(1.0, 5000.0)
        self.sp_sweep_tmin.setDecimals(2)
        self.sp_sweep_tmin.setValue(float(defaults["sweep_t_min"]))
        self.sp_sweep_tmin.setSuffix(" K")

        self.sp_sweep_tmax = QtWidgets.QDoubleSpinBox()
        self.sp_sweep_tmax.setRange(1.0, 5000.0)
        self.sp_sweep_tmax.setDecimals(2)
        self.sp_sweep_tmax.setValue(float(defaults["sweep_t_max"]))
        self.sp_sweep_tmax.setSuffix(" K")

        self.sp_sweep_tstep = QtWidgets.QDoubleSpinBox()
        self.sp_sweep_tstep.setRange(0.1, 2000.0)
        self.sp_sweep_tstep.setDecimals(2)
        self.sp_sweep_tstep.setValue(float(defaults["sweep_t_step"]))
        self.sp_sweep_tstep.setSuffix(" K")

        self.sp_steps = QtWidgets.QSpinBox()
        self.sp_steps.setRange(1, 100_000_000)
        self.sp_steps.setValue(int(defaults["num_steps"]))

        self.sp_save_every = QtWidgets.QSpinBox()
        self.sp_save_every.setRange(1, 1_000_000)
        self.sp_save_every.setValue(int(defaults["steps_per_save"]))

        self.sp_kmc_seed = QtWidgets.QSpinBox()
        self.sp_kmc_seed.setRange(0, 2_147_483_647)
        self.sp_kmc_seed.setValue(int(defaults["kmc_seed"]))

        self.sp_video_max_frames = QtWidgets.QSpinBox()
        self.sp_video_max_frames.setRange(2, 200_000)
        self.sp_video_max_frames.setValue(int(defaults["video_max_frames"]))

        self.sp_video_fps = QtWidgets.QSpinBox()
        self.sp_video_fps.setRange(1, 120)
        self.sp_video_fps.setValue(int(defaults["video_fps"]))

        self.chk_frame_videos = QtWidgets.QCheckBox("Generate 2D/3D vacancy migration videos")
        self.chk_frame_videos.setChecked(bool(defaults["generate_frame_videos"]))
        self.chk_frame_videos.toggled.connect(self._sync_frame_export)

        self.chk_db = QtWidgets.QCheckBox("Enable detailed-balance surrogate")
        self.chk_db.setChecked(bool(defaults["enable_detailed_balance"]))
        self.ed_interactions = QtWidgets.QLineEdit(str(defaults["interaction_matrix_text"]))
        self.ed_interactions.setToolTip("3x3 JSON matrix in eV, used only when detailed balance is enabled.")

        self.btn_run = QtWidgets.QPushButton("Run Simulation")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_open = QtWidgets.QPushButton("Open Output Folder")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_open.clicked.connect(self._on_open_output)

        # composition row
        comp_row = QtWidgets.QWidget()
        comp_lay = QtWidgets.QHBoxLayout(comp_row)
        comp_lay.setContentsMargins(0, 0, 0, 0)
        comp_lay.setSpacing(6)
        comp_lay.addWidget(QtWidgets.QLabel("Fe"))
        comp_lay.addWidget(self.sp_fe)
        comp_lay.addWidget(QtWidgets.QLabel("Ni"))
        comp_lay.addWidget(self.sp_ni)
        comp_lay.addWidget(QtWidgets.QLabel("Cr"))
        comp_lay.addWidget(self.sp_cr)

        structure_box, structure_form = self._new_form_box("Atomic Structure")
        self._add_form_row(structure_form, "Structure source", src_box)
        self._add_form_row(structure_form, "LAMMPS .data", self._row(self.ed_data, self.btn_data))
        self._add_form_row(
            structure_form,
            "Atom count / Vacancy id",
            self._compact_pairs_row(
                [
                    ("Atoms", self.sp_atoms),
                    ("Vacancy id", self.sp_vacancy),
                ]
            ),
        )
        self._add_form_row(structure_form, "Composition (%)", comp_row)
        self._add_form_row(structure_form, "Structure seed", self.sp_structure_seed)
        content_layout.addWidget(structure_box)
        content_layout.addSpacing(8)

        model_box, model_form = self._new_form_box("Model and Output")
        self._add_form_row(model_form, "ANN checkpoint", self._row(self.ed_ckpt, self.btn_ckpt))
        self._add_form_row(model_form, "Model metadata", self._row(self.ed_meta, self.btn_meta))
        self._add_form_row(model_form, "Output directory", self._row(self.ed_out, self.btn_out))
        self._add_form_row(model_form, "Descriptor cache size", self.sp_cache)
        content_layout.addWidget(model_box)
        content_layout.addSpacing(8)

        sim_box, sim_form = self._new_form_box("Simulation Controls")
        self._add_form_row(
            sim_form,
            "Thermal controls",
            self._compact_pairs_row(
                [
                    ("T [K]", self.sp_temperature),
                    ("Attempt Frequency (10^X Hz)", self.sp_nu_log10),
                ]
            ),
        )
        self._add_form_row(sim_form, "Temperature sweep", self.chk_sweep)
        self._add_form_row(
            sim_form,
            "Sweep range [K]",
            self._compact_pairs_row(
                [
                    ("Tmin", self.sp_sweep_tmin),
                    ("Tmax", self.sp_sweep_tmax),
                    ("Step", self.sp_sweep_tstep),
                ]
            ),
        )
        self._add_form_row(
            sim_form,
            "KMC controls",
            self._compact_pairs_row(
                [
                    ("Steps", self.sp_steps),
                    ("Save every", self.sp_save_every),
                    ("Seed", self.sp_kmc_seed),
                ]
            ),
        )

        video_settings_row = self._compact_pairs_row(
            [
                ("Max frames", self.sp_video_max_frames),
                ("FPS", self.sp_video_fps),
            ]
        )
        self._add_form_row(sim_form, "Frame export", self.chk_frame_videos)
        self._add_form_row(sim_form, "Video settings", video_settings_row)

        self._add_form_row(sim_form, "Detailed balance", self.chk_db)
        self._add_form_row(sim_form, "Interaction matrix (JSON)", self.ed_interactions)
        content_layout.addWidget(sim_box)
        content_layout.addSpacing(8)

        actions_box, actions_form = self._new_form_box("Actions")
        self._add_form_row(actions_form, "Run control", self._row(self.btn_run, self.btn_stop))
        self._add_form_row(actions_form, "Output", self._row(self.btn_open))
        content_layout.addWidget(actions_box)

        content_layout.addStretch(1)

        return panel

    def _new_form_box(self, title: str) -> tuple[QtWidgets.QWidget, QtWidgets.QFormLayout]:
        section = QtWidgets.QWidget()
        section_lay = QtWidgets.QVBoxLayout(section)
        section_lay.setContentsMargins(0, 0, 0, 0)
        section_lay.setSpacing(8)

        header = QtWidgets.QWidget()
        header_lay = QtWidgets.QHBoxLayout(header)
        header_lay.setContentsMargins(4, 0, 4, 0)
        header_lay.setSpacing(0)
        title_lbl = QtWidgets.QLabel(title)
        title_font = QtGui.QFont(title_lbl.font())
        title_font.setBold(True)
        title_font.setPointSizeF(max(11.5, float(title_font.pointSizeF()) + 0.8))
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet("QLabel { padding: 0 2px 0 1px; }")
        header_lay.addWidget(title_lbl, 0)
        header_lay.addStretch(1)
        section_lay.addWidget(header)

        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        frame.setFrameShadow(QtWidgets.QFrame.Plain)
        section_lay.addWidget(frame)

        form = QtWidgets.QFormLayout(frame)
        form.setContentsMargins(12, 12, 12, 12)
        form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        form.setFormAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)
        return section, form

    def _add_form_row(self, form: QtWidgets.QFormLayout, label: str, field: QtWidgets.QWidget) -> None:
        lbl = QtWidgets.QLabel(label)
        lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        lbl.setMinimumWidth(165)
        form.addRow(lbl, field)

    def _compact_pairs_row(self, entries: list[tuple[str, QtWidgets.QWidget]]) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        for idx, (name, widget) in enumerate(entries):
            lbl = QtWidgets.QLabel(name)
            lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            lay.addWidget(lbl)
            lay.addWidget(widget)
            if idx < len(entries) - 1:
                lay.addSpacing(4)
        lay.addStretch(1)
        return row

    def _build_right_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_status = QtWidgets.QLabel("Idle")
        top.addWidget(self.progress_bar, 2)
        top.addWidget(self.lbl_status, 1)
        v.addLayout(top)

        self.tabs = QtWidgets.QTabWidget()

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        log_tab = QtWidgets.QWidget()
        log_lay = QtWidgets.QVBoxLayout(log_tab)
        log_lay.addWidget(self.txt_log)
        self.tabs.addTab(log_tab, "Logs")

        obs_tab = QtWidgets.QWidget()
        obs_lay = QtWidgets.QVBoxLayout(obs_tab)
        row_metric = QtWidgets.QHBoxLayout()
        row_metric.setContentsMargins(0, 0, 0, 0)
        row_metric.setSpacing(8)
        row_metric.addWidget(QtWidgets.QLabel("Observable"))
        self.cb_metric = QtWidgets.QComboBox()
        self.cb_metric.addItems(
            [
                "MSD (A^2)",
                "Tracer Diffusion D(t)",
                "Migration Barrier Histogram (eV)",
                "Local Order vs T (Sweep)",
                "Jump Randomness R",
                "Cluster / Order Evolution",
                "2D Vacancy Migraation",
                "3D Vacancy Migration",
            ]
        )
        self.cb_metric.currentTextChanged.connect(self._refresh_plot)
        row_metric.addWidget(self.cb_metric, 1)
        obs_lay.addLayout(row_metric)

        self.lbl_temp = QtWidgets.QLabel("Temperature")
        self.cb_temp = QtWidgets.QComboBox()
        self.cb_temp.currentTextChanged.connect(self._refresh_plot)

        self.temp_selector_widget = QtWidgets.QWidget()
        temp_sel_lay = QtWidgets.QHBoxLayout(self.temp_selector_widget)
        temp_sel_lay.setContentsMargins(0, 0, 0, 0)
        temp_sel_lay.setSpacing(6)
        temp_sel_lay.addWidget(self.lbl_temp)
        temp_sel_lay.addWidget(self.cb_temp)

        row_selectors = QtWidgets.QHBoxLayout()
        row_selectors.setContentsMargins(0, 0, 0, 0)
        row_selectors.setSpacing(10)
        row_selectors.addWidget(self.temp_selector_widget, 0)
        row_selectors.addStretch(1)
        obs_lay.addLayout(row_selectors)

        self.obs_stack = QtWidgets.QStackedWidget()

        self.canvas = ObservableCanvas()
        plot_page = QtWidgets.QWidget()
        plot_lay = QtWidgets.QVBoxLayout(plot_page)
        plot_lay.setContentsMargins(0, 0, 0, 0)
        plot_lay.setSpacing(0)
        self.plot_scroll = QtWidgets.QScrollArea()
        self.plot_scroll.setWidgetResizable(False)
        self.plot_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.plot_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.plot_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.plot_scroll.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.plot_scroll.setMinimumWidth(0)
        self.plot_scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.plot_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        self.plot_scroll.setWidget(self.canvas)
        plot_lay.addWidget(self.plot_scroll, 1)
        self.obs_stack.addWidget(plot_page)

        media_page = QtWidgets.QWidget()
        media_lay = QtWidgets.QVBoxLayout(media_page)
        media_lay.setContentsMargins(0, 0, 0, 0)
        media_lay.setSpacing(6)
        self.lbl_media_status = QtWidgets.QLabel("Select a frames observable to preview media.")
        self.lbl_media_status.setWordWrap(True)
        self.lbl_media_status.setStyleSheet("color: #444;")
        self.lbl_media = QtWidgets.QLabel()
        self.lbl_media.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_media.setMinimumHeight(380)
        self.lbl_media.setStyleSheet("background: #fafafa; border: 1px solid #ddd;")
        media_lay.addWidget(self.lbl_media_status)
        media_lay.addWidget(self.lbl_media, 1)
        self.obs_stack.addWidget(media_page)

        obs_lay.addWidget(self.obs_stack, 1)
        self.tabs.addTab(obs_tab, "Observables")

        v.addWidget(self.tabs, 1)
        return panel

    def _configure_pct_spin(self, spin: QtWidgets.QDoubleSpinBox, value: float) -> None:
        spin.setRange(0.0, 100.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        spin.setSuffix(" %")
        spin.setValue(value)

    def _row(self, *widgets) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        for idx, widget in enumerate(widgets):
            stretch = 1 if (len(widgets) == 2 and idx == 0 and isinstance(widget, QtWidgets.QLineEdit)) else 0
            lay.addWidget(widget, stretch)
        return w

    def _file_button(self, line_edit: QtWidgets.QLineEdit, title: str, filt: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton("…")

        def choose() -> None:
            selected, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                title,
                str(Path(line_edit.text()).expanduser().parent),
                filt,
            )
            if selected:
                line_edit.setText(selected)

        button.clicked.connect(choose)
        return button

    def _dir_button(self, line_edit: QtWidgets.QLineEdit, title: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton("…")

        def choose() -> None:
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                title,
                str(Path(line_edit.text()).expanduser().parent),
            )
            if selected:
                line_edit.setText(selected)

        button.clicked.connect(choose)
        return button

    def _sync_structure_mode(self):
        use_file = self.rb_from_file.isChecked()
        self.ed_data.setEnabled(use_file)
        self.btn_data.setEnabled(use_file)

        for widget in (
            self.sp_atoms,
            self.sp_fe,
            self.sp_ni,
            self.sp_cr,
            self.sp_structure_seed,
        ):
            widget.setEnabled(not use_file)

    def _sync_sweep_mode(self):
        sweep = self.chk_sweep.isChecked()
        self.sp_temperature.setEnabled(not sweep)
        self.sp_sweep_tmin.setEnabled(sweep)
        self.sp_sweep_tmax.setEnabled(sweep)
        self.sp_sweep_tstep.setEnabled(sweep)

    def _sync_frame_export(self):
        enabled = self.chk_frame_videos.isChecked()
        self.sp_video_max_frames.setEnabled(enabled)
        self.sp_video_fps.setEnabled(enabled)

    def _current_metric_uses_temperature_x_axis(self) -> bool:
        return self.cb_metric.currentText() == "Local Order vs T (Sweep)"

    def _sync_temperature_selector(self) -> None:
        show = self._last_sweep_mode and (not self._current_metric_uses_temperature_x_axis())
        self.temp_selector_widget.setVisible(show)

    def _is_image_backed_metric(self, metric: str) -> bool:
        if self._is_media_metric(metric):
            return False
        if metric == "Local Order vs T (Sweep)":
            return True
        return metric in IMAGE_BACKED_METRIC_FILES

    def _observable_image_filename(self, metric: str) -> Optional[str]:
        if metric == "Local Order vs T (Sweep)":
            return "xij_norm_vs_T.png"
        return IMAGE_BACKED_METRIC_FILES.get(metric)

    def _resolve_observable_image_path(self, metric: str) -> Optional[Path]:
        filename = self._observable_image_filename(metric)
        if filename is None:
            return None

        if metric == "Local Order vs T (Sweep)":
            root_dir = self._last_run_output_dir
            if root_dir is None:
                root_dir = Path(self.ed_out.text().strip()).expanduser().resolve()
            return root_dir / filename

        run_dir: Optional[Path]
        if self._last_sweep_mode:
            run_dir = self._selected_sweep_run_dir()
            if run_dir is None and self._last_run_output_dir is not None:
                run_dir = self._last_run_output_dir
        else:
            run_dir = self._last_run_output_dir

        if run_dir is None:
            run_dir = Path(self.ed_out.text().strip()).expanduser().resolve()
        return run_dir / filename

    def _discover_sweep_temp_dirs(self, out_root: Path) -> dict[str, Path]:
        out: dict[str, Path] = {}
        if not out_root.exists():
            return out

        rx = re.compile(r"^T_(.+)K$")
        items: list[tuple[str, Path]] = []
        for d in out_root.iterdir():
            if not d.is_dir():
                continue
            m = rx.match(d.name)
            if not m:
                continue
            label = m.group(1)
            if (d / "observables.csv").exists():
                items.append((label, d))

        def _temp_sort_key(item: tuple[str, Path]) -> tuple[int, float, str]:
            label = item[0]
            try:
                return (0, float(label), label)
            except ValueError:
                return (1, float("inf"), label)

        for label, d in sorted(items, key=_temp_sort_key):
            out[label] = d
        return out

    def _selected_sweep_run_dir(self) -> Optional[Path]:
        key = self.cb_temp.currentText().strip()
        if not key:
            return None
        return self._sweep_temp_dirs.get(key)

    def _records_for_selected_sweep_temperature(self) -> list[dict]:
        run_dir = self._selected_sweep_run_dir()
        if run_dir is None:
            return []
        path = run_dir / "observables.csv"
        if not path.exists():
            return []
        try:
            df = pd.read_csv(path)
        except Exception:
            return []
        if df.empty:
            return []
        records = df.to_dict(orient="records")
        return self._merge_barriers_into_records(records, run_dir)

    def _load_chosen_barrier_map(self, run_dir: Optional[Path]) -> dict[int, float]:
        if run_dir is None:
            return {}
        path = run_dir / "time_log.csv"
        if not path.exists():
            return {}
        try:
            df = pd.read_csv(path, usecols=["step", "chosen_barrier_eV"])
        except Exception:
            return {}
        if df.empty:
            return {}

        out: dict[int, float] = {}
        steps = df["step"].to_numpy()
        barriers = df["chosen_barrier_eV"].to_numpy()
        for step, barrier in zip(steps, barriers):
            try:
                step_i = int(step)
                barrier_f = float(barrier)
            except (TypeError, ValueError):
                continue
            if np.isfinite(barrier_f):
                out[step_i] = barrier_f
        return out

    def _merge_barriers_into_records(self, records: list[dict], run_dir: Optional[Path]) -> list[dict]:
        if not records:
            return records
        barrier_by_step = self._load_chosen_barrier_map(run_dir)
        if not barrier_by_step:
            return records

        merged: list[dict] = []
        for record in records:
            enriched = dict(record)
            try:
                step_i = int(enriched.get("step"))
            except (TypeError, ValueError):
                merged.append(enriched)
                continue
            barrier = barrier_by_step.get(step_i)
            if barrier is not None:
                enriched["chosen_barrier_eV"] = barrier
            merged.append(enriched)
        return merged

    def _append_log(self, msg: str) -> None:
        msg_norm = msg.lower()
        if any(token in msg_norm for token in SUPPRESSED_LOG_SUBSTRINGS):
            return
        self.txt_log.appendPlainText(msg)
        bar = self.txt_log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _is_media_metric(self, metric: str) -> bool:
        return metric in {"2D Vacancy Migraation", "3D Vacancy Migration"}

    def _media_basename(self, metric: str) -> Optional[str]:
        if metric == "2D Vacancy Migraation":
            return "frames_2D"
        if metric == "3D Vacancy Migration":
            return "frames_3D"
        return None

    def _stop_media_movie(self) -> None:
        if self._media_movie is not None:
            self._media_movie.stop()
            self._media_movie.deleteLater()
            self._media_movie = None
        self.lbl_media.clear()

    def _refresh_media_view(self, metric: str) -> None:
        self.obs_stack.setCurrentIndex(1)
        self._stop_media_movie()

        base = self._media_basename(metric)
        if base is None:
            self.lbl_media_status.setText("No media selected.")
            return

        out_dir = self._selected_sweep_run_dir() if self._last_sweep_mode else self._last_run_output_dir
        if out_dir is None:
            out_dir = Path(self.ed_out.text().strip()).expanduser().resolve()

        gif_path = out_dir / f"{base}.gif"
        mp4_path = out_dir / f"{base}.mp4"

        if gif_path.exists():
            movie = QtGui.QMovie(str(gif_path))
            if not movie.isValid():
                self.lbl_media_status.setText(f"Found {gif_path.name}, but Qt could not decode it.")
                return
            self._media_movie = movie
            self.lbl_media.setMovie(movie)
            self.lbl_media_status.setText(f"Showing {gif_path.name} from {out_dir}")
            movie.start()
            return

        if mp4_path.exists():
            self.lbl_media_status.setText(
                f"Found {mp4_path.name}, but in-GUI preview currently supports GIF. Open output folder to view MP4."
            )
            return

        self.lbl_media_status.setText(
            f"No {base}.gif found in {out_dir}. Run simulation with frame videos enabled."
        )

    def _collect_config(self) -> dict:
        total_pct = self.sp_fe.value() + self.sp_ni.value() + self.sp_cr.value()
        if self.rb_generated.isChecked() and abs(total_pct - 100.0) > 1e-6:
            raise ValueError(f"Generated composition must sum to 100%. Got {total_pct:.6f}%")

        if self.chk_sweep.isChecked():
            if self.sp_sweep_tstep.value() <= 0:
                raise ValueError("Sweep temperature step must be > 0")
            if self.sp_sweep_tmax.value() < self.sp_sweep_tmin.value():
                raise ValueError("Sweep Tmax must be >= Tmin")

        return {
            "use_generated": bool(self.rb_generated.isChecked()),
            "data_path": self.ed_data.text().strip(),
            "num_atoms": int(self.sp_atoms.value()),
            "fe_pct": float(self.sp_fe.value()),
            "ni_pct": float(self.sp_ni.value()),
            "cr_pct": float(self.sp_cr.value()),
            "vacancy_atom_id": None if int(self.sp_vacancy.value()) < 0 else int(self.sp_vacancy.value()),
            "structure_seed": int(self.sp_structure_seed.value()),
            "checkpoint_path": self.ed_ckpt.text().strip(),
            "metadata_path": self.ed_meta.text().strip(),
            "output_dir": self.ed_out.text().strip(),
            "cache_size": int(self.sp_cache.value()),
            "temperature_K": float(self.sp_temperature.value()),
            "attempt_frequency": float(10.0 ** float(self.sp_nu_log10.value())),
            "attempt_frequency_log10": float(self.sp_nu_log10.value()),
            "enable_temp_sweep": bool(self.chk_sweep.isChecked()),
            "sweep_t_min": float(self.sp_sweep_tmin.value()),
            "sweep_t_max": float(self.sp_sweep_tmax.value()),
            "sweep_t_step": float(self.sp_sweep_tstep.value()),
            "num_steps": int(self.sp_steps.value()),
            "steps_per_save": int(self.sp_save_every.value()),
            "kmc_seed": int(self.sp_kmc_seed.value()),
            "generate_frame_videos": bool(self.chk_frame_videos.isChecked()),
            "video_max_frames": int(self.sp_video_max_frames.value()),
            "video_fps": int(self.sp_video_fps.value()),
            "enable_detailed_balance": bool(self.chk_db.isChecked()),
            "interaction_matrix_text": self.ed_interactions.text().strip(),
        }

    def _prepare_output_directory(self, out_dir: Path) -> Optional[Path]:
        try:
            resolved = out_dir.expanduser().resolve()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid output directory", str(exc))
            return None

        if resolved.exists() and not resolved.is_dir():
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid output directory",
                f"Selected output path exists but is not a directory:\n{resolved}",
            )
            return None

        try:
            resolved.mkdir(parents=True, exist_ok=True)
            has_contents = any(resolved.iterdir())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Output directory error",
                f"Cannot access output directory:\n{resolved}\n\n{exc}",
            )
            return None

        if has_contents:
            prompt = QtWidgets.QMessageBox(self)
            prompt.setWindowTitle("Clear Output Directory?")
            prompt.setIcon(QtWidgets.QMessageBox.NoIcon)
            prompt.setText(
                "The selected output directory is not empty.\n\n"
                f"{resolved}\n\n"
                "Do you want to clear all current contents before starting this run?"
            )
            prompt.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            prompt.setDefaultButton(QtWidgets.QMessageBox.No)
            answer = prompt.exec_()
            if answer != QtWidgets.QMessageBox.Yes:
                return None

        try:
            for child in resolved.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Failed to clear output directory",
                f"Could not clear directory:\n{resolved}\n\n{exc}",
            )
            return None

        return resolved

    def _on_run(self):
        if self._thread is not None and self._thread.isRunning():
            QtWidgets.QMessageBox.information(self, "Simulation running", "A simulation is already running.")
            return

        try:
            _preload_torch_main_thread()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "PyTorch import error", str(exc))
            return

        try:
            cfg = self._collect_config()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid input", str(exc))
            return

        prepared_out_dir = self._prepare_output_directory(Path(cfg["output_dir"]))
        if prepared_out_dir is None:
            self._append_log("Run cancelled: output directory was not cleared.")
            return
        cfg["output_dir"] = str(prepared_out_dir)

        self._observable_records = []
        self._last_run_output_dir = prepared_out_dir
        self._last_xij_csv_path = None
        self._last_sweep_mode = False
        self._sweep_temp_dirs = {}
        self.cb_temp.clear()
        self._sync_temperature_selector()
        self._refresh_plot()
        self.txt_log.clear()
        self.lbl_status.setText("Running")
        self.progress_bar.setValue(0)

        self._thread = QtCore.QThread(self)
        self._worker = SimulationWorker(cfg)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.observable.connect(self._on_observable)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker_thread)

        self._thread.start()

    def _on_stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._append_log("Stop requested.")

    def _on_open_output(self):
        out = Path(self.ed_out.text().strip()).expanduser().resolve()
        if not out.exists():
            QtWidgets.QMessageBox.information(self, "Output directory", f"Directory does not exist: {out}")
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(out)))

    @QtCore.pyqtSlot(int, int)
    def _on_progress(self, done: int, total: int):
        pct = 0 if total <= 0 else int(round(100.0 * done / total))
        self.progress_bar.setValue(max(0, min(100, pct)))

    @QtCore.pyqtSlot(dict)
    def _on_observable(self, record: dict):
        self._observable_records.append(record)
        if len(self._observable_records) % 5 == 0:
            self._refresh_plot()

    @QtCore.pyqtSlot(dict)
    def _on_finished(self, payload: dict):
        self.lbl_status.setText("Done")
        self.progress_bar.setValue(100)
        self._last_run_output_dir = Path(payload.get("output_dir"))
        xij_csv = payload.get("xij_csv_path")
        self._last_xij_csv_path = Path(xij_csv) if xij_csv else None
        self._last_sweep_mode = bool(payload.get("sweep_mode", False))
        if not self._last_sweep_mode:
            self._observable_records = self._merge_barriers_into_records(
                self._observable_records,
                self._last_run_output_dir,
            )

        self._sweep_temp_dirs = {}
        self.cb_temp.blockSignals(True)
        self.cb_temp.clear()
        if self._last_sweep_mode and self._last_run_output_dir is not None:
            self._sweep_temp_dirs = self._discover_sweep_temp_dirs(self._last_run_output_dir)
            for key in self._sweep_temp_dirs.keys():
                self.cb_temp.addItem(key)
        self.cb_temp.blockSignals(False)
        self._sync_temperature_selector()
        self._refresh_plot()

        diffusion = payload.get("diffusion_estimate")
        if diffusion is not None:
            self._append_log(f"Estimated diffusion coefficient: {diffusion:.3e} A^2/s")
        frames_2d = payload.get("frames_2d_path")
        frames_3d = payload.get("frames_3d_path")
        if frames_2d:
            self._append_log(f"Saved observable: {frames_2d}")
        if frames_3d:
            self._append_log(f"Saved observable: {frames_3d}")
        if payload.get("sweep_summary_path"):
            self._append_log(f"Sweep summary: {payload.get('sweep_summary_path')}")
        self._append_log(f"Output directory: {payload.get('output_dir')}")

    @QtCore.pyqtSlot(str)
    def _on_error(self, msg: str):
        self.lbl_status.setText("Error")
        self._append_log("[ERROR] " + msg)
        QtWidgets.QMessageBox.critical(self, "Simulation error", msg)

    def _cleanup_worker_thread(self):
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def _refresh_plot(self):
        self._sync_temperature_selector()
        metric = self.cb_metric.currentText()

        if self._is_media_metric(metric):
            self._refresh_media_view(metric)
            return

        self.obs_stack.setCurrentIndex(0)
        self._stop_media_movie()

        if not self._is_image_backed_metric(metric):
            self.canvas.plot_message("Unsupported observable selection.", title=metric)
            return

        image_path = self._resolve_observable_image_path(metric)
        if image_path is None:
            self.canvas.plot_message("Could not resolve image path.", title=metric)
            return

        if image_path.exists():
            viewport_size = self.plot_scroll.viewport().size()
            fit_within_px = (
                max(1, int(viewport_size.width()) - 8),
                max(1, int(viewport_size.height()) - 8),
            )
            self.canvas.plot_image(image_path, title=None, fit_within_px=fit_within_px)
            return

        if self._thread is not None and self._thread.isRunning():
            self.canvas.plot_message(
                "Image-backed observables are generated after run completion.",
                title=metric,
            )
            return

        self.canvas.plot_message(
            f"Observable image not found:\n{image_path}\n\nRun simulation to generate this output.",
            title=metric,
        )


def launch_gui() -> int:
    # Important: load torch on main thread before any worker thread starts.
    _preload_torch_main_thread()

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication([])

    app.setApplicationName("Production KMC-ANN")
    win = MainWindow()
    win.show()

    if owns_app:
        return app.exec_()
    return 0
