from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .constants import SPECIES_NAME_TO_TYPE


@dataclass
class ModelSettings:
    checkpoint_path: Path
    metadata_path: Path
    device: str = "cpu"
    cache_size: int = 200_000

    def validate(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        if self.cache_size < 0:
            raise ValueError("cache_size must be >= 0")


@dataclass
class StructureSettings:
    mode: str = "file"  # file | generated
    data_path: Optional[Path] = None
    num_atoms: int = 256
    composition_percent: Dict[str, float] = field(
        default_factory=lambda: {"Fe": 33.3333, "Ni": 33.3333, "Cr": 33.3334}
    )
    vacancy_atom_id: Optional[int] = None
    random_seed: int = 123

    def validate(self) -> None:
        if self.mode not in {"file", "generated"}:
            raise ValueError("Structure mode must be 'file' or 'generated'.")

        if self.mode == "file":
            if self.data_path is None:
                raise ValueError("data_path is required when structure mode is 'file'.")
            if not self.data_path.exists():
                raise FileNotFoundError(f"LAMMPS data file not found: {self.data_path}")

        if self.mode == "generated":
            if self.num_atoms <= 0:
                raise ValueError("num_atoms must be > 0")
            if self.num_atoms % 4 != 0:
                raise ValueError("For FCC generation, num_atoms must be a multiple of 4.")

            keys = set(self.composition_percent.keys())
            required = set(SPECIES_NAME_TO_TYPE.keys())
            if keys != required:
                raise ValueError(f"composition_percent must have exactly keys {sorted(required)}")

            total = sum(float(v) for v in self.composition_percent.values())
            if abs(total - 100.0) > 1e-6:
                raise ValueError(f"Composition must sum to 100%. Got {total:.6f}%")


@dataclass
class SimulationSettings:
    temperature_K: float = 1000.0
    attempt_frequency_s_inv: float = 1.0e13
    num_steps: int = 2000
    steps_per_save: int = 25
    generate_frame_videos: bool = True
    video_max_frames: int = 300
    video_fps: int = 12
    output_dir: Path = Path("runs/production_run")
    random_seed: int = 123
    enable_detailed_balance: bool = False
    interaction_matrix_eV: Optional[List[List[float]]] = None

    def validate(self) -> None:
        if self.temperature_K <= 0:
            raise ValueError("temperature_K must be > 0")
        if self.attempt_frequency_s_inv <= 0:
            raise ValueError("attempt_frequency_s_inv must be > 0")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be > 0")
        if self.steps_per_save <= 0:
            raise ValueError("steps_per_save must be > 0")
        if self.video_max_frames <= 0:
            raise ValueError("video_max_frames must be > 0")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be > 0")
        if self.enable_detailed_balance and self.interaction_matrix_eV is None:
            raise ValueError(
                "enable_detailed_balance=True requires interaction_matrix_eV (3x3 matrix)."
            )
        if self.interaction_matrix_eV is not None:
            if len(self.interaction_matrix_eV) != 3 or any(len(row) != 3 for row in self.interaction_matrix_eV):
                raise ValueError("interaction_matrix_eV must be 3x3.")


@dataclass
class RunSettings:
    model: ModelSettings
    structure: StructureSettings
    simulation: SimulationSettings

    def validate(self) -> None:
        self.model.validate()
        self.structure.validate()
        self.simulation.validate()
