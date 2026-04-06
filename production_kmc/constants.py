from __future__ import annotations

from typing import Dict

K_B_EV_PER_K = 8.617333262145e-5

SPECIES_NAME_TO_TYPE: Dict[str, int] = {
    "Fe": 1,
    "Ni": 2,
    "Cr": 3,
}
TYPE_TO_SPECIES_NAME: Dict[int, str] = {v: k for k, v in SPECIES_NAME_TO_TYPE.items()}

CANONICAL_SHELL_COUNTS: Dict[int, int] = {
    1: 12,
    2: 6,
    3: 24,
    4: 12,
    5: 24,
    6: 8,
    7: 48,
    8: 6,
    9: 36,
    10: 24,
}

DEFAULT_BARRIER_COLUMNS = [f"1NN_{i}" for i in range(1, 13)]
