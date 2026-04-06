from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np

from .structure import AtomicState

_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass
class LammpsData:
    state: AtomicState
    atom_style: str
    masses: Dict[int, float]


def _split_no_comment(line: str) -> List[str]:
    if "#" in line:
        line = line.split("#", 1)[0]
    return [tok for tok in line.strip().split() if tok]


def _try_float(tok: str) -> Optional[float]:
    tok = tok.strip()
    if not _NUM_RE.match(tok):
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def read_lammps_data(path: str | Path) -> LammpsData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LAMMPS data file not found: {path}")

    xlo = xhi = ylo = yhi = zlo = zhi = None
    masses: Dict[int, float] = {}
    atoms: List[Tuple[int, int, float, float, float]] = []
    atom_style = "atomic"

    mode: Optional[str] = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            s = line.strip()
            if not s:
                continue

            low = s.lower()
            if low.endswith("atoms # atomic"):
                mode = "atoms"
                atom_style = "atomic"
                continue
            if low.endswith("atoms # charge"):
                mode = "atoms"
                atom_style = "charge"
                continue
            if low == "masses":
                mode = "masses"
                continue
            if low in {"atoms", "bonds", "angles", "dihedrals", "impropers"}:
                mode = "atoms" if low == "atoms" else None
                if low == "atoms":
                    atom_style = "atomic"
                continue

            if " xlo xhi" in s:
                toks = _split_no_comment(s)
                xlo, xhi = map(float, toks[0:2])
                continue
            if " ylo yhi" in s:
                toks = _split_no_comment(s)
                ylo, yhi = map(float, toks[0:2])
                continue
            if " zlo zhi" in s:
                toks = _split_no_comment(s)
                zlo, zhi = map(float, toks[0:2])
                continue

            if mode == "masses":
                toks = _split_no_comment(s)
                if len(toks) >= 2 and toks[0].isdigit():
                    t = int(toks[0])
                    m = _try_float(toks[1])
                    if m is not None:
                        masses[t] = m
                continue

            if mode == "atoms":
                toks = _split_no_comment(s)
                if len(toks) < 5:
                    continue

                if atom_style == "atomic":
                    if _NUM_RE.match(toks[0]) and _NUM_RE.match(toks[1]) and all(
                        _NUM_RE.match(tok) for tok in toks[-3:]
                    ):
                        atom_id = int(float(toks[0]))
                        atom_type = int(float(toks[1]))
                        x, y, z = map(float, toks[-3:])
                        atoms.append((atom_id, atom_type, x, y, z))
                else:
                    # charge style common line: id mol type q x y z
                    if len(toks) >= 7 and all(
                        _NUM_RE.match(tok)
                        for tok in (toks[0], toks[2], toks[-3], toks[-2], toks[-1])
                    ):
                        atom_id = int(float(toks[0]))
                        atom_type = int(float(toks[2]))
                        x, y, z = map(float, toks[-3:])
                        atoms.append((atom_id, atom_type, x, y, z))
                    elif _NUM_RE.match(toks[0]) and _NUM_RE.match(toks[1]) and all(
                        _NUM_RE.match(tok) for tok in toks[-3:]
                    ):
                        atom_id = int(float(toks[0]))
                        atom_type = int(float(toks[1]))
                        x, y, z = map(float, toks[-3:])
                        atoms.append((atom_id, atom_type, x, y, z))
                continue

    if any(v is None for v in (xlo, xhi, ylo, yhi, zlo, zhi)):
        raise ValueError("Could not parse x/y/z bounds from LAMMPS data file")
    if not atoms:
        raise ValueError("No atoms parsed from LAMMPS data file")

    atoms.sort(key=lambda row: row[0])
    atom_ids = np.array([row[0] for row in atoms], dtype=np.int64)
    types = np.array([row[1] for row in atoms], dtype=np.int64)
    positions = np.array([[row[2], row[3], row[4]] for row in atoms], dtype=np.float64)
    bounds = np.array(
        [[float(xlo), float(xhi)], [float(ylo), float(yhi)], [float(zlo), float(zhi)]],
        dtype=np.float64,
    )

    state = AtomicState(atom_ids=atom_ids, positions=positions, types=types, bounds=bounds)
    return LammpsData(state=state, atom_style=atom_style, masses=masses)


def write_lammps_atomic(path: str | Path, state: AtomicState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_atoms = state.n_sites
    non_vac_types = state.types[state.types > 0]
    n_types = int(np.max(non_vac_types)) if non_vac_types.size else 1

    xlo, xhi = state.bounds[0]
    ylo, yhi = state.bounds[1]
    zlo, zhi = state.bounds[2]

    lines: List[str] = []
    lines.append("LAMMPS data file (written by Production KMC-ANN)\n")
    lines.append(f"{n_atoms} atoms\n")
    lines.append(f"{n_types} atom types\n\n")
    lines.append(f"{xlo:.8f} {xhi:.8f} xlo xhi\n")
    lines.append(f"{ylo:.8f} {yhi:.8f} ylo yhi\n")
    lines.append(f"{zlo:.8f} {zhi:.8f} zlo zhi\n\n")
    lines.append("Atoms # atomic\n")

    for atom_id, atom_type, xyz in zip(state.atom_ids, state.types, state.positions):
        lines.append(
            f"{int(atom_id)} {int(atom_type)} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}\n"
        )

    path.write_text("".join(lines), encoding="utf-8")
