#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PRODUCTION_ROOT = THIS_FILE.parents[1]
if str(PRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_ROOT))

# Keep matplotlib/font caches in a writable local directory. This avoids slow
# startup cache rebuilds and permission-related backend instability.
LOCAL_CACHE_ROOT = PRODUCTION_ROOT / ".cache"
LOCAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(LOCAL_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_ROOT / "matplotlib"))

from production_kmc.gui import launch_gui


if __name__ == "__main__":
    raise SystemExit(launch_gui())
