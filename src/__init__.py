"""Defense GMM-Maha experiment package."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = PROJECT_ROOT / "build" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
