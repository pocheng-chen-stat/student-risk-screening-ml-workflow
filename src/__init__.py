"""Student risk screening workflow package.

Set joblib/loky CPU count early to avoid Windows/Anaconda warnings about
missing `wmic` when estimating the number of physical CPU cores.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
