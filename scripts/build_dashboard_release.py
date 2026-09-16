"""Build and validate Module 9A's read-only dashboard evidence release."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Dashboard.data import build_dashboard_release  # noqa: E402


if __name__ == "__main__":
    _, _, metadata = build_dashboard_release()
    print(json.dumps(metadata, indent=2))
    if metadata["status"] != "READY":
        raise SystemExit("Dashboard release is NOT READY")
