"""Optional Home-page summary for the combined CSc 8830 course dashboard.

The combined dashboard may call ``get_module_summary()`` to show one status chip on this
module's Home card. The standalone app never uses it.

Evidence contract: the summary reports "Results available" only when every committed result
file below exists and both tracking-validation records state ``"status": "complete"``.
Otherwise it returns ``None`` and no chip is shown. It only reads small committed files: it
never runs OpenCV, recomputes metrics, judges result quality, writes files, or touches the
network.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

TRACKING_RECORDS = (
    "results/tracking/video_1/P1_record.json",
    "results/tracking/video_2/P1_record.json",
)
SUPPORTING_RESULTS = (
    "results/optical_flow/video_1/video_1_optical_flow_summary.json",
    "results/optical_flow/video_2/video_2_optical_flow_summary.json",
    "results/sfm/sfm_summary.json",
)


@dataclass(frozen=True)
class ModuleSummary:
    """One status chip: ``kind`` is a shared chip kind (neutral, info, success, warning, error)."""

    label: str
    kind: str


RESULTS_AVAILABLE = ModuleSummary("Results available", "success")


def _record_status(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data.get("status") if isinstance(data, dict) else None


def get_module_summary(repo_root: Path | None = None) -> ModuleSummary | None:
    """Return the Home status for this module, or ``None`` when the evidence is incomplete."""
    root = repo_root or _REPO_ROOT
    if not all((root / rel).is_file() for rel in SUPPORTING_RESULTS):
        return None
    if any(_record_status(root / rel) != "complete" for rel in TRACKING_RECORDS):
        return None
    return RESULTS_AVAILABLE
