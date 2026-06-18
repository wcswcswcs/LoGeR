from __future__ import annotations

from typing import Any


def geometry_stage_decision(stage1_summary: dict[str, Any]) -> dict[str, Any]:
    decision = str(stage1_summary.get("final_label") or stage1_summary.get("status") or "")
    allowed = decision in {"GO_STAGE1_SIGNIFICANT_MATCHING_BREAKTHROUGH", "GO_STAGE1_STRONG_CONFERENCE_GATE"}
    return {
        "phase": "v43_2_geometry_optimization_diagnostic",
        "stage2_allowed": bool(allowed),
        "status": "STAGE2_ALLOWED" if allowed else "STAGE2_BLOCKED_MATCHING_NOT_SIGNIFICANT",
        "reason": "Phase F passed Minimum Significant Gate" if allowed else "Phase F did not pass Minimum Significant Gate",
    }
