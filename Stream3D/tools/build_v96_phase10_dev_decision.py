#!/usr/bin/env python3
"""Build final v96 dev decision from Phase6/7/9 evidence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_PHASE7 = ROOT / "outputs/audit/v96_phase7_control_summary_w0020_segmented_r4_D3_sceneoffset"
DEFAULT_PHASE6_A = ROOT / "outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_sceneoffset_A_ranked_suppressed_more"
DEFAULT_PHASE6_C = ROOT / "outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_sceneoffset_ranked_suppressed_more"
DEFAULT_PHASE9 = ROOT / "outputs/audit/v96_phase9_error_decomposition_w0020_segmented_r4_D3_sceneoffset"
DEFAULT_OUT = ROOT / "outputs/audit/v96_phase10_dev_decision"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _best_pass(summary: dict[str, Any]) -> bool:
    best = summary.get("best_variant") or {}
    return bool(best.get("phase6_gate_pass"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _build_reason(phase6_a: dict[str, Any], phase6_c: dict[str, Any], phase7: dict[str, Any]) -> str:
    a_pass = _best_pass(phase6_a)
    c_pass = _best_pass(phase6_c)
    gap_rows = phase7.get("control_gap_rows") or []
    gap = gap_rows[0] if gap_rows else {}
    failure_labels = [
        str(row.get("blocker_label", ""))
        for row in (phase7.get("control_failure_rows") or [])
        if row.get("blocker_label")
    ]
    if phase7.get("decision") != "PASS_V96_PHASE7_CONTROL_ATTRIBUTION":
        if a_pass and not c_pass:
            return "Phase7 control attribution failed: A-family mask/set-cover proxy passes Phase6 while the full signed C-family remains Phase6 No-Go."
        if c_pass and not a_pass and "CONTROL_COVERAGE_INCOMPLETE" in failure_labels:
            return "Phase7 control attribution failed: full signed C-family passes the A-family proxy, but required controls are incomplete."
        if gap:
            return (
                "Phase7 control attribution failed under fullscope evaluation: "
                f"C Phase6 pass={c_pass}, A Phase6 pass={a_pass}; "
                f"gap_MV_AP={_num(gap.get('gap_MV_AP_window'))} vs required {_num(gap.get('required_gap_MV_AP_window'))}; "
                f"gap_MV_AP50={_num(gap.get('gap_MV_AP50_window'))} vs required {_num(gap.get('required_gap_MV_AP50_window'))}."
            )
        return "Phase7 control attribution failed; holdout/local2history promotion is not allowed."
    if phase6_c.get("decision") != "PASS_V96_PHASE6_RENDER_SNAP":
        return "Phase6 full signed C-family did not pass the render-snap gate; promotion is not allowed."
    return "Development gate remains conservative: no holdout/local2history promotion without complete downstream approval."


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    phase7_root = _project(args.phase7_root)
    phase6_a_root = _project(args.phase6_a_root)
    phase6_c_root = _project(args.phase6_c_root)
    phase9_root = _project(args.phase9_root)
    phase7 = _read_json(phase7_root / "summary.json")
    phase6_a = _read_json(phase6_a_root / "summary.json")
    phase6_c = _read_json(phase6_c_root / "summary.json")
    phase9 = _read_json(phase9_root / "summary.json")
    decision = {
        "schema": "stream4d_v96_phase10_dev_decision_v1",
        "phase_id": "v96_phase10_dev_decision",
        "created_at": _created_at(),
        "decision": "NO_GO_V96_DEV",
        "holdout_allowed": False,
        "local2history_allowed": False,
        "reason": _build_reason(phase6_a, phase6_c, phase7),
        "phase6_a_decision": phase6_a.get("decision"),
        "phase6_a_best_variant": phase6_a.get("best_variant"),
        "phase6_a_eval_frame_count": phase6_a.get("eval_frame_count"),
        "phase6_a_eval_frame_scope": phase6_a.get("eval_frame_scope"),
        "phase6_c_decision": phase6_c.get("decision"),
        "phase6_c_best_variant": phase6_c.get("best_variant"),
        "phase6_c_eval_frame_count": phase6_c.get("eval_frame_count"),
        "phase6_c_eval_frame_scope": phase6_c.get("eval_frame_scope"),
        "phase7_decision": phase7.get("decision"),
        "phase7_control_gap_rows": phase7.get("control_gap_rows"),
        "phase7_control_failure_rows": phase7.get("control_failure_rows"),
        "phase9_blocker_labels": phase9.get("blocker_labels"),
        "evidence_roots": {
            "phase6_a_root": _rel(phase6_a_root),
            "phase6_c_root": _rel(phase6_c_root),
            "phase7_root": _rel(phase7_root),
            "phase9_root": _rel(phase9_root),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(output_root / "final_dev_decision.json", decision)
    print(json.dumps({"decision": decision["decision"], "holdout_allowed": decision["holdout_allowed"], "output_root": _rel(output_root)}, sort_keys=True))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v96 final dev decision.")
    parser.add_argument("--phase7-root", default=str(DEFAULT_PHASE7))
    parser.add_argument("--phase6-a-root", default=str(DEFAULT_PHASE6_A))
    parser.add_argument("--phase6-c-root", default=str(DEFAULT_PHASE6_C))
    parser.add_argument("--phase9-root", default=str(DEFAULT_PHASE9))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
