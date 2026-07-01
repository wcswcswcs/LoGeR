#!/usr/bin/env python3
"""Summarize v96 Phase7 control attribution from completed Phase6 runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


PHASE_ID = "v96_phase7_control_summary"
RUN_ID = "v96_phase7_control_summary"
DEFAULT_REAL_C = ROOT / "outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_sceneoffset_ranked_suppressed_more"
DEFAULT_CONTROL_A = ROOT / "outputs/audit/v96_phase6_render_snap_w0020_segmented_r4_D3_sceneoffset_A_ranked_suppressed_more"
DEFAULT_OUT = ROOT / "outputs/audit/v96_phase7_control_summary_w0020_segmented_r4_D3_sceneoffset"


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _best_metric(root: Path) -> dict[str, Any]:
    summary = _load_json(root / "summary.json")
    best_variant = str((summary.get("best_variant") or {}).get("readout_variant") or "")
    rows = _read_csv(root / "render_variant_metric_rows.csv")
    for row in rows:
        if row.get("readout_variant") == best_variant:
            return {**row, "phase6_root": _rel(root), "family": summary.get("family", ""), "decision": summary.get("decision", "")}
    if rows:
        best = max(rows, key=lambda row: (_num(row.get("MV_AP50_window")), _num(row.get("MV_AP_window"))))
        return {**best, "phase6_root": _rel(root), "family": summary.get("family", ""), "decision": summary.get("decision", "")}
    return {}


def _parse_control_root(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"control root must be NAME=PATH, got {raw!r}")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"control root has empty NAME: {raw!r}")
    return name, _project(path.strip())


def run(args: argparse.Namespace) -> dict[str, Any]:
    real_root = _project(args.real_c_phase6_root)
    control_root = _project(args.control_a_phase6_root)
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    real = _best_metric(real_root)
    control_specs: list[tuple[str, Path, str]] = [
        ("C5_mask_only_setcover_proxy_A_family", control_root, "control_mask_only_setcover_proxy")
    ]
    for raw in args.control_phase6_root or []:
        control_name, root = _parse_control_root(raw)
        control_specs.append((control_name, root, "required_control"))
    controls: list[dict[str, Any]] = []
    for control_name, root, role in control_specs:
        metric = _best_metric(root)
        metric["control_name"] = control_name
        metric["variant_role"] = role
        controls.append(metric)
    variant_rows = [
        {
            "schema_version": "stream4d_v96_control_variant_metric_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_role": "real_full_signed_affinity_proxy",
            "control_name": "real_C_hybrid_cover_cluster_F5",
            "family": real.get("family", ""),
            "readout_variant": real.get("readout_variant", ""),
            "MV_AP_window": _num(real.get("MV_AP_window")),
            "MV_AP50_window": _num(real.get("MV_AP50_window")),
            "ScoreFreeMatch50_window": _num(real.get("ScoreFreeMatch50_window")),
            "phase6_gate_pass": real.get("phase6_gate_pass", ""),
            "phase6_root": real.get("phase6_root", ""),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]
    for control in controls:
        variant_rows.append(
            {
                "schema_version": "stream4d_v96_control_variant_metric_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_role": control.get("variant_role", "required_control"),
                "control_name": control.get("control_name", ""),
                "family": control.get("family", ""),
                "readout_variant": control.get("readout_variant", ""),
                "MV_AP_window": _num(control.get("MV_AP_window")),
                "MV_AP50_window": _num(control.get("MV_AP50_window")),
                "ScoreFreeMatch50_window": _num(control.get("ScoreFreeMatch50_window")),
                "phase6_gate_pass": control.get("phase6_gate_pass", ""),
                "phase6_root": control.get("phase6_root", ""),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    best_control_mv_ap = max((_num(row.get("MV_AP_window")) for row in controls), default=0.0)
    best_control_mv_ap50 = max((_num(row.get("MV_AP50_window")) for row in controls), default=0.0)
    best_control_for_mv_ap = max(controls, key=lambda row: _num(row.get("MV_AP_window")), default={})
    best_control_for_mv_ap50 = max(controls, key=lambda row: _num(row.get("MV_AP50_window")), default={})
    gap_rows = [
        {
            "schema_version": "stream4d_v96_control_gap_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gap_name": "real_C_minus_best_required_control",
            "real_MV_AP_window": _num(real.get("MV_AP_window")),
            "control_MV_AP_window": best_control_mv_ap,
            "control_MV_AP_control_name": best_control_for_mv_ap.get("control_name", ""),
            "gap_MV_AP_window": _num(real.get("MV_AP_window")) - best_control_mv_ap,
            "required_gap_MV_AP_window": 0.005,
            "real_MV_AP50_window": _num(real.get("MV_AP50_window")),
            "control_MV_AP50_window": best_control_mv_ap50,
            "control_MV_AP50_control_name": best_control_for_mv_ap50.get("control_name", ""),
            "gap_MV_AP50_window": _num(real.get("MV_AP50_window")) - best_control_mv_ap50,
            "required_gap_MV_AP50_window": 0.010,
            "pass": bool(
                _num(real.get("MV_AP_window")) >= best_control_mv_ap + 0.005
                and _num(real.get("MV_AP50_window")) >= best_control_mv_ap50 + 0.010
            ),
        }
    ]
    gap = gap_rows[0]
    by_name = {str(row.get("control_name", "")): row for row in controls}
    required_control_names = {
        "C0_semantic_only",
        "C1_mask_area_risk",
        "C2_shuffled_D4RT",
        "C3_no_temporal",
        "C4_random_micro_primitives",
        "C5_mask_only_setcover_proxy_A_family",
    }
    missing_controls = sorted(required_control_names - set(by_name))
    c2_gap_ap50 = _num(real.get("MV_AP50_window")) - _num(by_name.get("C2_shuffled_D4RT", {}).get("MV_AP50_window"))
    c3_gap_ap50 = _num(real.get("MV_AP50_window")) - _num(by_name.get("C3_no_temporal", {}).get("MV_AP50_window"))
    c2_specific_pass = "C2_shuffled_D4RT" in by_name and c2_gap_ap50 >= 0.010
    c3_specific_pass = "C3_no_temporal" in by_name and c3_gap_ap50 >= 0.005
    failure_rows: list[dict[str, Any]] = []
    if _num(real.get("MV_AP_window")) < best_control_mv_ap or _num(real.get("MV_AP50_window")) < best_control_mv_ap50:
        failure_rows.append(
            {
                "schema_version": "stream4d_v96_control_failure_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "failure": "best_required_control_beats_full_signed_affinity",
                "evidence": (
                    f"C MV_AP={_num(real.get('MV_AP_window'))}, MV_AP50={_num(real.get('MV_AP50_window'))}; "
                    f"best_control_MV_AP={best_control_mv_ap} ({best_control_for_mv_ap.get('control_name','')}); "
                    f"best_control_MV_AP50={best_control_mv_ap50} ({best_control_for_mv_ap50.get('control_name','')})"
                ),
                "blocker_label": "CONTROL_BIAS_REJECTED",
            }
        )
    elif not bool(gap["pass"]):
        failure_rows.append(
            {
                "schema_version": "stream4d_v96_control_failure_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "failure": "full_signed_affinity_control_margin_insufficient",
                "evidence": (
                    f"gap_MV_AP={gap['gap_MV_AP_window']} vs required {gap['required_gap_MV_AP_window']}; "
                    f"gap_MV_AP50={gap['gap_MV_AP50_window']} vs required {gap['required_gap_MV_AP50_window']}"
                ),
                "blocker_label": "CONTROL_MARGIN_INSUFFICIENT",
            }
        )
    if missing_controls:
        failure_rows.append(
            {
                "schema_version": "stream4d_v96_control_failure_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "failure": "required_controls_not_all_materialized",
                "evidence": "missing_controls=" + ",".join(missing_controls),
                "blocker_label": "CONTROL_COVERAGE_INCOMPLETE",
            }
        )
    if not c2_specific_pass:
        failure_rows.append(
            {
                "schema_version": "stream4d_v96_control_failure_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "failure": "shuffled_d4rt_gap_insufficient_or_missing",
                "evidence": f"real_minus_C2_MV_AP50={c2_gap_ap50} vs required 0.010",
                "blocker_label": "CONTROL_C2_GAP_INSUFFICIENT",
            }
        )
    if not c3_specific_pass:
        failure_rows.append(
            {
                "schema_version": "stream4d_v96_control_failure_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "failure": "no_temporal_gap_insufficient_or_missing",
                "evidence": f"real_minus_C3_MV_AP50={c3_gap_ap50} vs required 0.005",
                "blocker_label": "CONTROL_C3_GAP_INSUFFICIENT",
            }
        )
    controls_complete = not missing_controls
    phase7_pass = bool(gap["pass"]) and controls_complete and c2_specific_pass and c3_specific_pass
    if phase7_pass:
        conclusion = "The full signed C-family clears the required C0-C5 control margins under this evaluator, but Phase6 absolute dev gates still have to be checked separately."
    else:
        conclusion = "Phase7 remains No-Go because at least one required control coverage, margin, or C2/C3 specific gap gate failed."
    summary = {
        "schema": "stream4d_v96_phase7_control_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "PASS_V96_PHASE7_CONTROL_ATTRIBUTION" if phase7_pass else "NO_GO_V96_PHASE7_CONTROL_ATTRIBUTION",
        "output_root": _rel(output_root),
        "real_c_phase6_root": _rel(real_root),
        "control_a_phase6_root": _rel(control_root),
        "required_control_phase6_roots": {name: _rel(root) for name, root, _role in control_specs},
        "required_controls_complete": controls_complete,
        "missing_required_controls": missing_controls,
        "best_control_MV_AP_window": best_control_mv_ap,
        "best_control_MV_AP_control_name": best_control_for_mv_ap.get("control_name", ""),
        "best_control_MV_AP50_window": best_control_mv_ap50,
        "best_control_MV_AP50_control_name": best_control_for_mv_ap50.get("control_name", ""),
        "C2_real_minus_control_MV_AP50_window": c2_gap_ap50,
        "C2_specific_gap_pass": c2_specific_pass,
        "C3_real_minus_control_MV_AP50_window": c3_gap_ap50,
        "C3_specific_gap_pass": c3_specific_pass,
        "variant_metric_rows": variant_rows,
        "control_gap_rows": gap_rows,
        "control_failure_rows": failure_rows,
        "conclusion": conclusion,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_csv(output_root / "control_variant_metric_rows.csv", variant_rows)
    _write_csv(output_root / "control_gap_rows.csv", gap_rows)
    _write_csv(output_root / "control_failure_rows.csv", failure_rows)
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"decision": summary["decision"], "output_root": _rel(output_root), "gap_MV_AP50_window": gap_rows[0]["gap_MV_AP50_window"]}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v96 Phase7 control attribution summary.")
    parser.add_argument("--real-c-phase6-root", default=str(DEFAULT_REAL_C))
    parser.add_argument("--control-a-phase6-root", default=str(DEFAULT_CONTROL_A))
    parser.add_argument(
        "--control-phase6-root",
        action="append",
        default=[],
        help="Additional required control Phase6 root in NAME=PATH form, e.g. C2_shuffled_D4RT=outputs/audit/...",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
