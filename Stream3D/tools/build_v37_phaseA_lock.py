from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.run_v26_object_quality_diagnostics import _json_safe


PLAN_PATH = "docs/stream4d_v37_temporal_curriculum_masklet_plan.md"
V36_DECISION = "outputs/audit/v36_final_decision/decision_summary.json"
V36_SUMMARY_CSV = "outputs/audit/v36_external_downstream_assignment_watershed_all_masks_chain2/external_downstream_summary.csv"


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _int_from_file(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        return int(text.splitlines()[-1])
    except ValueError:
        return None


def _parse_unittest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    ran = re.search(r"Ran\s+(\d+)\s+tests?", text)
    ok = re.search(r"\nOK(?:\s+\([^)]*\))?\s*$", text) is not None
    return {
        "log_path": str(path),
        "log_exists": path.exists(),
        "test_count": int(ran.group(1)) if ran else None,
        "ok_marker": bool(ok),
    }


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    repo = root.parent
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    v36 = _read_json(root / V36_DECISION) or {}
    best = dict(v36.get("phaseF_external_downstream_assignment") or {})
    controls = dict(best.get("controls") or {})
    phase_validation = {
        "py_compile_first_exit_code": _int_from_file(out_dir / "py_compile.exit_code"),
        "unittest_first_exit_code": _int_from_file(out_dir / "unittest.exit_code"),
        "py_compile_exit_code": _int_from_file(out_dir / "py_compile_fullpath.exit_code"),
        "unittest_exit_code": _int_from_file(out_dir / "unittest_fullpath.exit_code"),
        "unittest": _parse_unittest(out_dir / "unittest_fullpath.log"),
    }
    py_pass = phase_validation["py_compile_exit_code"] == 0
    unit_pass = phase_validation["unittest_exit_code"] == 0 and bool(phase_validation["unittest"]["ok_marker"])
    route = str(best.get("best_route") or "")
    lock = {
        "plan": PLAN_PATH,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "v37_phaseA_lock",
        "source_files": {
            "v36_report": str(repo / "docs/stream4d_v36_masklet_first_object_identity_report.md"),
            "v36_recap": str(repo / "docs/stream4d_v36_实验结果复盘.md"),
            "v36_decision": str(root / V36_DECISION),
            "v36_external_summary_csv": str(root / V36_SUMMARY_CSV),
        },
        "source_exists": {
            "v36_report": (repo / "docs/stream4d_v36_masklet_first_object_identity_report.md").exists(),
            "v36_recap": (repo / "docs/stream4d_v36_实验结果复盘.md").exists(),
            "v36_decision": (root / V36_DECISION).exists(),
            "v36_external_summary_csv": (root / V36_SUMMARY_CSV).exists(),
        },
        "v36_state_loaded": bool(v36 and best),
        "v36_best_route": route,
        "v36_best_ARI": _finite(best.get("ARI")),
        "v36_best_purity": _finite(best.get("purity")),
        "v36_best_completeness": _finite(best.get("completeness")),
        "v36_best_unknown": _finite(best.get("unknown_tube_ratio")),
        "v36_best_scene0081_ARI": _finite(best.get("scene0081_ARI")),
        "v36_real_minus_shuffled": _finite(controls.get("real_minus_shuffled")),
        "v36_real_minus_no_temporal": _finite(controls.get("real_minus_no_temporal")),
        "v36_pass_3D_gate": bool(best.get("pass_3D_gate")),
        "v36_allowed_4d": bool(v36.get("allowed_4d")),
        "v36_allowed_ap": bool(v36.get("allowed_ap")),
        "validation": phase_validation,
        "py_compile_pass": bool(py_pass),
        "unittest_pass": bool(unit_pass),
        "phaseA_pass": bool(py_pass and unit_pass and v36 and route == "watershed:all_masks:real_support_chain2"),
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": [],
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "uses_d4rt_self_sim3": False,
        "uses_frozen_visual_backbone": False,
        "visual_backbone_name": "none",
        "mask_source": "none",
        "temporal_curriculum_enabled": False,
        "temporal_stage": "phaseA_lock",
        "geometry_field": "none",
        "coordinate_frame": "none",
        "alignment_source": "none",
    }
    (out_dir / "current_state_lock.json").write_text(
        json.dumps(_json_safe(lock), indent=2, sort_keys=True), encoding="utf-8"
    )
    return lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--output-root", default="outputs/audit/v37_phaseA_lock")
    args = parser.parse_args()
    print(json.dumps(_json_safe(build(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
