from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stream4d_native.v47_common import ROOT, write_csv, write_json, utc_now


PLAN_PATH = "docs/stream4d_v51_r2_mosaic_remask_lift_codex_plan.md"


def _rel(path: str | Path) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = ROOT / path_obj
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v51-r2 AP diagnostic policy/blocker summary.")
    parser.add_argument("--hypothesis-root", required=True)
    parser.add_argument("--output-root", default="outputs/audit/v51_r2_ap_diagnostic")
    args = parser.parse_args()
    hyp_root = ROOT / args.hypothesis_root if not Path(args.hypothesis_root).is_absolute() else Path(args.hypothesis_root)
    out = ROOT / args.output_root if not Path(args.output_root).is_absolute() else Path(args.output_root)
    summary_path = hyp_root / "hypothesis_selection_summary.json"
    hyp = json.loads(summary_path.read_text(encoding="utf-8"))
    native_materialization = bool(hyp.get("summary", {}).get("native_3d_materialization_available"))
    selected_object_count = int(hyp.get("summary", {}).get("selected_object_count") or 0)
    generic_exporter = ROOT / "stream4d/export_scannet.py"
    policy_rows: list[dict[str, Any]] = [
        {
            "row": "AP3_v51_r2_best_native_export",
            "selected_object_count": selected_object_count,
            "exporter_input_rows_available": selected_object_count > 0,
            "native_3d_materialization_available": native_materialization,
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "is_diagnostic_only": False,
            "forbidden_for_method_table": False,
            "status": "blocked_native_materialization_missing" if not native_materialization else "ready",
        },
        {
            "row": "AP6_v51_r2_best_rgbd_pose_mesh_bridge_diagnostic",
            "selected_object_count": selected_object_count,
            "exporter_input_rows_available": selected_object_count > 0,
            "native_3d_materialization_available": native_materialization,
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": True,
            "uses_rgbd_pose_mesh_for_export": True,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "status": "not_run_plan_requires_native_first",
        },
    ]
    metric_rows: list[dict[str, Any]] = []
    failure_rows = [
        {
            "failure_label": "NO_GO_NATIVE_MATERIALIZATION",
            "phase": "Phase10_AP",
            "evidence": "v51 selected component-set objects do not yet have a method-safe object-field-to-ScanNet-prediction materializer",
            "selected_object_count": selected_object_count,
            "native_3d_materialization_available": native_materialization,
            "generic_exporter_exists": generic_exporter.exists(),
            "hypothesis_summary": _rel(summary_path),
            "uses_gt_for_prediction": False,
        }
    ]
    gate = {
        "selected_object_count_gt0": selected_object_count > 0,
        "native_3d_materialization_available": native_materialization,
        "generic_exporter_exists": generic_exporter.exists(),
        "exporter_exit_code": None,
        "evaluator_exit_code": None,
        "prediction_file_exists": False,
        "ap_smoke_pass": False,
        "ap_diagnostic_ran": False,
        "uses_gt_for_prediction": False,
        "pass": False,
    }
    payload = {
        "phase": "v51_r2_ap_diagnostic",
        "created_at": utc_now(),
        "plan": PLAN_PATH,
        "hypothesis_root": _rel(hyp_root),
        "summary": {
            "selected_object_count": selected_object_count,
            "native_3d_materialization_available": native_materialization,
            "generic_exporter_exists": generic_exporter.exists(),
            "AP": None,
            "AP50": None,
            "AP25": None,
            "failure_label": "NO_GO_NATIVE_MATERIALIZATION",
        },
        "gate": gate,
        "policy_rows": policy_rows,
        "metric_rows": metric_rows,
        "failure_rows": failure_rows,
    }
    write_json(out / "ap_export_summary.json", payload)
    write_csv(out / "ap_policy_rows.csv", policy_rows)
    write_csv(out / "ap_metric_rows.csv", metric_rows)
    write_csv(out / "ap_failure_casebook.csv", failure_rows)
    print({"summary": f"{args.output_root}/ap_export_summary.json", "gate": gate, "failure": failure_rows[0]})


if __name__ == "__main__":
    main()
