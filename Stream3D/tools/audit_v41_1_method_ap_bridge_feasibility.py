"""Audit whether corrected v41.1 native ObjectFields can enter ScanNet AP as a method result."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _rel(path: Path) -> str:
    resolved = path if path.is_absolute() else ROOT / path
    try:
        return str(resolved.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _source_has(path: Path, text: str) -> bool:
    return text in path.read_text(encoding="utf-8")


def _npz_keys(path: Path) -> list[str]:
    import numpy as np

    with np.load(path) as payload:
        return sorted(str(key) for key in payload.files)


def _main(args: argparse.Namespace) -> dict[str, Any]:
    final_summary_path = Path(args.final_native_support_summary)
    smoke_npz_path = Path(args.native_points_npz)
    export_scannet_path = ROOT / "stream4d/export_scannet.py"
    evaluate_path = ROOT / "evaluation/evaluate.py"

    final_summary = _read_json(final_summary_path)
    npz_keys = _npz_keys(smoke_npz_path) if smoke_npz_path.exists() else []
    mesh_id_keys = {
        "mesh_vertex_id",
        "mesh_vertex_ids",
        "scene_point_id",
        "scene_point_ids",
        "scannet_vertex_id",
        "scannet_vertex_ids",
        "point_id",
        "point_ids",
    }
    available_mesh_keys = sorted(mesh_id_keys.intersection(npz_keys))
    aggregate = final_summary.get("aggregate_metrics", {})
    gate = final_summary.get("gate", {})

    checks = [
        {
            "check": "corrected native-support object gate is passed",
            "pass": gate.get("pass_native_support_metric_gate") is True,
            "evidence": _rel(final_summary_path),
            "detail": (
                f"ARI={aggregate.get('4D_ARI')}, purity={aggregate.get('4D_purity')}, "
                f"completeness={aggregate.get('4D_completeness')}"
            ),
        },
        {
            "check": "native-support artifact is explicitly not ScanNet AP",
            "pass": final_summary.get("AP_bridge_status") == "not_evaluated_native_support_metrics_only"
            and final_summary.get("real_method_ap_status") == "not_run",
            "evidence": _rel(final_summary_path),
            "detail": (
                f"AP_bridge_status={final_summary.get('AP_bridge_status')}, "
                f"real_method_ap_status={final_summary.get('real_method_ap_status')}"
            ),
        },
        {
            "check": "native-support prediction path does not use forbidden prediction sources",
            "pass": final_summary.get("prediction_uses_gt") is False
            and final_summary.get("prediction_uses_rgbd") is False
            and final_summary.get("prediction_uses_pose") is False
            and final_summary.get("prediction_uses_scannet_mesh") is False,
            "evidence": _rel(final_summary_path),
            "detail": "prediction_uses_gt/rgbd/pose/scannet_mesh are all false",
        },
        {
            "check": "native point artifact has no mesh vertex ids",
            "pass": smoke_npz_path.exists() and not available_mesh_keys,
            "evidence": _rel(smoke_npz_path) if smoke_npz_path.exists() else str(smoke_npz_path),
            "detail": f"npz_keys={npz_keys}; mesh_id_keys_present={available_mesh_keys}",
        },
        {
            "check": "ScanNet evaluator requires mesh-vertex masks",
            "pass": _source_has(evaluate_path, "wrong number of lines")
            and _source_has(evaluate_path, "vs #mesh vertices")
            and _source_has(evaluate_path, "pred_masks"),
            "evidence": _rel(evaluate_path),
            "detail": "evaluation checks prediction mask length against GT mesh vertex count",
        },
        {
            "check": "existing ScanNet export bridge materializes through forbidden RGB-D/pose/mesh",
            "pass": _source_has(export_scannet_path, "self.scene_points")
            and _source_has(export_scannet_path, "load_depth")
            and _source_has(export_scannet_path, "load_pose")
            and _source_has(export_scannet_path, "Diagnostic-only RGB-D bridge export"),
            "evidence": _rel(export_scannet_path),
            "detail": "ScanNetExporter builds mesh point ids through depth/pose/mesh backprojection paths",
        },
        {
            "check": "native D4RT AP exporter remains unimplemented",
            "pass": _source_has(export_scannet_path, "def export_d4rt_nn")
            and _source_has(export_scannet_path, "raise NotImplementedError")
            and _source_has(export_scannet_path, "scene-coordinate calibration path"),
            "evidence": _rel(export_scannet_path),
            "detail": "export_d4rt_nn still requires a scene-coordinate calibration path",
        },
    ]
    all_pass = all(bool(row["pass"]) for row in checks)
    status = (
        "NO_GO_METHOD_AP_BRIDGE_REQUIRES_MESH_VERTEX_MASKS_AND_MISSING_NATIVE_CALIBRATION"
        if all_pass
        else "METHOD_AP_BRIDGE_FEASIBILITY_INCONCLUSIVE"
    )
    out_dir = Path(args.output_root)
    summary = {
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checks_all_pass": all_pass,
        "method_ap_goal_reached": False,
        "native_support_gate_pass": gate.get("pass_native_support_metric_gate") is True,
        "real_method_ap_status": final_summary.get("real_method_ap_status"),
        "AP_bridge_status": final_summary.get("AP_bridge_status"),
        "blocker": "ScanNet AP requires mesh-vertex prediction masks, but v41.1 method-safe native support has no mesh vertex ids and the only existing materialization routes use forbidden RGB-D/pose/mesh or unimplemented native calibration.",
        "repair_direction_evaluated": [
            "Use corrected v41.1 native ObjectFields directly: blocked because native points have no mesh vertex ids.",
            "Use existing ScanNetExporter: diagnostic-only because it loads ScanNet mesh/depth/pose to produce vertex masks.",
            "Use export_d4rt_nn: blocked because it raises NotImplementedError for missing scene-coordinate calibration.",
        ],
        "final_native_support_summary": _rel(final_summary_path),
        "native_points_npz": _rel(smoke_npz_path) if smoke_npz_path.exists() else str(smoke_npz_path),
        "native_points_npz_keys": npz_keys,
        "checks": checks,
        "outputs": {
            "summary_json": _rel(out_dir / "method_ap_bridge_feasibility_summary.json"),
            "evidence_csv": _rel(out_dir / "method_ap_bridge_feasibility_checks.csv"),
            "answer_md": _rel(out_dir / "method_ap_bridge_feasibility_answer.md"),
        },
    }
    _write_json(out_dir / "method_ap_bridge_feasibility_summary.json", summary)
    _write_csv(out_dir / "method_ap_bridge_feasibility_checks.csv", checks)
    lines = [
        "# v41.1 Method AP Bridge Feasibility",
        "",
        f"Status: `{status}`",
        "",
        "| check | pass | evidence | detail |",
        "|---|---:|---|---|",
    ]
    for row in checks:
        lines.append(f"| {row['check']} | {row['pass']} | `{row['evidence']}` | {row['detail']} |")
    (out_dir / "method_ap_bridge_feasibility_answer.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-native-support-summary",
        default=(
            "outputs/audit/v41_1_native_support_metrics_probe5_sweep/"
            "offsetfix2_closure_rgb090_t035_m010_birthgate/native_support_metrics_summary.json"
        ),
    )
    parser.add_argument(
        "--native-points-npz",
        default="outputs/audit/v41_1_native_object_field_export_smoke/native_object_points.npz",
    )
    parser.add_argument("--output-root", default="outputs/audit/v41_1_method_ap_bridge_feasibility")
    return parser


def main() -> None:
    summary = _main(_parser().parse_args())
    print(json.dumps({"status": summary["status"], "checks_all_pass": summary["checks_all_pass"]}, indent=2))


if __name__ == "__main__":
    main()
