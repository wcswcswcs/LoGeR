"""Audit why v41.1 still lacks a method-compatible native D4RT AP row."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
AUDIT_ROOT = ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v41_1_native_ap_exporter_blocker"

V40_DASHBOARD = AUDIT_ROOT / "v40R_root_cause_dashboard/root_cause_dashboard.json"
V37_AP_EXPORT_SUMMARY = AUDIT_ROOT / "v37_ap_if_allowed_i4_sparse/ap_export_summary.json"
EXPORT_SCANNET = ROOT / "stream4d/export_scannet.py"
RUN_SCANNET = ROOT / "stream4d/run_scannet.py"
PROVIDER_REPLACEMENT = ROOT / "tools/run_v21_3_stream3d_provider_replacement.py"
V37_AP_TOOL = ROOT / "tools/run_v37_ap_if_allowed.py"


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def find_row(dashboard: dict[str, Any], row_id: str) -> dict[str, Any]:
    for row in dashboard.get("rows", []):
        if row.get("id") == row_id:
            return row
    raise KeyError(row_id)


def source_has(path: Path, pattern: str) -> bool:
    return pattern in path.read_text(encoding="utf-8")


def main() -> None:
    dashboard = load_json(V40_DASHBOARD)
    ap_export_summary = load_json(V37_AP_EXPORT_SUMMARY)
    s_d4rt_g6 = find_row(dashboard, "S-D4RT-G6")
    o_d4rt_native = find_row(dashboard, "O-D4RT-native")

    checks = [
        {
            "check": "O-D4RT-native dashboard row is unavailable",
            "pass": o_d4rt_native.get("status") == "not_available_current_run",
            "evidence": rel(V40_DASHBOARD),
            "detail": o_d4rt_native.get("purpose"),
        },
        {
            "check": "S-D4RT-G6 is Stream3D provider replacement, not v41.1 object-field AP",
            "pass": s_d4rt_g6.get("object_algorithm") == "Stream3D original"
            and s_d4rt_g6.get("materializer") == "Stream3D provider replacement path",
            "evidence": rel(V40_DASHBOARD),
            "detail": f"AP={s_d4rt_g6.get('AP')}, forbidden={s_d4rt_g6.get('forbidden_for_method_table')}",
        },
        {
            "check": "v37 AP exporter summary is diagnostic-only and uses RGB-D/pose/mesh",
            "pass": ap_export_summary.get("is_diagnostic_only") is True
            and ap_export_summary.get("is_method_result") is False
            and ap_export_summary.get("uses_rgbd_for_prediction") is True
            and ap_export_summary.get("uses_pose_for_prediction") is True
            and ap_export_summary.get("uses_scannet_mesh_for_prediction") is True,
            "evidence": rel(V37_AP_EXPORT_SUMMARY),
            "detail": f"mean_mesh_coverage={ap_export_summary.get('mean_mesh_coverage')}",
        },
        {
            "check": "legacy d4rt_nn export path is not implemented",
            "pass": source_has(EXPORT_SCANNET, "def export_d4rt_nn") and source_has(EXPORT_SCANNET, "raise NotImplementedError"),
            "evidence": rel(EXPORT_SCANNET),
            "detail": "export_d4rt_nn raises NotImplementedError for missing scene-coordinate calibration path.",
        },
        {
            "check": "RGB-D eval export path is explicitly forbidden for method table",
            "pass": source_has(EXPORT_SCANNET, "Diagnostic-only RGB-D bridge export")
            and source_has(EXPORT_SCANNET, "\"forbidden_for_method_table\": True"),
            "evidence": rel(EXPORT_SCANNET),
            "detail": "export_rgbd_eval writes a diagnostic-only prediction manifest.",
        },
        {
            "check": "v21 Stream3D provider replacement is diagnostic-only",
            "pass": source_has(PROVIDER_REPLACEMENT, "is_method_result=False")
            and source_has(PROVIDER_REPLACEMENT, "is_diagnostic_only=True")
            and source_has(PROVIDER_REPLACEMENT, "\"forbidden_for_method_table\": True"),
            "evidence": rel(PROVIDER_REPLACEMENT),
            "detail": "Provider replacement reruns Stream3D original algorithm and writes forbidden diagnostic manifests.",
        },
        {
            "check": "v37 AP tool uses ScanNetExporter backprojection bridge",
            "pass": source_has(V37_AP_TOOL, "ScanNetExporter")
            and source_has(V37_AP_TOOL, "export_support_mode=\"mask_backproject\"")
            and source_has(V37_AP_TOOL, "uses_rgbd_for_prediction"),
            "evidence": rel(V37_AP_TOOL),
            "detail": "Tool builds AP masks by mask backprojection through ScanNetExporter.",
        },
        {
            "check": "current AP exporters do not consume v41.1 semantic-material ObjectField",
            "pass": not any(
                source_has(path, pattern)
                for path in [EXPORT_SCANNET, RUN_SCANNET, V37_AP_TOOL]
                for pattern in ["SemanticMaterial", "ObjectField", "object_fields", "semantic_material"]
            ),
            "evidence": ", ".join(rel(path) for path in [EXPORT_SCANNET, RUN_SCANNET, V37_AP_TOOL]),
            "detail": "No object-field adapter entry point was found in the existing AP export tools.",
        },
    ]
    all_pass = all(bool(row["pass"]) for row in checks)
    summary = {
        "phase": "v41_1_native_ap_exporter_blocker",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "NO_GO_NATIVE_AP_EXPORTER_NOT_AVAILABLE" if all_pass else "CHECKS_INCONCLUSIVE",
        "checks_all_pass": all_pass,
        "blocker": "method_compatible_native_d4rt_ap_exporter_missing",
        "repair_direction_explored": [
            "Reuse Stream3D D4RT provider row: rejected as S-route provider replacement, not v41.1/O object-field materialization.",
            "Reuse v37 AP exporter: rejected as diagnostic-only RGB-D/pose/mesh materialization.",
            "Use stream4d d4rt_nn exporter: blocked because export_d4rt_nn is not implemented.",
            "Directly feed v41.1 ObjectField to existing exporters: blocked because current AP tools have no ObjectField/SemanticMaterial adapter.",
        ],
        "key_numbers": {
            "S_D4RT_G6_AP": s_d4rt_g6.get("AP"),
            "S_D4RT_G6_AP50": s_d4rt_g6.get("AP50"),
            "S_D4RT_G6_AP25": s_d4rt_g6.get("AP25"),
            "O_D4RT_native_status": o_d4rt_native.get("status"),
            "v37_ap_export_mean_mesh_coverage": ap_export_summary.get("mean_mesh_coverage"),
        },
        "checks": checks,
        "outputs": {
            "summary_json": rel(OUT_DIR / "native_ap_exporter_blocker_summary.json"),
            "evidence_csv": rel(OUT_DIR / "native_ap_exporter_blocker_evidence.csv"),
            "answer_md": rel(OUT_DIR / "native_ap_exporter_blocker_answer.md"),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "native_ap_exporter_blocker_evidence.csv", checks)
    write_json(OUT_DIR / "native_ap_exporter_blocker_summary.json", summary)
    lines = [
        "# v41.1 Native AP Exporter Blocker",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| check | pass | evidence | detail |",
        "|---|---:|---|---|",
    ]
    for row in checks:
        lines.append(f"| {row['check']} | {row['pass']} | `{row['evidence']}` | {row['detail']} |")
    (OUT_DIR / "native_ap_exporter_blocker_answer.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "checks_all_pass": all_pass}, indent=2))


if __name__ == "__main__":
    main()
