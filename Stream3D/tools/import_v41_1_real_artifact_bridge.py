"""Import real prior Stream3D/v37 identity-memory evidence for v41.1 audit.

This tool does not create new method metrics. It cross-checks existing v37/v40R
artifacts, records the real identity and 4D memory gates, and keeps the AP
export rows marked as diagnostic-only because they use the ScanNet RGB-D/pose/
mesh materialization bridge.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
AUDIT_ROOT = ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v41_1_real_artifact_bridge"

V37_LOCAL_DECISION = AUDIT_ROOT / "v37_adaptive_density_final_probe5/v37_final_decision/decision_summary.json"
V37_4D_DECISION = AUDIT_ROOT / "v37_4d_if_allowed_i4_sparse/4d_memory_decision.json"
V37_4D_SUMMARY = AUDIT_ROOT / "v37_4d_if_allowed_i4_sparse/4d_memory_summary.json"
V37_AP_EXPORT_SUMMARY = AUDIT_ROOT / "v37_ap_if_allowed_i4_sparse/ap_export_summary.json"
V40_FACT_LOCK = AUDIT_ROOT / "v40R_phaseA_lock/current_fact_lock.json"
V37_AP_EVAL_GLOB = ROOT / "data/evaluation/scannet"
MAIN_TABLE_DIR = AUDIT_ROOT / "v41_1_main_tables"

FLOAT_KEYS = [
    ("v37_ARI", ("best_metrics", "ARI")),
    ("v37_purity", ("best_metrics", "purity")),
    ("v37_completeness", ("best_metrics", "completeness")),
    ("v37_unknown_tube_ratio", ("best_metrics", "unknown_tube_ratio")),
]

FLOAT_4D_KEYS = [
    ("v37_4D_ARI", ("best_metrics", "4D_ARI")),
    ("v37_4D_purity", ("best_metrics", "4D_purity")),
    ("v37_4D_completeness", ("best_metrics", "4D_completeness")),
    ("v37_4D_temporal_span_mean", ("best_metrics", "temporal_span_mean")),
]


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def get_nested(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = data
    for key in keys:
        current = current[key]
    return current


def assert_close(name: str, left: Any, right: Any, *, tol: float = 1e-12) -> dict[str, Any]:
    left_f = float(left)
    right_f = float(right)
    delta = abs(left_f - right_f)
    if delta > tol:
        raise AssertionError(f"{name} mismatch: {left_f} vs {right_f} (delta={delta})")
    return {"key": name, "left": left_f, "right": right_f, "delta": delta, "pass": True}


def parse_eval_text(path: Path) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty eval text: {path}")
    parts = [part.strip() for part in lines[-1].split(",")]
    if len(parts) < 3:
        raise ValueError(f"could not parse AP row from {path}: {lines[-1]}")
    return {
        "eval_file": str(path.relative_to(REPO_ROOT)),
        "AP": float(parts[0]),
        "AP50": float(parts[1]),
        "AP25": float(parts[2]),
    }


def scan_v37_ap_eval_rows() -> list[dict[str, Any]]:
    rows = [parse_eval_text(path) for path in sorted(V37_AP_EVAL_GLOB.glob("v37_i4_sparse_ap_eval_probe5*class_agnostic.txt"))]
    if not rows:
        raise FileNotFoundError("no v37_i4_sparse_ap_eval_probe5*class_agnostic.txt files found")
    return sorted(rows, key=lambda row: float(row["AP"]), reverse=True)


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def build_rows(
    local_decision: dict[str, Any],
    memory_decision: dict[str, Any],
    ap_export_summary: dict[str, Any],
    ap_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    local = local_decision["best_metrics"]
    memory = memory_decision["best_metrics"]
    best_ap = ap_rows[0]
    raw_ap = next((row for row in ap_rows if row["eval_file"].endswith("v37_i4_sparse_ap_eval_probe5_class_agnostic.txt")), None)
    return [
        {
            "row": "v37 F31 real 3D temporal curriculum identity gate",
            "evidence_type": "imported_measured_v37_identity_gate",
            "status": local_decision.get("final_status"),
            "is_method_result": "true",
            "is_diagnostic_only": "false",
            "forbidden_for_method_table": "false",
            "uses_gt_for_prediction": bool_text(local_decision["manifest"].get("uses_gt_for_prediction")),
            "uses_rgbd_for_prediction": bool_text(local_decision["manifest"].get("uses_rgbd_for_prediction")),
            "uses_pose_for_prediction": bool_text(local_decision["manifest"].get("uses_pose_for_prediction")),
            "uses_scannet_mesh_for_prediction": bool_text(local_decision["manifest"].get("uses_scannet_mesh_for_prediction")),
            "uses_eval_sim3_for_prediction": bool_text(local_decision["manifest"].get("uses_eval_sim3_for_prediction")),
            "uses_d4rt_self_sim3": bool_text(local_decision["manifest"].get("uses_d4rt_self_sim3")),
            "stage_or_variant": local_decision.get("best_stage"),
            "ARI": local.get("ARI"),
            "purity": local.get("purity"),
            "completeness": local.get("completeness"),
            "unknown_tube_ratio": local.get("unknown_tube_ratio"),
            "same_frame_cannot_link_violations": local.get("same_frame_cannot_link_violations"),
            "AP": None,
            "AP50": None,
            "AP25": None,
            "source_artifact": rel(V37_LOCAL_DECISION),
            "note": "Real prior Stream3D/v37 identity gate; AP export not included in this row.",
        },
        {
            "row": "v37 I4 real 4D memory gate",
            "evidence_type": "imported_measured_v37_4d_memory_gate",
            "status": memory_decision.get("final_status"),
            "is_method_result": "true",
            "is_diagnostic_only": "false",
            "forbidden_for_method_table": "false",
            "uses_gt_for_prediction": "false",
            "uses_rgbd_for_prediction": "false",
            "uses_pose_for_prediction": "false",
            "uses_scannet_mesh_for_prediction": "false",
            "uses_eval_sim3_for_prediction": "false",
            "uses_d4rt_self_sim3": "true",
            "stage_or_variant": memory_decision.get("best_variant"),
            "ARI": memory.get("4D_ARI"),
            "purity": memory.get("4D_purity"),
            "completeness": memory.get("4D_completeness"),
            "unknown_tube_ratio": memory.get("unknown_tube_ratio"),
            "temporal_span_mean": memory.get("temporal_span_mean"),
            "real_minus_no_temporal": memory.get("real_minus_no_temporal"),
            "AP": None,
            "AP50": None,
            "AP25": None,
            "source_artifact": rel(V37_4D_DECISION),
            "note": "Real prior Stream3D/v37 4D memory gate; source explicitly says no AP export is included.",
        },
        {
            "row": "v37 I4 raw AP export",
            "evidence_type": "imported_measured_v37_eval_only_ap",
            "status": "diagnostic_only_forbidden_method",
            "is_method_result": "false",
            "is_diagnostic_only": "true",
            "forbidden_for_method_table": "true",
            "uses_gt_for_prediction": "false",
            "uses_rgbd_for_prediction": bool_text(ap_export_summary.get("uses_rgbd_for_prediction")),
            "uses_pose_for_prediction": bool_text(ap_export_summary.get("uses_pose_for_prediction")),
            "uses_scannet_mesh_for_prediction": bool_text(ap_export_summary.get("uses_scannet_mesh_for_prediction")),
            "uses_eval_sim3_for_prediction": "false",
            "uses_d4rt_self_sim3": "true",
            "stage_or_variant": ap_export_summary.get("variant"),
            "AP": None if raw_ap is None else raw_ap.get("AP"),
            "AP50": None if raw_ap is None else raw_ap.get("AP50"),
            "AP25": None if raw_ap is None else raw_ap.get("AP25"),
            "mean_mesh_coverage": ap_export_summary.get("mean_mesh_coverage"),
            "mean_covered_GT_instance_ratio": ap_export_summary.get("mean_covered_GT_instance_ratio"),
            "source_artifact": rel(V37_AP_EXPORT_SUMMARY),
            "note": "Existing AP export uses ScanNet RGB-D/pose/mesh bridge and is therefore diagnostic-only.",
        },
        {
            "row": "v37 I4 best AP postprocess export",
            "evidence_type": "imported_measured_v37_eval_only_ap_best_existing",
            "status": "diagnostic_only_forbidden_method",
            "is_method_result": "false",
            "is_diagnostic_only": "true",
            "forbidden_for_method_table": "true",
            "uses_gt_for_prediction": "false",
            "uses_rgbd_for_prediction": bool_text(ap_export_summary.get("uses_rgbd_for_prediction")),
            "uses_pose_for_prediction": bool_text(ap_export_summary.get("uses_pose_for_prediction")),
            "uses_scannet_mesh_for_prediction": bool_text(ap_export_summary.get("uses_scannet_mesh_for_prediction")),
            "uses_eval_sim3_for_prediction": "false",
            "uses_d4rt_self_sim3": "true",
            "stage_or_variant": Path(str(best_ap["eval_file"])).name,
            "AP": best_ap.get("AP"),
            "AP50": best_ap.get("AP50"),
            "AP25": best_ap.get("AP25"),
            "mean_mesh_coverage": ap_export_summary.get("mean_mesh_coverage"),
            "mean_covered_GT_instance_ratio": ap_export_summary.get("mean_covered_GT_instance_ratio"),
            "source_artifact": best_ap.get("eval_file"),
            "note": "Best existing v37 AP text after postprocess search; still not a method-compatible AP row.",
        },
        {
            "row": "v41.1 real method-compatible D4RT AP",
            "evidence_type": "not_run_missing_native_d4rt_ap_exporter",
            "status": "blocked_missing_method_compatible_native_d4rt_ap_exporter",
            "is_method_result": "false",
            "is_diagnostic_only": "false",
            "forbidden_for_method_table": "false",
            "uses_gt_for_prediction": "false",
            "uses_rgbd_for_prediction": "false",
            "uses_pose_for_prediction": "false",
            "uses_scannet_mesh_for_prediction": "false",
            "uses_eval_sim3_for_prediction": "false",
            "uses_d4rt_self_sim3": "true",
            "stage_or_variant": "v41.1 semantic-material inference",
            "AP": None,
            "AP50": None,
            "AP25": None,
            "source_artifact": "",
            "note": "No current exporter turns v41.1 object fields into AP masks without RGB-D/pose/mesh bridge.",
        },
    ]


def augment_main_tables(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    table1_path = MAIN_TABLE_DIR / "table1_material_to_object_fields.csv"
    manifest_path = MAIN_TABLE_DIR / "main_table_manifest.json"
    if table1_path.exists():
        original = [
            row
            for row in read_csv_rows(table1_path)
            if row.get("evidence_type")
            not in {
                "imported_measured_v37_identity_gate",
                "imported_measured_v37_4d_memory_gate",
                "not_run_missing_native_d4rt_ap_exporter",
            }
        ]
        fieldnames: list[str] = []
        for row in [*original, *rows]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        write_csv(table1_path, [*original, *rows[:2], rows[-1]], fieldnames)
    manifest = load_json(manifest_path) if manifest_path.exists() else {"phase": "v41_1_main_tables"}
    manifest.update(
        {
            "real_identity_memory_bridge": rel(OUT_DIR / "real_identity_bridge_summary.json"),
            "real_identity_memory_bridge_status": summary["bridge_status"],
            "real_identity_memory_bridge_evidence_type": "imported_measured_v37_identity_and_4d_memory",
            "real_v41_1_method_ap_status": summary["real_v41_1_method_ap_status"],
            "real_v41_1_method_ap_blocker": summary["blocker"],
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    local_decision = load_json(V37_LOCAL_DECISION)
    memory_decision = load_json(V37_4D_DECISION)
    memory_summary = load_json(V37_4D_SUMMARY)
    ap_export_summary = load_json(V37_AP_EXPORT_SUMMARY)
    fact_lock = load_json(V40_FACT_LOCK)
    locked = fact_lock["locked_prior_facts"]

    consistency_rows = []
    for fact_key, nested in FLOAT_KEYS:
        consistency_rows.append(assert_close(fact_key, locked[fact_key], get_nested(local_decision, nested)))
    for fact_key, nested in FLOAT_4D_KEYS:
        consistency_rows.append(assert_close(fact_key, locked[fact_key], get_nested(memory_decision, nested)))
    if locked["v37_stage"] != local_decision.get("best_stage"):
        raise AssertionError("v37_stage mismatch")
    if locked["v37_i4_variant"] != memory_decision.get("best_variant"):
        raise AssertionError("v37_i4_variant mismatch")
    if memory_decision.get("best_metrics") not in memory_summary:
        raise AssertionError("best 4D memory metrics not found in 4d_memory_summary.json")

    manifest = local_decision["manifest"]
    method_constraints_pass = bool(
        local_decision.get("final_status") == "GO_3D_TEMPORAL_CURRICULUM"
        and memory_decision.get("final_status") == "GO_4D_MEMORY"
        and manifest.get("uses_gt_for_prediction") is False
        and manifest.get("uses_rgbd_for_prediction") is False
        and manifest.get("uses_pose_for_prediction") is False
        and manifest.get("uses_scannet_mesh_for_prediction") is False
        and manifest.get("uses_eval_sim3_for_prediction") is False
        and manifest.get("uses_d4rt_self_sim3") is True
        and int(local_decision["best_metrics"].get("same_frame_cannot_link_violations", 1)) == 0
        and float(memory_decision["best_metrics"].get("unknown_tube_ratio", 0.0)) > 0.0
        and bool(memory_decision["best_metrics"].get("pass_4D_gate"))
    )
    ap_export_forbidden = bool(
        ap_export_summary.get("is_diagnostic_only") is True
        and ap_export_summary.get("is_method_result") is False
        and ap_export_summary.get("uses_rgbd_for_prediction") is True
        and ap_export_summary.get("uses_pose_for_prediction") is True
        and ap_export_summary.get("uses_scannet_mesh_for_prediction") is True
    )
    ap_rows = scan_v37_ap_eval_rows()
    bridge_rows = build_rows(local_decision, memory_decision, ap_export_summary, ap_rows)

    summary = {
        "phase": "v41_1_real_artifact_bridge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_artifacts": {
            "v37_local_decision": rel(V37_LOCAL_DECISION),
            "v37_4d_decision": rel(V37_4D_DECISION),
            "v37_4d_summary": rel(V37_4D_SUMMARY),
            "v37_ap_export_summary": rel(V37_AP_EXPORT_SUMMARY),
            "v40_fact_lock": rel(V40_FACT_LOCK),
        },
        "bridge_status": "PARTIAL_REAL_IDENTITY_MEMORY_PASS_AP_EXPORT_BLOCKED",
        "real_identity_memory_gate_pass": method_constraints_pass,
        "ap_export_existing_diagnostic_only": ap_export_forbidden,
        "real_v41_1_method_ap_status": "not_run",
        "blocker": "method_compatible_native_d4rt_ap_exporter_missing",
        "blocker_repair_attempted": [
            "Imported v37 real identity and 4D memory gates instead of relying on synthetic proxy only.",
            "Scanned existing v37 AP export/postprocess eval texts and selected the best diagnostic AP row.",
            "Inspected v37 AP exporter summary and kept AP rows forbidden because RGB-D/pose/mesh bridge is used.",
        ],
        "key_numbers": {
            "v37_ARI": local_decision["best_metrics"]["ARI"],
            "v37_purity": local_decision["best_metrics"]["purity"],
            "v37_completeness": local_decision["best_metrics"]["completeness"],
            "v37_same_frame_cannot_link_violations": local_decision["best_metrics"]["same_frame_cannot_link_violations"],
            "v37_unknown_tube_ratio": local_decision["best_metrics"]["unknown_tube_ratio"],
            "v37_4D_ARI": memory_decision["best_metrics"]["4D_ARI"],
            "v37_4D_purity": memory_decision["best_metrics"]["4D_purity"],
            "v37_4D_completeness": memory_decision["best_metrics"]["4D_completeness"],
            "v37_4D_temporal_span_mean": memory_decision["best_metrics"]["temporal_span_mean"],
            "v37_4D_unknown_tube_ratio": memory_decision["best_metrics"]["unknown_tube_ratio"],
            "v37_best_existing_diagnostic_AP": ap_rows[0]["AP"],
            "v37_best_existing_diagnostic_AP50": ap_rows[0]["AP50"],
            "v37_best_existing_diagnostic_AP25": ap_rows[0]["AP25"],
            "v37_mean_mesh_coverage": ap_export_summary.get("mean_mesh_coverage"),
            "v37_mean_covered_GT_instance_ratio": ap_export_summary.get("mean_covered_GT_instance_ratio"),
        },
        "consistency_checks": consistency_rows,
        "ap_eval_rows_scanned": ap_rows,
        "outputs": {
            "summary_json": rel(OUT_DIR / "real_identity_bridge_summary.json"),
            "rows_csv": rel(OUT_DIR / "real_identity_bridge_rows.csv"),
            "ap_eval_scan_csv": rel(OUT_DIR / "v37_ap_eval_scan.csv"),
            "answer_md": rel(OUT_DIR / "real_identity_bridge_answer.md"),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "real_identity_bridge_rows.csv", bridge_rows)
    write_csv(OUT_DIR / "v37_ap_eval_scan.csv", ap_rows)
    write_csv(OUT_DIR / "consistency_checks.csv", consistency_rows)
    write_json(OUT_DIR / "real_identity_bridge_summary.json", summary)
    augment_main_tables(bridge_rows, summary)

    answer_lines = [
        "# v41.1 Real Artifact Bridge",
        "",
        f"Status: `{summary['bridge_status']}`",
        "",
        "This bridge imports measured v37 Stream3D identity/memory evidence. It does not create new AP metrics.",
        "",
        f"- real identity/memory gate pass: `{method_constraints_pass}`",
        f"- AP export diagnostic-only: `{ap_export_forbidden}`",
        f"- blocker: `{summary['blocker']}`",
        f"- best existing diagnostic AP: `{ap_rows[0]['AP']}` from `{ap_rows[0]['eval_file']}`",
        "",
        "| row | evidence | status | AP | note |",
        "|---|---|---|---:|---|",
    ]
    for row in bridge_rows:
        answer_lines.append(
            "| {row} | {evidence} | {status} | {ap} | {note} |".format(
                row=row["row"],
                evidence=row["evidence_type"],
                status=row["status"],
                ap="" if row.get("AP") is None else row.get("AP"),
                note=row["note"],
            )
        )
    (OUT_DIR / "real_identity_bridge_answer.md").write_text("\n".join(answer_lines) + "\n", encoding="utf-8")
    print(json.dumps({"bridge_status": summary["bridge_status"], "real_identity_memory_gate_pass": method_constraints_pass, "real_v41_1_method_ap_status": summary["real_v41_1_method_ap_status"]}, indent=2))


if __name__ == "__main__":
    main()
