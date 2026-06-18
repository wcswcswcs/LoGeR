"""Build the Stream3D-first comparison lock for Stream4D v41.1.

This script intentionally imports measured v40R audit rows instead of inventing
new v41.1 metrics. v41.1 method rows that have not been implemented or run are
emitted as explicit not_run rows.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
V40_DASHBOARD = ROOT / "outputs/audit/v40R_root_cause_dashboard/root_cause_dashboard.json"
V40_FAIRNESS = ROOT / "outputs/audit/v40R_phaseB_fairness/input_fairness_manifest.json"
V40_DECISION = ROOT / "outputs/audit/v40R_final_decision/decision_summary.json"
V40_FACT_LOCK = ROOT / "outputs/audit/v40R_phaseA_lock/current_fact_lock.json"
OUT_DIR = ROOT / "outputs/audit/v41_1_stream3d_first_comparison"


BASE_COLUMNS = [
    "table4_label",
    "source_row_id",
    "status",
    "AP",
    "AP50",
    "AP25",
    "mean_predictions_per_scene",
    "delta_AP_vs_S_GT_v40R_G0",
    "delta_AP_vs_locked_same_support",
    "is_method_result",
    "is_diagnostic_only",
    "forbidden_for_method_table",
    "training_free",
    "uses_gt_for_prediction",
    "uses_rgbd_for_prediction",
    "uses_pose_for_prediction",
    "uses_scannet_mesh_for_prediction",
    "uses_eval_sim3_for_prediction",
    "geometry_backend",
    "object_algorithm",
    "materializer",
    "source_artifact",
    "note",
]


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def row_index(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("id")): row for row in rows}


def select_row(
    label: str,
    row_id: str,
    rows: Dict[str, Dict[str, Any]],
    s_gt_ap: Optional[float],
    same_support_ap: Optional[float],
    note: str,
) -> Dict[str, Any]:
    src = rows[row_id]
    ap = safe_float(src.get("AP"))
    return {
        "table4_label": label,
        "source_row_id": row_id,
        "status": src.get("status", ""),
        "AP": src.get("AP", ""),
        "AP50": src.get("AP50", ""),
        "AP25": src.get("AP25", ""),
        "mean_predictions_per_scene": src.get("mean_predictions_per_scene", ""),
        "delta_AP_vs_S_GT_v40R_G0": None if ap is None or s_gt_ap is None else ap - s_gt_ap,
        "delta_AP_vs_locked_same_support": None if ap is None or same_support_ap is None else ap - same_support_ap,
        "is_method_result": src.get("is_method_result", ""),
        "is_diagnostic_only": src.get("is_diagnostic_only", ""),
        "forbidden_for_method_table": src.get("forbidden_for_method_table", ""),
        "training_free": src.get("training_free", ""),
        "uses_gt_for_prediction": src.get("uses_gt_for_prediction", ""),
        "uses_rgbd_for_prediction": src.get("uses_rgbd_for_prediction", ""),
        "uses_pose_for_prediction": src.get("uses_pose_for_prediction", ""),
        "uses_scannet_mesh_for_prediction": src.get("uses_scannet_mesh_for_prediction", ""),
        "uses_eval_sim3_for_prediction": src.get("uses_eval_sim3_for_prediction", ""),
        "geometry_backend": src.get("geometry_backend", ""),
        "object_algorithm": src.get("object_algorithm", ""),
        "materializer": src.get("materializer", ""),
        "source_artifact": src.get("source_artifact", ""),
        "note": note,
    }


def not_run_row(label: str, row_id: str, note: str) -> Dict[str, Any]:
    return {
        "table4_label": label,
        "source_row_id": row_id,
        "status": "not_run_v41_1_not_implemented_yet",
        "AP": "",
        "AP50": "",
        "AP25": "",
        "mean_predictions_per_scene": "",
        "delta_AP_vs_S_GT_v40R_G0": "",
        "delta_AP_vs_locked_same_support": "",
        "is_method_result": "false",
        "is_diagnostic_only": "false",
        "forbidden_for_method_table": "false",
        "training_free": "true",
        "uses_gt_for_prediction": "false",
        "uses_rgbd_for_prediction": "false",
        "uses_pose_for_prediction": "false",
        "uses_scannet_mesh_for_prediction": "false",
        "uses_eval_sim3_for_prediction": "false",
        "geometry_backend": "",
        "object_algorithm": "v41.1 semantic-material inference",
        "materializer": "not implemented in this lock step",
        "source_artifact": "",
        "note": note,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in BASE_COLUMNS})


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    header = "| label | source | AP | AP50 | AP25 | status | note |\n"
    sep = "|---|---|---:|---:|---:|---|---|\n"
    body = []
    for row in rows:
        body.append(
            "| {label} | {source} | {ap} | {ap50} | {ap25} | {status} | {note} |".format(
                label=row["table4_label"],
                source=row["source_row_id"],
                ap=fmt(row["AP"]),
                ap50=fmt(row["AP50"]),
                ap25=fmt(row["AP25"]),
                status=row["status"],
                note=row["note"],
            )
        )
    return header + sep + "\n".join(body) + "\n"


def main() -> None:
    dashboard = load_json(V40_DASHBOARD)
    fairness = load_json(V40_FAIRNESS)
    decision = load_json(V40_DECISION)
    fact_lock = load_json(V40_FACT_LOCK)
    rows = row_index(dashboard["rows"])

    s_gt_ap = safe_float(rows["S-GT-v40R-G0"].get("AP"))
    same_support_ap = safe_float(rows["S-GT-locked-v38-same-support"].get("AP"))

    selected = [
        select_row(
            "Stream3D + RGB-D/pose geometry",
            "S-GT-v40R-G0",
            rows,
            s_gt_ap,
            same_support_ap,
            "v40R measured diagnostic baseline; not a method row because it uses ScanNet RGB-D/pose/mesh provider.",
        ),
        select_row(
            "Stream3D + D4RT self-canonical geometry",
            "S-D4RT-G6",
            rows,
            s_gt_ap,
            same_support_ap,
            "best v40R self-canonical D4RT repair available in dashboard.",
        ),
        select_row(
            "Stream3D + D4RT eval-Sim3 diagnostic geometry",
            "S-D4RT-Eval-G4",
            rows,
            s_gt_ap,
            same_support_ap,
            "diagnostic-only eval-Sim3/outlier repair; forbidden as method evidence.",
        ),
        select_row(
            "Old Stream4D candidate-first + GT/RGB-D geometry",
            "O-GTGeo-rescore-min100",
            rows,
            s_gt_ap,
            same_support_ap,
            "best simple O-GTGeo repair from v40R; still diagnostic-only and low.",
        ),
        select_row(
            "Old Stream4D compact materializer + GT/RGB-D geometry",
            "O-GTGeo-Compact-K2",
            rows,
            s_gt_ap,
            same_support_ap,
            "best compact materializer row from v40R dashboard.",
        ),
        select_row(
            "Old candidate pool GT oracle upper bound",
            "O-OldPoolOracle-B1",
            rows,
            s_gt_ap,
            same_support_ap,
            "GT diagnostic upper bound for old candidate pool; below 0.35 source-oracle gate.",
        ),
        select_row(
            "Old Stream4D D4RT-native AP row",
            "O-D4RT-native",
            rows,
            s_gt_ap,
            same_support_ap,
            "missing in v40R because no method-compatible native D4RT AP exporter existed.",
        ),
        not_run_row(
            "v41.1 semantic-material inference + D4RT",
            "V41_1-O-D4RT",
            "not run yet; this lock is intentionally created before v41.1 implementation.",
        ),
        not_run_row(
            "v41.1 semantic-material inference + GT/RGB-D diagnostic geometry",
            "V41_1-O-GTGeo-diagnostic",
            "not run yet; should stay diagnostic-only once implemented because GT/RGB-D geometry is forbidden for method prediction.",
        ),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_csv = OUT_DIR / "table4_static_bridge_stream3d_first.csv"
    summary_json = OUT_DIR / "stream3d_first_comparison_summary.json"
    answer_md = OUT_DIR / "stream3d_first_comparison_answer.md"
    manifest_json = OUT_DIR / "manifest.json"

    write_csv(table_csv, selected)

    summary = {
        "phase": "v41_1_stream3d_first_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "training_free": True,
            "no_new_metrics_fabricated": True,
            "v41_1_method_rows_not_run_yet": True,
            "diagnostic_rows_forbidden_for_method_claim": True,
        },
        "input_artifacts": {
            "v40_dashboard": str(V40_DASHBOARD.relative_to(REPO_ROOT)),
            "v40_fairness_manifest": str(V40_FAIRNESS.relative_to(REPO_ROOT)),
            "v40_decision": str(V40_DECISION.relative_to(REPO_ROOT)),
            "v40_fact_lock": str(V40_FACT_LOCK.relative_to(REPO_ROOT)),
        },
        "output_artifacts": {
            "table_csv": str(table_csv.relative_to(REPO_ROOT)),
            "answer_md": str(answer_md.relative_to(REPO_ROOT)),
            "manifest_json": str(manifest_json.relative_to(REPO_ROOT)),
        },
        "key_numbers": {
            "S_GT_v40R_G0_AP": s_gt_ap,
            "S_GT_locked_same_support_AP": same_support_ap,
            "S_D4RT_G6_AP": safe_float(rows["S-D4RT-G6"].get("AP")),
            "S_D4RT_Eval_G4_AP": safe_float(rows["S-D4RT-Eval-G4"].get("AP")),
            "O_GTGeo_rescore_min100_AP": safe_float(rows["O-GTGeo-rescore-min100"].get("AP")),
            "O_GTGeo_compact_K2_AP": safe_float(rows["O-GTGeo-Compact-K2"].get("AP")),
            "O_old_pool_oracle_B1_AP": safe_float(rows["O-OldPoolOracle-B1"].get("AP")),
            "O_D4RT_native_status": rows["O-D4RT-native"].get("status"),
            "v41_1_method_status": "not_run",
        },
        "v40_final_status": decision.get("final_status"),
        "v40_root_cause_classification": decision.get("root_cause_classification"),
        "v40_fairness_notes": fairness.get("notes", []),
        "v40_locked_prior_facts_subset": {
            k: fact_lock.get("locked_prior_facts", {}).get(k)
            for k in [
                "same_support_stream3d_AP",
                "same_support_stream3d_AP50",
                "same_support_stream3d_AP25",
                "v37_4D_ARI",
                "v37_4D_purity",
                "v37_4D_completeness",
                "v37_unknown_tube_ratio",
            ]
        },
        "interpretation": [
            "Stream3D RGB-D/pose diagnostic rows remain the strongest AP bridge reference.",
            "Stream3D with D4RT provider under v40R is far below Stream3D RGB-D/pose baseline.",
            "Old Stream4D candidate-first rows remain low even with GT/RGB-D diagnostic geometry.",
            "v41.1 rows are intentionally not filled before implementation and measurement.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    answer = [
        "# v41.1 Stream3D-first comparison lock",
        "",
        "This artifact is generated before v41.1 method implementation. It imports measured v40R benchmark rows and keeps v41.1 rows explicit as not_run.",
        "",
        markdown_table(selected),
        "## Interpretation",
        "",
        "- Stream3D RGB-D/pose diagnostic baseline is still the strongest AP bridge reference in this lock.",
        "- Stream3D + D4RT self-canonical repair is much lower than Stream3D + RGB-D/pose geometry.",
        "- Old Stream4D candidate-first rows are also low under GT/RGB-D diagnostic geometry, so the v41.1 route must solve object/source identity rather than just AP post-processing.",
        "- No v41.1 method superiority claim is made here because v41.1 method rows are not implemented or measured yet.",
        "",
    ]
    answer_md.write_text("\n".join(answer))

    manifest = {
        "phase": "v41_1_stream3d_first_comparison",
        "generated_files": [
            str(table_csv.relative_to(REPO_ROOT)),
            str(summary_json.relative_to(REPO_ROOT)),
            str(answer_md.relative_to(REPO_ROOT)),
        ],
        "source_files": [
            str(V40_DASHBOARD.relative_to(REPO_ROOT)),
            str(V40_FAIRNESS.relative_to(REPO_ROOT)),
            str(V40_DECISION.relative_to(REPO_ROOT)),
            str(V40_FACT_LOCK.relative_to(REPO_ROOT)),
        ],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(summary["output_artifacts"], indent=2))


if __name__ == "__main__":
    main()
