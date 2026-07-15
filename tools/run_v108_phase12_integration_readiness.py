#!/usr/bin/env python3
"""Phase12 full-online integration readiness ledger for v108.

This tool does not claim to run the full v108 system. It audits whether the
modules and scene artifacts required by Phase12 are actually available, and
writes a failure ledger for the missing full-online integration runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in fields})


def read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def latest_existing(patterns: list[str]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(ROOT.glob(pattern))
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.as_posix())[-1]


def artifact(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False, "path": "", "sha256": ""}
    return {"exists": True, "path": rel(path), "sha256": sha256_file(path)}


def component_rows() -> list[dict[str, Any]]:
    components = [
        {
            "component": "physical_gap_graph",
            "required_phase12": True,
            "code": "Stream3D/stream4d_v108/gap_hypothesis_graph.py",
            "latest_artifact": latest_existing([
                "Stream3D/outputs/audit/v108_phase4_gap_multiseed_sam2_*/phase4_gap_multiseed_sam2_shadow_summary.json"
            ]),
            "status": "shadow_candidate_generation_only",
            "blocking_gap": "not integrated into a full online runner or final mask selector",
        },
        {
            "component": "output_memory_dual_plane",
            "required_phase12": True,
            "code": "Stream3D/stream4d_v108/lifecycle.py",
            "latest_artifact": latest_existing([
                "Stream3D/outputs/audit/v108_phase7_visual_admission_audit_*/phase7_visual_admission_audit_summary.json",
                "Stream3D/outputs/audit/v108_phase6_probation_watcher_*/phase6_probation_watcher_shadow_summary.json",
            ]),
            "status": "policy_shadow_only",
            "blocking_gap": "no full-scene output-plane compositor connected to rolling labels",
        },
        {
            "component": "appearance_capsule",
            "required_phase12": True,
            "code": "Stream3D/stream4d_v108/appearance_capsule.py",
            "latest_artifact": latest_existing([
                "Stream3D/outputs/audit/v108_phase3_appearance_*/phase3_appearance_summary.json"
            ]),
            "status": "controlled_benchmark_only",
            "blocking_gap": "not wired into online full-scene selector",
        },
        {
            "component": "watcher_probation",
            "required_phase12": True,
            "code": "Stream3D/stream4d_v108/masklet_watcher.py",
            "latest_artifact": latest_existing([
                "Stream3D/outputs/audit/v108_phase6_probation_watcher_*/phase6_probation_watcher_shadow_summary.json"
            ]),
            "status": "shadow_tracks_only",
            "blocking_gap": "not running inside full-scene stream",
        },
        {
            "component": "transactions",
            "required_phase12": True,
            "code": "Stream3D/stream4d_v108/transaction_manager.py",
            "latest_artifact": latest_existing([
                "Stream3D/outputs/audit/v108_phase8_sparse_transaction_shadow_*/phase8_sparse_transaction_shadow_summary.json"
            ]),
            "status": "scheduler_shadow_only",
            "blocking_gap": "no real SAM2 memory mutation from user-accepted durable rows",
        },
        {
            "component": "growth_repair_demotion",
            "required_phase12": True,
            "code": "Stream3D/stream4d_v108/growth_repair.py",
            "latest_artifact": latest_existing([
                "Stream3D/outputs/audit/v108_phase9_growth_repair_*/phase9_growth_repair_shadow_summary.json"
            ]),
            "status": "diagnostic_shadow_only",
            "blocking_gap": "candidate_B repair and demotion transaction are not integrated",
        },
        {
            "component": "2d_reactivation",
            "required_phase12": True,
            "code": "tools/run_v108_phase10_2d_reactivation_shadow.py",
            "latest_artifact": latest_existing([
                "Stream3D/outputs/audit/v108_phase10_2d_reactivation_*/phase10_2d_reactivation_shadow_summary.json"
            ]),
            "status": "controlled_shadow_only",
            "blocking_gap": "not connected to online trigger or selector",
        },
        {
            "component": "geometry_reactivation",
            "required_phase12": True,
            "code": "tools/run_v108_phase11_lingbot_geometry_reactivation_shadow.py",
            "latest_artifact": latest_existing([
                "Stream3D/outputs/audit/v108_phase11_lingbot_geometry_reactivation_*/phase11_lingbot_geometry_reactivation_summary.json"
            ]),
            "status": "controlled_shadow_only",
            "blocking_gap": "not connected to online trigger or selector",
        },
        {
            "component": "full_online_v108_runner",
            "required_phase12": True,
            "code": "tools/run_v108_phase12_full_online.py",
            "latest_artifact": None,
            "status": "missing",
            "blocking_gap": "the required orchestrator does not exist yet",
        },
    ]
    rows: list[dict[str, Any]] = []
    for row in components:
        code_path = ROOT / str(row["code"])
        art = artifact(row["latest_artifact"])
        rows.append(
            {
                "component": row["component"],
                "required_phase12": bool(row["required_phase12"]),
                "code_path": row["code"],
                "code_exists": code_path.exists(),
                "code_sha256": sha256_file(code_path) if code_path.exists() else "",
                "latest_artifact_exists": art["exists"],
                "latest_artifact_path": art["path"],
                "latest_artifact_sha256": art["sha256"],
                "status": row["status"],
                "blocking_gap": row["blocking_gap"],
                "metrics_are_diagnostic_only": True,
            }
        )
    return rows


def scene_rows() -> list[dict[str, Any]]:
    required = [
        {
            "case_name": "scene0050_full99",
            "scene_id": "scene0050_00",
            "required_by_phase12": True,
            "summary_patterns": [
                "Stream3D/outputs/audit/v107_phase*_scene0050_*full99*/g3_scheduler_summary.json",
                "Stream3D/outputs/audit/v107_phase*_scene0050_*full99*/v107_phase8_g3_rolling_scheduler_smoke/summary.json",
            ],
        },
        {
            "case_name": "scene0050_full90",
            "scene_id": "scene0050_00",
            "required_by_phase12": True,
            "summary_patterns": [
                "Stream3D/outputs/audit/v108_phase1_candidate_scene0050_90f_*/v106_stateful_sam2_rolling_scene_stream/summary.json",
                "Stream3D/outputs/audit/v108_phase1_reference_scene0050_90f_*/v106_stateful_sam2_rolling_scene_stream/summary.json",
            ],
        },
        {
            "case_name": "scene0011_full90",
            "scene_id": "scene0011_00",
            "required_by_phase12": True,
            "summary_patterns": [
                "Stream3D/outputs/audit/v107_phase10_scene0011_*full90*/**/summary.json",
                "Stream3D/outputs/audit/v108_phase1_candidate_scene0011_90f_*/v106_stateful_sam2_rolling_scene_stream/summary.json",
            ],
        },
        {
            "case_name": "scene0030_full90",
            "scene_id": "scene0030_00",
            "required_by_phase12": True,
            "summary_patterns": [
                "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_*90*/v107_phase8_g3_rolling_scheduler_smoke/summary.json",
                "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_g3_scheduler90_*/g3_scheduler_summary.json",
            ],
        },
    ]
    rows: list[dict[str, Any]] = []
    for item in required:
        summary = latest_existing(item["summary_patterns"])
        summary_json = read_json_or_none(summary) if summary is not None else None
        rows.append(
            {
                "case_name": item["case_name"],
                "scene_id": item["scene_id"],
                "required_by_phase12": bool(item["required_by_phase12"]),
                "latest_summary_exists": summary is not None,
                "latest_summary_path": rel(summary) if summary is not None else "",
                "latest_summary_sha256": sha256_file(summary) if summary is not None else "",
                "schema_version": (summary_json or {}).get("schema_version", ""),
                "frame_count": len((summary_json or {}).get("records", [])),
                "status": "existing_baseline_or_v107_artifact" if summary is not None else "missing_required_scene_artifact",
                "phase12_full_online_output_exists": False,
                "visual_review_required": True,
                "metrics_are_diagnostic_only": True,
            }
        )
    return rows


def write_markdown(path: Path, component_rows_: list[dict[str, Any]], scene_rows_: list[dict[str, Any]]) -> None:
    lines = [
        "# v108 Phase12 Integration Readiness",
        "",
        "This is a readiness ledger, not a full online v108 run.",
        "Metrics and counts are diagnostic only. Full Phase12 quality still requires high-resolution visual review.",
        "",
        "## Component Status",
        "",
        "| Component | Status | Blocking Gap | Latest Artifact |",
        "|---|---|---|---|",
    ]
    for row in component_rows_:
        lines.append(
            f"| {row['component']} | {row['status']} | {row['blocking_gap']} | {row['latest_artifact_path']} |"
        )
    lines.extend(["", "## Required Scenes", "", "| Case | Status | Latest Summary | Frame Count |", "|---|---|---|---|"])
    for row in scene_rows_:
        lines.append(
            f"| {row['case_name']} | {row['status']} | {row['latest_summary_path']} | {row['frame_count']} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "NO_GO_FULL_ONLINE_RUNNER_MISSING.",
            "The next repair is to implement `tools/run_v108_phase12_full_online.py` as a real orchestrator, not a shadow summary merger.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    comps = component_rows()
    scenes = scene_rows()
    component_csv = output_root / "phase12_component_readiness_rows.csv"
    scene_csv = output_root / "phase12_required_scene_rows.csv"
    markdown_path = output_root / "phase12_integration_readiness.md"
    write_csv(component_csv, comps)
    write_csv(scene_csv, scenes)
    write_markdown(markdown_path, comps, scenes)
    missing_components = [row["component"] for row in comps if row["status"] == "missing"]
    shadow_only = [row["component"] for row in comps if "shadow" in str(row["status"])]
    missing_scenes = [row["case_name"] for row in scenes if not row["latest_summary_exists"]]
    summary = {
        "schema_version": "stream4d_v108_phase12_integration_readiness_v1",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "phase12_full_online_run_executed": False,
        "status": "NO_GO_FULL_ONLINE_RUNNER_MISSING" if missing_components else "READY_FOR_FULL_ONLINE_RUNNER",
        "missing_components": missing_components,
        "shadow_only_components": shadow_only,
        "missing_required_scene_artifacts": missing_scenes,
        "component_rows_csv": rel(component_csv),
        "component_rows_csv_sha256": sha256_file(component_csv),
        "scene_rows_csv": rel(scene_csv),
        "scene_rows_csv_sha256": sha256_file(scene_csv),
        "readiness_markdown": rel(markdown_path),
        "readiness_markdown_sha256": sha256_file(markdown_path),
        "acceptance_rule": "Metrics are diagnostic only; this readiness ledger is not a quality gate or final result.",
        "next_repair": "Implement a real Phase12 full-online orchestrator; do not treat this shadow ledger as full online v108.",
    }
    summary_path = output_root / "phase12_integration_readiness_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), "status": summary["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
