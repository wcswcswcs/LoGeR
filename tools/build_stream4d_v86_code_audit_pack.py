#!/usr/bin/env python3
"""Build a compact Stream4D v86 code/artifact audit packet."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = REPO_ROOT / "code_audit_pack"
TAG_PREFIX = "stream4d_v86_persistent_affinity_field_readout_core_audit"

DECLARED_PATHS = [
    "docs/stream4d_v86_persistent_affinity_field_readout_experiment_plan.md",
    "docs/stream4d_v86_执行日志.md",
    "docs/stream4d_v86_实验结果复盘.md",
    "Stream3D/tools/run_v86_persistent_affinity_field_readout.py",
    "Stream3D/stream4d_native/v75_soft_incidence.py",
    "Stream3D/tools/run_v80_cmap_af_l2h_pipeline.py",
    "Stream3D/tools/run_v81_history_anchored_cmap_af_l2h_pipeline.py",
    "Stream3D/tools/run_v82_revised_causal_tracklet_memory.py",
    "tools/build_stream4d_v86_code_audit_pack.py",
    "Stream3D/outputs/audit/v86_phase0_fact_lock/fact_lock_summary.json",
    "Stream3D/outputs/audit/v86_phase0_fact_lock/route_decision_rows.csv",
    "Stream3D/outputs/audit/v86_phase0_fact_lock/forbidden_path_rows.csv",
    "Stream3D/outputs/audit/v86_phase0_fact_lock/input_artifact_scope_rows.csv",
    "Stream3D/outputs/audit/v86_phase1_native_eval_contract/native_eval_contract.json",
    "Stream3D/outputs/audit/v86_phase1_native_eval_contract/native_eval_contract_rows.csv",
    "Stream3D/outputs/audit/v86_phase1_native_eval_contract/native_gt_label_audit_rows.csv",
    "Stream3D/outputs/audit/v86_phase1_native_eval_contract/control_suite_rows.csv",
    "Stream3D/outputs/audit/v86_phase5_native_membership/native_membership_summary.json",
    "Stream3D/outputs/audit/v86_phase5_native_membership/native_readout_variant_rows.csv",
    "Stream3D/outputs/audit/v86_phase5_native_membership/native_membership_rows.csv",
    "Stream3D/outputs/audit/v86_phase5_native_membership/native_object_rows.csv",
    "Stream3D/outputs/audit/v86_phase5_native_membership/native_conflict_rows.csv",
    "Stream3D/outputs/audit/v86_phase6_native_eval/native_eval_summary.json",
    "Stream3D/outputs/audit/v86_phase6_native_eval/native_metric_rows.csv",
    "Stream3D/outputs/audit/v86_phase6_native_eval/native_control_rows.csv",
    "Stream3D/outputs/audit/v86_phase6_native_eval/native_ap_curve_rows.csv",
    "Stream3D/outputs/audit/v86_phase6_native_eval/native_case_rows.csv",
    "Stream3D/outputs/audit/v86_config/frozen_method_config.json",
    "Stream3D/outputs/audit/v86_phase8_scene_exporter_audit/scene_exporter_audit_summary.json",
    "Stream3D/outputs/audit/v86_phase8_scene_exporter_audit/scene_exporter_route_rows.csv",
    "Stream3D/outputs/audit/v86_phase9_controls/control_summary.json",
    "Stream3D/outputs/audit/v86_phase9_controls/method_variant_rows.csv",
    "Stream3D/outputs/audit/v86_phase9_controls/decision_matrix_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/holdout_summary.json",
    "Stream3D/outputs/audit/v86_phase10_holdout/holdout_input_split_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/holdout_source_candidate_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/holdout_source_candidate_summary.json",
    "Stream3D/outputs/audit/v86_phase10_holdout/diagnostic_tentative_holdout_summary.json",
    "Stream3D/outputs/audit/v86_phase10_holdout/diagnostic_tentative_holdout_frame_mask_candidate_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/diagnostic_tentative_holdout_frame_mask_selected_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/diagnostic_tentative_holdout_native_assignment_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/diagnostic_tentative_holdout_native_metric_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/diagnostic_tentative_holdout_control_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/diagnostic_tentative_holdout_source_audit_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/holdout_repair_probe_summary.json",
    "Stream3D/outputs/audit/v86_phase10_holdout/holdout_repair_probe_variant_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/holdout_repair_probe_native_metric_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/holdout_repair_probe_control_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/holdout_repair_probe_source_audit_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/holdout_repair_probe_frame_mask_selected_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/holdout_native_metric_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/holdout_control_rows.csv",
    "Stream3D/outputs/audit/v86_phase10_holdout/holdout_failure_case_rows.csv",
    "Stream3D/outputs/audit/v86_phase11_casebook/final_decision.json",
    "Stream3D/outputs/audit/v86_phase11_casebook/failure_case_rows.csv",
    "Stream3D/outputs/audit/v86_phase11_casebook/success_case_rows.csv",
    "Stream3D/outputs/audit/v86_phase11_casebook/theory_update.md",
    "Stream3D/outputs/audit/v86_phase12_dev_tracklet_readout_repair/dev_tracklet_readout_summary.json",
    "Stream3D/outputs/audit/v86_phase12_dev_tracklet_readout_repair/dev_tracklet_readout_candidate_config.json",
    "Stream3D/outputs/audit/v86_phase12_dev_tracklet_readout_repair/dev_tracklet_readout_source_rows.csv",
    "Stream3D/outputs/audit/v86_phase12_dev_tracklet_readout_repair/dev_tracklet_readout_frame_mask_candidate_rows.csv",
    "Stream3D/outputs/audit/v86_phase12_dev_tracklet_readout_repair/dev_tracklet_readout_frame_mask_selected_rows.csv",
    "Stream3D/outputs/audit/v86_phase12_dev_tracklet_readout_repair/dev_tracklet_readout_variant_rows.csv",
    "Stream3D/outputs/audit/v86_phase12_dev_tracklet_readout_repair/dev_tracklet_readout_native_metric_rows.csv",
    "Stream3D/outputs/audit/v86_phase12_dev_tracklet_readout_repair/dev_tracklet_readout_control_rows.csv",
    "Stream3D/outputs/audit/v86_phase12_dev_tracklet_readout_repair/dev_tracklet_readout_source_audit_rows.csv",
    "Stream3D/outputs/audit/v86_phase13_candidate_freeze_holdout_audit/candidate_freeze_holdout_summary.json",
    "Stream3D/outputs/audit/v86_phase13_candidate_freeze_holdout_audit/candidate_freeze_holdout_candidate_config_snapshot.json",
    "Stream3D/outputs/audit/v86_phase13_candidate_freeze_holdout_audit/candidate_freeze_holdout_source_rows.csv",
    "Stream3D/outputs/audit/v86_phase13_candidate_freeze_holdout_audit/candidate_freeze_holdout_frame_mask_candidate_rows.csv",
    "Stream3D/outputs/audit/v86_phase13_candidate_freeze_holdout_audit/candidate_freeze_holdout_frame_mask_selected_rows.csv",
    "Stream3D/outputs/audit/v86_phase13_candidate_freeze_holdout_audit/candidate_freeze_holdout_variant_rows.csv",
    "Stream3D/outputs/audit/v86_phase13_candidate_freeze_holdout_audit/candidate_freeze_holdout_native_metric_rows.csv",
    "Stream3D/outputs/audit/v86_phase13_candidate_freeze_holdout_audit/candidate_freeze_holdout_control_rows.csv",
    "Stream3D/outputs/audit/v86_phase13_candidate_freeze_holdout_audit/candidate_freeze_holdout_source_audit_rows.csv",
    "Stream3D/outputs/audit/v86_phase14_fresh_holdout_availability_audit/fresh_holdout_availability_summary.json",
    "Stream3D/outputs/audit/v86_phase14_fresh_holdout_availability_audit/fresh_holdout_chunk_coverage_rows.csv",
    "Stream3D/outputs/audit/v86_phase14_fresh_holdout_availability_audit/fresh_holdout_artifact_audit_rows.csv",
    "Stream3D/outputs/audit/v86_phase14_fresh_holdout_availability_audit/fresh_holdout_candidate_rows.csv",
    "Stream3D/outputs/audit/v86_phase15_raw_substrate_availability_audit/raw_substrate_availability_summary.json",
    "Stream3D/outputs/audit/v86_phase15_raw_substrate_availability_audit/raw_substrate_source_artifact_rows.csv",
    "Stream3D/outputs/audit/v86_phase15_raw_substrate_availability_audit/raw_substrate_scene_chunk_rows.csv",
    "Stream3D/outputs/audit/v86_phase15_raw_substrate_availability_audit/formal_ready_uninspected_chain_rows.csv",
    "Stream3D/outputs/audit/v86_phase16_new_scene_pipeline_feasibility_audit/new_scene_pipeline_feasibility_summary.json",
    "Stream3D/outputs/audit/v86_phase16_new_scene_pipeline_feasibility_audit/new_scene_input_coverage_rows.csv",
    "Stream3D/outputs/audit/v86_phase16_new_scene_pipeline_feasibility_audit/new_scene_entrypoint_rows.csv",
    "Stream3D/outputs/audit/v86_phase16_new_scene_pipeline_feasibility_audit/new_scene_required_chain_rows.csv",
    "Stream3D/outputs/audit/v86_phase16_v75_soft_incidence_new_scene_smoke/incidence_summary.json",
    "Stream3D/outputs/audit/v86_phase16_v75_soft_incidence_new_scene_smoke/summary.json",
    "Stream3D/outputs/audit/v86_phase16_v75_soft_incidence_new_scene_smoke/missing_input_rows.csv",
    "Stream3D/outputs/audit/v86_phase17_new_scene_chain_repair_audit/new_scene_chain_repair_summary.json",
    "Stream3D/outputs/audit/v86_phase17_new_scene_chain_repair_audit/new_scene_chain_attempt_rows.csv",
    "Stream3D/outputs/audit/v86_phase17_v75_soft_incidence_new_scene_smoke/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v75_soft_incidence_new_scene_smoke/incidence_summary.json",
    "Stream3D/outputs/audit/v86_phase17_v75_soft_incidence_new_scene_smoke/source_rows.csv",
    "Stream3D/outputs/audit/v86_phase17_v75_soft_incidence_new_scene_max5/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v75_soft_incidence_new_scene_max5/incidence_summary.json",
    "Stream3D/outputs/audit/v86_phase17_v75_soft_incidence_new_scene_max5/incidence_variant_summary_rows.csv",
    "Stream3D/outputs/audit/v86_phase17_v75_soft_incidence_new_scene_max5/source_rows.csv",
    "Stream3D/outputs/audit/v86_phase17_v75_soft_incidence_scene0030_max10/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v75_soft_incidence_scene0030_max10/incidence_summary.json",
    "Stream3D/outputs/audit/v86_phase17_v75_soft_incidence_scene0030_max10/incidence_variant_summary_rows.csv",
    "Stream3D/outputs/audit/v86_phase17_v82_new_scene_phase1_proxyhash_smoke/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_new_scene_phase1_all_c0/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0081_phase1_c0/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0591_phase1_c0/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase1_c0_4/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase2_c0_4_proxyappearance_additive_app080/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase3_c0_4_proxyappearance_app080/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase4_c0_4_proxyappearance_app080_qproxy_entropy080/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase4_c0_4_proxyappearance_app080_qproxy_entropy080/q_control_rows.csv",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase4_c0_4_proxyappearance_app080_qproxy_entropy080/q_margin_rows.csv",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase5_c0_4_proxyappearance_app080_qproxy_entropy080_diag/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase5_c0_4_proxyappearance_app080_qproxy_entropy080_diag_candidates/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase5_c0_4_phase4q_method/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase5_c0_4_phase4q_method/local_slot_history_assignment_rows.csv",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase6_c0_4_phase4q_method_passthrough/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase7_c0_4_phase4q_method_passthrough/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase1_c0_3/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase1_c0_2/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase2_c0_2_proxyappearance_additive_app080/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase3_c0_2_proxyappearance_app080/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase1_c0_5/summary.json",
    "Stream3D/outputs/audit/v86_phase17_v82_scene0030_phase1_c0_9/summary.json",
    "Stream3D/outputs/audit/v85_phase7_renderable_materializer/native_carrier_diagnostic_summary.json",
    "Stream3D/outputs/audit/v85_phase7_renderable_materializer/native_carrier_evaluator_candidate_contract.json",
    "Stream3D/outputs/audit/v85_phase7_renderable_materializer/native_scene_vertex_export_route_summary.json",
    "Stream3D/outputs/audit/v85_phase6_history_query/q_summary.json",
    "Stream3D/outputs/audit/v85_phase10_casebook/final_decision.json",
    "Stream3D/outputs/audit/v84_holdout_replay_v82_phase5_weak_history/summary.json",
    "Stream3D/outputs/audit/v84_holdout_replay_v82_phase5_weak_history/local_slot_history_assignment_rows.csv",
    "Stream3D/outputs/audit/v84_holdout_replay_v82_phase6_strong_history/summary.json",
    "Stream3D/outputs/audit/v84_holdout_replay_v82_phase6_strong_history/adapter_rows.csv",
    "Stream3D/outputs/audit/v84_holdout_replay_v82_phase6_strong_history/cluster_rows.csv",
    "Stream3D/outputs/audit/v84_holdout_replay_v82_phase7_final_local/summary.json",
]

DIFF_PATHS = [
    "docs/stream4d_v86_执行日志.md",
    "docs/stream4d_v86_实验结果复盘.md",
    "Stream3D/tools/run_v86_persistent_affinity_field_readout.py",
    "Stream3D/stream4d_native/v75_soft_incidence.py",
    "Stream3D/tools/run_v80_cmap_af_l2h_pipeline.py",
    "Stream3D/tools/run_v81_history_anchored_cmap_af_l2h_pipeline.py",
    "Stream3D/tools/run_v82_revised_causal_tracklet_memory.py",
    "tools/build_stream4d_v86_code_audit_pack.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_text(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as exc:
        return f"$ {' '.join(cmd)}\nCOMMAND_NOT_FOUND: {exc}\n"
    return f"$ {' '.join(cmd)}\nreturncode={proc.returncode}\n\n{proc.stdout}"


def packet_files(packet_dir: Path) -> list[Path]:
    return sorted(path for path in packet_dir.rglob("*") if path.is_file())


def copy_declared(packet_dir: Path) -> tuple[list[str], list[str]]:
    copied: list[str] = []
    missing: list[str] = []
    for rel in DECLARED_PATHS:
        src = REPO_ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = packet_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    return copied, missing


def write_scope(packet_dir: Path, copied: list[str], missing: list[str]) -> None:
    lines = [
        "# Stream4D v86 Audit Scope",
        "",
        f"generated_at_utc: {datetime.utcnow().isoformat(timespec='seconds')}Z",
        "",
        "Purpose: package the v86 persistent affinity field readout runner, dual logs, final decision, and key evidence tables.",
        "",
        "Included:",
        "- v86 plan, execution log, retrospective log",
        "- v86 runner, v80/v81/v82 source dependencies used by Phase16 entrypoint audit, and pack builder",
        "- v86 frozen config plus Phase0/1/5/6/8/9/10/11 summaries and compact CSV evidence",
        "- Phase10 holdout input split audit showing whether registered holdout support exists",
        "- Phase10 holdout source-candidate audit separating raw D4RT observations from legal selected method readout",
        "- Phase10 diagnostic-only tentative holdout replay evidence, explicitly forbidden for method claim",
        "- Phase10 holdout repair-probe evidence for high-margin weak filters and Phase2 confirmed/repeated tracklet readouts",
        "- Phase12 dev-side tracklet readout repair evidence and candidate frozen-config handoff, explicitly not a formal holdout claim",
        "- Phase13 candidate-freeze holdout audit applying the dev-selected DV5 rule, explicitly marked as not a formal method claim",
        "- Phase14 fresh-holdout availability audit showing whether any uninspected artifact chunks remain for a formal v86 rerun",
        "- Phase15 raw-substrate availability audit showing unconsumed raw scenes exist but lack the formal-ready v86 readout chain",
        "- Phase16 new-scene pipeline feasibility audit plus v75 soft-incidence smoke showing the missing input-chain repairs",
        "- Phase17 new-scene chain repair audit: v75 root registration, v80/v82 chunk/input forwarding, proxy descriptor/Q repairs, and compact v82 Phase1-5 evidence",
        "- compact v84 holdout replay evidence showing diagnostic-only tentative rows and missing strong materializer inputs",
        "- selected v85 source summaries needed to audit provenance",
        "",
        "Excluded:",
        "- raw datasets, checkpoints, media, old audit packs",
        "- v85 40MB native support table and other predecessor row-level tables not needed for compact review",
        "",
        f"copied_file_count: {len(copied)}",
        f"missing_declared_path_count: {len(missing)}",
        "",
    ]
    write_text(packet_dir / "AUDIT_SCOPE.md", "\n".join(lines))
    write_text(packet_dir / "MISSING_DECLARED_PATHS.txt", "\n".join(missing) + ("\n" if missing else ""))


def write_context(packet_dir: Path) -> None:
    write_text(packet_dir / "GIT_STATUS_SHORT.txt", run_text(["git", "status", "--short"]))
    existing = [path for path in DIFF_PATHS if (REPO_ROOT / path).exists()]
    write_text(packet_dir / "SCOPED_GIT_DIFF.patch", run_text(["git", "diff", "--", *existing]) if existing else "")


def write_filelists(packet_dir: Path) -> dict[str, int]:
    hash_lines: list[str] = []
    for path in packet_files(packet_dir):
        rel = path.relative_to(packet_dir).as_posix()
        if rel in {"PAYLOAD_FILELIST.txt", "PAYLOAD_SHA256SUMS.txt"}:
            continue
        hash_lines.append(f"{sha256_file(path)}  {rel}")
    write_text(packet_dir / "PAYLOAD_SHA256SUMS.txt", "\n".join(hash_lines) + "\n")
    files = [path.relative_to(packet_dir).as_posix() for path in packet_files(packet_dir)]
    write_text(packet_dir / "PAYLOAD_FILELIST.txt", "\n".join(files) + "\n")
    return {"payload_file_count": len(files), "payload_hash_rows": len(hash_lines)}


def zip_packet(packet_dir: Path, zip_path: Path) -> list[str]:
    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in packet_files(packet_dir):
            rel = path.relative_to(packet_dir).as_posix()
            arc = f"{packet_dir.name}/{rel}"
            zf.write(path, arc)
            entries.append(arc)
    return entries


def validate(packet_dir: Path, zip_path: Path, entries: list[str]) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as zf:
        bad_entry = zf.testzip()
        actual_entries = sorted(zf.namelist())
    expected_entries = sorted(entries)
    diff = sorted(set(expected_entries) ^ set(actual_entries))
    write_text(zip_path.with_suffix(".zip_entries.txt"), "\n".join(actual_entries) + "\n")
    write_text(packet_dir.with_suffix(".entry_diff.txt"), "\n".join(diff) + ("\n" if diff else ""))
    sha_path = zip_path.with_suffix(".zip.sha256")
    write_text(sha_path, f"{sha256_file(zip_path)}  {zip_path.name}\n")
    unzip_summary = "OK\n" if bad_entry is None else f"BAD_ENTRY {bad_entry}\n"
    write_text(packet_dir.with_suffix(".unzip_test.txt"), unzip_summary)
    return {
        "zip_path": zip_path.relative_to(REPO_ROOT).as_posix(),
        "zip_sha256": sha256_file(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_entry_count": len(actual_entries),
        "unzip_test_ok": bad_entry is None,
        "entry_diff_count": len(diff),
    }


def main() -> None:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    tag = f"{TAG_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    packet_dir = PACK_ROOT / tag
    if packet_dir.exists():
        raise SystemExit(f"refusing to overwrite existing packet: {packet_dir}")
    packet_dir.mkdir(parents=True)

    copied, missing = copy_declared(packet_dir)
    write_scope(packet_dir, copied, missing)
    write_context(packet_dir)
    counts = write_filelists(packet_dir)
    zip_path = PACK_ROOT / f"{tag}.zip"
    entries = zip_packet(packet_dir, zip_path)
    validation = validate(packet_dir, zip_path, entries)
    summary: dict[str, Any] = {
        "schema": "stream4d_v86_code_audit_pack_summary_v1",
        "tag": tag,
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "copied_file_count": len(copied),
        "missing_declared_paths": missing,
        **counts,
        **validation,
    }
    write_text(packet_dir.with_suffix(".build_summary.json"), json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
