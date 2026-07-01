#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DOC_FILES = [
    "docs/stream4d_v71_gsrm_geo_semantic_representative_mask_selection_plan.md",
    "docs/stream4d_v71_执行日志.md",
    "docs/stream4d_v71_实验结果复盘.md",
]

CODE_FILES = [
    "tools/build_v71_code_audit_pack.py",
    "Stream3D/stream4d/__init__.py",
    "Stream3D/stream4d/scannet_stream.py",
    "Stream3D/stream4d_native/__init__.py",
    "Stream3D/stream4d_native/frozen_feature_adapter.py",
    "Stream3D/stream4d_native/v67_local_baselines.py",
    "Stream3D/stream4d_native/v67_mask_universe.py",
    "Stream3D/stream4d_native/v71_fact_lock.py",
    "Stream3D/stream4d_native/v71_candidate_bank.py",
    "Stream3D/stream4d_native/v71_d4rt_atoms.py",
    "Stream3D/stream4d_native/v71_semantic_features.py",
    "Stream3D/stream4d_native/v71_key_atoms.py",
    "Stream3D/stream4d_native/v71_representative_setcover.py",
    "Stream3D/tools/run_v65_scene_multiview_ap.py",
    "Stream3D/tools/run_v66_local_chunk_eval.py",
    "Stream3D/tools/run_v66_scene_mv_ap_probe5.py",
    "Stream3D/tools/run_v71_fact_lock.py",
    "Stream3D/tools/build_v71_candidate_bank.py",
    "Stream3D/tools/build_v71_d4rt_atoms.py",
    "Stream3D/tools/extract_v71_semantic_features.py",
    "Stream3D/tools/run_v71_key_atoms.py",
    "Stream3D/tools/run_v71_representative_setcover.py",
    "Stream3D/tools/diagnose_v71_oracle_budget_sweep.py",
    "Stream3D/tools/diagnose_v71_group_representative_repair.py",
    "Stream3D/tools/diagnose_v71_adjacent_mask_track_repair.py",
    "Stream3D/tools/diagnose_v71_d4rt_carrier_track_repair.py",
    "Stream3D/tools/diagnose_v71_objectness_proxy_separability.py",
]

ARTIFACT_DIRS = [
    "Stream3D/outputs/audit/v71_phase0_fact_lock",
    "Stream3D/outputs/audit/v71_candidate_bank",
    "Stream3D/outputs/audit/v71_d4rt_atoms",
    "Stream3D/outputs/audit/v71_d4rt_atoms_smoke_scene0011",
    "Stream3D/outputs/audit/v71_d4rt_atoms_smoke_scene0011_min8",
    "Stream3D/outputs/audit/v71_d4rt_atoms_smoke_scene0011_reliability_fix",
    "Stream3D/outputs/audit/v71_d4rt_atoms_smoke_scene0011_schemafix",
    "Stream3D/outputs/audit/v71_semantic_features",
    "Stream3D/outputs/audit/v71_semantic_features_part_gpu6",
    "Stream3D/outputs/audit/v71_semantic_features_part_gpu7",
    "Stream3D/outputs/audit/v71_semantic_features_smoke512",
    "Stream3D/outputs/audit/v71_semantic_features_smoke512_cov",
    "Stream3D/outputs/audit/v71_key_atoms",
    "Stream3D/outputs/audit/v71_key_atoms_schemafix_smoke",
    "Stream3D/outputs/audit/v71_key_atoms_smoke",
    "Stream3D/outputs/audit/v71_key_atoms_smoke2",
    "Stream3D/outputs/audit/v71_representative_setcover",
    "Stream3D/outputs/audit/v71_representative_setcover_debug3",
    "Stream3D/outputs/audit/v71_representative_setcover_debug3_areatarget",
    "Stream3D/outputs/audit/v71_representative_setcover_debug3_cleanmid",
    "Stream3D/outputs/audit/v71_representative_setcover_debug3_max128",
    "Stream3D/outputs/audit/v71_representative_setcover_debug3_noprefilter",
    "Stream3D/outputs/audit/v71_representative_setcover_debug3_objectnessrank",
    "Stream3D/outputs/audit/v71_representative_setcover_debug3_oraclefix",
    "Stream3D/outputs/audit/v71_representative_setcover_debug3_sourceflag",
    "Stream3D/outputs/audit/v71_representative_setcover_oracle65_allmask",
    "Stream3D/outputs/audit/v71_representative_setcover_oracle128_allmask",
    "Stream3D/outputs/audit/v71_representative_setcover_oracle_candidate_budget",
    "Stream3D/outputs/audit/v71_representative_setcover_oracle_budget_sweep12",
    "Stream3D/outputs/audit/v71_representative_setcover_oracle_budget_sweep12_script",
    "Stream3D/outputs/audit/v71_representative_setcover_smoke_scene0011",
    "Stream3D/outputs/audit/v71_group_representative_repair12",
    "Stream3D/outputs/audit/v71_group_representative_repair12_risky",
    "Stream3D/outputs/audit/v71_group_representative_repair12_riskfields",
    "Stream3D/outputs/audit/v71_adjacent_mask_track_repair12",
    "Stream3D/outputs/audit/v71_representative_setcover_oracle_budget_sweep12_highbudget",
    "Stream3D/outputs/audit/v71_representative_setcover_debug12_highbudget192",
    "Stream3D/outputs/audit/v71_d4rt_carrier_track_repair12",
    "Stream3D/outputs/audit/v71_objectness_proxy_separability12",
    "Stream3D/outputs/audit/v71_visualizations",
]

EXCLUDED_SEGMENTS = {"__pycache__", ".git", "code_audit_pack", "data", "ckpts", "wandb"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".zip"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_rel(rel_path: str) -> dict[str, Any]:
    path = REPO_ROOT / rel_path
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_excluded(rel_path: Path) -> bool:
    if any(part in EXCLUDED_SEGMENTS for part in rel_path.parts):
        return True
    return rel_path.suffix.lower() in EXCLUDED_SUFFIXES


def add_existing_file(rel: str, files: set[Path], missing: list[str]) -> None:
    path = REPO_ROOT / rel
    rel_path = Path(rel)
    if not path.exists():
        missing.append(rel)
        return
    if path.is_file() and not is_excluded(rel_path):
        files.add(rel_path)


def collect_files() -> tuple[list[Path], list[str], list[str]]:
    files: set[Path] = set()
    missing: list[str] = []
    skipped_empty_dirs: list[str] = []

    for rel in DOC_FILES + CODE_FILES:
        add_existing_file(rel, files, missing)

    for rel_dir in ARTIFACT_DIRS:
        root = REPO_ROOT / rel_dir
        if not root.exists():
            missing.append(rel_dir)
            continue
        count_before = len(files)
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel_path = path.relative_to(REPO_ROOT)
            if is_excluded(rel_path):
                continue
            files.add(rel_path)
        if len(files) == count_before:
            skipped_empty_dirs.append(rel_dir)

    return sorted(files, key=lambda p: p.as_posix()), missing, skipped_empty_dirs


def copy_payload(files: list[Path], stage_dir: Path) -> None:
    for rel_path in files:
        src = REPO_ROOT / rel_path
        dst = stage_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_summary(tag: str, files: list[Path], missing: list[str], skipped_empty_dirs: list[str]) -> dict[str, Any]:
    setcover = read_json_rel("Stream3D/outputs/audit/v71_representative_setcover/setcover_summary.json")
    group = read_json_rel("Stream3D/outputs/audit/v71_group_representative_repair12_riskfields/group_repair_summary.json")
    key_atoms = read_json_rel("Stream3D/outputs/audit/v71_key_atoms/key_atom_summary.json")
    oracle_sweep = read_json_rel(
        "Stream3D/outputs/audit/v71_representative_setcover_oracle_budget_sweep12_script/oracle_budget_sweep_summary.json"
    )
    highbudget_oracle = read_json_rel(
        "Stream3D/outputs/audit/v71_representative_setcover_oracle_budget_sweep12_highbudget/oracle_budget_sweep_summary.json"
    )
    highbudget_setcover = read_json_rel(
        "Stream3D/outputs/audit/v71_representative_setcover_debug12_highbudget192/setcover_summary.json"
    )
    adjacent_track = read_json_rel("Stream3D/outputs/audit/v71_adjacent_mask_track_repair12/adjacent_track_summary.json")
    carrier_track = read_json_rel("Stream3D/outputs/audit/v71_d4rt_carrier_track_repair12/d4rt_carrier_track_summary.json")
    objectness_proxy = read_json_rel(
        "Stream3D/outputs/audit/v71_objectness_proxy_separability12/objectness_proxy_summary.json"
    )
    total_size = sum((REPO_ROOT / rel).stat().st_size for rel in files)
    return {
        "tag": tag,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "payload_file_count_source": len(files),
        "payload_source_bytes": total_size,
        "missing_requested_paths": missing,
        "skipped_empty_artifact_dirs": skipped_empty_dirs,
        "scope": {
            "docs": DOC_FILES,
            "code_files": CODE_FILES,
            "artifact_dirs": ARTIFACT_DIRS,
            "exclusion_segments": sorted(EXCLUDED_SEGMENTS),
            "exclusion_suffixes": sorted(EXCLUDED_SUFFIXES),
        },
        "current_decisions_from_artifacts": {
            "phase4_key_atoms_decision": key_atoms.get("decision"),
            "phase5_representative_setcover_decision": setcover.get("decision"),
            "phase5_processed_chunk_count": setcover.get("processed_chunk_count"),
            "phase5_best_method_variant": (setcover.get("best_method") or {}).get("variant"),
            "phase5_best_method_oracle_SF50": (setcover.get("best_method") or {}).get("representative_oracle_SF50"),
            "phase5_best_method_GT_best_IoU_mean": (setcover.get("best_method") or {}).get("representative_GT_best_IoU_mean"),
            "group_repair_decision": group.get("decision"),
            "group_repair_best_variant": group.get("best_variant"),
            "group_repair_best_variant_local_SF50": group.get("best_variant_local_SF50"),
            "oracle_budget_sweep_decision": oracle_sweep.get("decision"),
            "oracle_budget_sweep_processed_chunk_count": oracle_sweep.get("processed_chunk_count"),
            "highbudget_oracle_sweep_decision": highbudget_oracle.get("decision"),
            "highbudget_oracle_sweep_processed_chunk_count": highbudget_oracle.get("processed_chunk_count"),
            "highbudget_setcover_decision": highbudget_setcover.get("decision"),
            "highbudget_setcover_best_method_variant": (highbudget_setcover.get("best_method") or {}).get("variant"),
            "highbudget_setcover_best_method_oracle_SF50": (highbudget_setcover.get("best_method") or {}).get(
                "representative_oracle_SF50"
            ),
            "adjacent_track_decision": adjacent_track.get("decision"),
            "adjacent_track_best_variant": adjacent_track.get("best_variant"),
            "adjacent_track_best_variant_local_SF50": adjacent_track.get("best_variant_local_SF50"),
            "carrier_track_decision": carrier_track.get("decision"),
            "carrier_track_best_variant": carrier_track.get("best_variant"),
            "carrier_track_best_variant_local_SF50": carrier_track.get("best_variant_local_SF50"),
            "objectness_proxy_decision": objectness_proxy.get("decision"),
            "objectness_proxy_processed_chunk_count": objectness_proxy.get("processed_chunk_count"),
            "objectness_proxy_best_non_oracle_variant": objectness_proxy.get("best_non_oracle_variant"),
            "objectness_proxy_best_non_oracle_budget": objectness_proxy.get("best_non_oracle_budget"),
            "objectness_proxy_best_non_oracle_oracle_SF50": objectness_proxy.get(
                "best_non_oracle_representative_oracle_SF50"
            ),
            "objectness_proxy_best_non_oracle_GT_best_IoU_mean": objectness_proxy.get(
                "best_non_oracle_GT_best_IoU_mean"
            ),
            "objectness_proxy_best_non_oracle_broad_large_selected_rate": objectness_proxy.get(
                "best_non_oracle_broad_large_selected_rate"
            ),
            "objectness_proxy_best_non_oracle_underseg_proxy_selected_rate": objectness_proxy.get(
                "best_non_oracle_underseg_proxy_selected_rate"
            ),
        },
    }


def build_readme(summary: dict[str, Any]) -> str:
    decisions = summary["current_decisions_from_artifacts"]
    return "\n".join(
        [
            "# Stream4D v71 GSRM Core Audit Pack",
            "",
            "This archive is a compact reviewer packet for the current v71 frontier.",
            "It contains the v71 plan, execution log, retrospective log, core implementation files, direct helper dependencies, and selected v71 audit artifacts.",
            "",
            "## Current Artifact Decisions",
            "",
            f"- Phase4 key atoms: {decisions.get('phase4_key_atoms_decision')}",
            f"- Phase5 representative set cover: {decisions.get('phase5_representative_setcover_decision')}",
            f"- Phase5 processed chunks: {decisions.get('phase5_processed_chunk_count')}",
            f"- Phase5 best method variant: {decisions.get('phase5_best_method_variant')}",
            f"- Phase5 best method oracle SF50: {decisions.get('phase5_best_method_oracle_SF50')}",
            f"- Phase5 best method GT best IoU mean: {decisions.get('phase5_best_method_GT_best_IoU_mean')}",
            f"- Group repair decision: {decisions.get('group_repair_decision')}",
            f"- Group repair best variant: {decisions.get('group_repair_best_variant')}",
            f"- Group repair best variant local SF50: {decisions.get('group_repair_best_variant_local_SF50')}",
            f"- Oracle budget sweep decision: {decisions.get('oracle_budget_sweep_decision')}",
            f"- High-budget oracle sweep decision: {decisions.get('highbudget_oracle_sweep_decision')}",
            f"- High-budget setcover decision: {decisions.get('highbudget_setcover_decision')}",
            f"- High-budget setcover best method oracle SF50: {decisions.get('highbudget_setcover_best_method_oracle_SF50')}",
            f"- Adjacent track decision: {decisions.get('adjacent_track_decision')}",
            f"- Adjacent track best variant local SF50: {decisions.get('adjacent_track_best_variant_local_SF50')}",
            f"- D4RT carrier track decision: {decisions.get('carrier_track_decision')}",
            f"- D4RT carrier track best variant local SF50: {decisions.get('carrier_track_best_variant_local_SF50')}",
            f"- Objectness proxy decision: {decisions.get('objectness_proxy_decision')}",
            f"- Objectness proxy best non-oracle variant: {decisions.get('objectness_proxy_best_non_oracle_variant')}",
            f"- Objectness proxy best non-oracle budget: {decisions.get('objectness_proxy_best_non_oracle_budget')}",
            f"- Objectness proxy best non-oracle oracle SF50: {decisions.get('objectness_proxy_best_non_oracle_oracle_SF50')}",
            f"- Objectness proxy best non-oracle GT best IoU mean: {decisions.get('objectness_proxy_best_non_oracle_GT_best_IoU_mean')}",
            f"- Objectness proxy best non-oracle broad rate: {decisions.get('objectness_proxy_best_non_oracle_broad_large_selected_rate')}",
            f"- Objectness proxy best non-oracle underseg rate: {decisions.get('objectness_proxy_best_non_oracle_underseg_proxy_selected_rate')}",
            "",
            "## Revalidation Sidecars",
            "",
            "The directory next to the zip contains:",
            "",
            "- *_FILELIST.txt",
            "- *_PAYLOAD_SHA256SUMS.txt",
            "- *_ZIP_FILELIST.txt",
            "- *_ZIPTEST.log",
            "- *_ENTRY_DIFF.log",
            "- *_EXCLUDED_PATH_CHECK.log",
            "- *.zip.sha256",
            "- *_zip_sha256_check.log",
            "",
            "The archive intentionally excludes raw datasets, checkpoints/weights, caches, prior audit zips, and Python bytecode.",
            "",
        ]
    )


def zip_stage(stage_dir: Path, zip_path: Path, tag: str) -> list[str]:
    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(stage_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(stage_dir)
            arcname = f"{tag}/{rel.as_posix()}"
            zf.write(path, arcname)
            entries.append(arcname)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="")
    parser.add_argument("--dest", default="code_audit_pack")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tag = args.tag or f"stream4d_v71_gsrm_phase5_no_go_repair_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dest = (REPO_ROOT / args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    stage_dir = dest / tag
    zip_path = dest / f"{tag}.zip"
    if (stage_dir.exists() or zip_path.exists()) and not args.force:
        raise SystemExit(f"Refusing to overwrite existing package path for tag={tag}; pass --force if intentional")
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    if zip_path.exists():
        zip_path.unlink()
    for old in dest.glob(f"{tag}_*"):
        if old.is_file():
            old.unlink()
    old_sha = dest / f"{tag}.zip.sha256"
    if old_sha.exists():
        old_sha.unlink()

    files, missing, skipped_empty_dirs = collect_files()
    stage_dir.mkdir(parents=True, exist_ok=True)
    copy_payload(files, stage_dir)

    summary = build_summary(tag, files, missing, skipped_empty_dirs)
    source_filelist = "\n".join(rel.as_posix() for rel in files) + "\n"
    payload_hash_lines = [f"{sha256_file(REPO_ROOT / rel)}  {rel.as_posix()}" for rel in files]
    write_text(stage_dir / "SOURCE_FILELIST.txt", source_filelist)
    write_text(stage_dir / "SOURCE_SHA256SUMS.txt", "\n".join(payload_hash_lines) + "\n")
    write_text(stage_dir / "PACKAGE_SUMMARY.json", json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    write_text(stage_dir / "PACKET_README.md", build_readme(summary))

    entries = zip_stage(stage_dir, zip_path, tag)
    zip_sha = sha256_file(zip_path)
    write_text(dest / f"{tag}.zip.sha256", f"{zip_sha}  {zip_path.name}\n")
    write_text(dest / f"{tag}_FILELIST.txt", source_filelist)
    write_text(dest / f"{tag}_PAYLOAD_SHA256SUMS.txt", "\n".join(payload_hash_lines) + "\n")
    write_text(dest / f"{tag}_ZIP_FILELIST.txt", "\n".join(entries) + "\n")
    write_text(dest / f"{tag}_PACKAGE_SUMMARY.json", json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
    write_text(dest / f"{tag}_ZIPTEST.log", "OK\n" if bad is None else f"BAD_ENTRY {bad}\n")

    stage_entries = sorted(f"{tag}/{p.relative_to(stage_dir).as_posix()}" for p in stage_dir.rglob("*") if p.is_file())
    diff_lines: list[str] = []
    for entry in sorted(set(stage_entries) - set(entries)):
        diff_lines.append(f"missing_from_zip {entry}")
    for entry in sorted(set(entries) - set(stage_entries)):
        diff_lines.append(f"extra_in_zip {entry}")
    write_text(dest / f"{tag}_ENTRY_DIFF.log", "\n".join(diff_lines) + ("\n" if diff_lines else ""))

    exclusion_hits: list[str] = []
    for entry in entries:
        rel = Path(entry)
        if any(part in EXCLUDED_SEGMENTS for part in rel.parts[1:]) or rel.suffix.lower() in EXCLUDED_SUFFIXES:
            exclusion_hits.append(entry)
    write_text(dest / f"{tag}_EXCLUDED_PATH_CHECK.log", "\n".join(exclusion_hits) + ("\n" if exclusion_hits else ""))

    latest = dest / "LATEST_stream4d_v71_gsrm_phase5_no_go_repair_audit.txt"
    write_text(latest, zip_path.name + "\n")

    print(json.dumps({
        "tag": tag,
        "zip_path": str(zip_path.relative_to(REPO_ROOT)),
        "zip_sha256": zip_sha,
        "payload_file_count_source": len(files),
        "zip_entry_count": len(entries),
        "payload_source_bytes": summary["payload_source_bytes"],
        "zip_bytes": zip_path.stat().st_size,
        "missing_requested_paths": missing,
        "skipped_empty_artifact_dirs": skipped_empty_dirs,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
