#!/usr/bin/env python3
"""Build a code-focused Stream4D v99 review packet."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "code_audit_pack"
NAME_PREFIX = "stream4d_v99_core_code_audit"


CODE_FILES = [
    "tools/build_stream4d_v99_code_audit_pack.py",
    "docs/stream4d_v99_f2_anchored_da3_d4rt_reliable_primitive_affinity_plan.md",
    "docs/stream4d_v99_执行日志.md",
    "docs/stream4d_v99_实验结果复盘.md",
    "Stream3D/tools/build_v99_phase0_fact_lock.py",
    "Stream3D/tools/build_v99_phase1_f2_base_reproduction.py",
    "Stream3D/tools/check_mv_ap_contract.py",
    "Stream3D/tools/run_v65_scene_multiview_ap.py",
    "Stream3D/tools/run_v89_recalc_point_projected_mv_ap.py",
    "Stream3D/tools/run_v90_geo_semantic_witness_cover.py",
    "Stream3D/tools/run_v91_phase4_adaptive_uncertainty_materialization.py",
    "Stream3D/tools/build_v91_radio_mask_features.py",
    "Stream3D/tools/merge_v91_radio_mask_feature_stores.py",
    "Stream3D/tools/build_v95_phase1_physical_source_registry.py",
    "Stream3D/tools/build_v96_phase2_d4rt_micro_tracks.py",
    "Stream3D/tools/build_v96_phase2_segment_aggregate.py",
    "Stream3D/tools/build_v97_phase2_d4rt_micro_tracks.py",
    "Stream3D/tools/build_v97_phase2_full_aggregate.py",
    "Stream3D/tools/build_v97_phase2_overlap_stitch_micro_tracks.py",
    "Stream3D/tools/build_v97_phase2_source_preserving_query_repair.py",
    "Stream3D/tools/build_v97_phase3_triton_incidence.py",
    "Stream3D/tools/build_v97_phase4_micro_affinity_feature.py",
    "Stream3D/tools/build_v97_phase5_object_birth.py",
    "Stream3D/tools/build_v97_phase6_render_splat.py",
    "Stream3D/tools/build_v97_phase7_support_iou_readout.py",
    "Stream3D/tools/build_v97_phase9_failure_decomposition.py",
    "Stream3D/tools/prepare_v98_1_da3_full_dev_inputs.py",
    "Stream3D/tools/prepare_v98_1_da3_variant_smoke.py",
    "Stream3D/tools/build_v98_1_contract_artifacts.py",
    "Stream3D/tools/build_v98_1_canonical_mv_metrics.py",
    "Stream3D/tools/build_v98_1_canonical_score_repair.py",
    "Stream3D/tools/build_v98_1_holdout_source_registry.py",
    "Stream3D/tools/build_v98_1_phase4_phase5_geometry_method.py",
    "Stream3D/tools/build_v98_1_phase6_to_phase12_affinity_eval.py",
    "Stream3D/tools/build_v98_1_canonical_holdout_metrics.py",
    "Stream3D/tools/build_v98_1_holdout_control_frame_mask_rows.py",
    "Stream3D/tools/build_v98_1_canonical_scene_metrics.py",
    "Stream3D/tools/build_v98_1_da3_variant_geometry_quality.py",
    "Stream3D/tools/build_v98_1_da3_single_scene_geometry_quality.py",
    "Stream3D/tools/build_v98_1_da3_d4rt_geometry_comparison.py",
    "Stream3D/tools/build_v98_1_d4rt_dense_geometry_comparison.py",
    "Stream3D/tools/export_v98_1_da3_gt_geometry_visual.py",
    "Stream3D/tools/run_v98_1_dense_geometry_visualization_pipeline.py",
    "Stream3D/tools/serve_v98_1_da3_gt_sim3_viewer.py",
    "Stream3D/tools/serve_v98_1_da3_gt_dense_rgb_sim3_viewer.py",
    "Stream3D/tools/serve_v98_1_da3_variant_geometry_viewer.py",
    "Stream3D/geometry_provider/__init__.py",
    "Stream3D/geometry_provider/common.py",
    "Stream3D/stream4d/__init__.py",
    "Stream3D/stream4d/carrier_store.py",
    "Stream3D/stream4d/scannet_stream.py",
    "Stream3D/stream4d/d4rt_adapter.py",
    "Stream3D/stream4d_native/__init__.py",
    "Stream3D/stream4d_native/chunk_alignment.py",
    "Stream3D/stream4d_native/d4rt_scene_builder.py",
    "Stream3D/stream4d_native/frozen_feature_adapter.py",
    "Stream3D/stream4d_native/occupancy_dense_tracker.py",
    "Stream3D/stream4d_native/occupancy_state.py",
    "Stream3D/stream4d_native/self_stitch.py",
    "Stream3D/stream4d_native/sim3.py",
    "Stream3D/stream4d_native/v65_visualization_export.py",
    "Stream3D/stream4d_native/v91_mask_feature_store.py",
]

README = """# Stream4D v99 core code audit packet

This archive intentionally contains code plus the v99 plan/execution/retrospective
documents needed to audit the current Phase0/Phase1 state.

Included:
- v99 Phase0 fact-lock and Phase1 F2 chunk-causal reproduction scripts
- v99 plan, execution log, and experiment retrospective
- MV_AP contract/evaluator code
- v98.1 F2 provenance scripts and related v91-v98.1 helper scripts
- required local helper modules imported by those scripts
- package README, filelist, payload SHA256 sums, selected git diff, and git status

Not included:
- Stream3D/outputs/audit artifacts
- datasets, caches, checkpoints, model binaries, CSV outputs, masks, or prior audit packs

Review context:
Stream4D v99 evaluates whether the v98.1 F2 baseline can be reproduced under a
strict chunk-causal contract before DA3/D4RT increments are allowed.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        return exc.output


def copy_code(staging: Path) -> list[str]:
    copied: list[str] = []
    for rel in CODE_FILES:
        src = ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(rel)
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    return copied


def write_metadata(staging: Path, copied: list[str]) -> None:
    (staging / "README_AUDIT.md").write_text(README, encoding="utf-8")
    (staging / "GIT_STATUS_SHORT.txt").write_text(run_text(["git", "status", "--short"]), encoding="utf-8")
    (staging / "GIT_DIFF_SELECTED_CODE.patch").write_text(
        run_text(["git", "diff", "--", *copied]), encoding="utf-8"
    )
    (staging / "NO_ARTIFACTS_INCLUDED.txt").write_text(
        "This is a code-focused packet. No files under Stream3D/outputs/audit are included.\n",
        encoding="utf-8",
    )


def write_filelist_and_hashes(staging: Path) -> list[str]:
    files = sorted(
        p.relative_to(staging).as_posix()
        for p in staging.rglob("*")
        if p.is_file() and p.name != "PAYLOAD_SHA256SUMS.txt"
    )
    (staging / "PAYLOAD_FILELIST.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
    files = sorted(
        p.relative_to(staging).as_posix()
        for p in staging.rglob("*")
        if p.is_file() and p.name != "PAYLOAD_SHA256SUMS.txt"
    )
    (staging / "PAYLOAD_SHA256SUMS.txt").write_text(
        "\n".join(f"{sha256(staging / name)}  {name}" for name in files) + "\n",
        encoding="utf-8",
    )
    return sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())


def make_zip(staging: Path, zip_path: Path) -> list[str]:
    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src in sorted(p for p in staging.rglob("*") if p.is_file()):
            arcname = src.relative_to(staging).as_posix()
            zf.write(src, arcname)
            entries.append(arcname)
    return entries


def validate_zip(zip_path: Path, expected_entries: list[str]) -> dict[str, object]:
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        entries = sorted(name for name in zf.namelist() if not name.endswith("/"))
    missing = sorted(set(expected_entries) - set(entries))
    extra = sorted(set(entries) - set(expected_entries))
    artifact_entries = [name for name in entries if name.startswith("Stream3D/outputs/audit/")]
    payload_hash_ok = False
    with tempfile.TemporaryDirectory(prefix="stream4d_v99_code_audit_extract_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)
        sums = tmp_path / "PAYLOAD_SHA256SUMS.txt"
        if sums.is_file():
            payload_hash_ok = True
            for line in sums.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                digest, name = line.split("  ", 1)
                if sha256(tmp_path / name) != digest:
                    payload_hash_ok = False
                    break
    return {
        "zip_test_ok": bad is None,
        "bad_zip_member": bad,
        "entry_count": len(entries),
        "expected_entry_count": len(expected_entries),
        "entry_parity_ok": not missing and not extra,
        "missing_entries": missing,
        "extra_entries": extra,
        "payload_hash_ok": payload_hash_ok,
        "artifact_entries_found": artifact_entries,
        "artifact_exclusion_ok": not artifact_entries,
    }


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pack_name = f"{NAME_PREFIX}_{timestamp}"
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    staging = PACK_ROOT / f".{pack_name}_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        copied = copy_code(staging)
        write_metadata(staging, copied)
        expected_entries = write_filelist_and_hashes(staging)
        zip_path = PACK_ROOT / f"{pack_name}.zip"
        entries = make_zip(staging, zip_path)
        archive_sha = sha256(zip_path)
        (PACK_ROOT / f"{zip_path.name}.sha256").write_text(
            f"{archive_sha}  {zip_path.name}\n", encoding="utf-8"
        )
        validation = validate_zip(zip_path, expected_entries)
        validation.update(
            {
                "archive": zip_path.as_posix(),
                "archive_sha256": archive_sha,
                "archive_size_bytes": zip_path.stat().st_size,
                "payload_file_count": len(expected_entries),
                "code_file_count": len(copied),
                "source_files": copied,
            }
        )
        (PACK_ROOT / f"{zip_path.name}.validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        ok = (
            validation["zip_test_ok"]
            and validation["entry_parity_ok"]
            and validation["payload_hash_ok"]
            and validation["artifact_exclusion_ok"]
            and len(entries) == len(expected_entries)
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if ok else 2
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
