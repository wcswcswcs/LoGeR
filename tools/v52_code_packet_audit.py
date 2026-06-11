#!/usr/bin/env python3
"""Build and audit the ACL2 v52 supplemental code packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "run_pipeline_abc.py",
    "run_pipeline_abc_v2.py",
    "run_geometry_backbone_inference.py",
    "inference_dynamic_cue_extractor.py",
    "loger/utils/rotation.py",
    "loger/pipeline/hybrid_memory_controller.py",
    "loger/pipeline/ttt_write_controller.py",
    "loger/pipeline/semantic_prior_generator.py",
    "loger/pipeline/dynamic_cue_extractor.py",
    "loger/pipeline/geometry_backbone.py",
    "loger/models/pi3.py",
    "loger/models/layers/attention.py",
    "tools/kitti_trajectory_diagnostics.py",
    "tools/run_attention_cue_experiment.sh",
    "tools/run_v45_full_candidate.sh",
    "tools/run_v46b_factorial_candidate.sh",
    "tools/run_v47_adaptive_ttt_writer_candidate.sh",
    "tools/v45_report.py",
    "tools/v46b_factorial_report.py",
    "tools/v47_adaptive_ttt_writer_report.py",
    "tools/v52_c9_attribution_report.py",
    "tools/v52_adaptive_failure_audit.py",
    "tools/v52_support_alias_unit_audit.py",
    "tools/v52_phase0_debug_audit.py",
    "tools/v52_code_packet_audit.py",
    "tools/v52_runtime_profile_report.py",
    "eval/long_eval_script/kitti_benchmark",
    "docs/ACL2_v52_C9Clean_AdaptiveTTT_SemanticGeometry_Experiment_Plan.md",
    "docs/ACL2_v52_C9Clean_AdaptiveTTT_SemanticGeometry_执行日志.md",
    "docs/ACL2_v52_C9Clean_AdaptiveTTT_SemanticGeometry_实验复盘.md",
    "docs/C9_P0_R2_Pipelinev2_Configuration_Explainer.md",
    "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase0_code_audit/bugfix_report.md",
    "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase0_code_audit/support_alias_unit_audit_summary.json",
    "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase0_code_audit/support_alias_unit_audit_summary.csv",
    "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase0_code_audit/adaptive_writer_debug_field_audit.json",
    "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase0_code_audit/code_packet_completeness_audit.md",
    "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase1_c9_attribution/c9_component_attribution_report.md",
    "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase1_c9_attribution/phase1_attribution_summary.json",
    "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase2_adaptive_failure_audit/adaptive_failure_autopsy.md",
    "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase2_adaptive_failure_audit/phase2_adaptive_failure_summary.json",
    "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_full/report_R2_noswa_done/v47_adaptive_ttt_writer_report.md",
    "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_full/report_R2_noswa_done/v47_adaptive_ttt_writer_registry.csv",
    "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/runtime_efficiency_audit/v52_runtime_profile_summary.md",
    "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/runtime_efficiency_audit/v52_runtime_profile_summary.json",
    "results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/runtime_efficiency_audit/v52_runtime_profile_summary.csv",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = [dict(row) for row in rows]
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_capture(args: List[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    packet_root = out_dir / "acl2_v52_code_audit_packet"
    files_root = packet_root / "files"
    if packet_root.exists():
        shutil.rmtree(packet_root)
    files_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for rel in REQUIRED_PATHS:
        src = REPO_ROOT / rel
        dst = files_root / rel
        exists = src.exists()
        row: Dict[str, object] = {
            "path": rel,
            "repo_exists": bool(exists),
            "packaged": False,
            "is_dir": bool(src.is_dir()) if exists else False,
            "bytes": "",
            "sha256": "",
        }
        if exists:
            if src.is_dir():
                shutil.copytree(src, dst)
                file_count = 0
                byte_count = 0
                digest = hashlib.sha256()
                for file_path in sorted(dst.rglob("*")):
                    if file_path.is_file():
                        file_count += 1
                        data = file_path.read_bytes()
                        byte_count += len(data)
                        digest.update(str(file_path.relative_to(packet_root)).encode("utf-8"))
                        digest.update(data)
                row["file_count"] = file_count
                row["bytes"] = byte_count
                row["sha256"] = digest.hexdigest()
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                row["bytes"] = dst.stat().st_size
                row["sha256"] = _sha256(dst)
            row["packaged"] = True
        rows.append(row)

    manifest_path = packet_root / "ACL2_v52_code_packet_manifest.csv"
    _write_csv(manifest_path, rows)
    (packet_root / "ACL2_v52_code_packet_git_status.txt").write_text(
        _git_capture(["status", "--short", "--", *REQUIRED_PATHS]),
        encoding="utf-8",
    )
    (packet_root / "ACL2_v52_code_packet_git_diff.patch").write_text(
        _git_capture(["diff", "--", *REQUIRED_PATHS]),
        encoding="utf-8",
    )

    zip_path = out_dir / "ACL2_v52_code_audit_packet.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(packet_root.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(out_dir))
    filelist_path = out_dir / "ACL2_v52_code_audit_packet_filelist.txt"
    with zipfile.ZipFile(zip_path, "r") as archive:
        filelist_path.write_text(
            "\n".join(sorted(archive.namelist())) + "\n",
            encoding="utf-8",
        )
    zip_sha = _sha256(zip_path)
    sha_path = out_dir / "ACL2_v52_code_audit_packet.zip.sha256"
    sha_path.write_text(f"{zip_sha}  {zip_path.name}\n", encoding="utf-8")

    missing = [str(row["path"]) for row in rows if not row["repo_exists"]]
    unpackaged = [str(row["path"]) for row in rows if not row["packaged"]]
    old_pack_note = (
        "The previous v45-v51 audit packet was sufficient for historical code review, "
        "but missed several supporting files needed by a fresh reproducer. "
        "This v52 supplemental packet packages the required files from the current repository paths."
    )
    report_lines = [
        "# ACL2 v52 Code Packet Completeness Audit",
        "",
        f"required_paths = {len(REQUIRED_PATHS)}",
        f"missing_in_repo = {len(missing)}",
        f"unpackaged = {len(unpackaged)}",
        f"zip = {zip_path}",
        f"zip_sha256 = {zip_sha}",
        f"filelist = {filelist_path}",
        "",
        old_pack_note,
        "",
        "## Missing Paths",
        "",
    ]
    report_lines.extend(f"- {path}" for path in missing)
    if not missing:
        report_lines.append("- none")
    report_lines.extend([
        "",
        "## Manifest",
        "",
        f"- {manifest_path}",
    ])
    (out_dir / "code_packet_completeness_audit.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    if missing or unpackaged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
