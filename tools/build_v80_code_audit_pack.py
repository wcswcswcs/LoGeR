#!/usr/bin/env python3
"""Build a compact ACL2 v80 code/artifact review packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable


RESULT_ROOT = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final"
)

DOC_FILES = [
    Path("docs/ACL2_v80TF_MultiSeq_GoodBadCase_SemanticThreeMemoryControl_Plan.md"),
    Path("docs/ACL2_v80TF_MultiSeq_GoodBadCase_SemanticThreeMemoryControl_执行日志.md"),
    Path("docs/ACL2_v80TF_MultiSeq_GoodBadCase_SemanticThreeMemoryControl_实验结果复盘.md"),
]

CODE_FILES = [
    Path("tools/audit_v80_multiseq_semantic_memory_artifacts.py"),
    Path("tools/build_v80_three_memory_good_bad_case_bank.py"),
    Path("tools/summarize_v80_case_bank.py"),
    Path("tools/visualize_v80_case_pca_qkv_ttt_panels.py"),
    Path("tools/audit_v80_visual_case_panels.py"),
    Path("tools/review_v80_visual_case_insights.py"),
    Path("tools/run_v80_read_swa_direct_hook_smoke.py"),
    Path("tools/audit_v80_direct_hook_repair_artifacts.py"),
    Path("tools/build_v80_direct_hook_enhanced_visual_panels.py"),
    Path("tools/run_v78_ttt_five_chunk_visual_probe.py"),
    Path("tools/visualize_v78_phase4_ttt_output_separated.py"),
    Path("tools/review_v78_ttt_five_chunk_visual_probe.py"),
    Path("tools/build_v80_code_audit_pack.py"),
    Path("tools/audit_v80_phase2_multiseq_ttt_abs_error_overlap.py"),
    Path("tools/run_v80_ttt_selected_geometry_visual_probe.py"),
    Path("tools/diagnose_v80_ttt_selected_visual_probe_geometry.py"),
    Path("tools/build_v80_error_semantic_overlap_support.py"),
    Path("tools/build_v80_selected_write_support_map.py"),
    Path("tools/audit_v80_selected_write_low_support_coverage.py"),
    Path("tools/audit_v80_selected_write_support_batch_insights.py"),
    Path("tools/run_v80_downstream_probe_batch.py"),
    Path("tools/merge_v80_downstream_probe_summaries.py"),
    Path("tools/analyze_v80_selected_write_insight_matrix.py"),
    Path("tools/run_v80_ttt_long_case_accelerated_smoke.py"),
    Path("tools/audit_v80_current_action_evidence_matrix.py"),
    Path("tools/audit_v80_formal_no_go_review.py"),
    Path("loger/models/layers/attention.py"),
    Path("loger/models/pi3.py"),
    Path("loger/pipeline/hybrid_memory_controller.py"),
    Path("loger/pipeline/ttt_write_controller.py"),
    Path("run_pipeline_abc.py"),
    Path("run_pipeline_abc_v2.py"),
    Path("configs/v79tf_semantic_three_memory_rules.yaml"),
]

SUMMARY_DIRS = [
    RESULT_ROOT / "phase0_multiseq_artifact_audit",
    RESULT_ROOT / "phase1_three_memory_case_bank",
    RESULT_ROOT / "phase2_case_visual_confirmation",
    RESULT_ROOT / "phase2_direct_hook_repair" / "direct_hook_audit",
    RESULT_ROOT / "phase2_direct_hook_repair" / "direct_hook_audit_seqaware",
    RESULT_ROOT / "phase2_direct_hook_repair" / "direct_hook_audit_seqaware_after_seq01_full",
    RESULT_ROOT / "phase2_direct_hook_repair" / "direct_hook_audit_seqaware_after_seq05_full",
    RESULT_ROOT / "phase2_direct_hook_enhanced_visual_review",
    RESULT_ROOT / "phase2_direct_hook_enhanced_visual_review_seq01_all",
    RESULT_ROOT / "phase2_direct_hook_enhanced_visual_review_seq05_all",
    RESULT_ROOT / "phase10_current_action_evidence_matrix_20260622_2210",
    RESULT_ROOT / "phase10_formal_no_go_review_20260622_2214",
    RESULT_ROOT / "phase10_phase2_multiseq_ttt_abs_error_overlap_20260622_2223",
    RESULT_ROOT / "phase10_seq00_chunk142_selected_write_low_support_map_20260622_2211",
    RESULT_ROOT / "phase10_seq02_chunk66_abs_error_selected_write_map_20260623_0005",
    RESULT_ROOT / "phase10_seq02_chunk69_abs_error_selected_write_map_20260623_0005",
    RESULT_ROOT / "phase10_seq05_chunk83_abs_error_selected_write_map_20260622_2237",
    RESULT_ROOT / "phase10_seq05_abs_error_selected_write_support_maps_highctrl_20260622_2322",
    RESULT_ROOT / "phase10_selected_write_extra_insights_20260623_0041",
    RESULT_ROOT / "phase10_selected_write_extra_insights_thr045_20260623_0045",
    RESULT_ROOT / "phase10_selected_write_extra_insights_thr040_20260623_0045",
    RESULT_ROOT / "phase10_seq05_chunk83_84_downstream_probe_20260623_0055",
    RESULT_ROOT / "phase10_seq02_downstream_probe_batch_smoke_chunk62_20260623_0123",
    RESULT_ROOT / "phase10_seq02_downstream_probe_batch_remaining4_3parallel_20260623_0140",
    RESULT_ROOT / "phase10_seq02_downstream_probe_batch_combined5_20260623_0152",
    RESULT_ROOT / "phase10_downstream_sign_probe_3case_20260623_0226",
    RESULT_ROOT / "phase10_downstream_sign_probe_combined8_20260623_0233",
    RESULT_ROOT / "phase10_selected_write_insight_matrix_20260623_0235",
]

WRAPPER_DIRS = [
    RESULT_ROOT / "phase2_direct_hook_repair" / "read_swa_seq01_chunks010_013",
    RESULT_ROOT / "phase2_direct_hook_repair" / "read_swa_seq01_mid_extra_chunks008_011_014",
    RESULT_ROOT / "phase2_direct_hook_repair" / "read_swa_seq01_full_case_missing_chunks001_002_016_017_026_029_030",
    RESULT_ROOT / "phase2_direct_hook_repair" / "read_swa_seq05_full_case_chunks000_007_008_020_021_067_068_079_080_081_082_092",
    RESULT_ROOT / "phase2_direct_hook_repair" / "ttt_seq01_chunks005_010",
    RESULT_ROOT / "phase2_direct_hook_repair" / "ttt_seq01_full_case_missing_chunks003_004_027_032",
    RESULT_ROOT / "phase2_direct_hook_repair" / "ttt_seq05_full_case_chunks005_009_020_024_075_083",
]

LIGHT_EXTS = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".png",
    ".html",
}

FORBIDDEN_EXTS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".bin",
}

FORBIDDEN_PARTS = {
    "__pycache__",
    ".git",
    "code_audit_pack",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str], cwd: Path, out_file: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        f"$ {' '.join(cmd)}\n"
        f"returncode={proc.returncode}\n\n"
        f"[stdout]\n{proc.stdout}\n\n"
        f"[stderr]\n{proc.stderr}\n",
        encoding="utf-8",
    )
    return proc


def should_copy(path: Path, include_png: bool) -> bool:
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        return False
    if path.suffix in FORBIDDEN_EXTS:
        return False
    if path.suffix == ".png":
        return include_png
    return path.suffix in LIGHT_EXTS


def iter_light_files(root: Path, include_png: bool) -> Iterable[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and should_copy(path, include_png=include_png):
            files.append(path)
    return sorted(files)


def copy_file(src: Path, packet_root: Path, copied: set[Path]) -> None:
    if not src.is_file():
        return
    rel = src
    if rel in copied:
        return
    dst = packet_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.add(rel)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def list_payload_files(packet_root: Path) -> list[Path]:
    return sorted(p for p in packet_root.rglob("*") if p.is_file())


def write_payload_sidecars(packet_root: Path) -> dict[str, int]:
    filelist = packet_root / "PAYLOAD_FILELIST.txt"
    hashes = packet_root / "PAYLOAD_SHA256SUMS.txt"
    filelist.touch()
    hashes.touch()
    files = list_payload_files(packet_root)
    write_text(filelist, "\n".join(str(p.relative_to(packet_root)) for p in files) + "\n")
    lines = []
    for path in files:
        rel = path.relative_to(packet_root)
        if rel in {Path("PAYLOAD_FILELIST.txt"), Path("PAYLOAD_SHA256SUMS.txt")}:
            continue
        lines.append(f"{sha256_file(path)}  {rel}")
    write_text(hashes, "\n".join(lines) + "\n")
    return {"payload_files": len(files), "payload_hash_rows": len(lines)}


def zip_packet(packet_root: Path, zip_path: Path) -> list[str]:
    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in list_payload_files(packet_root):
            arcname = Path(packet_root.name) / path.relative_to(packet_root)
            zf.write(path, arcname)
            entries.append(str(arcname))
    return sorted(entries)


def strip_top(entry: str) -> str:
    parts = Path(entry).parts
    return str(Path(*parts[1:])) if len(parts) > 1 else ""


def write_manifest(packet_root: Path, tag: str, copied_count: int) -> None:
    manifest = {
        "schema": "acl2_v80tf_code_audit_pack_v1",
        "tag": tag,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "partial_progress_review_packet",
        "full_v80_objective_complete": False,
        "full_phase2_action_ready_gate_pass": False,
        "known_complete_scope": {
            "phase1_case_bank_cases": 72,
            "direct_confirmed_sequences": ["01", "05"],
            "direct_confirmed_cases": 36,
            "remaining_missing_sequences": ["00", "02"],
        },
        "copied_source_files_before_sidecars": copied_count,
        "exclusion_policy": {
            "forbidden_extensions": sorted(FORBIDDEN_EXTS),
            "forbidden_path_parts": sorted(FORBIDDEN_PARTS),
            "raw_tensor_and_checkpoint_artifacts": "excluded; original workspace paths are preserved in manifests/logs",
        },
        "included_sections": [
            "v80 docs and dual logs",
            "v80 and supporting direct-hook code",
            "phase0/phase1/phase2 summary artifacts",
            "seq-aware direct-hook audits",
            "seq01/seq05 enhanced visual reviews and PNG panels",
            "wrapper manifests and summaries for READ/SWA and TTT direct-hook runs",
            "phase10 selected-write low-support and downstream-sign diagnostic artifacts",
            "git status/diff and validation sidecars",
        ],
    }
    write_text(packet_root / "PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_readme(packet_root: Path, tag: str) -> None:
    text = f"""# ACL2 v80 TF Semantic Three Memory Control Audit Pack

Tag: {tag}

This is a compact reviewer bundle for the current partial v80 state. It does
not claim the full v80 objective is complete.

Current evidence:

- Phase1 case bank has 72 cases across seq00/01/02/05.
- Direct-hook enhanced review is complete for seq01 and seq05 only: 36 cases.
- seq00 and seq02 still lack direct-hook evidence.
- full_phase2_action_ready_gate_pass remains false.
- Phase10 selected-write/downstream diagnostics are included for review.
- Latest boundary remains diagnostic-only; method_gate_claimed is false.

Large raw tensors, checkpoints, caches, and old code_audit_pack content are
excluded. The original artifact paths remain in the logs and manifests.
"""
    write_text(packet_root / "README.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=f"acl2_v80tf_seq01_seq05_direct_hook_partial_audit_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--out-dir", default="code_audit_pack")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd()
    out_dir = Path(args.out_dir)
    packet_root = out_dir / args.tag
    zip_path = out_dir / f"{args.tag}.zip"

    if packet_root.exists():
        if not args.overwrite:
            raise SystemExit(f"packet root already exists: {packet_root}")
        shutil.rmtree(packet_root)
    if zip_path.exists() and not args.overwrite:
        raise SystemExit(f"zip already exists: {zip_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    packet_root.mkdir(parents=True)

    copied: set[Path] = set()
    for src in DOC_FILES + CODE_FILES:
        copy_file(src, packet_root, copied)
    for src in SUMMARY_DIRS:
        include_png = src.name in {
            "phase2_direct_hook_enhanced_visual_review",
            "phase2_direct_hook_enhanced_visual_review_seq01_all",
            "phase2_direct_hook_enhanced_visual_review_seq05_all",
        }
        for path in iter_light_files(src, include_png=include_png):
            copy_file(path, packet_root, copied)
    for src in WRAPPER_DIRS:
        for path in iter_light_files(src, include_png=False):
            if path.name.endswith((".json", ".jsonl", ".log", ".md", ".txt", ".csv", ".yaml", ".yml")):
                copy_file(path, packet_root, copied)

    write_readme(packet_root, args.tag)
    write_manifest(packet_root, args.tag, len(copied))

    run_cmd(["git", "status", "--short"], repo, packet_root / "_audit_metadata" / "git_status_short.txt")
    run_cmd(
        [
            "git",
            "diff",
            "--",
            "docs/ACL2_v80TF_MultiSeq_GoodBadCase_SemanticThreeMemoryControl_Plan.md",
            "docs/ACL2_v80TF_MultiSeq_GoodBadCase_SemanticThreeMemoryControl_执行日志.md",
            "docs/ACL2_v80TF_MultiSeq_GoodBadCase_SemanticThreeMemoryControl_实验结果复盘.md",
            "tools/audit_v80_direct_hook_repair_artifacts.py",
            "tools/build_v80_direct_hook_enhanced_visual_panels.py",
            "tools/build_v80_code_audit_pack.py",
            "tools/run_v80_read_swa_direct_hook_smoke.py",
            "tools/visualize_v80_case_pca_qkv_ttt_panels.py",
            "loger/models/layers/attention.py",
            "loger/models/pi3.py",
            "loger/pipeline/hybrid_memory_controller.py",
            "loger/pipeline/ttt_write_controller.py",
            "run_pipeline_abc.py",
            "run_pipeline_abc_v2.py",
        ],
        repo,
        packet_root / "_audit_metadata" / "git_diff_relevant.patch",
    )
    py_compile_files = [str(p) for p in CODE_FILES if p.suffix == ".py" and p.exists()]
    run_cmd([sys.executable, "-m", "py_compile", *py_compile_files], repo, packet_root / "VALIDATION_PY_COMPILE.txt")

    write_text(
        packet_root / "EXCLUDED_PATHS.md",
        "\n".join(
            [
                "# Excluded Paths",
                "",
                "Excluded by policy: raw tensors/checkpoints/caches and old audit packs.",
                "",
                "Forbidden extensions:",
                ", ".join(sorted(FORBIDDEN_EXTS)),
                "",
                "Forbidden path parts:",
                ", ".join(sorted(FORBIDDEN_PARTS)),
                "",
            ]
        ),
    )
    sidecar_counts = write_payload_sidecars(packet_root)

    entries = zip_packet(packet_root, zip_path)
    write_text(out_dir / f"{args.tag}.zip_entries.txt", "\n".join(entries) + "\n")
    write_text(out_dir / f"{args.tag}.zip.sha256", f"{sha256_file(zip_path)}  {zip_path.name}\n")
    shutil.copy2(packet_root / "PAYLOAD_FILELIST.txt", out_dir / f"{args.tag}.filelist.txt")
    shutil.copy2(packet_root / "PAYLOAD_SHA256SUMS.txt", out_dir / f"{args.tag}.payload.sha256")

    payload_check = run_cmd(["sha256sum", "-c", "PAYLOAD_SHA256SUMS.txt"], packet_root, out_dir / f"{args.tag}.payload_sha256_check.txt")
    zip_check = run_cmd(["sha256sum", "-c", f"{args.tag}.zip.sha256"], out_dir, out_dir / f"{args.tag}.zip_sha256_check.txt")
    unzip_test = run_cmd(["unzip", "-t", zip_path.name], out_dir, out_dir / f"{args.tag}.unzip_test.txt")
    zip_list = run_cmd(["unzip", "-Z1", zip_path.name], out_dir, out_dir / f"{args.tag}.zip_entries.files_relative.txt")

    payload_files = [p.read_text(encoding="utf-8").strip() for p in [packet_root / "PAYLOAD_FILELIST.txt"]][0].splitlines()
    zip_payload_files = sorted(strip_top(line) for line in zip_list.stdout.splitlines() if strip_top(line))
    diff_lines = []
    if sorted(payload_files) != zip_payload_files:
        only_payload = sorted(set(payload_files) - set(zip_payload_files))
        only_zip = sorted(set(zip_payload_files) - set(payload_files))
        diff_lines.append("MISMATCH")
        diff_lines.extend(f"only_payload {p}" for p in only_payload[:100])
        diff_lines.extend(f"only_zip {p}" for p in only_zip[:100])
    else:
        diff_lines.append("PASS filelist matches zip entries after stripping packet root")
    write_text(out_dir / f"{args.tag}.files_only_diff.txt", "\n".join(diff_lines) + "\n")

    violations = []
    for entry in entries:
        rel = Path(strip_top(entry))
        if rel.suffix in FORBIDDEN_EXTS:
            violations.append(entry)
        if any(part in FORBIDDEN_PARTS for part in rel.parts):
            violations.append(entry)
    excluded_text = "PASS no forbidden paths/extensions found\n" if not violations else "\n".join(["FAIL", *violations]) + "\n"
    write_text(out_dir / f"{args.tag}.excluded_path_check.txt", excluded_text)

    result = {
        "tag": args.tag,
        "packet_root": str(packet_root),
        "zip": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "payload_files": sidecar_counts["payload_files"],
        "zip_entries": len(entries),
        "payload_sha256_check_returncode": payload_check.returncode,
        "zip_sha256_check_returncode": zip_check.returncode,
        "unzip_test_returncode": unzip_test.returncode,
        "files_only_diff_pass": diff_lines[0].startswith("PASS"),
        "excluded_path_check_pass": not violations,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(
        [
            payload_check.returncode == 0,
            zip_check.returncode == 0,
            unzip_test.returncode == 0,
            diff_lines[0].startswith("PASS"),
            not violations,
        ]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
