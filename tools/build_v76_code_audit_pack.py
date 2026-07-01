#!/usr/bin/env python3
"""Build a compact Stream4D v76 code/audit artifact packet.

The packet is intentionally selective: it includes the pipeline, plan/logs,
formal decision artifacts, and the small/medium evidence tables needed to audit
the current conclusion. It excludes raw data, caches, checkpoints, media, and
large intermediate edge rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = REPO_ROOT / "code_audit_pack"

CORE_PATHS = [
    "tools/build_v76_code_audit_pack.py",
    "Stream3D/tools/run_v76_cmap_l2h_pipeline.py",
    "Stream3D/tools/run_v66_local_chunk_eval.py",
    "Stream3D/tools/run_v68_edge_audit.py",
    "Stream3D/stream4d_native/v75_fact_lock.py",
    "Stream3D/stream4d_native/v75_soft_incidence.py",
    "docs/stream4d_v76_cmap_l2h_fragment_role_hierarchy_experiment_plan.md",
    "docs/stream4d_v76_执行日志.md",
    "docs/stream4d_v76_实验结果复盘.md",
]

ARTIFACT_ROOTS = [
    "Stream3D/outputs/audit/v68_edge_audit_dinov2",
    "Stream3D/outputs/audit/v76_phase0_fact_lock_r2",
    "Stream3D/outputs/audit/v76_phase1_headroom_r2",
    "Stream3D/outputs/audit/v76_phase2_fragment_role_graph_r2",
    "Stream3D/outputs/audit/v76_phase3_role_propagation_r4_component_conflict_gate",
    "Stream3D/outputs/audit/v76_phase4_role_hierarchy_r4_component_conflict_gate",
    "Stream3D/outputs/audit/v76_phase5_hierarchical_local_cut_r14_v68_edge_coherence",
    "Stream3D/outputs/audit/v76_phase6_attribution_r14_v68_edge_coherence",
    "Stream3D/outputs/audit/v76_phase7_local2history_r14_v68_edge_coherence",
    "Stream3D/outputs/audit/v76_cmap_l2h_pipeline_r14_v68_edge_coherence",
    "Stream3D/outputs/audit/v76_final_decision_r14_v68_edge_coherence",
    "Stream3D/outputs/audit/v76_phase6_attribution_r14_v68_edge_coherence_caseEfix",
    "Stream3D/outputs/audit/v76_phase7_local2history_r14_v68_edge_coherence_caseEfix",
    "Stream3D/outputs/audit/v76_cmap_l2h_pipeline_r14_v68_edge_coherence_caseEfix",
    "Stream3D/outputs/audit/v76_final_decision_r14_v68_edge_coherence_caseEfix",
    "Stream3D/outputs/audit/v76_phase5_hierarchical_local_cut_r15_relaxed_merge",
    "Stream3D/outputs/audit/v76_phase6_attribution_r15_relaxed_merge",
    "Stream3D/outputs/audit/v76_cmap_l2h_pipeline_r15_relaxed_merge",
    "Stream3D/outputs/audit/v76_final_decision_r15_relaxed_merge",
    "Stream3D/outputs/audit/v76_phase5_hierarchical_local_cut_r16_moderate_merge",
    "Stream3D/outputs/audit/v76_phase6_attribution_r16_moderate_merge",
    "Stream3D/outputs/audit/v76_cmap_l2h_pipeline_r16_moderate_merge",
    "Stream3D/outputs/audit/v76_final_decision_r16_moderate_merge",
    "Stream3D/outputs/audit/v76_phase5_color_probe_r11",
    "Stream3D/outputs/audit/v76_phase5_dino_semantic_probe_r12",
    "Stream3D/outputs/audit/v76_phase5_v68_edge_probe_r13",
    "Stream3D/outputs/audit/v76_phase5_component_edge_merge_probe_r17",
    "Stream3D/outputs/audit/v76_phase5_all_candidate_component_vote_probe_r18",
]

EXCLUDED_SUBSTRINGS = [
    "/__pycache__/",
    "/code_audit_pack/",
    "/data/",
    "/checkpoints/",
    "/weights/",
]

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".ckpt",
    ".npy",
    ".npz",
    ".png",
    ".jpg",
    ".jpeg",
    ".mp4",
    ".mov",
    ".avi",
}

EXCLUDED_FILENAMES = {
    "edge_rows.csv",  # v68 raw edge table is large; summary and sha rows are enough for this review pack.
}

KEY_ARTIFACTS = {
    "r14_final": "Stream3D/outputs/audit/v76_final_decision_r14_v68_edge_coherence/final_decision.json",
    "caseEfix_final": "Stream3D/outputs/audit/v76_final_decision_r14_v68_edge_coherence_caseEfix/final_decision.json",
    "r15_final": "Stream3D/outputs/audit/v76_final_decision_r15_relaxed_merge/final_decision.json",
    "r16_final": "Stream3D/outputs/audit/v76_final_decision_r16_moderate_merge/final_decision.json",
    "r18_probe": "Stream3D/outputs/audit/v76_phase5_all_candidate_component_vote_probe_r18/summary.json",
}


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def should_exclude(path: Path) -> bool:
    rel = "/" + repo_rel(path)
    if any(part in rel for part in EXCLUDED_SUBSTRINGS):
        return True
    if path.name in EXCLUDED_FILENAMES:
        return True
    return path.suffix.lower() in EXCLUDED_SUFFIXES


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_text(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as exc:
        return f"COMMAND_NOT_FOUND: {cmd[0]} ({exc})\n"
    return proc.stdout


def copy_one(src_rel: str, packet_dir: Path, missing: list[str], copied: list[Path]) -> None:
    src = REPO_ROOT / src_rel
    if not src.exists():
        missing.append(src_rel)
        return
    if src.is_file():
        if should_exclude(src):
            return
        dst = packet_dir / src_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
        return
    for path in sorted(src.rglob("*")):
        if not path.is_file() or should_exclude(path):
            continue
        dst = packet_dir / repo_rel(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        copied.append(dst)


def read_key_artifact(rel_path: str) -> dict | None:
    path = REPO_ROOT / rel_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - summary should record malformed evidence, not hide it.
        return {"read_error": str(exc)}


def write_packet_readme(packet_dir: Path, packet_name: str, missing: list[str]) -> None:
    key_data = {name: read_key_artifact(path) for name, path in KEY_ARTIFACTS.items()}
    readme = f"""# Stream4D v76 CMAP L2H Audit Packet

Packet name: `{packet_name}`
Repository root: `{REPO_ROOT}`
Created at: `{datetime.now().isoformat(timespec="seconds")}`

## Scope

This packet contains the v76 core pipeline code, experiment plan, execution log,
experiment retrospective, formal decision artifacts, and compact evidence tables
for the current Stream4D CMAP local/local2history investigation.

It intentionally excludes raw datasets, caches, checkpoints, image/video media,
the local `code_audit_pack` directory, and the large v68 `edge_rows.csv`
intermediate. The packet keeps v68 summaries and hash/provenance rows where
available.

## Current Verified Conclusion

The current audited status is `NO_GO` / blocked before local2history promotion:
the Phase5 non-GT local cut remains below the required gate. The best formal
r14/r15/r16 score observed in the logs and final decision artifacts is the
`LC19_rgb_v68_edge_component_expand_f1_0p02` local variant with:

- `sf50 = 0.34594375775296826`
- `ap50 = 0.058352455741693905`
- `gt_iou = 0.36618969589471817`

The Phase4 oracle headroom artifact remains substantially higher
(`oracle_best_iou = 0.5298801296169717`), so the active bottleneck is the
non-GT local cut/adapter scoring path rather than local2history.

## Key Artifacts Embedded

```json
{json.dumps(key_data, indent=2, ensure_ascii=False, sort_keys=True)}
```

## Reproduction / Audit Pointers

- Pipeline: `Stream3D/tools/run_v76_cmap_l2h_pipeline.py`
- Evaluator dependency: `Stream3D/tools/run_v66_local_chunk_eval.py`
- Plan: `docs/stream4d_v76_cmap_l2h_fragment_role_hierarchy_experiment_plan.md`
- Execution log: `docs/stream4d_v76_执行日志.md`
- Retrospective log: `docs/stream4d_v76_实验结果复盘.md`
- File list: `PACKET_FILELIST.txt`
- Payload hashes: `PAYLOAD_SHA256SUMS.txt` (`PAYLOAD_SHA256SUMS.txt`
  itself is excluded from its own hash list; the zip sidecar covers the full
  archive bytes)

## Missing Expected Paths

```text
{os.linesep.join(missing) if missing else "none"}
```
"""
    (packet_dir / "PACKET_README.md").write_text(readme, encoding="utf-8")


def write_git_context(packet_dir: Path) -> None:
    (packet_dir / "GIT_STATUS_SHORT.txt").write_text(
        run_text(["git", "status", "--short"]),
        encoding="utf-8",
    )
    (packet_dir / "GIT_DIFF_STAT.txt").write_text(
        run_text(["git", "diff", "--stat", "--", "Stream3D/tools/run_v76_cmap_l2h_pipeline.py"]),
        encoding="utf-8",
    )
    (packet_dir / "PY_COMPILE_CHECK.txt").write_text(
        run_text(
            [
                "/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python",
                "-m",
                "py_compile",
                "Stream3D/tools/run_v76_cmap_l2h_pipeline.py",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest(packet_dir: Path, packet_name: str, missing: list[str]) -> list[str]:
    metadata_names = {
        "PACKET_FILELIST.txt",
        "PACKET_MANIFEST.json",
        "PAYLOAD_SHA256SUMS.txt",
    }
    files = sorted(path for path in packet_dir.rglob("*") if path.is_file())
    rel_files = sorted({path.relative_to(packet_dir).as_posix() for path in files} | metadata_names)
    (packet_dir / "PACKET_FILELIST.txt").write_text(
        "\n".join(rel_files) + "\n",
        encoding="utf-8",
    )

    files = sorted(path for path in packet_dir.rglob("*") if path.is_file())
    manifest_files = []
    for path in files:
        rel = path.relative_to(packet_dir).as_posix()
        digest = sha256_file(path)
        manifest_files.append({"path": rel, "size_bytes": path.stat().st_size, "sha256": digest})

    manifest = {
        "packet_name": packet_name,
        "repo_root": str(REPO_ROOT),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "missing_expected_paths": missing,
        "planned_payload_file_count": len(rel_files),
        "manifested_file_count_before_manifest_and_hash": len(manifest_files),
        "total_manifested_bytes_before_manifest_and_hash": sum(item["size_bytes"] for item in manifest_files),
        "payload_hash_policy": {
            "hash_file": "PAYLOAD_SHA256SUMS.txt",
            "excluded_from_own_hash_file": ["PAYLOAD_SHA256SUMS.txt"],
            "full_archive_hash_sidecar": f"{packet_name}.zip.sha256",
        },
        "exclusion_policy": {
            "excluded_substrings": EXCLUDED_SUBSTRINGS,
            "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
            "excluded_filenames": sorted(EXCLUDED_FILENAMES),
        },
        "files": manifest_files,
    }
    (packet_dir / "PACKET_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    hash_files = sorted(path for path in packet_dir.rglob("*") if path.is_file() and path.name != "PAYLOAD_SHA256SUMS.txt")
    hash_lines = []
    for path in hash_files:
        rel = path.relative_to(packet_dir).as_posix()
        hash_lines.append(f"{sha256_file(path)}  {rel}")
    (packet_dir / "PAYLOAD_SHA256SUMS.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    return sorted(path.relative_to(packet_dir).as_posix() for path in packet_dir.rglob("*") if path.is_file())


def write_zip(packet_dir: Path, packet_name: str) -> Path:
    zip_path = PACK_ROOT / f"{packet_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(packet_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(packet_dir).as_posix()
                zf.write(path, arcname=f"{packet_name}/{rel}")
    return zip_path


def validate_packet(packet_dir: Path, packet_name: str, zip_path: Path, rel_files: list[str]) -> dict:
    sidecar_base = PACK_ROOT / packet_name
    zip_digest = sha256_file(zip_path)
    (PACK_ROOT / f"{packet_name}.zip.sha256").write_text(
        f"{zip_digest}  {zip_path.name}\n",
        encoding="utf-8",
    )

    zip_list = sorted(zipfile.ZipFile(zip_path).namelist())
    expected_zip_list = sorted(f"{packet_name}/{rel}" for rel in rel_files)
    missing_entries = sorted(set(expected_zip_list) - set(zip_list))
    extra_entries = sorted(set(zip_list) - set(expected_zip_list))
    entry_diff_text = [
        f"expected_entries={len(expected_zip_list)}",
        f"actual_entries={len(zip_list)}",
        f"missing_entries={len(missing_entries)}",
        f"extra_entries={len(extra_entries)}",
        "",
        "[missing]",
        *missing_entries,
        "",
        "[extra]",
        *extra_entries,
        "",
    ]
    (PACK_ROOT / f"{packet_name}.entry_diff.txt").write_text("\n".join(entry_diff_text), encoding="utf-8")

    unzip_test = run_text(["unzip", "-t", zip_path.name], cwd=PACK_ROOT)
    (PACK_ROOT / f"{packet_name}.unzip_test.txt").write_text(unzip_test, encoding="utf-8")

    sha_check = run_text(["sha256sum", "-c", "PAYLOAD_SHA256SUMS.txt"], cwd=packet_dir)
    (PACK_ROOT / f"{packet_name}.payload_sha256_check.txt").write_text(sha_check, encoding="utf-8")

    excluded_hits: list[str] = []
    for rel in rel_files:
        probe = "/" + rel
        if any(part in probe for part in EXCLUDED_SUBSTRINGS):
            excluded_hits.append(rel)
        elif Path(rel).name in EXCLUDED_FILENAMES:
            excluded_hits.append(rel)
        elif Path(rel).suffix.lower() in EXCLUDED_SUFFIXES:
            excluded_hits.append(rel)
    (PACK_ROOT / f"{packet_name}.excluded_path_check.txt").write_text(
        "\n".join(excluded_hits) + ("\n" if excluded_hits else "PASS: no excluded paths in payload\n"),
        encoding="utf-8",
    )

    return {
        "packet_dir": str(packet_dir),
        "zip_path": str(zip_path),
        "zip_sha256": zip_digest,
        "payload_file_count": len(rel_files),
        "zip_entry_count": len(zip_list),
        "zip_size_bytes": zip_path.stat().st_size,
        "entry_diff_missing": len(missing_entries),
        "entry_diff_extra": len(extra_entries),
        "unzip_test_ok": "No errors detected" in unzip_test,
        "payload_sha256_check_ok": "FAILED" not in sha_check and "No such file" not in sha_check,
        "excluded_path_hits": excluded_hits,
        "sidecars": [
            str(PACK_ROOT / f"{packet_name}.zip.sha256"),
            str(PACK_ROOT / f"{packet_name}.entry_diff.txt"),
            str(PACK_ROOT / f"{packet_name}.unzip_test.txt"),
            str(PACK_ROOT / f"{packet_name}.payload_sha256_check.txt"),
            str(PACK_ROOT / f"{packet_name}.excluded_path_check.txt"),
        ],
    }


def build(packet_name: str) -> dict:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    packet_dir = PACK_ROOT / packet_name
    if packet_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing packet dir: {packet_dir}")
    packet_dir.mkdir(parents=True)

    missing: list[str] = []
    copied: list[Path] = []
    for rel in CORE_PATHS:
        copy_one(rel, packet_dir, missing, copied)
    for rel in ARTIFACT_ROOTS:
        copy_one(rel, packet_dir, missing, copied)

    write_packet_readme(packet_dir, packet_name, missing)
    write_git_context(packet_dir)
    rel_files = write_manifest(packet_dir, packet_name, missing)
    zip_path = write_zip(packet_dir, packet_name)
    validation = validate_packet(packet_dir, packet_name, zip_path, rel_files)
    validation["missing_expected_paths"] = missing
    validation["copied_source_count_before_metadata"] = len(copied)

    (PACK_ROOT / f"{packet_name}.build_summary.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default=f"stream4d_v76_cmap_l2h_core_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Packet directory/archive basename under code_audit_pack/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(args.name)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    if (
        result["entry_diff_missing"]
        or result["entry_diff_extra"]
        or not result["unzip_test_ok"]
        or not result["payload_sha256_check_ok"]
        or result["excluded_path_hits"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
