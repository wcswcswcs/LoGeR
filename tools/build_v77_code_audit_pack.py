#!/usr/bin/env python3
"""Build the Stream4D v77 CMAP-MDL L2H audit packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = REPO_ROOT / "code_audit_pack"

CORE_FILES = [
    "tools/build_v77_code_audit_pack.py",
    "Stream3D/tools/run_v77_cmap_mdl_l2h_pipeline.py",
    "Stream3D/tools/run_v76_cmap_l2h_pipeline.py",
    "Stream3D/tools/run_v66_local_chunk_eval.py",
    "docs/stream4d_v77_cmap_mdl_l2h_experiment_plan.md",
    "docs/stream4d_v77_执行日志.md",
    "docs/stream4d_v77_实验结果复盘.md",
]

ARTIFACT_ROOTS = [
    "Stream3D/outputs/audit/v77_phase0_fact_lock",
    "Stream3D/outputs/audit/v77_phase1_cache",
    "Stream3D/outputs/audit/v77_phase2_candidate_hierarchy",
    "Stream3D/outputs/audit/v77_phase3_cmap_mdl_local",
    "Stream3D/outputs/audit/v77_phase4_gap_casebook",
    "Stream3D/outputs/audit/v77_phase5_local_controls",
    "Stream3D/outputs/audit/v77_phase6_local_decision",
    "Stream3D/outputs/audit/v77_phase7_local2history",
    "Stream3D/outputs/audit/v77_final_decision",
    "Stream3D/outputs/audit/v77_cmap_mdl_l2h_pipeline",
]

SOURCE_EVIDENCE = [
    "Stream3D/outputs/audit/v76_final_decision_r14_v68_edge_coherence_caseEfix/final_decision.json",
    "Stream3D/outputs/audit/v76_phase2_fragment_role_graph_r2/fragment_role_node_rows.csv",
    "Stream3D/outputs/audit/v76_phase2_fragment_role_graph_r2/fragment_role_edge_rows.csv",
    "Stream3D/outputs/audit/v76_phase4_role_hierarchy_r4_component_conflict_gate/hierarchy_summary.json",
    "Stream3D/outputs/audit/v76_phase4_role_hierarchy_r4_component_conflict_gate/method_cut_rows.csv",
    "Stream3D/outputs/audit/v76_phase4_role_hierarchy_r4_component_conflict_gate/oracle_cut_rows.csv",
    "Stream3D/outputs/audit/v76_phase4_role_hierarchy_r4_component_conflict_gate/hierarchy_node_rows.csv",
    "Stream3D/outputs/audit/v76_phase4_role_hierarchy_r4_component_conflict_gate/hierarchy_edge_rows.csv",
    "Stream3D/outputs/audit/v76_phase5_hierarchical_local_cut_r14_v68_edge_coherence/local_cut_summary.json",
    "Stream3D/outputs/audit/v76_phase5_hierarchical_local_cut_r14_v68_edge_coherence/local_cut_metric_rows.csv",
    "Stream3D/outputs/audit/v76_phase5_hierarchical_local_cut_r14_v68_edge_coherence/cut_variant_summary_rows.csv",
    "Stream3D/outputs/audit/v76_phase5_hierarchical_local_cut_r14_v68_edge_coherence/adapter_rows.csv",
    "Stream3D/outputs/audit/v76_phase5_hierarchical_local_cut_r14_v68_edge_coherence/local_slot_rows.csv",
    "Stream3D/outputs/audit/v76_phase6_attribution_r14_v68_edge_coherence_caseEfix/attribution_summary.json",
    "Stream3D/outputs/audit/v75_phase1_soft_incidence/incidence_summary.json",
    "Stream3D/outputs/audit/v75_phase5_local_cut_r30_mixed_oracle/control_rows.csv",
    "Stream3D/outputs/audit/v75_phase5_local_cut_r30_mixed_oracle/adapter_candidate_rows.csv",
    "Stream3D/outputs/audit/v75_phase5_local_cut_r19_nms_default/local_slot_metric_rows.csv",
    "Stream3D/outputs/audit/v75_phase5_local_cut_r20_nms_aggregatefix/local_slot_metric_rows.csv",
    "Stream3D/outputs/audit/v75_phase5_local_cut_r21_one_mask_per_slot_frame/local_slot_metric_rows.csv",
    "Stream3D/outputs/audit/v75_phase5_local_cut_r22_strict_one_mask/local_slot_metric_rows.csv",
    "Stream3D/outputs/audit/v75_phase5_local_cut_r23_merge_competing/local_slot_metric_rows.csv",
    "Stream3D/outputs/audit/v75_phase5_local_cut_r24_demote_broad/local_slot_metric_rows.csv",
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
    ".pkl",
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


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def should_exclude(path: Path) -> bool:
    rel = "/" + repo_rel(path)
    return any(part in rel for part in EXCLUDED_SUBSTRINGS) or path.suffix.lower() in EXCLUDED_SUFFIXES


def payload_rel_excluded(rel: str) -> bool:
    probe = "/" + rel
    substrings = [part for part in EXCLUDED_SUBSTRINGS if part != "/code_audit_pack/"]
    return any(part in probe for part in substrings) or Path(rel).suffix.lower() in EXCLUDED_SUFFIXES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    try:
        proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError as exc:
        return f"COMMAND_NOT_FOUND: {cmd[0]} ({exc})\n"
    return proc.stdout


def copy_path(src_rel: str, packet_dir: Path, missing: list[str], copied: list[Path]) -> None:
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
        if path.is_file() and not should_exclude(path):
            dst = packet_dir / repo_rel(path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            copied.append(dst)


def write_readme(packet_dir: Path, packet_name: str, missing: list[str]) -> None:
    final_path = REPO_ROOT / "Stream3D/outputs/audit/v77_final_decision/final_decision.json"
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else {}
    readme = f"""# Stream4D v77 CMAP-MDL L2H Audit Packet

Packet: `{packet_name}`
Created: `{datetime.now().isoformat(timespec="seconds")}`
Repository: `{REPO_ROOT}`

## Scope

This packet contains the v77 canonical pipeline, the v77 plan, the two required
logs, v77 output artifacts, and the compact v75/v76 source evidence needed to
audit the cached finite-candidate CMAP-MDL decision.

The packet excludes raw data, checkpoints, generated pickle caches, media,
`__pycache__`, and nested `code_audit_pack` contents.

## Final Decision

```json
{json.dumps(final, indent=2, ensure_ascii=False, sort_keys=True)}
```

## Reproduction

Main run:

```bash
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python Stream3D/tools/run_v77_cmap_mdl_l2h_pipeline.py --stop-after final --scenes scene0011_00,scene0050_00 --max-chunks 0
```

Validation files:

- `PACKET_FILELIST.txt`
- `PAYLOAD_SHA256SUMS.txt`
- `PACKET_MANIFEST.json`

Missing expected paths:

```text
{chr(10).join(missing) if missing else "none"}
```
"""
    (packet_dir / "PACKET_README.md").write_text(readme, encoding="utf-8")


def write_metadata(packet_dir: Path, packet_name: str, missing: list[str]) -> list[str]:
    (packet_dir / "GIT_STATUS_SHORT.txt").write_text(run_text(["git", "status", "--short"]), encoding="utf-8")
    (packet_dir / "PY_COMPILE_CHECK.txt").write_text(
        run_text([
            "/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python",
            "-m",
            "py_compile",
            "Stream3D/tools/run_v77_cmap_mdl_l2h_pipeline.py",
            "tools/build_v77_code_audit_pack.py",
        ]),
        encoding="utf-8",
    )
    files = sorted(path for path in packet_dir.rglob("*") if path.is_file())
    rel_files = sorted({path.relative_to(packet_dir).as_posix() for path in files} | {"PACKET_FILELIST.txt", "PACKET_MANIFEST.json", "PAYLOAD_SHA256SUMS.txt"})
    (packet_dir / "PACKET_FILELIST.txt").write_text("\n".join(rel_files) + "\n", encoding="utf-8")

    manifest = {
        "packet_name": packet_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "missing_expected_paths": missing,
        "payload_hash_policy": {
            "excluded_from_own_hash_file": ["PAYLOAD_SHA256SUMS.txt"],
            "full_archive_hash_sidecar": f"{packet_name}.zip.sha256",
        },
        "exclusion_policy": {
            "excluded_substrings": EXCLUDED_SUBSTRINGS,
            "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
        },
        "files": [
            {
                "path": path.relative_to(packet_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    (packet_dir / "PACKET_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    hash_lines = []
    for path in sorted(p for p in packet_dir.rglob("*") if p.is_file() and p.name != "PAYLOAD_SHA256SUMS.txt"):
        hash_lines.append(f"{sha256_file(path)}  {path.relative_to(packet_dir).as_posix()}")
    (packet_dir / "PAYLOAD_SHA256SUMS.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    return sorted(path.relative_to(packet_dir).as_posix() for path in packet_dir.rglob("*") if path.is_file())


def zip_packet(packet_dir: Path, packet_name: str) -> Path:
    zip_path = PACK_ROOT / f"{packet_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(packet_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=f"{packet_name}/{path.relative_to(packet_dir).as_posix()}")
    return zip_path


def validate(packet_dir: Path, packet_name: str, zip_path: Path, rel_files: list[str]) -> dict[str, object]:
    zip_sha = sha256_file(zip_path)
    (PACK_ROOT / f"{packet_name}.zip.sha256").write_text(f"{zip_sha}  {zip_path.name}\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path) as zf:
        zip_entries = sorted(zf.namelist())
    expected_entries = sorted(f"{packet_name}/{rel}" for rel in rel_files)
    missing = sorted(set(expected_entries) - set(zip_entries))
    extra = sorted(set(zip_entries) - set(expected_entries))
    (PACK_ROOT / f"{packet_name}.entry_diff.txt").write_text(
        "\n".join([
            f"expected_entries={len(expected_entries)}",
            f"actual_entries={len(zip_entries)}",
            f"missing_entries={len(missing)}",
            f"extra_entries={len(extra)}",
            "",
            "[missing]",
            *missing,
            "",
            "[extra]",
            *extra,
            "",
        ]),
        encoding="utf-8",
    )
    unzip_test = run_text(["unzip", "-t", zip_path.name], cwd=PACK_ROOT)
    sha_check = run_text(["sha256sum", "-c", "PAYLOAD_SHA256SUMS.txt"], cwd=packet_dir)
    excluded_hits = [rel for rel in rel_files if payload_rel_excluded(rel)]
    (PACK_ROOT / f"{packet_name}.unzip_test.txt").write_text(unzip_test, encoding="utf-8")
    (PACK_ROOT / f"{packet_name}.payload_sha256_check.txt").write_text(sha_check, encoding="utf-8")
    (PACK_ROOT / f"{packet_name}.excluded_path_check.txt").write_text(
        "\n".join(excluded_hits) + ("\n" if excluded_hits else "PASS: no excluded paths in payload\n"),
        encoding="utf-8",
    )
    return {
        "packet_dir": str(packet_dir),
        "zip_path": str(zip_path),
        "zip_sha256": zip_sha,
        "zip_size_bytes": zip_path.stat().st_size,
        "payload_file_count": len(rel_files),
        "zip_entry_count": len(zip_entries),
        "entry_diff_missing": len(missing),
        "entry_diff_extra": len(extra),
        "unzip_test_ok": "No errors detected" in unzip_test,
        "payload_sha256_check_ok": "FAILED" not in sha_check and "No such file" not in sha_check,
        "excluded_path_hits": excluded_hits,
    }


def build(packet_name: str) -> dict[str, object]:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    packet_dir = PACK_ROOT / packet_name
    if packet_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing packet dir: {packet_dir}")
    packet_dir.mkdir(parents=True)
    missing: list[str] = []
    copied: list[Path] = []
    for rel in [*CORE_FILES, *SOURCE_EVIDENCE]:
        copy_path(rel, packet_dir, missing, copied)
    for rel in ARTIFACT_ROOTS:
        copy_path(rel, packet_dir, missing, copied)
    write_readme(packet_dir, packet_name, missing)
    rel_files = write_metadata(packet_dir, packet_name, missing)
    zip_path = zip_packet(packet_dir, packet_name)
    result = validate(packet_dir, packet_name, zip_path, rel_files)
    result["missing_expected_paths"] = missing
    result["copied_source_count_before_metadata"] = len(copied)
    (PACK_ROOT / f"{packet_name}.build_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=f"stream4d_v77_cmap_mdl_l2h_core_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    args = parser.parse_args()
    result = build(args.name)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    failed = result["entry_diff_missing"] or result["entry_diff_extra"] or not result["unzip_test_ok"] or not result["payload_sha256_check_ok"] or result["excluded_path_hits"]
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
