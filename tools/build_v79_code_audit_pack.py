#!/usr/bin/env python3
"""Build the Stream4D v79 CMAP-AF-L2H core audit packet."""

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
    "tools/build_v79_code_audit_pack.py",
    "Stream3D/tools/run_v79_cmap_af_l2h_pipeline.py",
    "Stream3D/tools/summarize_v79_repair_sweep.py",
    "Stream3D/tools/run_v77_cmap_mdl_l2h_pipeline.py",
    "Stream3D/tools/run_v76_cmap_l2h_pipeline.py",
    "Stream3D/tools/run_v75_cmap_l2h_pipeline.py",
    "Stream3D/tools/run_v75_soft_incidence.py",
    "Stream3D/tools/run_v66_local_chunk_eval.py",
    "docs/stream4d_v79_cmap_af_l2h_experiment_plan.md",
    "docs/stream4d_v79_执行日志.md",
    "docs/stream4d_v79_实验结果复盘.md",
    "docs/stream4d_v77_cmap_mdl_l2h_experiment_plan.md",
    "docs/stream4d_v77_执行日志.md",
    "docs/stream4d_v77_实验结果复盘.md",
    "docs/stream4d_v75_cmap_local_l2h_experiment_plan.md",
]

V79_ARTIFACT_ROOTS = [
    "Stream3D/outputs/audit/v79_phase0_fact_lock",
    "Stream3D/outputs/audit/v79_phase1_affinity_features",
    "Stream3D/outputs/audit/v79_phase2_neighbor_graph",
    "Stream3D/outputs/audit/v79_phase3_carrier_clustering",
    "Stream3D/outputs/audit/v79_phase4_cluster_adapter",
    "Stream3D/outputs/audit/v79_phase5_controls",
    "Stream3D/outputs/audit/v79_phase6_casebook",
    "Stream3D/outputs/audit/v79_phase7_local2history",
    "Stream3D/outputs/audit/v79_final_decision",
    "Stream3D/outputs/audit/v79_cmap_af_l2h_pipeline",
    "Stream3D/outputs/audit/v79_cmap_af_l2h_pipeline_r18_semproto_control_r8base",
    "Stream3D/outputs/audit/v79_phase1_affinity_features_r18_semproto_control_r8base",
    "Stream3D/outputs/audit/v79_phase2_neighbor_graph_r18_semproto_control_r8base",
    "Stream3D/outputs/audit/v79_phase3_carrier_clustering_r18_semproto_control_r8base",
    "Stream3D/outputs/audit/v79_phase4_cluster_adapter_r18_semproto_control_r8base",
    "Stream3D/outputs/audit/v79_phase5_controls_r18_semproto_control_r8base",
    "Stream3D/outputs/audit/v79_phase6_casebook_r18_semproto_control_r8base",
    "Stream3D/outputs/audit/v79_phase7_local2history_r18_semproto_control_r8base",
    "Stream3D/outputs/audit/v79_final_decision_r18_semproto_control_r8base",
    "Stream3D/outputs/audit/v79_hybrid_diag_pipeline_r19_w015",
    "Stream3D/outputs/audit/v79_hybrid_diag_phase1_r19_w015",
    "Stream3D/outputs/audit/v79_hybrid_diag_phase2_r19_w015",
    "Stream3D/outputs/audit/v79_hybrid_diag_pipeline_r20_w035",
    "Stream3D/outputs/audit/v79_hybrid_diag_phase1_r20_w035",
    "Stream3D/outputs/audit/v79_hybrid_diag_phase2_r20_w035",
    "Stream3D/outputs/audit/v79_repair_sweep_summary",
]

SOURCE_EVIDENCE = [
    "Stream3D/outputs/audit/v71_semantic_features",
    "Stream3D/outputs/audit/v75_phase1_soft_incidence/incidence_summary.json",
    "Stream3D/outputs/audit/v75_phase1_soft_incidence/incidence_chunk_rows.csv",
    "Stream3D/outputs/audit/v75_phase1_soft_incidence/source_rows.csv",
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


def load_json_or_empty(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"json_decode_error": str(path)}


def write_readme(packet_dir: Path, packet_name: str, missing: list[str]) -> None:
    final_path = REPO_ROOT / "Stream3D/outputs/audit/v79_final_decision_r18_semproto_control_r8base/final_decision.json"
    pipeline_summary_path = REPO_ROOT / "Stream3D/outputs/audit/v79_cmap_af_l2h_pipeline_r18_semproto_control_r8base/pipeline_summary.json"
    sweep_summary_path = REPO_ROOT / "Stream3D/outputs/audit/v79_repair_sweep_summary/sweep_summary.json"
    v77_final_path = REPO_ROOT / "Stream3D/outputs/audit/v77_final_decision/final_decision.json"
    v75_summary_path = REPO_ROOT / "Stream3D/outputs/audit/v75_phase1_soft_incidence/incidence_summary.json"
    final = load_json_or_empty(final_path)
    pipeline_summary = load_json_or_empty(pipeline_summary_path)
    sweep_summary = load_json_or_empty(sweep_summary_path)
    v77_final = load_json_or_empty(v77_final_path)
    v75_summary = load_json_or_empty(v75_summary_path)
    v79_roots = [root for root in V79_ARTIFACT_ROOTS if (REPO_ROOT / root).exists()]
    readme = f"""# Stream4D v79 CMAP-AF-L2H Core Audit Packet

Packet: `{packet_name}`
Created: `{datetime.now().isoformat(timespec="seconds")}`
Repository: `{REPO_ROOT}`

## Scope

	This packet contains the v79 CMAP-AF-L2H plan, dual logs, one-file v79
	pipeline, repair sweep summarizer, source runner dependencies, and compact
	source evidence from v71/v75/v77.
It is a reviewer packet, not a whole-worktree dump.

	The packet excludes raw data, checkpoints, tensor/cache files, media,
	`__pycache__`, and nested `code_audit_pack` contents. The large v75
	`incidence_rows.csv` file is intentionally excluded; the package keeps only the
	v75 summary, chunk rows, and source rows needed to identify the source run.

## Current v79 Output State

```json
	{json.dumps(final, indent=2, ensure_ascii=False, sort_keys=True) if final else "{}"}
	```

	Current r18 pipeline summary:

	```json
	{json.dumps(pipeline_summary, indent=2, ensure_ascii=False, sort_keys=True) if pipeline_summary else "{}"}
	```

	Repair sweep summary:

	```json
	{json.dumps(sweep_summary, indent=2, ensure_ascii=False, sort_keys=True) if sweep_summary else "{}"}
	```

Detected v79 artifact roots:

```text
{chr(10).join(v79_roots) if v79_roots else "none"}
```

## Source Evidence Anchors

v75 soft-incidence summary:

```json
{json.dumps(v75_summary, indent=2, ensure_ascii=False, sort_keys=True)}
```

v77 final decision:

```json
{json.dumps(v77_final, indent=2, ensure_ascii=False, sort_keys=True)}
```

## Reproduction Commands

Syntax check:

	```bash
	/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile Stream3D/tools/run_v79_cmap_af_l2h_pipeline.py Stream3D/tools/summarize_v79_repair_sweep.py tools/build_v79_code_audit_pack.py
	```

	Main r18 semantic-prototype-control probe:

	```bash
	CUDA_VISIBLE_DEVICES=6 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python Stream3D/tools/run_v79_cmap_af_l2h_pipeline.py --stop-after final --scenes scene0011_00,scene0050_00 --max-chunks 4 --projection-dim 2048 --idf-power 2.0 --specificity-power 2.0 --large-mask-penalty 12.0 --object-large-mask-area 0.25 --top-k 12 --graph-threshold 0.35 --cluster-threshold 0.35 --neighbor-pruning mutual --cluster-algorithm connected --adapter-min-f1 0.05 --adapter-min-precision 0.30 --semantic-feature-rows outputs/audit/v71_semantic_features/mask_feature_rows.csv --pipeline-root outputs/audit/v79_cmap_af_l2h_pipeline_r18_semproto_control_r8base --phase1-output-root outputs/audit/v79_phase1_affinity_features_r18_semproto_control_r8base --phase2-output-root outputs/audit/v79_phase2_neighbor_graph_r18_semproto_control_r8base --phase3-output-root outputs/audit/v79_phase3_carrier_clustering_r18_semproto_control_r8base --phase4-output-root outputs/audit/v79_phase4_cluster_adapter_r18_semproto_control_r8base --phase5-output-root outputs/audit/v79_phase5_controls_r18_semproto_control_r8base --phase6-output-root outputs/audit/v79_phase6_casebook_r18_semproto_control_r8base --phase7-output-root outputs/audit/v79_phase7_local2history_r18_semproto_control_r8base --final-output-root outputs/audit/v79_final_decision_r18_semproto_control_r8base
	```

	Hybrid diagnostic probes:

	```bash
	CUDA_VISIBLE_DEVICES=6 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python Stream3D/tools/run_v79_cmap_af_l2h_pipeline.py --stop-after phase2 --scenes scene0011_00,scene0050_00 --max-chunks 4 --projection-dim 2048 --idf-power 2.0 --specificity-power 2.0 --large-mask-penalty 12.0 --object-large-mask-area 0.25 --top-k 12 --graph-threshold 0.35 --cluster-threshold 0.35 --neighbor-pruning mutual --cluster-algorithm connected --adapter-min-f1 0.05 --adapter-min-precision 0.30 --semantic-feature-rows outputs/audit/v71_semantic_features/mask_feature_rows.csv --hybrid-semantic-weight 0.15 --pipeline-root outputs/audit/v79_hybrid_diag_pipeline_r19_w015 --phase1-output-root outputs/audit/v79_hybrid_diag_phase1_r19_w015 --phase2-output-root outputs/audit/v79_hybrid_diag_phase2_r19_w015

	CUDA_VISIBLE_DEVICES=7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python Stream3D/tools/run_v79_cmap_af_l2h_pipeline.py --stop-after phase2 --scenes scene0011_00,scene0050_00 --max-chunks 4 --projection-dim 2048 --idf-power 2.0 --specificity-power 2.0 --large-mask-penalty 12.0 --object-large-mask-area 0.25 --top-k 12 --graph-threshold 0.35 --cluster-threshold 0.35 --neighbor-pruning mutual --cluster-algorithm connected --adapter-min-f1 0.05 --adapter-min-precision 0.30 --semantic-feature-rows outputs/audit/v71_semantic_features/mask_feature_rows.csv --hybrid-semantic-weight 0.35 --pipeline-root outputs/audit/v79_hybrid_diag_pipeline_r20_w035 --phase1-output-root outputs/audit/v79_hybrid_diag_phase1_r20_w035 --phase2-output-root outputs/audit/v79_hybrid_diag_phase2_r20_w035
	```

Validation files:

- `PACKET_FILELIST.txt`
- `PACKET_MANIFEST.json`
- `PAYLOAD_SHA256SUMS.txt`

Missing expected paths:

```text
{chr(10).join(missing) if missing else "none"}
```
"""
    (packet_dir / "PACKET_README.md").write_text(readme, encoding="utf-8")


def write_metadata(packet_dir: Path, packet_name: str, missing: list[str]) -> list[str]:
    (packet_dir / "GIT_STATUS_SHORT.txt").write_text(run_text(["git", "status", "--short"]), encoding="utf-8")
    (packet_dir / "PY_COMPILE_CHECK.txt").write_text(
        run_text(
            [
                "/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python",
                "-m",
                "py_compile",
                "Stream3D/tools/run_v79_cmap_af_l2h_pipeline.py",
                "Stream3D/tools/summarize_v79_repair_sweep.py",
                "tools/build_v79_code_audit_pack.py",
            ]
        ),
        encoding="utf-8",
    )

    planned = sorted(
        {path.relative_to(packet_dir).as_posix() for path in packet_dir.rglob("*") if path.is_file()}
        | {"PACKET_FILELIST.txt", "PACKET_MANIFEST.json", "PAYLOAD_SHA256SUMS.txt"}
    )
    (packet_dir / "PACKET_FILELIST.txt").write_text("\n".join(planned) + "\n", encoding="utf-8")

    manifest_files = sorted(
        path for path in packet_dir.rglob("*") if path.is_file() and path.name not in {"PACKET_MANIFEST.json", "PAYLOAD_SHA256SUMS.txt"}
    )
    manifest = {
        "packet_name": packet_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "scope": "stream4d_v79_cmap_af_l2h_core_code_and_compact_artifacts",
        "missing_expected_paths": missing,
        "v79_artifact_roots_detected": [root for root in V79_ARTIFACT_ROOTS if (REPO_ROOT / root).exists()],
        "payload_hash_policy": {
            "excluded_from_own_hash_file": ["PAYLOAD_SHA256SUMS.txt"],
            "manifest_self_hash_excluded": True,
            "full_archive_hash_sidecar": f"{packet_name}.zip.sha256",
        },
        "exclusion_policy": {
            "excluded_substrings": EXCLUDED_SUBSTRINGS,
            "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
            "explicit_large_source_exclusions": [
                "Stream3D/outputs/audit/v75_phase1_soft_incidence/incidence_rows.csv"
            ],
        },
        "files": [
            {
                "path": path.relative_to(packet_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in manifest_files
        ],
    }
    (packet_dir / "PACKET_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

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

    payload_check = run_text(["sha256sum", "-c", "PAYLOAD_SHA256SUMS.txt"], cwd=packet_dir)
    (PACK_ROOT / f"{packet_name}.payload_sha256_check.txt").write_text(payload_check, encoding="utf-8")

    unzip_check = run_text(["unzip", "-t", zip_path.name], cwd=PACK_ROOT)
    (PACK_ROOT / f"{packet_name}.unzip_test.txt").write_text(unzip_check, encoding="utf-8")

    with zipfile.ZipFile(zip_path) as zf:
        zip_entries = sorted(zf.namelist())
        testzip_bad = zf.testzip()

    expected_entries = sorted(f"{packet_name}/{rel}" for rel in rel_files)
    missing_entries = sorted(set(expected_entries) - set(zip_entries))
    extra_entries = sorted(set(zip_entries) - set(expected_entries))
    (PACK_ROOT / f"{packet_name}.entry_diff.txt").write_text(
        "\n".join(
            [
                f"expected_entries={len(expected_entries)}",
                f"actual_entries={len(zip_entries)}",
                f"missing_entries={len(missing_entries)}",
                f"extra_entries={len(extra_entries)}",
                "",
                "[missing]",
                *missing_entries,
                "",
                "[extra]",
                *extra_entries,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    excluded_hits = []
    for entry in zip_entries:
        payload_rel = entry.split("/", 1)[1] if "/" in entry else entry
        if payload_rel_excluded(payload_rel):
            excluded_hits.append(entry)
    (PACK_ROOT / f"{packet_name}.excluded_path_check.txt").write_text(
        "NO_EXCLUDED_PATHS_FOUND\n" if not excluded_hits else "\n".join(excluded_hits) + "\n",
        encoding="utf-8",
    )

    summary = {
        "packet_name": packet_name,
        "zip_path": repo_rel(zip_path),
        "zip_sha256": zip_sha,
        "payload_file_count": len(rel_files),
        "zip_entry_count": len(zip_entries),
        "entry_parity_pass": not missing_entries and not extra_entries,
        "excluded_path_check_pass": not excluded_hits,
        "payload_sha256_check_pass": "FAILED" not in payload_check and "No such file" not in payload_check,
        "unzip_test_pass": "No errors detected" in unzip_check or testzip_bad is None,
        "zipfile_testzip_bad_entry": testzip_bad,
        "sidecars": [
            f"{packet_name}.zip.sha256",
            f"{packet_name}.payload_sha256_check.txt",
            f"{packet_name}.unzip_test.txt",
            f"{packet_name}.entry_diff.txt",
            f"{packet_name}.excluded_path_check.txt",
            f"{packet_name}.build_summary.json",
        ],
    }
    (PACK_ROOT / f"{packet_name}.build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def next_packet_name(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{prefix}_{timestamp}"
    packet_name = base
    suffix = 2
    while (PACK_ROOT / f"{packet_name}.zip").exists() or (PACK_ROOT / packet_name).exists():
        packet_name = f"{base}_r{suffix}"
        suffix += 1
    return packet_name


def build(prefix: str, keep_staging: bool) -> dict[str, object]:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    packet_name = next_packet_name(prefix)
    packet_dir = PACK_ROOT / packet_name
    if packet_dir.exists():
        shutil.rmtree(packet_dir)
    packet_dir.mkdir(parents=True)

    missing: list[str] = []
    copied: list[Path] = []
    for rel in CORE_FILES + V79_ARTIFACT_ROOTS + SOURCE_EVIDENCE:
        copy_path(rel, packet_dir, missing, copied)

    write_readme(packet_dir, packet_name, missing)
    rel_files = write_metadata(packet_dir, packet_name, missing)
    zip_path = zip_packet(packet_dir, packet_name)
    summary = validate(packet_dir, packet_name, zip_path, rel_files)
    summary["staging_dir"] = repo_rel(packet_dir)
    summary["copied_file_count_before_metadata"] = len(copied)
    summary["missing_expected_paths"] = missing
    (PACK_ROOT / f"{packet_name}.build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not keep_staging:
        shutil.rmtree(packet_dir)
        summary["staging_removed"] = True
        (PACK_ROOT / f"{packet_name}.build_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        summary["staging_removed"] = False
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="stream4d_v79_cmap_af_l2h_core_audit")
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()
    summary = build(args.prefix, args.keep_staging)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
