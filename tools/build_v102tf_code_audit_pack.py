#!/usr/bin/env python3
"""Build a compact ACL2 v102 code/evidence audit packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "code_audit_pack"
PACK_PREFIX = "acl2_v102_drift_source_autopsy_semantic_oracle_core_audit"
RESULT_ROOT = ROOT / "results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control"

DOC_FILES = [
    "docs/ACL2_v102TF_DriftSourceAutopsy_SemanticOracleMemoryControl_ExperimentPlan_v2.md",
    "docs/ACL2_v102TF_DriftSourceAutopsy_SemanticOracleMemoryControl_执行日志.md",
    "docs/ACL2_v102TF_DriftSourceAutopsy_SemanticOracleMemoryControl_实验结果复盘.md",
]

MANDATORY_CODE_FILES = [
    "tools/build_v102tf_drift_source_autopsy_semantic_oracle_memory_control.py",
    "tools/build_v102tf_rgb_semantic_overlay_panels.py",
    "tools/build_v102tf_trajectory_error_overlay_panels.py",
    "tools/build_v102tf_local_point_residual_overlay_panels.py",
    "tools/audit_v102tf_stage3_local_geometry_oracle_repair.py",
    "tools/audit_v102tf_stage3_full_control_semantic_rotation.py",
    "tools/audit_v102tf_stage3_clean_handoff_candidate_expansion.py",
    "tools/audit_v102tf_broader_drift_onset_trace_extension.py",
    "tools/audit_v102tf_read_local_oracle_bridge.py",
    "tools/audit_v102tf_legacy_cue_case_alignment.py",
    "tools/audit_v102tf_historical_clean_target_extension.py",
    "tools/audit_v102tf_strict_clean_handoff_materialization_repair.py",
    "tools/audit_v102tf_exhaustive_clean_handoff_target_mining.py",
    "tools/audit_v102tf_action_surface_true_l3_upper_bound_feasibility.py",
    "tools/audit_v102tf_state_machine_hook_readiness.py",
    "tools/build_v102tf_state_machine_scaffold_trace_targets.py",
    "tools/audit_v102tf_state_machine_scaffold_trace_closure.py",
    "tools/audit_v102tf_state_machine_action_probe_closure.py",
    "tools/evaluate_v102tf_state_machine_trace_runs.py",
    "tools/run_v101tf_stage_c_seed_bridge_target_traces.py",
    "tools/build_v102tf_code_audit_pack.py",
    "run_pipeline_abc_v2.py",
    "run_geometry_backbone_inference.py",
    "loger/models/layers/attention.py",
    "loger/models/layers/block.py",
    "loger/models/pi3.py",
    "loger/pipeline/geometry_backbone.py",
    "loger/pipeline/hybrid_memory_controller.py",
    "loger/pipeline/semantic_prior_generator.py",
    "loger/pipeline/ttt_write_controller.py",
]

CODE_GLOBS = [
    "tools/build_v96tf_*.py",
    "tools/build_v97tf_*.py",
    "tools/build_v98tf_*.py",
    "tools/build_v99tf_*.py",
    "tools/build_v100tf_*.py",
    "tools/build_v101tf_*.py",
    "tools/build_v102tf_*.py",
    "tools/audit_v100tf_*.py",
    "tools/audit_v101tf_*.py",
    "tools/audit_v102tf_*.py",
    "tools/run_v96tf_*.py",
    "tools/run_v98tf_*.py",
    "tools/run_v101tf_*.py",
]

EXCLUDED_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".mp4",
    ".mov",
    ".avi",
}

EXCLUDED_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "run.log",
    "stage4_memory_action_probe_hold_prev_reference_soft2_v1",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        return exc.output


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}: {exc}", "path": str(path)}
    return payload if isinstance(payload, dict) else {"value": payload}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def should_exclude(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_NAMES:
        return True
    if path.name in EXCLUDED_NAMES:
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return False


def add_file(
    src: Path,
    dst_rel: str,
    pack_dir: Path,
    manifest: list[dict[str, object]],
    *,
    category: str,
) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = pack_dir / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "category": category,
            "source_path": rel(src),
            "archive_path": dst_rel,
            "size_bytes": dst.stat().st_size,
            "sha256": sha256(dst),
        }
    )


def collect_code_files() -> list[str]:
    files = set(MANDATORY_CODE_FILES)
    for pattern in CODE_GLOBS:
        for path in ROOT.glob(pattern):
            if path.is_file() and not should_exclude(path):
                files.add(rel(path))
    return sorted(files)


def copy_docs(pack_dir: Path, manifest: list[dict[str, object]]) -> list[str]:
    copied: list[str] = []
    for item in DOC_FILES:
        src = ROOT / item
        add_file(src, item, pack_dir, manifest, category="doc")
        copied.append(item)
    return copied


def copy_code(pack_dir: Path, manifest: list[dict[str, object]]) -> list[str]:
    copied: list[str] = []
    missing: list[str] = []
    for item in collect_code_files():
        src = ROOT / item
        if item in MANDATORY_CODE_FILES and not src.is_file():
            missing.append(item)
            continue
        if not src.is_file():
            continue
        add_file(src, item, pack_dir, manifest, category="code")
        copied.append(item)
    if missing:
        raise FileNotFoundError("missing mandatory code files: " + ", ".join(missing))
    return copied


def copy_artifacts(pack_dir: Path, manifest: list[dict[str, object]]) -> list[str]:
    if not RESULT_ROOT.is_dir():
        raise FileNotFoundError(RESULT_ROOT)
    copied: list[str] = []
    for src in sorted(p for p in RESULT_ROOT.rglob("*") if p.is_file()):
        rel_src = rel(src)
        if should_exclude(src):
            continue
        dst_rel = "artifacts/" + rel_src
        add_file(src, dst_rel, pack_dir, manifest, category="artifact")
        copied.append(rel_src)
    return copied


def write_manifest(pack_dir: Path, manifest: list[dict[str, object]]) -> None:
    keys = ["category", "source_path", "archive_path", "size_bytes", "sha256"]
    lines = ["\t".join(keys)]
    for row in sorted(manifest, key=lambda r: str(r["archive_path"])):
        lines.append("\t".join(str(row[key]) for key in keys))
    manifest_path = pack_dir / "PACK_MANIFEST.tsv"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (pack_dir / "PACK_MANIFEST.sha256").write_text(
        f"{sha256(manifest_path)}  PACK_MANIFEST.tsv\n",
        encoding="utf-8",
    )


def write_payload_hashes(pack_dir: Path) -> list[str]:
    files = sorted(
        p.relative_to(pack_dir).as_posix()
        for p in pack_dir.rglob("*")
        if p.is_file() and p.name != "PAYLOAD_SHA256SUMS.txt"
    )
    (pack_dir / "PAYLOAD_FILELIST.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
    files = sorted(
        p.relative_to(pack_dir).as_posix()
        for p in pack_dir.rglob("*")
        if p.is_file() and p.name != "PAYLOAD_SHA256SUMS.txt"
    )
    (pack_dir / "PAYLOAD_SHA256SUMS.txt").write_text(
        "\n".join(f"{sha256(pack_dir / name)}  {name}" for name in files) + "\n",
        encoding="utf-8",
    )
    return sorted(p.relative_to(pack_dir).as_posix() for p in pack_dir.rglob("*") if p.is_file())


def write_metadata(pack_dir: Path, code_files: list[str], artifact_files: list[str], doc_files: list[str]) -> None:
    final_decision = read_json(RESULT_ROOT / "final_decision/final_decision.json")
    completion = read_json(RESULT_ROOT / "final_decision/completion_audit_summary.json")
    metadata_dir = pack_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "SOURCE_GIT_HEAD.txt").write_text(run_text(["git", "rev-parse", "HEAD"]), encoding="utf-8")
    (metadata_dir / "SOURCE_GIT_STATUS_SHORT.txt").write_text(
        run_text(["git", "status", "--short"]),
        encoding="utf-8",
    )
    relevant = sorted(set(code_files + doc_files))
    (metadata_dir / "SOURCE_GIT_STATUS_RELEVANT.txt").write_text(
        run_text(["git", "status", "--short", "--", *relevant]) if relevant else "",
        encoding="utf-8",
    )
    (metadata_dir / "SOURCE_GIT_DIFF_RELEVANT.patch").write_text(
        run_text(["git", "diff", "--", *relevant]) if relevant else "",
        encoding="utf-8",
    )
    (metadata_dir / "BUILD_COMMANDS.txt").write_text(
        "\n".join(
            [
                "python3 -m py_compile tools/build_v102tf_drift_source_autopsy_semantic_oracle_memory_control.py",
                "python3 -m py_compile tools/build_v102tf_rgb_semantic_overlay_panels.py",
                "python3 -m py_compile tools/build_v102tf_trajectory_error_overlay_panels.py",
                "python3 -m py_compile tools/build_v102tf_local_point_residual_overlay_panels.py",
                "python3 -m py_compile tools/audit_v102tf_stage3_local_geometry_oracle_repair.py",
                "python3 -m py_compile tools/audit_v102tf_stage3_full_control_semantic_rotation.py",
                "python3 -m py_compile tools/audit_v102tf_stage3_clean_handoff_candidate_expansion.py",
                "python3 -m py_compile tools/audit_v102tf_broader_drift_onset_trace_extension.py",
                "python3 -m py_compile tools/audit_v102tf_read_local_oracle_bridge.py",
                "python3 -m py_compile tools/audit_v102tf_legacy_cue_case_alignment.py",
                "python3 -m py_compile tools/audit_v102tf_historical_clean_target_extension.py",
                "python3 -m py_compile tools/audit_v102tf_strict_clean_handoff_materialization_repair.py",
                "python3 -m py_compile tools/audit_v102tf_exhaustive_clean_handoff_target_mining.py",
                "python3 -m py_compile tools/audit_v102tf_action_surface_true_l3_upper_bound_feasibility.py",
                "python3 -m py_compile tools/audit_v102tf_state_machine_hook_readiness.py",
                "python3 -m py_compile tools/build_v102tf_state_machine_scaffold_trace_targets.py",
                "python3 -m py_compile tools/audit_v102tf_state_machine_scaffold_trace_closure.py",
                "python3 -m py_compile tools/audit_v102tf_state_machine_action_probe_closure.py",
                "python3 -m py_compile tools/audit_v102tf_ttt_write_to_use_chain_closure.py",
                "python3 -m py_compile tools/evaluate_v102tf_state_machine_trace_runs.py",
                "python3 -m py_compile tools/run_v101tf_stage_c_seed_bridge_target_traces.py",
                "python3 tools/build_v102tf_drift_source_autopsy_semantic_oracle_memory_control.py",
                "/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/build_v102tf_rgb_semantic_overlay_panels.py",
                "python3 tools/build_v102tf_trajectory_error_overlay_panels.py",
                "/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/build_v102tf_local_point_residual_overlay_panels.py",
                "python3 tools/audit_v102tf_stage3_local_geometry_oracle_repair.py",
                "python3 tools/audit_v102tf_stage3_full_control_semantic_rotation.py",
                "python3 tools/audit_v102tf_stage3_clean_handoff_candidate_expansion.py",
                "/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/audit_v102tf_broader_drift_onset_trace_extension.py",
                "python3 tools/audit_v102tf_read_local_oracle_bridge.py",
                "python3 tools/audit_v102tf_legacy_cue_case_alignment.py",
                "python3 tools/audit_v102tf_historical_clean_target_extension.py",
                "python3 tools/audit_v102tf_strict_clean_handoff_materialization_repair.py",
                "python3 tools/audit_v102tf_exhaustive_clean_handoff_target_mining.py",
                "python3 tools/audit_v102tf_action_surface_true_l3_upper_bound_feasibility.py",
                "python3 tools/audit_v102tf_state_machine_hook_readiness.py",
                "python3 tools/build_v102tf_state_machine_scaffold_trace_targets.py",
                "/mnt/data/users/chengshun.wang/miniconda3/bin/conda run --no-capture-output -n loger python tools/audit_v102tf_state_machine_scaffold_trace_closure.py",
                "/mnt/data/users/chengshun.wang/miniconda3/bin/conda run --no-capture-output -n loger python tools/audit_v102tf_state_machine_action_probe_closure.py",
                "python3 tools/audit_v102tf_ttt_write_to_use_chain_closure.py",
                "python3 tools/evaluate_v102tf_state_machine_trace_runs.py",
                "python3 tools/build_v102tf_drift_source_autopsy_semantic_oracle_memory_control.py",
                "python3 -m py_compile tools/build_v102tf_code_audit_pack.py",
                "python3 tools/build_v102tf_code_audit_pack.py",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    readme = f"""# ACL2 v102 Core Audit Packet

Scope:
- v102 plan, execution log, retrospective log
- v102 drift-source/semantic-oracle builder and related ACL2 memory-control source
- v102 lightweight result artifacts under results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control
- package metadata, manifests, checksums, and validation sidecars

Final decision snapshot:
- final_taxonomy: {final_decision.get("final_taxonomy")}
- goal_achieved: {final_decision.get("goal_achieved")}
- runtime_action_allowed: {final_decision.get("runtime_action_allowed")}
- full_method_success: {final_decision.get("full_method_success")}
- passed_requirements: {completion.get("passed_requirement_count")}/{completion.get("requirement_count")}
- failed_requirements: {completion.get("failed_requirement_count")}

Not included:
- raw datasets
- checkpoints or model binaries
- tensor dumps such as .pt/.npy/.npz
- run.log files and caches
- previous code_audit_pack archives
- aborted wrong-root hold-prev-reference soft2 scratch directory

Counts:
- docs: {len(doc_files)}
- code files: {len(code_files)}
- artifact files: {len(artifact_files)}
"""
    (pack_dir / "PACK_README.md").write_text(readme, encoding="utf-8")
    validation = {
        "schema": "acl2_v102_code_audit_pack_pre_zip_v1",
        "final_decision": final_decision,
        "completion_audit": completion,
        "doc_file_count": len(doc_files),
        "code_file_count": len(code_files),
        "artifact_file_count": len(artifact_files),
        "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
        "excluded_names": sorted(EXCLUDED_NAMES),
    }
    (pack_dir / "PACK_VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_zip(pack_dir: Path, zip_path: Path) -> list[str]:
    root_name = pack_dir.name
    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src in sorted(p for p in pack_dir.rglob("*") if p.is_file()):
            arcname = f"{root_name}/{src.relative_to(pack_dir).as_posix()}"
            zf.write(src, arcname)
            entries.append(arcname)
    return entries


def verify_payload_hashes(extract_root: Path, pack_name: str) -> tuple[bool, list[str]]:
    root = extract_root / pack_name
    sums = root / "PAYLOAD_SHA256SUMS.txt"
    failures: list[str] = []
    if not sums.is_file():
        return False, ["missing PAYLOAD_SHA256SUMS.txt"]
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        path = root / name
        if not path.is_file():
            failures.append(f"missing:{name}")
        elif sha256(path) != digest:
            failures.append(f"sha256_mismatch:{name}")
    return not failures, failures


def validate_zip(zip_path: Path, expected_entries: list[str], pack_name: str) -> dict[str, object]:
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        entries = sorted(name for name in zf.namelist() if not name.endswith("/"))
    missing = sorted(set(expected_entries) - set(entries))
    extra = sorted(set(entries) - set(expected_entries))
    excluded_hits = [
        name
        for name in entries
        if Path(name).suffix.lower() in EXCLUDED_SUFFIXES
        or Path(name).name in EXCLUDED_NAMES
        or set(Path(name).parts) & EXCLUDED_NAMES
        or "/code_audit_pack/" in name
    ]
    with tempfile.TemporaryDirectory(prefix="acl2_v102_pack_extract_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)
        payload_hash_ok, payload_hash_failures = verify_payload_hashes(tmp_path, pack_name)
    return {
        "zip_test_ok": bad is None,
        "bad_zip_member": bad,
        "entry_count": len(entries),
        "expected_entry_count": len(expected_entries),
        "entry_parity_ok": not missing and not extra,
        "missing_entries": missing,
        "extra_entries": extra,
        "payload_hash_ok": payload_hash_ok,
        "payload_hash_failures": payload_hash_failures,
        "excluded_path_check_ok": not excluded_hits,
        "excluded_path_hits": excluded_hits,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="", help="Optional exact pack directory/archive stem.")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing pack with the same name.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pack_name = args.name or f"{PACK_PREFIX}_{timestamp}"
    pack_dir = PACK_ROOT / pack_name
    zip_path = PACK_ROOT / f"{pack_name}.zip"
    if (pack_dir.exists() or zip_path.exists()) and not args.overwrite:
        raise FileExistsError(f"{pack_name} already exists; use --overwrite or a new --name")
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    if zip_path.exists():
        zip_path.unlink()
    pack_dir.mkdir(parents=True)
    manifest: list[dict[str, object]] = []
    try:
        docs = copy_docs(pack_dir, manifest)
        code = copy_code(pack_dir, manifest)
        artifacts = copy_artifacts(pack_dir, manifest)
        write_metadata(pack_dir, code, artifacts, docs)
        write_manifest(pack_dir, manifest)
        payload_entries = write_payload_hashes(pack_dir)
        expected_zip_entries = [f"{pack_name}/{name}" for name in payload_entries]
        make_zip(pack_dir, zip_path)
        archive_sha = sha256(zip_path)
        validation = validate_zip(zip_path, expected_zip_entries, pack_name)
        validation.update(
            {
                "schema": "acl2_v102_code_audit_pack_zip_validation_v1",
                "pack_dir": pack_dir.as_posix(),
                "archive": zip_path.as_posix(),
                "archive_sha256": archive_sha,
                "archive_size_bytes": zip_path.stat().st_size,
                "doc_file_count": len(docs),
                "code_file_count": len(code),
                "artifact_file_count": len(artifacts),
                "payload_file_count": len(payload_entries),
            }
        )
        (PACK_ROOT / f"{zip_path.name}.sha256").write_text(
            f"{archive_sha}  {zip_path.name}\n",
            encoding="utf-8",
        )
        unzip_text = "OK\n" if validation["zip_test_ok"] else f"BAD {validation['bad_zip_member']}\n"
        (PACK_ROOT / f"{zip_path.name}.unzip_test.txt").write_text(unzip_text, encoding="utf-8")
        payload_text = "OK\n" if validation["payload_hash_ok"] else "\n".join(validation["payload_hash_failures"]) + "\n"
        (PACK_ROOT / f"{zip_path.name}.payload_sha256_check.txt").write_text(payload_text, encoding="utf-8")
        (PACK_ROOT / f"{zip_path.name}.validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ok = (
            validation["zip_test_ok"]
            and validation["entry_parity_ok"]
            and validation["payload_hash_ok"]
            and validation["excluded_path_check_ok"]
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if ok else 2
    except Exception:
        shutil.rmtree(pack_dir, ignore_errors=True)
        if zip_path.exists():
            zip_path.unlink()
        raise


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
