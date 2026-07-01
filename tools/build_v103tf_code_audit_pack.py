#!/usr/bin/env python3
"""Build a compact ACL2 v103 code/evidence audit packet."""

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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "code_audit_pack"
PACK_PREFIX = "acl2_v103_semantic_geometric_evidence_core_audit"
RESULT_ROOT = ROOT / "results/acl2_v103tf_semantic_geometric_evidence_eligibility_readswa_ttt_memory_control"

DOC_FILES = [
    "docs/ACL2_v103TF_SemanticGeometricEvidenceEligibility_READSWA_TTT_MemoryControl_ExperimentPlan_v2.md",
    "docs/ACL2_v103TF_SemanticGeometricEvidenceEligibility_READSWA_TTT_MemoryControl_执行日志.md",
    "docs/ACL2_v103TF_SemanticGeometricEvidenceEligibility_READSWA_TTT_MemoryControl_实验结果复盘.md",
]

CODE_FILES = [
    "tools/build_v103tf_semantic_geometric_evidence_eligibility_readswa_ttt_memory_control.py",
    "tools/build_v103tf_code_audit_pack.py",
]

EXCLUDED_SUFFIXES = {
    ".avi",
    ".ckpt",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
}

EXCLUDED_NAMES = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "code_audit_pack",
    "run.log",
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


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}: {exc}", "path": rel(path)}
    return data if isinstance(data, dict) else {"value": data}


def should_exclude(path: Path) -> bool:
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    if path.name in EXCLUDED_NAMES:
        return True
    return bool(set(path.parts) & EXCLUDED_NAMES)


def add_file(
    src: Path,
    dst_rel: str,
    pack_dir: Path,
    manifest: list[dict[str, Any]],
    *,
    category: str,
) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    if should_exclude(src):
        raise ValueError(f"excluded source requested: {src}")
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


def copy_docs(pack_dir: Path, manifest: list[dict[str, Any]]) -> list[str]:
    copied: list[str] = []
    for item in DOC_FILES:
        add_file(ROOT / item, item, pack_dir, manifest, category="doc")
        copied.append(item)
    return copied


def copy_code(pack_dir: Path, manifest: list[dict[str, Any]]) -> list[str]:
    copied: list[str] = []
    for item in CODE_FILES:
        add_file(ROOT / item, item, pack_dir, manifest, category="code")
        copied.append(item)
    return copied


def copy_artifacts(pack_dir: Path, manifest: list[dict[str, Any]]) -> list[str]:
    if not RESULT_ROOT.is_dir():
        raise FileNotFoundError(RESULT_ROOT)
    copied: list[str] = []
    for src in sorted(p for p in RESULT_ROOT.rglob("*") if p.is_file()):
        if should_exclude(src):
            continue
        rel_src = rel(src)
        add_file(src, "artifacts/" + rel_src, pack_dir, manifest, category="artifact")
        copied.append(rel_src)
    return copied


def write_manifest(pack_dir: Path, manifest: list[dict[str, Any]]) -> None:
    keys = ["category", "source_path", "archive_path", "size_bytes", "sha256"]
    lines = ["\t".join(keys)]
    for row in sorted(manifest, key=lambda item: str(item["archive_path"])):
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
    stage0 = read_json(RESULT_ROOT / "stage0_evidence_ledger/stage0_summary.json")
    stage1 = read_json(RESULT_ROOT / "stage1_focused_drift_source_case_preparation/stage1_summary.json")
    stage2 = read_json(RESULT_ROOT / "stage2_evidence_eligibility_feature_materialization/stage2_summary.json")
    stage3 = read_json(
        RESULT_ROOT / "stage3_branch_b_semantic_correspondence_oracle/stage3_branch_b_summary.json"
    )
    metadata_dir = pack_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    relevant = sorted(set(code_files + doc_files))
    (metadata_dir / "SOURCE_GIT_HEAD.txt").write_text(run_text(["git", "rev-parse", "HEAD"]), encoding="utf-8")
    (metadata_dir / "SOURCE_GIT_STATUS_SHORT.txt").write_text(
        run_text(["git", "status", "--short"]),
        encoding="utf-8",
    )
    (metadata_dir / "SOURCE_GIT_STATUS_RELEVANT.txt").write_text(
        run_text(["git", "status", "--short", "--", *relevant]),
        encoding="utf-8",
    )
    (metadata_dir / "SOURCE_GIT_DIFF_RELEVANT.patch").write_text(
        run_text(["git", "diff", "--", *relevant]),
        encoding="utf-8",
    )
    (metadata_dir / "BUILD_COMMANDS.txt").write_text(
        "\n".join(
            [
                "python3 -m py_compile tools/build_v103tf_semantic_geometric_evidence_eligibility_readswa_ttt_memory_control.py",
                "python3 tools/build_v103tf_semantic_geometric_evidence_eligibility_readswa_ttt_memory_control.py",
                "python3 -m py_compile tools/build_v103tf_code_audit_pack.py",
                "python3 tools/build_v103tf_code_audit_pack.py",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    readme = f"""# ACL2 v103 Core Code/Evidence Audit Packet

Scope:
- v103 plan, execution log, and experiment retrospective log
- v103 artifact-backed evidence eligibility builder
- v103 code-audit packaging script
- lightweight v103 result artifacts under results/acl2_v103tf_semantic_geometric_evidence_eligibility_readswa_ttt_memory_control
- package metadata, manifests, payload checksums, selected git state, and validation sidecars

Current decision snapshot:
- final_taxonomy: {final_decision.get("final_taxonomy")}
- goal_achieved: {final_decision.get("goal_achieved")}
- full_method_success: {final_decision.get("full_method_success")}
- runtime_action_allowed: {final_decision.get("runtime_action_allowed")}
- stage0_pass: {final_decision.get("stage0_pass")}
- stage1_pass: {final_decision.get("stage1_pass")}
- stage2_feature_materialization_pass: {final_decision.get("stage2_feature_materialization_pass")}
- stage3_branch_b_oracle_pass: {final_decision.get("stage3_branch_b_oracle_pass")}

Not included:
- raw datasets
- checkpoints or model binaries
- tensor dumps such as .pt/.npy/.npz
- run logs, caches, or Python cache directories
- previous code_audit_pack archives
- unrelated dirty workspace files

Counts:
- docs: {len(doc_files)}
- code files: {len(code_files)}
- artifact files: {len(artifact_files)}
"""
    (pack_dir / "PACK_README.md").write_text(readme, encoding="utf-8")
    validation = {
        "schema": "acl2_v103_code_audit_pack_pre_zip_v1",
        "final_decision": final_decision,
        "stage0_summary": stage0,
        "stage1_summary": stage1,
        "stage2_summary": stage2,
        "stage3_branch_b_summary": stage3,
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


def validate_zip(zip_path: Path, expected_entries: list[str], pack_name: str) -> dict[str, Any]:
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
        or bool(set(Path(name).parts) & EXCLUDED_NAMES)
    ]
    with tempfile.TemporaryDirectory(prefix="acl2_v103_pack_extract_") as tmp:
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
    manifest: list[dict[str, Any]] = []
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
                "schema": "acl2_v103_code_audit_pack_zip_validation_v1",
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
