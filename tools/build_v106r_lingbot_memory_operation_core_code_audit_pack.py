#!/usr/bin/env python3
"""Build a source-only ACL2 v106R LingBot memory-operation audit packet."""

from __future__ import annotations

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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "code_audit_pack"
PACK_PREFIX = "acl2_v106r_lingbot_memory_operation_core_code"
SELF = Path("tools/build_v106r_lingbot_memory_operation_core_code_audit_pack.py")

SOURCE_FILES = [
    "tools/build_v106r_stage0_v105_evidence_freeze.py",
    "tools/build_v106r_stage1_memory_operation_map.py",
    "tools/build_v106r_stage1_targeted_trace_configs.py",
    "tools/run_v106r_stage1_targeted_trace_manifest.py",
    "tools/build_v106r_stage1_targeted_trace_summary.py",
    "tools/finalize_v106r_stage1_no_memory_lever.py",
    "tools/build_v105tf_lingbot_stage3_oracle.py",
    "third_party/lingbot-map/benchmark/prepare.py",
    "third_party/lingbot-map/benchmark/run_worker.py",
    "third_party/lingbot-map/benchmark/methods/lingbot_map.py",
    "third_party/lingbot-map/benchmark/io/image.py",
    "third_party/lingbot-map/benchmark/io/intrinsics.py",
    "third_party/lingbot-map/lingbot_map/heads/camera_head.py",
    "third_party/lingbot-map/lingbot_map/layers/attention.py",
    "third_party/lingbot-map/lingbot_map/layers/flashinfer_cache.py",
    "third_party/lingbot-map/lingbot_map/models/gct_base.py",
    "third_party/lingbot-map/lingbot_map/models/gct_stream_window_v2.py",
    SELF.as_posix(),
]

EXCLUDED_SUFFIXES = {
    ".avi",
    ".bin",
    ".ckpt",
    ".exr",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".tar",
    ".tgz",
}

EXCLUDED_NAMES = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "checkpoints",
    "code_audit_pack",
    "raw_trace",
    "results",
    "wandb",
    "workspace",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        output = getattr(exc, "output", "")
        return f"{type(exc).__name__}: {exc}\n{output}"


def should_exclude(path: Path) -> bool:
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return bool(set(path.parts) & EXCLUDED_NAMES)


def add_file(src_rel: str, pack_dir: Path, manifest: list[dict[str, Any]]) -> None:
    src = ROOT / src_rel
    if not src.is_file():
        manifest.append(
            {
                "source_path": src_rel,
                "archive_path": "",
                "size_bytes": 0,
                "sha256": "",
                "status": "missing",
            }
        )
        return
    if should_exclude(Path(src_rel)):
        raise ValueError(f"excluded source requested for code-only packet: {src_rel}")
    dst = pack_dir / src_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "source_path": src_rel,
            "archive_path": src_rel,
            "size_bytes": dst.stat().st_size,
            "sha256": sha256(dst),
            "status": "included",
        }
    )


def write_manifest(pack_dir: Path, manifest: list[dict[str, Any]]) -> None:
    keys = ["source_path", "archive_path", "size_bytes", "sha256", "status"]
    rows = ["\t".join(keys)]
    for row in sorted(manifest, key=lambda item: str(item["source_path"])):
        rows.append("\t".join(str(row.get(key, "")) for key in keys))
    (pack_dir / "PACK_MANIFEST.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_metadata(pack_dir: Path, manifest: list[dict[str, Any]]) -> None:
    metadata = pack_dir / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    included = sorted(row["source_path"] for row in manifest if row["status"] == "included")
    lingbot_included = sorted(
        item.removeprefix("third_party/lingbot-map/")
        for item in included
        if item.startswith("third_party/lingbot-map/")
    )
    (metadata / "SOURCE_GIT_HEAD.txt").write_text(run_text(["git", "rev-parse", "HEAD"]), encoding="utf-8")
    (metadata / "SOURCE_GIT_STATUS_RELEVANT.txt").write_text(
        run_text(["git", "status", "--short", "--", *included]),
        encoding="utf-8",
    )
    (metadata / "SOURCE_GIT_DIFF_RELEVANT.patch").write_text(
        run_text(["git", "diff", "--", *included]),
        encoding="utf-8",
    )
    (metadata / "LINGBOT_GIT_HEAD.txt").write_text(
        run_text(["git", "-C", "third_party/lingbot-map", "rev-parse", "HEAD"]),
        encoding="utf-8",
    )
    (metadata / "LINGBOT_GIT_STATUS_RELEVANT.txt").write_text(
        run_text(["git", "-C", "third_party/lingbot-map", "status", "--short", "--", *lingbot_included]),
        encoding="utf-8",
    )
    (metadata / "LINGBOT_GIT_DIFF_RELEVANT.patch").write_text(
        run_text(["git", "-C", "third_party/lingbot-map", "diff", "--", *lingbot_included]),
        encoding="utf-8",
    )
    missing = [row["source_path"] for row in manifest if row["status"] != "included"]
    readme = f"""# ACL2 v106R LingBot Memory Operation Source-Only Audit Packet

Scope:
- Source code only.
- No docs, logs, result artifacts, configs, raw traces, workspaces, checkpoints, or data.
- Includes v106R experiment builders/runners/summarizers/finalizer, the v105 semantic-loader dependency used by v106R summary, and LingBot trace/cache core source files.

Included source files: {len(included)}
Missing requested source files: {len(missing)}

Missing list:
{json.dumps(missing, ensure_ascii=False, indent=2)}

Validation files outside the zip:
- `<archive>.zip.sha256`
- `<archive>.zip.sha256_check.txt`
- `<archive>.zip.payload_sha256_check.txt`
- `<archive>.zip.unzip_test.txt`
- `<archive>.zip.entry_parity.txt`
- `<archive>.zip.exclusion_check.txt`
- `<archive>.zip.summary.json`
"""
    (pack_dir / "PACK_README.md").write_text(readme, encoding="utf-8")
    commands = [
        "python -m py_compile tools/build_v106r_lingbot_memory_operation_core_code_audit_pack.py",
        "python tools/build_v106r_lingbot_memory_operation_core_code_audit_pack.py",
        "cd code_audit_pack && sha256sum -c <archive>.zip.sha256",
        "unzip -tq code_audit_pack/<archive>.zip",
        "cd <extracted-packet-root> && sha256sum -c PAYLOAD_SHA256SUMS.txt",
    ]
    (metadata / "BUILD_AND_VERIFY_COMMANDS.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")


def write_payload_hashes(pack_dir: Path) -> list[str]:
    filelist = pack_dir / "PAYLOAD_FILELIST.txt"
    hashes = pack_dir / "PAYLOAD_SHA256SUMS.txt"
    hashes.write_text("", encoding="utf-8")
    files = sorted(path.relative_to(pack_dir).as_posix() for path in pack_dir.rglob("*") if path.is_file())
    filelist.write_text("\n".join(files) + "\n", encoding="utf-8")
    files = sorted(path.relative_to(pack_dir).as_posix() for path in pack_dir.rglob("*") if path.is_file())
    hash_files = [name for name in files if name != "PAYLOAD_SHA256SUMS.txt"]
    hashes.write_text("\n".join(f"{sha256(pack_dir / name)}  {name}" for name in hash_files) + "\n", encoding="utf-8")
    return sorted(path.relative_to(pack_dir).as_posix() for path in pack_dir.rglob("*") if path.is_file())


def create_zip(pack_dir: Path, zip_path: Path) -> list[str]:
    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for src in sorted(path for path in pack_dir.rglob("*") if path.is_file()):
            arcname = src.relative_to(pack_dir).as_posix()
            archive.write(src, arcname)
            entries.append(arcname)
    return entries


def check_payload_hashes(zip_path: Path, sidecar: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="acl2_v106r_code_pack_verify_") as tmp:
        tmpdir = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmpdir)
        proc = subprocess.run(
            ["sha256sum", "-c", "PAYLOAD_SHA256SUMS.txt"],
            cwd=tmpdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        sidecar.write_text(proc.stdout, encoding="utf-8")
        return proc.returncode


def check_archive_sha(sha_sidecar: Path, sidecar: Path) -> int:
    proc = subprocess.run(
        ["sha256sum", "-c", sha_sidecar.name],
        cwd=PACK_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    sidecar.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode


def check_unzip(zip_path: Path, sidecar: Path) -> int:
    proc = subprocess.run(
        ["unzip", "-tq", str(zip_path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    sidecar.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode


def check_entry_parity(payload_files: list[str], zip_entries: list[str], sidecar: Path) -> int:
    payload_set = set(payload_files)
    zip_set = set(zip_entries)
    data = {
        "payload_file_count": len(payload_files),
        "zip_entry_count": len(zip_entries),
        "missing_from_zip": sorted(payload_set - zip_set),
        "extra_in_zip": sorted(zip_set - payload_set),
    }
    data["entry_parity_ok"] = not data["missing_from_zip"] and not data["extra_in_zip"]
    sidecar.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if data["entry_parity_ok"] else 1


def check_exclusions(zip_entries: list[str], sidecar: Path) -> int:
    violations: list[str] = []
    for entry in zip_entries:
        path = Path(entry)
        if should_exclude(path):
            violations.append(entry)
        if entry.startswith(("docs/", "results/", "code_audit_pack/", "configs/")):
            violations.append(entry)
    data = {
        "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
        "excluded_names": sorted(EXCLUDED_NAMES),
        "violation_count": len(sorted(set(violations))),
        "violations": sorted(set(violations)),
        "exclusion_check_ok": not violations,
    }
    sidecar.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if data["exclusion_check_ok"] else 1


def build() -> dict[str, Any]:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = os.environ.get("V106R_CODE_PACK_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = PACK_ROOT / f"{PACK_PREFIX}_{timestamp}.zip"
    with tempfile.TemporaryDirectory(prefix=f"{PACK_PREFIX}_") as tmp:
        pack_dir = Path(tmp) / f"{PACK_PREFIX}_{timestamp}"
        pack_dir.mkdir(parents=True)
        manifest: list[dict[str, Any]] = []
        for src_rel in SOURCE_FILES:
            add_file(src_rel, pack_dir, manifest)
        write_manifest(pack_dir, manifest)
        write_metadata(pack_dir, manifest)
        payload_files = write_payload_hashes(pack_dir)
        zip_entries = create_zip(pack_dir, zip_path)

    sha_sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sha_sidecar.write_text(f"{sha256(zip_path)}  {zip_path.name}\n", encoding="utf-8")
    sha_check = zip_path.with_suffix(zip_path.suffix + ".sha256_check.txt")
    payload_check = zip_path.with_suffix(zip_path.suffix + ".payload_sha256_check.txt")
    unzip_test = zip_path.with_suffix(zip_path.suffix + ".unzip_test.txt")
    entry_parity = zip_path.with_suffix(zip_path.suffix + ".entry_parity.txt")
    exclusion_check = zip_path.with_suffix(zip_path.suffix + ".exclusion_check.txt")
    sha_rc = check_archive_sha(sha_sidecar, sha_check)
    payload_rc = check_payload_hashes(zip_path, payload_check)
    unzip_rc = check_unzip(zip_path, unzip_test)
    parity_rc = check_entry_parity(payload_files, zip_entries, entry_parity)
    exclusion_rc = check_exclusions(zip_entries, exclusion_check)
    summary = {
        "schema": "acl2_v106r_core_code_audit_pack_summary_v1",
        "archive": zip_path.relative_to(ROOT).as_posix(),
        "archive_size_bytes": zip_path.stat().st_size,
        "archive_sha256": sha256(zip_path),
        "payload_file_count": len(payload_files),
        "zip_entry_count": len(zip_entries),
        "archive_sha256_check_returncode": sha_rc,
        "payload_sha256_check_returncode": payload_rc,
        "unzip_test_returncode": unzip_rc,
        "entry_parity_returncode": parity_rc,
        "exclusion_check_returncode": exclusion_rc,
        "validation_ok": sha_rc == 0 and payload_rc == 0 and unzip_rc == 0 and parity_rc == 0 and exclusion_rc == 0,
        "sidecars": [
            sha_sidecar.relative_to(ROOT).as_posix(),
            sha_check.relative_to(ROOT).as_posix(),
            payload_check.relative_to(ROOT).as_posix(),
            unzip_test.relative_to(ROOT).as_posix(),
            entry_parity.relative_to(ROOT).as_posix(),
            exclusion_check.relative_to(ROOT).as_posix(),
        ],
        "source_file_count_requested": len(SOURCE_FILES),
    }
    summary_path = zip_path.with_suffix(zip_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (PACK_ROOT / ".latest_acl2_v106r_code_pack_tag").write_text(zip_path.stem + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["validation_ok"]:
        raise SystemExit(1)
    return summary


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
