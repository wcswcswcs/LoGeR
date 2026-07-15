#!/usr/bin/env python3
"""Build a compact ACL2 v119/v118 core-code audit packet."""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "code_audit_pack"
PACK_PREFIX = "acl2_v119tf_core_code_audit"
SELF = Path("tools/build_v119tf_core_code_audit_pack.py")

V118_RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
V119_RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
V119_CARRIER_AWARE_RESULT_ROOT = (
    ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_carrier_aware_augmented"
)

SOURCE_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ARTIFACT_SUFFIXES = {".csv", ".json", ".md", ".txt"}
SELECTED_BINARY_ARTIFACTS = {
    "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_carrier_aware_augmented/"
    "V119_CHS_CARRIER_EVIDENCE_ROWS.parquet",
}

EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "cache",
    "checkpoints",
    "code_audit_pack",
    "data",
    "logs",
    "output",
    "outputs",
    "report",
    "runtime_full",
    "runtime_full_thread8",
    "tmp",
    "traces",
    "wandb",
    "workspace",
}

SOURCE_EXCLUDED_PARTS = (EXCLUDED_PARTS - {"artifacts", "data", "report"}) | {"results"}

EXCLUDED_SUFFIXES = {
    ".avi",
    ".bin",
    ".ckpt",
    ".gz",
    ".jsonl",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".tar",
    ".tgz",
    ".zip",
}

MAX_ARTIFACT_BYTES = 1_000_000
SKIPPED_DETAIL_LIMIT = 5_000


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(args: list[str], *, cwd: Path = ROOT, timeout: int = 120) -> str:
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return proc.stdout
    except Exception as exc:  # pragma: no cover - audit capture only
        return f"{type(exc).__name__}: {exc}\n"


def has_excluded_part(path: Path) -> bool:
    return bool(set(path.parts) & EXCLUDED_PARTS)


def has_excluded_source_part(path: Path) -> bool:
    return bool(set(path.parts) & SOURCE_EXCLUDED_PARTS)


def source_allowed(path: Path) -> bool:
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if path.suffix.lower() not in SOURCE_SUFFIXES:
        return False
    return not has_excluded_source_part(path)


def artifact_allowed(path: Path) -> tuple[bool, str]:
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False, "excluded_suffix"
    if path.suffix.lower() not in ARTIFACT_SUFFIXES:
        return False, "not_lightweight_text_artifact"
    if has_excluded_part(path):
        return False, "excluded_runtime_or_bulk_dir"
    size = path.stat().st_size
    if size > MAX_ARTIFACT_BYTES:
        return False, "artifact_size_over_1MB"
    return True, "included"


def add_path(rows: dict[str, dict[str, Any]], category: str, src: Path) -> None:
    src = src.resolve()
    if not src.is_file():
        return
    src_rel = rel(src)
    if src_rel in rows:
        existing = str(rows[src_rel]["category"])
        if category not in existing.split("+"):
            rows[src_rel]["category"] = f"{existing}+{category}"
        return
    rows[src_rel] = {
        "category": category,
        "source_path": src_rel,
        "archive_path": src_rel,
        "size_bytes": src.stat().st_size,
        "sha256": sha256(src),
        "status": "selected",
    }


def add_tree(rows: dict[str, dict[str, Any]], category: str, base: Path, suffixes: set[str]) -> None:
    if not base.exists():
        return
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel_path = Path(rel(path))
        if rel_path.suffix.lower() not in suffixes:
            continue
        if source_allowed(rel_path):
            add_path(rows, category, path)


def collect_docs(rows: dict[str, dict[str, Any]]) -> None:
    for path in sorted((ROOT / "docs").glob("ACL2_v118TF_OperationSpecificSemanticCarrierCalibration*")):
        add_path(rows, "docs_v118", path)
    for path in sorted((ROOT / "docs").glob("ACL2_v119TF_SemanticAddressableGeometryCarrierRouting_RepresentationRepair*")):
        add_path(rows, "docs_v119", path)
    for path in sorted(
        (ROOT / "docs").glob("ACL2_v119TF_SemanticAddressableGeometryCarrierRouting_CarrierAwareAugmented*")
    ):
        add_path(rows, "docs_v119_carrier_aware", path)


def collect_tools(rows: dict[str, dict[str, Any]]) -> None:
    add_path(rows, "packaging_code", ROOT / SELF)
    for path in sorted((ROOT / "tools").glob("*v118tf*")):
        if path.is_file() and source_allowed(Path(rel(path))):
            add_path(rows, "tools_v118tf", path)
    for path in sorted((ROOT / "tools").glob("*v119tf*")):
        if path.is_file() and source_allowed(Path(rel(path))):
            add_path(rows, "tools_v119tf", path)
    for path in [
        "run_geometry_backbone_inference.py",
        "run_pipeline_abc.py",
        "run_pipeline_abc_v2.py",
        "run_video_masklet_front_end.py",
        "run_video_masklet_front_end_v2.py",
        "run_video_masklet_front_end_v3.py",
    ]:
        add_path(rows, "top_level_runners", ROOT / path)


def collect_core_runtime(rows: dict[str, dict[str, Any]]) -> None:
    add_tree(rows, "loger_core", ROOT / "loger", SOURCE_SUFFIXES)
    add_tree(rows, "eval_core", ROOT / "eval", SOURCE_SUFFIXES)

    lingbot_roots = [
        "third_party/lingbot-map/README.md",
        "third_party/lingbot-map/gct_profile.py",
        "third_party/lingbot-map/demo.py",
        "third_party/lingbot-map/benchmark",
        "third_party/lingbot-map/lingbot_map",
    ]
    for item in lingbot_roots:
        path = ROOT / item
        if path.is_dir():
            add_tree(rows, "lingbot_core", path, SOURCE_SUFFIXES)
        else:
            add_path(rows, "lingbot_core", path)

    horizon_roots = [
        "third_party/HorizonStream/README.md",
        "third_party/HorizonStream/configs",
        "third_party/HorizonStream/horizonstream",
        "third_party/HorizonStream/infer.py",
        "third_party/HorizonStream/run_pipeline.py",
        "third_party/HorizonStream/scripts",
    ]
    for item in horizon_roots:
        path = ROOT / item
        if path.is_dir():
            add_tree(rows, "horizonstream_core", path, SOURCE_SUFFIXES)
        else:
            add_path(rows, "horizonstream_core", path)


def record_skip(
    skipped: list[dict[str, Any]],
    skipped_summary: dict[str, dict[str, int]],
    path: str,
    reason: str,
    size_bytes: int,
) -> None:
    bucket = skipped_summary.setdefault(reason, {"count": 0, "bytes": 0})
    bucket["count"] += 1
    bucket["bytes"] += size_bytes
    if len(skipped) < SKIPPED_DETAIL_LIMIT:
        skipped.append({"path": path, "reason": reason, "size_bytes": size_bytes})


def collect_artifacts(rows: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    skipped: list[dict[str, Any]] = []
    skipped_summary: dict[str, dict[str, int]] = {}
    for category, root in [
        ("artifacts_v118_lightweight", V118_RESULT_ROOT),
        ("artifacts_v119_representation_repair_lightweight", V119_RESULT_ROOT),
        ("artifacts_v119_carrier_aware_lightweight", V119_CARRIER_AWARE_RESULT_ROOT),
    ]:
        if not root.exists():
            record_skip(skipped, skipped_summary, rel(root), "result_root_missing", 0)
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel_path = Path(rel(path))
            if rel_path.as_posix() in SELECTED_BINARY_ARTIFACTS and path.stat().st_size <= MAX_ARTIFACT_BYTES:
                add_path(rows, f"{category}_selected_binary", path)
                continue
            ok, reason = artifact_allowed(rel_path)
            if ok:
                add_path(rows, category, path)
            else:
                record_skip(skipped, skipped_summary, rel_path.as_posix(), reason, path.stat().st_size)
    return skipped, skipped_summary


def copy_payload(pack_dir: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        src = ROOT / str(row["source_path"])
        dst = pack_dir / str(row["archive_path"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_manifest(pack_dir: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["category", "source_path", "archive_path", "size_bytes", "sha256", "status"]
    with (pack_dir / "PACK_MANIFEST.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_skipped(pack_dir: Path, skipped: list[dict[str, Any]], skipped_summary: dict[str, dict[str, int]]) -> None:
    fields = ["path", "reason", "size_bytes"]
    with (pack_dir / "metadata/SKIPPED_ARTIFACTS.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(skipped)
        total_skipped = sum(row["count"] for row in skipped_summary.values())
        if total_skipped > len(skipped):
            writer.writerow(
                {
                    "path": "__TRUNCATED__",
                    "reason": f"showing_first_{len(skipped)}_of_{total_skipped}",
                    "size_bytes": 0,
                }
            )
    (pack_dir / "metadata/SKIPPED_ARTIFACTS_SUMMARY.json").write_text(
        json.dumps(skipped_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_metadata(
    pack_dir: Path,
    rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    skipped_summary: dict[str, dict[str, int]],
    created_at: str,
) -> None:
    metadata_dir = pack_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    (metadata_dir / "SOURCE_GIT_HEAD.txt").write_text(run_text(["git", "rev-parse", "HEAD"]), encoding="utf-8")
    (metadata_dir / "SOURCE_GIT_STATUS_FULL.txt").write_text(run_text(["git", "status", "--short"], timeout=240), encoding="utf-8")
    (metadata_dir / "LINGBOT_GIT_HEAD.txt").write_text(
        run_text(["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party/lingbot-map"),
        encoding="utf-8",
    )
    (metadata_dir / "LINGBOT_GIT_STATUS.txt").write_text(
        run_text(["git", "status", "--short"], cwd=ROOT / "third_party/lingbot-map"),
        encoding="utf-8",
    )
    (metadata_dir / "HORIZONSTREAM_GIT_HEAD.txt").write_text(
        run_text(["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party/HorizonStream"),
        encoding="utf-8",
    )
    (metadata_dir / "HORIZONSTREAM_GIT_STATUS.txt").write_text(
        run_text(["git", "status", "--short"], cwd=ROOT / "third_party/HorizonStream"),
        encoding="utf-8",
    )

    category_counts: dict[str, int] = {}
    category_bytes: dict[str, int] = {}
    for row in rows:
        category = str(row["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
        category_bytes[category] = category_bytes.get(category, 0) + int(row["size_bytes"])

    observed_state = {
        "schema": "acl2_v119tf_core_code_audit_pack_v1",
        "created_at_local": created_at,
        "pack_prefix": PACK_PREFIX,
        "source_file_count": len(rows),
        "source_total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "category_counts": category_counts,
        "category_bytes": category_bytes,
        "v118_result_root": rel(V118_RESULT_ROOT),
        "v118_result_root_exists": V118_RESULT_ROOT.exists(),
        "v119_result_root": rel(V119_RESULT_ROOT),
        "v119_result_root_exists": V119_RESULT_ROOT.exists(),
        "v119_carrier_aware_result_root": rel(V119_CARRIER_AWARE_RESULT_ROOT),
        "v119_carrier_aware_result_root_exists": V119_CARRIER_AWARE_RESULT_ROOT.exists(),
        "skipped_artifact_count": sum(row["count"] for row in skipped_summary.values()),
        "skipped_artifact_detail_rows": len(skipped),
        "skipped_artifact_detail_limit": SKIPPED_DETAIL_LIMIT,
        "skipped_artifact_summary": skipped_summary,
        "exclusion_policy": {
            "excluded_parts": sorted(EXCLUDED_PARTS),
            "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
            "max_artifact_bytes": MAX_ARTIFACT_BYTES,
        },
        "truthfulness_note": (
            "This is a source-focused/current-audit package. It does not claim that the v119 "
            "experiment goal or representation/runtime gates have passed."
        ),
    }
    (metadata_dir / "OBSERVED_STATE.json").write_text(
        json.dumps(observed_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_skipped(pack_dir, skipped, skipped_summary)

    commands = [
        "python3 -m py_compile tools/build_v119tf_core_code_audit_pack.py",
        "python3 tools/build_v119tf_core_code_audit_pack.py",
        "sha256sum -c code_audit_pack/<archive>.zip.sha256",
        "unzip -tq code_audit_pack/<archive>.zip",
        "cd code_audit_pack/<payload_dir> && sha256sum -c PAYLOAD_SHA256SUMS.txt",
    ]
    (metadata_dir / "BUILD_AND_VERIFY_COMMANDS.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    readme = f"""# ACL2 v119/v118 Core-Code Audit Packet

Created at: {created_at}

Scope:
- ACL2 v119 representation-repair and carrier-aware augmented plans/logs currently present in this checkout.
- ACL2 v118 plan, live logs, v118 builder/runner code, and lightweight v118 decision/summary artifacts.
- LoGeR, LingBot-Map, and HorizonStream code needed to review the semantic-memory carrier paths referenced by v118/v119.
- Packaging script, manifest, file hashes, zip test, and entry parity evidence.

Explicit exclusions:
- Checkpoints, raw data, caches, large runtime traces, jsonl traces, workspaces, report data dumps, and old audit packs.
- This packet is for code/audit review only; it does not claim v119 gates have passed.

Key verification files:
- `PACK_MANIFEST.tsv`
- `PAYLOAD_SHA256SUMS.txt`
- `metadata/OBSERVED_STATE.json`
- `metadata/SKIPPED_ARTIFACTS.tsv`
- sibling files next to the zip: `*.zip.sha256`, `*_zip_test.txt`, `*_zip_entry_parity.diff`, `*_validation_summary.txt`
"""
    (pack_dir / "PACKAGE_README.md").write_text(readme, encoding="utf-8")


def payload_files(pack_dir: Path) -> list[Path]:
    return sorted(path for path in pack_dir.rglob("*") if path.is_file())


def write_payload_hashes(pack_dir: Path) -> None:
    rows = []
    for path in payload_files(pack_dir):
        if path.name == "PAYLOAD_SHA256SUMS.txt":
            continue
        rows.append(f"{sha256(path)}  {path.relative_to(pack_dir).as_posix()}")
    (pack_dir / "PAYLOAD_SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_file_list(pack_dir: Path) -> None:
    entries = [path.relative_to(pack_dir).as_posix() for path in payload_files(pack_dir)]
    (pack_dir / "FILE_LIST.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")


def make_zip(pack_dir: Path, zip_path: Path) -> None:
    root_name = pack_dir.name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in payload_files(pack_dir):
            arcname = Path(root_name) / path.relative_to(pack_dir)
            zf.write(path, arcname.as_posix())


def validate_pack(pack_dir: Path, zip_path: Path, tag: str) -> dict[str, Any]:
    payload_rel = [path.relative_to(pack_dir).as_posix() for path in payload_files(pack_dir)]
    payload_list_path = PACK_ROOT / f"{tag}_payload_filelist_from_find.txt"
    payload_list_path.write_text("\n".join(payload_rel) + "\n", encoding="utf-8")

    with zipfile.ZipFile(zip_path) as zf:
        bad_member = zf.testzip()
        zip_entries = sorted(zf.namelist())

    zip_entries_path = PACK_ROOT / f"{tag}_zip_entries.txt"
    zip_entries_path.write_text("\n".join(zip_entries) + "\n", encoding="utf-8")

    prefix = pack_dir.name + "/"
    stripped_entries = sorted(entry.removeprefix(prefix) for entry in zip_entries if not entry.endswith("/"))
    stripped_path = PACK_ROOT / f"{tag}_zip_entries_stripped.txt"
    stripped_path.write_text("\n".join(stripped_entries) + "\n", encoding="utf-8")

    parity_diff = list(
        difflib.unified_diff(
            payload_rel,
            stripped_entries,
            fromfile="payload_filelist",
            tofile="zip_entries_stripped",
            lineterm="",
        )
    )
    parity_path = PACK_ROOT / f"{tag}_zip_entry_parity.diff"
    parity_path.write_text("\n".join(parity_diff) + ("\n" if parity_diff else ""), encoding="utf-8")

    zip_test = run_text(["unzip", "-tq", zip_path.as_posix()], timeout=240)
    zip_test_path = PACK_ROOT / f"{tag}_zip_test.txt"
    zip_test_path.write_text(zip_test, encoding="utf-8")

    zip_hash = sha256(zip_path)
    zip_sha_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    zip_sha_path.write_text(f"{zip_hash}  {zip_path.name}\n", encoding="utf-8")

    validation = {
        "zip_path": rel(zip_path),
        "zip_sha256": zip_hash,
        "zip_size_bytes": zip_path.stat().st_size,
        "payload_dir": rel(pack_dir),
        "payload_file_count": len(payload_rel),
        "zip_entry_count": len(stripped_entries),
        "zipfile_test_bad_member": bad_member,
        "entry_parity_pass": payload_rel == stripped_entries,
        "zip_test_output_file": rel(zip_test_path),
        "zip_entry_parity_diff": rel(parity_path),
        "zip_sha256_file": rel(zip_sha_path),
    }
    validation_path = PACK_ROOT / f"{tag}_validation_summary.txt"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def main() -> int:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    tag = f"{PACK_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pack_dir = PACK_ROOT / tag
    zip_path = PACK_ROOT / f"{tag}.zip"

    rows_by_path: dict[str, dict[str, Any]] = {}
    collect_docs(rows_by_path)
    collect_tools(rows_by_path)
    collect_core_runtime(rows_by_path)
    skipped, skipped_summary = collect_artifacts(rows_by_path)

    rows = sorted(rows_by_path.values(), key=lambda item: (str(item["category"]), str(item["source_path"])))
    if not rows:
        raise RuntimeError("no files selected for audit packet")

    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    if pack_dir.exists() or zip_path.exists():
        raise FileExistsError(f"refusing to overwrite existing pack tag: {tag}")
    pack_dir.mkdir(parents=True)

    copy_payload(pack_dir, rows)
    write_manifest(pack_dir, rows)
    write_metadata(pack_dir, rows, skipped, skipped_summary, created_at)
    write_payload_hashes(pack_dir)
    write_file_list(pack_dir)
    make_zip(pack_dir, zip_path)
    validation = validate_pack(pack_dir, zip_path, tag)

    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if validation["entry_parity_pass"] and validation["zipfile_test_bad_member"] is None else 2


if __name__ == "__main__":
    sys.exit(main())
