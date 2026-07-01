#!/usr/bin/env python3
"""Build a compact ACL2 v105-TF dual-track core-code audit packet.

This packet is intentionally source-centric.  It captures the v105 plan,
the LingBot-Map GCT/GCA interfaces named by that plan, the LoGeR memory /
semantic pipeline files that reviewers need for context, and the v104
carry-forward scripts/docs that v105 explicitly builds on.
"""

from __future__ import annotations

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
PACK_PREFIX = "acl2_v105tf_dualtrack_lingbot_loger_core_code"
RESULT_ROOT = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
SELF = Path("tools/build_v105tf_dualtrack_lingbot_loger_core_code_audit_pack.py")

DOC_FILES = [
    "docs/ACL2_v105TF_DualTrack_LingBotMap_LoGeR_EvidenceEligibility_ExperimentPlan.md",
    "docs/ACL2_v105TF_DualTrack_LingBotMap_LoGeR_EvidenceEligibility_执行日志.md",
    "docs/ACL2_v105TF_DualTrack_LingBotMap_LoGeR_EvidenceEligibility_实验结果复盘.md",
    "docs/ACL2_v104TF_StrictProvider_EvidenceEligibilityStateMachine_MemoryControl_ExperimentPlan.md",
    "docs/ACL2_v104TF_StrictProvider_EvidenceEligibilityStateMachine_MemoryControl_执行日志.md",
    "docs/ACL2_v104TF_StrictProvider_EvidenceEligibilityStateMachine_MemoryControl_实验结果复盘.md",
]

LINGBOT_CORE_FILES = [
    "third_party/lingbot-map/README.md",
    "third_party/lingbot-map/benchmark/README.md",
    "third_party/lingbot-map/benchmark/README_zh.md",
    "third_party/lingbot-map/benchmark/configs/kitti.yaml",
    "third_party/lingbot-map/benchmark/methods/lingbot_map.py",
    "third_party/lingbot-map/lingbot_map/models/gct_stream.py",
    "third_party/lingbot-map/lingbot_map/models/gct_stream_window.py",
    "third_party/lingbot-map/lingbot_map/aggregator/stream.py",
    "third_party/lingbot-map/lingbot_map/layers/attention.py",
    "third_party/lingbot-map/lingbot_map/layers/flashinfer_cache.py",
]

LOGER_CORE_FILES = [
    "run_geometry_backbone_inference.py",
    "run_pipeline_abc.py",
    "run_pipeline_abc_v2.py",
    "loger/models/pi3.py",
    "loger/models/layers/attention.py",
    "loger/models/layers/block.py",
    "loger/pipeline/geometry_backbone.py",
    "loger/pipeline/hybrid_memory_controller.py",
    "loger/pipeline/semantic_prior_generator.py",
    "loger/pipeline/ttt_write_controller.py",
]

V104_CARRY_FORWARD_CODE = [
    "tools/build_v104tf_strict_provider_evidence_eligibility_state_machine_memory_control.py",
    "tools/run_v104tf_no_trace_pose_sha_parity.py",
]

V105_EXPERIMENT_CODE = [
    "tools/build_v105tf_dualtrack_lingbot_loger_stage0_audit.py",
    "tools/build_v105tf_lingbot_stage1_configs.py",
    "tools/build_v105tf_lingbot_stage1_debug_metrics.py",
    "tools/build_v105tf_lingbot_stage1_full_metrics.py",
    "tools/build_v105tf_lingbot_stage2_trace_configs.py",
    "tools/build_v105tf_lingbot_stage2_trace_summary.py",
    "tools/build_v105tf_lingbot_stage3_decision.py",
    "tools/build_v105tf_lingbot_stage3_oracle.py",
    "tools/build_v105tf_lingbot_stage3_oracle_sweep.py",
    "tools/build_v105tf_lingbot_stage4_action_configs.py",
    "tools/build_v105tf_lingbot_stage4_action_summary.py",
    "tools/build_v105tf_lingbot_stage4_headlocal_policy_sweep.py",
    "tools/build_v105tf_lingbot_stage4_headlocal_trace_configs.py",
    "tools/build_v105tf_lingbot_stage4_safety_policy_sweep.py",
    "tools/build_v105tf_stage1_loger_comparison.py",
    "tools/build_v105tf_stage5_cross_platform_decision.py",
    "tools/run_v105tf_lingbot_stage4_action_manifest.py",
    "tools/run_v105tf_lingbot_stage4_headlocal_manifest.py",
]

PACKER_FILES = [
    SELF.as_posix(),
]

EXCLUDED_SUFFIXES = {
    ".avi",
    ".bin",
    ".ckpt",
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
    "code_audit_pack",
    "results",
    "wandb",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


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


def add_file(src_rel: str, pack_dir: Path, manifest: list[dict[str, Any]], *, category: str) -> None:
    src = ROOT / src_rel
    if not src.is_file():
        manifest.append(
            {
                "category": category,
                "source_path": src_rel,
                "archive_path": "",
                "size_bytes": 0,
                "sha256": "",
                "status": "missing",
            }
        )
        return
    if should_exclude(Path(src_rel)):
        raise ValueError(f"excluded source requested: {src_rel}")
    dst = pack_dir / src_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "category": category,
            "source_path": src_rel,
            "archive_path": src_rel,
            "size_bytes": dst.stat().st_size,
            "sha256": sha256(dst),
            "status": "included",
        }
    )


def collect_sources() -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    for item in DOC_FILES:
        sources.append(("doc", item))
    for item in LINGBOT_CORE_FILES:
        sources.append(("lingbot_core", item))
    for item in LOGER_CORE_FILES:
        sources.append(("loger_core", item))
    for item in V104_CARRY_FORWARD_CODE:
        sources.append(("v104_carry_forward_code", item))
    for item in V105_EXPERIMENT_CODE:
        sources.append(("v105_experiment_code", item))
    for item in PACKER_FILES:
        sources.append(("packaging_code", item))
    return sources


def write_manifest(pack_dir: Path, manifest: list[dict[str, Any]]) -> None:
    keys = ["category", "source_path", "archive_path", "size_bytes", "sha256", "status"]
    lines = ["\t".join(keys)]
    for row in sorted(manifest, key=lambda item: (str(item["category"]), str(item["source_path"]))):
        lines.append("\t".join(str(row.get(key, "")) for key in keys))
    path = pack_dir / "PACK_MANIFEST.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def observed_v105_state() -> dict[str, Any]:
    docs = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").glob("*v105*"))
    tools = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tools").glob("*v105*"))
    result_files: list[str] = []
    result_stage_dirs: list[str] = []
    if RESULT_ROOT.exists():
        result_files = sorted(p.relative_to(ROOT).as_posix() for p in RESULT_ROOT.rglob("*") if p.is_file())
        result_stage_dirs = sorted(
            p.relative_to(ROOT).as_posix() for p in RESULT_ROOT.iterdir() if p.is_dir()
        )
    expected_logs = [
        "docs/ACL2_v105TF_DualTrack_LingBotMap_LoGeR_EvidenceEligibility_执行日志.md",
        "docs/ACL2_v105TF_DualTrack_LingBotMap_LoGeR_EvidenceEligibility_实验结果复盘.md",
    ]
    return {
        "schema": "acl2_v105tf_observed_state_v1",
        "result_root": RESULT_ROOT.relative_to(ROOT).as_posix(),
        "result_root_exists": RESULT_ROOT.exists(),
        "result_stage_dirs": result_stage_dirs,
        "result_file_count": len(result_files),
        "v105_docs": docs,
        "v105_tools": tools,
        "expected_v105_logs": expected_logs,
        "expected_v105_logs_present": {item: (ROOT / item).is_file() for item in expected_logs},
        "note": "This is a core-code audit packet, not evidence that the v105 experiment gates passed.",
    }


def write_metadata(pack_dir: Path, manifest: list[dict[str, Any]]) -> None:
    metadata_dir = pack_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    included = sorted(row["source_path"] for row in manifest if row.get("status") == "included")
    lingbot_included = sorted(
        item.removeprefix("third_party/lingbot-map/")
        for item in included
        if item.startswith("third_party/lingbot-map/")
    )
    (metadata_dir / "SOURCE_GIT_HEAD.txt").write_text(run_text(["git", "rev-parse", "HEAD"]), encoding="utf-8")
    (metadata_dir / "SOURCE_GIT_STATUS_RELEVANT.txt").write_text(
        run_text(["git", "status", "--short", "--", *included]),
        encoding="utf-8",
    )
    (metadata_dir / "SOURCE_GIT_DIFF_RELEVANT.patch").write_text(
        run_text(["git", "diff", "--", *included]),
        encoding="utf-8",
    )
    (metadata_dir / "LINGBOT_GIT_HEAD.txt").write_text(
        run_text(["git", "-C", "third_party/lingbot-map", "rev-parse", "HEAD"]),
        encoding="utf-8",
    )
    (metadata_dir / "LINGBOT_GIT_STATUS_RELEVANT.txt").write_text(
        run_text(["git", "-C", "third_party/lingbot-map", "status", "--short", "--", *lingbot_included]),
        encoding="utf-8",
    )
    (metadata_dir / "LINGBOT_GIT_DIFF_RELEVANT.patch").write_text(
        run_text(["git", "-C", "third_party/lingbot-map", "diff", "--", *lingbot_included]),
        encoding="utf-8",
    )
    commands = [
        "python3 -m py_compile tools/build_v105tf_dualtrack_lingbot_loger_core_code_audit_pack.py",
        "python3 tools/build_v105tf_dualtrack_lingbot_loger_core_code_audit_pack.py",
        "cd code_audit_pack && sha256sum -c <archive>.zip.sha256",
        "unzip -tq code_audit_pack/<archive>.zip",
        "cd <extracted-packet-root> && sha256sum -c PAYLOAD_SHA256SUMS.txt",
    ]
    (metadata_dir / "BUILD_AND_VERIFY_COMMANDS.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    state = observed_v105_state()
    (metadata_dir / "OBSERVED_V105_STATE.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    missing = [row["source_path"] for row in manifest if row.get("status") != "included"]
    included_by_category: dict[str, int] = {}
    for row in manifest:
        if row.get("status") == "included":
            included_by_category[row["category"]] = included_by_category.get(row["category"], 0) + 1
    readme = f"""# ACL2 v105-TF Dual-Track Core-Code Audit Packet

Scope:
- v105 experiment plan for Dual-Track LingBot-Map / LoGeR evidence eligibility.
- LingBot-Map GCT/GCA and benchmark interface files explicitly named by the v105 plan.
- LoGeR geometry backbone, semantic prior, hybrid memory, TTT write controller, and pipeline entrypoints relevant to semantic-geometric memory management.
- v104 strict-provider carry-forward scripts and documents referenced by v105 as prior evidence context.
- This packaging script, manifest, source git status/diff for included paths, payload hashes, and validation sidecars.
- LingBot-Map nested checkout HEAD/status/diff for included LingBot paths.

Audit boundary:
- Source-centric core-code packet only.
- No raw datasets, checkpoints, tensor dumps, caches, previous audit packs, or bulk result directories.
- The packet does not claim v105 experiment completion. Observed v105 state is recorded in `metadata/OBSERVED_V105_STATE.json`.

Observed v105 state at build time:
- result_root_exists: {state["result_root_exists"]}
- result_file_count: {state["result_file_count"]}
- expected_v105_logs_present: {json.dumps(state["expected_v105_logs_present"], ensure_ascii=False, sort_keys=True)}

Included file counts by category:
{json.dumps(included_by_category, ensure_ascii=False, indent=2, sort_keys=True)}

Missing requested source files:
{json.dumps(missing, ensure_ascii=False, indent=2)}

Validation files outside the zip:
- `<archive>.zip.sha256`
- `<archive>.payload_sha256_check.txt`
- `<archive>.unzip_test.txt`
- `<archive>.entry_parity.txt`
- `<archive>.exclusion_check.txt`
- `<archive>.summary.json`
"""
    (pack_dir / "PACK_README.md").write_text(readme, encoding="utf-8")


def write_payload_hashes(pack_dir: Path) -> list[str]:
    filelist_path = pack_dir / "PAYLOAD_FILELIST.txt"
    hash_path = pack_dir / "PAYLOAD_SHA256SUMS.txt"
    hash_path.write_text("", encoding="utf-8")
    files = sorted(p.relative_to(pack_dir).as_posix() for p in pack_dir.rglob("*") if p.is_file())
    filelist_path.write_text("\n".join(files) + "\n", encoding="utf-8")
    files = sorted(p.relative_to(pack_dir).as_posix() for p in pack_dir.rglob("*") if p.is_file())
    hash_files = [name for name in files if name != "PAYLOAD_SHA256SUMS.txt"]
    hash_path.write_text(
        "\n".join(f"{sha256(pack_dir / name)}  {name}" for name in hash_files) + "\n",
        encoding="utf-8",
    )
    return sorted(p.relative_to(pack_dir).as_posix() for p in pack_dir.rglob("*") if p.is_file())


def create_zip(pack_dir: Path, zip_path: Path) -> list[str]:
    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for src in sorted(p for p in pack_dir.rglob("*") if p.is_file()):
            arcname = src.relative_to(pack_dir).as_posix()
            zf.write(src, arcname)
            entries.append(arcname)
    return entries


def check_payload_hashes(zip_path: Path, sidecar_path: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="acl2_v105_pack_verify_") as tmp:
        tmpdir = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)
        proc = subprocess.run(
            ["sha256sum", "-c", "PAYLOAD_SHA256SUMS.txt"],
            cwd=tmpdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        sidecar_path.write_text(proc.stdout, encoding="utf-8")
        return proc.returncode


def check_unzip(zip_path: Path, sidecar_path: Path) -> int:
    proc = subprocess.run(
        ["unzip", "-tq", str(zip_path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    sidecar_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode


def check_entry_parity(payload_files: list[str], zip_entries: list[str], sidecar_path: Path) -> int:
    payload_set = set(payload_files)
    zip_set = set(zip_entries)
    missing = sorted(payload_set - zip_set)
    extra = sorted(zip_set - payload_set)
    data = {
        "payload_file_count": len(payload_files),
        "zip_entry_count": len(zip_entries),
        "missing_from_zip": missing,
        "extra_in_zip": extra,
        "entry_parity_ok": not missing and not extra,
    }
    sidecar_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if data["entry_parity_ok"] else 1


def check_exclusions(zip_entries: list[str], sidecar_path: Path) -> int:
    violations: list[str] = []
    for entry in zip_entries:
        path = Path(entry)
        if should_exclude(path):
            violations.append(entry)
        if entry.startswith("results/") or entry.startswith("code_audit_pack/"):
            violations.append(entry)
    data = {
        "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
        "excluded_names": sorted(EXCLUDED_NAMES),
        "violation_count": len(violations),
        "violations": sorted(set(violations)),
        "exclusion_check_ok": not violations,
    }
    sidecar_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if data["exclusion_check_ok"] else 1


def build() -> dict[str, Any]:
    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = PACK_ROOT / f"{PACK_PREFIX}_{timestamp}.zip"
    with tempfile.TemporaryDirectory(prefix=f"{PACK_PREFIX}_") as tmp:
        pack_dir = Path(tmp) / f"{PACK_PREFIX}_{timestamp}"
        pack_dir.mkdir(parents=True)
        manifest: list[dict[str, Any]] = []
        for category, src_rel in collect_sources():
            add_file(src_rel, pack_dir, manifest, category=category)
        write_manifest(pack_dir, manifest)
        write_metadata(pack_dir, manifest)
        payload_files = write_payload_hashes(pack_dir)
        zip_entries = create_zip(pack_dir, zip_path)

    sha_sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sha_sidecar.write_text(f"{sha256(zip_path)}  {zip_path.name}\n", encoding="utf-8")
    payload_check = zip_path.with_suffix(zip_path.suffix + ".payload_sha256_check.txt")
    unzip_test = zip_path.with_suffix(zip_path.suffix + ".unzip_test.txt")
    entry_parity = zip_path.with_suffix(zip_path.suffix + ".entry_parity.txt")
    exclusion_check = zip_path.with_suffix(zip_path.suffix + ".exclusion_check.txt")
    payload_rc = check_payload_hashes(zip_path, payload_check)
    unzip_rc = check_unzip(zip_path, unzip_test)
    parity_rc = check_entry_parity(payload_files, zip_entries, entry_parity)
    exclusion_rc = check_exclusions(zip_entries, exclusion_check)
    summary = {
        "schema": "acl2_v105tf_core_code_audit_pack_summary_v1",
        "archive": zip_path.relative_to(ROOT).as_posix(),
        "archive_size_bytes": zip_path.stat().st_size,
        "archive_sha256": sha256(zip_path),
        "payload_file_count": len(payload_files),
        "zip_entry_count": len(zip_entries),
        "payload_sha256_check_returncode": payload_rc,
        "unzip_test_returncode": unzip_rc,
        "entry_parity_returncode": parity_rc,
        "exclusion_check_returncode": exclusion_rc,
        "validation_ok": payload_rc == 0 and unzip_rc == 0 and parity_rc == 0 and exclusion_rc == 0,
        "sidecars": [
            sha_sidecar.relative_to(ROOT).as_posix(),
            payload_check.relative_to(ROOT).as_posix(),
            unzip_test.relative_to(ROOT).as_posix(),
            entry_parity.relative_to(ROOT).as_posix(),
            exclusion_check.relative_to(ROOT).as_posix(),
        ],
        "observed_v105_state": observed_v105_state(),
    }
    summary_path = zip_path.with_suffix(zip_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["validation_ok"]:
        raise SystemExit(1)
    return summary


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
