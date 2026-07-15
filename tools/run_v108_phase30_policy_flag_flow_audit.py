#!/usr/bin/env python3
"""Audit v108 policy callsites for safe attestation-flag flow."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SAFE_POLICY_READY_NAMES = {
    "policy_user_attestation_verified",
    "activation_attestation_ready_for_policy",
}
TARGET_ATTRS = {"evaluate", "suggest_from_shadow_stats"}
TARGET_CLASS_NAMES = {"DelayedAdmissionPolicy", "GrowthRepairPlanner"}
DEFAULT_SCAN_ROOTS = [
    ROOT / "tools",
    ROOT / "Stream3D" / "stream4d_v108",
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def expr_text(source: str, node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.get_source_segment(source, node) or ast.unparse(node)
    except Exception:
        return ""


def collect_python_files(paths: list[Path]) -> list[Path]:
    out: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            out.add(path)
            continue
        if path.is_dir():
            out.update(p for p in path.rglob("*.py") if p.is_file())
    return sorted(out)


def imports_target_classes(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name in TARGET_CLASS_NAMES for alias in node.names):
                return True
        elif isinstance(node, ast.Import):
            if any(alias.name.endswith("stream4d_v108.lifecycle") or alias.name.endswith("stream4d_v108.growth_repair") for alias in node.names):
                return True
    return False


def name_set(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def attestation_arg_status(node: ast.AST | None, source: str) -> tuple[str, str, str]:
    if node is None:
        return "omitted_defaults_false", "safe", ""
    text = expr_text(source, node)
    if isinstance(node, ast.Constant):
        if node.value is True:
            return "constant_true", "unsafe", text
        if node.value is False:
            return "constant_false", "safe", text
    names = name_set(node)
    if names & SAFE_POLICY_READY_NAMES:
        return "policy_ready_expression", "safe", text
    if "user_attestation_verified" in names:
        return "raw_user_attestation_verified_expression", "unsafe", text
    return "other_expression", "review", text


def call_kind(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        if node.func.attr == "suggest_from_shadow_stats":
            return "GrowthRepairPlanner.suggest_from_shadow_stats"
        if node.func.attr == "evaluate":
            return "DelayedAdmissionPolicy.evaluate_candidate"
    return "unknown"


def audit_file(path: Path) -> list[dict[str, Any]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    if not imports_target_classes(tree) and "DelayedAdmissionPolicy" not in source and "GrowthRepairPlanner" not in source:
        return []
    rows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr not in TARGET_ATTRS:
            continue
        keyword = next((kw for kw in node.keywords if kw.arg == "user_attestation_verified"), None)
        status, severity, text = attestation_arg_status(keyword.value if keyword else None, source)
        rows.append(
            {
                "file": rel(path),
                "line": int(getattr(node, "lineno", -1)),
                "call_kind": call_kind(node),
                "function_attr": node.func.attr,
                "user_attestation_argument_status": status,
                "severity": severity,
                "argument_source": text,
                "note": (
                    "omitted defaults false and is safe; any true value must come from policy_user_attestation_verified "
                    "or activation_attestation_ready_for_policy, not raw user_attestation_verified"
                ),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scan-root", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    scan_roots = [resolve_path(item) for item in args.scan_root] if args.scan_root else DEFAULT_SCAN_ROOTS
    files = collect_python_files(scan_roots)
    rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for path in files:
        try:
            rows.extend(audit_file(path))
        except Exception as exc:
            parse_errors.append({"file": rel(path), "error": str(exc)})

    unsafe_rows = [row for row in rows if row["severity"] == "unsafe"]
    review_rows = [row for row in rows if row["severity"] == "review"]
    safe_rows = [row for row in rows if row["severity"] == "safe"]

    rows_csv = output_root / "phase30_policy_flag_flow_rows.csv"
    rows_json = output_root / "phase30_policy_flag_flow_rows.json"
    write_csv(rows_csv, rows)
    write_json(rows_json, {"schema_version": "stream4d_v108_phase30_policy_flag_flow_rows_v1", "records": rows})

    summary = {
        "schema_version": "stream4d_v108_phase30_policy_flag_flow_summary_v1",
        "status": "POLICY_FLAG_FLOW_AUDIT_PASS" if not unsafe_rows and not parse_errors else "POLICY_FLAG_FLOW_AUDIT_FAIL",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "scan_roots": [rel(path) for path in scan_roots],
        "scanned_python_file_count": int(len(files)),
        "policy_callsite_count": int(len(rows)),
        "safe_callsite_count": int(len(safe_rows)),
        "review_callsite_count": int(len(review_rows)),
        "unsafe_callsite_count": int(len(unsafe_rows)),
        "parse_error_count": int(len(parse_errors)),
        "unsafe_rows": unsafe_rows,
        "review_rows": review_rows,
        "parse_errors": parse_errors,
        "rows_csv": rel(rows_csv),
        "rows_csv_sha256": sha256_file(rows_csv),
        "rows_json": rel(rows_json),
        "rows_json_sha256": sha256_file(rows_json),
        "note": (
            "Safe current-state behavior means policy callsites either omit user_attestation_verified, "
            "which defaults to false, or derive it from policy-ready Phase20 fields. This audit is not "
            "visual acceptance and does not apply SAM2 memory."
        ),
    }
    summary_path = output_root / "phase30_policy_flag_flow_summary.json"
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": rel(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "status": summary["status"],
                "policy_callsite_count": int(summary["policy_callsite_count"]),
                "unsafe_callsite_count": int(summary["unsafe_callsite_count"]),
                "review_callsite_count": int(summary["review_callsite_count"]),
                "parse_error_count": int(summary["parse_error_count"]),
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "POLICY_FLAG_FLOW_AUDIT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
