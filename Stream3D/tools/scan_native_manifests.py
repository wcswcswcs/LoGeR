from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any


FORBIDDEN_IMPORT_FRAGMENTS = (
    "scannet_stream",
    "mask_backprojection",
    "export_scannet",
    "reliable_densifier",
    "materialize_scannet",
)
FORBIDDEN_CALL_NAMES = (
    "load_depth",
    "load_pose",
    "load_intrinsics",
    "read_point_cloud",
    "get_depth",
    "get_extrinsic",
    "get_intrinsics",
)
PREDICTION_FORBIDDEN_FLAGS = (
    "uses_rgbd_for_prediction",
    "uses_pose_for_prediction",
    "uses_scannet_mesh_for_prediction",
    "uses_gt_for_prediction",
    "uses_gt_sim3_for_prediction",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _stream3d_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _scan_source_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"syntax_error:{exc.lineno}:{exc.msg}"]
    reasons: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if any(fragment in name for fragment in FORBIDDEN_IMPORT_FRAGMENTS):
                    reasons.append(f"forbidden_import:{name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if any(fragment in module for fragment in FORBIDDEN_IMPORT_FRAGMENTS):
                reasons.append(f"forbidden_import_from:{module}")
        elif isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in FORBIDDEN_CALL_NAMES:
                reasons.append(f"forbidden_call:{name}")
    return sorted(set(reasons))


def _scan_native_sources(root: Path) -> list[dict[str, Any]]:
    source_dir = root / "stream4d_native"
    rows: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("*.py")):
        reasons = _scan_source_file(path)
        rows.append(
            {
                "kind": "source",
                "path": str(path.relative_to(root)),
                "forbidden": bool(reasons),
                "reasons": ",".join(reasons),
            }
        )
    return rows


def _scan_prediction_manifests(root: Path, manifest_glob: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob(manifest_glob)):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append(
                {
                    "kind": "manifest",
                    "path": str(path.relative_to(root)),
                    "forbidden": True,
                    "reasons": f"manifest_parse_error:{type(exc).__name__}",
                }
            )
            continue
        if not bool(manifest.get("is_method_result", False)):
            forbidden_flags: list[str] = []
        else:
            forbidden_flags = [flag for flag in PREDICTION_FORBIDDEN_FLAGS if bool(manifest.get(flag, False))]
            if bool(manifest.get("is_diagnostic_only", False)):
                forbidden_flags.append("is_diagnostic_only_method_result")
        rows.append(
            {
                "kind": "manifest",
                "path": str(path.relative_to(root)),
                "forbidden": bool(forbidden_flags),
                "reasons": ",".join(forbidden_flags),
            }
        )
    return rows


def _write_outputs(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    rows = payload["rows"]
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["kind", "path", "forbidden", "reasons"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Native Manifest And Source Guard Scan",
        "",
        "## Summary",
        "",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Rows", "", "| kind | path | forbidden | reasons |", "|---|---|---:|---|"])
    for row in rows:
        lines.append(f"| {row['kind']} | `{row['path']}` | {row['forbidden']} | {row['reasons']} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan D4RT-native code and native manifests for forbidden prediction geometry.")
    parser.add_argument("--root", default=None, help="Stream3D root. Defaults to the package root.")
    parser.add_argument("--manifest-glob", default="data/prediction/*native*/*config_manifest.json")
    parser.add_argument("--output", default="outputs/audit/v21_3_phaseA/native_manifest_scan.md")
    parser.add_argument("--require-no-gt-prediction", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve() if args.root else _stream3d_root()
    rows = _scan_native_sources(root) + _scan_prediction_manifests(root, args.manifest_glob)
    forbidden_count = int(sum(1 for row in rows if row["forbidden"]))
    summary = {
        "stream3d_root": str(root),
        "manifest_glob": str(args.manifest_glob),
        "source_files_scanned": int(sum(1 for row in rows if row["kind"] == "source")),
        "manifest_files_scanned": int(sum(1 for row in rows if row["kind"] == "manifest")),
        "forbidden_import_count": forbidden_count,
        "method_path_forbidden_imports_count": forbidden_count,
        "num_method_configs_with_gt_or_rgbd_geometry": int(
            sum(1 for row in rows if row["kind"] == "manifest" and row["forbidden"])
        ),
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    payload = {"summary": summary, "rows": rows}
    _write_outputs(output, payload)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    if args.require_no_gt_prediction and forbidden_count > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
