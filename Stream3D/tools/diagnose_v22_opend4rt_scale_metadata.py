from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent

SCALE_KEY_PATTERN = re.compile(r"(mean[_-]?depth|depth[_-]?scale|scale[_-]?depth|metric[_-]?scale|xyz[_-]?scale|z[_-]?scale)", re.IGNORECASE)
OUTPUT_KEY_PATTERN = re.compile(r'"([^"]+)"\s*:\s*self\.([A-Za-z_][A-Za-z0-9_]*)')
RETURN_KEY_PATTERN = re.compile(r'"([^"]+)"\s*:\s*([A-Za-z_][A-Za-z0-9_]*)')


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _find_scale_like_keys(keys: list[str]) -> list[str]:
    return sorted([key for key in keys if SCALE_KEY_PATTERN.search(str(key))])


def _extract_model_output_keys(text: str) -> list[str]:
    return sorted({match.group(1) for match in OUTPUT_KEY_PATTERN.finditer(text)})


def _has_explicit_scale_head(output_keys: list[str]) -> bool:
    return bool(_find_scale_like_keys(output_keys))


def _loss_has_independent_mean_depth_normalization(text: str) -> bool:
    return all(
        token in text
        for token in [
            "masked_mean_per_sample(depth, mask)",
            "out = out / scale",
            "pred = self._xyz_preprocess(pred, m, use_norm, use_log)",
            "gt = self._xyz_preprocess(gt, m, use_norm, use_log)",
        ]
    )


def _config_enables_mean_depth_normalization(text: str) -> bool:
    return bool(re.search(r"normalize_by_mean_depth\s*:\s*true\b", text, flags=re.IGNORECASE))


def _schema_has_metric_xyz(text: str) -> bool:
    return "Depth and 3D coordinates are in meters" in text and "y_xyz_cam_tcam" in text


def _schema_has_mean_depth_scale_field(text: str) -> bool:
    schema_fields = re.findall(r"`([^`]+)`", text)
    return bool(_find_scale_like_keys(schema_fields))


def _infer_return_keys(text: str) -> list[str]:
    start = text.find("return {")
    if start < 0:
        return []
    return sorted({match.group(1) for match in RETURN_KEY_PATTERN.finditer(text[start:])})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return "NA"
        return f"{numeric:.6f}"
    return str(value)


def _npz_key_rows(cache_root: Path, max_files: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(cache_root.glob("*/carriers_window*.npz"))
    for path in paths[: max(0, int(max_files))]:
        try:
            with np.load(path) as data:
                keys = [str(key) for key in data.files]
                scale_like = _find_scale_like_keys(keys)
                depth_like = sorted([key for key in keys if "depth" in key.lower()])
                rows.append(
                    {
                        "path": str(path),
                        "num_keys": len(keys),
                        "keys": ",".join(keys),
                        "scale_like_keys": ",".join(scale_like),
                        "depth_like_keys": ",".join(depth_like),
                        "has_scale_like_key": bool(scale_like),
                        "has_depth_like_key": bool(depth_like),
                    }
                )
        except Exception as exc:  # pragma: no cover - defensive audit path
            rows.append(
                {
                    "path": str(path),
                    "num_keys": 0,
                    "keys": "",
                    "scale_like_keys": "",
                    "depth_like_keys": "",
                    "has_scale_like_key": False,
                    "has_depth_like_key": False,
                    "error": repr(exc),
                }
            )
    return rows


def _source_evidence_rows(
    *,
    loss_path: Path,
    train_cfg_path: Path,
    heads_path: Path,
    infer_path: Path,
    schema_path: Path,
    loss_text: str,
    train_cfg_text: str,
    heads_text: str,
    infer_text: str,
    schema_text: str,
) -> list[dict[str, Any]]:
    output_keys = _extract_model_output_keys(heads_text)
    infer_keys = _infer_return_keys(infer_text)
    return [
        {
            "check": "loss_config_enables_mean_depth_normalization",
            "path": str(train_cfg_path),
            "pass": _config_enables_mean_depth_normalization(train_cfg_text),
            "evidence": "normalize_by_mean_depth: true",
        },
        {
            "check": "loss_normalizes_pred_and_gt_independently",
            "path": str(loss_path),
            "pass": _loss_has_independent_mean_depth_normalization(loss_text),
            "evidence": "pred and gt both call _xyz_preprocess; _xyz_preprocess divides by each tensor mean abs-z",
        },
        {
            "check": "model_has_no_explicit_scale_or_depth_head",
            "path": str(heads_path),
            "pass": not _has_explicit_scale_head(output_keys),
            "evidence": ",".join(output_keys),
        },
        {
            "check": "inference_return_has_no_scale_metadata",
            "path": str(infer_path),
            "pass": not _find_scale_like_keys(infer_keys),
            "evidence": ",".join(infer_keys),
        },
        {
            "check": "schema_metric_xyz_is_meter_scale",
            "path": str(schema_path),
            "pass": _schema_has_metric_xyz(schema_text),
            "evidence": "schema says depth/3D are meters and includes y_xyz_cam_tcam",
        },
        {
            "check": "schema_has_no_mean_depth_scale_field",
            "path": str(schema_path),
            "pass": not _schema_has_mean_depth_scale_field(schema_text),
            "evidence": "no query_pool/meta field matching mean-depth/depth-scale/metric-scale",
        },
    ]


def _write_md(
    path: Path,
    *,
    metadata: dict[str, Any],
    source_rows: list[dict[str, Any]],
    cache_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = [
        "# v22.17 OpenD4RT scale metadata audit",
        "",
        "This diagnostic checks whether the current OpenD4RT training/inference/cache path exposes a non-GT mean-depth or metric-scale side channel that could undo the `xyz_3d` loss normalization.",
        "",
        "## Summary",
        "",
        "| item | value |",
        "|---|---:|",
    ]
    for key in [
        "loss_config_enables_mean_depth_normalization",
        "loss_normalizes_pred_and_gt_independently",
        "model_has_explicit_scale_or_depth_head",
        "inference_return_has_scale_metadata",
        "schema_has_metric_xyz",
        "schema_has_mean_depth_scale_field",
        "cache_files_scanned",
        "cache_files_with_scale_like_keys",
        "cache_files_with_depth_like_keys",
        "method_result",
    ]:
        lines.append(f"| {key} | {_fmt(metadata.get(key))} |")

    lines.extend(
        [
            "",
            "## Source Evidence",
            "",
            "| check | pass | path | evidence |",
            "|---|---:|---|---|",
        ]
    )
    for row in source_rows:
        lines.append(f"| {row['check']} | {_fmt(row.get('pass'))} | `{row['path']}` | {row.get('evidence', '')} |")

    lines.extend(
        [
            "",
            "## Cache Keys",
            "",
            "| path | scale-like keys | depth-like keys |",
            "|---|---|---|",
        ]
    )
    for row in cache_rows:
        lines.append(
            f"| `{row['path']}` | {row.get('scale_like_keys') or 'NA'} | {row.get('depth_like_keys') or 'NA'} |"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "The current path has metric training targets, but the enabled loss removes each sample's mean-depth scale independently for prediction and GT. No explicit scale/depth head, inference return field, schema field, or Stream3D carrier-cache key preserves that normalization scale. This is diagnostic-only evidence: it does not create a reportable method row, and it points the next repair toward adding/learning an actual scale anchor rather than mining a hidden one from existing artifacts.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    *,
    opend4rt_root: Path,
    cache_root: Path,
    audit_root: Path,
    max_cache_files: int,
) -> dict[str, Any]:
    loss_path = opend4rt_root / "src/losses/d4rt_loss.py"
    train_cfg_path = opend4rt_root / "configs/train_effective.yaml"
    heads_path = opend4rt_root / "src/model/heads.py"
    infer_path = opend4rt_root / "infer_track_3d.py"
    schema_path = opend4rt_root / "docs/data_schema.md"

    loss_text = _read_text(loss_path)
    train_cfg_text = _read_text(train_cfg_path)
    heads_text = _read_text(heads_path)
    infer_text = _read_text(infer_path)
    schema_text = _read_text(schema_path)

    output_keys = _extract_model_output_keys(heads_text)
    infer_keys = _infer_return_keys(infer_text)
    source_rows = _source_evidence_rows(
        loss_path=loss_path,
        train_cfg_path=train_cfg_path,
        heads_path=heads_path,
        infer_path=infer_path,
        schema_path=schema_path,
        loss_text=loss_text,
        train_cfg_text=train_cfg_text,
        heads_text=heads_text,
        infer_text=infer_text,
        schema_text=schema_text,
    )
    cache_rows = _npz_key_rows(cache_root, max_files=max_cache_files)

    metadata = {
        "opend4rt_root": str(opend4rt_root),
        "cache_root": str(cache_root),
        "loss_config_enables_mean_depth_normalization": _config_enables_mean_depth_normalization(train_cfg_text),
        "loss_normalizes_pred_and_gt_independently": _loss_has_independent_mean_depth_normalization(loss_text),
        "model_output_keys": output_keys,
        "model_has_explicit_scale_or_depth_head": _has_explicit_scale_head(output_keys),
        "inference_return_keys": infer_keys,
        "inference_return_has_scale_metadata": bool(_find_scale_like_keys(infer_keys)),
        "schema_has_metric_xyz": _schema_has_metric_xyz(schema_text),
        "schema_has_mean_depth_scale_field": _schema_has_mean_depth_scale_field(schema_text),
        "cache_files_scanned": len(cache_rows),
        "cache_files_with_scale_like_keys": int(sum(1 for row in cache_rows if row.get("has_scale_like_key"))),
        "cache_files_with_depth_like_keys": int(sum(1 for row in cache_rows if row.get("has_depth_like_key"))),
        "method_result": False,
        "diagnostic_only": True,
    }

    audit_root.mkdir(parents=True, exist_ok=True)
    _write_csv(audit_root / "opend4rt_scale_metadata_source_evidence.csv", source_rows)
    _write_csv(audit_root / "opend4rt_scale_metadata_cache_keys.csv", cache_rows)
    (audit_root / "opend4rt_scale_metadata_source_evidence.json").write_text(
        json.dumps(_json_safe(source_rows), indent=2), encoding="utf-8"
    )
    (audit_root / "opend4rt_scale_metadata_cache_keys.json").write_text(
        json.dumps(_json_safe(cache_rows), indent=2), encoding="utf-8"
    )
    (audit_root / "opend4rt_scale_metadata_metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2), encoding="utf-8"
    )
    _write_md(
        audit_root / "opend4rt_scale_metadata_audit.md",
        metadata=metadata,
        source_rows=source_rows,
        cache_rows=cache_rows,
    )
    return metadata


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit whether OpenD4RT exposes mean-depth scale metadata.")
    parser.add_argument("--opend4rt-root", default=str(REPO_ROOT / "Open-d4rt"))
    parser.add_argument("--cache-root", default=str(STREAM3D_ROOT / "outputs/stream4d_debug_v22_local_xyz_probe5_r1"))
    parser.add_argument("--audit-root", default=str(STREAM3D_ROOT / "outputs/audit/v22_17_opend4rt_scale_metadata"))
    parser.add_argument("--max-cache-files", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    metadata = run_audit(
        opend4rt_root=Path(args.opend4rt_root),
        cache_root=Path(args.cache_root),
        audit_root=Path(args.audit_root),
        max_cache_files=int(args.max_cache_files),
    )
    print(f"Wrote v22.17 OpenD4RT scale metadata audit to {args.audit_root}")
    print(json.dumps(_json_safe(metadata), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
