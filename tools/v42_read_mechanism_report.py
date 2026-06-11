#!/usr/bin/env python3
"""v42 READ mechanism audit from landed/proxy artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _write_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# ACL2 v42 READ Mechanism Report",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "mechanism_decision",
        "explainability_level",
        "selected_bad_chunks",
        "proxy_source",
        "proxy_overlays_copied",
        "scalar_attention_mass_rows",
        "sky_causality_supported",
        "general_anomaly_supported",
        "static_anchor_misdamage_risk",
    ]:
        lines.append(f"- `{key}`: `{summary.get(key)}`")
    lines.extend([
        "",
        "Boundary:",
        "",
        "```text",
        str(summary.get("boundary", "")),
        "```",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-json", required=True, type=Path)
    parser.add_argument("--v41-mechanism-dir", default="results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase2_read_mechanism", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    selected_payload = _read_json(args.selected_json)
    selected_chunks = [int(v) for v in selected_payload.get("selected_bad_chunks", [])]
    v41_summary = _read_json(args.v41_mechanism_dir / "v41_read_mechanism_summary.json")
    scalar_rows = _read_csv(args.v41_mechanism_dir / "scalar_attention_mass_rows.csv")
    attribution = _read_csv(args.v41_mechanism_dir / "read_a2_a4_attribution.csv")

    copied = 0
    copied += int(_copy_if_exists(
        args.v41_mechanism_dir / "overlays/chunk010_proxy_rgb_frame_strip.png",
        args.out_dir / "overlays/chunk010_proxy_rgb_frame_strip.png",
    ))
    copied += int(_copy_if_exists(
        args.v41_mechanism_dir / "overlays/chunk010_proxy_semantic_mask_overlay.png",
        args.out_dir / "overlays/chunk010_proxy_semantic_mask_overlay.png",
    ))
    copied += int(_copy_if_exists(
        args.v41_mechanism_dir / "overlays/chunk010_proxy_appearance_anomaly_heatmap.png",
        args.out_dir / "overlays/chunk010_proxy_appearance_anomaly_heatmap.png",
    ))
    _copy_if_exists(
        args.v41_mechanism_dir / "overlays/spatial_attention_boundary.json",
        args.out_dir / "overlays/spatial_attention_boundary.json",
    )
    for filename in [
        "read_a2_a4_attribution.csv",
        "per_label_removed_source_mass.csv",
        "action_mask_overlap.csv",
        "read_a4_sky_causality_report.md",
        "READ_A2_A4_attribution_report.md",
        "scalar_attention_mass_rows.csv",
    ]:
        _copy_if_exists(args.v41_mechanism_dir / filename, args.out_dir / filename)

    sky_supported = bool(v41_summary.get("sky_causality_proven", False))
    mechanism_decision = str(v41_summary.get("mechanism_decision") or "B_general_high_influence_anomaly_preferred")
    explainability_level = "proxy_v41_chunk010_plus_scalar_mass" if 10 in selected_chunks else "incomplete_explainability_selected_chunk_not_chunk010_proxy"
    summary = {
        "selected_bad_chunks": selected_chunks,
        "proxy_source": str(args.v41_mechanism_dir),
        "proxy_overlays_copied": int(copied),
        "scalar_attention_mass_rows": len(scalar_rows),
        "attribution_rows": len(attribution),
        "mechanism_decision": mechanism_decision,
        "explainability_level": explainability_level,
        "sky_causality_supported": sky_supported,
        "general_anomaly_supported": mechanism_decision == "B_general_high_influence_anomaly_preferred",
        "static_anchor_misdamage_risk": "not_proven_from_spatial_maps",
        "boundary": (
            "v42 reuses landed v41 scalar attention-mass/proxy overlay evidence where selected chunks overlap chunk010. "
            "Per-label spatial source-attention, READ affected masks, and before/after tensor maps are still not landed; "
            "missing spatial evidence is marked incomplete rather than reconstructed."
        ),
    }
    _write_json(args.out_dir / "v42_read_mechanism_summary.json", summary)
    _write_md(args.out_dir / "read_mechanism_report.md", summary)
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

