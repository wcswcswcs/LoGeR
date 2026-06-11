#!/usr/bin/env python3
"""Write v41 READ_A2/A4/A5 mechanism attribution from landed artifacts.

Spatial attention/source tensors are not reconstructed.  When only candidate
level summaries are available, the output explicitly marks the evidence level.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


V40_ROOT = Path("results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30")
V39_REPORT = Path("results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal/phase0_semantic_appearance/report_R1")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _finite(value: Any, default: float = 0.0) -> float:
    out = _f(value, default)
    return out if math.isfinite(out) else default


def _read_selected_chunks(root: Path) -> List[int]:
    path = root / "phase1_health_detector/selected_bad_chunks.json"
    if not path.exists():
        return [10]
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = [int(x) for x in data.get("bad_chunks", [])]
    return chunks or [10]


def _scan_attention_mass(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rollout_root = root / "phase2_read_mechanism/h3_R1/rollouts"
    if not rollout_root.exists():
        return rows
    for run_dir in sorted(p for p in rollout_root.iterdir() if p.is_dir()):
        for jsonl in ("hook_effect_summary.jsonl", "hmc_state_hash.jsonl", "context_skip_summary.jsonl"):
            path = run_dir / jsonl
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    flat: Dict[str, Any] = {}
                    _flatten(record, flat)
                    mass_keys = [k for k in flat if "attention_mass" in k or "removed_before" in k or "removed_after" in k]
                    if not mass_keys:
                        continue
                    row = {"run_dir": str(run_dir), "file": jsonl}
                    for key in mass_keys:
                        value = flat[key]
                        if isinstance(value, (int, float, str, bool)) or value is None:
                            row[key] = value
                    rows.append(row)
    return rows


def _flatten(value: Any, out: Dict[str, Any], prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(item, out, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for idx, item in enumerate(value[:8]):
            _flatten(item, out, f"{prefix}[{idx}]")
    else:
        out[prefix] = value


def _copy_proxy_overlays(v39_report: Path, out_dir: Path, chunks: List[int]) -> List[str]:
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for chunk in chunks:
        for kind in ("rgb_frame_strip", "semantic_mask_overlay", "appearance_anomaly_heatmap"):
            src = v39_report / f"{kind}_chunk{chunk:03d}.png"
            if not src.exists():
                continue
            dst = overlay_dir / f"chunk{chunk:03d}_proxy_{kind}.png"
            shutil.copy2(src, dst)
            copied.append(str(dst))
    boundary = overlay_dir / "spatial_attention_boundary.json"
    boundary.write_text(
        json.dumps(
            {
                "proxy_overlays_copied": copied,
                "missing_required_spatial_tensors": [
                    "D_g_map",
                    "source_attention_mass_map_before_action",
                    "source_attention_mass_map_after_action",
                    "candidate_affected_source_mask",
                    "static_anchor_source_map",
                    "high_D_source_map",
                    "high_influence_anomaly_map",
                ],
                "boundary": "RGB/semantic/appearance proxy overlays copied from v39; missing spatial attention tensors are not reconstructed.",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v40-root", type=Path, default=V40_ROOT)
    parser.add_argument("--v39-report", type=Path, default=V39_REPORT)
    args = parser.parse_args()

    out_dir = args.root / "phase2_read_mechanism"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_chunks = _read_selected_chunks(args.root)

    effects = _read_csv(args.v40_root / "phase2a_read/report_h10_R1/read_h10_effects.csv")
    candidates = {
        "READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT": "general_high_influence_anomaly",
        "READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH": "sky_app_anomaly_source_mass_gated",
        "READ_A5_STATIC_ANCHOR_RESCUE_ONLY": "static_anchor_rescue_only",
    }
    attribution: List[Dict[str, Any]] = []
    for row in effects:
        if row.get("candidate") not in candidates:
            continue
        chunk = int(_finite(row.get("chunk"), -1))
        if chunk not in selected_chunks:
            continue
        frame_removed = _finite(row.get("frame_attention_num_context_source_skip_applied"), 0.0)
        chunk_removed = _finite(row.get("chunk_attention_num_context_source_skip_applied"), 0.0)
        attribution.append({
            "parent": row.get("parent"),
            "chunk": chunk,
            "candidate": row.get("candidate"),
            "mechanism_family": candidates[row.get("candidate")],
            "ATE_delta_vs_base": _f(row.get("ATE_delta_vs_base")),
            "rolling_100f_best_delta_vs_base": _f(row.get("rolling_100f_best_delta_vs_base")),
            "intersection_200_300_delta_vs_base": _f(row.get("intersection_200_300_delta_vs_base")),
            "intersection_400_600_delta_vs_base": _f(row.get("intersection_400_600_delta_vs_base")),
            "frame_context_skip_applied": frame_removed,
            "chunk_context_skip_applied": chunk_removed,
            "mean_context_source_keep_ratio_frame": _f(row.get("frame_attention_mean_context_source_keep_ratio")),
            "mean_context_source_keep_ratio_chunk": _f(row.get("chunk_attention_mean_context_source_keep_ratio")),
            "removed_attention_mass_total": "scalar_attention_mass_available_only_if_h3_instrumented",
            "evidence_level": "trajectory_metrics_plus_hook_counts; no per-label spatial attention map",
        })

    _write_csv(out_dir / "read_a2_a4_attribution.csv", attribution)

    source_mass = _read_csv(args.v39_report / "per_semantic_source_attention_mass.csv")
    label_rows: List[Dict[str, Any]] = []
    label_map = {
        "READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT": "mixed_or_unspecified",
        "READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH": "sky",
        "READ_A5_STATIC_ANCHOR_RESCUE_ONLY": "road/building/wall/fence",
    }
    for candidate, target in label_map.items():
        for row in source_mass:
            label = row.get("semantic_target_from_policy_name") or row.get("label") or "unknown"
            if target != "mixed_or_unspecified" and target not in label:
                continue
            label_rows.append({
                "candidate": candidate,
                "semantic_label_or_proxy": label,
                "path": row.get("path"),
                "per_label_source_mass_before": row.get("attention_mass_removed_before", ""),
                "per_label_source_mass_after": row.get("attention_mass_removed_after", ""),
                "per_label_removed_source_mass": row.get("attention_mass_removed_before", ""),
                "candidate_level_influence_mass": row.get("influence_mass", ""),
                "evidence_status": row.get("evidence_status", "candidate_level_path_summary_not_per_label"),
            })
    if not label_rows:
        label_rows.append({
            "candidate": "ALL",
            "semantic_label_or_proxy": "explainability_missing",
            "path": "NA",
            "per_label_source_mass_before": "",
            "per_label_source_mass_after": "",
            "per_label_removed_source_mass": "",
            "candidate_level_influence_mass": "",
            "evidence_status": "per_label_source_attention_not_landed",
        })
    _write_csv(out_dir / "per_label_removed_source_mass.csv", label_rows)

    overlap_rows = []
    for candidate, target in label_map.items():
        overlap_rows.append({
            "candidate": candidate,
            "target_proxy": target,
            "jaccard_affected_mask_sky": "explainability_missing",
            "jaccard_affected_mask_high_influence_anomaly": "explainability_missing",
            "jaccard_affected_mask_static_anchor": "explainability_missing",
            "affected_token_count": "explainability_missing",
            "evidence_status": "candidate_affected_source_mask_not_landed",
        })
    _write_csv(out_dir / "action_mask_overlap.csv", overlap_rows)

    attention_rows = _scan_attention_mass(args.root)
    _write_csv(out_dir / "scalar_attention_mass_rows.csv", attention_rows)
    copied = _copy_proxy_overlays(args.v39_report, out_dir, selected_chunks)

    best_a2 = min(
        [r for r in attribution if r["candidate"] == "READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT"],
        key=lambda r: _finite(r["ATE_delta_vs_base"], float("inf")),
        default=None,
    )
    best_a4 = min(
        [r for r in attribution if r["candidate"] == "READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH"],
        key=lambda r: _finite(r["intersection_200_300_delta_vs_base"], float("inf")),
        default=None,
    )
    sky_proven = False
    decision = "B_general_high_influence_anomaly_preferred"
    reason = "READ_A2 has the safer v40 full short-ATE behavior, while READ_A4 has strong stress-window signal but regresses downstream in v40."
    if attention_rows:
        reason += " Supplemental h3 scalar attention-mass rows landed, but per-label spatial affected masks remain unavailable."
    else:
        reason += " Supplemental scalar attention-mass rows are not available yet."

    summary = {
        "selected_chunks": selected_chunks,
        "mechanism_decision": decision,
        "sky_causality_proven": sky_proven,
        "best_READ_A2_ATE_delta": best_a2.get("ATE_delta_vs_base") if best_a2 else None,
        "best_READ_A4_stress_delta": best_a4.get("intersection_200_300_delta_vs_base") if best_a4 else None,
        "scalar_attention_mass_rows": len(attention_rows),
        "proxy_overlays_copied": len(copied),
        "evidence_level": "candidate_level_metrics_plus_proxy_overlays; per-label spatial attention and affected masks missing",
        "reason": reason,
    }
    _write_json(out_dir / "v41_read_mechanism_summary.json", summary)

    lines = [
        "# v41 READ A2/A4 Causality Report",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "Decision:",
        "",
        "```text",
        reason,
        "Sky-specific causality is not proven because per-label source-attention maps and affected source masks are not landed.",
        "The continuation candidate should prioritize general high-influence anomaly READ (R1), with R2 kept as a diagnostic/negative contrast.",
        "```",
        "",
        "Boundary:",
        "",
        "```text",
        "RGB / semantic mask / appearance anomaly overlays are proxy visualizations copied from v39.",
        "D_g maps, source attention before/after maps, candidate affected masks, and static-anchor maps are not reconstructed.",
        "```",
        "",
    ]
    (out_dir / "read_a4_sky_causality_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "READ_A2_A4_attribution_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
