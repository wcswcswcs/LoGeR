#!/usr/bin/env python3
"""Summarize ACL2 v67 Phase-2 carrier discovery artifacts."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loger.pipeline.semantic_prior_generator import SEMANTIC_FINE_ID_TO_LABEL


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out: Dict[str, Any] = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True, ensure_ascii=False)
                out[field] = "" if value is None else value
            writer.writerow(out)


def _float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_summary(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _result_ate(run_dir: str) -> Optional[float]:
    p = Path(run_dir) / "results_sim3" / "results_ate.txt"
    if not p.exists():
        return None
    vals = p.read_text(encoding="utf-8").strip().split()
    if not vals:
        return None
    return _float(vals[0])


def _run_metric(segment: Dict[str, Any], run: str, key: str) -> Optional[float]:
    node = ((segment.get("runs") or {}).get(run) or {})
    if key.startswith("interval:"):
        interval = key.split(":", 1)[1]
        return _float((node.get("interval_ate_rmse_m") or {}).get(interval))
    if key == "overall_ate":
        return _float((node.get("overall_local_sim3") or {}).get("ate_rmse_m"))
    if key == "overlap_to_future":
        return _float((node.get("chunk_overlap_to_future_ate") or {}).get("mean"))
    if key == "scale_cv":
        return _float((node.get("chunk_scale_cv") or {}).get("mean"))
    if key == "head_to_tail":
        return _float((node.get("chunk_head_to_tail_ate") or {}).get("mean"))
    return None


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a - b)


def _build_registry(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        comp = row.get("semantic_composition") or []
        top_label = comp[0] if isinstance(comp, list) and comp else {}
        out.append({
            "chunk_idx": row.get("chunk_idx"),
            "start_frame": row.get("start_frame"),
            "end_frame": row.get("end_frame"),
            "token_group_name": row.get("token_group_name"),
            "control_of": row.get("control_of"),
            "supported": row.get("supported"),
            "support_reason": row.get("support_reason"),
            "eligible_tokens": row.get("eligible_tokens"),
            "selected_tokens": row.get("selected_tokens"),
            "selected_ratio": row.get("selected_ratio"),
            "D_mean": row.get("D_mean"),
            "D_q80": row.get("D_q80"),
            "D_q90": row.get("D_q90"),
            "source_attention_mean": row.get("source_attention_mean"),
            "source_attention_status": row.get("source_attention_status"),
            "spatial_cell_coverage": row.get("spatial_cell_coverage"),
            "frame_coverage": row.get("frame_coverage"),
            "head_overlap_selected_frac": row.get("head_overlap_selected_frac"),
            "tail_overlap_selected_frac": row.get("tail_overlap_selected_frac"),
            "group_structure_frac": row.get("group_structure_frac"),
            "group_static_frac": row.get("group_static_frac"),
            "group_movable_frac": row.get("group_movable_frac"),
            "group_lowstuff_frac": row.get("group_lowstuff_frac"),
            "group_uncertain_frac": row.get("group_uncertain_frac"),
            "top_label_id": top_label.get("label_id") if isinstance(top_label, dict) else None,
            "top_label_name": top_label.get("label_name") if isinstance(top_label, dict) else None,
            "top_label_frac": top_label.get("frac") if isinstance(top_label, dict) else None,
            "semantic_composition": comp,
        })
    return out


def _global_label_fracs(carrier_jsonl: Path, token_summary_globs: Optional[List[str]] = None) -> Dict[int, Dict[int, float]]:
    root = carrier_jsonl.parent
    out: Dict[int, Dict[int, float]] = {}
    paths: List[Path] = []
    if token_summary_globs:
        for pattern in token_summary_globs:
            paths.extend(Path(p) for p in glob.glob(pattern))
    else:
        paths = list(root.glob("token_summary_chunk_*.pt"))
    for path in sorted(set(paths)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        chunk_idx = int(payload.get("chunk_idx"))
        labels = payload.get("L_patch")
        eligible = payload.get("eligible_patch")
        if labels is None:
            continue
        labels = labels.detach().cpu().long().reshape(-1)
        if eligible is not None and int(eligible.numel()) == int(labels.numel()):
            mask = eligible.detach().cpu().bool().reshape(-1)
            labels = labels[mask]
        if labels.numel() <= 0:
            continue
        uniq, counts = torch.unique(labels, return_counts=True)
        denom = max(int(labels.numel()), 1)
        out[chunk_idx] = {int(u.item()): float(c.item() / denom) for u, c in zip(uniq, counts)}
    return out


def _build_enrichment(
    carrier_jsonl: Path,
    rows: List[Dict[str, Any]],
    *,
    token_summary_globs: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    base = _global_label_fracs(carrier_jsonl, token_summary_globs=token_summary_globs)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not row.get("supported") or int(row.get("selected_tokens") or 0) <= 0:
            continue
        chunk_idx = int(row.get("chunk_idx"))
        global_fracs = base.get(chunk_idx, {})
        for item in row.get("semantic_composition") or []:
            if not isinstance(item, dict):
                continue
            label_id = int(item.get("label_id"))
            frac = _float(item.get("frac")) or 0.0
            bg = float(global_fracs.get(label_id, 0.0))
            enrich = (frac / bg) if bg > 0 else None
            out.append({
                "chunk_idx": chunk_idx,
                "token_group_name": row.get("token_group_name"),
                "control_of": row.get("control_of"),
                "selected_tokens": row.get("selected_tokens"),
                "label_id": label_id,
                "label_name": item.get("label_name") or SEMANTIC_FINE_ID_TO_LABEL.get(label_id, str(label_id)),
                "group_frac": frac,
                "chunk_global_frac": bg,
                "enrichment": enrich,
                "semantic_explanation_positive": bool(enrich is not None and enrich >= 2.0),
            })
    return out


def _build_effects(segment: Dict[str, Any], action: Dict[str, Any]) -> List[Dict[str, Any]]:
    run_to_group = {
        "r1_fine": "R1_SEMZ_FINE_READ_CUE",
        "r1_random": "R1_RANDOM_SAME_MASS_SEMZ_FINE",
        "r2a": "G7_SKY_HIGH_D",
        "r2b": "G7_SKY_HIGH_D",
        "r2c": "G7_SKY_HIGH_D",
        "r2b_shuffled": "G14_SHUFFLED_SEMANTIC_FOR_G7_SKY_HIGH_D",
        "r3": "G3_HIGH_D_Q80",
        "r3_random": "G13_RANDOM_SAME_MASS_FOR_G3_HIGH_D_Q80",
    }
    runs = segment.get("runs") or {}
    action_runs = action.get("runs") or {}
    effects: List[Dict[str, Any]] = []
    for run in sorted(runs.keys()):
        group = run_to_group.get(run, "baseline_or_unmapped")
        row: Dict[str, Any] = {
            "run": run,
            "token_group_name": group,
            "intervention_path": "READ_frame_attention" if group != "baseline_or_unmapped" else "baseline",
            "intervention_strength": "",
            "ATE_96_or_704_delta": _delta(_run_metric(segment, run, "overall_ate"), _run_metric(segment, "h35like", "overall_ate")),
            "overall_ate": _run_metric(segment, run, "overall_ate"),
            "local_window_200_300_ate": _run_metric(segment, run, "interval:200:300"),
            "local_window_290_390_ate": _run_metric(segment, run, "interval:290:390"),
            "interval_400_600_ate": _run_metric(segment, run, "interval:400:600"),
            "future_after_overlap_ate": _run_metric(segment, run, "overlap_to_future"),
            "tail_after_overlap_delta": "",
            "head_to_tail_ate": _run_metric(segment, run, "head_to_tail"),
            "scale_cv": _run_metric(segment, run, "scale_cv"),
            "delta_vs_h35like_200_300": _delta(_run_metric(segment, run, "interval:200:300"), _run_metric(segment, "h35like", "interval:200:300")),
            "delta_vs_h35like_290_390": _delta(_run_metric(segment, run, "interval:290:390"), _run_metric(segment, "h35like", "interval:290:390")),
            "delta_vs_h35like_overlap_to_future": _delta(_run_metric(segment, run, "overlap_to_future"), _run_metric(segment, "h35like", "overlap_to_future")),
            "delta_vs_h35like_scale_cv": _delta(_run_metric(segment, run, "scale_cv"), _run_metric(segment, "h35like", "scale_cv")),
            "affected_mass_before": (action_runs.get(run) or {}).get("frame_attention_mass_removed_before_mean"),
            "affected_mass_after": (action_runs.get(run) or {}).get("frame_attention_mass_removed_after_mean"),
            "source_attention_mean": "",
            "D_mean": "",
            "semantic_composition": "",
            "beats_random": "",
            "beats_shuffled": "",
        }
        if run == "r2a":
            shuf = _run_metric(segment, "r2b_shuffled", "overall_ate")
            row["beats_shuffled"] = bool(row["overall_ate"] is not None and shuf is not None and row["overall_ate"] < shuf)
        if run == "r3":
            rand = _run_metric(segment, "r3_random", "overall_ate")
            row["beats_random"] = bool(row["overall_ate"] is not None and rand is not None and row["overall_ate"] < rand)
        effects.append(row)
    return effects


def _write_report(out_dir: Path, registry: List[Dict[str, Any]], enrich: List[Dict[str, Any]], effects: List[Dict[str, Any]]) -> None:
    unsupported = [r for r in registry if str(r.get("supported")).lower() in {"false", "0"}]
    supported = [r for r in registry if str(r.get("supported")).lower() not in {"false", "0"}]
    top_enrich = sorted(
        [r for r in enrich if _float(r.get("enrichment")) is not None],
        key=lambda r: float(r["enrichment"]),
        reverse=True,
    )[:12]
    lines = [
        "# ACL2 v67 Phase 2 Carrier Discovery Report",
        "",
        "Delta convention: candidate - baseline; negative means improvement.",
        "",
        f"Registry rows: {len(registry)}",
        f"Supported rows: {len(supported)}",
        f"Unsupported rows: {len(unsupported)}",
        "",
        "Unsupported official groups are retained in the registry instead of being silently skipped.",
        "At this point source-attention top-token groups are unavailable because no per-source-token attention map is persisted.",
        "",
        "## Top Semantic Enrichment",
    ]
    if top_enrich:
        for row in top_enrich:
            lines.append(
                f"- chunk {row['chunk_idx']} {row['token_group_name']} label={row['label_name']} "
                f"group_frac={float(row['group_frac']):.6f} global_frac={float(row['chunk_global_frac']):.6f} "
                f"enrichment={float(row['enrichment']):.3f}"
            )
    else:
        lines.append("- No enrichment rows available.")
    lines += ["", "## Existing Intervention Effects"]
    for row in effects:
        if row.get("token_group_name") == "baseline_or_unmapped":
            continue
        lines.append(
            f"- {row['run']} / {row['token_group_name']}: overall={row.get('overall_ate')} "
            f"delta_vs_h35like={row.get('ATE_96_or_704_delta')} "
            f"delta_200_300={row.get('delta_vs_h35like_200_300')} "
            f"delta_overlap_future={row.get('delta_vs_h35like_overlap_to_future')} "
            f"mass_before={row.get('affected_mass_before')}"
        )
    (out_dir / "carrier_discovery_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier-jsonl", required=True)
    parser.add_argument("--segment-summary", default="")
    parser.add_argument("--action-summary", default="")
    parser.add_argument("--token-summary-glob", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    carrier_jsonl = Path(args.carrier_jsonl)
    out_dir = Path(args.out_dir)
    rows = _read_jsonl(carrier_jsonl)
    registry = _build_registry(rows)
    enrichment = _build_enrichment(carrier_jsonl, rows, token_summary_globs=args.token_summary_glob)
    effects = _build_effects(_load_summary(args.segment_summary), _load_summary(args.action_summary))

    _write_csv(
        out_dir / "token_group_registry.csv",
        registry,
        [
            "chunk_idx", "start_frame", "end_frame", "token_group_name", "control_of",
            "supported", "support_reason", "eligible_tokens", "selected_tokens",
            "selected_ratio", "D_mean", "D_q80", "D_q90", "source_attention_mean",
            "source_attention_status", "spatial_cell_coverage", "frame_coverage",
            "head_overlap_selected_frac", "tail_overlap_selected_frac",
            "group_structure_frac", "group_static_frac", "group_movable_frac",
            "group_lowstuff_frac", "group_uncertain_frac", "top_label_id",
            "top_label_name", "top_label_frac", "semantic_composition",
        ],
    )
    _write_csv(
        out_dir / "semantic_enrichment_of_high_impact_tokens.csv",
        enrichment,
        [
            "chunk_idx", "token_group_name", "control_of", "selected_tokens",
            "label_id", "label_name", "group_frac", "chunk_global_frac",
            "enrichment", "semantic_explanation_positive",
        ],
    )
    _write_csv(
        out_dir / "token_group_effects.csv",
        effects,
        [
            "run", "token_group_name", "intervention_path", "intervention_strength",
            "affected_mass_before", "affected_mass_after", "local_window_200_300_ate",
            "local_window_290_390_ate", "interval_400_600_ate", "future_after_overlap_ate",
            "tail_after_overlap_delta", "head_to_tail_ate", "scale_cv", "overall_ate",
            "ATE_96_or_704_delta", "delta_vs_h35like_200_300",
            "delta_vs_h35like_290_390", "delta_vs_h35like_overlap_to_future",
            "delta_vs_h35like_scale_cv", "D_mean", "source_attention_mean",
            "semantic_composition", "beats_random", "beats_shuffled",
        ],
    )
    _write_report(out_dir, registry, enrichment, effects)
    print(json.dumps({
        "registry_rows": len(registry),
        "enrichment_rows": len(enrichment),
        "effect_rows": len(effects),
        "out_dir": str(out_dir),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
