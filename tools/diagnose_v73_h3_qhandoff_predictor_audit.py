#!/usr/bin/env python3
"""Audit whether H3 handoff signals predict Phase-E chunk success."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


FEATURES = [
    "q_handoff",
    "overlap_residual",
    "qscale_factor",
    "effective_alpha",
    "stable_mean",
    "risk_mean",
    "remaining_valid_ratio",
    "reset_relative_index",
    "chunk_norm",
    "scale",
    "component_consistency_proxy",
    "component_top_mass_ratio",
    "component_count_norm",
    "component_stable_variance",
    "component_risk_variance",
]

LABELS = ["head_tail_pass", "overlap_pass"]


def _read_trace_row(path: Path) -> Dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty trace: {path}")
    return rows[-1]


def _component_consistency_features(trace: Dict[str, Any]) -> Dict[str, float]:
    """Compute an auditable RADIO component consistency proxy from sidecar tensors.

    The current online traces do not save per-component Sim(3) fits. This proxy is
    therefore diagnostic-only: it measures component-internal semantic/temporal
    consistency, not an actual component transform residual.
    """
    path_raw = trace.get("semantic_merge_radio_component_path")
    if not path_raw:
        return {
            "component_consistency_proxy": 0.0,
            "component_top_mass_ratio": 0.0,
            "component_count_norm": 0.0,
            "component_stable_variance": 0.0,
            "component_risk_variance": 0.0,
        }
    path = Path(str(path_raw))
    if not path.exists():
        return {
            "component_consistency_proxy": 0.0,
            "component_top_mass_ratio": 0.0,
            "component_count_norm": 0.0,
            "component_stable_variance": 0.0,
            "component_risk_variance": 0.0,
        }
    sidecar = torch.load(path, map_location="cpu")
    component = sidecar.get("object_component_id")
    if not torch.is_tensor(component):
        return {
            "component_consistency_proxy": 0.0,
            "component_top_mass_ratio": 0.0,
            "component_count_norm": 0.0,
            "component_stable_variance": 0.0,
            "component_risk_variance": 0.0,
        }
    component = component.long()
    valid = component >= 0
    if not bool(valid.any().item()):
        return {
            "component_consistency_proxy": 0.0,
            "component_top_mass_ratio": 0.0,
            "component_count_norm": 0.0,
            "component_stable_variance": 0.0,
            "component_risk_variance": 0.0,
        }

    def _tensor(name: str, default: float) -> torch.Tensor:
        value = sidecar.get(name)
        if torch.is_tensor(value):
            return value.float().clamp(0.0, 1.0)
        return torch.full_like(component, float(default), dtype=torch.float32)

    conf = _tensor("radio_confidence", 1.0)
    temporal = _tensor("temporal_stability", 0.0)
    interior = _tensor("object_interior_score", 0.0)
    boundary = _tensor("object_boundary_score", 0.0)
    static = _tensor("radio_static_score", 0.0)
    dynamic = _tensor("radio_dynamic_score", 0.0)
    lowtrust = _tensor("radio_lowtrust_score", 0.0)
    sky = _tensor("radio_sky_context_score", 0.0)

    stable = (conf * temporal * (0.45 * static + 0.35 * interior + 0.20 * (1.0 - boundary))).clamp(0.0, 1.0)
    risk = torch.maximum(torch.maximum(dynamic, 0.50 * lowtrust), torch.maximum(0.25 * sky, boundary * dynamic)).clamp(0.0, 1.0)
    component_flat = component[valid].reshape(-1)
    stable_flat = stable[valid].reshape(-1)
    risk_flat = risk[valid].reshape(-1)
    conf_flat = conf[valid].reshape(-1)
    temporal_flat = temporal[valid].reshape(-1)
    interior_flat = interior[valid].reshape(-1)
    boundary_flat = boundary[valid].reshape(-1)

    unique, inverse, counts = torch.unique(component_flat, return_inverse=True, return_counts=True)
    count_f = counts.float()
    comp_count = int(unique.numel())
    top_mass_ratio = float(count_f.max().item() / max(1.0, float(count_f.sum().item())))
    weight = count_f / count_f.sum().clamp_min(1.0)

    stable_means = torch.zeros(comp_count, dtype=torch.float32)
    risk_means = torch.zeros(comp_count, dtype=torch.float32)
    stable_sq_means = torch.zeros(comp_count, dtype=torch.float32)
    risk_sq_means = torch.zeros(comp_count, dtype=torch.float32)
    consistency_means = torch.zeros(comp_count, dtype=torch.float32)
    consistency = (conf_flat * temporal_flat * interior_flat * (1.0 - boundary_flat)).clamp(0.0, 1.0)
    stable_means.scatter_add_(0, inverse, stable_flat)
    risk_means.scatter_add_(0, inverse, risk_flat)
    stable_sq_means.scatter_add_(0, inverse, stable_flat * stable_flat)
    risk_sq_means.scatter_add_(0, inverse, risk_flat * risk_flat)
    consistency_means.scatter_add_(0, inverse, consistency)
    stable_means = stable_means / count_f.clamp_min(1.0)
    risk_means = risk_means / count_f.clamp_min(1.0)
    stable_var = (stable_sq_means / count_f.clamp_min(1.0) - stable_means * stable_means).clamp_min(0.0)
    risk_var = (risk_sq_means / count_f.clamp_min(1.0) - risk_means * risk_means).clamp_min(0.0)
    consistency_means = consistency_means / count_f.clamp_min(1.0)

    return {
        "component_consistency_proxy": float((consistency_means * weight).sum().item()),
        "component_top_mass_ratio": top_mass_ratio,
        "component_count_norm": float(min(1.0, comp_count / 128.0)),
        "component_stable_variance": float((stable_var * weight).sum().item()),
        "component_risk_variance": float((risk_var * weight).sum().item()),
    }


def _auc_score(values: Iterable[float], labels: Iterable[bool]) -> Optional[float]:
    pairs = list(zip(values, labels))
    pos = [value for value, label in pairs if label]
    neg = [value for value, label in pairs if not label]
    if not pos or not neg:
        return None
    wins = 0
    ties = 0
    for p_value in pos:
        for n_value in neg:
            if p_value > n_value:
                wins += 1
            elif p_value == n_value:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _build_rows(root: Path, chunks: List[int], candidate: str, decisions_name: str) -> List[Dict[str, Any]]:
    decisions_path = root / decisions_name
    decisions: Dict[int, Dict[str, str]] = {}
    with decisions_path.open(newline="") as f:
        for row in csv.DictReader(f):
            decisions[int(row["chunk"])] = row
    rows: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        trace = _read_trace_row(root / f"chunk{chunk:02d}" / candidate / "merge_state_trace.jsonl")
        component_features = _component_consistency_features(trace)
        decision = decisions[chunk]
        rows.append(
            {
                "chunk": chunk,
                "reset_relative_index": index / max(1, len(chunks) - 1),
                "chunk_norm": chunk / max(chunks),
                "q_handoff": trace.get("semantic_merge_radio_handoff_qscale_observability"),
                "overlap_residual": trace.get("semantic_merge_overlap_residual"),
                "qscale_factor": trace.get("semantic_merge_qscale_factor"),
                "effective_alpha": trace.get("semantic_merge_effective_blend_alpha"),
                "stable_mean": trace.get("semantic_merge_radio_component_stable_mean"),
                "risk_mean": trace.get("semantic_merge_radio_component_risk_mean"),
                "remaining_valid_ratio": trace.get("semantic_merge_remaining_valid_ratio"),
                "scale": trace.get("semantic_merge_scale"),
                **component_features,
                "head_tail_improvement": float(decision["head_tail_improvement_vs_baseline_ratio"]),
                "head_tail_pass": decision["head_tail_phaseE_chunk_pass"] == "True",
                "head_tail_beats_controls": decision["head_tail_beats_controls"] == "True",
                "overlap_improvement": float(decision["overlap_improvement_vs_baseline_ratio"]),
                "overlap_pass": decision["overlap_phaseE_chunk_pass"] == "True",
                "overlap_beats_controls": decision["overlap_beats_controls"] == "True",
            }
        )
    return rows


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for label in LABELS:
        labels = [bool(row[label]) for row in rows]
        positive_count = sum(labels)
        for feature in FEATURES:
            values = [float(row[feature]) for row in rows]
            for direction in ("desc", "asc"):
                ranked_values = values if direction == "desc" else [-value for value in values]
                auc = _auc_score(ranked_values, labels)
                order = sorted(range(len(rows)), key=lambda i: ranked_values[i], reverse=True)
                top5 = order[:5]
                top5_precision = sum(1 for i in top5 if labels[i]) / 5.0
                summary_rows.append(
                    {
                        "label": label,
                        "positive_count": positive_count,
                        "feature": feature,
                        "direction": direction,
                        "auc": auc,
                        "top5_precision": top5_precision,
                        "top5_chunks": ",".join(str(rows[i]["chunk"]) for i in top5),
                        "passes_h3_predictor_gate": bool(
                            positive_count >= 4
                            and auc is not None
                            and auc >= 0.70
                            and top5_precision >= 0.40
                        ),
                    }
                )
    summary_rows.sort(key=lambda row: (row["label"], -(row["auc"] if row["auc"] is not None else -1.0), -row["top5_precision"]))
    return summary_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--chunks", required=True, help="Comma-separated target chunk ids.")
    parser.add_argument("--candidate", default="radio_qscale")
    parser.add_argument("--decisions-name", default="phaseE_multichunk_decisions_full11.csv")
    parser.add_argument("--out-prefix", default="h3_qhandoff_predictor")
    args = parser.parse_args()

    root = Path(args.root)
    chunks = [int(x) for x in args.chunks.split(",") if x.strip()]
    rows = _build_rows(root, chunks, str(args.candidate), str(args.decisions_name))
    summary_rows = _summarize(rows)
    best_by_label = {label: next(row for row in summary_rows if row["label"] == label) for label in LABELS}

    rows_path = root / f"{args.out_prefix}_rows.csv"
    summary_path = root / f"{args.out_prefix}_summary.csv"
    json_path = root / f"{args.out_prefix}_summary.json"
    _write_csv(rows_path, rows)
    _write_csv(summary_path, summary_rows)
    payload = {
        "schema": "v73_h3_qhandoff_predictor_audit_v1",
        "root": str(root),
        "target_chunks": chunks,
        "row_count": len(rows),
        "labels": {
            label: {
                "positive_count": sum(bool(row[label]) for row in rows),
                "best_single_feature": best_by_label[label],
            }
            for label in LABELS
        },
        "gate_rule": "H3 predictor deploy requires AUC >=0.70, top5 precision >=0.40, and >=4/11 positive chunks; this audit uses single-feature rank diagnostics only, not a deployed selector.",
        "deployable": False,
        "reason": "positive_count is below 4/11 for both head_tail_pass and overlap_pass; Q_handoff features are diagnostic only.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote_rows={rows_path}")
    print(f"wrote_summary_csv={summary_path}")
    print(f"wrote_summary_json={json_path}")


if __name__ == "__main__":
    main()
