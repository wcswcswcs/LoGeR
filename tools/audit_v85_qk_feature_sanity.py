#!/usr/bin/env python3
"""Audit v85 Phase2 Q/K feature sanity by layer/head."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F


DEFAULT_FEATURE_DIR = Path("results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase2_qk_feature_bank")
DEFAULT_ANCHOR_ROWS = Path(
    "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase1_anchor_pair_universe/anchor_pair_rows.csv"
)
DEFAULT_PHASE1_SUMMARY = Path(
    "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase1_anchor_pair_universe/anchor_pair_sufficiency_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--anchor-rows", type=Path, default=DEFAULT_ANCHOR_ROWS)
    parser.add_argument("--phase1-summary", type=Path, default=DEFAULT_PHASE1_SUMMARY)
    parser.add_argument("--min-feature-availability", type=float, default=0.90)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def is_labelled_nonstress(row: Mapping[str, Any]) -> bool:
    return row.get("case_label") in {"bad", "good"} and row.get("quality_label") != "low_conf_stress"


def matrix_rank(features: torch.Tensor) -> int:
    if features.numel() == 0 or features.ndim != 2:
        return 0
    return int(torch.linalg.matrix_rank(features.float()).item())


def audit_group(
    key: tuple[str, str],
    indices: list[int],
    index_rows: list[dict[str, str]],
    q_features: torch.Tensor,
    k_features: torch.Tensor,
    labelled_nonstress_denominator: int,
    min_feature_availability: float,
) -> dict[str, Any]:
    q = q_features[indices].float()
    k = k_features[indices].float()
    labelled_indices = [
        i for i in indices if is_labelled_nonstress(index_rows[i])
    ]
    q_labelled = q_features[labelled_indices].float() if labelled_indices else torch.empty(0, q.shape[-1])
    k_labelled = k_features[labelled_indices].float() if labelled_indices else torch.empty(0, k.shape[-1])
    availability = len(labelled_indices) / labelled_nonstress_denominator if labelled_nonstress_denominator else 0.0
    q_nan_count = int(torch.isnan(q_labelled).sum().item()) if q_labelled.numel() else 0
    k_nan_count = int(torch.isnan(k_labelled).sum().item()) if k_labelled.numel() else 0
    q_zero_count = int((torch.linalg.norm(q_labelled, dim=1) <= 1e-12).sum().item()) if q_labelled.numel() else 0
    k_zero_count = int((torch.linalg.norm(k_labelled, dim=1) <= 1e-12).sum().item()) if k_labelled.numel() else 0
    q_rank = matrix_rank(q_labelled)
    k_rank = matrix_rank(k_labelled)
    dim = int(q.shape[-1]) if q.ndim == 2 and q.shape else 0
    min_rank = min(4, max(1, dim // 2))
    if q_labelled.numel() and k_labelled.numel():
        cos = F.cosine_similarity(q_labelled, k_labelled, dim=1)
        residual = torch.linalg.norm(q_labelled - k_labelled, dim=1)
        q_norm = torch.linalg.norm(q_labelled, dim=1)
        k_norm = torch.linalg.norm(k_labelled, dim=1)
    else:
        cos = residual = q_norm = k_norm = torch.empty(0)
    eligible = (
        availability >= min_feature_availability
        and q_rank >= min_rank
        and k_rank >= min_rank
        and q_nan_count == 0
        and k_nan_count == 0
        and q_zero_count == 0
        and k_zero_count == 0
    )
    feature_sources = sorted({index_rows[i].get("feature_source_path", "") for i in indices if index_rows[i].get("feature_source_path")})
    schemas = sorted({index_rows[i].get("feature_schema", "") for i in indices if index_rows[i].get("feature_schema")})
    return {
        "seq": "*",
        "prev_chunk": "*",
        "curr_chunk": "*",
        "pair_id": "*",
        "layer_id": key[0],
        "head_id": key[1],
        "feature_dim_raw": dim,
        "feature_dim_projected": dim,
        "projection_method": "preexisting_pca_swa_dump",
        "labelled_nonstress_available_rows": len(labelled_indices),
        "labelled_nonstress_denominator": labelled_nonstress_denominator,
        "feature_availability": availability,
        "q_norm_mean": float(q_norm.mean().item()) if q_norm.numel() else None,
        "k_norm_mean": float(k_norm.mean().item()) if k_norm.numel() else None,
        "q_rank": q_rank,
        "k_rank": k_rank,
        "q_nan_count": q_nan_count,
        "k_nan_count": k_nan_count,
        "q_zero_count": q_zero_count,
        "k_zero_count": k_zero_count,
        "cosine_qk_identity_mean": float(cos.mean().item()) if cos.numel() else None,
        "identity_residual_mean": float(residual.mean().item()) if residual.numel() else None,
        "feature_source_path": "multiple:%d" % len(feature_sources) if len(feature_sources) > 1 else (feature_sources[0] if feature_sources else ""),
        "feature_schema": ",".join(schemas),
        "authority_source": "direct_pca_swa_q_cache_k_dump",
        "eligible_layer_head": eligible,
    }


def main() -> None:
    args = parse_args()
    feature_path = args.feature_dir / "qk_anchor_features.pt"
    index_path = args.feature_dir / "qk_anchor_feature_index.csv"
    payload = torch_load(feature_path)
    index_rows = read_csv(index_path)
    anchor_rows = read_csv(args.anchor_rows)
    phase1_summary = json.loads(args.phase1_summary.read_text(encoding="utf-8")) if args.phase1_summary.exists() else {}

    q_features = payload["q_features"].float()
    k_features = payload["k_features"].float()
    labelled_nonstress_denominator = sum(1 for row in anchor_rows if is_labelled_nonstress(row))
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(index_rows):
        groups[(str(row.get("layer_id")), str(row.get("head_id")))].append(i)

    rows = [
        audit_group(
            key,
            indices,
            index_rows,
            q_features,
            k_features,
            labelled_nonstress_denominator,
            args.min_feature_availability,
        )
        for key, indices in sorted(groups.items())
    ]
    eligible = [row for row in rows if row["eligible_layer_head"]]
    phase2_feature_gate_pass = bool(eligible)
    phase1_gate_pass = bool(phase1_summary.get("phase1_gate_pass", False))
    summary = {
        "phase": "Phase2_qk_feature_sanity",
        "phase2_feature_gate_pass": phase2_feature_gate_pass,
        "eligible_layer_head_count": len(eligible),
        "eligible_layer_heads": [
            {"layer_id": row["layer_id"], "head_id": row["head_id"]} for row in eligible
        ],
        "feature_entry_count": len(index_rows),
        "q_feature_shape": list(q_features.shape),
        "k_feature_shape": list(k_features.shape),
        "labelled_nonstress_denominator": labelled_nonstress_denominator,
        "phase1_gate_pass": phase1_gate_pass,
        "phase1_fail_reasons": phase1_summary.get("fail_reasons", []),
        "can_enter_phase3_alignment": bool(phase2_feature_gate_pass and phase1_gate_pass),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "notes": [
            "Feature sanity can pass while Phase1 anchor support still blocks Phase3.",
            "Authority source is direct PCA SWA Q/cache-K dump; route mass is not available here.",
        ],
    }
    write_csv(args.feature_dir / "feature_sanity_by_layer_head.csv", rows)
    write_json(args.feature_dir / "feature_sanity_summary.json", summary)
    write_report(args.feature_dir / "feature_sanity_report.md", summary, rows)
    print(f"phase2_feature_gate_pass={str(phase2_feature_gate_pass).lower()}")
    print(f"eligible_layer_head_count={len(eligible)}")
    print(f"can_enter_phase3_alignment={str(summary['can_enter_phase3_alignment']).lower()}")
    print(f"phase1_gate_pass={str(phase1_gate_pass).lower()}")
    print(f"phase1_fail_reasons={','.join(summary['phase1_fail_reasons']) if summary['phase1_fail_reasons'] else 'none'}")


def write_report(path: Path, summary: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> None:
    lines = [
        "# Phase2 Q/K Feature Sanity Audit",
        "",
        f"- Phase2 feature gate pass: `{summary['phase2_feature_gate_pass']}`",
        f"- Eligible layer/head count: `{summary['eligible_layer_head_count']}`",
        f"- Phase1 gate pass: `{summary['phase1_gate_pass']}`",
        f"- Can enter Phase3 alignment: `{summary['can_enter_phase3_alignment']}`",
        "",
        "## Layer/Head Rows",
        "",
        "| layer | head | availability | q_rank | k_rank | cos_mean | residual_mean | eligible |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {layer_id} | {head_id} | {feature_availability} | {q_rank} | {k_rank} | "
            "{cosine_qk_identity_mean} | {identity_residual_mean} | {eligible_layer_head} |".format(**row)
        )
    lines.extend(["", "## Notes", ""])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
