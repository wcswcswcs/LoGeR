#!/usr/bin/env python3
"""Posthoc SWA-stable READ redirection corridor probe for ACL2 v79."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_v79_read_swa_overlap_alignment_probe import (
    DEFAULT_LEDGER,
    DEFAULT_PHASE3_ROOT,
    DEFAULT_READ_DUMP,
    _feature_path,
    _first_primary_pair_row,
    _jsonable,
    _load_pt,
    _patch_overlap_from_feature,
    _positive_stats,
    _read_overlap,
    _stats,
    _write_csv,
)


def _finite_flat(values: torch.Tensor) -> torch.Tensor:
    x = values.detach().cpu().float().reshape(-1)
    return x[torch.isfinite(x)]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    x = values.detach().cpu().float().reshape(-1)
    m = mask.detach().cpu().bool().reshape(-1)
    if int(x.numel()) != int(m.numel()) or not bool(m.any().item()):
        return None
    return float(x[m].mean().item())


def _topk_mask(values: torch.Tensor, k: int, allowed: Optional[torch.Tensor] = None) -> torch.Tensor:
    flat = values.detach().cpu().float().reshape(-1)
    if allowed is None:
        allowed_flat = torch.isfinite(flat)
    else:
        allowed_flat = allowed.detach().cpu().bool().reshape(-1) & torch.isfinite(flat)
    idx = torch.nonzero(allowed_flat, as_tuple=False).reshape(-1)
    out = torch.zeros_like(flat, dtype=torch.bool)
    if int(idx.numel()) <= 0 or int(k) <= 0:
        return out.reshape(values.shape)
    take = min(int(k), int(idx.numel()))
    chosen_local = torch.topk(flat[idx], k=take, largest=True, sorted=False).indices
    out[idx[chosen_local]] = True
    return out.reshape(values.shape)


def _frame_counts(mask: torch.Tensor) -> List[int]:
    m = mask.detach().cpu().bool()
    if m.ndim != 3:
        return []
    return [int(v) for v in m.reshape(int(m.shape[0]), -1).sum(dim=1).tolist()]


def _repeat_counts(mask: torch.Tensor) -> Dict[str, int]:
    m = mask.detach().cpu().bool()
    if m.ndim != 3:
        return {"spatial_repeat_ge2": 0, "spatial_repeat_all": 0}
    repeat = m.sum(dim=0)
    return {
        "spatial_repeat_ge2": int((repeat >= 2).sum().item()),
        "spatial_repeat_all": int((repeat == int(m.shape[0])).sum().item()),
    }


def _relative_delta(candidate: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if candidate is None or baseline is None or abs(float(baseline)) < 1.0e-12:
        return None
    return float((float(candidate) - float(baseline)) / abs(float(baseline)))


def _row_for_mask(
    *,
    name: str,
    mask: torch.Tensor,
    read_active: torch.Tensor,
    read_score: torch.Tensor,
    stable_score: torch.Tensor,
    stable_control: torch.Tensor,
    disagreement_score: torch.Tensor,
    disagreement_control: torch.Tensor,
    random_stable_score: torch.Tensor,
    random_disagreement_score: torch.Tensor,
    dyn: torch.Tensor,
    key_avg: torch.Tensor,
    qk_var: torch.Tensor,
    occlusion: torch.Tensor,
    uncertainty: torch.Tensor,
) -> Dict[str, Any]:
    flat_mask = mask.detach().cpu().bool().reshape(-1)
    flat_read = read_active.detach().cpu().bool().reshape(-1)
    denom = int(flat_mask.numel())
    count = int(flat_mask.sum().item())
    read_inter = flat_mask & flat_read
    read_inter_count = int(read_inter.sum().item())
    repeat_counts = _repeat_counts(mask)
    return {
        "mask": name,
        "count": count,
        "mass": float(count / max(denom, 1)),
        "frame_counts": json.dumps(_frame_counts(mask), ensure_ascii=False),
        "spatial_repeat_ge2": repeat_counts["spatial_repeat_ge2"],
        "spatial_repeat_all": repeat_counts["spatial_repeat_all"],
        "read_intersection_count": read_inter_count,
        "given_read": float(read_inter_count / max(int(flat_read.sum().item()), 1)),
        "read_given_mask": float(read_inter_count / max(count, 1)),
        "read_score_mean": _masked_mean(read_score, flat_mask),
        "stable_score_mean": _masked_mean(stable_score, flat_mask),
        "stable_control_mean": _masked_mean(stable_control, flat_mask),
        "disagreement_score_mean": _masked_mean(disagreement_score, flat_mask),
        "disagreement_control_mean": _masked_mean(disagreement_control, flat_mask),
        "random_stable_score_mean": _masked_mean(random_stable_score, flat_mask),
        "random_disagreement_score_mean": _masked_mean(random_disagreement_score, flat_mask),
        "dyn_mean": _masked_mean(dyn, flat_mask),
        "key_avg_mean": _masked_mean(key_avg, flat_mask),
        "qk_var_mean": _masked_mean(qk_var, flat_mask),
        "occlusion_mean": _masked_mean(occlusion, flat_mask),
        "uncertainty_mean": _masked_mean(uncertainty, flat_mask),
    }


def _improvement_summary(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Optional[float]]:
    fields = [
        "dyn_mean",
        "key_avg_mean",
        "qk_var_mean",
        "disagreement_score_mean",
        "occlusion_mean",
        "uncertainty_mean",
        "stable_score_mean",
    ]
    return {f"{field}_relative_delta": _relative_delta(candidate.get(field), baseline.get(field)) for field in fields}


def build_probe(args: argparse.Namespace) -> Dict[str, Any]:
    read_payload = _load_pt(args.read_dump)
    stable_payload = _load_pt(_feature_path(args.phase3_root, case=args.stable_case, kind="source_replace", layer=args.layer, chunk=args.chunk))
    disagreement_payload = _load_pt(_feature_path(args.phase3_root, case=args.disagreement_case, kind="source_gate", layer=args.layer, chunk=args.chunk))
    random_stable_payload = _load_pt(_feature_path(args.phase3_root, case=args.random_stable_case, kind="source_replace", layer=args.layer, chunk=args.chunk))
    random_disagreement_payload = _load_pt(_feature_path(args.phase3_root, case=args.random_disagreement_case, kind="source_gate", layer=args.layer, chunk=args.chunk))

    overlap_frames = int(stable_payload.get("overlap_frames_effective", args.overlap_frames) or args.overlap_frames)
    read_active = _read_overlap(read_payload, "read_active_q90_patch", overlap_frames).bool()
    read_score = _read_overlap(read_payload, "read_patch_final", overlap_frames).float()
    dyn = _read_overlap(read_payload, "dyn_patch", overlap_frames).float()
    key_avg = _read_overlap(read_payload, "key_avg_patch", overlap_frames).float()
    qk_var = _read_overlap(read_payload, "qk_var_patch", overlap_frames).float()
    occlusion = _read_overlap(read_payload, "occlusion_patch", overlap_frames).float()
    uncertainty = _read_overlap(read_payload, "uncertainty_patch", overlap_frames).float()

    stable_score = _patch_overlap_from_feature(stable_payload, "score_overlap")
    stable_control = _patch_overlap_from_feature(stable_payload, "control_overlap")
    disagreement_score = _patch_overlap_from_feature(disagreement_payload, "score_overlap")
    disagreement_control = _patch_overlap_from_feature(disagreement_payload, "control_overlap")
    random_stable_score = _patch_overlap_from_feature(random_stable_payload, "score_overlap")
    random_disagreement_score = _patch_overlap_from_feature(random_disagreement_payload, "score_overlap")

    min_frames = min(
        int(read_active.shape[0]),
        int(stable_score.shape[0]),
        int(disagreement_score.shape[0]),
        int(random_stable_score.shape[0]),
        int(random_disagreement_score.shape[0]),
    )
    read_active = read_active[:min_frames]
    read_score = read_score[:min_frames]
    dyn = dyn[:min_frames]
    key_avg = key_avg[:min_frames]
    qk_var = qk_var[:min_frames]
    occlusion = occlusion[:min_frames]
    uncertainty = uncertainty[:min_frames]
    stable_score = stable_score[:min_frames]
    stable_control = stable_control[:min_frames]
    disagreement_score = disagreement_score[:min_frames]
    disagreement_control = disagreement_control[:min_frames]
    random_stable_score = random_stable_score[:min_frames]
    random_disagreement_score = random_disagreement_score[:min_frames]

    stable_positive_stats = _positive_stats(stable_score)
    random_stable_positive_stats = _positive_stats(random_stable_score)
    disagreement_stats = _stats(disagreement_score)
    random_disagreement_stats = _stats(random_disagreement_score)
    read_count = int(read_active.sum().item())

    stable_positive = stable_score > 0.0
    random_stable_positive = random_stable_score > 0.0
    low_boundary50 = disagreement_score <= float(disagreement_stats["q50"])
    low_boundary25 = disagreement_score <= float(disagreement_stats["q25"])
    random_low_boundary50 = random_disagreement_score <= float(random_disagreement_stats["q50"])
    random_low_boundary25 = random_disagreement_score <= float(random_disagreement_stats["q25"])

    masks = {
        "current_read_active_q90": read_active,
        "swa_stable_positive": stable_positive,
        "swa_stable_positive_top_read_mass": _topk_mask(stable_score, read_count, stable_positive),
        "swa_stable_positive_top50": stable_score >= float(stable_positive_stats["q50"]),
        "swa_stable_positive_top25": stable_score >= float(stable_positive_stats["q75"]),
        "swa_stable_low_boundary50": stable_positive & low_boundary50,
        "swa_stable_low_boundary25": stable_positive & low_boundary25,
        "swa_stable_low_boundary50_top_read_mass": _topk_mask(stable_score, read_count, stable_positive & low_boundary50),
        "swa_stable_low_boundary25_top_read_mass": _topk_mask(stable_score, read_count, stable_positive & low_boundary25),
        "random_stable_positive": random_stable_positive,
        "random_stable_positive_top_read_mass": _topk_mask(random_stable_score, read_count, random_stable_positive),
        "random_stable_positive_top50": random_stable_score >= float(random_stable_positive_stats["q50"]),
        "random_stable_positive_top25": random_stable_score >= float(random_stable_positive_stats["q75"]),
        "random_stable_low_boundary50": random_stable_positive & random_low_boundary50,
        "random_stable_low_boundary25": random_stable_positive & random_low_boundary25,
        "random_stable_low_boundary50_top_read_mass": _topk_mask(random_stable_score, read_count, random_stable_positive & random_low_boundary50),
        "random_stable_low_boundary25_top_read_mass": _topk_mask(random_stable_score, read_count, random_stable_positive & random_low_boundary25),
    }

    rows = [
        _row_for_mask(
            name=name,
            mask=mask,
            read_active=read_active,
            read_score=read_score,
            stable_score=stable_score,
            stable_control=stable_control,
            disagreement_score=disagreement_score,
            disagreement_control=disagreement_control,
            random_stable_score=random_stable_score,
            random_disagreement_score=random_disagreement_score,
            dyn=dyn,
            key_avg=key_avg,
            qk_var=qk_var,
            occlusion=occlusion,
            uncertainty=uncertainty,
        )
        for name, mask in masks.items()
    ]
    rows_by_name = {str(row["mask"]): row for row in rows}
    current = rows_by_name["current_read_active_q90"]
    candidate = rows_by_name[str(args.candidate_mask)]
    random_candidate_name = str(args.random_candidate_mask)
    random_candidate = rows_by_name[random_candidate_name]
    vs_current = _improvement_summary(candidate, current)
    vs_random = _improvement_summary(candidate, random_candidate)

    required_fields = ["dyn_mean", "qk_var_mean", "disagreement_score_mean"]
    candidate_better_than_current = all(
        vs_current.get(f"{field}_relative_delta") is not None
        and float(vs_current[f"{field}_relative_delta"]) <= -float(args.min_relative_improvement)
        for field in required_fields
    )
    candidate_better_than_random = all(
        vs_random.get(f"{field}_relative_delta") is not None
        and float(vs_random[f"{field}_relative_delta"]) <= -float(args.min_relative_improvement)
        for field in required_fields
    )
    count_ok = int(candidate["count"]) >= int(args.min_candidate_tokens)
    repeat_ok = int(candidate["spatial_repeat_ge2"]) >= int(args.min_repeat_tokens)
    redirection_viable = bool(count_ok and repeat_ok and candidate_better_than_current and candidate_better_than_random)

    ledger_row = _first_primary_pair_row(args.ledger, f"{int(args.chunk) - 1}-{int(args.chunk)}")
    summary = {
        "schema": "acl2_v79_read_redirection_corridor_probe_v1",
        "chunk": int(args.chunk),
        "overlap_frames": int(min_frames),
        "read_dump": str(args.read_dump),
        "phase3_root": str(args.phase3_root),
        "stable_case": str(args.stable_case),
        "disagreement_case": str(args.disagreement_case),
        "random_stable_case": str(args.random_stable_case),
        "random_disagreement_case": str(args.random_disagreement_case),
        "stable_score_stats": _stats(stable_score),
        "stable_positive_score_stats": stable_positive_stats,
        "disagreement_score_stats": disagreement_stats,
        "random_stable_score_stats": _stats(random_stable_score),
        "random_stable_positive_score_stats": random_stable_positive_stats,
        "random_disagreement_score_stats": random_disagreement_stats,
        "read_active_count": read_count,
        "candidate_mask": str(args.candidate_mask),
        "random_candidate_mask": random_candidate_name,
        "candidate_count": int(candidate["count"]),
        "candidate_mass": float(candidate["mass"]),
        "random_candidate_count": int(random_candidate["count"]),
        "random_candidate_mass": float(random_candidate["mass"]),
        "candidate_vs_current": vs_current,
        "candidate_vs_random": vs_random,
        "candidate_better_than_current": bool(candidate_better_than_current),
        "candidate_better_than_random": bool(candidate_better_than_random),
        "count_ok": bool(count_ok),
        "repeat_ok": bool(repeat_ok),
        "redirection_viable_for_smoke": redirection_viable,
        "viability_rule": (
            "candidate_count>=min_candidate_tokens and spatial_repeat_ge2>=min_repeat_tokens "
            "and dyn/qk_var/disagreement relative deltas <= -min_relative_improvement "
            "against both current READ and random same-mass candidate"
        ),
        "ledger_rank1_pair_fields": {
            key: ledger_row.get(key)
            for key in (
                "chunk_pair",
                "future_after_overlap",
                "boundary_jump",
                "raw_overlap_residual",
                "stable_overlap_mass",
                "harm_overlap_mass",
                "context_overlap_mass",
                "road_edge_confidence_mean",
                "semantic_boundary_density_mean",
            )
            if key in ledger_row
        },
        "rows": rows,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-root", type=Path, default=DEFAULT_PHASE3_ROOT)
    parser.add_argument("--read-dump", type=Path, default=DEFAULT_READ_DUMP)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=9)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--overlap-frames", type=int, default=3)
    parser.add_argument("--stable-case", default="P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST")
    parser.add_argument("--disagreement-case", default="P9_10_SOURCE_GATE_DISAGREEMENT_K_LAST")
    parser.add_argument("--random-stable-case", default="P9_41_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_V_LAST")
    parser.add_argument("--random-disagreement-case", default="P9_11_SOURCE_GATE_DISAGREEMENT_RANDOM_SAME_MASS_K_LAST")
    parser.add_argument("--candidate-mask", default="swa_stable_low_boundary50_top_read_mass")
    parser.add_argument("--random-candidate-mask", default="random_stable_low_boundary50_top_read_mass")
    parser.add_argument("--min-candidate-tokens", type=int, default=128)
    parser.add_argument("--min-repeat-tokens", type=int, default=16)
    parser.add_argument("--min-relative-improvement", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_probe(args)
    summary_path = args.out_dir / "read_redirection_corridor_summary.json"
    csv_path = args.out_dir / "read_redirection_corridor_masks.csv"
    md_path = args.out_dir / "read_redirection_corridor_observations.md"
    summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, [dict(row) for row in summary["rows"]])
    lines = [
        "# READ redirection corridor observations",
        "",
        f"- chunk: {summary['chunk']}",
        f"- overlap_frames: {summary['overlap_frames']}",
        f"- candidate_mask: {summary['candidate_mask']}",
        f"- candidate_count: {summary['candidate_count']}",
        f"- random_candidate_count: {summary['random_candidate_count']}",
        f"- candidate_better_than_current: {summary['candidate_better_than_current']}",
        f"- candidate_better_than_random: {summary['candidate_better_than_random']}",
        f"- redirection_viable_for_smoke: {summary['redirection_viable_for_smoke']}",
        "",
        "This is a posthoc diagnostic only; it does not claim a method gate pass.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(_jsonable({k: summary[k] for k in (
        "candidate_mask",
        "candidate_count",
        "candidate_mass",
        "random_candidate_mask",
        "random_candidate_count",
        "random_candidate_mass",
        "candidate_vs_current",
        "candidate_vs_random",
        "candidate_better_than_current",
        "candidate_better_than_random",
        "count_ok",
        "repeat_ok",
        "redirection_viable_for_smoke",
    )}), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_summary={summary_path}")
    print(f"wrote_csv={csv_path}")
    print(f"wrote_observations={md_path}")


if __name__ == "__main__":
    main()
