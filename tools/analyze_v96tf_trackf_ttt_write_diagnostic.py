#!/usr/bin/env python3
"""Analyze ACL2 v96 Track F TTT write diagnostic dumps.

The input dumps are produced by ``--ttt_spatial_post_delta_map_dump_dir``.
This script is diagnostic-only: it computes spatial write-risk enrichment
from saved patch maps and never changes runtime memory behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import torch


ROOT = Path("results/acl2_v96tf_vggt4d_informed_semantic_gauge_preserving_memory_control")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def _to_float_tensor(value: Any) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    t = value.detach().cpu().float()
    if t.numel() == 0:
        return None
    return t


def _to_float_tensor_dict(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        t = _to_float_tensor(tensor)
        if t is not None:
            out[str(key)] = t
    return out


def _collapse_write_map(t: torch.Tensor) -> torch.Tensor:
    # Existing dump shape is usually [layer_branch, T, H, W].
    if t.ndim == 4:
        return t.mean(dim=0)
    if t.ndim == 3:
        return t
    raise ValueError(f"unsupported write map shape: {tuple(t.shape)}")


def _normalize_nonnegative(t: torch.Tensor) -> torch.Tensor:
    t = torch.nan_to_num(t.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    hi = torch.quantile(t.flatten(), 0.95) if t.numel() > 0 else torch.tensor(0.0)
    scale = float(hi.item())
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = float(t.max().item()) if t.numel() else 0.0
    if scale <= 1e-12:
        return torch.zeros_like(t)
    return (t / scale).clamp(0.0, 1.0)


def _same_mass_random_mean(write: torch.Tensor, count: int, *, samples: int = 64) -> float:
    flat = write.flatten()
    n = int(flat.numel())
    k = max(1, min(int(count), n))
    if n <= 0:
        return math.nan
    gen = torch.Generator(device="cpu")
    gen.manual_seed(9600 + k + n)
    vals = []
    for _ in range(int(samples)):
        idx = torch.randperm(n, generator=gen)[:k]
        vals.append(float(flat[idx].mean().item()))
    return float(sum(vals) / len(vals)) if vals else math.nan


def _mask_enrichment(write: torch.Tensor, mask: torch.Tensor) -> tuple[float, float, float, int]:
    mask = mask.detach().cpu().bool()
    flat_write = write.flatten()
    flat_mask = mask.flatten()
    count = int(flat_mask.sum().item())
    if count <= 0:
        return math.nan, math.nan, math.nan, 0
    actual = float(flat_write[flat_mask].mean().item())
    random = _same_mass_random_mean(write, count)
    return actual, random, actual - random, count


def _top_fraction_enrichment(write: torch.Tensor, score: torch.Tensor, *, frac: float = 0.10) -> tuple[float, float, float, int]:
    flat_score = score.flatten()
    n = int(flat_score.numel())
    if n <= 0:
        return math.nan, math.nan, math.nan, 0
    count = max(1, int(math.ceil(float(n) * float(frac))))
    top_idx = torch.topk(flat_score, k=count, largest=True).indices
    flat_write = write.flatten()
    actual = float(flat_write[top_idx].mean().item())
    random = _same_mass_random_mean(write, count)
    return actual, random, actual - random, count


def _best_component_enrichment(
    write_candidates: dict[str, torch.Tensor],
    risk_candidates: dict[str, torch.Tensor],
) -> dict[str, Any]:
    best = {
        "component_write_source": "missing",
        "component_risk_source": "missing",
        "component_actual": math.nan,
        "component_random": math.nan,
        "component_enrichment": math.nan,
        "component_token_count": 0,
        "component_pair_enrichments": [],
    }
    pair_rows: list[dict[str, Any]] = []
    for write_name, write_map in sorted(write_candidates.items()):
        write = _normalize_nonnegative(_collapse_write_map(write_map))
        for risk_name, risk_map in sorted(risk_candidates.items()):
            if tuple(write.shape) != tuple(risk_map.shape):
                continue
            actual, random, enrich, count = _top_fraction_enrichment(
                write,
                _normalize_nonnegative(risk_map),
            )
            if not math.isfinite(float(enrich)):
                continue
            pair_rows.append({
                "write_source": write_name,
                "risk_source": risk_name,
                "actual": actual,
                "random": random,
                "enrichment": enrich,
                "token_count": count,
            })
            if (
                not math.isfinite(float(best["component_enrichment"]))
                or float(enrich) > float(best["component_enrichment"])
            ):
                best = {
                    "component_write_source": write_name,
                    "component_risk_source": risk_name,
                    "component_actual": actual,
                    "component_random": random,
                    "component_enrichment": enrich,
                    "component_token_count": count,
                    "component_pair_enrichments": pair_rows,
                }
    best["component_pair_enrichments"] = pair_rows
    return best


def _safe_mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def _safe_median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(median(vals)) if vals else None


def _case_bucket(case_id: str, atlas: dict[str, dict[str, str]]) -> str:
    row = atlas.get(case_id, {})
    labels = row.get("action_response_labels", "")
    if "GOOD_PROTECTION" in labels:
        return "good_control"
    if "TTT_WRITE_RISK_DIAGNOSTIC" in row.get("v95_case_bucket", ""):
        return "ttt_write_risk"
    return "other"


def analyze_dump(path: Path, case_id: str, bucket: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    prior = _to_float_tensor(payload.get("ttt_write_prior_patch"))
    d_tok = _to_float_tensor(payload.get("D_tok_patch"))
    role = _to_float_tensor(payload.get("R_ttt_tok_patch"))
    conflict_exact = _to_float_tensor(payload.get("C_ttt_conflict_tok_patch"))
    scale_exact = _to_float_tensor(payload.get("S_scale_risk_tok_patch"))
    conflict_replay = _to_float_tensor(payload.get("C_ttt_conflict_replay_contribution_patch"))
    scale_replay = _to_float_tensor(payload.get("S_scale_risk_replay_contribution_patch"))
    replay_energy = _to_float_tensor(payload.get("U_ttt_write_replay_contribution_patch"))
    replay_energy_by_branch = _to_float_tensor_dict(
        payload.get("U_ttt_write_replay_contribution_by_branch_patch")
    )
    replay_conflict_by_branch = _to_float_tensor_dict(
        payload.get("C_ttt_conflict_replay_contribution_by_branch_patch")
    )
    replay_scale_by_branch = _to_float_tensor_dict(
        payload.get("S_scale_risk_replay_contribution_by_branch_patch")
    )
    conflict_proxy = _to_float_tensor(payload.get("C_ttt_conflict_proxy_patch"))
    scale_proxy = _to_float_tensor(payload.get("S_scale_risk_proxy_patch"))
    non_proxy_condition_available = any(
        x is not None for x in (conflict_exact, scale_exact, conflict_replay, scale_replay)
    )
    conflict = (
        conflict_exact
        if conflict_exact is not None
        else conflict_replay
        if conflict_replay is not None
        else None
        if non_proxy_condition_available
        else conflict_proxy
    )
    scale = (
        scale_exact
        if scale_exact is not None
        else scale_replay
        if scale_replay is not None
        else None
        if non_proxy_condition_available
        else scale_proxy
    )
    conflict_source = (
        "exact_token"
        if conflict_exact is not None
        else "replay_contribution"
        if conflict_replay is not None
        else "proxy"
        if conflict_proxy is not None and not non_proxy_condition_available
        else "missing"
    )
    scale_source = (
        "exact_token"
        if scale_exact is not None
        else "replay_contribution"
        if scale_replay is not None
        else "proxy"
        if scale_proxy is not None and not non_proxy_condition_available
        else "missing"
    )
    condition_map_source = (
        "exact_token"
        if conflict_source == "exact_token" or scale_source == "exact_token"
        else "replay_contribution"
        if conflict_source == "replay_contribution" or scale_source == "replay_contribution"
        else "proxy"
        if conflict_source == "proxy" or scale_source == "proxy"
        else "missing"
    )
    action = _to_float_tensor(payload.get("action_delta_norm_projection_patch"))
    committed = _to_float_tensor(payload.get("committed_post_delta_norm_projection_patch"))
    if prior is None or d_tok is None or action is None:
        raise ValueError(f"missing required Track F tensors in {path}")
    write = _normalize_nonnegative(_collapse_write_map(action))
    committed_write = _normalize_nonnegative(_collapse_write_map(committed)) if committed is not None else write
    risk_parts = [d_tok]
    if conflict is not None:
        risk_parts.append(conflict)
    if scale is not None:
        risk_parts.append(scale)
    risk = torch.stack([_normalize_nonnegative(x) for x in risk_parts], dim=0).amax(dim=0)
    semantic_scale_risk = torch.zeros_like(risk)
    if conflict is not None:
        semantic_scale_risk = torch.maximum(semantic_scale_risk, _normalize_nonnegative(conflict))
    if scale is not None:
        semantic_scale_risk = torch.maximum(semantic_scale_risk, _normalize_nonnegative(scale))
    write_candidates: dict[str, torch.Tensor] = {"action_projection": action}
    if replay_energy is not None:
        write_candidates["update_replay_all"] = replay_energy
    for branch_key, branch_energy in replay_energy_by_branch.items():
        write_candidates[f"update_replay_branch{branch_key}"] = branch_energy
    risk_candidates: dict[str, torch.Tensor] = {
        "combined_risk": risk,
        "d_tok": d_tok,
    }
    if conflict is not None:
        risk_candidates["conflict_all"] = conflict
    if scale is not None:
        risk_candidates["scale_all"] = scale
    if conflict is not None or scale is not None:
        risk_candidates["semantic_scale_all"] = semantic_scale_risk
    for branch_key, branch_conflict in replay_conflict_by_branch.items():
        risk_candidates[f"conflict_branch{branch_key}"] = branch_conflict
    for branch_key, branch_scale in replay_scale_by_branch.items():
        risk_candidates[f"scale_branch{branch_key}"] = branch_scale
    for branch_key in sorted(set(replay_conflict_by_branch) | set(replay_scale_by_branch)):
        parts = []
        if branch_key in replay_conflict_by_branch:
            parts.append(_normalize_nonnegative(replay_conflict_by_branch[branch_key]))
        if branch_key in replay_scale_by_branch:
            parts.append(_normalize_nonnegative(replay_scale_by_branch[branch_key]))
        if parts:
            risk_candidates[f"semantic_scale_branch{branch_key}"] = torch.stack(parts, dim=0).amax(dim=0)
    component_best = _best_component_enrichment(write_candidates, risk_candidates)

    flat_risk = risk.flatten()
    k = max(1, int(math.ceil(float(flat_risk.numel()) * 0.10)))
    top_idx = torch.topk(flat_risk, k=k, largest=True).indices
    low_idx = torch.topk(flat_risk, k=k, largest=False).indices
    flat_write = write.flatten()
    high_risk_write = float(flat_write[top_idx].mean().item())
    low_risk_write = float(flat_write[low_idx].mean().item())
    random_write = _same_mass_random_mean(write, k)
    enrichment = high_risk_write - random_write
    role_negative_actual, role_negative_random, role_negative_enrich, role_negative_count = (
        _mask_enrichment(write, role == 3) if role is not None else (math.nan, math.nan, math.nan, 0)
    )
    role_positive_actual, role_positive_random, role_positive_enrich, role_positive_count = (
        _mask_enrichment(write, role == 1) if role is not None else (math.nan, math.nan, math.nan, 0)
    )
    role_neutral_actual, role_neutral_random, role_neutral_enrich, role_neutral_count = (
        _mask_enrichment(write, role == 2) if role is not None else (math.nan, math.nan, math.nan, 0)
    )
    d_tok_actual, d_tok_random, d_tok_enrich, d_tok_top_count = _top_fraction_enrichment(
        write, _normalize_nonnegative(d_tok)
    )
    conflict_actual, conflict_random, conflict_enrich, conflict_top_count = (
        _top_fraction_enrichment(write, _normalize_nonnegative(conflict))
        if conflict is not None
        else (math.nan, math.nan, math.nan, 0)
    )
    scale_actual, scale_random, scale_enrich, scale_top_count = (
        _top_fraction_enrichment(write, _normalize_nonnegative(scale))
        if scale is not None
        else (math.nan, math.nan, math.nan, 0)
    )
    semantic_scale_actual, semantic_scale_random, semantic_scale_enrich, semantic_scale_top_count = (
        _top_fraction_enrichment(write, _normalize_nonnegative(semantic_scale_risk))
        if conflict is not None or scale is not None
        else (math.nan, math.nan, math.nan, 0)
    )
    candidate_enrichments = [
        value
        for value in (
            enrichment,
            role_negative_enrich,
            d_tok_enrich,
            conflict_enrich,
            scale_enrich,
            semantic_scale_enrich,
        )
        if math.isfinite(float(value))
    ]
    best_risk_enrichment = max(candidate_enrichments) if candidate_enrichments else math.nan

    return {
        "case_id": case_id,
        "bucket": bucket,
        "dump_path": str(path),
        "chunk_idx": int(payload.get("chunk_idx", -1)),
        "start_frame": int(payload.get("start_frame", -1)),
        "end_frame": int(payload.get("end_frame", -1)),
        "schema": str(payload.get("schema", "")),
        "condition_map_source": condition_map_source,
        "conflict_map_source": conflict_source,
        "scale_risk_map_source": scale_source,
        "condition_proxy_not_exact": condition_map_source == "proxy",
        "condition_replay_contribution_not_runtime_eligible": condition_map_source == "replay_contribution",
        "condition_map_provenance": json.dumps(payload.get("condition_map_provenance") or {}, sort_keys=True),
        "write_debug_available": bool((payload.get("write_debug_scalar_summary") or {}).get("probe_ttt_write_debug_available")),
        "layer_branch_rows": int(len(payload.get("layer_branch_rows") or [])),
        "persistent_write_mass": float((write * (1.0 - risk)).mean().item()),
        "transient_write_mass": float((write * _normalize_nonnegative(d_tok)).mean().item()),
        "no_write_mass": float((write * semantic_scale_risk).mean().item()),
        "write_risk_mean": float(risk.mean().item()),
        "write_risk_q90": float(torch.quantile(risk.flatten(), 0.90).item()),
        "write_mass_mean": float(write.mean().item()),
        "committed_write_mass_mean": float(committed_write.mean().item()),
        "high_risk_write_mass": high_risk_write,
        "low_risk_write_mass": low_risk_write,
        "same_mass_random_write_mass": random_write,
        "write_risk_enrichment": enrichment,
        "role_negative_short_token_count": role_negative_count,
        "role_negative_short_write_mass": role_negative_actual,
        "role_negative_short_random_write_mass": role_negative_random,
        "role_negative_short_enrichment": role_negative_enrich,
        "role_positive_long_token_count": role_positive_count,
        "role_positive_long_write_mass": role_positive_actual,
        "role_positive_long_random_write_mass": role_positive_random,
        "role_positive_long_enrichment": role_positive_enrich,
        "role_neutral_keep_token_count": role_neutral_count,
        "role_neutral_keep_write_mass": role_neutral_actual,
        "role_neutral_keep_random_write_mass": role_neutral_random,
        "role_neutral_keep_enrichment": role_neutral_enrich,
        "d_tok_top10_token_count": d_tok_top_count,
        "d_tok_top10_write_mass": d_tok_actual,
        "d_tok_top10_random_write_mass": d_tok_random,
        "d_tok_top10_enrichment": d_tok_enrich,
        "conflict_top10_token_count": conflict_top_count,
        "conflict_top10_write_mass": conflict_actual,
        "conflict_top10_random_write_mass": conflict_random,
        "conflict_top10_enrichment": conflict_enrich,
        "scale_risk_top10_token_count": scale_top_count,
        "scale_risk_top10_write_mass": scale_actual,
        "scale_risk_top10_random_write_mass": scale_random,
        "scale_risk_top10_enrichment": scale_enrich,
        "semantic_scale_top10_token_count": semantic_scale_top_count,
        "semantic_scale_top10_write_mass": semantic_scale_actual,
        "semantic_scale_top10_random_write_mass": semantic_scale_random,
        "semantic_scale_top10_enrichment": semantic_scale_enrich,
        "best_decomposed_risk_enrichment": best_risk_enrichment,
        "best_component_risk_enrichment": component_best["component_enrichment"],
        "best_component_write_source": component_best["component_write_source"],
        "best_component_risk_source": component_best["component_risk_source"],
        "best_component_actual_write_mass": component_best["component_actual"],
        "best_component_random_write_mass": component_best["component_random"],
        "best_component_token_count": component_best["component_token_count"],
        "component_write_candidate_count": len(write_candidates),
        "component_risk_candidate_count": len(risk_candidates),
        "component_pair_enrichments_json": json.dumps(
            component_best.get("component_pair_enrichments") or [],
            sort_keys=True,
        ),
        "prior_mean": float(prior.mean().item()),
        "prior_min": float(prior.min().item()),
        "prior_max": float(prior.max().item()),
        "d_tok_mean": float(d_tok.mean().item()),
        "d_tok_q90": float(torch.quantile(d_tok.flatten(), 0.90).item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atlas-rows", type=Path, default=ROOT / "trackA_case_response_atlas" / "rows.csv")
    parser.add_argument("--enrichment-threshold", type=float, default=0.05)
    parser.add_argument("--good-fp-threshold", type=float, default=0.25)
    args = parser.parse_args()

    atlas = {row["case_id"]: row for row in _read_rows(args.atlas_rows)}
    chunk_rows: list[dict[str, Any]] = []
    for path in sorted(args.input_root.glob("*/TTT_PROBE/ttt_spatial_post_delta_maps/*.pt")):
        case_id = path.parts[-4]
        chunk_rows.append(analyze_dump(path, case_id, _case_bucket(case_id, atlas)))

    if not chunk_rows:
        raise SystemExit(f"no TTT spatial dump .pt files found under {args.input_root}")

    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in chunk_rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    case_rows: list[dict[str, Any]] = []
    for case_id, rows in sorted(by_case.items()):
        bucket = str(rows[0]["bucket"])
        case_rows.append(
            {
                "case_id": case_id,
                "bucket": bucket,
                "chunk_count": len(rows),
                "condition_map_source": (
                    "exact_token"
                    if any(str(r.get("condition_map_source")) == "exact_token" for r in rows)
                    else "replay_contribution"
                    if any(str(r.get("condition_map_source")) == "replay_contribution" for r in rows)
                    else "proxy"
                    if any(str(r.get("condition_map_source")) == "proxy" for r in rows)
                    else "missing"
                ),
                "conflict_map_source": (
                    "exact_token"
                    if any(str(r.get("conflict_map_source")) == "exact_token" for r in rows)
                    else "replay_contribution"
                    if any(str(r.get("conflict_map_source")) == "replay_contribution" for r in rows)
                    else "proxy"
                    if any(str(r.get("conflict_map_source")) == "proxy" for r in rows)
                    else "missing"
                ),
                "scale_risk_map_source": (
                    "exact_token"
                    if any(str(r.get("scale_risk_map_source")) == "exact_token" for r in rows)
                    else "replay_contribution"
                    if any(str(r.get("scale_risk_map_source")) == "replay_contribution" for r in rows)
                    else "proxy"
                    if any(str(r.get("scale_risk_map_source")) == "proxy" for r in rows)
                    else "missing"
                ),
                "condition_proxy_not_exact": any(bool(r.get("condition_proxy_not_exact")) for r in rows),
                "condition_replay_contribution_not_runtime_eligible": any(
                    bool(r.get("condition_replay_contribution_not_runtime_eligible")) for r in rows
                ),
                "write_debug_available": all(bool(r["write_debug_available"]) for r in rows),
                "layer_branch_rows_min": min(int(r["layer_branch_rows"]) for r in rows),
                "persistent_write_mass": _safe_mean([r["persistent_write_mass"] for r in rows]),
                "transient_write_mass": _safe_mean([r["transient_write_mass"] for r in rows]),
                "no_write_mass": _safe_mean([r["no_write_mass"] for r in rows]),
                "write_risk_mean": _safe_mean([r["write_risk_mean"] for r in rows]),
                "write_risk_q90": _safe_mean([r["write_risk_q90"] for r in rows]),
                "write_mass_mean": _safe_mean([r["write_mass_mean"] for r in rows]),
                "high_risk_write_mass": _safe_mean([r["high_risk_write_mass"] for r in rows]),
                "same_mass_random_write_mass": _safe_mean([r["same_mass_random_write_mass"] for r in rows]),
                "write_risk_enrichment": _safe_mean([r["write_risk_enrichment"] for r in rows]),
                "role_negative_short_token_count": _safe_mean([r["role_negative_short_token_count"] for r in rows]),
                "role_negative_short_write_mass": _safe_mean([r["role_negative_short_write_mass"] for r in rows]),
                "role_negative_short_enrichment": _safe_mean([r["role_negative_short_enrichment"] for r in rows]),
                "role_positive_long_token_count": _safe_mean([r["role_positive_long_token_count"] for r in rows]),
                "role_positive_long_write_mass": _safe_mean([r["role_positive_long_write_mass"] for r in rows]),
                "role_positive_long_enrichment": _safe_mean([r["role_positive_long_enrichment"] for r in rows]),
                "role_neutral_keep_token_count": _safe_mean([r["role_neutral_keep_token_count"] for r in rows]),
                "role_neutral_keep_write_mass": _safe_mean([r["role_neutral_keep_write_mass"] for r in rows]),
                "role_neutral_keep_enrichment": _safe_mean([r["role_neutral_keep_enrichment"] for r in rows]),
                "d_tok_top10_enrichment": _safe_mean([r["d_tok_top10_enrichment"] for r in rows]),
                "conflict_top10_enrichment": _safe_mean([r["conflict_top10_enrichment"] for r in rows]),
                "scale_risk_top10_enrichment": _safe_mean([r["scale_risk_top10_enrichment"] for r in rows]),
                "semantic_scale_top10_enrichment": _safe_mean([r["semantic_scale_top10_enrichment"] for r in rows]),
                "best_decomposed_risk_enrichment": _safe_mean([r["best_decomposed_risk_enrichment"] for r in rows]),
                "best_component_risk_enrichment": _safe_mean([r["best_component_risk_enrichment"] for r in rows]),
                "gate_risk_enrichment": max(
                    x for x in (
                        _safe_mean([r["best_decomposed_risk_enrichment"] for r in rows]),
                        _safe_mean([r["best_component_risk_enrichment"] for r in rows]),
                    )
                    if x is not None
                ),
                "best_component_write_source": max(
                    (str(r.get("best_component_write_source", "")) for r in rows),
                    key=lambda src: sum(
                        1 for r in rows if str(r.get("best_component_write_source", "")) == src
                    ),
                ),
                "best_component_risk_source": max(
                    (str(r.get("best_component_risk_source", "")) for r in rows),
                    key=lambda src: sum(
                        1 for r in rows if str(r.get("best_component_risk_source", "")) == src
                    ),
                ),
                "component_write_candidate_count": _safe_mean([r["component_write_candidate_count"] for r in rows]),
                "component_risk_candidate_count": _safe_mean([r["component_risk_candidate_count"] for r in rows]),
                "prior_mean": _safe_mean([r["prior_mean"] for r in rows]),
                "prior_min": _safe_mean([r["prior_min"] for r in rows]),
                "prior_max": _safe_mean([r["prior_max"] for r in rows]),
                "d_tok_mean": _safe_mean([r["d_tok_mean"] for r in rows]),
                "d_tok_q90": _safe_mean([r["d_tok_q90"] for r in rows]),
            }
        )

    component_pair_by_case: dict[str, dict[tuple[str, str], list[float]]] = {}
    case_bucket_by_id = {str(row["case_id"]): str(row["bucket"]) for row in case_rows}
    for row in chunk_rows:
        case_id = str(row["case_id"])
        try:
            pair_rows = json.loads(str(row.get("component_pair_enrichments_json") or "[]"))
        except json.JSONDecodeError:
            pair_rows = []
        for pair_row in pair_rows:
            if not isinstance(pair_row, dict):
                continue
            write_source = str(pair_row.get("write_source", ""))
            risk_source = str(pair_row.get("risk_source", ""))
            if not write_source or not risk_source:
                continue
            try:
                enrich = float(pair_row.get("enrichment"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(enrich):
                continue
            component_pair_by_case.setdefault(case_id, {}).setdefault(
                (write_source, risk_source),
                [],
            ).append(enrich)

    component_pairs = sorted({
        pair
        for case_vals in component_pair_by_case.values()
        for pair in case_vals.keys()
    })
    component_pair_rows: list[dict[str, Any]] = []
    for write_source, risk_source in component_pairs:
        risk_vals: list[float] = []
        good_vals: list[float] = []
        for case_id, case_vals in component_pair_by_case.items():
            vals = case_vals.get((write_source, risk_source))
            if not vals:
                continue
            case_mean = float(sum(vals) / len(vals))
            bucket = case_bucket_by_id.get(case_id, "")
            if bucket == "ttt_write_risk":
                risk_vals.append(case_mean)
            elif bucket == "good_control":
                good_vals.append(case_mean)
        if len(risk_vals) < 3 or len(good_vals) < 3:
            continue
        risk_median = float(median(risk_vals))
        risk_mean = float(sum(risk_vals) / len(risk_vals))
        good_mean = float(sum(good_vals) / len(good_vals))
        good_fp_rate = float(sum(v >= float(args.enrichment_threshold) for v in good_vals) / len(good_vals))
        component_pair_rows.append({
            "write_source": write_source,
            "risk_source": risk_source,
            "risk_case_count": len(risk_vals),
            "good_control_count": len(good_vals),
            "risk_median_enrichment": risk_median,
            "risk_mean_enrichment": risk_mean,
            "good_mean_enrichment": good_mean,
            "good_false_positive_rate": good_fp_rate,
            "gate_pass": bool(
                risk_median >= float(args.enrichment_threshold)
                and good_fp_rate <= float(args.good_fp_threshold)
            ),
        })
    component_pair_rows.sort(
        key=lambda row: (
            not bool(row.get("gate_pass", False)),
            float(row.get("good_false_positive_rate", 1.0)),
            -float(row.get("risk_median_enrichment", -1.0)),
        )
    )

    fixed_pair_pass_rows = [row for row in component_pair_rows if bool(row.get("gate_pass", False))]
    fixed_component_pair_gate_pass = bool(fixed_pair_pass_rows)
    safety_pair_rows = [
        row for row in component_pair_rows
        if float(row.get("good_false_positive_rate", 1.0)) <= float(args.good_fp_threshold)
    ]
    if fixed_pair_pass_rows:
        best_fixed_pair = max(
            fixed_pair_pass_rows,
            key=lambda row: float(row.get("risk_median_enrichment", -1.0)),
        )
    elif safety_pair_rows:
        best_fixed_pair = max(
            safety_pair_rows,
            key=lambda row: float(row.get("risk_median_enrichment", -1.0)),
        )
    elif component_pair_rows:
        best_fixed_pair = max(
            component_pair_rows,
            key=lambda row: (
                -float(row.get("good_false_positive_rate", 1.0)),
                float(row.get("risk_median_enrichment", -1.0)),
            ),
        )
    else:
        best_fixed_pair = {}

    risk_rows = [r for r in case_rows if r["bucket"] == "ttt_write_risk"]
    good_rows = [r for r in case_rows if r["bucket"] == "good_control"]
    threshold = float(args.enrichment_threshold)
    risk_enrichment_vals = [float(r["write_risk_enrichment"]) for r in risk_rows if r["write_risk_enrichment"] is not None]
    decomposed_risk_vals = [
        float(r["best_decomposed_risk_enrichment"])
        for r in risk_rows
        if r.get("best_decomposed_risk_enrichment") is not None
    ]
    component_risk_vals = [
        float(r["best_component_risk_enrichment"])
        for r in risk_rows
        if r.get("best_component_risk_enrichment") is not None
    ]
    gate_risk_vals = [
        float(r["gate_risk_enrichment"])
        for r in risk_rows
        if r.get("gate_risk_enrichment") is not None
    ]
    d_tok_enrichment_vals = [
        float(r["d_tok_top10_enrichment"]) for r in risk_rows if r.get("d_tok_top10_enrichment") is not None
    ]
    conflict_enrichment_vals = [
        float(r["conflict_top10_enrichment"]) for r in risk_rows if r.get("conflict_top10_enrichment") is not None
    ]
    scale_enrichment_vals = [
        float(r["scale_risk_top10_enrichment"]) for r in risk_rows if r.get("scale_risk_top10_enrichment") is not None
    ]
    semantic_scale_enrichment_vals = [
        float(r["semantic_scale_top10_enrichment"])
        for r in risk_rows
        if r.get("semantic_scale_top10_enrichment") is not None
    ]
    role_negative_enrichment_vals = [
        float(r["role_negative_short_enrichment"])
        for r in risk_rows
        if r.get("role_negative_short_enrichment") is not None
    ]
    good_fp = [
        r for r in good_rows
        if r.get("gate_risk_enrichment") is not None
        and float(r["gate_risk_enrichment"]) >= threshold
    ]
    visual_support_exists = all(bool(r["write_debug_available"]) and int(r["layer_branch_rows_min"]) > 0 for r in case_rows)
    exact_condition_chunk_count = sum(1 for row in chunk_rows if row.get("condition_map_source") == "exact_token")
    replay_condition_chunk_count = sum(1 for row in chunk_rows if row.get("condition_map_source") == "replay_contribution")
    proxy_condition_chunk_count = sum(1 for row in chunk_rows if row.get("condition_map_source") == "proxy")
    missing_condition_chunk_count = sum(1 for row in chunk_rows if row.get("condition_map_source") == "missing")
    condition_map_source = (
        "exact_token"
        if exact_condition_chunk_count > 0
        else "replay_contribution"
        if replay_condition_chunk_count > 0
        else "proxy"
        if proxy_condition_chunk_count > 0
        else "missing"
    )
    risk_case_count_pass = len(risk_rows) >= 3
    risk_enrichment_pass = (_safe_median(gate_risk_vals) or -1.0) >= threshold
    good_false_positive_rate = float(len(good_fp) / max(1, len(good_rows))) if good_rows else 1.0
    good_fp_pass = good_false_positive_rate <= float(args.good_fp_threshold)
    gate_pass = bool(
        risk_case_count_pass
        and risk_enrichment_pass
        and good_fp_pass
        and fixed_component_pair_gate_pass
        and visual_support_exists
    )
    classification_prefix = "TTT_WRITE_DIAGNOSTIC"
    if condition_map_source == "replay_contribution":
        classification_prefix = "TTT_WRITE_REPLAY_CONTRIBUTION_CONDITION_DIAGNOSTIC"
    elif condition_map_source == "proxy":
        classification_prefix = "TTT_WRITE_PROXY_CONDITION_DIAGNOSTIC"
    elif condition_map_source == "missing":
        classification_prefix = "TTT_WRITE_NO_CONDITION_MAP_DIAGNOSTIC"

    summary = {
        "schema": "acl2_v96_trackf_ttt_write_diagnostic_summary_v1",
        "input_root": str(args.input_root),
        "chunk_row_count": len(chunk_rows),
        "case_count": len(case_rows),
        "ttt_write_risk_case_count": len(risk_rows),
        "good_control_count": len(good_rows),
        "condition_map_source": condition_map_source,
        "exact_condition_chunk_count": exact_condition_chunk_count,
        "replay_condition_chunk_count": replay_condition_chunk_count,
        "proxy_condition_chunk_count": proxy_condition_chunk_count,
        "missing_condition_chunk_count": missing_condition_chunk_count,
        "proxy_condition_not_runtime_eligible": condition_map_source == "proxy",
        "replay_condition_not_runtime_eligible": condition_map_source == "replay_contribution",
        "visual_support_exists": visual_support_exists,
        "median_risk_write_enrichment": _safe_median(risk_enrichment_vals),
        "mean_risk_write_enrichment": _safe_mean(risk_enrichment_vals),
        "median_decomposed_risk_enrichment": _safe_median(decomposed_risk_vals),
        "mean_decomposed_risk_enrichment": _safe_mean(decomposed_risk_vals),
        "median_component_risk_enrichment": _safe_median(component_risk_vals),
        "mean_component_risk_enrichment": _safe_mean(component_risk_vals),
        "median_gate_risk_enrichment": _safe_median(gate_risk_vals),
        "mean_gate_risk_enrichment": _safe_mean(gate_risk_vals),
        "component_pair_count": len(component_pair_rows),
        "fixed_component_pair_gate_pass": fixed_component_pair_gate_pass,
        "fixed_component_pair_pass_count": len(fixed_pair_pass_rows),
        "best_fixed_component_write_source": best_fixed_pair.get("write_source"),
        "best_fixed_component_risk_source": best_fixed_pair.get("risk_source"),
        "best_fixed_component_risk_median_enrichment": best_fixed_pair.get("risk_median_enrichment"),
        "best_fixed_component_risk_mean_enrichment": best_fixed_pair.get("risk_mean_enrichment"),
        "best_fixed_component_good_mean_enrichment": best_fixed_pair.get("good_mean_enrichment"),
        "best_fixed_component_good_false_positive_rate": best_fixed_pair.get("good_false_positive_rate"),
        "median_d_tok_top10_enrichment": _safe_median(d_tok_enrichment_vals),
        "median_conflict_top10_enrichment": _safe_median(conflict_enrichment_vals),
        "median_scale_risk_top10_enrichment": _safe_median(scale_enrichment_vals),
        "median_semantic_scale_top10_enrichment": _safe_median(semantic_scale_enrichment_vals),
        "median_role_negative_short_enrichment": _safe_median(role_negative_enrichment_vals),
        "good_false_positive_rate": good_false_positive_rate,
        "gate_rule": "risk cases >= 3; oracle median write-risk enrichment >= 0.05 over same-mass random; oracle good false positive <= 0.25; at least one fixed component write/risk pair must also satisfy risk median >= 0.05 and good false positive <= 0.25; visual support exists",
        "risk_case_count_pass": risk_case_count_pass,
        "risk_enrichment_pass": risk_enrichment_pass,
        "good_false_positive_pass": good_fp_pass,
        "gate_pass": gate_pass,
        "runtime_ttt_action_allowed": False,
        "method_success": False,
        "full_method_success": False,
        "classification": (
            f"{classification_prefix}_GATE_PASS_NO_RUNTIME_ACTION"
            if gate_pass
            else f"{classification_prefix}_GATE_FAIL_NO_RUNTIME_ACTION"
        ),
    }

    fields = list(case_rows[0].keys())
    chunk_fields = list(chunk_rows[0].keys())
    _write_rows(args.output_dir / "case_rows.csv", case_rows, fields)
    _write_rows(args.output_dir / "chunk_rows.csv", chunk_rows, chunk_fields)
    if component_pair_rows:
        _write_rows(
            args.output_dir / "component_pair_rows.csv",
            component_pair_rows,
            list(component_pair_rows[0].keys()),
        )
    _write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "failure_report.md").write_text(
        "# Track F TTT Write Diagnostic\n\n"
        f"classification = {summary['classification']}\n\n"
        f"gate_pass = {summary['gate_pass']}\n\n"
        f"ttt_write_risk_case_count = {summary['ttt_write_risk_case_count']}\n\n"
        f"median_risk_write_enrichment = {summary['median_risk_write_enrichment']}\n\n"
        f"median_decomposed_risk_enrichment = {summary['median_decomposed_risk_enrichment']}\n\n"
        f"median_component_risk_enrichment = {summary['median_component_risk_enrichment']}\n\n"
        f"median_gate_risk_enrichment = {summary['median_gate_risk_enrichment']}\n\n"
        f"fixed_component_pair_gate_pass = {summary['fixed_component_pair_gate_pass']}\n\n"
        f"fixed_component_pair_pass_count = {summary['fixed_component_pair_pass_count']}\n\n"
        f"best_fixed_component_write_source = {summary['best_fixed_component_write_source']}\n\n"
        f"best_fixed_component_risk_source = {summary['best_fixed_component_risk_source']}\n\n"
        f"best_fixed_component_risk_median_enrichment = {summary['best_fixed_component_risk_median_enrichment']}\n\n"
        f"best_fixed_component_good_false_positive_rate = {summary['best_fixed_component_good_false_positive_rate']}\n\n"
        f"median_d_tok_top10_enrichment = {summary['median_d_tok_top10_enrichment']}\n\n"
        f"median_conflict_top10_enrichment = {summary['median_conflict_top10_enrichment']}\n\n"
        f"median_scale_risk_top10_enrichment = {summary['median_scale_risk_top10_enrichment']}\n\n"
        f"median_semantic_scale_top10_enrichment = {summary['median_semantic_scale_top10_enrichment']}\n\n"
        f"median_role_negative_short_enrichment = {summary['median_role_negative_short_enrichment']}\n\n"
        f"good_false_positive_rate = {summary['good_false_positive_rate']}\n\n"
        "runtime_ttt_action_allowed = false\n",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
