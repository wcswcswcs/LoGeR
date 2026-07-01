#!/usr/bin/env python3
"""Posthoc READ-vs-SWA overlap carrier diagnostic for ACL2 v79."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


DEFAULT_PHASE3_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/"
    "report_final/phase3_semantic_swa_handoff/source_side_phase9_subset_chunk09"
)
DEFAULT_READ_DUMP = Path(
    "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/"
    "report_final/phase7_semantic_pca_qkv_ttt_rediscovery/"
    "read_ttt_hs12_key_stable_full_1609/chunk11/"
    "HS12_READ_ACTIVE_KEY_STABLE_TTT_POS_SEM/read_cue_patch_dumps/"
    "chunk_009_read_cue_patch.pt"
)
DEFAULT_LEDGER = Path(
    "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/"
    "report_final/phase1_current_bad_target_mining_with_semantic_diagnosis/"
    "adjacent_semantic_handoff_targets.csv"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _finite_values(values: torch.Tensor) -> torch.Tensor:
    x = values.detach().cpu().float().reshape(-1)
    return x[torch.isfinite(x)]


def _stats(values: torch.Tensor) -> Dict[str, Optional[float]]:
    x = _finite_values(values)
    if int(x.numel()) <= 0:
        return {"mean": None, "q10": None, "q25": None, "q50": None, "q75": None, "q80": None, "q90": None, "max": None}
    return {
        "mean": float(x.mean().item()),
        "q10": float(torch.quantile(x, 0.10).item()),
        "q25": float(torch.quantile(x, 0.25).item()),
        "q50": float(torch.quantile(x, 0.50).item()),
        "q75": float(torch.quantile(x, 0.75).item()),
        "q80": float(torch.quantile(x, 0.80).item()),
        "q90": float(torch.quantile(x, 0.90).item()),
        "max": float(x.max().item()),
    }


def _positive_stats(values: torch.Tensor) -> Dict[str, Optional[float]]:
    x = _finite_values(values)
    x = x[x > 0.0]
    out = _stats(x)
    out["positive_count"] = int(x.numel())
    return out


def _mask_stats(mask: torch.Tensor, denom: int) -> Dict[str, Any]:
    count = int(mask.detach().cpu().bool().sum().item())
    return {"count": count, "mass": float(count / max(int(denom), 1))}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    x = values.detach().cpu().float().reshape(-1)
    m = mask.detach().cpu().bool().reshape(-1)
    if int(x.numel()) != int(m.numel()) or not bool(m.any().item()):
        return None
    return float(x[m].mean().item())


def _load_pt(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected dict payload: {path}")
    return payload


def _feature_path(root: Path, *, case: str, kind: str, layer: int, chunk: int) -> Path:
    path = (
        root
        / f"chunk{int(chunk):02d}"
        / case
        / "swa_overlap_feature_maps"
        / f"chunk_{int(chunk):03d}_swa_overlap_{kind}_layer_{int(layer):02d}.pt"
    )
    if path.is_file():
        return path
    candidates = sorted((root / f"chunk{int(chunk):02d}" / case / "swa_overlap_feature_maps").glob(f"chunk_{int(chunk):03d}_swa_overlap_{kind}_layer_*.pt"))
    if not candidates:
        raise FileNotFoundError(path)
    return candidates[-1]


def _patch_overlap_from_feature(payload: Dict[str, Any], key: str) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise RuntimeError(f"SWA feature payload lacks tensor {key}")
    x = value.detach().cpu().float()
    if x.ndim == 3:
        x = x[0]
    if x.ndim != 2:
        raise RuntimeError(f"Unsupported {key} shape: {tuple(x.shape)}")
    tokens_per_frame = int(payload.get("tokens_per_frame", int(x.shape[-1])) or int(x.shape[-1]))
    patch_count = 19 * 66
    patch_start = max(0, int(tokens_per_frame) - patch_count)
    patch_end = patch_start + patch_count
    if int(x.shape[-1]) < patch_end:
        raise RuntimeError(f"Not enough tokens for 19x66 patches: shape={tuple(x.shape)}")
    return x[:, patch_start:patch_end].reshape(int(x.shape[0]), 19, 66)


def _read_overlap(read_payload: Dict[str, Any], key: str, overlap_frames: int) -> torch.Tensor:
    tensors = read_payload.get("tensors")
    if not isinstance(tensors, dict) or not torch.is_tensor(tensors.get(key)):
        raise RuntimeError(f"READ dump lacks tensor {key}")
    x = tensors[key].detach().cpu()
    if x.ndim != 3:
        raise RuntimeError(f"Unsupported READ tensor {key} shape: {tuple(x.shape)}")
    return x[: int(overlap_frames)].float()


def _first_primary_pair_row(path: Path, pair: str) -> Dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("target_scope") == "primary_kitti01_v53_h35" and row.get("chunk_pair") == pair:
                return dict(row)
    return {}


def _row_for_mask(
    *,
    name: str,
    mask: torch.Tensor,
    read_active: torch.Tensor,
    stable_score: torch.Tensor,
    disagreement_score: torch.Tensor,
    stable_control: torch.Tensor,
    disagreement_control: torch.Tensor,
    dyn: torch.Tensor,
    key_avg: torch.Tensor,
    qk_var: torch.Tensor,
) -> Dict[str, Any]:
    flat_mask = mask.detach().cpu().bool().reshape(-1)
    flat_read = read_active.detach().cpu().bool().reshape(-1)
    denom = int(flat_mask.numel())
    read_count = int(flat_read.sum().item())
    count = int(flat_mask.sum().item())
    read_inter = flat_mask & flat_read
    read_inter_count = int(read_inter.sum().item())
    return {
        "mask": name,
        "count": count,
        "mass": float(count / max(denom, 1)),
        "read_intersection_count": read_inter_count,
        "read_intersection_mass": float(read_inter_count / max(denom, 1)),
        "given_read": float(read_inter_count / max(read_count, 1)),
        "read_given_mask": float(read_inter_count / max(count, 1)),
        "stable_score_mean": _masked_mean(stable_score, flat_mask),
        "disagreement_score_mean": _masked_mean(disagreement_score, flat_mask),
        "stable_control_mean": _masked_mean(stable_control, flat_mask),
        "disagreement_control_mean": _masked_mean(disagreement_control, flat_mask),
        "dyn_mean": _masked_mean(dyn, flat_mask),
        "key_avg_mean": _masked_mean(key_avg, flat_mask),
        "qk_var_mean": _masked_mean(qk_var, flat_mask),
        "read_dyn_mean": _masked_mean(dyn, read_inter),
        "read_key_avg_mean": _masked_mean(key_avg, read_inter),
        "read_qk_var_mean": _masked_mean(qk_var, read_inter),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_probe(args: argparse.Namespace) -> Dict[str, Any]:
    read_payload = _load_pt(args.read_dump)
    stable_payload = _load_pt(_feature_path(args.phase3_root, case=args.stable_case, kind="source_replace", layer=args.layer, chunk=args.chunk))
    disagreement_payload = _load_pt(_feature_path(args.phase3_root, case=args.disagreement_case, kind="source_gate", layer=args.layer, chunk=args.chunk))
    random_stable_payload = _load_pt(_feature_path(args.phase3_root, case=args.random_stable_case, kind="source_replace", layer=args.layer, chunk=args.chunk))
    random_disagreement_payload = _load_pt(_feature_path(args.phase3_root, case=args.random_disagreement_case, kind="source_gate", layer=args.layer, chunk=args.chunk))

    overlap_frames = int(stable_payload.get("overlap_frames_effective", args.overlap_frames) or args.overlap_frames)
    read_active = _read_overlap(read_payload, "read_active_q90_patch", overlap_frames).bool()
    dyn = _read_overlap(read_payload, "dyn_patch", overlap_frames).float()
    key_avg = _read_overlap(read_payload, "key_avg_patch", overlap_frames).float()
    qk_var = _read_overlap(read_payload, "qk_var_patch", overlap_frames).float()

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
    dyn = dyn[:min_frames]
    key_avg = key_avg[:min_frames]
    qk_var = qk_var[:min_frames]
    stable_score = stable_score[:min_frames]
    stable_control = stable_control[:min_frames]
    disagreement_score = disagreement_score[:min_frames]
    disagreement_control = disagreement_control[:min_frames]
    random_stable_score = random_stable_score[:min_frames]
    random_disagreement_score = random_disagreement_score[:min_frames]

    stable_stats = _stats(stable_score)
    stable_positive_stats = _positive_stats(stable_score)
    disagreement_stats = _stats(disagreement_score)
    random_stable_stats = _stats(random_stable_score)
    random_stable_positive_stats = _positive_stats(random_stable_score)
    random_disagreement_stats = _stats(random_disagreement_score)

    stable_positive = stable_score > 0.0
    stable_positive_top50 = stable_score >= float(stable_positive_stats["q50"])
    stable_positive_top25 = stable_score >= float(stable_positive_stats["q75"])
    low_boundary50 = disagreement_score <= float(disagreement_stats["q50"])
    low_boundary25 = disagreement_score <= float(disagreement_stats["q25"])
    random_stable_positive = random_stable_score > 0.0
    random_stable_positive_top50 = random_stable_score >= float(random_stable_positive_stats["q50"])
    random_low_boundary50 = random_disagreement_score <= float(random_disagreement_stats["q50"])

    rows = [
        _row_for_mask(
            name="read_active_q90",
            mask=read_active,
            read_active=read_active,
            stable_score=stable_score,
            disagreement_score=disagreement_score,
            stable_control=stable_control,
            disagreement_control=disagreement_control,
            dyn=dyn,
            key_avg=key_avg,
            qk_var=qk_var,
        ),
        _row_for_mask(
            name="stable_agreement_positive",
            mask=stable_positive,
            read_active=read_active,
            stable_score=stable_score,
            disagreement_score=disagreement_score,
            stable_control=stable_control,
            disagreement_control=disagreement_control,
            dyn=dyn,
            key_avg=key_avg,
            qk_var=qk_var,
        ),
        _row_for_mask(
            name="stable_agreement_positive_top50",
            mask=stable_positive_top50,
            read_active=read_active,
            stable_score=stable_score,
            disagreement_score=disagreement_score,
            stable_control=stable_control,
            disagreement_control=disagreement_control,
            dyn=dyn,
            key_avg=key_avg,
            qk_var=qk_var,
        ),
        _row_for_mask(
            name="stable_agreement_positive_top25",
            mask=stable_positive_top25,
            read_active=read_active,
            stable_score=stable_score,
            disagreement_score=disagreement_score,
            stable_control=stable_control,
            disagreement_control=disagreement_control,
            dyn=dyn,
            key_avg=key_avg,
            qk_var=qk_var,
        ),
        _row_for_mask(
            name="low_boundary_disagreement_bottom50",
            mask=low_boundary50,
            read_active=read_active,
            stable_score=stable_score,
            disagreement_score=disagreement_score,
            stable_control=stable_control,
            disagreement_control=disagreement_control,
            dyn=dyn,
            key_avg=key_avg,
            qk_var=qk_var,
        ),
        _row_for_mask(
            name="low_boundary_disagreement_bottom25",
            mask=low_boundary25,
            read_active=read_active,
            stable_score=stable_score,
            disagreement_score=disagreement_score,
            stable_control=stable_control,
            disagreement_control=disagreement_control,
            dyn=dyn,
            key_avg=key_avg,
            qk_var=qk_var,
        ),
        _row_for_mask(
            name="candidate_read_x_stable_positive_x_low_boundary50",
            mask=read_active & stable_positive & low_boundary50,
            read_active=read_active,
            stable_score=stable_score,
            disagreement_score=disagreement_score,
            stable_control=stable_control,
            disagreement_control=disagreement_control,
            dyn=dyn,
            key_avg=key_avg,
            qk_var=qk_var,
        ),
        _row_for_mask(
            name="candidate_read_x_stable_positive_x_low_boundary25",
            mask=read_active & stable_positive & low_boundary25,
            read_active=read_active,
            stable_score=stable_score,
            disagreement_score=disagreement_score,
            stable_control=stable_control,
            disagreement_control=disagreement_control,
            dyn=dyn,
            key_avg=key_avg,
            qk_var=qk_var,
        ),
        _row_for_mask(
            name="candidate_read_x_stable_positive_top50_x_low_boundary50",
            mask=read_active & stable_positive_top50 & low_boundary50,
            read_active=read_active,
            stable_score=stable_score,
            disagreement_score=disagreement_score,
            stable_control=stable_control,
            disagreement_control=disagreement_control,
            dyn=dyn,
            key_avg=key_avg,
            qk_var=qk_var,
        ),
        _row_for_mask(
            name="candidate_read_x_stable_positive_top25_x_low_boundary50",
            mask=read_active & stable_positive_top25 & low_boundary50,
            read_active=read_active,
            stable_score=stable_score,
            disagreement_score=disagreement_score,
            stable_control=stable_control,
            disagreement_control=disagreement_control,
            dyn=dyn,
            key_avg=key_avg,
            qk_var=qk_var,
        ),
        _row_for_mask(
            name="random_control_read_x_stable_positive_x_low_boundary50",
            mask=read_active & random_stable_positive & random_low_boundary50,
            read_active=read_active,
            stable_score=stable_score,
            disagreement_score=disagreement_score,
            stable_control=stable_control,
            disagreement_control=disagreement_control,
            dyn=dyn,
            key_avg=key_avg,
            qk_var=qk_var,
        ),
        _row_for_mask(
            name="random_control_read_x_stable_positive_top50_x_low_boundary50",
            mask=read_active & random_stable_positive_top50 & random_low_boundary50,
            read_active=read_active,
            stable_score=stable_score,
            disagreement_score=disagreement_score,
            stable_control=stable_control,
            disagreement_control=disagreement_control,
            dyn=dyn,
            key_avg=key_avg,
            qk_var=qk_var,
        ),
    ]
    rows_by_name = {str(row["mask"]): row for row in rows}
    candidate = rows_by_name["candidate_read_x_stable_positive_x_low_boundary50"]
    random_candidate = rows_by_name["random_control_read_x_stable_positive_x_low_boundary50"]
    ledger_row = _first_primary_pair_row(args.ledger, f"{int(args.chunk) - 1}-{int(args.chunk)}")
    carrier_count = int(candidate["count"])
    carrier_mass = float(candidate["mass"])
    random_count = int(random_candidate["count"])
    viable = bool(
        carrier_count >= int(args.min_candidate_tokens)
        and carrier_mass >= float(args.min_candidate_mass)
        and carrier_count > random_count
    )
    summary = {
        "schema": "acl2_v79_read_swa_overlap_alignment_probe_v1",
        "chunk": int(args.chunk),
        "overlap_frames": int(min_frames),
        "read_dump": str(args.read_dump),
        "phase3_root": str(args.phase3_root),
        "stable_case": str(args.stable_case),
        "disagreement_case": str(args.disagreement_case),
        "random_stable_case": str(args.random_stable_case),
        "random_disagreement_case": str(args.random_disagreement_case),
        "stable_feature_path": str(_feature_path(args.phase3_root, case=args.stable_case, kind="source_replace", layer=args.layer, chunk=args.chunk)),
        "disagreement_feature_path": str(_feature_path(args.phase3_root, case=args.disagreement_case, kind="source_gate", layer=args.layer, chunk=args.chunk)),
        "stable_score_stats": stable_stats,
        "stable_positive_score_stats": stable_positive_stats,
        "disagreement_score_stats": disagreement_stats,
        "random_stable_score_stats": random_stable_stats,
        "random_stable_positive_score_stats": random_stable_positive_stats,
        "random_disagreement_score_stats": random_disagreement_stats,
        "read_active_mass": _mask_stats(read_active, int(read_active.numel())),
        "candidate_mask": "candidate_read_x_stable_positive_x_low_boundary50",
        "candidate_count": carrier_count,
        "candidate_mass": carrier_mass,
        "random_control_count": random_count,
        "random_control_mass": float(random_candidate["mass"]),
        "candidate_minus_random_count": int(carrier_count - random_count),
        "carrier_viable_for_smoke": viable,
        "viability_rule": (
            "candidate_count>=min_candidate_tokens and candidate_mass>=min_candidate_mass "
            "and candidate_count>random_control_count"
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
    parser.add_argument("--min-candidate-tokens", type=int, default=64)
    parser.add_argument("--min-candidate-mass", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_probe(args)
    summary_path = args.out_dir / "read_swa_overlap_alignment_summary.json"
    csv_path = args.out_dir / "read_swa_overlap_alignment_masks.csv"
    md_path = args.out_dir / "read_swa_overlap_alignment_observations.md"
    summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, [dict(row) for row in summary["rows"]])
    md_lines = [
        "# READ-SWA overlap alignment observations",
        "",
        f"- chunk: {summary['chunk']}",
        f"- overlap_frames: {summary['overlap_frames']}",
        f"- candidate_count: {summary['candidate_count']}",
        f"- candidate_mass: {summary['candidate_mass']}",
        f"- random_control_count: {summary['random_control_count']}",
        f"- carrier_viable_for_smoke: {summary['carrier_viable_for_smoke']}",
        f"- stable_score_q80: {summary['stable_score_stats']['q80']}",
        f"- disagreement_score_q50: {summary['disagreement_score_stats']['q50']}",
        "",
        "This is a posthoc diagnostic only; it does not claim a method gate pass.",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(_jsonable({k: summary[k] for k in (
        "candidate_count",
        "candidate_mass",
        "random_control_count",
        "random_control_mass",
        "carrier_viable_for_smoke",
        "stable_score_stats",
        "disagreement_score_stats",
    )}), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_summary={summary_path}")
    print(f"wrote_csv={csv_path}")
    print(f"wrote_observations={md_path}")


if __name__ == "__main__":
    main()
