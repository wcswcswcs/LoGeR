#!/usr/bin/env python3
"""Compare HS1 vs HS23/HS24 READ query masks and pose drift for v79.

This is a diagnostic-only tool.  It reads existing READ cue dumps and TUM
trajectories; it does not rerun LoGeR or claim a method gate pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_v78_bad_window_tables import (  # noqa: E402
    DEFAULT_GT_ROOT,
    _evaluate_run,
)


DEFAULT_SOURCEBOOST_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/"
    "report_final/phase7_semantic_pca_qkv_ttt_rediscovery/"
    "sourceboost_qkattn_midtailq_hs23_hs24_smoke_1917/chunk11"
)

DEFAULT_TRAJ_CASES = (
    "HS0_NATIVE_TTT_PROBE",
    "HS1_READ_ONLY_BEST_SEM",
    "HS8_GEOMETRY_ONLY_HANDSHAKE",
    "HS9_RANDOM_ROLE_HANDSHAKE",
    "HS23_SWA_STABLE_TOP25_QKATTN_SOURCEBOOST_MIDTAILQ_TTT_POS_SEM",
    "HS24_RANDOM_SWA_STABLE_TOP25_QKATTN_SOURCEBOOST_MIDTAILQ_TTT_POS_SEM",
)

DEFAULT_READ_CASES = (
    "HS1_READ_ONLY_BEST_SEM",
    "HS8_GEOMETRY_ONLY_HANDSHAKE",
    "HS9_RANDOM_ROLE_HANDSHAKE",
    "HS23_SWA_STABLE_TOP25_QKATTN_SOURCEBOOST_MIDTAILQ_TTT_POS_SEM",
    "HS24_RANDOM_SWA_STABLE_TOP25_QKATTN_SOURCEBOOST_MIDTAILQ_TTT_POS_SEM",
)

FOCUS_SINGLE_CHUNKS = {9, 10}
FOCUS_PAIRS = {"8-9", "9-10"}
FOCUS_WINDOWS = {"7-8-9-10-11"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _jsonable(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def _load_pt(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _tensor(payload: Dict[str, Any], key: str) -> torch.Tensor:
    tensors = payload.get("tensors")
    if isinstance(tensors, dict) and isinstance(tensors.get(key), torch.Tensor):
        return tensors[key].detach().cpu().float()
    value = payload.get(key)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    raise KeyError(f"{key} not found in {payload.get('schema', 'payload')}")


def _finite_mean(values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Optional[float]:
    x = values.detach().cpu().float()
    if mask is not None:
        m = mask.detach().cpu().bool()
        if tuple(m.shape) != tuple(x.shape):
            raise ValueError(f"mask shape {tuple(m.shape)} != values shape {tuple(x.shape)}")
        x = x[m]
    x = x[torch.isfinite(x)]
    if int(x.numel()) <= 0:
        return None
    return float(x.mean().item())


def _masked_fraction(mask: torch.Tensor, condition: torch.Tensor) -> Optional[float]:
    m = mask.detach().cpu().bool()
    c = condition.detach().cpu().bool()
    if tuple(m.shape) != tuple(c.shape):
        raise ValueError(f"mask shape {tuple(m.shape)} != condition shape {tuple(c.shape)}")
    denom = int(m.sum().item())
    if denom <= 0:
        return None
    return float((m & c).sum().item() / float(denom))


def _quantile_condition(values: torch.Tensor, q: float, *, high: bool) -> torch.Tensor:
    x = values.detach().cpu().float()
    finite = x[torch.isfinite(x)]
    if int(finite.numel()) <= 0:
        return torch.zeros_like(x, dtype=torch.bool)
    threshold = float(torch.quantile(finite, float(q)).item())
    return x >= threshold if high else x <= threshold


def _case_dir(root: Path, case: str) -> Path:
    return root / case


def _dump_paths(run_dir: Path) -> List[Path]:
    dump_dir = run_dir / "read_cue_patch_dumps"
    return sorted(dump_dir.glob("chunk_*_read_cue_patch.pt"))


def _chunk_from_dump(path: Path) -> int:
    return int(path.name.split("_")[1])


def _read_stats_for_dump(case: str, path: Path) -> Dict[str, Any]:
    payload = _load_pt(path)
    chunk = int(payload.get("chunk_idx", _chunk_from_dump(path)))
    read = _tensor(payload, "read_patch_final")
    read_q90 = _tensor(payload, "read_active_q90_patch") > 0.5
    dyn = _tensor(payload, "dyn_patch")
    key_avg = _tensor(payload, "key_avg_patch")
    qk_var = _tensor(payload, "qk_var_patch")
    conf = _tensor(payload, "confidence_patch")
    unc = _tensor(payload, "uncertainty_patch")
    occ = _tensor(payload, "occlusion_patch")
    high_dyn = _quantile_condition(dyn, 0.90, high=True)
    high_qk = _quantile_condition(qk_var, 0.90, high=True)
    low_conf = _quantile_condition(conf, 0.10, high=False)
    high_unc = _quantile_condition(unc, 0.90, high=True)
    high_occ = _quantile_condition(occ, 0.90, high=True)
    return {
        "case": case,
        "chunk": chunk,
        "dump": str(path),
        "num_frames": int(payload.get("num_frames", read.shape[0])),
        "patch_grid": json.dumps(payload.get("patch_grid", list(read.shape[-2:])), ensure_ascii=False),
        "read_q90_count": int(read_q90.sum().item()),
        "read_q90_mass": float(read_q90.float().mean().item()),
        "read_mean": _finite_mean(read),
        "read_q90_read_mean": _finite_mean(read, read_q90),
        "read_q90_dyn_mean": _finite_mean(dyn, read_q90),
        "read_q90_key_avg_mean": _finite_mean(key_avg, read_q90),
        "read_q90_qk_var_mean": _finite_mean(qk_var, read_q90),
        "read_q90_confidence_mean": _finite_mean(conf, read_q90),
        "read_q90_uncertainty_mean": _finite_mean(unc, read_q90),
        "read_q90_occlusion_mean": _finite_mean(occ, read_q90),
        "read_q90_high_dyn_frac": _masked_fraction(read_q90, high_dyn),
        "read_q90_high_qk_var_frac": _masked_fraction(read_q90, high_qk),
        "read_q90_low_confidence_frac": _masked_fraction(read_q90, low_conf),
        "read_q90_high_uncertainty_frac": _masked_fraction(read_q90, high_unc),
        "read_q90_high_occlusion_frac": _masked_fraction(read_q90, high_occ),
    }


def _read_mask_payload(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    payload = _load_pt(path)
    read = _tensor(payload, "read_patch_final")
    mask = _tensor(payload, "read_active_q90_patch") > 0.5
    return read, mask


def _compare_masks(
    *,
    reference_case: str,
    reference_path: Path,
    case: str,
    path: Path,
) -> Dict[str, Any]:
    ref_read, ref_mask = _read_mask_payload(reference_path)
    read, mask = _read_mask_payload(path)
    if tuple(ref_mask.shape) != tuple(mask.shape):
        raise ValueError(f"shape mismatch for {case}: {tuple(ref_mask.shape)} vs {tuple(mask.shape)}")
    inter = ref_mask & mask
    union = ref_mask | mask
    ref_count = int(ref_mask.sum().item())
    count = int(mask.sum().item())
    union_count = int(union.sum().item())
    inter_count = int(inter.sum().item())
    added = mask & ~ref_mask
    removed = ref_mask & ~mask
    read_diff = (read - ref_read).abs()
    return {
        "reference_case": reference_case,
        "case": case,
        "chunk": _chunk_from_dump(path),
        "reference_dump": str(reference_path),
        "dump": str(path),
        "reference_count": ref_count,
        "case_count": count,
        "intersection_count": inter_count,
        "union_count": union_count,
        "q90_iou": float(inter_count / max(union_count, 1)),
        "same_fraction_of_reference": float(inter_count / max(ref_count, 1)),
        "same_fraction_of_case": float(inter_count / max(count, 1)),
        "added_count": int(added.sum().item()),
        "removed_count": int(removed.sum().item()),
        "read_patch_mean_abs_diff": float(read_diff.mean().item()),
        "read_patch_max_abs_diff": float(read_diff.max().item()),
    }


def _run_bad_window_tables(
    *,
    root: Path,
    cases: Sequence[str],
    gt_root: Path,
    chunk_size: int,
    chunk_overlap: int,
    min_coverage: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    singles: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []
    windows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for case in cases:
        traj = _case_dir(root, case) / "01.txt"
        if not traj.is_file():
            continue
        s_rows, p_rows, w_rows, summary = _evaluate_run(
            name=case,
            seq="01",
            path=traj,
            gt_root=gt_root,
            chunk_size=int(chunk_size),
            overlap=int(chunk_overlap),
            min_coverage=float(min_coverage),
        )
        singles.extend(s_rows)
        pairs.extend(p_rows)
        windows.extend(w_rows)
        summaries.append(summary)
    return singles, pairs, windows, summaries


def _focus_metric_rows(
    singles: Sequence[Dict[str, Any]],
    pairs: Sequence[Dict[str, Any]],
    windows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in singles:
        if int(row.get("chunk_id", -1)) in FOCUS_SINGLE_CHUNKS:
            rows.append(
                {
                    "kind": "single",
                    "run": row.get("run"),
                    "target": f"chunk{row.get('chunk_id')}",
                    "frame_start": row.get("chunk_start_frame"),
                    "frame_end": row.get("chunk_end_frame"),
                    "primary_metric": "local_sim3_rmse_m",
                    "primary_value": row.get("local_sim3_rmse_m"),
                    "global_sim3_rmse_m": row.get("global_sim3_chunk_rmse_m"),
                    "scale": row.get("local_sim3_scale"),
                }
            )
    for row in pairs:
        if str(row.get("chunk_pair")) in FOCUS_PAIRS:
            rows.append(
                {
                    "kind": "pair",
                    "run": row.get("run"),
                    "target": row.get("chunk_pair"),
                    "frame_start": row.get("pair_start_frame"),
                    "frame_end": row.get("pair_end_frame"),
                    "boundary_frame": row.get("boundary_frame"),
                    "primary_metric": "tail3_to_future_from_boundary_sim3_rmse_m",
                    "primary_value": row.get("tail3_to_future_from_boundary_sim3_rmse_m"),
                    "tail3_to_head3_sim3_rmse_m": row.get("tail3_to_head3_sim3_rmse_m"),
                    "boundary_step_error_global_sim3_m": row.get("boundary_step_error_global_sim3_m"),
                    "pair_joint_sim3_rmse_m": row.get("pair_joint_sim3_rmse_m"),
                    "scale": row.get("pair_joint_sim3_scale"),
                }
            )
    for row in windows:
        if str(row.get("window_chunks")) in FOCUS_WINDOWS:
            rows.append(
                {
                    "kind": "window5",
                    "run": row.get("run"),
                    "target": row.get("window_chunks"),
                    "frame_start": row.get("window_start_frame"),
                    "frame_end": row.get("window_end_frame"),
                    "primary_metric": "window5_joint_sim3_rmse_m",
                    "primary_value": row.get("window5_joint_sim3_rmse_m"),
                    "window5_subchunk_scale_cv": row.get("window5_subchunk_scale_cv"),
                    "global_sim3_rmse_m": row.get("global_sim3_window5_rmse_m"),
                    "scale": row.get("window5_joint_sim3_scale"),
                }
            )
    return rows


def _best_by_target(rows: Sequence[Dict[str, Any]], target_case: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("kind")), str(row.get("target"))), []).append(dict(row))
    comparisons: List[Dict[str, Any]] = []
    for (kind, target), group in sorted(grouped.items()):
        valid = [row for row in group if row.get("primary_value") not in (None, "")]
        if not valid:
            continue
        valid.sort(key=lambda row: float(row["primary_value"]))
        best = valid[0]
        candidate = next((row for row in valid if row.get("run") == target_case), None)
        hs1 = next((row for row in valid if row.get("run") == "HS1_READ_ONLY_BEST_SEM"), None)
        if candidate is None:
            continue
        comparisons.append(
            {
                "kind": kind,
                "target": target,
                "candidate": target_case,
                "candidate_primary_value": candidate.get("primary_value"),
                "best_run": best.get("run"),
                "best_primary_value": best.get("primary_value"),
                "hs1_primary_value": hs1.get("primary_value") if hs1 else None,
                "candidate_minus_best": float(candidate["primary_value"]) - float(best["primary_value"]),
                "candidate_minus_hs1": (
                    float(candidate["primary_value"]) - float(hs1["primary_value"])
                    if hs1 and hs1.get("primary_value") is not None
                    else None
                ),
            }
        )
    return comparisons


def build_probe(args: argparse.Namespace) -> Dict[str, Any]:
    root = args.sourceboost_root
    read_stats: List[Dict[str, Any]] = []
    for case in args.read_cases:
        for path in _dump_paths(_case_dir(root, case)):
            read_stats.append(_read_stats_for_dump(case, path))

    reference_paths = {
        _chunk_from_dump(path): path
        for path in _dump_paths(_case_dir(root, args.reference_case))
    }
    mask_comparisons: List[Dict[str, Any]] = []
    for case in args.read_cases:
        if case == args.reference_case:
            continue
        for path in _dump_paths(_case_dir(root, case)):
            chunk = _chunk_from_dump(path)
            if chunk not in reference_paths:
                continue
            mask_comparisons.append(
                _compare_masks(
                    reference_case=args.reference_case,
                    reference_path=reference_paths[chunk],
                    case=case,
                    path=path,
                )
            )

    singles, pairs, windows, summaries = _run_bad_window_tables(
        root=root,
        cases=args.traj_cases,
        gt_root=args.gt_root,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_coverage=args.min_coverage,
    )
    focus_rows = _focus_metric_rows(singles, pairs, windows)
    hs23_comparisons = _best_by_target(focus_rows, args.candidate_case)

    hs23_vs_hs1 = [
        row
        for row in mask_comparisons
        if row.get("case") == args.candidate_case
    ]
    min_iou = min((float(row["q90_iou"]) for row in hs23_vs_hs1), default=None)
    max_patch_diff = max((float(row["read_patch_max_abs_diff"]) for row in hs23_vs_hs1), default=None)
    same_query_as_hs1 = bool(
        hs23_vs_hs1
        and min_iou is not None
        and float(min_iou) >= 0.999
        and max_patch_diff is not None
        and float(max_patch_diff) <= 1.0e-8
    )
    candidate_beats_hs1_any_focus = any(
        row.get("candidate_minus_hs1") is not None and float(row["candidate_minus_hs1"]) < 0.0
        for row in hs23_comparisons
    )
    interpretation = (
        "HS23 reuses the HS1 READ query mask; the failure is therefore downstream of "
        "the READ selection, in source/TTT injection or handoff propagation."
        if same_query_as_hs1
        else "HS23 READ query differs from HS1; query selection remains a live suspect."
    )
    if candidate_beats_hs1_any_focus:
        interpretation += " HS23 beats HS1 on at least one focus metric, but strict gate still requires all controls."
    else:
        interpretation += " HS23 does not beat HS1 on focus drift metrics."

    summary = {
        "schema": "acl2_v79_hs1_hs23_query_pose_contrast_v1",
        "diagnostic_only": True,
        "sourceboost_root": str(root),
        "reference_case": args.reference_case,
        "candidate_case": args.candidate_case,
        "read_cases": list(args.read_cases),
        "trajectory_cases": list(args.traj_cases),
        "same_query_as_hs1": same_query_as_hs1,
        "hs23_vs_hs1_min_q90_iou": min_iou,
        "hs23_vs_hs1_max_read_patch_abs_diff": max_patch_diff,
        "candidate_beats_hs1_any_focus": candidate_beats_hs1_any_focus,
        "interpretation": interpretation,
        "read_stats": read_stats,
        "mask_comparisons": mask_comparisons,
        "focus_metric_rows": focus_rows,
        "candidate_focus_comparisons": hs23_comparisons,
        "trajectory_summaries": summaries,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sourceboost-root", type=Path, default=DEFAULT_SOURCEBOOST_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-case", default="HS1_READ_ONLY_BEST_SEM")
    parser.add_argument("--candidate-case", default="HS23_SWA_STABLE_TOP25_QKATTN_SOURCEBOOST_MIDTAILQ_TTT_POS_SEM")
    parser.add_argument("--read-cases", nargs="+", default=list(DEFAULT_READ_CASES))
    parser.add_argument("--traj-cases", nargs="+", default=list(DEFAULT_TRAJ_CASES))
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_probe(args)
    summary_path = args.out_dir / "query_pose_contrast_summary.json"
    read_stats_path = args.out_dir / "read_query_stats.csv"
    mask_cmp_path = args.out_dir / "read_query_mask_comparison.csv"
    focus_path = args.out_dir / "focus_pose_metrics.csv"
    candidate_cmp_path = args.out_dir / "candidate_focus_comparison.csv"
    md_path = args.out_dir / "query_pose_contrast_observations.md"

    summary_path.write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(read_stats_path, summary["read_stats"])
    _write_csv(mask_cmp_path, summary["mask_comparisons"])
    _write_csv(focus_path, summary["focus_metric_rows"])
    _write_csv(candidate_cmp_path, summary["candidate_focus_comparisons"])

    lines = [
        "# HS1 vs HS23 query/pose contrast",
        "",
        f"- diagnostic_only: {summary['diagnostic_only']}",
        f"- same_query_as_hs1: {summary['same_query_as_hs1']}",
        f"- hs23_vs_hs1_min_q90_iou: {summary['hs23_vs_hs1_min_q90_iou']}",
        f"- hs23_vs_hs1_max_read_patch_abs_diff: {summary['hs23_vs_hs1_max_read_patch_abs_diff']}",
        f"- candidate_beats_hs1_any_focus: {summary['candidate_beats_hs1_any_focus']}",
        f"- interpretation: {summary['interpretation']}",
        "",
        "This file is diagnostic only and does not claim v79TF success.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            _jsonable(
                {
                    "same_query_as_hs1": summary["same_query_as_hs1"],
                    "hs23_vs_hs1_min_q90_iou": summary["hs23_vs_hs1_min_q90_iou"],
                    "hs23_vs_hs1_max_read_patch_abs_diff": summary["hs23_vs_hs1_max_read_patch_abs_diff"],
                    "candidate_beats_hs1_any_focus": summary["candidate_beats_hs1_any_focus"],
                    "interpretation": summary["interpretation"],
                    "out_dir": args.out_dir,
                }
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote_summary={summary_path}")
    print(f"wrote_read_stats={read_stats_path}")
    print(f"wrote_mask_comparison={mask_cmp_path}")
    print(f"wrote_focus_metrics={focus_path}")
    print(f"wrote_candidate_comparison={candidate_cmp_path}")
    print(f"wrote_observations={md_path}")


if __name__ == "__main__":
    main()
