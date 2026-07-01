#!/usr/bin/env python3
"""Audit Phase2 multiseq TTT selected regions against absolute geometry error.

This diagnostic intentionally does not claim the v80 method gate.  It uses
existing Phase2 native trajectories and TTT post-delta maps to rank held-out
seq02/seq05 chunks for the next targeted selected-write support probes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_v80_ttt_geometry_error_visual_bridge import _load_aligned_run  # noqa: E402
from tools.visualize_v78_phase4_ttt_output_separated import (  # noqa: E402
    _delta_map,
    _load_semantic,
    _mask_from_ids,
    _semantic_patch,
    _torch_load,
)


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final"
)
DEFAULT_CASE_BANK = REPORT_ROOT / "phase1_three_memory_case_bank/long_five_chunk_cases.csv"
DEFAULT_OUT_DIR = REPORT_ROOT / "phase10_phase2_multiseq_ttt_abs_error_overlap_20260622_2223"
DEFAULT_DATA_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset")
PHASE2_ROOTS = {
    "02": REPORT_ROOT / "phase2_direct_hook_repair/ttt_seq02_full_case_chunks025_029_041_046_062_070",
    "05": REPORT_ROOT / "phase2_direct_hook_repair/ttt_seq05_full_case_chunks005_009_020_024_075_083",
}
RISK_LABELS = {
    "void",
    "sky",
    "tree",
    "grass",
    "other_plant",
    "flower",
    "car",
    "truck",
    "bicycle",
    "motorcycle",
    "person",
}


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return _clean(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
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
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _csv_list(text: str) -> list[str]:
    return [part.strip().zfill(2) for part in str(text or "").split(",") if part.strip()]


def _find_masklet(seq: str, chunk: int, stage_root: Path) -> Path:
    matches = sorted((stage_root / seq / "stage_c_cache_semantic_chunks").glob(f"chunk_{int(chunk):03d}_*/masklet.pt"))
    if not matches:
        raise FileNotFoundError(f"missing stage-C masklet: seq={seq} chunk={chunk}")
    return matches[0]


def _risk_ids(label_names: Iterable[str]) -> set[int]:
    wanted = {name.lower().replace(" ", "_") for name in RISK_LABELS}
    out: set[int] = set()
    for idx, raw in enumerate(label_names):
        norm = str(raw).lower().replace(" ", "_")
        if norm in wanted:
            out.add(int(idx))
    return out


def _case_memberships(case_rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in case_rows:
        seq = str(row.get("seq") or "").zfill(2)
        try:
            start = int(float(row.get("chunk_start", "")))
            end = int(float(row.get("chunk_end", "")))
        except ValueError:
            continue
        for chunk in range(start, end + 1):
            out.setdefault((seq, int(chunk)), []).append(row)
    return out


def _trajectory_errors(seq: str, chunk_dir: Path, data_root: Path) -> dict[str, Any]:
    native = chunk_dir / "T0_NATIVE_PCA" / f"{seq}.txt"
    write_probe = chunk_dir / "T0_TTT_WRITE_PROBE_DELTA" / f"{seq}.txt"
    gt = data_root / "poses" / f"{seq}.txt"
    base = _load_aligned_run(native, gt)
    cand = _load_aligned_run(write_probe, gt) if write_probe.is_file() else None
    base_by_frame = {int(frame): float(base["err_m"][idx]) for idx, frame in enumerate(base["frames"])}
    delta_by_frame: dict[int, float] = {}
    if cand is not None:
        cand_index = {int(frame): idx for idx, frame in enumerate(cand["frames"])}
        for idx, frame in enumerate(base["frames"]):
            frame_int = int(frame)
            if frame_int in cand_index:
                delta_by_frame[frame_int] = float(cand["err_m"][cand_index[frame_int]] - base["err_m"][idx])
    return {
        "native_trajectory": str(native),
        "write_probe_trajectory": str(write_probe),
        "base_by_frame": base_by_frame,
        "delta_by_frame": delta_by_frame,
    }


def _analyze_chunk(
    *,
    seq: str,
    chunk: int,
    chunk_dir: Path,
    case_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    post_delta_path = (
        chunk_dir
        / "T0_TTT_WRITE_PROBE_DELTA"
        / "ttt_spatial_post_delta_maps"
        / f"chunk_{int(chunk):03d}_ttt_spatial_post_delta_map.pt"
    )
    if not post_delta_path.is_file():
        raise FileNotFoundError(f"missing post-delta: {post_delta_path}")
    post_delta = _torch_load(post_delta_path)
    start_frame = int(post_delta.get("start_frame", int(chunk) * (int(args.chunk_size) - int(args.chunk_overlap))))
    num_frames = int(post_delta.get("num_frames", int(args.chunk_size)))
    overlap = min(int(args.overlap), num_frames)
    masklet = _find_masklet(seq, chunk, Path(args.stage_c_root))
    semantic = _load_semantic(masklet)
    risk_ids = _risk_ids(semantic["label_names"])
    traj = _trajectory_errors(seq, chunk_dir, Path(args.data_root))
    runtime_grid = (int(args.runtime_grid[0]), int(args.runtime_grid[1]))

    frame_rows: list[dict[str, Any]] = []
    selected_runtime_total = 0
    selected_risk_total = 0
    risk_total = 0
    baseline_errors: list[float] = []
    candidate_deltas: list[float] = []
    weighted_scores: list[float] = []
    for local in range(overlap):
        global_frame = int(start_frame + local)
        d_geo = _delta_map(post_delta, "D_tok_patch", local)
        if not torch.is_tensor(d_geo):
            continue
        d_geo = d_geo.detach().cpu().float()
        visual_h, visual_w = int(d_geo.shape[0]), int(d_geo.shape[1])
        labels, _sem_img, conf_patch = _semantic_patch(semantic, local, (visual_w, visual_h))
        dyn_mask = _mask_from_ids(labels, semantic["dynamic_ids"])
        risk_mask = _mask_from_ids(labels, risk_ids)
        high_d = d_geo > torch.quantile(d_geo.reshape(-1), float(args.d_tok_quantile))
        low_conf = conf_patch < float(args.low_conf_threshold)
        selected_visual = dyn_mask | high_d | low_conf
        risk_visual = risk_mask | low_conf
        selected_runtime = F.interpolate(
            selected_visual.float()[None, None],
            size=runtime_grid,
            mode="nearest",
        ).squeeze(0).squeeze(0).bool()
        risk_runtime = F.interpolate(
            risk_visual.float()[None, None],
            size=runtime_grid,
            mode="nearest",
        ).squeeze(0).squeeze(0).bool()
        selected_risk = selected_runtime & risk_runtime
        baseline_error = traj["base_by_frame"].get(global_frame)
        if baseline_error is not None:
            baseline_errors.append(float(baseline_error))
        delta = traj["delta_by_frame"].get(global_frame)
        if delta is not None:
            candidate_deltas.append(float(delta))
        selected_mass = int(selected_runtime.sum().item())
        risk_mass = int(risk_runtime.sum().item())
        selected_risk_mass = int(selected_risk.sum().item())
        selected_risk_ratio = float(selected_risk_mass / max(1, selected_mass))
        error_badness = float(min(1.0, max(0.0, float(baseline_error or 0.0) / max(float(args.error_scale_m), 1e-6))))
        weighted = selected_risk_ratio * error_badness
        weighted_scores.append(weighted)
        selected_runtime_total += selected_mass
        selected_risk_total += selected_risk_mass
        risk_total += risk_mass
        frame_rows.append(
            {
                "seq": seq,
                "chunk": int(chunk),
                "local_frame": int(local),
                "global_frame": global_frame,
                "baseline_abs_error_m": baseline_error,
                "write_probe_delta_vs_native_m": delta,
                "error_badness_scale_m": float(args.error_scale_m),
                "error_badness": error_badness,
                "selected_runtime_mass": selected_mass,
                "risk_runtime_mass": risk_mass,
                "selected_risk_mass": selected_risk_mass,
                "selected_risk_given_selected": selected_risk_ratio,
                "risk_given_runtime": float(risk_mass / max(1, int(runtime_grid[0]) * int(runtime_grid[1]))),
                "weighted_selected_risk_error": weighted,
            }
        )

    case_types = sorted({str(row.get("case_type")) for row in case_rows})
    j_values = [
        float(row["J_long"])
        for row in case_rows
        if str(row.get("J_long", "")).strip() and str(row.get("J_long", "")).strip().lower() != "nan"
    ]
    return {
        "seq": seq,
        "chunk": int(chunk),
        "case_types": case_types,
        "case_ranks": [
            {"case_type": row.get("case_type"), "case_rank": row.get("case_rank"), "window": f"{row.get('chunk_start')}-{row.get('chunk_end')}"}
            for row in case_rows
        ],
        "max_J_long_membership": max(j_values) if j_values else None,
        "start_frame": int(start_frame),
        "overlap": int(overlap),
        "runtime_grid": [int(runtime_grid[0]), int(runtime_grid[1])],
        "post_delta_path": str(post_delta_path),
        "stage_c_masklet": str(masklet),
        "native_trajectory": traj["native_trajectory"],
        "write_probe_trajectory": traj["write_probe_trajectory"],
        "baseline_abs_error_mean_m": sum(baseline_errors) / len(baseline_errors) if baseline_errors else None,
        "baseline_abs_error_max_m": max(baseline_errors) if baseline_errors else None,
        "write_probe_delta_mean_m": sum(candidate_deltas) / len(candidate_deltas) if candidate_deltas else None,
        "write_probe_delta_max_m": max(candidate_deltas) if candidate_deltas else None,
        "selected_runtime_mass": int(selected_runtime_total),
        "risk_runtime_mass": int(risk_total),
        "selected_risk_mass": int(selected_risk_total),
        "selected_risk_given_selected": float(selected_risk_total / max(1, selected_runtime_total)),
        "weighted_selected_risk_error_mean": sum(weighted_scores) / len(weighted_scores) if weighted_scores else None,
        "frame_rows": frame_rows,
        "diagnostic_positive_flag": bool(
            case_rows
            and "bad" in case_types
            and selected_runtime_total > 0
            and float(selected_risk_total / max(1, selected_runtime_total)) >= float(args.min_selected_risk_ratio)
            and (sum(baseline_errors) / len(baseline_errors) if baseline_errors else 0.0) >= float(args.min_abs_error_m)
        ),
        "diagnostic_good_safety_flag": bool(
            case_rows
            and "good" in case_types
            and (
                selected_runtime_total == 0
                or float(selected_risk_total / max(1, selected_runtime_total)) <= float(args.good_max_selected_risk_ratio)
                or (sum(baseline_errors) / len(baseline_errors) if baseline_errors else 0.0) < float(args.min_abs_error_m)
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-bank", type=Path, default=DEFAULT_CASE_BANK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seqs", default="02,05")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--stage-c-root", type=Path, default=Path("results/kitti_preprocess"))
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--runtime-grid", nargs=2, type=int, default=(19, 66))
    parser.add_argument("--d-tok-quantile", type=float, default=0.75)
    parser.add_argument("--low-conf-threshold", type=float, default=0.55)
    parser.add_argument("--error-scale-m", type=float, default=1.0)
    parser.add_argument("--min-selected-risk-ratio", type=float, default=0.50)
    parser.add_argument("--good-max-selected-risk-ratio", type=float, default=0.35)
    parser.add_argument("--min-abs-error-m", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_rows = _read_csv(Path(args.case_bank))
    memberships = _case_memberships(case_rows)
    seqs = set(_csv_list(args.seqs))
    chunk_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for seq in sorted(seqs):
        root = PHASE2_ROOTS.get(seq)
        if root is None or not root.exists():
            errors.append({"seq": seq, "error": f"missing_phase2_root:{root}"})
            continue
        for chunk_dir in sorted(root.glob("chunk*")):
            try:
                chunk = int(chunk_dir.name.replace("chunk", ""))
            except ValueError:
                continue
            case_members = memberships.get((seq, chunk), [])
            if not case_members:
                continue
            try:
                row = _analyze_chunk(seq=seq, chunk=chunk, chunk_dir=chunk_dir, case_rows=case_members, args=args)
                frame_rows.extend(row.pop("frame_rows"))
                chunk_rows.append(row)
            except Exception as exc:  # noqa: BLE001
                errors.append({"seq": seq, "chunk": chunk, "error": repr(exc)})

    ranked_bad = sorted(
        [row for row in chunk_rows if "bad" in row.get("case_types", [])],
        key=lambda row: (
            bool(row.get("diagnostic_positive_flag")),
            float(row.get("weighted_selected_risk_error_mean") or -1.0),
            float(row.get("baseline_abs_error_mean_m") or -1.0),
        ),
        reverse=True,
    )
    ranked_good = sorted(
        [row for row in chunk_rows if "good" in row.get("case_types", [])],
        key=lambda row: (
            bool(row.get("diagnostic_good_safety_flag")),
            -(float(row.get("selected_risk_given_selected") or 0.0)),
            -(float(row.get("baseline_abs_error_mean_m") or 0.0)),
        ),
        reverse=True,
    )
    summary = {
        "schema": "acl2_v80_phase2_multiseq_ttt_abs_error_overlap_audit_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "note": (
            "This audit uses native absolute trajectory error and TTT post-delta/semantic risk overlap. "
            "It is not the selected-write low-support held-out gate and must not be mixed into method success."
        ),
        "seqs": sorted(seqs),
        "chunk_rows": len(chunk_rows),
        "frame_rows": len(frame_rows),
        "errors": errors,
        "diagnostic_positive_bad_chunks": [
            {"seq": row["seq"], "chunk": row["chunk"]} for row in chunk_rows if row.get("diagnostic_positive_flag")
        ],
        "diagnostic_good_safety_chunks": [
            {"seq": row["seq"], "chunk": row["chunk"]} for row in chunk_rows if row.get("diagnostic_good_safety_flag")
        ],
        "top_bad_candidates": [
            {
                "seq": row["seq"],
                "chunk": row["chunk"],
                "case_types": row["case_types"],
                "baseline_abs_error_mean_m": row["baseline_abs_error_mean_m"],
                "selected_risk_given_selected": row["selected_risk_given_selected"],
                "weighted_selected_risk_error_mean": row["weighted_selected_risk_error_mean"],
                "post_delta_path": row["post_delta_path"],
            }
            for row in ranked_bad[:8]
        ],
        "top_good_safety_candidates": [
            {
                "seq": row["seq"],
                "chunk": row["chunk"],
                "case_types": row["case_types"],
                "baseline_abs_error_mean_m": row["baseline_abs_error_mean_m"],
                "selected_risk_given_selected": row["selected_risk_given_selected"],
                "weighted_selected_risk_error_mean": row["weighted_selected_risk_error_mean"],
                "post_delta_path": row["post_delta_path"],
            }
            for row in ranked_good[:8]
        ],
        "outputs": {
            "chunk_rows_csv": str(args.out_dir / "phase2_abs_error_overlap_chunk_rows.csv"),
            "frame_rows_csv": str(args.out_dir / "phase2_abs_error_overlap_frame_rows.csv"),
            "summary_json": str(args.out_dir / "phase2_abs_error_overlap_summary.json"),
        },
    }
    _write_csv(args.out_dir / "phase2_abs_error_overlap_chunk_rows.csv", chunk_rows)
    _write_csv(args.out_dir / "phase2_abs_error_overlap_frame_rows.csv", frame_rows)
    _write_json(args.out_dir / "phase2_abs_error_overlap_summary.json", summary)
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
