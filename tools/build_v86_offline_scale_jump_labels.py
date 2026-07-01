#!/usr/bin/env python3
"""Build ACL2 v86 offline scale-jump labels from landed chunk trajectories."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import _load_kitti_gt, _load_tum_prediction, _rmse, _umeyama_sim3  # noqa: E402
from v86_soft_latent_utils import safe_int, seq_norm, write_csv, write_json  # noqa: E402


DEFAULT_PHASE1 = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase1_soft_pair_universe")
DEFAULT_OUT = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase4_offline_scale_labels")
DEFAULT_GT_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    return parser.parse_args()


def _chunk_dir_from_feature(path_text: Any) -> Path | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.name.startswith("chunk_") and path.suffix == ".pt" and path.parent.name == "pca_features":
        return path.parent.parent
    return None


def _trajectory_path(seq: str, feature_path: Any) -> Path | None:
    chunk_dir = _chunk_dir_from_feature(feature_path)
    if chunk_dir is None:
        return None
    candidate = chunk_dir / f"{seq}.txt"
    return candidate if candidate.exists() else None


def _eval_traj(seq: str, traj: Path, gt_root: Path) -> dict[str, Any]:
    gt_path = gt_root / f"{seq}.txt"
    if not gt_path.exists():
        return {"trajectory": str(traj), "valid": False, "missing_reason": f"missing_gt:{gt_path}"}
    try:
        _, _, gt_pos = _load_kitti_gt(gt_path)
        frames, _, raw_pos = _load_tum_prediction(traj, gt_pos.shape[0])
        scale, _, _ = _umeyama_sim3(raw_pos, gt_pos[frames], with_scale=True)
        aligned_err = raw_pos - gt_pos[frames]
        rmse = _rmse(np.linalg.norm(aligned_err, axis=1))
    except Exception as exc:  # noqa: BLE001
        return {"trajectory": str(traj), "valid": False, "missing_reason": f"{type(exc).__name__}:{exc}"}
    return {
        "trajectory": str(traj),
        "valid": True,
        "frame_count": int(frames.size),
        "frame_start": int(frames.min()) if frames.size else None,
        "frame_end_exclusive": int(frames.max()) + 1 if frames.size else None,
        "chunk_sim3_scale": float(scale),
        "raw_position_rmse_before_alignment_proxy": float(rmse),
    }


def main() -> None:
    args = parse_args()
    pair_rows = pd.read_csv(args.phase1_dir / "soft_pair_by_seq_chunk.csv")
    soft_rows = pd.read_csv(args.phase1_dir / "soft_pair_rows.csv")

    source_by_chunk: dict[tuple[str, int], str] = {}
    for _, row in soft_rows.drop_duplicates(["seq", "curr_chunk", "feature_source_path"]).iterrows():
        seq = seq_norm(row.get("seq"))
        curr = safe_int(row.get("curr_chunk"))
        if curr is None:
            continue
        path = str(row.get("feature_source_path") or "")
        if path and (seq, curr) not in source_by_chunk:
            source_by_chunk[(seq, curr)] = path

    chunk_eval: dict[tuple[str, int], dict[str, Any]] = {}
    for key, source in source_by_chunk.items():
        seq, chunk = key
        traj = _trajectory_path(seq, source)
        if traj is None:
            chunk_eval[key] = {"valid": False, "trajectory": "", "missing_reason": "missing_chunk_trajectory"}
        else:
            chunk_eval[key] = _eval_traj(seq, traj, args.gt_root)

    rows: list[dict[str, Any]] = []
    for _, pair in pair_rows.iterrows():
        seq = seq_norm(pair.get("seq"))
        prev = int(pair["prev_chunk"])
        curr = int(pair["curr_chunk"])
        prev_eval = chunk_eval.get((seq, prev), {"valid": False, "missing_reason": "prev_chunk_scale_missing"})
        curr_eval = chunk_eval.get((seq, curr), {"valid": False, "missing_reason": "curr_chunk_scale_missing"})
        prev_scale = prev_eval.get("chunk_sim3_scale") if prev_eval.get("valid") else None
        curr_scale = curr_eval.get("chunk_sim3_scale") if curr_eval.get("valid") else None
        if prev_scale is not None and curr_scale is not None and prev_scale > 0 and curr_scale > 0:
            jump = math.log(float(curr_scale)) - math.log(float(prev_scale))
            abs_jump = abs(jump)
            sign = 1 if jump > 0 else (-1 if jump < 0 else 0)
            reason = ""
        else:
            jump = None
            abs_jump = None
            sign = None
            reason = ";".join(
                str(item)
                for item in [prev_eval.get("missing_reason"), curr_eval.get("missing_reason")]
                if item
            )
        rows.append(
            {
                "seq": seq,
                "prev_chunk": prev,
                "curr_chunk": curr,
                "case_label": pair.get("case_label"),
                "quality_label": pair.get("quality_label"),
                "chunk_scale_prev": prev_scale,
                "chunk_scale_curr": curr_scale,
                "adjacent_log_scale_jump": jump,
                "abs_log_scale_jump": abs_jump,
                "scale_jump_sign": sign,
                "full_ATE_contribution_proxy": curr_eval.get("raw_position_rmse_before_alignment_proxy"),
                "future_after_overlap": "",
                "boundary_jump": "",
                "overlap_scale_residual": "",
                "prev_trajectory": prev_eval.get("trajectory"),
                "curr_trajectory": curr_eval.get("trajectory"),
                "scale_label_available": jump is not None,
                "missing_reason": reason,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "offline_scale_jump_rows.csv", rows)
    labelled = [row for row in rows if row["case_label"] in {"bad", "good"}]
    available = [row for row in rows if row["scale_label_available"]]
    summary = {
        "phase": "Phase4_offline_scale_jump_labels",
        "pair_rows": len(rows),
        "scale_label_available_rows": len(available),
        "labelled_rows": len(labelled),
        "labelled_scale_label_available_rows": sum(1 for row in labelled if row["scale_label_available"]),
        "sequence_coverage": len({row["seq"] for row in available}),
        "gt_root": str(args.gt_root),
        "note": "Offline Sim3 scale labels are diagnostic only and are not runtime features or triggers.",
    }
    write_json(args.out_dir / "offline_scale_jump_summary.json", summary)
    print(f"scale_label_available_rows={summary['scale_label_available_rows']}")
    print(f"labelled_scale_label_available_rows={summary['labelled_scale_label_available_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")


if __name__ == "__main__":
    main()
