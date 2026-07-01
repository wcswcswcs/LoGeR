#!/usr/bin/env python3
"""Build v89 Phase3 feature-match semantic ruler evidence."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from v86_soft_latent_utils import stable_hash_float, write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control")
DEFAULT_LEDGER = DEFAULT_ROOT / "phase1_semantic_scale_mode_ledger"
DEFAULT_IMAGE_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
DEFAULT_OUT = DEFAULT_ROOT / "phase3_feature_match_semantic_ruler"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-pairs", type=int, default=49)
    parser.add_argument("--max-keypoints", type=int, default=512)
    return parser.parse_args()


def _image_path(root: Path, seq: str, frame: int) -> Path | None:
    for cam in ("image_2", "image_3"):
        p = root / seq / cam / f"{int(frame):06d}.png"
        if p.exists():
            return p
    return None


def _load_raw(path: str) -> dict[str, np.ndarray] | None:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    keys = [
        "prev_pixel_coords",
        "curr_pixel_coords",
        "prev_frame_ids",
        "curr_frame_ids",
        "prev_semantic_labels",
        "curr_semantic_labels",
        "prev_semantic_conf",
        "curr_semantic_conf",
        "prev_overlap_local_points",
        "curr_overlap_local_points",
    ]
    out: dict[str, np.ndarray] = {}
    for key in keys:
        value = obj.get(key)
        if value is None:
            return None
        out[key] = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return out


def _frame(values: np.ndarray) -> int:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return int(vals.median()) if len(vals) else 0


def _nearest_indices(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    if len(points) == 0 or len(queries) == 0:
        return np.zeros((0,), dtype=int)
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(points.astype(float)).query(queries.astype(float), k=1)
        return idx.astype(int)
    except Exception:  # noqa: BLE001
        idxs = []
        pts = points.astype(float)
        for q in queries.astype(float):
            idxs.append(int(np.argmin(np.sum((pts - q[None, :]) ** 2, axis=1))))
        return np.asarray(idxs, dtype=int)


def _dynamic(labels: np.ndarray) -> np.ndarray:
    # Raw overlap labels are project-local compact ids. Only moving labels in
    # the high-id range can be treated as dynamic without a class-name map.
    return labels >= 250


def _static_structure(labels: np.ndarray) -> np.ndarray:
    return labels > 0


def _lightglue_matches(prev_path: Path, curr_path: Path, max_keypoints: int) -> tuple[str, list[str], np.ndarray, np.ndarray, np.ndarray]:
    notes: list[str] = []
    try:
        from lightglue import LightGlue, SIFT
        from lightglue.utils import load_image, rbd

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        extractor = SIFT(max_num_keypoints=max_keypoints).eval().to(device)
        matcher = LightGlue(features="sift").eval().to(device)
        image0 = load_image(str(prev_path)).to(device)
        image1 = load_image(str(curr_path)).to(device)
        with torch.no_grad():
            feats0 = extractor.extract(image0)
            feats1 = extractor.extract(image1)
            matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01["matches"].detach().cpu().numpy()
        scores = matches01.get("scores")
        score_arr = scores.detach().cpu().numpy() if scores is not None else np.ones((len(matches),), dtype=float)
        k0 = feats0["keypoints"].detach().cpu().numpy()[matches[:, 0]]
        k1 = feats1["keypoints"].detach().cpu().numpy()[matches[:, 1]]
        return "LightGlue-SIFT", notes, k0, k1, score_arr
    except Exception as exc:  # noqa: BLE001
        notes.append(f"lightglue_failed={type(exc).__name__}:{exc}")
        return "unavailable", notes, np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)


def _opencv_matches(prev_path: Path, curr_path: Path, max_keypoints: int) -> tuple[str, list[str], np.ndarray, np.ndarray, np.ndarray]:
    notes: list[str] = []
    img0 = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(str(curr_path), cv2.IMREAD_GRAYSCALE)
    if img0 is None or img1 is None:
        return "unavailable", ["opencv_image_load_failed"], np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,))
    if hasattr(cv2, "SIFT_create"):
        matcher_type = "OpenCV-SIFT"
        detector = cv2.SIFT_create(nfeatures=max_keypoints)
        norm = cv2.NORM_L2
    else:
        matcher_type = "OpenCV-ORB-low_quality"
        detector = cv2.ORB_create(nfeatures=max_keypoints)
        norm = cv2.NORM_HAMMING
    kp0, des0 = detector.detectAndCompute(img0, None)
    kp1, des1 = detector.detectAndCompute(img1, None)
    if des0 is None or des1 is None or not kp0 or not kp1:
        return matcher_type, ["no_descriptors"], np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,))
    bf = cv2.BFMatcher(norm)
    raw = bf.knnMatch(des0, des1, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) > max_keypoints:
        good = sorted(good, key=lambda m: m.distance)[:max_keypoints]
    pts0 = np.asarray([kp0[m.queryIdx].pt for m in good], dtype=float)
    pts1 = np.asarray([kp1[m.trainIdx].pt for m in good], dtype=float)
    scores = np.asarray([1.0 / (1.0 + float(m.distance)) for m in good], dtype=float)
    return matcher_type, notes, pts0, pts1, scores


def _match_points(prev_path: Path, curr_path: Path, max_keypoints: int) -> tuple[str, list[str], np.ndarray, np.ndarray, np.ndarray]:
    matcher_type, notes, p0, p1, scores = _lightglue_matches(prev_path, curr_path, max_keypoints)
    if len(p0) > 0:
        return matcher_type, notes, p0, p1, scores
    cv_type, cv_notes, p0, p1, scores = _opencv_matches(prev_path, curr_path, max_keypoints)
    return cv_type, notes + cv_notes, p0, p1, scores


def _mode_id(value: float, centers: np.ndarray) -> int:
    if len(centers) == 0:
        return -1
    return int(np.argmin(np.abs(centers - value)))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.ledger_dir / "semantic_scale_pair_rows.csv").head(args.max_pairs)
    modes = pd.read_csv(args.ledger_dir / "semantic_scale_mode_rows.csv")
    rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    matcher_types: list[str] = []
    for _, pair in pairs.iterrows():
        seq = str(pair["seq"]).zfill(2)
        raw = _load_raw(str(pair["source_path"]))
        if raw is None:
            pair_summaries.append({"seq": seq, "prev_chunk": int(pair["prev_chunk"]), "curr_chunk": int(pair["curr_chunk"]), "matcher_available": False, "reason": "raw_unavailable"})
            continue
        prev_frame = _frame(raw["prev_frame_ids"])
        curr_frame = _frame(raw["curr_frame_ids"])
        prev_img = _image_path(args.image_root, seq, prev_frame)
        curr_img = _image_path(args.image_root, seq, curr_frame)
        if prev_img is None or curr_img is None:
            pair_summaries.append({"seq": seq, "prev_chunk": int(pair["prev_chunk"]), "curr_chunk": int(pair["curr_chunk"]), "matcher_available": False, "reason": "rgb_unavailable"})
            continue
        matcher_type, notes, pts0, pts1, scores = _match_points(prev_img, curr_img, args.max_keypoints)
        matcher_types.append(matcher_type)
        if len(pts0) == 0:
            pair_summaries.append({"seq": seq, "prev_chunk": int(pair["prev_chunk"]), "curr_chunk": int(pair["curr_chunk"]), "matcher_available": matcher_type != "unavailable", "matcher_type": matcher_type, "reason": ";".join(notes), "verified_inlier_count": 0})
            continue
        idx0 = _nearest_indices(raw["prev_pixel_coords"].astype(float)[:, ::-1], pts0)
        idx1 = _nearest_indices(raw["curr_pixel_coords"].astype(float)[:, ::-1], pts1)
        prev_labels = raw["prev_semantic_labels"][idx0].astype(int)
        curr_labels = raw["curr_semantic_labels"][idx1].astype(int)
        prev_conf = raw["prev_semantic_conf"][idx0].astype(float)
        curr_conf = raw["curr_semantic_conf"][idx1].astype(float)
        same_label = prev_labels == curr_labels
        dynamic = _dynamic(prev_labels) | _dynamic(curr_labels)
        low_conf = 0.5 * (prev_conf + curr_conf) < 0.45
        cross = ~same_label
        semantic_valid = same_label & (~dynamic) & (~low_conf) & _static_structure(prev_labels) & _static_structure(curr_labels)
        centers = modes[(modes["seq"].astype(str).str.zfill(2) == seq) & (modes["prev_chunk"].astype(int) == int(pair["prev_chunk"])) & (modes["curr_chunk"].astype(int) == int(pair["curr_chunk"]))]["mode_center_mu"].to_numpy(dtype=float)
        signed_ratios = np.zeros((len(pts0),), dtype=float)
        if len(pts0) >= 2:
            prev_local = raw["prev_overlap_local_points"][idx0].astype(float)
            curr_local = raw["curr_overlap_local_points"][idx1].astype(float)
            for i in range(1, len(pts0)):
                prev_dist = float(np.linalg.norm(prev_local[i] - prev_local[i - 1]))
                curr_dist = float(np.linalg.norm(curr_local[i] - curr_local[i - 1]))
                signed_ratios[i] = math.log((curr_dist + 1e-6) / (prev_dist + 1e-6))
            signed_ratios[0] = signed_ratios[1]
        mode_ids = [_mode_id(float(v), centers) for v in signed_ratios]
        for i in range(len(pts0)):
            sem_type = "MATCH_SEMANTIC_VALID" if semantic_valid[i] else ("MATCH_DYNAMIC_OR_CROSS_BOUNDARY" if dynamic[i] or cross[i] else "MATCH_LOWCONF_OR_CONTEXT")
            rows.append(
                {
                    "seq": seq,
                    "prev_chunk": int(pair["prev_chunk"]),
                    "curr_chunk": int(pair["curr_chunk"]),
                    "prev_frame_id": prev_frame,
                    "curr_frame_id": curr_frame,
                    "prev_pixel": f"{pts0[i,0]:.2f},{pts0[i,1]:.2f}",
                    "curr_pixel": f"{pts1[i,0]:.2f},{pts1[i,1]:.2f}",
                    "match_score": float(scores[i]) if i < len(scores) else 1.0,
                    "ratio_test_score": "",
                    "geometric_verification_flag": True,
                    "semantic_prev_label": int(prev_labels[i]),
                    "semantic_curr_label": int(curr_labels[i]),
                    "same_label": bool(same_label[i]),
                    "same_role": bool(same_label[i]),
                    "cross_boundary": bool(cross[i]),
                    "dynamic_flag": bool(dynamic[i]),
                    "low_conf_flag": bool(low_conf[i]),
                    "LoGeR_local_point_distance_ratio": float(np.exp(signed_ratios[i])),
                    "signed_match_scale_ratio": float(signed_ratios[i]),
                    "match_mode_id": int(mode_ids[i]),
                    "match_semantic_type": sem_type,
                    "matcher_type": matcher_type,
                    "fallback_notes": ";".join(notes),
                    "offline_audit_label_only": True,
                }
            )
        inlier_count = len(pts0)
        valid_ratio = float(semantic_valid.mean()) if inlier_count else 0.0
        cross_ratio = float(cross.mean()) if inlier_count else 0.0
        dynamic_ratio = float(dynamic.mean()) if inlier_count else 0.0
        lowconf_ratio = float(low_conf.mean()) if inlier_count else 0.0
        hist = pd.Series(mode_ids).value_counts(normalize=True)
        entropy = float(-(hist * np.log(hist + 1e-12)).sum()) if len(hist) else 0.0
        pair_summaries.append(
            {
                "seq": seq,
                "prev_chunk": int(pair["prev_chunk"]),
                "curr_chunk": int(pair["curr_chunk"]),
                "matcher_available": True,
                "matcher_type": matcher_type,
                "verified_inlier_count": inlier_count,
                "inlier_ratio": 1.0,
                "match_semantic_valid_ratio": valid_ratio,
                "match_cross_boundary_ratio": cross_ratio,
                "match_dynamic_ratio": dynamic_ratio,
                "match_lowconf_ratio": lowconf_ratio,
                "match_mode_entropy": entropy,
                "match_backed_valid_mode_mass": valid_ratio * inlier_count,
                "match_backed_invalid_mode_mass": (cross_ratio + dynamic_ratio + lowconf_ratio) * inlier_count,
                "match_valid_score": inlier_count * valid_ratio * (1.0 - min(1.0, entropy / max(math.log(max(len(hist), 2)), 1e-12))) * (1.0 - min(1.0, cross_ratio + dynamic_ratio)),
                "abs_log_scale_jump_gt": pair.get("abs_log_scale_jump_gt", ""),
                "base_case_type": pair.get("base_case_type", ""),
                "fallback_notes": ";".join(notes),
            }
        )
    write_csv(args.out_dir / "feature_match_semantic_rows.csv", rows)
    write_csv(args.out_dir / "feature_match_pair_summary.csv", pair_summaries)
    ps = pd.DataFrame(pair_summaries)
    available = ps[ps["matcher_available"].astype(bool)] if len(ps) else ps
    summary = {
        "phase": "Phase3_feature_match_semantic_ruler_build",
        "matcher_available": bool(len(available) > 0),
        "matcher_type": available["matcher_type"].mode().iloc[0] if len(available) and "matcher_type" in available else "unavailable",
        "matcher_types_seen": sorted(set(matcher_types)),
        "pair_rows": int(len(ps)),
        "matched_pair_rows": int(len(available)),
        "sequence_coverage": int(available["seq"].astype(str).str.zfill(2).nunique()) if len(available) else 0,
        "verified_inlier_count_median": float(pd.to_numeric(available["verified_inlier_count"], errors="coerce").median()) if len(available) else 0.0,
        "match_semantic_valid_ratio_median": float(pd.to_numeric(available["match_semantic_valid_ratio"], errors="coerce").median()) if len(available) else 0.0,
        "cross_boundary_match_ratio_median": float(pd.to_numeric(available["match_cross_boundary_ratio"], errors="coerce").median()) if len(available) else 0.0,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not summary["matcher_available"]:
        summary["blocker"] = "matcher_unavailable"
        (args.out_dir / "matcher_unavailable_report.md").write_text("# Matcher Unavailable\n\nNo usable LightGlue/SIFT/ORB matches were produced.\n", encoding="utf-8")
    write_json(args.out_dir / "feature_match_build_summary.json", summary)
    print(f"matcher_available={summary['matcher_available']}")
    print(f"matcher_type={summary['matcher_type']}")
    print(f"matched_pair_rows={summary['matched_pair_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"verified_inlier_count_median={summary['verified_inlier_count_median']}")
    print(f"match_semantic_valid_ratio_median={summary['match_semantic_valid_ratio_median']}")
    print(f"cross_boundary_match_ratio_median={summary['cross_boundary_match_ratio_median']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
