from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .self_stitch import residual_diagnostics, scene_scale_from_points


@dataclass(frozen=True)
class Sim3Transform:
    scale: float
    rot: np.ndarray
    trans: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale": float(self.scale),
            "rot": np.asarray(self.rot, dtype=np.float64),
            "trans": np.asarray(self.trans, dtype=np.float64),
        }


def _coerce_sim3(value: Sim3Transform | dict[str, Any] | tuple[Any, Any, Any]) -> Sim3Transform:
    if isinstance(value, Sim3Transform):
        return value
    if isinstance(value, dict):
        rot = value.get("rot", value.get("rotation"))
        trans = value.get("trans", value.get("translation"))
        return Sim3Transform(float(value["scale"]), np.asarray(rot, dtype=np.float64), np.asarray(trans, dtype=np.float64))
    scale, rot, trans = value
    return Sim3Transform(float(scale), np.asarray(rot, dtype=np.float64), np.asarray(trans, dtype=np.float64))


def _umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> Sim3Transform | None:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2 or src.shape != dst.shape or src.shape[1] != 3 or src.shape[0] < 3:
        return None

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    cov = (src_centered.T @ dst_centered) / float(src.shape[0])
    u, s, vt = np.linalg.svd(cov)
    d = np.ones((3,), dtype=np.float64)
    if np.linalg.det(vt.T @ u.T) < 0:
        d[-1] = -1.0
    rot = vt.T @ np.diag(d) @ u.T
    var_src = float((src_centered**2).sum() / float(src.shape[0]))
    if var_src <= 1e-12:
        return None
    scale = float((s * d).sum() / var_src)
    trans = mu_dst - scale * (rot @ mu_src)
    if not (np.isfinite(scale) and np.isfinite(rot).all() and np.isfinite(trans).all()):
        return None
    return Sim3Transform(scale=scale, rot=rot, trans=trans)


def fit_sim3_umeyama(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Fit ``target ~= scale * rotation * source + translation``.

    This public helper is the single packaged Sim3 implementation used by
    providers and diagnostic tools. It preserves the legacy dict keys consumed
    by the v21.x scripts while avoiding imports from ``tools/`` in production
    provider code.
    """

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if source.shape[0] < 4:
        raise ValueError("at least 4 anchors are required for Sim3 fit")

    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    if source.shape[0] < 4:
        raise ValueError("at least 4 finite anchors are required for Sim3 fit")

    fit = _umeyama_sim3(source, target)
    if fit is None:
        raise ValueError("degenerate source anchors")
    transformed = float(fit.scale) * (source @ fit.rot.T) + fit.trans
    residual = np.linalg.norm(transformed - target, axis=1)
    return {
        "scale": float(fit.scale),
        "rotation": fit.rot.astype(np.float64),
        "rot": fit.rot.astype(np.float64),
        "rotation_det": float(np.linalg.det(fit.rot)),
        "translation": fit.trans.astype(np.float64),
        "trans": fit.trans.astype(np.float64),
        "residual": residual.astype(np.float64),
        "anchor_count": int(source.shape[0]),
    }


def apply_sim3_to_xyz(
    xyz: np.ndarray,
    scale: float | None = None,
    rot: np.ndarray | None = None,
    trans: np.ndarray | None = None,
    transform: Sim3Transform | dict[str, Any] | tuple[Any, Any, Any] | None = None,
) -> np.ndarray:
    """Apply ``scale * rot @ xyz + trans`` while preserving batch shape."""

    if transform is not None:
        parsed = _coerce_sim3(transform)
        scale = parsed.scale
        rot = parsed.rot
        trans = parsed.trans
    if scale is None or rot is None or trans is None:
        raise ValueError("scale, rot and trans are required unless transform is provided")
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.shape[-1] != 3:
        raise ValueError(f"xyz last dimension must be 3, got {pts.shape}")
    out = np.full_like(pts, np.nan, dtype=np.float64)
    flat = pts.reshape(-1, 3)
    valid = np.isfinite(flat).all(axis=1)
    if np.any(valid):
        r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
        t = np.asarray(trans, dtype=np.float64).reshape(3)
        out.reshape(-1, 3)[valid] = (float(scale) * (r @ flat[valid].T)).T + t
    return out.astype(np.float32)


def estimate_overlap_sim3(
    prev_xyz_qt3: np.ndarray,
    curr_xyz_qt3: np.ndarray,
    prev_vis_qt: np.ndarray,
    curr_vis_qt: np.ndarray,
    prev_conf_qt: np.ndarray | None = None,
    curr_conf_qt: np.ndarray | None = None,
    *,
    min_points: int = 16,
    min_inlier_ratio: float = 0.50,
    residual_quantile: float = 90.0,
    confidence_keep_ratio: float = 0.85,
) -> dict[str, Any] | None:
    """Estimate curr->prev Sim3 from overlapping D4RT points.

    Inputs are shaped ``[Q, T, 3]`` for xyz and ``[Q, T]`` for visibility/confidence.
    The function uses only D4RT xyz, visibility and confidence. GT/RGB-D alignment
    belongs in evaluation-only tools.
    """

    prev = np.asarray(prev_xyz_qt3, dtype=np.float64)
    curr = np.asarray(curr_xyz_qt3, dtype=np.float64)
    if prev.shape != curr.shape or prev.ndim != 3 or prev.shape[-1] != 3:
        return None
    valid = (
        np.asarray(prev_vis_qt, dtype=bool)
        & np.asarray(curr_vis_qt, dtype=bool)
        & np.isfinite(prev).all(axis=-1)
        & np.isfinite(curr).all(axis=-1)
    )
    if int(np.count_nonzero(valid)) < int(min_points):
        return None

    src = curr[valid]
    dst = prev[valid]
    if prev_conf_qt is not None and curr_conf_qt is not None:
        score = np.minimum(np.asarray(prev_conf_qt, dtype=np.float64)[valid], np.asarray(curr_conf_qt, dtype=np.float64)[valid])
        score = np.nan_to_num(score, nan=-np.inf)
        keep_count = max(int(min_points), int(round(score.shape[0] * float(np.clip(confidence_keep_ratio, 0.0, 1.0)))))
        if score.shape[0] > keep_count:
            keep = np.argsort(score)[-keep_count:]
            src = src[keep]
            dst = dst[keep]

    fit = _umeyama_sim3(src, dst)
    if fit is None:
        return None
    pred = apply_sim3_to_xyz(src, transform=fit)
    residual = np.linalg.norm(pred.astype(np.float64) - dst, axis=1)
    if residual.size == 0 or not np.isfinite(residual).all():
        return None
    metrics = residual_diagnostics(residual, scene_scale=scene_scale_from_points(dst))
    inlier_ratio = float(metrics["inlier_ratio_abs010"] or 0.0)
    threshold = float(metrics["mad_threshold"] or np.percentile(residual, residual_quantile))
    threshold = max(threshold, 0.10)
    inliers = residual <= threshold
    if int(np.count_nonzero(inliers)) < int(min_points) or inlier_ratio < float(min_inlier_ratio):
        return None

    refined = _umeyama_sim3(src[inliers], dst[inliers])
    if refined is None:
        return None
    refined_pred = apply_sim3_to_xyz(src[inliers], transform=refined)
    refined_residual = np.linalg.norm(refined_pred.astype(np.float64) - dst[inliers], axis=1)
    refined_metrics = residual_diagnostics(refined_residual, scene_scale=scene_scale_from_points(dst[inliers]))
    return {
        "scale": float(refined.scale),
        "rot": refined.rot.astype(np.float64),
        "trans": refined.trans.astype(np.float64),
        "num_candidates": int(src.shape[0]),
        "num_inliers": int(np.count_nonzero(inliers)),
        "inlier_ratio": inlier_ratio,
        **refined_metrics,
    }


def compose_sim3(a: Sim3Transform | dict[str, Any] | tuple[Any, Any, Any], b: Sim3Transform | dict[str, Any] | tuple[Any, Any, Any]) -> Sim3Transform:
    """Return the transform equivalent to applying ``a`` then ``b``."""

    first = _coerce_sim3(a)
    second = _coerce_sim3(b)
    scale = second.scale * first.scale
    rot = second.rot @ first.rot
    trans = second.scale * (second.rot @ first.trans) + second.trans
    return Sim3Transform(scale=float(scale), rot=rot.astype(np.float64), trans=trans.astype(np.float64))


def invert_sim3(t: Sim3Transform | dict[str, Any] | tuple[Any, Any, Any]) -> Sim3Transform:
    parsed = _coerce_sim3(t)
    inv_scale = 1.0 / float(parsed.scale)
    inv_rot = parsed.rot.T
    inv_trans = -inv_scale * (inv_rot @ parsed.trans)
    return Sim3Transform(scale=float(inv_scale), rot=inv_rot.astype(np.float64), trans=inv_trans.astype(np.float64))
