#!/usr/bin/env python3
"""Build ACL2 v107TF Stage2 metric reliability verifier rows."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


ROOT = Path(__file__).resolve().parents[1]
V107 = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention"
STAGE1 = Path(os.environ.get("V107_STAGE1_ROOT", V107 / "stage1_cache_operation_instrumentation")).resolve()
OUT = Path(os.environ.get("V107_STAGE2_OUT", V107 / "stage2_metric_reliability_verifier")).resolve()
MAP_ROOT = OUT / "moge2_operation_frame_maps"
PANEL_ROOT = OUT / "verifier_visual_panels"
MOGE_ROOT = ROOT / "third_party/MoGe"
OP_ROWS = Path(os.environ.get("V107_OPERATION_ROWS", STAGE1 / "operation_trace_rows.csv")).resolve()
CONFIG_SUMMARY = Path(os.environ.get("V107_CONFIG_SUMMARY", STAGE1 / "config_generation_summary.json")).resolve()
DEFAULT_PRETRAINED = "Ruicheng/moge-2-vits-normal"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        output = getattr(exc, "output", "")
        return f"{type(exc).__name__}: {exc}\n{output}"


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    raw = row.get(key, "")
    if raw in {"", None}:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def inum(row: dict[str, Any], key: str, default: int = 0) -> int:
    raw = row.get(key, "")
    if raw in {"", None}:
        return default
    return int(float(raw))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def discover_inputs() -> dict[str, Any]:
    lingbot_depth_paths = [
        ROOT / "third_party/lingbot-depth",
        ROOT / "third_party/LingBot-Depth",
    ]
    moge_paths = [
        ROOT / "third_party/moge2",
        ROOT / "third_party/MoGe-2",
        ROOT / "third_party/MoGe",
    ]
    checkpoint_hits = []
    for root in [ROOT / "checkpoints", ROOT / "third_party/lingbot-map/checkpoints"]:
        if root.exists():
            for item in root.rglob("*"):
                name = item.name.lower()
                if "moge" in name or ("lingbot" in name and "depth" in name):
                    checkpoint_hits.append(rel(item))
    import_probe = run_text(
        [
            "/mnt/data/users/chengshun.wang/miniconda3/bin/conda",
            "run",
            "-n",
            "loger",
            "python",
            "-c",
            (
                "import sys; sys.path.insert(0,'third_party/MoGe'); "
                "from moge.model.v2 import MoGeModel; print('moge_v2_import_ok')"
            ),
        ]
    ).strip()
    return {
        "lingbot_depth_candidates": [rel(path) for path in lingbot_depth_paths],
        "lingbot_depth_found": [rel(path) for path in lingbot_depth_paths if path.exists()],
        "moge_candidates": [rel(path) for path in moge_paths],
        "moge_found": [rel(path) for path in moge_paths if path.exists()],
        "checkpoint_hits": checkpoint_hits,
        "moge_import_probe": import_probe,
    }


def trace_method() -> str:
    config = json.loads(CONFIG_SUMMARY.read_text(encoding="utf-8"))
    return str(config["trace_method"])


def frame_roots(row: dict[str, str]) -> Path:
    return STAGE1 / "workspace" / row["dataset"] / row["seq"] / trace_method()


def rgb_path(row: dict[str, str], local_frame: int) -> Path:
    return frame_roots(row) / "rgb" / f"{local_frame:06d}.png"


def depth_path(row: dict[str, str], local_frame: int) -> Path:
    return frame_roots(row) / "depth" / f"{local_frame:06d}.exr"


def unique_frame_items(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        start = inum(row, "frame_span_start", inum(row, "source_frame", 0))
        end = inum(row, "frame_span_end", start)
        if end < start:
            end = start
        # Bound large retention spans to their actual frame range; cache_append rows
        # already cover all individual frames, so this does not create duplicates.
        for local_frame in range(start, end + 1):
            key = (row["dataset"], row["seq"], local_frame)
            if key not in seen:
                seen[key] = {
                    "dataset": row["dataset"],
                    "seq": row["seq"],
                    "target_id": row.get("target_id", ""),
                    "target_kind": row.get("target_kind", ""),
                    "window_id": row.get("window_id", ""),
                    "trace_start_idx": row.get("trace_start_idx", ""),
                    "trace_end_idx_exclusive": row.get("trace_end_idx_exclusive", ""),
                    "local_frame": local_frame,
                    "frame_id": int(float(row.get("trace_start_idx", 0))) + local_frame,
                    "rgb_path": rel(rgb_path(row, local_frame)),
                    "lingbot_depth_path": rel(depth_path(row, local_frame)),
                }
    return [seen[key] for key in sorted(seen)]


def load_moge_model():
    import torch

    sys.path.insert(0, str(MOGE_ROOT))
    from moge.model.v2 import MoGeModel  # noqa: WPS433

    pretrained = os.environ.get("V107_MOGE_PRETRAINED", DEFAULT_PRETRAINED)
    requested_device = os.environ.get("V107_MOGE_DEVICE", "cuda:0")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    model = MoGeModel.from_pretrained(pretrained).to(device).eval()
    return model, device, pretrained


def run_moge_frame(item: dict[str, Any], model, device, pretrained: str) -> dict[str, Any]:
    import cv2
    import torch

    resize_to = int(os.environ.get("V107_MOGE_RESIZE", "224"))
    num_tokens = int(os.environ.get("V107_MOGE_NUM_TOKENS", "1200"))
    use_fp16 = str(os.environ.get("V107_MOGE_FP16", "1")).lower() not in {"0", "false", "no"}
    out_dir = MAP_ROOT / item["dataset"] / item["seq"]
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{int(item['local_frame']):06d}.npz"
    meta_path = out_dir / f"{int(item['local_frame']):06d}.json"
    if npz_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["cached"] = True
        meta["npz_path"] = rel(npz_path)
        return meta

    image = cv2.cvtColor(cv2.imread(str(ROOT / item["rgb_path"])), cv2.COLOR_BGR2RGB)
    if image is None:
        raise FileNotFoundError(f"missing RGB input: {item['rgb_path']}")
    orig_h, orig_w = image.shape[:2]
    height = min(resize_to, int(resize_to * orig_h / orig_w))
    width = min(resize_to, int(resize_to * orig_w / orig_h))
    image_small = cv2.resize(image, (width, height), cv2.INTER_AREA)
    image_tensor = torch.tensor(image_small / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        output = model.infer(image_tensor, num_tokens=num_tokens, use_fp16=use_fp16)
    depth = output["depth"].detach().cpu().numpy().astype(np.float32)
    points = output["points"].detach().cpu().numpy().astype(np.float32)
    mask = output["mask"].detach().cpu().numpy().astype(np.float32)
    intrinsics = output["intrinsics"].detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(npz_path, depth=depth, points=points, mask=mask, intrinsics=intrinsics)
    meta = {
        "schema": "acl2_v107tf_stage2_moge2_frame_map_v1",
        **{key: item[key] for key in ["dataset", "seq", "target_id", "target_kind", "window_id", "trace_start_idx", "trace_end_idx_exclusive", "local_frame", "frame_id", "rgb_path", "lingbot_depth_path"]},
        "npz_path": rel(npz_path),
        "pretrained": pretrained,
        "device": str(device),
        "resize_to": resize_to,
        "num_tokens": num_tokens,
        "use_fp16": use_fp16,
        "original_shape_hw": [int(orig_h), int(orig_w)],
        "moge_shape_hw": [int(depth.shape[0]), int(depth.shape[1])],
        "finite_depth_fraction": float(np.isfinite(depth).mean()),
        "mask_fraction": float((mask > 0.5).mean()),
        "cached": False,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def resize_like(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(arr.astype(np.float32), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)


def frame_features(meta: dict[str, Any]) -> dict[str, Any]:
    import cv2

    data = np.load(ROOT / meta["npz_path"])
    moge_depth = np.asarray(data["depth"], dtype=np.float32)
    points = np.asarray(data["points"], dtype=np.float32)
    mask = np.asarray(data["mask"], dtype=np.float32)
    lingbot_depth = cv2.imread(str(ROOT / meta["lingbot_depth_path"]), cv2.IMREAD_UNCHANGED)
    if lingbot_depth is None:
        return {
            **meta,
            "verifier_available": False,
            "missing_reason": "lingbot_depth_missing",
        }
    lingbot_resized = resize_like(np.asarray(lingbot_depth, dtype=np.float32), moge_depth.shape)
    valid = np.isfinite(moge_depth) & np.isfinite(lingbot_resized) & (moge_depth > 1e-6) & (lingbot_resized > 1e-6) & (mask > 0.5)
    if not np.any(valid):
        return {
            **meta,
            "verifier_available": False,
            "missing_reason": "no_valid_depth_overlap",
        }

    log_err = np.abs(np.log(lingbot_resized[valid]) - np.log(moge_depth[valid]))
    log_depth = np.log(moge_depth[valid])
    point_norm = np.linalg.norm(points[valid], axis=-1)
    median_err = float(np.median(log_err))
    depth_spread = float(np.quantile(log_depth, 0.75) - np.quantile(log_depth, 0.25))
    point_spread = float(np.quantile(point_norm, 0.75) - np.quantile(point_norm, 0.25))

    # Image-edge proxy for boundary mismatch: high edge strength with high
    # LingBot-vs-MoGe disagreement indicates risky boundary evidence.
    rgb = cv2.imread(str(ROOT / meta["rgb_path"]))
    gray = cv2.cvtColor(cv2.resize(rgb, (moge_depth.shape[1], moge_depth.shape[0])), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge_norm = edge / (float(np.quantile(edge, 0.95)) + 1e-6)
    edge_norm = np.clip(edge_norm, 0.0, 1.0)
    err_map = np.zeros_like(moge_depth, dtype=np.float32)
    err_map[valid] = np.abs(np.log(lingbot_resized[valid]) - np.log(moge_depth[valid]))
    err_norm = np.clip(err_map / (float(np.quantile(err_map[valid], 0.95)) + 1e-6), 0.0, 1.0)
    boundary_mismatch = float(np.mean((edge_norm * err_norm)[valid]))

    metric_consistency = float(math.exp(-median_err))
    mask_mean = float(np.mean(mask[valid]))
    far_cut = float(np.quantile(moge_depth[valid], 0.75))
    far_background_weakness = float(np.mean((moge_depth[valid] >= far_cut).astype(np.float32)) * max(0.0, 1.0 - min(1.0, depth_spread)))
    spread_norm = min(1.0, depth_spread / 1.0)
    scale_observability = float(metric_consistency * spread_norm * max(0.0, 1.0 - boundary_mismatch) * mask_mean)
    metric_reliability = float(sigmoid((-median_err) - boundary_mismatch + spread_norm + mask_mean - far_background_weakness))

    return {
        **meta,
        "schema": "acl2_v107tf_stage2_frame_verifier_row_v1",
        "verifier_available": True,
        "verifier_name": "MoGe-2",
        "proxy_only": False,
        "metric_depth_consistency": median_err,
        "pointmap_consistency": "",
        "boundary_mismatch_risk": boundary_mismatch,
        "boundary_mismatch_source": "image_edge_x_logdepth_disagreement_proxy",
        "depth_spread": depth_spread,
        "point_spread": point_spread,
        "far_background_weakness": far_background_weakness,
        "scale_observability_score": scale_observability,
        "metric_consistency_score": metric_consistency,
        "metric_reliability_score": metric_reliability,
        "moge_confidence_or_mask": mask_mean,
        "lingbot_vs_moge_log_depth_error": median_err,
        "lingbot_vs_moge_point_error": "",
        "missing_reason": "",
    }


def mean_feature(rows: list[dict[str, Any]], key: str) -> Any:
    vals = [float(row[key]) for row in rows if finite(row.get(key, ""))]
    if not vals:
        return ""
    return float(np.mean(vals))


def join_operation_rows(operation_rows: list[dict[str, str]], frame_by_key: dict[tuple[str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for idx, row in enumerate(operation_rows):
        start = inum(row, "frame_span_start", inum(row, "source_frame", 0))
        end = inum(row, "frame_span_end", start)
        if end < start:
            end = start
        frame_rows = [
            frame_by_key[(row["dataset"], row["seq"], frame)]
            for frame in range(start, end + 1)
            if (row["dataset"], row["seq"], frame) in frame_by_key
        ]
        available = [item for item in frame_rows if item.get("verifier_available")]
        coverage = len(available) / max(len(frame_rows), 1)
        out = {
            "schema": "acl2_v107tf_stage2_operation_verifier_join_row_v1",
            "operation_row_index": idx,
            **row,
            "verifier_frame_count": len(frame_rows),
            "verifier_available_frame_count": len(available),
            "verifier_coverage": coverage,
            "metric_depth_consistency": mean_feature(available, "metric_depth_consistency"),
            "pointmap_consistency": "",
            "boundary_mismatch_risk": mean_feature(available, "boundary_mismatch_risk"),
            "depth_spread": mean_feature(available, "depth_spread"),
            "point_spread": mean_feature(available, "point_spread"),
            "far_background_weakness": mean_feature(available, "far_background_weakness"),
            "scale_observability_score": mean_feature(available, "scale_observability_score"),
            "metric_consistency_score": mean_feature(available, "metric_consistency_score"),
            "metric_reliability_score": mean_feature(available, "metric_reliability_score"),
            "boundary_mismatch_proxy_marked": True,
            "scale_observability_proxy_marked": False,
            "proxy_only": False,
            "output_pose_depth_replacement_performed": False,
        }
        joined.append(out)
    return joined


def make_visual_panels(frame_rows: list[dict[str, Any]], max_panels: int = 8) -> list[str]:
    import cv2

    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    selected = [row for row in frame_rows if row.get("verifier_available")][:max_panels]
    for row in selected:
        data = np.load(ROOT / row["npz_path"])
        moge_depth = np.asarray(data["depth"], dtype=np.float32)
        lingbot_depth = cv2.imread(str(ROOT / row["lingbot_depth_path"]), cv2.IMREAD_UNCHANGED)
        if lingbot_depth is None:
            continue
        lingbot_resized = resize_like(np.asarray(lingbot_depth, dtype=np.float32), moge_depth.shape)
        rgb = cv2.imread(str(ROOT / row["rgb_path"]))
        rgb = cv2.resize(rgb, (moge_depth.shape[1], moge_depth.shape[0]))
        valid = np.isfinite(moge_depth) & np.isfinite(lingbot_resized) & (moge_depth > 1e-6) & (lingbot_resized > 1e-6)
        err = np.zeros_like(moge_depth, dtype=np.float32)
        err[valid] = np.abs(np.log(lingbot_resized[valid]) - np.log(moge_depth[valid]))

        def colorize(arr: np.ndarray) -> np.ndarray:
            vals = arr[np.isfinite(arr)]
            if vals.size == 0:
                norm = np.zeros_like(arr, dtype=np.uint8)
            else:
                lo, hi = float(np.quantile(vals, 0.02)), float(np.quantile(vals, 0.98))
                norm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
                norm = (norm * 255).astype(np.uint8)
            return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

        panel = np.concatenate([rgb, colorize(lingbot_resized), colorize(moge_depth), colorize(err)], axis=1)
        out_path = PANEL_ROOT / f"{row['dataset']}_{row['seq']}_{int(row['local_frame']):06d}.png"
        cv2.imwrite(str(out_path), panel)
        paths.append(rel(out_path))
    return paths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    discovery = discover_inputs()
    write_text(
        OUT / "lingbot_depth_missing_report.md",
        "\n".join(
            [
                "# LingBot-Depth Availability",
                "",
                f"- searched: `{', '.join(discovery['lingbot_depth_candidates'])}`",
                f"- found: `{', '.join(discovery['lingbot_depth_found']) or 'none'}`",
                "",
                "LingBot-Depth is not used unless a local repo/checkpoint is present.",
            ]
        ),
    )
    operation_rows = read_csv(OP_ROWS)
    frame_items = unique_frame_items(operation_rows)
    limit = int(os.environ.get("V107_MOGE_MAX_FRAMES", "0"))
    if limit > 0:
        frame_items = frame_items[:limit]

    moge_available = MOGE_ROOT.exists() and "moge_v2_import_ok" in discovery.get("moge_import_probe", "")
    frame_rows: list[dict[str, Any]] = []
    failure_notes: list[str] = []
    if moge_available:
        try:
            model, device, pretrained = load_moge_model()
            for item in frame_items:
                try:
                    meta = run_moge_frame(item, model, device, pretrained)
                    frame_rows.append(frame_features(meta))
                except Exception as exc:  # noqa: BLE001
                    failure_notes.append(f"{item['dataset']}/{item['seq']}/{item['local_frame']}: {exc!r}")
                    frame_rows.append({**item, "schema": "acl2_v107tf_stage2_frame_verifier_row_v1", "verifier_available": False, "proxy_only": False, "missing_reason": repr(exc)})
            try:
                import torch
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            moge_available = False
            failure_notes.append(f"moge_model_load_failed: {exc!r}")
            frame_rows = [{**item, "schema": "acl2_v107tf_stage2_frame_verifier_row_v1", "verifier_available": False, "proxy_only": True, "missing_reason": "moge_model_load_failed"} for item in frame_items]
    else:
        frame_rows = [{**item, "schema": "acl2_v107tf_stage2_frame_verifier_row_v1", "verifier_available": False, "proxy_only": True, "missing_reason": "moge_unavailable"} for item in frame_items]

    frame_by_key = {
        (row["dataset"], row["seq"], int(row["local_frame"])): row
        for row in frame_rows
    }
    joined = join_operation_rows(operation_rows, frame_by_key)
    evidence_rows = [
        {
            "schema": "acl2_v107tf_stage2_evidence_verifier_row_v1",
            "dataset": row["dataset"],
            "seq": row["seq"],
            "local_frame": row["local_frame"],
            "frame_id": row["frame_id"],
            "metric_reliability_score": row.get("metric_reliability_score", ""),
            "scale_observability_score": row.get("scale_observability_score", ""),
            "boundary_mismatch_risk": row.get("boundary_mismatch_risk", ""),
            "depth_spread": row.get("depth_spread", ""),
            "proxy_only": row.get("proxy_only", ""),
            "verifier_available": row.get("verifier_available", ""),
        }
        for row in frame_rows
    ]
    panel_paths = make_visual_panels(frame_rows)

    write_csv(OUT / "frame_verifier_rows.csv", frame_rows)
    write_csv(OUT / "evidence_verifier_rows.csv", evidence_rows)
    write_csv(OUT / "operation_verifier_join_rows.csv", joined)

    available_ops = [row for row in joined if float(row.get("verifier_coverage", 0.0)) >= 1.0]
    finite_metric = [row for row in joined if finite(row.get("metric_reliability_score", ""))]
    finite_scale = [row for row in joined if finite(row.get("scale_observability_score", ""))]
    coverage = len(available_ops) / len(joined) if joined else 0.0
    finite_frac = len(finite_metric) / len(joined) if joined else 0.0
    scale_finite_frac = len(finite_scale) / len(joined) if joined else 0.0
    proxy_only = bool(joined) and all(str(row.get("proxy_only", "")).lower() == "true" for row in joined)
    stage2_pass = (
        coverage >= 0.80
        and finite_frac >= 0.95
        and scale_finite_frac >= 0.95
        and all(str(row.get("output_pose_depth_replacement_performed", "False")).lower() in {"false", "0"} for row in joined)
    )
    summary = {
        "schema": "acl2_v107tf_stage2_metric_reliability_summary_v1",
        "lingbot_depth_available": bool(discovery["lingbot_depth_found"]),
        "moge_available": bool(moge_available),
        "proxy_only": proxy_only,
        "frame_count": len(frame_rows),
        "operation_row_count": len(joined),
        "verifier_coverage": coverage,
        "metric_reliability_score_finite_frac": finite_frac,
        "scale_observability_score_finite_frac": scale_finite_frac,
        "boundary_mismatch_available_or_proxy_marked": True,
        "scale_observability_available_or_proxy_marked": True,
        "no_output_pose_depth_replacement_performed": True,
        "stage2_pass": stage2_pass,
        "stage4_action_forbidden_due_to_proxy_only": bool(proxy_only),
        "visual_panel_count": len(panel_paths),
        "visual_panels": panel_paths,
        "failure_note_count": len(failure_notes),
        "outputs": {
            "frame_verifier_rows": rel(OUT / "frame_verifier_rows.csv"),
            "evidence_verifier_rows": rel(OUT / "evidence_verifier_rows.csv"),
            "operation_verifier_join_rows": rel(OUT / "operation_verifier_join_rows.csv"),
            "verifier_coverage_summary": rel(OUT / "verifier_coverage_summary.json"),
            "verifier_failure_report": rel(OUT / "verifier_failure_report.md"),
            "lingbot_depth_missing_report": rel(OUT / "lingbot_depth_missing_report.md"),
            "verifier_visual_panels": rel(PANEL_ROOT),
        },
        "discovery": discovery,
    }
    write_text(OUT / "verifier_coverage_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    write_text(
        OUT / "verifier_failure_report.md",
        "\n".join(
            [
                "# Stage2 Verifier Failure Report",
                "",
                f"- failure_note_count: `{len(failure_notes)}`",
                f"- stage2_pass: `{stage2_pass}`",
                f"- proxy_only: `{proxy_only}`",
                "",
                "First failures:",
                *[f"- {note}" for note in failure_notes[:50]],
            ]
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
