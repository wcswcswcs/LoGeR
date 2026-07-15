#!/usr/bin/env python3
"""Build ACL2 v106 Stage2 MoGe-2 verifier, with explicit proxy fallback."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V106 = ROOT / "results/acl2_v106tf_lingbot_semantic_aware_memory_role_control"
STAGE1 = V106 / "stage1_selected_evidence_materialization"
OUT = V106 / "stage2_moge_metric_verifier"
HEAD_TRACE_ROWS = V105 / "stage4_lingbot_headlocal_trace/headlocal_trace_semantic_key_rows.csv"
BASELINE_WORKSPACE = V105 / "stage2_gca_trace/workspace"
RGB_WORKSPACE = V105 / "stage1_lingbot_baseline/workspace/kitti_v105_00_01_02_05"
MOGE_ROOT = ROOT / "third_party/MoGe"
MOGE_MAP_ROOT = OUT / "moge2_selected_keyframe_maps"

SPECIAL_TOKENS = 6
PATCH_H = 20
PATCH_W = 36
TARGET_H = 280
TARGET_W = 504
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


def run_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        output = getattr(exc, "output", "")
        return f"{type(exc).__name__}: {exc}\n{output}"


def as_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    return float(raw)


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    raw = row.get(key, "")
    if raw == "":
        return default
    return int(float(raw))


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def all_constant(values: list[float]) -> bool:
    vals = [float(x) for x in values if finite(x)]
    if not vals:
        return True
    return max(vals) - min(vals) <= 1e-12


def discover_moge() -> dict[str, Any]:
    candidates = [
        ROOT / "third_party/MoGe-2",
        ROOT / "third_party/moge2",
        ROOT / "third_party/MoGe",
        ROOT / "checkpoints",
        ROOT / "third_party/lingbot-map/checkpoints",
    ]
    found: list[str] = []
    for candidate in candidates:
        if candidate.exists():
            if candidate.is_dir():
                for item in candidate.rglob("*"):
                    if "moge" in item.name.lower():
                        found.append(item.relative_to(ROOT).as_posix())
            elif "moge" in candidate.name.lower():
                found.append(candidate.relative_to(ROOT).as_posix())
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
                "from moge.model.v2 import MoGeModel; "
                "print('moge_v2_import_ok')"
            ),
        ]
    )
    return {
        "candidate_roots": [path.relative_to(ROOT).as_posix() for path in candidates],
        "found_moge_like_paths": sorted(set(found)),
        "third_party_find_output": run_text(["find", "third_party", "-maxdepth", "3", "-iname", "*moge*"]),
        "repo_find_output": run_text(["find", ".", "-maxdepth", "4", "-iname", "*moge*"]),
        "moge_root_exists": MOGE_ROOT.exists(),
        "moge_v2_import_probe": import_probe.strip(),
    }


def trace_rows_by_evidence() -> dict[tuple[str, int, int], list[dict[str, str]]]:
    rows: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(HEAD_TRACE_ROWS):
        if row.get("current_sample_position", "") == "" or row.get("head_idx", "") == "":
            continue
        key = (row["seq"], as_int(row, "current_sample_position"), as_int(row, "head_idx"))
        rows[key].append(row)
    return rows


def rgb_path(seq: str, sample_position: int) -> Path:
    return RGB_WORKSPACE / seq / "gt/rgb" / f"{sample_position:06d}.png"


def lingbot_depth_path(seq: str, sample_position: int) -> Path:
    dataset = f"kitti_v105_seq{seq}_trace32"
    return (
        BASELINE_WORKSPACE
        / dataset
        / seq
        / "lingbot_map_stream_default_stage2_notrace/depth"
        / f"{sample_position:06d}.exr"
    )


def patch_region(patch_index: int, height: int, width: int) -> tuple[slice, slice]:
    row = patch_index // PATCH_W
    col = patch_index % PATCH_W
    y0 = int(round(row * TARGET_H / PATCH_H * height / TARGET_H))
    y1 = int(round((row + 1) * TARGET_H / PATCH_H * height / TARGET_H))
    x0 = int(round(col * TARGET_W / PATCH_W * width / TARGET_W))
    x1 = int(round((col + 1) * TARGET_W / PATCH_W * width / TARGET_W))
    y0, y1 = max(0, min(height, y0)), max(0, min(height, y1))
    x0, x1 = max(0, min(width, x0)), max(0, min(width, x1))
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    return slice(y0, y1), slice(x0, x1)


def resize_like(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(arr.astype(np.float32), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)


def run_moge_maps(unique_frames: set[tuple[str, int]]) -> dict[tuple[str, int], dict[str, Any]]:
    import cv2
    import torch
    import utils3d

    sys.path.insert(0, str(MOGE_ROOT))
    from moge.model.v2 import MoGeModel  # noqa: WPS433

    pretrained = os.environ.get("V106_MOGE_PRETRAINED", DEFAULT_PRETRAINED)
    requested_device = os.environ.get("V106_MOGE_DEVICE", "cuda:0")
    device = torch.device(requested_device if torch.cuda.is_available() or not requested_device.startswith("cuda") else "cpu")
    resize_to = int(os.environ.get("V106_MOGE_RESIZE", "224"))
    num_tokens = int(os.environ.get("V106_MOGE_NUM_TOKENS", "1200"))
    use_fp16 = str(os.environ.get("V106_MOGE_FP16", "1")).lower() not in {"0", "false", "no"}

    MOGE_MAP_ROOT.mkdir(parents=True, exist_ok=True)
    model = MoGeModel.from_pretrained(pretrained).to(device).eval()
    outputs: dict[tuple[str, int], dict[str, Any]] = {}
    for seq, sample_pos in sorted(unique_frames):
        out_dir = MOGE_MAP_ROOT / seq
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / f"{sample_pos:06d}.npz"
        meta_path = out_dir / f"{sample_pos:06d}.json"
        rgb = rgb_path(seq, sample_pos)
        if npz_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            outputs[(seq, sample_pos)] = {
                "npz_path": npz_path,
                "meta_path": meta_path,
                "depth_shape": tuple(meta.get("depth_shape", [])),
                "pretrained": meta.get("pretrained", pretrained),
                "device": meta.get("device", str(device)),
                "resize_to": meta.get("resize_to", resize_to),
                "num_tokens": meta.get("num_tokens", num_tokens),
                "cached": True,
            }
            continue
        image = cv2.cvtColor(cv2.imread(str(rgb)), cv2.COLOR_BGR2RGB)
        if image is None:
            raise FileNotFoundError(f"missing RGB input for MoGe: {rgb}")
        orig_h, orig_w = image.shape[:2]
        height = min(resize_to, int(resize_to * orig_h / orig_w))
        width = min(resize_to, int(resize_to * orig_w / orig_h))
        image = cv2.resize(image, (width, height), cv2.INTER_AREA)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        output = model.infer(image_tensor, num_tokens=num_tokens, use_fp16=use_fp16)
        points = output["points"].detach().cpu().numpy().astype(np.float32)
        depth = output["depth"].detach().cpu().numpy().astype(np.float32)
        mask = output["mask"].detach().cpu().numpy().astype(np.float32)
        intrinsics = output["intrinsics"].detach().cpu().numpy().astype(np.float32)
        fov_x, fov_y = utils3d.np.intrinsics_to_fov(intrinsics)
        np.savez_compressed(npz_path, depth=depth, points=points, mask=mask, intrinsics=intrinsics)
        meta = {
            "schema": "acl2_v106tf_moge2_frame_map_v1",
            "seq_id": seq,
            "sample_position": sample_pos,
            "rgb_path": rgb.relative_to(ROOT).as_posix(),
            "npz_path": npz_path.relative_to(ROOT).as_posix(),
            "pretrained": pretrained,
            "device": str(device),
            "resize_to": resize_to,
            "num_tokens": num_tokens,
            "use_fp16": use_fp16,
            "original_shape_hw": [int(orig_h), int(orig_w)],
            "depth_shape": [int(depth.shape[0]), int(depth.shape[1])],
            "finite_depth_fraction": float(np.isfinite(depth).mean()),
            "mask_fraction": float((mask > 0.5).mean()),
            "fov_x_deg": float(np.rad2deg(fov_x)),
            "fov_y_deg": float(np.rad2deg(fov_y)),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs[(seq, sample_pos)] = {
            "npz_path": npz_path,
            "meta_path": meta_path,
            "depth_shape": depth.shape,
            "pretrained": pretrained,
            "device": str(device),
            "resize_to": resize_to,
            "num_tokens": num_tokens,
            "cached": False,
        }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


def collect_trace_regions(enriched: list[dict[str, str]], traces: dict[tuple[str, int, int], list[dict[str, str]]]) -> set[tuple[str, int]]:
    frames: set[tuple[str, int]] = set()
    for row in enriched:
        key = (row["seq_id"], as_int(row, "frame_id"), as_int(row, "head_id"))
        for trace in traces.get(key, []):
            if trace.get("key_token_role") != "patch":
                continue
            patch_index = as_int(trace, "key_token_offset", -1) - SPECIAL_TOKENS
            if 0 <= patch_index < PATCH_H * PATCH_W:
                frames.add((row["seq_id"], as_int(trace, "key_sample_position")))
    return frames


def summarize_moge_region(
    row: dict[str, str],
    trace_rows: list[dict[str, str]],
    maps: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    import cv2

    log_errors: list[float] = []
    moge_log_depths: list[float] = []
    point_norms: list[float] = []
    mask_vals: list[float] = []
    used_regions = 0
    used_weights: list[float] = []
    map_paths: list[str] = []
    depth_shape = ""

    for trace in trace_rows:
        if trace.get("key_token_role") != "patch":
            continue
        patch_index = as_int(trace, "key_token_offset", -1) - SPECIAL_TOKENS
        if not (0 <= patch_index < PATCH_H * PATCH_W):
            continue
        seq = row["seq_id"]
        key_pos = as_int(trace, "key_sample_position")
        info = maps.get((seq, key_pos))
        if not info:
            continue
        data = np.load(info["npz_path"])
        moge_depth = np.asarray(data["depth"], dtype=np.float32)
        points = np.asarray(data["points"], dtype=np.float32)
        mask = np.asarray(data["mask"], dtype=np.float32)
        depth_shape = f"{moge_depth.shape[0]}x{moge_depth.shape[1]}"
        lingbot_path = lingbot_depth_path(seq, key_pos)
        lingbot_depth = cv2.imread(str(lingbot_path), cv2.IMREAD_UNCHANGED)
        if lingbot_depth is None:
            continue
        lingbot_resized = resize_like(np.asarray(lingbot_depth, dtype=np.float32), moge_depth.shape)
        y_slice, x_slice = patch_region(patch_index, *moge_depth.shape)
        md = moge_depth[y_slice, x_slice]
        ld = lingbot_resized[y_slice, x_slice]
        pm = points[y_slice, x_slice, :]
        mm = mask[y_slice, x_slice]
        valid = np.isfinite(md) & np.isfinite(ld) & (md > 1e-6) & (ld > 1e-6) & (mm > 0.5)
        if not np.any(valid):
            continue
        err = np.abs(np.log(ld[valid]) - np.log(md[valid]))
        log_errors.extend(err.astype(np.float64).tolist())
        moge_log_depths.extend(np.log(md[valid]).astype(np.float64).tolist())
        point_norms.extend(np.linalg.norm(pm[valid], axis=-1).astype(np.float64).tolist())
        mask_vals.extend(mm[np.isfinite(mm)].astype(np.float64).tolist())
        used_regions += 1
        used_weights.append(as_float(trace, "attention_weight", 0.0))
        map_paths.append(info["npz_path"].relative_to(ROOT).as_posix())

    if not log_errors:
        return {
            "moge_region_available": False,
            "moge_region_count": 0,
            "moge_depth": "",
            "moge_pointmap_xyz": "",
            "lingbot_vs_moge_log_depth_error": "",
            "metric_consistency_score": "",
            "depth_spread": "",
            "point_spread": "",
            "moge_confidence_or_mask": "",
            "moge_region_scope": "attention_topk_key_patch_regions",
            "moge_output_shape": depth_shape,
        }

    err_arr = np.asarray(log_errors, dtype=np.float64)
    log_depth_arr = np.asarray(moge_log_depths, dtype=np.float64)
    point_arr = np.asarray(point_norms, dtype=np.float64)
    depth_spread = float(np.quantile(log_depth_arr, 0.75) - np.quantile(log_depth_arr, 0.25))
    point_spread = float(np.quantile(point_arr, 0.75) - np.quantile(point_arr, 0.25))
    median_err = float(np.median(err_arr))
    return {
        "moge_region_available": True,
        "moge_region_count": used_regions,
        "moge_attention_weight_sum": float(sum(used_weights)),
        "moge_depth": ";".join(sorted(set(map_paths))),
        "moge_pointmap_xyz": ";".join(sorted(set(map_paths))),
        "lingbot_vs_moge_log_depth_error": median_err,
        "lingbot_vs_moge_point_error": "",
        "metric_consistency_score": float(math.exp(-median_err)),
        "metric_consistency_source": "moge2_vits_normal_attention_topk_key_patch_region",
        "depth_spread": depth_spread,
        "point_spread": point_spread,
        "moge_confidence_or_mask": float(np.mean(mask_vals)) if mask_vals else "",
        "moge_region_scope": "attention_topk_key_patch_regions",
        "moge_output_shape": depth_shape,
    }


def proxy_rows(enriched: list[dict[str, str]]) -> tuple[list[dict[str, Any]], float]:
    depth_values = [as_float(row, "depth_spread_proxy", 0.0) for row in enriched]
    tau = float(np.median(np.asarray(depth_values, dtype=np.float64))) if depth_values else 1.0
    tau = tau if tau > 1e-9 else 1.0
    rows: list[dict[str, Any]] = []
    for row in enriched:
        baseline = as_float(row, "baseline_L3", 0.0)
        action = as_float(row, "action_L3", baseline)
        rel_change = abs(action - baseline) / max(abs(baseline), 1e-6)
        metric_consistency_proxy = float(math.exp(-rel_change))
        boundary = as_float(row, "boundary_risk", 1.0)
        depth_spread = as_float(row, "depth_spread_proxy", 0.0)
        point_spread = as_float(row, "point_spread_proxy", 0.0)
        scale_observability = metric_consistency_proxy * min(1.0, depth_spread / tau) * max(0.0, 1.0 - boundary)
        semantic_role = row.get("semantic_role", "unknown")
        far_weak = 1.0 if semantic_role in {"vegetation_or_weak_context", "sky_or_lowobs"} else max(0.0, 1.0 - scale_observability)
        rows.append(
            {
                "schema": "acl2_v106tf_stage2_moge_or_proxy_verifier_row_v1",
                "seq_id": row["seq_id"],
                "frame_id": row["frame_id"],
                "head_id": row["head_id"],
                "token_group_id": row["token_group_id"],
                "label_type": row["label_type"],
                "moge_depth": "",
                "moge_pointmap_xyz": "",
                "lingbot_depth": row.get("baseline_depth_exr", ""),
                "lingbot_vs_moge_log_depth_error": "",
                "lingbot_vs_moge_point_error": "",
                "metric_consistency_score": metric_consistency_proxy,
                "metric_consistency_source": "lingbot_self_consistency_proxy",
                "boundary_mismatch_score": boundary,
                "depth_spread": depth_spread,
                "point_spread": point_spread,
                "scale_observability_score": scale_observability,
                "far_background_weakness_score": far_weak,
                "moge_confidence_or_mask": "",
                "proxy_only": True,
                "moge_available": False,
                "moge_coverage": 0.0,
                "moge_region_scope": "",
                "moge_output_shape": "",
                "semantic_role": semantic_role,
                "baseline_L3": row.get("baseline_L3", ""),
                "action_L3": row.get("action_L3", ""),
                "bad_improvement": row.get("bad_improvement", ""),
                "good_harm": row.get("good_harm", ""),
            }
        )
    return rows, tau


def moge_rows(
    enriched: list[dict[str, str]],
    traces: dict[tuple[str, int, int], list[dict[str, str]]],
    maps: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    partial: list[dict[str, Any]] = []
    for row in enriched:
        key = (row["seq_id"], as_int(row, "frame_id"), as_int(row, "head_id"))
        region = summarize_moge_region(row, traces.get(key, []), maps)
        partial.append({**row, **region})
    tau_values = [as_float(row, "depth_spread", 0.0) for row in partial if row.get("moge_region_available")]
    tau = float(np.median(np.asarray(tau_values, dtype=np.float64))) if tau_values else 1.0
    tau = tau if tau > 1e-9 else 1.0
    rows: list[dict[str, Any]] = []
    for row in partial:
        boundary = as_float(row, "boundary_risk", 1.0)
        metric = as_float(row, "metric_consistency_score", 0.0)
        depth_spread = as_float(row, "depth_spread", 0.0)
        semantic_role = row.get("semantic_role", "unknown")
        scale_observability = metric * min(1.0, depth_spread / tau) * max(0.0, 1.0 - boundary)
        far_weak = 1.0 if semantic_role in {"vegetation_or_weak_context", "sky_or_lowobs"} else max(0.0, 1.0 - scale_observability)
        rows.append(
            {
                "schema": "acl2_v106tf_stage2_moge_or_proxy_verifier_row_v2",
                "seq_id": row["seq_id"],
                "frame_id": row["frame_id"],
                "head_id": row["head_id"],
                "token_group_id": row["token_group_id"],
                "label_type": row["label_type"],
                "moge_depth": row.get("moge_depth", ""),
                "moge_pointmap_xyz": row.get("moge_pointmap_xyz", ""),
                "lingbot_depth": row.get("baseline_depth_exr", ""),
                "lingbot_vs_moge_log_depth_error": row.get("lingbot_vs_moge_log_depth_error", ""),
                "lingbot_vs_moge_point_error": row.get("lingbot_vs_moge_point_error", ""),
                "metric_consistency_score": row.get("metric_consistency_score", ""),
                "metric_consistency_source": row.get("metric_consistency_source", ""),
                "boundary_mismatch_score": boundary,
                "depth_spread": depth_spread,
                "point_spread": row.get("point_spread", ""),
                "scale_observability_score": scale_observability,
                "far_background_weakness_score": far_weak,
                "moge_confidence_or_mask": row.get("moge_confidence_or_mask", ""),
                "proxy_only": False,
                "moge_available": True,
                "moge_coverage": 1.0 if row.get("moge_region_available") else 0.0,
                "moge_region_available": row.get("moge_region_available", False),
                "moge_region_count": row.get("moge_region_count", 0),
                "moge_attention_weight_sum": row.get("moge_attention_weight_sum", ""),
                "moge_region_scope": row.get("moge_region_scope", ""),
                "moge_output_shape": row.get("moge_output_shape", ""),
                "moge_model": DEFAULT_PRETRAINED,
                "semantic_role": semantic_role,
                "baseline_L3": row.get("baseline_L3", ""),
                "action_L3": row.get("action_L3", ""),
                "bad_improvement": row.get("bad_improvement", ""),
                "good_harm": row.get("good_harm", ""),
            }
        )
    return rows, tau


def write_common_reports(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    discovery: dict[str, Any],
    tau_name: str,
    tau: float,
) -> None:
    bad_rows = [row for row in rows if row["label_type"] == "bad_selected"]
    good_rows = [row for row in rows if row["label_type"] == "good_selected"]
    audit = f"""# Stage2 MoGe-2 Local Availability Audit

MoGe available: `{str(summary["moge_available"]).lower()}`

Checked candidate roots:
```json
{json.dumps(discovery["candidate_roots"], ensure_ascii=False, indent=2)}
```

MoGe-like paths found:
```json
{json.dumps(discovery["found_moge_like_paths"][:200], ensure_ascii=False, indent=2)}
```

MoGe v2 import probe:
```text
{discovery["moge_v2_import_probe"]}
```

Decision:
- `proxy_only={summary["proxy_only"]}`
- `moge_coverage={summary["moge_coverage"]}`
- `{tau_name}={tau}`
"""
    (OUT / "moge_availability_audit.md").write_text(audit, encoding="utf-8")
    alignment = f"""# Stage2 LingBot / MoGe Alignment Report

Rows: `{len(rows)}`
MoGe coverage: `{summary["moge_coverage"]}`
proxy_only: `{summary["proxy_only"]}`
metric_consistency_not_all_constant: `{summary["metric_consistency_not_all_constant"]}`
boundary_mismatch_not_all_zero: `{summary["boundary_mismatch_not_all_zero"]}`
scale_observability_not_all_constant: `{summary["scale_observability_not_all_constant"]}`

Metric provenance:
- `metric_consistency_source` is recorded per row.
- For real MoGe rows, LingBot baseline depth is resized to the MoGe output map shape and compared on traced attention top-k key patch regions.
- `lingbot_vs_moge_point_error` is left blank because current LingBot artifacts expose depth maps, not point maps.
"""
    (OUT / "moge_lingbot_alignment_report.md").write_text(alignment, encoding="utf-8")

    def panel(label: str, subset: list[dict[str, Any]]) -> str:
        if not subset:
            return f"## {label}\n\nNo rows.\n"
        metric = np.asarray([as_float(row, "metric_consistency_score") for row in subset], dtype=np.float64)
        obs = np.asarray([as_float(row, "scale_observability_score") for row in subset], dtype=np.float64)
        boundary = np.asarray([as_float(row, "boundary_mismatch_score") for row in subset], dtype=np.float64)
        return (
            f"## {label}\n\n"
            f"- rows: `{len(subset)}`\n"
            f"- metric_consistency_median: `{float(np.nanmedian(metric))}`\n"
            f"- scale_observability_median: `{float(np.nanmedian(obs))}`\n"
            f"- boundary_mismatch_median: `{float(np.nanmedian(boundary))}`\n"
        )

    panels = "# Stage2 Metric Consistency Panels\n\n" + panel("bad_selected", bad_rows) + "\n" + panel("good_selected", good_rows)
    (OUT / "metric_consistency_panels.md").write_text(panels, encoding="utf-8")


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    enriched = read_csv(STAGE1 / "selected_evidence_enriched_rows.csv")
    discovery = discover_moge()
    traces = trace_rows_by_evidence()
    use_moge = MOGE_ROOT.exists() and "moge_v2_import_ok" in discovery.get("moge_v2_import_probe", "")

    mode = "proxy"
    note = ""
    if use_moge:
        try:
            unique_frames = collect_trace_regions(enriched, traces)
            maps = run_moge_maps(unique_frames)
            rows, tau = moge_rows(enriched, traces, maps)
            mode = "moge2"
            note = (
                "MoGe-2 VITS-normal ran on selected evidence key-frame patch regions. "
                "No LingBot pointmap exists, so point error remains blank."
            )
        except Exception as exc:  # noqa: BLE001
            rows, tau = proxy_rows(enriched)
            note = f"MoGe-2 attempt failed and proxy fallback was used: {exc!r}"
    else:
        rows, tau = proxy_rows(enriched)
        note = "MoGe-2 was not locally importable; rows use LingBot self-consistency proxy only."

    metric_vals = [as_float(row, "metric_consistency_score") for row in rows]
    boundary_vals = [as_float(row, "boundary_mismatch_score") for row in rows]
    obs_vals = [as_float(row, "scale_observability_score") for row in rows]
    bad_rows = [row for row in rows if row["label_type"] == "bad_selected"]
    good_rows = [row for row in rows if row["label_type"] == "good_selected"]
    moge_covered_rows = [row for row in rows if as_float(row, "moge_coverage", 0.0) >= 1.0]
    coverage = len(moge_covered_rows) / len(rows) if rows else 0.0
    real_moge = mode == "moge2" and coverage >= 0.80
    summary = {
        "schema": "acl2_v106tf_stage2_moge_metric_verifier_summary_v2",
        "moge_available": real_moge,
        "moge_coverage": coverage,
        "proxy_only": not real_moge,
        "moge_proxy_or_missing": not real_moge,
        "moge_mode": mode,
        "moge_pretrained": os.environ.get("V106_MOGE_PRETRAINED", DEFAULT_PRETRAINED),
        "moge_resize": int(os.environ.get("V106_MOGE_RESIZE", "224")),
        "moge_num_tokens": int(os.environ.get("V106_MOGE_NUM_TOKENS", "1200")),
        "moge_region_scope": "attention_topk_key_patch_regions" if real_moge else "",
        "selected_evidence_rows": len(rows),
        "proxy_verifier_coverage": 1.0 if rows else 0.0,
        "selected_bad_rows": len(bad_rows),
        "selected_good_rows": len(good_rows),
        "selected_bad_good_have_comparable_proxy_coverage": bool(bad_rows and good_rows),
        "selected_bad_good_have_comparable_moge_coverage": bool(
            all(as_float(row, "moge_coverage", 0.0) >= 1.0 for row in bad_rows + good_rows)
        ) if rows else False,
        "metric_consistency_not_all_constant": not all_constant(metric_vals),
        "boundary_mismatch_not_all_zero": any(abs(x) > 1e-12 for x in boundary_vals if finite(x)),
        "scale_observability_not_all_constant": not all_constant(obs_vals),
        "tau_depth_moge_or_proxy_median": tau,
        "stage2_moge_pass": (
            real_moge
            and not all_constant(metric_vals)
            and any(abs(x) > 1e-12 for x in boundary_vals if finite(x))
            and not all_constant(obs_vals)
            and bool(bad_rows and good_rows)
        ),
        "stage2_proxy_ready_for_stage3_diagnostic": bool(rows),
        "stage4_moge_based_action_promotion_allowed": False,
        "outputs": {
            "moge_verifier_rows": (OUT / "moge_verifier_rows.csv").relative_to(ROOT).as_posix(),
            "moge_availability_audit": (OUT / "moge_availability_audit.md").relative_to(ROOT).as_posix(),
            "moge_lingbot_alignment_report": (OUT / "moge_lingbot_alignment_report.md").relative_to(ROOT).as_posix(),
            "metric_consistency_panels": (OUT / "metric_consistency_panels.md").relative_to(ROOT).as_posix(),
            "stage2_summary": (OUT / "stage2_summary.json").relative_to(ROOT).as_posix(),
            "moge_map_root": MOGE_MAP_ROOT.relative_to(ROOT).as_posix(),
        },
        "discovery": discovery,
        "note": note,
    }
    if summary["stage2_moge_pass"]:
        summary["stage4_moge_based_action_promotion_allowed"] = True

    write_csv(OUT / "moge_verifier_rows.csv", rows)
    (OUT / "stage2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_common_reports(
        rows=rows,
        summary=summary,
        discovery=discovery,
        tau_name="tau_depth_moge_or_proxy_median",
        tau=tau,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
