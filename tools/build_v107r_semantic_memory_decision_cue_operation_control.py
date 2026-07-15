#!/usr/bin/env python3
"""Build ACL2 v107R semantic memory decision cue artifacts.

This is a diagnostic builder. It does not change LingBot runtime behavior and
does not use MoGe/LingBot-Depth/GT as runtime cues.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V107TF = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention"
V107R = ROOT / "results/acl2_v107r_lingbot_semantic_memory_decision_cue_operation_control"

V107TF_STAGE1 = V107TF / "stage1_cache_operation_instrumentation"
V107TF_STAGE3 = V107TF / "stage3_operation_discovery"
V107TF_OP_ROWS = V107TF_STAGE3 / "operation_discovery_rows.csv"
V107TF_TARGETS = V107TF_STAGE1 / "target_manifest.csv"
V107TF_LC_TARGETS = V107TF_STAGE3 / "length_control_safe96/target_manifest.csv"

STAGE0 = V107R / "stage0_v107r_evidence_freeze"
STAGE1 = V107R / "stage1_semantic_cue_bank"
STAGE2 = V107R / "stage2_operation_cue_join"
STAGE3 = V107R / "stage3_operation_cue_matrix"
STAGE4 = V107R / "stage4_memory_role_disambiguation"
STAGE5 = V107R / "stage5_action_surface_selection"
STAGE6 = V107R / "stage6_runtime_pilot_or_blocked"
STAGE7 = V107R / "stage7_full_validation_or_blocked"
FINAL = V107R / "final_decision"

SEM_ROOT = ROOT / "results/kitti_preprocess"
CHUNK_RE = re.compile(r"_(\d{6})_(\d{6})$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        raw = row.get(key, "")
        if raw in {"", None}:
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def inum(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = fnum(row, key, float(default))
    return int(value) if math.isfinite(value) else default


def bval(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).lower() in {"true", "1", "yes"}


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def quantile(values: list[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    xv = [p[0] for p in pairs]
    yv = [p[1] for p in pairs]
    mx = sum(xv) / len(xv)
    my = sum(yv) / len(yv)
    vx = sum((x - mx) ** 2 for x in xv)
    vy = sum((y - my) ** 2 for y in yv)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def load_length_matched_operation_rows() -> list[dict[str, str]]:
    rows = read_csv(V107TF_OP_ROWS)
    out = [row for row in rows if row.get("universe") == "length_matched_96f"]
    if not out:
        out = rows
    return out


def load_targets() -> dict[str, dict[str, str]]:
    targets: dict[str, dict[str, str]] = {}
    for path in [V107TF_TARGETS, V107TF_LC_TARGETS]:
        if not path.is_file():
            continue
        for row in read_csv(path):
            targets[row["target_id"]] = row
    return targets


def stage0() -> dict[str, Any]:
    STAGE0.mkdir(parents=True, exist_ok=True)
    v107tf_s0 = load_json(V107TF / "stage0_evidence_freeze/stage0_summary.json")
    v107tf_s1 = load_json(V107TF_STAGE1 / "operation_trace_summary.json")
    v107tf_s2 = load_json(V107TF / "stage2_metric_reliability_verifier/verifier_coverage_summary.json")
    v107tf_s3 = load_json(V107TF_STAGE3 / "stage3_summary.json")
    v105_v106r = load_json(V107TF / "stage0_evidence_freeze/v105_v106r_known_facts.json")

    op_rows_exists = V107TF_OP_ROWS.is_file()
    op_rows = load_length_matched_operation_rows() if op_rows_exists else []
    operation_types = sorted({row.get("operation_type", "") for row in op_rows if row.get("operation_type")})
    required_non_readout_seen = sorted(set(operation_types) & {"initialization", "cache_append", "retention", "eviction", "budget_keep", "budget_drop", "special_token_update", "trajectory_write"})
    semantic_paths = [
        SEM_ROOT / seq / "stage_c_cache_semantic_chunks"
        for seq in ["00", "01", "02", "05"]
    ]
    semantic_cache_available = {seq: path.is_dir() for seq, path in zip(["00", "01", "02", "05"], semantic_paths)}
    known_facts = {
        "schema": "acl2_v107r_known_facts_v1",
        "v107tf_stage0_pass": v107tf_s0.get("stage0_pass", ""),
        "v107tf_stage1_trace_parity_pass": v107tf_s1.get("trace_parity_pass", ""),
        "v107tf_stage1_operation_row_count": v107tf_s1.get("operation_row_count", ""),
        "v107tf_stage1_required_gate_operation_types_seen": v107tf_s1.get("required_gate_operation_types_seen", []),
        "v107tf_stage2_pass": v107tf_s2.get("stage2_pass", ""),
        "v107tf_stage2_verifier_coverage": v107tf_s2.get("verifier_coverage", ""),
        "v107tf_stage3_pass": v107tf_s3.get("stage3_pass", ""),
        "v107tf_final_taxonomy_if_stop_here": v107tf_s3.get("final_taxonomy_if_stop_here", ""),
        "v107tf_length_matched_case_count": (v107tf_s3.get("length_matched_96f") or {}).get("case_count", ""),
        "v107tf_length_matched_operation_row_count": len(op_rows),
        "v107tf_semantic_available_in_stage3": v107tf_s3.get("semantic_available", ""),
        "v105_full_kitti_baseline_by_seq": v105_v106r.get("v105_full_kitti_baseline_by_seq", {}),
        "v105_headlocal_relaxed_good_median_harm": v105_v106r.get("v105_headlocal_relaxed_good_median_harm", ""),
        "v106r_taxonomy": v105_v106r.get("v106r_taxonomy", ""),
        "v106r_targeted_operation_types_present": v105_v106r.get("v106r_targeted_operation_types_present", []),
        "v106r_targeted_missing_operation_types": v105_v106r.get("v106r_targeted_missing_operation_types", []),
        "v107r_semantic_cache_available_by_seq": semantic_cache_available,
        "v107r_runtime_action_allowed": False,
        "v107r_runtime_cue_forbidden": ["MoGe-2", "LingBot-Depth", "GT pose/depth", "SLAM", "post-hoc Sim3"],
        "v107r_operation_rows_source": rel(V107TF_OP_ROWS),
        "v107r_operation_types_from_v107tf_length_matched": operation_types,
        "v107r_required_non_readout_seen": required_non_readout_seen,
    }
    missing = []
    if not op_rows:
        missing.append(rel(V107TF_OP_ROWS))
    for seq, available in semantic_cache_available.items():
        if not available:
            missing.append(rel(SEM_ROOT / seq / "stage_c_cache_semantic_chunks"))
    known_facts["missing_required_artifacts"] = missing
    known_facts["stage0_pass"] = (not missing) and bool(v107tf_s1.get("trace_parity_pass")) and bool(required_non_readout_seen)

    write_json(STAGE0 / "v107r_known_facts.json", known_facts)
    write_json(STAGE0 / "stage0_summary.json", {
        "schema": "acl2_v107r_stage0_summary_v1",
        "stage0_pass": known_facts["stage0_pass"],
        "missing_required_artifacts": missing,
        "runtime_action_allowed": False,
        "operation_rows_source": rel(V107TF_OP_ROWS),
    })

    forbidden = """# v107R Forbidden Repeat List

These paths are not allowed as v107R method claims. They may appear only as
negative controls, guardrails, or offline evaluation evidence.

- frame-level semantic_geometry_write_filter
- frame-level context-only demotion
- semantic-only reject filter
- headlocal relaxed context-only demotion as direct method
- readout attention mass threshold action
- head-selected-count threshold action
- semantic-label-only action
- external depth / MoGe / LingBot-Depth verifier as runtime cue
- post-hoc Sim(3) / SLAM / pose graph correction
- LoGeR provider expansion / query-soft resurrection

Carry-forward evidence:

- v105 headlocal relaxed action moved bad L3 but caused severe good harm.
- v106R targeted trace observed readout only and did not find a memory lever.
- v107TF observed non-readout cache operations, but operation discovery did not pass same-count controls.
"""
    write_text(STAGE0 / "forbidden_repeat_list.md", forbidden)

    allowed_ops = [
        ("readout", "diagnostic_only_until_stage5", "local window/context read path"),
        ("initialization", "allowed_after_stage3_stage4_pass", "anchor/scale-frame initialization"),
        ("append_or_write", "allowed_after_stage3_stage4_pass", "cache append or trajectory write admission"),
        ("update", "allowed_after_stage3_stage4_pass", "semantic-aware update admission if exposed"),
        ("retention", "allowed_after_stage3_stage4_pass", "cache/memory retention policy"),
        ("eviction", "allowed_after_stage3_stage4_pass", "cache/memory eviction policy"),
        ("budget_keep_drop", "allowed_after_stage3_stage4_pass", "fixed-budget keep/drop policy"),
        ("local_reference_separation", "allowed_after_stage3_stage4_pass", "preserve local readout while blocking reference update"),
        ("special_token_lifecycle", "allowed_after_stage3_stage4_pass", "camera/register/scale/trajectory special-token path"),
    ]
    write_csv(
        STAGE0 / "allowed_lingbot_memory_operations.csv",
        [
            {
                "schema": "acl2_v107r_allowed_lingbot_memory_operation_v1",
                "operation_type": op,
                "stage0_status": status,
                "notes": notes,
                "runtime_action_allowed_at_stage0": False,
            }
            for op, status, notes in allowed_ops
        ],
    )
    no_external = """# Why No External Depth Or Postprocessing

v107R runtime cues are restricted to semantic cache features and LingBot internal
state. MoGe-2, LingBot-Depth, GT poses/depth, SLAM, pose graphs, and post-hoc
Sim(3) are forbidden as runtime inputs. They may be used only as offline
evaluation or as already frozen historical verifier evidence.

This builder therefore does not consume v107TF MoGe verifier columns as cue
features in Stage3. L3/L4 values are used only as labels/evaluation targets.
"""
    write_text(STAGE0 / "why_no_external_depth_or_postprocessing.md", no_external)
    if missing:
        write_text(STAGE0 / "stage0_missing_artifacts_report.md", "\n".join(["# Missing Artifacts", "", *[f"- {m}" for m in missing]]))
    return known_facts


def parse_grid_from_trace(op_rows: list[dict[str, str]]) -> dict[str, Any]:
    image_rows = [row for row in op_rows if row.get("token_type") == "image_patch"]
    patch_count_candidates = sorted({inum(row, "token_count") for row in image_rows if inum(row, "token_count") > 0})
    min_patch_count = min((c for c in patch_count_candidates if c <= 2000), default=720)
    patch_start_idx = min((inum(row, "token_index") for row in image_rows if inum(row, "token_index") >= 0), default=6)
    patch_size = 14
    target_w = 504
    target_h = 280
    ds_cfg = next((V107TF / "configs/datasets").glob("*.yaml"), None)
    if ds_cfg is not None:
        text = ds_cfg.read_text(encoding="utf-8")
        m = re.search(r"_target_size:\s*\[(\d+),\s*(\d+)\]", text)
        if m:
            target_w, target_h = int(m.group(1)), int(m.group(2))
    meth_cfg = V107TF / "configs/methods/lingbot_map_v107tf_stage1_operation_trace.yaml"
    if meth_cfg.is_file():
        text = meth_cfg.read_text(encoding="utf-8")
        m = re.search(r"_patch_size:\s*(\d+)", text)
        if m:
            patch_size = int(m.group(1))
    patch_grid_h = target_h // patch_size
    patch_grid_w = target_w // patch_size
    inferred_patch_count = patch_grid_h * patch_grid_w
    return {
        "target_width": target_w,
        "target_height": target_h,
        "patch_size": patch_size,
        "patch_grid_h": patch_grid_h,
        "patch_grid_w": patch_grid_w,
        "image_patch_token_count_from_trace_min": min_patch_count,
        "image_patch_token_count_candidates_from_trace": patch_count_candidates,
        "patch_start_idx": patch_start_idx,
        "special_token_count": patch_start_idx,
        "token_alignment_pass": inferred_patch_count == min_patch_count and patch_start_idx == 6,
        "inferred_patch_count": inferred_patch_count,
        "evidence": [
            rel(ds_cfg) if ds_cfg is not None else "dataset_config_missing",
            rel(meth_cfg) if meth_cfg.is_file() else "method_config_missing",
            rel(V107TF_OP_ROWS),
            "third_party/lingbot-map/lingbot_map/aggregator/stream.py:self.patch_start_idx=1+num_register_tokens+1",
            "third_party/lingbot-map/benchmark/datasets/kitti.py:_cover_fit_center_crop",
        ],
    }


def write_token_grid(grid: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    special_names = ["camera_pose_token", "register_token_0", "register_token_1", "register_token_2", "register_token_3", "scale_frame_token"]
    for token_id, name in enumerate(special_names[: grid["special_token_count"]]):
        rows.append(
            {
                "schema": "acl2_v107r_token_grid_row_v1",
                "token_id": token_id,
                "token_type": name if not name.startswith("register") else "register_token",
                "patch_y": "",
                "patch_x": "",
                "is_special_token": True,
                "has_direct_semantic_patch": False,
            }
        )
    for py in range(grid["patch_grid_h"]):
        for px in range(grid["patch_grid_w"]):
            patch_idx = py * grid["patch_grid_w"] + px
            rows.append(
                {
                    "schema": "acl2_v107r_token_grid_row_v1",
                    "token_id": grid["patch_start_idx"] + patch_idx,
                    "token_type": "image_patch",
                    "patch_y": py,
                    "patch_x": px,
                    "is_special_token": False,
                    "has_direct_semantic_patch": True,
                }
            )
    write_csv(STAGE1 / "token_grid_rows.csv", rows)


def role_for_label(label_name: str, confidence: float, purity: float, void_name: str = "void") -> tuple[str, str]:
    name = (label_name or "").lower()
    if not name or name == void_name or "void" in name or confidence < 0.45:
        return "unknown_lowtrust", "unknown"
    if purity < 0.70:
        return "semantic_boundary", "mixed"
    dynamic = ["car", "person", "rider", "cyclist", "bicycle", "motorcycle", "bus", "truck", "vehicle", "train"]
    stable = ["building", "house", "wall", "fence", "handrail", "pole", "sign", "traffic", "pillar", "bridge", "guardrail", "bench"]
    layout = ["curb", "road edge", "crosswalk", "lane", "roadblock"]
    ground = ["road", "ground", "sidewalk", "terrain", "floor"]
    vegetation = ["tree", "grass", "vegetation", "flower"]
    sky = ["sky", "background"]
    if any(x in name for x in dynamic):
        return "dynamic_transient", "thing"
    if any(x in name for x in stable):
        return "stable_structure", "stuff"
    if any(x in name for x in layout):
        return "road_boundary_or_layout", "stuff"
    if any(x in name for x in ground):
        return "ground_or_road_weak", "stuff"
    if any(x in name for x in vegetation):
        return "vegetation_weak_context", "stuff"
    if any(x in name for x in sky):
        return "sky_or_lowobs", "stuff"
    return "unknown_lowtrust", "unknown"


def build_chunk_index(seq: str) -> list[tuple[int, int, Path]]:
    chunk_root = SEM_ROOT / seq / "stage_c_cache_semantic_chunks"
    chunks: list[tuple[int, int, Path]] = []
    for path in sorted(chunk_root.glob("chunk_*/masklet.pt")):
        match = CHUNK_RE.search(path.parent.name)
        if not match:
            continue
        chunks.append((int(match.group(1)), int(match.group(2)), path))
    return chunks


def find_chunk(chunks: list[tuple[int, int, Path]], frame_id: int) -> tuple[int, int, Path] | None:
    for start, end, path in chunks:
        if start <= frame_id < end:
            return start, end, path
    return None


def cover_fit_resize_2d(tensor: Any, target_h: int, target_w: int, mode: str) -> Any:
    import torch.nn.functional as F

    h, w = int(tensor.shape[-2]), int(tensor.shape[-1])
    scale = max(target_w / w, target_h / h)
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))
    x0 = (resized_w - target_w) // 2
    y0 = (resized_h - target_h) // 2
    x = tensor.float().unsqueeze(0).unsqueeze(0)
    if mode == "nearest":
        out = F.interpolate(x, size=(resized_h, resized_w), mode="nearest")
    else:
        out = F.interpolate(x, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    return out[0, 0, y0 : y0 + target_h, x0 : x0 + target_w]


def cover_fit_resize_mask_stack(mask_stack: Any, target_h: int, target_w: int) -> Any:
    import torch.nn.functional as F

    if mask_stack.numel() == 0:
        return mask_stack
    h, w = int(mask_stack.shape[-2]), int(mask_stack.shape[-1])
    scale = max(target_w / w, target_h / h)
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))
    x0 = (resized_w - target_w) // 2
    y0 = (resized_h - target_h) // 2
    x = mask_stack.float().unsqueeze(1)
    out = F.interpolate(x, size=(resized_h, resized_w), mode="nearest")
    return out[:, 0, y0 : y0 + target_h, x0 : x0 + target_w]


def patchify_projected_frame(label_frame: Any, conf_frame: Any, grid: dict[str, Any], label_names: list[str], void_id: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    patch_h = grid["patch_grid_h"]
    patch_w = grid["patch_grid_w"]
    patch = grid["patch_size"]
    labels = label_frame.long().view(patch_h, patch, patch_w, patch).permute(0, 2, 1, 3).reshape(patch_h * patch_w, patch * patch)
    conf = conf_frame.float().view(patch_h, patch, patch_w, patch).permute(0, 2, 1, 3).reshape(patch_h * patch_w, patch * patch)
    num_classes = max(int(labels.max().item()) + 1, len(label_names), void_id + 1)
    one_hot = torch.nn.functional.one_hot(labels.clamp(min=0, max=num_classes - 1), num_classes=num_classes)
    counts = one_hot.sum(dim=1)
    dominant = counts.argmax(dim=1)
    dominant_counts = counts.max(dim=1).values.float()
    purity = dominant_counts / float(patch * patch)
    conf_mean = conf.mean(dim=1)
    rows: list[dict[str, Any]] = []
    role_counter: Counter[str] = Counter()
    role_trust: Counter[str] = Counter()
    nonvoid = 0
    trusts: list[float] = []
    purities: list[float] = []
    confidences: list[float] = []
    for patch_idx in range(patch_h * patch_w):
        label_id = int(dominant[patch_idx].item())
        label_name = label_names[label_id] if 0 <= label_id < len(label_names) else f"label_{label_id}"
        cval = float(conf_mean[patch_idx].item())
        pval = float(purity[patch_idx].item())
        trust = cval * (pval ** 2)
        role, thing_or_stuff = role_for_label(label_name, cval, pval)
        py, px = divmod(patch_idx, patch_w)
        role_counter[role] += 1
        role_trust[role] += trust
        nonvoid += int(label_id != void_id)
        trusts.append(trust)
        purities.append(pval)
        confidences.append(cval)
        rows.append(
            {
                "patch_index": patch_idx,
                "token_id": grid["patch_start_idx"] + patch_idx,
                "patch_y": py,
                "patch_x": px,
                "dominant_label": label_id,
                "label_name": label_name,
                "semantic_confidence": cval,
                "patch_purity": pval,
                "semantic_trust": trust,
                "semantic_boundary_risk": 1.0 - pval,
                "semantic_role": role,
                "thing_or_stuff": thing_or_stuff,
            }
        )
    denom = float(patch_h * patch_w)
    summary = {
        "patch_count": int(denom),
        "semantic_patch_nonvoid_ratio": nonvoid / denom if denom else 0.0,
        "semantic_patch_purity_mean": mean(purities),
        "semantic_patch_purity_p10": quantile(purities, 0.10),
        "semantic_confidence_mean": mean(confidences),
        "semantic_trust_mean": mean(trusts),
    }
    for role in [
        "stable_structure",
        "road_boundary_or_layout",
        "ground_or_road_weak",
        "dynamic_transient",
        "vegetation_weak_context",
        "sky_or_lowobs",
        "semantic_boundary",
        "unknown_lowtrust",
    ]:
        summary[f"{role}_patch_frac"] = role_counter[role] / denom if denom else 0.0
        summary[f"{role}_mass"] = role_trust[role] / denom if denom else 0.0
    summary["frame_semantic_update_value"] = (
        summary["stable_structure_mass"]
        + summary["road_boundary_or_layout_mass"]
        - summary["dynamic_transient_mass"]
        - summary["semantic_boundary_mass"]
        - summary["unknown_lowtrust_mass"]
        - summary["sky_or_lowobs_mass"]
    )
    return rows, summary


def patch_identity_rows(mask_frame: Any, grid: dict[str, Any], seed_ids: list[Any], component_prefix: str) -> dict[int, dict[str, Any]]:
    import torch

    if mask_frame is None or mask_frame.numel() == 0:
        return {}
    patch_h = grid["patch_grid_h"]
    patch_w = grid["patch_grid_w"]
    patch = grid["patch_size"]
    masks = mask_frame.float().view(mask_frame.shape[0], patch_h, patch, patch_w, patch)
    counts = masks.sum(dim=(2, 4))
    max_counts, max_idx = counts.max(dim=0)
    out: dict[int, dict[str, Any]] = {}
    for py in range(patch_h):
        for px in range(patch_w):
            patch_idx = py * patch_w + px
            area = float(max_counts[py, px].item())
            if area < max(4.0, 0.05 * patch * patch):
                continue
            local_idx = int(max_idx[py, px].item())
            seed = seed_ids[local_idx] if local_idx < len(seed_ids) else ""
            out[patch_idx] = {
                "masklet_id": f"{component_prefix}:m{local_idx}",
                "component_id": f"{component_prefix}:m{local_idx}",
                "seed_global_track_idx": seed,
                "masklet_patch_coverage": area / float(patch * patch),
            }
    return out


def semantic_cache_stage1(op_rows: list[dict[str, str]], grid: dict[str, Any]) -> dict[str, Any]:
    import torch

    STAGE1.mkdir(parents=True, exist_ok=True)
    write_token_grid(grid)

    target_frames_by_seq: dict[str, set[int]] = defaultdict(set)
    target_frame_targets: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in op_rows:
        seq = row.get("seq") or row.get("sequence_id")
        if not seq:
            continue
        current = inum(row, "frame_id", -1)
        if current >= 0:
            target_frames_by_seq[seq].add(current)
            target_frame_targets[(seq, current)].add(row.get("target_id", ""))
        trace_start = inum(row, "trace_start_idx", -1)
        span_start = inum(row, "frame_span_start", -1)
        span_end = inum(row, "frame_span_end", span_start)
        if trace_start >= 0 and span_start >= 0:
            for fid in range(trace_start + span_start, trace_start + span_end + 1):
                target_frames_by_seq[seq].add(fid)
                target_frame_targets[(seq, fid)].add(row.get("target_id", ""))

    token_fields = [
        "schema",
        "seq",
        "frame_id",
        "target_ids",
        "token_id",
        "patch_y",
        "patch_x",
        "dominant_label",
        "label_name",
        "semantic_confidence",
        "patch_purity",
        "semantic_trust",
        "semantic_boundary_risk",
        "semantic_role",
        "thing_or_stuff",
        "masklet_id",
        "component_id",
        "seed_global_track_idx",
        "masklet_patch_coverage",
        "semantic_class_fallback_only",
        "runtime_available",
    ]
    frame_summary_rows: list[dict[str, Any]] = []
    role_mapping_rows: dict[str, dict[str, Any]] = {}
    continuity: dict[tuple[str, str], dict[str, Any]] = {}
    missing_frames: list[dict[str, Any]] = []
    total_target_frames = sum(len(v) for v in target_frames_by_seq.values())
    processed_frames = 0
    total_patch_rows = 0
    total_nonvoid_patches = 0
    total_identity_patches = 0
    purity_values: list[float] = []

    with (STAGE1 / "token_semantic_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=token_fields)
        writer.writeheader()
        for seq in sorted(target_frames_by_seq):
            frames = sorted(target_frames_by_seq[seq])
            chunks = build_chunk_index(seq)
            chunk_to_frames: dict[Path, list[int]] = defaultdict(list)
            for frame_id in frames:
                chunk = find_chunk(chunks, frame_id)
                if chunk is None:
                    missing_frames.append({"seq": seq, "frame_id": frame_id, "reason": "no_stage_c_chunk_covering_frame"})
                    continue
                start, _end, path = chunk
                chunk_to_frames[path].append(frame_id)

            for chunk_path, chunk_frames in sorted(chunk_to_frames.items(), key=lambda kv: kv[0].as_posix()):
                payload = torch.load(chunk_path, map_location="cpu", weights_only=False)
                sem = payload.get("semantic_segmentation", {})
                label_maps = sem.get("label_maps")
                conf_maps = sem.get("confidence_maps")
                label_names = list(sem.get("label_names", []))
                label_to_id = sem.get("label_to_id", {}) or {}
                global_start = int(sem.get("global_start_frame", CHUNK_RE.search(chunk_path.parent.name).group(1)))
                void_id = int(label_to_id.get("void", 0))
                seed_ids = payload.get("seed_global_track_idx", []) or []
                m_mask = payload.get("M_mask")
                if label_maps is None or conf_maps is None:
                    for frame_id in chunk_frames:
                        missing_frames.append({"seq": seq, "frame_id": frame_id, "reason": "chunk_missing_label_or_confidence_maps"})
                    continue

                for frame_id in chunk_frames:
                    local_idx = frame_id - global_start
                    if local_idx < 0 or local_idx >= int(label_maps.shape[0]):
                        missing_frames.append({"seq": seq, "frame_id": frame_id, "reason": "frame_not_in_loaded_chunk"})
                        continue
                    label_proj = cover_fit_resize_2d(label_maps[local_idx], grid["target_height"], grid["target_width"], "nearest").long()
                    conf_proj = cover_fit_resize_2d(conf_maps[local_idx], grid["target_height"], grid["target_width"], "bilinear").float()
                    patch_rows, summary = patchify_projected_frame(label_proj, conf_proj, grid, label_names, void_id)
                    identity: dict[int, dict[str, Any]] = {}
                    if m_mask is not None and m_mask.numel() > 0 and local_idx < int(m_mask.shape[1]):
                        mask_proj = cover_fit_resize_mask_stack(m_mask[:, local_idx], grid["target_height"], grid["target_width"])
                        identity = patch_identity_rows(mask_proj, grid, seed_ids, f"{seq}:{chunk_path.parent.name}:f{frame_id}")
                    target_ids = sorted(t for t in target_frame_targets.get((seq, frame_id), set()) if t)
                    identity_count = 0
                    nonvoid_count = 0
                    for prow in patch_rows:
                        ident = identity.get(int(prow["patch_index"]), {})
                        identity_count += int(bool(ident))
                        nonvoid_count += int(int(prow["dominant_label"]) != void_id)
                        purity_values.append(float(prow["patch_purity"]))
                        seed = str(ident.get("seed_global_track_idx", ""))
                        if seed:
                            key = (seq, seed)
                            entry = continuity.setdefault(
                                key,
                                {
                                    "schema": "acl2_v107r_semantic_continuity_row_v1",
                                    "seq": seq,
                                    "seed_global_track_idx": seed,
                                    "first_frame": frame_id,
                                    "last_frame": frame_id,
                                    "observation_frame_count": 0,
                                    "token_observation_count": 0,
                                    "semantic_roles": Counter(),
                                    "label_names": Counter(),
                                },
                            )
                            entry["first_frame"] = min(int(entry["first_frame"]), frame_id)
                            entry["last_frame"] = max(int(entry["last_frame"]), frame_id)
                            entry.setdefault("_frames", set()).add(frame_id)
                            entry["token_observation_count"] += 1
                            entry["semantic_roles"][str(prow["semantic_role"])] += 1
                            entry["label_names"][str(prow["label_name"])] += 1
                        role_mapping_rows.setdefault(
                            str(prow["label_name"]),
                            {
                                "schema": "acl2_v107r_semantic_role_mapping_row_v1",
                                "label_name": prow["label_name"],
                                "semantic_role": prow["semantic_role"],
                                "thing_or_stuff": prow["thing_or_stuff"],
                                "mapping_rule": "label_name_keyword_plus_confidence_purity_guard",
                            },
                        )
                        writer.writerow(
                            {
                                "schema": "acl2_v107r_token_semantic_row_v1",
                                "seq": seq,
                                "frame_id": frame_id,
                                "target_ids": ";".join(target_ids),
                                "token_id": prow["token_id"],
                                "patch_y": prow["patch_y"],
                                "patch_x": prow["patch_x"],
                                "dominant_label": prow["dominant_label"],
                                "label_name": prow["label_name"],
                                "semantic_confidence": prow["semantic_confidence"],
                                "patch_purity": prow["patch_purity"],
                                "semantic_trust": prow["semantic_trust"],
                                "semantic_boundary_risk": prow["semantic_boundary_risk"],
                                "semantic_role": prow["semantic_role"],
                                "thing_or_stuff": prow["thing_or_stuff"],
                                "masklet_id": ident.get("masklet_id", ""),
                                "component_id": ident.get("component_id", ""),
                                "seed_global_track_idx": ident.get("seed_global_track_idx", ""),
                                "masklet_patch_coverage": ident.get("masklet_patch_coverage", ""),
                                "semantic_class_fallback_only": not bool(ident),
                                "runtime_available": True,
                            }
                        )
                    total_patch_rows += len(patch_rows)
                    total_nonvoid_patches += nonvoid_count
                    total_identity_patches += identity_count
                    processed_frames += 1
                    frame_summary_rows.append(
                        {
                            "schema": "acl2_v107r_frame_semantic_summary_row_v1",
                            "seq": seq,
                            "frame_id": frame_id,
                            "target_ids": ";".join(target_ids),
                            "source_chunk": rel(chunk_path),
                            "runtime_available": True,
                            "confidence_maps_available": True,
                            "semantic_projection_available": True,
                            "semantic_class_fallback_only_frame_frac": 1.0 - (identity_count / len(patch_rows) if patch_rows else 0.0),
                            "masklet_identity_patch_coverage": identity_count / len(patch_rows) if patch_rows else 0.0,
                            **summary,
                        }
                    )
                del payload

    for entry in continuity.values():
        frames = entry.pop("_frames", set())
        entry["observation_frame_count"] = len(frames)
        entry["semantic_roles"] = ";".join(f"{k}:{v}" for k, v in entry["semantic_roles"].most_common())
        entry["label_names"] = ";".join(f"{k}:{v}" for k, v in entry["label_names"].most_common())
        entry["same_seed_continuity"] = len(frames) >= 2
    write_csv(STAGE1 / "semantic_continuity_rows.csv", list(continuity.values()))
    write_csv(STAGE1 / "frame_semantic_summary.csv", frame_summary_rows)
    write_csv(STAGE1 / "semantic_role_mapping.csv", sorted(role_mapping_rows.values(), key=lambda r: str(r["label_name"])))
    write_csv(STAGE1 / "missing_semantic_frames.csv", missing_frames, fieldnames=["seq", "frame_id", "reason"])

    projection_coverage = processed_frames / total_target_frames if total_target_frames else 0.0
    nonvoid_ratio = total_nonvoid_patches / total_patch_rows if total_patch_rows else 0.0
    identity_patch_coverage = total_identity_patches / total_patch_rows if total_patch_rows else 0.0
    role_coverage = 1.0 if total_patch_rows else 0.0
    purity_mean = mean(purity_values)
    summary = {
        "schema": "acl2_v107r_stage1_semantic_cue_summary_v1",
        "stage1_pass": bool(
            grid["token_alignment_pass"]
            and projection_coverage >= 0.95
            and nonvoid_ratio >= 0.95
            and purity_mean >= 0.70
            and role_coverage >= 0.95
        ),
        "token_alignment_pass": grid["token_alignment_pass"],
        "semantic_projection_coverage": projection_coverage,
        "targeted_frame_count": total_target_frames,
        "processed_frame_count": processed_frames,
        "missing_frame_count": len(missing_frames),
        "semantic_patch_nonvoid_ratio": nonvoid_ratio,
        "semantic_patch_purity_mean": purity_mean,
        "semantic_patch_purity_p10": quantile(purity_values, 0.10),
        "semantic_role_coverage": role_coverage,
        "masklet_identity_patch_coverage": identity_patch_coverage,
        "semantic_class_fallback_only_patch_frac": 1.0 - identity_patch_coverage,
        "label_only_ablation": False,
        "main_semantic_trust_run": True,
        "runtime_available": True,
        "target_width": grid["target_width"],
        "target_height": grid["target_height"],
        "patch_grid_h": grid["patch_grid_h"],
        "patch_grid_w": grid["patch_grid_w"],
        "patch_start_idx": grid["patch_start_idx"],
        "special_token_count": grid["special_token_count"],
        "token_semantic_rows": rel(STAGE1 / "token_semantic_rows.csv"),
        "frame_semantic_summary": rel(STAGE1 / "frame_semantic_summary.csv"),
        "semantic_continuity_rows": rel(STAGE1 / "semantic_continuity_rows.csv"),
    }
    write_json(STAGE1 / "semantic_cue_summary.json", summary)
    audit = [
        "# v107R Token Grid Audit",
        "",
        f"- target_size: {grid['target_width']}x{grid['target_height']}",
        f"- patch_size: {grid['patch_size']}",
        f"- patch_grid: {grid['patch_grid_h']}x{grid['patch_grid_w']} = {grid['inferred_patch_count']}",
        f"- trace image_patch token count candidates: {grid['image_patch_token_count_candidates_from_trace']}",
        f"- patch_start_idx / special_token_count: {grid['patch_start_idx']}",
        f"- token_alignment_pass: {grid['token_alignment_pass']}",
        "",
        "Evidence:",
        *[f"- {item}" for item in grid["evidence"]],
        "",
        "Semantic maps are projected with the same cover-fit center-crop rule used by the KITTI dataset wrapper, then patchified into 14x14 LingBot image patches. Special tokens are recorded in token_grid_rows.csv but do not receive direct dense semantic patches; Stage2 joins them through frame-level semantic summaries.",
    ]
    write_text(STAGE1 / "token_grid_audit.md", "\n".join(audit))
    if not summary["stage1_pass"]:
        write_text(
            STAGE1 / "TOKEN_SEMANTIC_ALIGNMENT_FAIL.md",
            "# TOKEN_SEMANTIC_ALIGNMENT_FAIL\n\n"
            f"stage1_pass={summary['stage1_pass']}\n\n"
            f"token_alignment_pass={grid['token_alignment_pass']}, projection_coverage={projection_coverage}, "
            f"nonvoid_ratio={nonvoid_ratio}, purity_mean={purity_mean}, role_coverage={role_coverage}.\n",
        )
    return summary


def aggregate_frame_summaries(frames: list[dict[str, Any]]) -> dict[str, float]:
    cols = [
        "stable_structure_mass",
        "road_boundary_or_layout_mass",
        "ground_or_road_weak_mass",
        "dynamic_transient_mass",
        "vegetation_weak_context_mass",
        "sky_or_lowobs_mass",
        "semantic_boundary_mass",
        "unknown_lowtrust_mass",
        "frame_semantic_update_value",
        "semantic_patch_nonvoid_ratio",
        "semantic_patch_purity_mean",
        "semantic_trust_mean",
        "semantic_confidence_mean",
        "masklet_identity_patch_coverage",
        "semantic_class_fallback_only_frame_frac",
    ]
    return {col: mean([fnum(row, col) for row in frames]) for col in cols}


def normalize_operation_type(op: str) -> str:
    mapping = {
        "cache_append": "append_or_write",
        "trajectory_write": "append_or_write",
        "budget_keep": "budget_keep_drop",
        "budget_drop": "budget_keep_drop",
        "special_token_update": "special_token_lifecycle",
    }
    return mapping.get(op, op)


def stage2_join(op_rows: list[dict[str, str]]) -> dict[str, Any]:
    STAGE2.mkdir(parents=True, exist_ok=True)
    frame_rows = read_csv(STAGE1 / "frame_semantic_summary.csv") if (STAGE1 / "frame_semantic_summary.csv").is_file() else []
    frame_map = {(row["seq"], int(float(row["frame_id"]))): row for row in frame_rows}
    out_rows: list[dict[str, Any]] = []
    joined_count = 0
    for idx, row in enumerate(op_rows):
        seq = row.get("seq") or row.get("sequence_id", "")
        trace_start = inum(row, "trace_start_idx", -1)
        span_start = inum(row, "frame_span_start", -1)
        span_end = inum(row, "frame_span_end", span_start)
        if trace_start >= 0 and span_start >= 0:
            join_frames = list(range(trace_start + span_start, trace_start + span_end + 1))
        else:
            join_frames = [inum(row, "frame_id", -1)]
        available = [frame_map[(seq, fid)] for fid in join_frames if (seq, fid) in frame_map]
        if not available and (seq, inum(row, "frame_id", -1)) in frame_map:
            available = [frame_map[(seq, inum(row, "frame_id", -1))]]
        semantic = aggregate_frame_summaries(available) if available else {}
        joined = bool(available)
        joined_count += int(joined)
        semantic_join_coverage = len(available) / len(join_frames) if join_frames else 0.0
        op = row.get("operation_type", "")
        out_rows.append(
            {
                "schema": "acl2_v107r_operation_semantic_row_v1",
                "operation_id": idx,
                "source_row_index": row.get("operation_row_index", idx),
                "target_id": row.get("target_id", ""),
                "seq": seq,
                "target_kind": row.get("target_kind", ""),
                "safe_good_or_bad_label": row.get("safe_good_or_bad_label", ""),
                "current_frame": row.get("frame_id", ""),
                "semantic_join_frame_start": join_frames[0] if join_frames else "",
                "semantic_join_frame_end": join_frames[-1] if join_frames else "",
                "semantic_join_frame_count": len(join_frames),
                "semantic_join_available_frame_count": len(available),
                "semantic_join_coverage": semantic_join_coverage,
                "semantic_runtime_available": joined,
                "operation_type": op,
                "plan_operation_type": normalize_operation_type(op),
                "memory_path": row.get("context_path", ""),
                "token_type": row.get("token_type", ""),
                "token_id": row.get("token_index", ""),
                "token_count": row.get("token_count", ""),
                "source_frame": row.get("source_frame", ""),
                "source_frame_global_start": join_frames[0] if join_frames else "",
                "source_frame_age": row.get("source_age", ""),
                "keyframe_flag": row.get("is_keyframe", ""),
                "scale_frame_flag": row.get("is_scale_frame", ""),
                "anchor_context_flag": row.get("context_path", "") == "anchor_context",
                "trajectory_memory_flag": row.get("is_trajectory_memory", ""),
                "local_window_flag": row.get("context_path", "") == "local_pose_reference_window",
                "cache_keep_drop_status": "keep" if op == "budget_keep" else ("drop" if op in {"budget_drop", "eviction"} else ""),
                "retention_duration": row.get("source_age", "") if op in {"retention", "budget_keep"} else "",
                "write_frequency": 1 if op in {"cache_append", "trajectory_write", "special_token_update"} else 0,
                "read_frequency": 0,
                "runtime_available": True,
                "L3_handoff_penalty_nearby": row.get("L3_handoff_penalty_nearby", ""),
                "L4_adjacent_log_scale_jump_proxy": row.get("L4_adjacent_log_scale_jump_proxy", ""),
                "local_window_support": row.get("local_window_support", ""),
                **semantic,
            }
        )
    write_csv(STAGE2 / "operation_semantic_rows.csv", out_rows)

    actual_counts = Counter(row.get("operation_type", "") for row in out_rows)
    plan_counts = Counter(row.get("plan_operation_type", "") for row in out_rows)
    required_plan_ops = [
        "readout",
        "initialization",
        "append_or_write",
        "update",
        "retention",
        "eviction",
        "budget_keep_drop",
        "local_reference_separation",
        "special_token_lifecycle",
    ]
    coverage_rows = []
    for op in required_plan_ops:
        coverage_rows.append(
            {
                "schema": "acl2_v107r_operation_type_coverage_row_v1",
                "plan_operation_type": op,
                "observed": plan_counts.get(op, 0) > 0,
                "row_count": plan_counts.get(op, 0),
                "source_operation_types": ";".join(sorted({r["operation_type"] for r in out_rows if r["plan_operation_type"] == op})),
            }
        )
    write_csv(STAGE2 / "operation_type_coverage.csv", coverage_rows)
    missing = [row["plan_operation_type"] for row in coverage_rows if not row["observed"]]
    write_text(
        STAGE2 / "missing_operation_type_report.md",
        "# v107R Missing Operation Type Report\n\n"
        + "\n".join(f"- {op}" for op in missing)
        + "\n\nObserved v107TF operation types: "
        + ", ".join(f"{k}:{v}" for k, v in sorted(actual_counts.items()))
        + "\n",
    )
    trace_summary = load_json(V107TF_STAGE1 / "operation_trace_summary.json")
    semantic_join_coverage = joined_count / len(out_rows) if out_rows else 0.0
    observed_actual = [k for k, v in actual_counts.items() if v > 0]
    stage2_pass = bool(
        any(op != "readout" for op in observed_actual)
        and len(observed_actual) >= 3
        and any(op in actual_counts for op in ["initialization", "cache_append", "retention", "eviction", "budget_keep", "budget_drop"])
        and semantic_join_coverage >= 0.80
        and bool(trace_summary.get("trace_parity_pass"))
    )
    summary = {
        "schema": "acl2_v107r_stage2_summary_v1",
        "stage2_pass": stage2_pass,
        "operation_row_count": len(out_rows),
        "actual_operation_type_count": len(observed_actual),
        "actual_operation_types": sorted(observed_actual),
        "plan_operation_type_count": len([r for r in coverage_rows if r["observed"]]),
        "plan_missing_operation_types": missing,
        "non_readout_operation_present": any(op != "readout" for op in observed_actual),
        "semantic_join_coverage": semantic_join_coverage,
        "trace_parity_pass": bool(trace_summary.get("trace_parity_pass")),
        "semantic_runtime_available": semantic_join_coverage >= 0.80,
        "operation_semantic_rows": rel(STAGE2 / "operation_semantic_rows.csv"),
        "operation_type_coverage": rel(STAGE2 / "operation_type_coverage.csv"),
    }
    write_json(STAGE2 / "stage2_summary.json", summary)
    if not stage2_pass:
        write_text(STAGE2 / "CACHE_OPERATION_OBSERVABILITY_BLOCKED.md", "# CACHE_OPERATION_OBSERVABILITY_BLOCKED\n\nStage2 did not meet operation observability join gates.\n")
    return summary


def bool_float(value: Any) -> float:
    return 1.0 if str(value).lower() in {"true", "1", "yes"} else 0.0


def build_case_features(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    scopes = ["all"] + sorted({row["operation_type"] for row in rows}) + sorted({f"plan:{row['plan_operation_type']}" for row in rows})
    cases: dict[tuple[str, str], dict[str, Any]] = {}
    by_target_scope: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_target_scope[(row["target_id"], "all")].append(row)
        by_target_scope[(row["target_id"], row["operation_type"])].append(row)
        by_target_scope[(row["target_id"], f"plan:{row['plan_operation_type']}")].append(row)
    labels_by_target: dict[str, dict[str, Any]] = {}
    for row in rows:
        labels_by_target.setdefault(
            row["target_id"],
            {
                "target_id": row["target_id"],
                "seq": row["seq"],
                "target_kind": row["target_kind"],
                "label": "bad" if row["target_kind"].startswith("high_l3") else "safe_good",
                "L3": fnum(row, "L3_handoff_penalty_nearby"),
                "L4": fnum(row, "L4_adjacent_log_scale_jump_proxy"),
            },
        )
    for target_id, base in labels_by_target.items():
        for scope in scopes:
            group = by_target_scope.get((target_id, scope), [])
            if not group:
                continue
            feat: dict[str, Any] = {
                **base,
                "operation_scope": scope,
                "available": True,
                "operation_row_count": len(group),
                "source_age_mean": mean([fnum(r, "source_frame_age") for r in group]),
                "cache_token_count_mean": mean([fnum(r, "token_count") for r in group]),
                "is_keyframe_frac": mean([bool_float(r.get("keyframe_flag")) for r in group]),
                "is_scale_frame_frac": mean([bool_float(r.get("scale_frame_flag")) for r in group]),
                "trajectory_memory_frac": mean([bool_float(r.get("trajectory_memory_flag")) for r in group]),
                "local_window_frac": mean([bool_float(r.get("local_window_flag")) for r in group]),
                "anchor_context_frac": mean([bool_float(r.get("anchor_context_flag")) for r in group]),
                "semantic_join_coverage_mean": mean([fnum(r, "semantic_join_coverage") for r in group]),
            }
            semantic_cols = [
                "stable_structure_mass",
                "road_boundary_or_layout_mass",
                "ground_or_road_weak_mass",
                "dynamic_transient_mass",
                "vegetation_weak_context_mass",
                "sky_or_lowobs_mass",
                "semantic_boundary_mass",
                "unknown_lowtrust_mass",
                "frame_semantic_update_value",
                "semantic_patch_nonvoid_ratio",
                "semantic_patch_purity_mean",
                "semantic_trust_mean",
                "semantic_confidence_mean",
                "masklet_identity_patch_coverage",
                "semantic_class_fallback_only_frame_frac",
            ]
            for col in semantic_cols:
                feat[f"{col}_mean"] = mean([fnum(r, col) for r in group])
                feat[f"{col}_max"] = max([fnum(r, col) for r in group if finite(r.get(col, ""))], default=float("nan"))
            cases[(target_id, scope)] = feat
    return cases


def evaluate_selection(cases: list[dict[str, Any]], selected: list[bool]) -> dict[str, Any]:
    bad_total = sum(1 for c in cases if c["label"] == "bad")
    good_total = sum(1 for c in cases if c["label"] != "bad")
    bad_sel = sum(1 for c, s in zip(cases, selected) if s and c["label"] == "bad")
    good_sel = sum(1 for c, s in zip(cases, selected) if s and c["label"] != "bad")
    bad_recall = bad_sel / bad_total if bad_total else 0.0
    good_fpr = good_sel / good_total if good_total else 0.0
    ba = 0.5 * (bad_recall + (1.0 - good_fpr))
    mask = [1.0 if s else 0.0 for s in selected]
    abs_corr_l3 = abs(pearson(mask, [c["L3"] for c in cases]))
    abs_corr_l4 = abs(pearson(mask, [c["L4"] for c in cases]))
    return {
        "selected_count": sum(selected),
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "balanced_accuracy": ba,
        "abs_corr_L3": abs_corr_l3,
        "abs_corr_L4_or_rolling": abs_corr_l4,
        "selected_bad_count": bad_sel,
        "selected_good_count": good_sel,
    }


def exact_same_count_p95(cases: list[dict[str, Any]], selected_count: int) -> tuple[float, float]:
    n = len(cases)
    if selected_count <= 0 or selected_count >= n:
        return 0.5, 0.5
    bas: list[float] = []
    for combo in itertools.combinations(range(n), selected_count):
        mask = [i in combo for i in range(n)]
        bas.append(float(evaluate_selection(cases, mask)["balanced_accuracy"]))
    return mean(bas), quantile(bas, 0.95)


def rotated_shuffle_margin(cases: list[dict[str, Any]], values: list[float], threshold: float, direction: str, observed_ba: float) -> float:
    if len(cases) < 3:
        return float("nan")
    bas = []
    for shift in range(1, len(values)):
        shifted = values[shift:] + values[:shift]
        if direction == ">=":
            selected = [v >= threshold for v in shifted]
        else:
            selected = [v <= threshold for v in shifted]
        bas.append(float(evaluate_selection(cases, selected)["balanced_accuracy"]))
    return observed_ba - max(bas) if bas else float("nan")


def role_rotation_margin(cases: list[dict[str, Any]], feature: str, direction: str, threshold: float, observed_ba: float) -> float:
    role_sets = [
        "stable_structure_mass_mean",
        "road_boundary_or_layout_mass_mean",
        "ground_or_road_weak_mass_mean",
        "dynamic_transient_mass_mean",
        "semantic_boundary_mass_mean",
        "unknown_lowtrust_mass_mean",
    ]
    if feature not in role_sets:
        return float("nan")
    idx = role_sets.index(feature)
    bas = []
    for shift in range(1, len(role_sets)):
        rotated_feature = role_sets[(idx + shift) % len(role_sets)]
        values = [float(c.get(rotated_feature, float("nan"))) for c in cases]
        if not all(math.isfinite(v) for v in values):
            continue
        selected = [v >= threshold for v in values] if direction == ">=" else [v <= threshold for v in values]
        bas.append(float(evaluate_selection(cases, selected)["balanced_accuracy"]))
    return observed_ba - max(bas) if bas else float("nan")


def stage3_matrix() -> dict[str, Any]:
    STAGE3.mkdir(parents=True, exist_ok=True)
    rows = read_csv(STAGE2 / "operation_semantic_rows.csv")
    cases_by_scope = build_case_features(rows)
    scope_names = sorted({scope for _tid, scope in cases_by_scope})
    internal_features = [
        "operation_row_count",
        "source_age_mean",
        "cache_token_count_mean",
        "is_keyframe_frac",
        "is_scale_frame_frac",
        "trajectory_memory_frac",
        "local_window_frac",
        "anchor_context_frac",
    ]
    semantic_features = [
        "stable_structure_mass_mean",
        "road_boundary_or_layout_mass_mean",
        "ground_or_road_weak_mass_mean",
        "dynamic_transient_mass_mean",
        "vegetation_weak_context_mass_mean",
        "sky_or_lowobs_mass_mean",
        "semantic_boundary_mass_mean",
        "unknown_lowtrust_mass_mean",
        "frame_semantic_update_value_mean",
        "semantic_patch_purity_mean_mean",
        "semantic_trust_mean_mean",
        "semantic_confidence_mean_mean",
        "masklet_identity_patch_coverage_mean",
        "semantic_class_fallback_only_frame_frac_mean",
    ]
    metric_rows: list[dict[str, Any]] = []
    by_operation_increment: dict[str, dict[str, Any]] = {}
    for scope in scope_names:
        cases = [case for (tid, sc), case in cases_by_scope.items() if sc == scope]
        cases = sorted(cases, key=lambda c: c["target_id"])
        if len(cases) < 4:
            continue
        seq_cov = len({c["seq"] for c in cases})
        for family, features in [("internal_only", internal_features), ("semantic_only", semantic_features), ("semantic_plus_internal", semantic_features)]:
            if family == "semantic_only" and scope != "all":
                continue
            if family == "semantic_plus_internal" and scope == "all":
                continue
            if family == "internal_only" and scope.startswith("plan:"):
                pass
            for feature in features:
                values = [float(c.get(feature, float("nan"))) for c in cases]
                values = [v for v in values if math.isfinite(v)]
                if len(values) < 4 or max(values) == min(values):
                    continue
                threshold = quantile(values, 0.5)
                for direction in [">=", "<="]:
                    all_values = [float(c.get(feature, float("nan"))) for c in cases]
                    selected = [(v >= threshold if direction == ">=" else v <= threshold) if math.isfinite(v) else False for v in all_values]
                    if sum(selected) == 0:
                        continue
                    ev = evaluate_selection(cases, selected)
                    rand_mean, rand_p95 = exact_same_count_p95(cases, int(ev["selected_count"]))
                    shuffle_margin = rotated_shuffle_margin(cases, all_values, threshold, direction, float(ev["balanced_accuracy"])) if family.startswith("semantic") else float("nan")
                    role_margin = role_rotation_margin(cases, feature, direction, threshold, float(ev["balanced_accuracy"])) if family.startswith("semantic") else float("nan")
                    diagnostic_pass = bool(
                        len(cases) >= 8
                        and seq_cov >= 3
                        and ev["bad_recall"] >= 0.65
                        and ev["good_FPR"] <= 0.35
                        and (ev["abs_corr_L3"] >= 0.30 or ev["abs_corr_L4_or_rolling"] >= 0.30)
                        and (ev["balanced_accuracy"] - rand_p95) >= 0.05
                    )
                    metric_rows.append(
                        {
                            "schema": "acl2_v107r_operation_cue_pattern_metric_v1",
                            "operation_type": scope,
                            "cue_family": family,
                            "pattern_name": f"{feature}{direction}{threshold:.6g}",
                            "feature": feature,
                            "direction": direction,
                            "threshold": threshold,
                            "available_row_count": sum(int(c["operation_row_count"]) for c in cases),
                            "available_case_or_window_count": len(cases),
                            "sequence_coverage": seq_cov,
                            **ev,
                            "same_count_random_BA_mean": rand_mean,
                            "same_count_random_BA_p95": rand_p95,
                            "same_count_random_margin": float(ev["balanced_accuracy"]) - rand_p95,
                            "semantic_shuffle_margin": shuffle_margin,
                            "role_rotation_margin": role_margin,
                            "diagnostic_pass": diagnostic_pass,
                            "runtime_cue_external_depth_used": False,
                        }
                    )
    metric_rows.sort(
        key=lambda r: (
            bool(r.get("diagnostic_pass")),
            float(r.get("same_count_random_margin", -999)),
            float(r.get("balanced_accuracy", -999)),
            -float(r.get("good_FPR", 999)),
        ),
        reverse=True,
    )
    write_csv(STAGE3 / "operation_cue_pattern_metrics.csv", metric_rows)
    write_csv(STAGE3 / "top_memory_lever_candidates.csv", metric_rows[:50])

    for scope in scope_names:
        rows_scope = [r for r in metric_rows if r["operation_type"] == scope]
        if not rows_scope:
            continue
        best_internal = max([r for r in rows_scope if r["cue_family"] == "internal_only"], key=lambda r: float(r["balanced_accuracy"]), default=None)
        best_semantic = max([r for r in rows_scope if r["cue_family"] == "semantic_only"], key=lambda r: float(r["balanced_accuracy"]), default=None)
        best_plus = max([r for r in rows_scope if r["cue_family"] == "semantic_plus_internal"], key=lambda r: float(r["balanced_accuracy"]), default=None)
        if best_plus is None:
            continue
        internal_pattern = best_internal["pattern_name"] if best_internal else "operation_scope_presence_all_cases_constant_baseline"
        internal_ba = float(best_internal["balanced_accuracy"]) if best_internal else 0.5
        internal_fpr = float(best_internal["good_FPR"]) if best_internal else 1.0
        plus_ba = float(best_plus["balanced_accuracy"])
        plus_fpr = float(best_plus["good_FPR"])
        delta_ba = plus_ba - internal_ba
        fpr_reduction = internal_fpr - plus_fpr
        raw_semantic_increment_pass = delta_ba >= 0.05 or fpr_reduction >= 0.10
        controlled_semantic_increment_pass = raw_semantic_increment_pass and bool(best_plus.get("diagnostic_pass"))
        by_operation_increment[scope] = {
            "schema": "acl2_v107r_semantic_increment_by_operation_v1",
            "operation_type": scope,
            "internal_only_baseline_pattern": internal_pattern,
            "internal_only_baseline_BA": internal_ba,
            "internal_only_good_FPR": internal_fpr,
            "semantic_only_best_pattern": best_semantic["pattern_name"] if best_semantic else "",
            "semantic_only_BA": float(best_semantic["balanced_accuracy"]) if best_semantic else "",
            "semantic_plus_internal_pattern": best_plus["pattern_name"],
            "semantic_plus_internal_BA": plus_ba,
            "semantic_plus_internal_good_FPR": plus_fpr,
            "semantic_increment_BA": delta_ba,
            "semantic_increment_FPR_reduction": fpr_reduction,
            "raw_semantic_increment_pass": raw_semantic_increment_pass,
            "semantic_increment_pass": controlled_semantic_increment_pass,
            "semantic_plus_internal_diagnostic_pass": bool(best_plus.get("diagnostic_pass")),
        }
    increment_rows = list(by_operation_increment.values())
    increment_rows.sort(key=lambda r: (bool(r["semantic_increment_pass"]), bool(r["raw_semantic_increment_pass"]), float(r["semantic_increment_BA"])), reverse=True)
    write_csv(STAGE3 / "semantic_increment_by_operation.csv", increment_rows)

    diagnostic_pass_count = sum(1 for r in metric_rows if r.get("diagnostic_pass"))
    raw_semantic_increment_pass_count = sum(1 for r in increment_rows if r.get("raw_semantic_increment_pass"))
    semantic_increment_pass_count = sum(1 for r in increment_rows if r.get("semantic_increment_pass"))
    semantic_action_ready_count = semantic_increment_pass_count
    top = metric_rows[0] if metric_rows else {}
    summary = {
        "schema": "acl2_v107r_stage3_summary_v1",
        "stage3_diagnostic_pass": diagnostic_pass_count > 0,
        "stage3_semantic_increment_pass": semantic_increment_pass_count > 0,
        "stage3_action_entry_pass": semantic_action_ready_count > 0,
        "diagnostic_pass_count": diagnostic_pass_count,
        "raw_semantic_increment_pass_count": raw_semantic_increment_pass_count,
        "semantic_increment_pass_count": semantic_increment_pass_count,
        "semantic_action_ready_count": semantic_action_ready_count,
        "candidate_count": len(metric_rows),
        "operation_increment_row_count": len(increment_rows),
        "top_candidate": top,
        "runtime_external_depth_cue_used": False,
    }
    write_json(STAGE3 / "stage3_summary.json", summary)

    panel_lines = [
        "# v107R Operation x Cue Failure Panels",
        "",
        f"- candidate_count: {len(metric_rows)}",
        f"- diagnostic_pass_count: {diagnostic_pass_count}",
        f"- semantic_increment_pass_count: {semantic_increment_pass_count}",
        f"- semantic_action_ready_count: {semantic_action_ready_count}",
        "",
        "Top candidates:",
    ]
    for row in metric_rows[:10]:
        panel_lines.append(
            f"- {row['operation_type']} / {row['cue_family']} / {row['pattern_name']}: "
            f"BA={float(row['balanced_accuracy']):.3f}, bad_recall={float(row['bad_recall']):.3f}, "
            f"good_FPR={float(row['good_FPR']):.3f}, same_count_margin={float(row['same_count_random_margin']):.3f}, "
            f"diagnostic_pass={row['diagnostic_pass']}"
        )
    write_text(STAGE3 / "operation_cue_failure_panels.md", "\n".join(panel_lines))

    if not (summary["stage3_semantic_increment_pass"] and summary["stage3_action_entry_pass"]):
        lines = [
            "# semantic_increment_failure",
            "",
            "v107R Stage3 did not produce a semantic+internal action-entry candidate.",
            "",
            f"- diagnostic_pass_count: {diagnostic_pass_count}",
            f"- raw_semantic_increment_pass_count: {raw_semantic_increment_pass_count}",
            f"- semantic_increment_pass_count: {semantic_increment_pass_count}",
            f"- semantic_action_ready_count: {semantic_action_ready_count}",
            "",
            "Interpretation:",
            "",
            "- Semantic cue bank and operation join were materialized. Some patterns can show raw improvement versus a constant internal baseline, but none clear the same-count diagnostic control, so the controlled semantic increment gate is not passed.",
            "- No runtime action is allowed from this run.",
            "- MoGe/LingBot-Depth verifier columns were not used as runtime cue features in this matrix.",
        ]
        write_text(STAGE3 / "semantic_increment_failure.md", "\n".join(lines))
    return summary


def write_blocked_later_stages(reason: str, taxonomy: str) -> None:
    write_csv(STAGE4 / "memory_role_rows.csv", [])
    write_csv(STAGE4 / "role_by_operation_summary.csv", [])
    write_text(STAGE4 / "DISAMBIGUATION_FAIL.md", f"# DISAMBIGUATION_FAIL\n\nBlocked before Stage4: {reason}\n")
    write_json(STAGE4 / "stage4_summary.json", {"schema": "acl2_v107r_stage4_summary_v1", "stage4_pass": False, "blocked_reason": reason})

    write_csv(STAGE5 / "action_candidates.csv", [])
    write_csv(STAGE5 / "action_config_manifest.csv", [])
    write_csv(STAGE5 / "control_manifest.csv", [])
    write_json(STAGE5 / "stage5_summary.json", {"schema": "acl2_v107r_stage5_summary_v1", "stage5_pass": False, "blocked_reason": reason})

    write_csv(STAGE6 / "action_metric_rows.csv", [])
    write_text(STAGE6 / "good_harm_attribution.md", f"# good_harm_attribution\n\nNo runtime action was run because {reason}.\n")
    write_json(STAGE6 / "stage6_summary.json", {"schema": "acl2_v107r_stage6_summary_v1", "stage6_pass": False, "blocked_reason": reason})

    write_csv(STAGE7 / "full_validation_metrics.csv", [])
    write_json(STAGE7 / "stage7_summary.json", {"schema": "acl2_v107r_stage7_summary_v1", "stage7_pass": False, "blocked_reason": reason, "taxonomy": taxonomy})


def final_decision(stage0_summary: dict[str, Any], stage1_summary: dict[str, Any], stage2_summary: dict[str, Any], stage3_summary: dict[str, Any]) -> dict[str, Any]:
    if not stage0_summary.get("stage0_pass"):
        taxonomy = "SEMANTIC_CUE_BANK_BLOCKED"
        reason = "Stage0 required artifacts missing or v107TF operation trace unavailable."
    elif not stage1_summary.get("stage1_pass"):
        taxonomy = "SEMANTIC_CUE_BANK_BLOCKED"
        reason = "Stage1 semantic token alignment/cue-bank gate failed."
    elif not stage2_summary.get("stage2_pass"):
        taxonomy = "CACHE_OPERATION_OBSERVABILITY_BLOCKED"
        reason = "Stage2 operation semantic join or trace parity gate failed."
    elif not (stage3_summary.get("stage3_semantic_increment_pass") and stage3_summary.get("stage3_action_entry_pass")):
        taxonomy = "SEMANTIC_INCREMENT_FAIL_INTERNAL_ONLY_DOMINATES"
        reason = "Stage3 semantic+internal candidates did not pass semantic increment plus diagnostic controls."
    else:
        taxonomy = "ACTION_SURFACE_PASS_RUNTIME_READY"
        reason = "Stage3 produced a semantic+internal action-entry candidate; Stage4/5 action selection is required before any runtime pilot."

    write_blocked_later_stages(reason, taxonomy) if taxonomy != "ACTION_SURFACE_PASS_RUNTIME_READY" else None
    decision = {
        "schema": "acl2_v107r_final_decision_v1",
        "taxonomy": taxonomy,
        "reason": reason,
        "stage0_pass": stage0_summary.get("stage0_pass"),
        "stage1_pass": stage1_summary.get("stage1_pass"),
        "stage2_pass": stage2_summary.get("stage2_pass"),
        "stage3_diagnostic_pass": stage3_summary.get("stage3_diagnostic_pass"),
        "stage3_semantic_increment_pass": stage3_summary.get("stage3_semantic_increment_pass"),
        "stage3_action_entry_pass": stage3_summary.get("stage3_action_entry_pass"),
        "runtime_action_run": False,
        "runtime_external_depth_cue_used": False,
        "full_validation_run": False,
        "result_root": rel(V107R),
    }
    write_json(FINAL / "final_decision.json", decision)
    report = [
        "# ACL2 v107R Final Report",
        "",
        f"Taxonomy: `{taxonomy}`",
        "",
        f"Reason: {reason}",
        "",
        "Gate summary:",
        "",
        f"- Stage0 pass: {decision['stage0_pass']}",
        f"- Stage1 pass: {decision['stage1_pass']}",
        f"- Stage2 pass: {decision['stage2_pass']}",
        f"- Stage3 diagnostic pass: {decision['stage3_diagnostic_pass']}",
        f"- Stage3 semantic increment pass: {decision['stage3_semantic_increment_pass']}",
        f"- Stage3 action entry pass: {decision['stage3_action_entry_pass']}",
        "",
        "Key evidence files:",
        "",
        f"- {rel(STAGE1 / 'semantic_cue_summary.json')}",
        f"- {rel(STAGE2 / 'stage2_summary.json')}",
        f"- {rel(STAGE3 / 'stage3_summary.json')}",
        f"- {rel(STAGE3 / 'semantic_increment_by_operation.csv')}",
        "",
        "No LingBot runtime behavior was changed in v107R. No external depth, MoGe, GT, SLAM, or post-hoc Sim(3) was used as a runtime cue.",
    ]
    write_text(FINAL / "final_report.md", "\n".join(report))
    return decision


def main() -> None:
    V107R.mkdir(parents=True, exist_ok=True)
    s0 = stage0()
    op_rows = load_length_matched_operation_rows()
    grid = parse_grid_from_trace(op_rows)
    s1 = semantic_cache_stage1(op_rows, grid)
    s2 = stage2_join(op_rows)
    s3 = stage3_matrix()
    final_decision(s0, s1, s2, s3)


if __name__ == "__main__":
    main()
