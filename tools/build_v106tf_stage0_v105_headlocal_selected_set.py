#!/usr/bin/env python3
"""Reconstruct the ACL2 v106 Stage0 selected evidence set from v105 artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V106 = ROOT / "results/acl2_v106tf_lingbot_semantic_aware_memory_role_control"
OUT = V106 / "stage0_v105_headlocal_selected_set"

STAGE2 = V105 / "stage2_gca_trace"
STAGE3 = V105 / "stage3_lingbot_oracle"
STAGE4 = V105 / "stage4_lingbot_action_pilot_or_blocked"
STAGE4_HEAD = V105 / "stage4_lingbot_headlocal_trace"

ACTION_LABEL = "semantic_headlocal_relaxed_context_only_demote"
BASELINE_METHOD = "lingbot_map_stream_default_stage2_notrace"

SELECTED_ROWS = STAGE4_HEAD / "headlocal_relaxed_selected_rows.csv"
HEAD_FEATURE_ROWS = STAGE4_HEAD / "headlocal_frame_head_features.csv"
HEAD_TRACE_ROWS = STAGE4_HEAD / "headlocal_trace_semantic_key_rows.csv"
FRAME_SEMANTIC_ROWS = STAGE3 / "frame_semantic_geometry_rows.csv"
ACTION_CONFIG_ROWS = STAGE4 / "action_config_rows.csv"
ACTION_METRIC_ROWS = STAGE4 / "action_metric_rows.csv"
STAGE4_SUMMARY = STAGE4 / "stage4_summary.json"


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


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    raw = row.get(key, "")
    if raw == "":
        return default
    return int(float(raw))


def as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    raw = row.get(key, "")
    if raw == "":
        return default
    return float(raw)


def parse_indices(raw: str) -> list[int]:
    return [int(x) for x in str(raw or "").split(";") if x != ""]


def parse_heads(raw: str) -> list[int]:
    heads: list[int] = []
    for part in str(raw or "").split(";"):
        part = part.strip()
        if not part:
            continue
        part = part.removeprefix("h")
        for token in part.split(","):
            token = token.strip().removeprefix("h")
            if token:
                heads.append(int(token))
    return sorted(set(heads))


def load_traj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frames: list[int] = []
    mats: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 13:
                raise ValueError(f"bad trajectory row in {path}: {line[:120]}")
            frames.append(int(vals[0]))
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = np.asarray(vals[1:], dtype=np.float64).reshape(3, 4)
            mats.append(mat)
    return np.asarray(frames, dtype=np.int64), np.stack(mats, axis=0)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if len(src) < 3:
        return 1.0, np.eye(3), np.zeros(3)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    x = src - mu_src
    y = dst - mu_dst
    var_src = float(np.mean(np.sum(x * x, axis=1)))
    if var_src <= 1e-12:
        return 1.0, np.eye(3), mu_dst - mu_src
    cov = (y.T @ x) / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1
    rot = u @ np.diag(d) @ vt
    scale = float(np.sum(s * d) / var_src)
    trans = mu_dst - scale * (rot @ mu_src)
    return scale, rot, trans


def sim3_residuals(gt_path: Path, pred_path: Path) -> dict[int, float]:
    gt_frames, gt = load_traj(gt_path)
    pred_frames, pred = load_traj(pred_path)
    if not np.array_equal(gt_frames, pred_frames):
        raise ValueError(f"frame mismatch: {gt_path} vs {pred_path}")
    gt_pos = gt[:, :3, 3]
    pred_pos = pred[:, :3, 3]
    scale, rot, trans = umeyama(pred_pos, gt_pos)
    aligned = scale * (pred_pos @ rot.T) + trans
    residual = np.linalg.norm(aligned - gt_pos, axis=1)
    return {int(frame): float(err) for frame, err in zip(gt_frames, residual)}


def residual_maps(seq: str) -> tuple[dict[int, float], dict[int, float]]:
    dataset = f"kitti_v105_seq{seq}_trace32"
    method = f"lingbot_map_stage4_{ACTION_LABEL}_seq{seq}"
    gt_traj = STAGE2 / f"workspace/{dataset}/{seq}/gt/traj.txt"
    baseline_traj = STAGE2 / f"workspace/{dataset}/{seq}/{BASELINE_METHOD}/traj.txt"
    action_traj = STAGE4 / f"workspace/{dataset}/{seq}/{method}/traj.txt"
    return sim3_residuals(gt_traj, baseline_traj), sim3_residuals(gt_traj, action_traj)


def dominant_context_role(head_feature: dict[str, str] | None) -> str:
    if not head_feature:
        return ""
    candidates = {
        "scale_reference_context": as_float(head_feature, "scale_reference_context_attention_frac", 0.0),
        "local_pose_reference_window": as_float(head_feature, "local_window_context_attention_frac", 0.0),
        "current_or_latest_frame": as_float(head_feature, "current_or_latest_frame_attention_frac", 0.0),
    }
    return max(candidates.items(), key=lambda item: item[1])[0]


def label_type(row: dict[str, str]) -> str:
    if parse_bool(row.get("bad_label")):
        return "bad_selected"
    if parse_bool(row.get("good_label")):
        return "good_selected"
    return "neutral_selected"


def top_trace_roles(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    counts = Counter(row.get("key_semantic_role", "") for row in rows if row.get("key_semantic_role", ""))
    return ";".join(f"{key}:{value}" for key, value in counts.most_common())


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = read_csv(SELECTED_ROWS)
    head_features = {
        (row["seq"], as_int(row, "sample_position"), as_int(row, "head_idx")): row
        for row in read_csv(HEAD_FEATURE_ROWS)
    }
    semantic_rows = {
        (row["seq"], as_int(row, "sample_position")): row
        for row in read_csv(FRAME_SEMANTIC_ROWS)
    }
    action_configs = [
        row for row in read_csv(ACTION_CONFIG_ROWS)
        if row.get("action_label") == ACTION_LABEL
    ]
    action_metrics = [
        row for row in read_csv(ACTION_METRIC_ROWS)
        if row.get("action_label") == ACTION_LABEL
    ]
    stage4_summary = json.loads(STAGE4_SUMMARY.read_text(encoding="utf-8"))
    policy_summary = json.loads((STAGE4_HEAD / "headlocal_policy_summary.json").read_text(encoding="utf-8"))

    trace_rows_by_key: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(HEAD_TRACE_ROWS):
        raw_pos = row.get("current_sample_position", "")
        raw_head = row.get("head_idx", "")
        if raw_pos == "" or raw_head == "":
            continue
        trace_rows_by_key[(row["seq"], int(float(raw_pos)), int(float(raw_head)))].append(row)

    residual_by_seq = {seq: residual_maps(seq) for seq in sorted({row["seq"] for row in selected})}

    rows: list[dict[str, Any]] = []
    for source in selected:
        seq = source["seq"]
        sample_position = as_int(source, "sample_position")
        semantic = semantic_rows.get((seq, sample_position), {})
        baseline_res, action_res = residual_by_seq[seq]
        baseline_l3 = baseline_res.get(sample_position, float("nan"))
        action_l3 = action_res.get(sample_position, float("nan"))
        denom = max(abs(baseline_l3), 1e-9)
        signed_improvement = (baseline_l3 - action_l3) / denom
        good_harm = (action_l3 - baseline_l3) / denom
        current_label = label_type(source)
        if current_label == "bad_selected":
            improvement_or_harm = signed_improvement
            improvement_or_harm_type = "bad_improvement"
        elif current_label == "good_selected":
            improvement_or_harm = good_harm
            improvement_or_harm_type = "good_harm"
        else:
            improvement_or_harm = signed_improvement
            improvement_or_harm_type = "signed_improvement_neutral"

        for head_id in parse_heads(source.get("selected_heads", "")):
            head_feature = head_features.get((seq, sample_position, head_id))
            selected_trace_rows = trace_rows_by_key.get((seq, sample_position, head_id), [])
            rows.append(
                {
                    "schema": "acl2_v106tf_stage0_selected_evidence_row_v1",
                    "seq_id": seq,
                    "frame_id": sample_position,
                    "original_frame": source.get("original_frame", ""),
                    "local_window_or_chunk_id": semantic.get("semantic_chunk", ""),
                    "head_id": head_id,
                    "context_role": dominant_context_role(head_feature),
                    "token_group_id": f"{seq}:{sample_position}:h{head_id}",
                    "selected_by_v105_policy": True,
                    "label_type": current_label,
                    "baseline_L3": baseline_l3,
                    "action_L3": action_l3,
                    "improvement_or_harm": improvement_or_harm,
                    "improvement_or_harm_type": improvement_or_harm_type,
                    "bad_improvement": signed_improvement,
                    "good_harm": good_harm,
                    "action_policy_id": source.get("policy", ""),
                    "source_artifact_path": SELECTED_ROWS.relative_to(ROOT).as_posix(),
                    "action_label": ACTION_LABEL,
                    "selected_heads_raw": source.get("selected_heads", ""),
                    "sim3_residual_m_v105_semantic_row": source.get("sim3_residual_m", ""),
                    "bad_label": source.get("bad_label", ""),
                    "good_label": source.get("good_label", ""),
                    "top_labels": source.get("top_labels", ""),
                    "semantic_source": semantic.get("semantic_source", ""),
                    "patch_count": semantic.get("patch_count", ""),
                    "scale_reference_patch_frac": semantic.get("scale_reference_patch_frac", ""),
                    "local_registration_patch_frac": semantic.get("local_registration_patch_frac", ""),
                    "context_only_patch_frac": semantic.get("context_only_patch_frac", ""),
                    "reject_unreliable_patch_frac": semantic.get("reject_unreliable_patch_frac", ""),
                    "semantic_confidence_mean": semantic.get("semantic_confidence_mean", ""),
                    "semantic_confidence_p10": semantic.get("semantic_confidence_p10", ""),
                    "head_trace_topk_rows": head_feature.get("head_trace_topk_rows", "") if head_feature else "",
                    "head_trace_topk_attention_sum": head_feature.get("head_trace_topk_attention_sum", "") if head_feature else "",
                    "scale_reference_context_attention_frac": head_feature.get("scale_reference_context_attention_frac", "") if head_feature else "",
                    "local_window_context_attention_frac": head_feature.get("local_window_context_attention_frac", "") if head_feature else "",
                    "current_or_latest_frame_attention_frac": head_feature.get("current_or_latest_frame_attention_frac", "") if head_feature else "",
                    "semantic_scale_reference_attention_frac": head_feature.get("semantic_scale_reference_attention_frac", "") if head_feature else "",
                    "semantic_local_registration_attention_frac": head_feature.get("semantic_local_registration_attention_frac", "") if head_feature else "",
                    "semantic_context_only_attention_frac": head_feature.get("semantic_context_only_attention_frac", "") if head_feature else "",
                    "semantic_reject_unreliable_attention_frac": head_feature.get("semantic_reject_unreliable_attention_frac", "") if head_feature else "",
                    "trace_rows_for_frame_head": len(selected_trace_rows),
                    "trace_top_semantic_roles": top_trace_roles(selected_trace_rows),
                    "trace_source_artifact_path": HEAD_TRACE_ROWS.relative_to(ROOT).as_posix(),
                }
            )

    hard_negative_rows = [row for row in rows if row["label_type"] == "good_selected"]

    by_seq_positions: dict[str, list[int]] = defaultdict(list)
    for row in selected:
        by_seq_positions[row["seq"]].append(as_int(row, "sample_position"))

    forced_consistency: dict[str, Any] = {}
    for cfg in action_configs:
        seq = cfg["seq"]
        selected_positions = sorted(set(by_seq_positions[seq]))
        forced = parse_indices(cfg.get("force_non_keyframe_indices", ""))
        forced_consistency[seq] = {
            "selected_positions": selected_positions,
            "forced_indices": forced,
            "match": selected_positions == forced,
            "config": cfg.get("config", ""),
            "method": cfg.get("method", ""),
            "trace_file": cfg.get("trace_file", ""),
            "action_file": cfg.get("action_file", ""),
        }

    label_counts = Counter(row["label_type"] for row in rows)
    frame_label_counts = Counter(label_type(row) for row in selected)
    trace_materialized_rows = sum(1 for row in rows if int(row["trace_rows_for_frame_head"]) > 0)
    head_feature_materialized_rows = sum(1 for row in rows if row["head_trace_topk_rows"] != "")
    selected_frame_count = len(selected)
    selected_evidence_count = len(rows)
    trace_coverage = trace_materialized_rows / selected_evidence_count if selected_evidence_count else 0.0
    head_feature_coverage = head_feature_materialized_rows / selected_evidence_count if selected_evidence_count else 0.0

    metric_by_seq = {row["seq"]: row for row in action_metrics}
    stage4_metric = stage4_summary.get("semantic_headlocal_relaxed_context_only_metrics", {})
    summary = {
        "schema": "acl2_v106tf_stage0_v105_selected_set_summary_v1",
        "action_label": ACTION_LABEL,
        "selected_frame_count": selected_frame_count,
        "selected_evidence_count_after_head_expansion": selected_evidence_count,
        "selected_frame_label_counts": dict(frame_label_counts),
        "selected_evidence_label_counts": dict(label_counts),
        "hard_negative_evidence_rows": len(hard_negative_rows),
        "forced_consistency": forced_consistency,
        "forced_consistency_all_match": all(item["match"] for item in forced_consistency.values()),
        "head_feature_coverage": head_feature_coverage,
        "trace_frame_head_coverage": trace_coverage,
        "stage4_semantic_headlocal_relaxed_metrics": stage4_metric,
        "per_seq_action_metric_rows": metric_by_seq,
        "v105_relaxed_policy_summary": policy_summary.get("relaxed_candidate", {}),
        "outputs": {
            "selected_evidence_rows": (OUT / "selected_evidence_rows.csv").relative_to(ROOT).as_posix(),
            "hard_negative_rows": (OUT / "hard_negative_rows.csv").relative_to(ROOT).as_posix(),
            "selected_set_reconstruction_report": (OUT / "selected_set_reconstruction_report.md").relative_to(ROOT).as_posix(),
            "action_surface_frozen_manifest": (OUT / "action_surface_frozen_manifest.json").relative_to(ROOT).as_posix(),
            "stage0_summary": (OUT / "stage0_summary.json").relative_to(ROOT).as_posix(),
        },
        "stage0_selected_set_reconstructed": bool(rows) and all(item["match"] for item in forced_consistency.values()),
        "note": "baseline_L3/action_L3 are per-frame Sim3 residuals recomputed from v105 baseline/action traj.txt files.",
    }

    write_csv(OUT / "selected_evidence_rows.csv", rows)
    write_csv(OUT / "hard_negative_rows.csv", hard_negative_rows)
    (OUT / "stage0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "acl2_v106tf_action_surface_frozen_manifest_v1",
        "action_label": ACTION_LABEL,
        "source_policy": policy_summary.get("relaxed_candidate", {}).get("policy", ""),
        "selected_positions": {seq: data["selected_positions"] for seq, data in forced_consistency.items()},
        "forced_indices": {seq: data["forced_indices"] for seq, data in forced_consistency.items()},
        "forced_consistency_all_match": summary["forced_consistency_all_match"],
        "v105_stage4_metric": stage4_metric,
        "source_artifacts": {
            "selected_rows": SELECTED_ROWS.relative_to(ROOT).as_posix(),
            "head_features": HEAD_FEATURE_ROWS.relative_to(ROOT).as_posix(),
            "head_trace_rows": HEAD_TRACE_ROWS.relative_to(ROOT).as_posix(),
            "frame_semantic_geometry_rows": FRAME_SEMANTIC_ROWS.relative_to(ROOT).as_posix(),
            "action_config_rows": ACTION_CONFIG_ROWS.relative_to(ROOT).as_posix(),
            "action_metric_rows": ACTION_METRIC_ROWS.relative_to(ROOT).as_posix(),
            "stage4_summary": STAGE4_SUMMARY.relative_to(ROOT).as_posix(),
        },
    }
    (OUT / "action_surface_frozen_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = f"""# Stage0 v105 Head-Local Selected Set Reconstruction

Source action surface: `{ACTION_LABEL}`

## Inputs

- selected rows: `{SELECTED_ROWS.relative_to(ROOT).as_posix()}`
- head features: `{HEAD_FEATURE_ROWS.relative_to(ROOT).as_posix()}`
- head trace semantic rows: `{HEAD_TRACE_ROWS.relative_to(ROOT).as_posix()}`
- semantic/geometric frame rows: `{FRAME_SEMANTIC_ROWS.relative_to(ROOT).as_posix()}`
- action configs: `{ACTION_CONFIG_ROWS.relative_to(ROOT).as_posix()}`
- action metrics: `{ACTION_METRIC_ROWS.relative_to(ROOT).as_posix()}`
- stage4 summary: `{STAGE4_SUMMARY.relative_to(ROOT).as_posix()}`

## Reconstruction Result

- selected_frame_count: `{selected_frame_count}`
- selected_evidence_count_after_head_expansion: `{selected_evidence_count}`
- selected_frame_label_counts: `{dict(frame_label_counts)}`
- selected_evidence_label_counts: `{dict(label_counts)}`
- hard_negative_evidence_rows: `{len(hard_negative_rows)}`
- forced_consistency_all_match: `{summary["forced_consistency_all_match"]}`
- head_feature_coverage: `{head_feature_coverage}`
- trace_frame_head_coverage: `{trace_coverage}`

The selected evidence table expands v105 frame-level selected rows by `selected_heads`.
`baseline_L3` and `action_L3` are not copied from aggregate metrics; they are per-frame Sim3 residuals recomputed from v105 `traj.txt` files using the same Umeyama alignment procedure as v105 Stage4 action summary.

## Forced Index Consistency

```json
{json.dumps(forced_consistency, ensure_ascii=False, indent=2, sort_keys=True)}
```

## v105 Stage4 Metric Snapshot

```json
{json.dumps(stage4_metric, ensure_ascii=False, indent=2, sort_keys=True)}
```

## Boundary

Stage0 only reconstructs and freezes the selected set. It does not prove Stage1 materialization, Stage3 disambiguation, or Stage4 action readiness.
"""
    (OUT / "selected_set_reconstruction_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> int:
    summary = build()
    if not summary["stage0_selected_set_reconstructed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
