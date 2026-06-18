from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d_native.object_field_v42 import build_v42_object_fields_from_alignment_rows, object_field_rows
from stream4d_native.semantic_material_factor_graph import summarize_v42_factor_graph, summary_row
from tools.run_v28_proposal_oracle import _cluster_metrics


ROOT = Path(__file__).resolve().parents[1]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key, "")) for key in keys})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_role_tube_ids(role_root: Path, *, scene: str, variant: str, source: str) -> set[int]:
    rows = _read_csv(role_root / "tube_role_rows.csv")
    out: set[int] = set()
    for row in rows:
        if str(row.get("scene", "")) != str(scene):
            continue
        if str(row.get("variant", "")) != str(variant):
            continue
        if str(row.get("source", "")) != str(source):
            continue
        text = str(row.get("tube_id", "")).strip()
        if text:
            out.add(int(text))
    return out


def _parse_json_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    return [int(v) for v in json.loads(text)]


def _load_token_gt(
    part_graph_root: Path,
    *,
    scene: str,
    variant: str,
    source: str,
) -> dict[int, int]:
    rows = _read_csv(part_graph_root / variant / scene / "part_token_rows.csv")
    out: dict[int, int] = {}
    for row in rows:
        if str(row.get("source", "")) != str(source):
            continue
        token_text = str(row.get("token_id", "")).strip()
        gt_text = str(row.get("diagnostic_gt_instance", "")).strip()
        if not token_text:
            continue
        if not gt_text:
            out[int(token_text)] = 0
            continue
        out[int(token_text)] = int(gt_text)
    return out


def _diagnostic_token_metrics(
    object_rows: list[dict[str, Any]],
    token_gt: dict[int, int],
    *,
    unknown_label_base: int = 1_000_000,
) -> tuple[dict[str, Any], dict[int, int], dict[int, int]]:
    labels_pred: dict[int, int] = {}
    claimed: set[int] = set()
    for row in object_rows:
        object_id = int(row["object_id"])
        for token_id in _parse_json_list(row.get("semantic_masklet_ids", "")):
            labels_pred[int(token_id)] = object_id
            claimed.add(int(token_id))
    unknown_index = 0
    for token_id, gt in sorted(token_gt.items()):
        if int(gt) <= 0:
            continue
        if int(token_id) in labels_pred:
            continue
        labels_pred[int(token_id)] = int(unknown_label_base + unknown_index)
        unknown_index += 1
    gt_labeled = {int(token_id): int(gt) for token_id, gt in token_gt.items() if int(gt) > 0 and int(token_id) in labels_pred}
    metrics = _cluster_metrics(labels_pred, gt_labeled) if labels_pred else {
        "ari": None,
        "purity": None,
        "completeness": None,
        "overmerge": None,
        "oversplit": None,
        "labeled_tube_count": 0,
    }
    metrics.update(
        {
            "diagnostic_labeled_token_count": int(len(gt_labeled)),
            "diagnostic_assigned_labeled_token_count": int(
                sum(1 for token_id in gt_labeled if int(token_id) in claimed)
            ),
            "diagnostic_unknown_labeled_token_count": int(
                sum(1 for token_id in gt_labeled if int(token_id) not in claimed)
            ),
            "diagnostic_unknown_labeled_token_ratio": float(
                sum(1 for token_id in gt_labeled if int(token_id) not in claimed) / max(len(gt_labeled), 1)
            ),
        }
    )
    return metrics, labels_pred, gt_labeled


def _offset_labels(
    scene_index: int,
    labels_pred: dict[int, int],
    gt_labels: dict[int, int],
) -> tuple[dict[int, int], dict[int, int]]:
    key_base = int(scene_index) * 10_000_000
    pred_base = int(scene_index) * 10_000_000
    gt_base = int(scene_index) * 10_000_000
    return (
        {key_base + int(token_id): pred_base + int(label) for token_id, label in labels_pred.items()},
        {
            key_base + int(token_id): gt_base + int(label)
            for token_id, label in gt_labels.items()
            if int(token_id) in labels_pred and int(label) > 0
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v42 full semantic-material factor graph diagnostic.")
    parser.add_argument("--alignment-root", required=True)
    parser.add_argument("--role-root", required=True)
    parser.add_argument("--part-graph-root", required=True)
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--variant", default="Q5")
    parser.add_argument("--sources", default="dinov2_maskcut")
    parser.add_argument("--selected-column", default="selected_O4_semantic_part_gated_robust_trim")
    parser.add_argument("--safe-merge-semantic-affinity", type=float, default=-1.0)
    parser.add_argument("--safe-merge-object-affinity", type=float, default=-1.0)
    parser.add_argument("--safe-merge-max-visible-outside-conflict", type=float, default=0.35)
    parser.add_argument("--max-material-union-count", type=int, default=-1)
    parser.add_argument("--max-fields", type=int, default=300)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    alignment_root = ROOT / str(args.alignment_root)
    role_root = ROOT / str(args.role_root)
    part_graph_root = ROOT / str(args.part_graph_root)
    output_root = ROOT / str(args.output_root)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    sources = [item.strip() for item in str(args.sources).split(",") if item.strip()]
    variant = str(args.variant)
    alignment_rows_all = _read_csv(alignment_root / "alignment_rows.csv")

    object_rows: list[dict[str, Any]] = []
    hard_scene_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    aggregate_pred: dict[int, int] = {}
    aggregate_gt: dict[int, int] = {}
    for scene_index, scene in enumerate(scenes):
        for source in sources:
            rows = [
                row
                for row in alignment_rows_all
                if str(row.get("scene", "")) == scene
                and str(row.get("variant", "")) == variant
                and str(row.get("source", "")) == source
            ]
            fields = build_v42_object_fields_from_alignment_rows(
                rows,
                selected_column=str(args.selected_column),
                safe_merge_semantic_affinity=None
                if float(args.safe_merge_semantic_affinity) < 0.0
                else float(args.safe_merge_semantic_affinity),
                safe_merge_object_affinity=None
                if float(args.safe_merge_object_affinity) < 0.0
                else float(args.safe_merge_object_affinity),
                safe_merge_max_visible_outside_conflict=float(args.safe_merge_max_visible_outside_conflict),
                max_material_union_count=None
                if int(args.max_material_union_count) < 0
                else int(args.max_material_union_count),
                max_fields=int(args.max_fields),
            )
            all_tubes = _load_role_tube_ids(role_root, scene=scene, variant=variant, source=source)
            summary = summarize_v42_factor_graph(fields, all_tube_ids=all_tubes, scene_count=1)
            current_object_rows = object_field_rows(fields, scene=scene, variant=variant, source=source)
            token_gt = _load_token_gt(part_graph_root, scene=scene, variant=variant, source=source)
            token_metrics, labels_pred, gt_labeled = _diagnostic_token_metrics(
                current_object_rows,
                token_gt,
                unknown_label_base=(scene_index + 1) * 1_000_000,
            )
            global_pred, global_gt = _offset_labels(scene_index, labels_pred, gt_labeled)
            aggregate_pred.update(global_pred)
            aggregate_gt.update(global_gt)
            object_rows.extend(current_object_rows)
            row = summary_row(summary, scene=scene, variant=variant, source=source)
            row.update(
                {
                    "4D_ARI": token_metrics.get("ari"),
                    "purity": token_metrics.get("purity"),
                    "completeness": token_metrics.get("completeness"),
                    "overmerge": token_metrics.get("overmerge"),
                    "oversplit": token_metrics.get("oversplit"),
                    "diagnostic_labeled_token_count": token_metrics.get("diagnostic_labeled_token_count"),
                    "diagnostic_assigned_labeled_token_count": token_metrics.get(
                        "diagnostic_assigned_labeled_token_count"
                    ),
                    "diagnostic_unknown_labeled_token_count": token_metrics.get(
                        "diagnostic_unknown_labeled_token_count"
                    ),
                    "diagnostic_unknown_labeled_token_ratio": token_metrics.get(
                        "diagnostic_unknown_labeled_token_ratio"
                    ),
                    "gt_used_only_for_scoring": True,
                    "metric_scope": "semantic_token_diagnostic_not_ap",
                }
            )
            metric_proxy_pass = bool(
                row["phase6_proxy_constraints_pass"]
                and row["4D_ARI"] is not None
                and float(row["purity"]) >= 0.86
                and float(row["completeness"]) >= 0.52
                and (scene != "scene0081_01" or float(row["4D_ARI"]) >= 0.20)
            )
            row["phase6_metric_proxy_pass"] = metric_proxy_pass
            hard_scene_rows.append(row)
            manifests.append(
                {
                    "scene": scene,
                    "variant": variant,
                    "source": source,
                    "alignment_edge_count": int(len(rows)),
                    "selected_column": str(args.selected_column),
                    "safe_merge_semantic_affinity": None
                    if float(args.safe_merge_semantic_affinity) < 0.0
                    else float(args.safe_merge_semantic_affinity),
                    "safe_merge_object_affinity": None
                    if float(args.safe_merge_object_affinity) < 0.0
                    else float(args.safe_merge_object_affinity),
                    "safe_merge_max_visible_outside_conflict": float(args.safe_merge_max_visible_outside_conflict),
                    "max_material_union_count": None
                    if int(args.max_material_union_count) < 0
                    else int(args.max_material_union_count),
                    "object_field_count": int(len(fields)),
                    "role_tube_count": int(len(all_tubes)),
                    "phase6_proxy_constraints_pass": bool(summary.phase6_proxy_constraints_pass),
                    "phase6_metric_proxy_pass": bool(metric_proxy_pass),
                    "phase6_gate_pass": bool(summary.phase6_gate_pass),
                    "phase6_gate_blocker": summary.phase6_gate_blocker,
                }
            )

    aggregate_metrics = _cluster_metrics(aggregate_pred, aggregate_gt) if aggregate_pred else {
        "ari": None,
        "purity": None,
        "completeness": None,
        "overmerge": None,
        "oversplit": None,
        "labeled_tube_count": 0,
    }
    v41_reference = {
        "scope_note": "loaded references are prior global/native-support artifacts, not same hard-scene semantic-token scope",
        "comparable_scope": False,
    }
    for ref_path in [
        ROOT / "outputs/audit/v41_1_real_artifact_bridge/real_identity_bridge_summary.json",
        ROOT / "outputs/audit/v41_1_object_fields/object_field_summary.json",
    ]:
        if ref_path.exists():
            try:
                payload = json.loads(ref_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            v41_reference[str(ref_path.relative_to(ROOT))] = payload.get("key_numbers", payload)
    aggregate = {
        "phase": "v42_full_semantic_material_factor_graph_diagnostic",
        "alignment_root": str(alignment_root),
        "role_root": str(role_root),
        "part_graph_root": str(part_graph_root),
        "selected_column": str(args.selected_column),
        "safe_merge_semantic_affinity": None
        if float(args.safe_merge_semantic_affinity) < 0.0
        else float(args.safe_merge_semantic_affinity),
        "safe_merge_object_affinity": None
        if float(args.safe_merge_object_affinity) < 0.0
        else float(args.safe_merge_object_affinity),
        "safe_merge_max_visible_outside_conflict": float(args.safe_merge_max_visible_outside_conflict),
        "max_material_union_count": None
        if int(args.max_material_union_count) < 0
        else int(args.max_material_union_count),
        "scene_count": int(len(scenes)),
        "object_field_count": int(len(object_rows)),
        "mean_predictions_per_scene": float(len(object_rows) / max(len(scenes), 1)),
        "birth_from_d4rt_tube_count": int(sum(int(row["birth_from_d4rt_tube_count"]) for row in hard_scene_rows)),
        "duplicate_rate_max": float(max([float(row["duplicate_rate"]) for row in hard_scene_rows], default=0.0)),
        "conflict_rate_max": float(max([float(row["conflict_rate"]) for row in hard_scene_rows], default=0.0)),
        "phase6_proxy_constraints_pass": bool(all(row["phase6_proxy_constraints_pass"] for row in hard_scene_rows)),
        "diagnostic_4D_ARI": aggregate_metrics.get("ari"),
        "diagnostic_purity": aggregate_metrics.get("purity"),
        "diagnostic_completeness": aggregate_metrics.get("completeness"),
        "diagnostic_overmerge": aggregate_metrics.get("overmerge"),
        "diagnostic_oversplit": aggregate_metrics.get("oversplit"),
        "scene0081_ARI": next(
            (row.get("4D_ARI") for row in hard_scene_rows if row.get("scene") == "scene0081_01"),
            None,
        ),
        "scene0591_completeness": next(
            (row.get("completeness") for row in hard_scene_rows if row.get("scene") == "scene0591_00"),
            None,
        ),
        "phase6_metric_proxy_pass": bool(all(row.get("phase6_metric_proxy_pass") for row in hard_scene_rows)),
        "phase6_gate_pass": False,
        "phase6_gate_blocker": "semantic-token diagnostic metrics are not a method-compatible AP/native 4D bridge and are not same-scope comparable to v41.1",
        "v41_reference": v41_reference,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "measurement_uses_metric_geometry": False,
        "manifests": manifests,
    }
    _write_csv(output_root / "object_field_rows.csv", object_rows)
    _write_csv(output_root / "hard_scene_rows.csv", hard_scene_rows)
    _write_json(output_root / "factor_graph_summary.json", aggregate)
    print(
        json.dumps(
            _json_safe(
                {
                    "output_root": str(output_root),
                    "object_field_rows": len(object_rows),
                    "hard_scene_rows": len(hard_scene_rows),
                    "phase6_proxy_constraints_pass": aggregate["phase6_proxy_constraints_pass"],
                    "phase6_gate_pass": aggregate["phase6_gate_pass"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
