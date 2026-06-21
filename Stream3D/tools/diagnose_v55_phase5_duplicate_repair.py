from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from stream4d_native.v53_local_objectlets import weighted_partition_metrics


ROOT = Path(__file__).resolve().parents[1]


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _parse_mask_observation_id(mask_observation_id: str) -> tuple[str, int, int] | None:
    try:
        scene, frame_id, mask_id = str(mask_observation_id).rsplit(":", 2)
    except ValueError:
        return None
    return scene, int(frame_id), int(mask_id)


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics_for(
    *,
    support_rows: list[tuple[str, str, str, float]],
    component_to_prediction: dict[tuple[str, str], str],
) -> dict[str, float]:
    assignments = []
    for scene, component_id, gt_id, weight in support_rows:
        prediction = component_to_prediction.get((scene, component_id), f"unknown:{scene}:{component_id}")
        assignments.append((prediction, gt_id, weight))
    return weighted_partition_metrics(assignments)


def build_duplicate_repair_diagnostic(
    *,
    history_component_rows_path: str | Path,
    history_rows_path: str | Path,
    anchor_birth_rows_path: str | Path,
    support_rows_path: str | Path,
    support_variant: str = "R0_visible_tau0.05",
    shared_thresholds: tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200, 500),
) -> dict[str, Any]:
    history_component_rows = _read_csv(_project(history_component_rows_path))
    history_rows = _read_csv(_project(history_rows_path))
    anchor_birth_rows = _read_csv(_project(anchor_birth_rows_path))

    history_scene: dict[str, str] = {}
    history_gt: dict[str, str] = {}
    anchor_chunk: dict[str, str] = {}
    for row in history_rows:
        history_id = str(row.get("history_id"))
        history_scene[history_id] = str(row.get("scene"))
        history_gt[history_id] = str(row.get("dominant_gt_diagnostic") or "")
        anchor_chunk[history_id] = str(row.get("anchor_chunk_id") or "")

    source_mask: dict[str, str] = {}
    for row in anchor_birth_rows:
        if str(row.get("accepted_birth")).lower() != "true":
            continue
        source_mask[str(row.get("birth_object_id"))] = str(row.get("source_mask_observation_id") or "")

    component_to_histories: dict[tuple[str, str], set[str]] = defaultdict(set)
    history_to_components: dict[str, set[str]] = defaultdict(set)
    for row in history_component_rows:
        scene = str(row.get("scene"))
        history_id = str(row.get("history_id"))
        component_id = str(row.get("component_id"))
        component_to_histories[(scene, component_id)].add(history_id)
        history_to_components[history_id].add(component_id)

    pair_shared: Counter[tuple[str, str]] = Counter()
    for histories in component_to_histories.values():
        if len(histories) < 2:
            continue
        for left, right in combinations(sorted(histories), 2):
            pair_shared[(left, right)] += 1

    pair_rows: list[dict[str, Any]] = []
    for (left, right), shared_count in sorted(pair_shared.items(), key=lambda item: (-item[1], item[0])):
        left_source = source_mask.get(left, "")
        right_source = source_mask.get(right, "")
        left_parsed = _parse_mask_observation_id(left_source)
        right_parsed = _parse_mask_observation_id(right_source)
        same_frame_conflict = False
        if left_parsed is not None and right_parsed is not None:
            same_frame_conflict = (
                left_parsed[0] == right_parsed[0]
                and left_parsed[1] == right_parsed[1]
                and left_parsed[2] != right_parsed[2]
            )
        min_size = max(min(len(history_to_components[left]), len(history_to_components[right])), 1)
        pair_rows.append(
            {
                "scene": history_scene[left],
                "left_history_id": left,
                "right_history_id": right,
                "shared_component_count": int(shared_count),
                "shared_min_ratio": float(shared_count / min_size),
                "same_frame_conflict": bool(same_frame_conflict),
                "same_anchor_chunk": bool(anchor_chunk.get(left) == anchor_chunk.get(right) and anchor_chunk.get(left)),
                "left_source_mask_observation_id": left_source,
                "right_source_mask_observation_id": right_source,
                "same_gt_diagnostic": bool(history_gt.get(left) == history_gt.get(right) and history_gt.get(left)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    scenes = set(history_scene.values())
    support_rows: list[tuple[str, str, str, float]] = []
    with _project(support_rows_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("variant")) != support_variant:
                continue
            scene = str(row.get("scene"))
            if scene not in scenes:
                continue
            gt_id = str(row.get("diagnostic_gt_instance") or "")
            if not gt_id or gt_id == "0":
                continue
            support_rows.append(
                (
                    scene,
                    str(row.get("component_id")),
                    f"{scene}|gt:{gt_id}",
                    float(max(int(row.get("support_count") or 1), 1)),
                )
            )

    base_component_to_history: dict[tuple[str, str], str] = {}
    for history_id, components in history_to_components.items():
        scene = history_scene[history_id]
        for component_id in components:
            key = (scene, component_id)
            if key not in base_component_to_history or history_id < base_component_to_history[key]:
                base_component_to_history[key] = history_id

    baseline_metrics = _metrics_for(
        support_rows=support_rows,
        component_to_prediction=base_component_to_history,
    )

    threshold_rows: list[dict[str, Any]] = []
    history_ids = sorted(history_scene)
    for shared_min in shared_thresholds:
        selected_pairs = [row for row in pair_rows if int(row["shared_component_count"]) >= int(shared_min)]
        uf = UnionFind(history_ids)
        for row in selected_pairs:
            uf.union(str(row["left_history_id"]), str(row["right_history_id"]))
        repaired_assignment = {
            key: uf.find(history_id) for key, history_id in base_component_to_history.items()
        }
        metrics = _metrics_for(support_rows=support_rows, component_to_prediction=repaired_assignment)
        no_same_frame_pairs = [row for row in selected_pairs if not row["same_frame_conflict"]]
        no_conflict_no_same_anchor_pairs = [
            row for row in no_same_frame_pairs if not row["same_anchor_chunk"]
        ]
        threshold_rows.append(
            {
                "shared_min": int(shared_min),
                "merge_pair_count": len(selected_pairs),
                "merged_group_count": len({uf.find(history_id) for history_id in history_ids}),
                "same_gt_pair_rate_diagnostic": (
                    sum(bool(row["same_gt_diagnostic"]) for row in selected_pairs) / len(selected_pairs)
                    if selected_pairs
                    else None
                ),
                "same_frame_conflict_pair_count": sum(bool(row["same_frame_conflict"]) for row in selected_pairs),
                "same_anchor_chunk_pair_count": sum(bool(row["same_anchor_chunk"]) for row in selected_pairs),
                "no_same_frame_conflict_pair_count": len(no_same_frame_pairs),
                "no_same_frame_conflict_same_gt_rate_diagnostic": (
                    sum(bool(row["same_gt_diagnostic"]) for row in no_same_frame_pairs) / len(no_same_frame_pairs)
                    if no_same_frame_pairs
                    else None
                ),
                "no_conflict_no_same_anchor_chunk_pair_count": len(no_conflict_no_same_anchor_pairs),
                "no_conflict_no_same_anchor_chunk_same_gt_rate_diagnostic": (
                    sum(bool(row["same_gt_diagnostic"]) for row in no_conflict_no_same_anchor_pairs)
                    / len(no_conflict_no_same_anchor_pairs)
                    if no_conflict_no_same_anchor_pairs
                    else None
                ),
                "ARI": metrics["ARI"],
                "purity": metrics["purity"],
                "completeness": metrics["completeness"],
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    return {
        "phase": "v55_phase5_duplicate_repair_diagnostic",
        "history_count": len(history_scene),
        "components_with_multi_history_count": sum(
            1 for histories in component_to_histories.values() if len(histories) > 1
        ),
        "candidate_pair_count": len(pair_rows),
        "baseline_metrics": baseline_metrics,
        "top_pair_rows": pair_rows[:10],
        "threshold_rows": threshold_rows,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Stream4D v55 Phase5 duplicate repair candidates.")
    parser.add_argument(
        "--history-component-rows",
        default="outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_plus_cosupport_seed038_componentrows_probe/history_component_rows.csv",
    )
    parser.add_argument(
        "--history-rows",
        default="outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_plus_cosupport_seed038_componentrows_probe/history_rows.csv",
    )
    parser.add_argument("--anchor-birth-rows", default="outputs/audit/v55_anchor_birth/anchor_birth_rows.csv")
    parser.add_argument(
        "--support-rows",
        default="outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/mask_component_support_rows.csv",
    )
    parser.add_argument("--support-variant", default="R0_visible_tau0.05")
    parser.add_argument("--output-root", default="outputs/audit/v55_phase5_duplicate_repair_diagnostic")
    args = parser.parse_args()

    payload = build_duplicate_repair_diagnostic(
        history_component_rows_path=args.history_component_rows,
        history_rows_path=args.history_rows,
        anchor_birth_rows_path=args.anchor_birth_rows,
        support_rows_path=args.support_rows,
        support_variant=args.support_variant,
    )
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "duplicate_repair_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    _write_csv(out / "duplicate_pair_rows.csv", payload["top_pair_rows"])
    _write_csv(out / "duplicate_threshold_rows.csv", payload["threshold_rows"])
    print(
        {
            "duplicate_repair_summary": str(out / "duplicate_repair_summary.json"),
            "candidate_pair_count": payload["candidate_pair_count"],
            "components_with_multi_history_count": payload["components_with_multi_history_count"],
        }
    )


if __name__ == "__main__":
    main()
