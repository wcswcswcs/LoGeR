from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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
            writer.writerow({key: row.get(key, "") for key in keys})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _bool_str(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _has_usable_observation(row: dict[str, str], scannet_root: Path) -> bool:
    try:
        observations = json.loads(row.get("support_observation_ids_json") or "[]")
    except json.JSONDecodeError:
        return False
    for token in observations:
        parts = str(token).split(":")
        if len(parts) != 4 or parts[0] != "m":
            continue
        scene, frame_s = parts[1], parts[2]
        if (scannet_root / scene / "output_Cropformer" / "mask" / f"{int(frame_s)}.png").exists():
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build diagnostic-negative active-query candidates for v62 follow-up controls.")
    parser.add_argument("--novelty-csv", default="Stream3D/outputs/audit/v62_increment_attribution/novelty_material_rows.csv")
    parser.add_argument("--scannet-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    rows = _read_csv(Path(args.novelty_csv))
    scannet_root = Path(args.scannet_root)
    candidates: list[dict[str, Any]] = []
    skipped = {"no_support_observation": 0, "diagnostic_positive": 0, "no_usable_mask": 0}
    for row in rows:
        observations = row.get("support_observation_ids_json") or "[]"
        try:
            obs_list = json.loads(observations)
        except json.JSONDecodeError:
            obs_list = []
        if not obs_list:
            skipped["no_support_observation"] += 1
            continue
        can_confirm = _bool_str(row.get("can_enter_confirmed_core")) and _bool_str(row.get("diagnostic_exact_match"))
        can_quarantine = _bool_str(row.get("can_enter_quarantine")) and row.get("state") in {"shared", "quarantine"}
        if can_confirm or can_quarantine:
            skipped["diagnostic_positive"] += 1
            continue
        if not _has_usable_observation(row, scannet_root):
            skipped["no_usable_mask"] += 1
            continue
        candidates.append(
            {
                "query_candidate_id": f"v62neg_{len(candidates):08d}",
                "material_node_id": row["material_node_id"],
                "scene": row.get("scene", ""),
                "component_id": row.get("component_id", ""),
                "candidate_source": "diagnostic_negative",
                "state": row.get("state", ""),
                "novelty_type": row.get("novelty_type", ""),
                "support_observation_count": row.get("support_observation_count", 0),
                "has_material_boundary_source": True,
                "has_existing_query_outcome": False,
                "valid_material_evidence": "",
                "confirm_or_quarantine_outcome": "",
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    output_root = Path(args.output_root)
    _write_csv(output_root / "diagnostic_negative_query_candidates.csv", candidates)
    scene_counts: dict[str, int] = {}
    for row in candidates:
        scene = str(row.get("scene", ""))
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
    _write_json(
        output_root / "diagnostic_negative_candidate_summary.json",
        {
            "candidate_count": len(candidates),
            "scene_counts": scene_counts,
            "skipped_counts": skipped,
            "method_note": "Diagnostic-negative candidate construction for v62 active-query control exploration; not part of original v62 final gate.",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
    )
    print(json.dumps({"candidate_count": len(candidates), "scene_counts": scene_counts, "skipped_counts": skipped}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
