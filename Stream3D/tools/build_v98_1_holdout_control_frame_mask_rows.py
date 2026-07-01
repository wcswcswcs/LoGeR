#!/usr/bin/env python3
"""Materialize v98.1 holdout control selected frame-mask rows for canonical AP."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_v98_1_phase6_to_phase12_affinity_eval as phase6_12  # noqa: E402


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
        return REPO_ROOT / p
    return ROOT / p


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rows", required=True)
    parser.add_argument("--surfel-rows", required=True)
    parser.add_argument("--surfel-obs-rows", required=True)
    parser.add_argument("--output-rows", required=True)
    parser.add_argument("--preview-rows", default="")
    parser.add_argument("--min-pred-pixels", type=int, default=1)
    parser.add_argument("--min-gt-pixels", type=int, default=1)
    args = parser.parse_args()

    phase6_12.SOURCE_ROWS = _project(args.source_rows)
    phase6_12.SURFEL_ROWS = _project(args.surfel_rows)
    phase6_12.SURFEL_OBS_ROWS = _project(args.surfel_obs_rows)

    source = phase6_12.load_source_context()
    ctx = phase6_12.load_surfel_context()
    selected_all: list[dict[str, Any]] = []
    preview_all: list[dict[str, Any]] = []
    for control_id in phase6_12.CONTROL_CONFIGS:
        emissions, _object_rows = phase6_12.build_control_emissions(control_id, ctx, source)
        _eval_pack, preview_rows, selected_rows, _pixel_collision_count = phase6_12.evaluate_emissions(
            control_id,
            emissions,
            source,
            min_pred_pixels=args.min_pred_pixels,
            min_gt_pixels=args.min_gt_pixels,
        )
        selected_all.extend(selected_rows)
        preview_all.extend(preview_rows)
    _write_csv(_project(args.output_rows), selected_all)
    if args.preview_rows:
        _write_csv(_project(args.preview_rows), preview_all)
    print({"control_selected_row_count": len(selected_all), "control_preview_row_count": len(preview_all)})


if __name__ == "__main__":
    main()
