#!/usr/bin/env python3
"""Select v30 diverse masklet causal-bank wave-1 interventions.

The v30 plan explicitly asks for a diverse masklet causal bank, not just the
top projected-support road masklet used in v29C. This selector uses landed
masklet-3D alignment evidence only. If D_g/conflict/scale per-masklet features
are unavailable, it records that limitation instead of inventing them.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        text = str(row.get(key, "")).strip()
        return float(text) if text else default
    except Exception:
        return default


def _b(row: Dict[str, str], key: str) -> Optional[bool]:
    text = str(row.get(key, "")).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _masklet_key(row: Dict[str, str]) -> str:
    return f"{int(_f(row, 'chunk_idx', -1))}:{int(_f(row, 'masklet_id', -1))}"


def _choose(
    rows: Sequence[Dict[str, str]],
    *,
    category: str,
    predicate: Callable[[Dict[str, str]], bool],
    key: Callable[[Dict[str, str]], tuple[float, ...]],
    used: set[str],
) -> Optional[Dict[str, object]]:
    candidates = [row for row in rows if predicate(row) and _masklet_key(row) not in used]
    if not candidates:
        return None
    row = max(candidates, key=key)
    used.add(_masklet_key(row))
    out: Dict[str, object] = dict(row)
    out["selection_category"] = category
    out["selection_key"] = "landed_alignment_proxy"
    return out


def _common(row: Dict[str, object]) -> Dict[str, object]:
    return {
        "selection_category": row.get("selection_category", ""),
        "chunk_idx": int(float(str(row.get("chunk_idx", -1)) or -1)),
        "masklet_id": int(float(str(row.get("masklet_id", -1)) or -1)),
        "fine_label_pred": row.get("fine_label_pred", ""),
        "video_group_pred": row.get("video_group_pred", ""),
        "projected_majority_semantic_name": row.get("projected_majority_semantic_name", ""),
        "projected_majority_group": row.get("projected_majority_group", ""),
        "projected_pixel_support_count": int(float(str(row.get("projected_pixel_support_count", 0)) or 0)),
        "agreement_pred_vs_projected": row.get("agreement_pred_vs_projected", ""),
        "q_3d": row.get("q_3d", ""),
        "t_mask": row.get("t_mask", ""),
        "area_mean": row.get("area_mean", ""),
        "mask_temporal_iou_mean": row.get("mask_temporal_iou_mean", ""),
        "feature_gap": "D_g/conflict/scale/source_attention unavailable per masklet in landed v29C alignment",
    }


def _planned_interventions(selected: Sequence[Dict[str, object]], max_rows: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in selected:
        cat = str(row.get("selection_category", ""))
        common = _common(row)
        if cat in {"road_high_support", "structure_high_support", "large_area"}:
            actions = [
                ("global", "source_skip"),
                ("swa", "swa_anchor_keep"),
                ("ttt", "ttt_positive"),
            ]
        elif cat in {"vegetation_high_support", "grass_or_terrain", "low_trust", "disagreement"}:
            actions = [
                ("global", "source_skip"),
                ("swa", "swa_remove"),
                ("ttt", "ttt_negative"),
            ]
        elif cat in {"movable_projected"}:
            actions = [
                ("global", "source_skip"),
                ("ttt", "ttt_negative"),
            ]
        elif cat in {"sky_or_low_projection"}:
            actions = [
                ("frame", "source_skip"),
                ("global", "source_skip"),
            ]
        else:
            actions = [("global", "source_skip")]
        for path, action in actions:
            rows.append({
                **common,
                "intervention_path": path,
                "intervention_action": action,
                "candidate_alias": {
                    ("frame", "source_skip"): "V30_MASKLET_FRAME_SKIP",
                    ("global", "source_skip"): "V30_MASKLET_GLOBAL_SKIP",
                    ("swa", "swa_anchor_keep"): "V30_MASKLET_SWA_ANCHOR",
                    ("swa", "swa_remove"): "V30_MASKLET_SWA_REMOVE",
                    ("ttt", "ttt_positive"): "V30_MASKLET_TTT_POS",
                    ("ttt", "ttt_negative"): "V30_MASKLET_TTT_NEG",
                }.get((path, action), "V30_MASKLET_GLOBAL_SKIP"),
                "run_prefix": (
                    "V30_H4_WAVE1_R1_"
                    f"{cat}_m{int(common['masklet_id']):02d}_{path}_{action}"
                ),
            })
            if len(rows) >= max_rows:
                return rows
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-csv", default="results/kitti01_hmc_v2/acl2_v29c_semantickitti_download_projection_videomasklet/masklet_3d_alignment/masklet_alignment.csv")
    parser.add_argument("--out-dir", default="results/kitti01_hmc_v2/acl2_v30_semanticcue_causalmemorybank_target30/track_e_masklet_selection")
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--max-masklets", type=int, default=8)
    parser.add_argument("--max-interventions", type=int, default=16)
    args = parser.parse_args()

    rows = [row for row in _read_csv(Path(args.alignment_csv)) if int(_f(row, "chunk_idx", -999)) == int(args.chunk)]
    used: set[str] = set()
    selectors = [
        (
            "road_high_support",
            lambda r: r.get("fine_label_pred") == "road",
            lambda r: (_f(r, "projected_pixel_support_count"), _f(r, "t_mask")),
        ),
        (
            "structure_high_support",
            lambda r: r.get("fine_label_pred") in {"building", "wall", "fence", "sidewalk"},
            lambda r: (_f(r, "projected_pixel_support_count"), _f(r, "t_mask")),
        ),
        (
            "vegetation_high_support",
            lambda r: r.get("fine_label_pred") == "vegetation",
            lambda r: (_f(r, "projected_pixel_support_count"), _f(r, "area_mean")),
        ),
        (
            "grass_or_terrain",
            lambda r: r.get("fine_label_pred") == "grass" or r.get("projected_majority_semantic_name") == "terrain",
            lambda r: (_f(r, "projected_pixel_support_count"), _f(r, "area_mean")),
        ),
        (
            "movable_projected",
            lambda r: "car" in str(r.get("projected_majority_semantic_name", "")) or "person" in str(r.get("projected_majority_semantic_name", "")),
            lambda r: (_f(r, "projected_pixel_support_count"), _f(r, "t_mask")),
        ),
        (
            "disagreement",
            lambda r: _b(r, "agreement_pred_vs_projected") is False,
            lambda r: (_f(r, "projected_pixel_support_count"), _f(r, "q_3d")),
        ),
        (
            "low_trust",
            lambda r: _f(r, "projected_pixel_support_count") > 0,
            lambda r: (-_f(r, "t_mask"), _f(r, "projected_pixel_support_count")),
        ),
        (
            "large_area",
            lambda r: True,
            lambda r: (_f(r, "area_mean"), _f(r, "projected_pixel_support_count")),
        ),
        (
            "sky_or_low_projection",
            lambda r: r.get("fine_label_pred") == "sky" or _f(r, "projected_pixel_support_count") < 50,
            lambda r: (_f(r, "area_mean"), -_f(r, "projected_pixel_support_count")),
        ),
    ]

    selected: List[Dict[str, object]] = []
    for category, predicate, key in selectors:
        row = _choose(rows, category=category, predicate=predicate, key=key, used=used)
        if row is not None:
            selected.append(row)
        if len(selected) >= int(args.max_masklets):
            break

    interventions = _planned_interventions(selected, int(args.max_interventions))
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "selected_masklets.csv", [_common(row) for row in selected])
    _write_csv(out_dir / "planned_interventions.csv", interventions)
    summary = {
        "chunk": int(args.chunk),
        "available_alignment_rows": len(rows),
        "selected_masklets": len(selected),
        "planned_interventions": len(interventions),
        "selected_categories": [row.get("selection_category", "") for row in selected],
        "selection_gate_pass": bool(selected and interventions),
        "feature_gap": "Per-masklet D_g/conflict/scale/source-attention features were not present in v29C alignment; wave1 uses landed alignment/trust/projection proxies.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if summary["selection_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
