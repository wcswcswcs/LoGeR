#!/usr/bin/env python3
"""Track A action distinguishability audit for v30 planned interventions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


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


def _masklet_key(row: Dict[str, str]) -> str:
    return f"{int(_f(row, 'chunk_idx', -1))}:{int(_f(row, 'masklet_id', -1))}"


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-masklets-csv", required=True)
    parser.add_argument("--planned-interventions-csv", required=True)
    parser.add_argument("--out-dir", default="results/kitti01_hmc_v2/acl2_v30_semanticcue_causalmemorybank_target30/track_a_action_audit")
    args = parser.parse_args()

    selected = _read_csv(Path(args.selected_masklets_csv))
    planned = _read_csv(Path(args.planned_interventions_csv))
    total = max(1, len(selected))
    all_keys = {_masklet_key(row) for row in selected}
    action_sets: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    summary_rows: List[Dict[str, object]] = []
    for row in planned:
        key = _masklet_key(row)
        path = row.get("intervention_path", "")
        action = row.get("intervention_action", "")
        action_sets[(path, action)].add(key)
    for (path, action), keys in sorted(action_sets.items()):
        summary_rows.append({
            "path": path,
            "action": action,
            "masklet_count": len(keys),
            "masklet_ratio": len(keys) / total,
            "source_keep_ratio": len(keys & all_keys) / total if path in {"frame", "global"} and action == "source_keep" else "",
            "source_drop_ratio": len(keys & all_keys) / total if path in {"frame", "global"} and "skip" in action else "",
            "swa_action_mass": len(keys & all_keys) / total if path == "swa" else "",
            "ttt_role_mass": len(keys & all_keys) / total if path == "ttt" else "",
        })

    pairs: List[Dict[str, object]] = []
    items = sorted(action_sets.items())
    gate_pass = False
    for i, ((path_a, action_a), set_a) in enumerate(items):
        for path_b, action_b in [k for k, _ in items[i + 1:]]:
            set_b = action_sets[(path_b, action_b)]
            jac = _jaccard(set_a, set_b)
            ratio_diff = abs(len(set_a) / total - len(set_b) / total)
            path_same = path_a == path_b
            row = {
                "left_path": path_a,
                "left_action": action_a,
                "right_path": path_b,
                "right_action": action_b,
                "same_path": path_same,
                "left_count": len(set_a),
                "right_count": len(set_b),
                "jaccard": jac,
                "ratio_difference": ratio_diff,
                "action_gate_pass": bool(jac <= 0.85 or ratio_diff >= 0.05),
            }
            if path_same:
                gate_pass = gate_pass or bool(row["action_gate_pass"])
            pairs.append(row)

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "action_tensor_summary.csv", summary_rows)
    _write_csv(out_dir / "action_jaccard_matrix.csv", pairs)
    # Compatibility outputs requested by the v30 plan. These are offline action
    # masks only; runtime attention/SWA/TTT effects are filled by rollout logs.
    _write_csv(out_dir / "per_path_source_keep_ratio.csv", summary_rows)
    _write_csv(out_dir / "per_label_role_mass.csv", [
        {
            "fine_label_pred": row.get("fine_label_pred", ""),
            "selection_category": row.get("selection_category", ""),
            "planned_actions": ";".join(
                f"{p.get('intervention_path')}:{p.get('intervention_action')}"
                for p in planned
                if _masklet_key(p) == _masklet_key(row)
            ),
        }
        for row in selected
    ])
    _write_csv(out_dir / "attention_mass_removed.csv", [{"status": "runtime_attention_mass_removed_not_available_offline"}])
    _write_csv(out_dir / "swa_cache_removed_mass.csv", [{"status": "runtime_swa_cache_mass_removed_not_available_offline"}])
    _write_csv(out_dir / "ttt_role_mass.csv", summary_rows)
    summary = {
        "track": "v30_track_a_action_distinguishability",
        "selected_masklets": len(selected),
        "planned_interventions": len(planned),
        "same_path_action_distinguishability_gate_pass": gate_pass,
        "gate_rule": "For same-path planned actions, Jaccard <= 0.85 or ratio difference >= 0.05.",
        "boundary": "Offline action distinguishability only; runtime hook collapse must be checked after rollout.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "action_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if planned else 1


if __name__ == "__main__":
    raise SystemExit(main())
