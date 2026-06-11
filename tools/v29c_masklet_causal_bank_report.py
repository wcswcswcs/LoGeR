#!/usr/bin/env python3
"""Summarize v29C masklet causal-bank rollout rows.

The input deltas come from the normal candidate-bank report. This tool only
reformats landed trajectory metrics and runtime intervention metadata; it does
not infer or fabricate missing ATE rows.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _f(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _interventions(run_dir: Path) -> List[Dict[str, object]]:
    path = run_dir / "hmc_state_hash.jsonl"
    out: List[Dict[str, object]] = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = row.get("v29c_masklet_intervention")
            if isinstance(meta, dict) and meta.get("enabled"):
                out.append(meta)
    return out


def _mode(values: List[object]) -> object:
    if not values:
        return ""
    counter = Counter(str(v) for v in values)
    return counter.most_common(1)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-delta-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    delta_rows = _read_csv(Path(args.candidate_delta_csv))
    causal_rows: List[Dict[str, object]] = []
    for row in delta_rows:
        run_dir = Path(str(row.get("run_dir", "")))
        metas = _interventions(run_dir)
        ate_delta = _f(row.get("ATE_delta_vs_H9"))
        seg_delta = _f(row.get("intersection_200_300_delta_vs_H9"))
        downstream_delta = _f(row.get("intersection_400_600_delta_vs_H9"))
        token_counts = [_f(m.get("token_count")) for m in metas]
        support_counts = [_f(m.get("projected_pixel_support_count")) for m in metas]
        causal_rows.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "horizon": row.get("horizon", ""),
                "path": _mode([m.get("path", "") for m in metas]),
                "action": _mode([m.get("action", "") for m in metas]),
                "selected_masklet_count": len(metas),
                "dominant_fine_label": _mode([m.get("fine_label_pred", "") for m in metas]),
                "dominant_projected_label": _mode([m.get("projected_majority_semantic_name", "") for m in metas]),
                "mean_selected_token_count": sum(token_counts) / max(len(token_counts), 1),
                "mean_projected_support_count": sum(support_counts) / max(len(support_counts), 1),
                "ATE_delta_vs_H9": ate_delta,
                "ATE_causal_effect_base_minus_intervention": -ate_delta,
                "intersection_200_300_delta_vs_H9": seg_delta,
                "intersection_200_300_effect": -seg_delta,
                "intersection_400_600_delta_vs_H9": downstream_delta,
                "raw_trans_max_diff": _f(row.get("raw_trans_max_diff")),
                "oracle_gate_pass": bool((-ate_delta) >= 3.0 or (-seg_delta) >= 5.0),
                "run_dir": str(run_dir),
            }
        )

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "causal_effects.csv", causal_rows)
    by_label: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    by_path: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in causal_rows:
        by_label[str(row.get("dominant_fine_label", ""))].append(row)
        by_path[str(row.get("path", ""))].append(row)

    label_rows = []
    for label, rows in sorted(by_label.items()):
        label_rows.append({
            "dominant_fine_label": label,
            "rows": len(rows),
            "best_ATE_effect": max(_f(r.get("ATE_causal_effect_base_minus_intervention")) for r in rows),
            "best_200_300_effect": max(_f(r.get("intersection_200_300_effect")) for r in rows),
        })
    path_rows = []
    for path, rows in sorted(by_path.items()):
        path_rows.append({
            "path": path,
            "rows": len(rows),
            "best_ATE_effect": max(_f(r.get("ATE_causal_effect_base_minus_intervention")) for r in rows),
            "best_200_300_effect": max(_f(r.get("intersection_200_300_effect")) for r in rows),
        })
    _write_csv(out_dir / "causal_effects_by_label.csv", label_rows)
    _write_csv(out_dir / "causal_effects_by_path.csv", path_rows)
    positives = sorted(causal_rows, key=lambda r: _f(r.get("ATE_causal_effect_base_minus_intervention")), reverse=True)
    negatives = sorted(causal_rows, key=lambda r: _f(r.get("ATE_causal_effect_base_minus_intervention")))
    _write_csv(out_dir / "top_positive_masklets.csv", positives[:10])
    _write_csv(out_dir / "top_negative_masklets.csv", negatives[:10])
    summary = {
        "rows": len(causal_rows),
        "oracle_gate_pass": any(bool(r.get("oracle_gate_pass")) for r in causal_rows),
        "best_ATE_effect": max((_f(r.get("ATE_causal_effect_base_minus_intervention")) for r in causal_rows), default=0.0),
        "best_200_300_effect": max((_f(r.get("intersection_200_300_effect")) for r in causal_rows), default=0.0),
        "counts_as_online_ttt_write_success": False,
        "note": "Short causal-bank rollout summary; gate uses landed trajectory deltas only.",
    }
    (out_dir / "causal_bank_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
