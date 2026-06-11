#!/usr/bin/env python3
"""Augment v38 Track 0 action/influence atlas outputs.

The upstream atlas intentionally records missing spatial/per-label tensor
granularity as explicit missing evidence. This postprocess creates the v38
file names requested by the plan without reconstructing absent data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _write_bar(path: Path, labels: List[str], series: Dict[str, List[float]], title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 0.8 / max(len(series), 1)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(labels) + 3), 4.8))
    for idx, (name, values) in enumerate(series.items()):
        ax.bar(x + idx * width, values, width=width, label=name)
    ax.set_xticks(x + width * (max(len(series), 1) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _write_heatmap(path: Path, row_labels: List[str], col_labels: List[str], matrix: List[List[float]], title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(col_labels) + 3), max(4, 0.32 * len(row_labels) + 2)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", required=True, type=Path)
    args = parser.parse_args()
    atlas_dir = args.atlas_dir

    per_path = _read_csv(atlas_dir / "semantic_path_action_influence.csv")
    atlas = _read_csv(atlas_dir / "semantic_influence_atlas.csv")
    summary_path = atlas_dir / "phase0_action_influence_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    removed_rows: List[Dict[str, Any]] = []
    for row in per_path:
        removed_rows.append({
            "parent": row.get("parent"),
            "chunk": row.get("chunk"),
            "candidate": row.get("candidate"),
            "path": row.get("path"),
            "attention_mass_available": row.get("attention_mass_available"),
            "attention_mass_removed_before": row.get("attention_mass_removed_before"),
            "attention_mass_removed_after": row.get("attention_mass_removed_after"),
            "attention_mass_retained_before": row.get("attention_mass_retained_before"),
            "attention_mass_retained_after": row.get("attention_mass_retained_after"),
            "evidence_status": row.get("explainability_status") or "landed_summary_only",
        })
    _write_csv(atlas_dir / "attention_mass_removed_before_after.csv", removed_rows)

    missing_files = [
        "frame_source_keep_ratio_by_label.csv",
        "global_source_keep_ratio_by_label.csv",
        "swa_overlap_keep_ratio_by_label.csv",
        "swa_nonoverlap_keep_ratio_by_label.csv",
        "ttt_role_mass_by_label.csv",
        "source_attention_mass_by_label.csv",
        "swa_source_attention_mass_by_label.csv",
        "ttt_post_zp_update_norm_by_label.csv",
    ]
    missing_row = {
        "status": "explainability_missing",
        "reason": "v38 rollout artifacts did not land per-label/per-masklet tensor summaries for this field; aggregate path summaries are available in semantic_path_action_influence.csv",
    }
    for name in missing_files:
        _write_csv(atlas_dir / name, [missing_row])

    by_candidate: Dict[str, List[Dict[str, str]]] = {}
    for row in per_path:
        by_candidate.setdefault(str(row.get("candidate")), []).append(row)
    labels = sorted(by_candidate)
    source_series = {
        "frame_removed_before": [
            _mean(_float(r.get("attention_mass_removed_before"), float("nan")) for r in by_candidate[c] if r.get("path") == "frame_attention")
            for c in labels
        ],
        "global_removed_before": [
            _mean(_float(r.get("attention_mass_removed_before"), float("nan")) for r in by_candidate[c] if r.get("path") == "chunk_attention")
            for c in labels
        ],
    }
    source_bar = _write_bar(
        atlas_dir / "source_attention_mass_removed_bar.png",
        labels,
        source_series,
        "v38 Track0 source attention mass removed before compaction",
    )
    swa_bar = _write_bar(
        atlas_dir / "swa_overlap_nonoverlap_keep_bar.png",
        labels,
        {
            "swa_keep_ratio": [
                _mean(_float(r.get("keep_ratio"), float("nan")) for r in by_candidate[c] if r.get("path") == "swa_read")
                for c in labels
            ],
            "swa_negative_role_mass": [
                _mean(_float(r.get("negative_role_mass"), float("nan")) for r in by_candidate[c] if r.get("path") == "swa_read")
                for c in labels
            ],
        },
        "v38 Track0 SWA aggregate keep / negative-role proxy",
    )
    ttt_bar = _write_bar(
        atlas_dir / "ttt_role_mass_by_label_bar.png",
        labels,
        {
            "ttt_positive_role_mass": [
                _mean(_float(r.get("positive_role_mass"), float("nan")) for r in by_candidate[c] if r.get("path") == "ttt_apply")
                for c in labels
            ],
            "ttt_negative_role_mass": [
                _mean(_float(r.get("negative_role_mass"), float("nan")) for r in by_candidate[c] if r.get("path") == "ttt_apply")
                for c in labels
            ],
            "ttt_no_write_role_mass": [
                _mean(_float(r.get("no_write_role_mass"), float("nan")) for r in by_candidate[c] if r.get("path") == "ttt_apply")
                for c in labels
            ],
        },
        "v38 Track0 TTT aggregate role-mass proxy",
    )

    chunks = sorted({str(row.get("chunk")) for row in atlas})
    candidates = sorted({str(row.get("candidate")) for row in atlas})
    matrix: List[List[float]] = []
    for chunk in chunks:
        row_vals: List[float] = []
        for candidate in candidates:
            vals = [
                _float(row.get("influence_max"), 0.0)
                for row in atlas
                if str(row.get("chunk")) == chunk and str(row.get("candidate")) == candidate
            ]
            row_vals.append(_mean(vals))
        matrix.append(row_vals)
    grid = _write_heatmap(
        atlas_dir / "influence_atlas_by_chunk.png",
        [f"chunk{c}" for c in chunks],
        candidates,
        matrix,
        "v38 Track0 influence atlas by chunk",
    )

    post_summary = {
        "attention_mass_removed_before_after_rows": len(removed_rows),
        "per_label_files_status": "explainability_missing_when_not_landed",
        "source_attention_mass_removed_bar": bool(source_bar),
        "swa_overlap_nonoverlap_keep_bar": bool(swa_bar),
        "ttt_role_mass_by_label_bar": bool(ttt_bar),
        "influence_atlas_by_chunk": bool(grid),
        "source_summary": summary,
    }
    _write_json(atlas_dir / "v38_action_influence_postprocess_summary.json", post_summary)
    print(json.dumps(post_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

