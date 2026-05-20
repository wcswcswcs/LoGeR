#!/usr/bin/env python3
"""Build ACL2 v19 drift-state autopsy tables from landed experiment artifacts.

The script is deliberately diagnostic-only: it reads existing boundary and
short-rollout metric CSV files, recomputes simple correlations/trade-off
summaries, and writes audit tables/figures. It does not alter trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: object) -> float:
    try:
        if value in ("", None, "nan", "NaN", "None"):
            return float("nan")
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, int]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.shape[0] < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan"), int(x.shape[0])
    xr = _rankdata(x)
    yr = _rankdata(y)
    return float(np.corrcoef(xr, yr)[0, 1]), int(x.shape[0])


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
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


def _json_default(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, rows: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _boundary_rows(path: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in _read_csv(path):
        run_id = row.get("run") or row.get("run_id") or row.get("Run") or ""
        if not run_id:
            continue
        out.append(
            {
                "row_type": "full_online_boundary",
                "run_id": run_id,
                "family": row.get("result_class", "online_boundary"),
                "counts_as_ttt_write": row.get("counts_as_ttt_write", ""),
                "diagnostic_only": row.get("diagnostic_only", "False"),
                "chunk_id": "",
                "horizon": "",
                "ATE": _to_float(row.get("ate_rmse")),
                "Rot": _to_float(row.get("rot_rmse")),
                "RPE_t": _to_float(row.get("rpe_t")),
                "RPE_r": _to_float(row.get("rpe_r")),
                "FinalErr": _to_float(row.get("final_err")),
                "YawRMSE": _to_float(row.get("yaw_rmse")),
                "Sim3Scale": _to_float(row.get("sim3_scale")),
                "seg_200_300": _to_float(row.get("seg_200_300")),
                "seg_400_600": _to_float(row.get("seg_400_600")),
                "ATE_delta_vs_H9": float("nan"),
                "seg_200_300_delta_vs_H9": float("nan"),
                "seg_400_600_delta_vs_H9": float("nan"),
                "action_signal_proxy": _to_float(row.get("memory_ttt_mean_rel_diff_max")),
                "source_path": row.get("source_path", ""),
            }
        )
    return out


def _candidate_rows(paths: Iterable[Path]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for path in paths:
        for row in _read_csv(path):
            if row.get("candidate_id") == "K1_H9":
                continue
            out.append(
                {
                    "row_type": "short_rollout_candidate",
                    "run_id": row.get("candidate_id", ""),
                    "family": row.get("family", "unknown"),
                    "counts_as_ttt_write": "False",
                    "diagnostic_only": row.get("diagnostic_only_short_rollout", "True"),
                    "chunk_id": row.get("chunk_id", ""),
                    "horizon": row.get("horizon", ""),
                    "ATE": _to_float(row.get("ATE_horizon")),
                    "Rot": _to_float(row.get("Rot_horizon")),
                    "RPE_t": float("nan"),
                    "RPE_r": float("nan"),
                    "FinalErr": _to_float(row.get("FinalErr_horizon")),
                    "YawRMSE": float("nan"),
                    "Sim3Scale": _to_float(row.get("alignment_scale")),
                    "seg_200_300": _to_float(row.get("intersection_200_300_ATE")),
                    "seg_400_600": _to_float(row.get("intersection_400_600_ATE")),
                    "ATE_delta_vs_H9": _to_float(row.get("ATE_delta_vs_H9")),
                    "seg_200_300_delta_vs_H9": _to_float(row.get("intersection_200_300_delta_vs_H9")),
                    "seg_400_600_delta_vs_H9": _to_float(row.get("intersection_400_600_delta_vs_H9")),
                    "action_signal_proxy": _to_float(row.get("raw_trans_max_diff")),
                    "raw_pose_max_abs_diff": _to_float(row.get("raw_pose_max_abs_diff")),
                    "alignment_scale_delta_vs_H9": _to_float(row.get("alignment_scale_delta_vs_H9")),
                    "counts_as_ttt_write_success": row.get("counts_as_ttt_write_success", "False"),
                    "source_path": str(path),
                    "run_dir": row.get("run_dir", ""),
                }
            )
    return out


def _plot_outputs(out_dir: Path, rows: List[Dict[str, object]], summary: Mapping[str, object]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        _write_json(out_dir / "plot_error.json", {"error": repr(exc)})
        return

    global_dir = out_dir / "global_drift_dashboard"
    action_dir = out_dir / "ttt_action_dashboard"
    global_dir.mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)

    boundary = [row for row in rows if row["row_type"] == "full_online_boundary"]
    if boundary:
        names = [str(row["run_id"]) for row in boundary]
        x = np.arange(len(names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - width / 2, [_to_float(row["seg_200_300"]) for row in boundary], width, label="[200,300)")
        ax.bar(x + width / 2, [_to_float(row["seg_400_600"]) for row in boundary], width, label="[400,600)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("ATE RMSE (m)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(global_dir / "boundary_segment_ate_H9_C9_WINGAM.png", dpi=160)
        plt.close(fig)

    candidates = [row for row in rows if row["row_type"] == "short_rollout_candidate"]
    finite_trade = [
        row
        for row in candidates
        if math.isfinite(_to_float(row["seg_200_300_delta_vs_H9"]))
        and math.isfinite(_to_float(row["seg_400_600_delta_vs_H9"]))
    ]
    if finite_trade:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            [_to_float(row["seg_200_300_delta_vs_H9"]) for row in finite_trade],
            [_to_float(row["seg_400_600_delta_vs_H9"]) for row in finite_trade],
            s=18,
            alpha=0.75,
        )
        ax.axvline(-5.0, color="tab:green", linestyle="--", linewidth=1)
        ax.axhline(1.0, color="tab:red", linestyle="--", linewidth=1)
        ax.set_xlabel("[200,300) delta vs H9 (m)")
        ax.set_ylabel("[400,600) delta vs H9 (m)")
        ax.set_title("Segment Trade-off")
        fig.tight_layout()
        fig.savefig(global_dir / "segment_200_300_vs_400_600_tradeoff.png", dpi=160)
        plt.close(fig)

    h10 = [row for row in candidates if str(row.get("horizon")) in {"10", "15"} and math.isfinite(_to_float(row["ATE_delta_vs_H9"]))]
    if h10:
        best_by_family: Dict[str, float] = {}
        for row in h10:
            family = str(row["family"])
            value = _to_float(row["ATE_delta_vs_H9"])
            best_by_family[family] = min(best_by_family.get(family, float("inf")), value)
        labels = sorted(best_by_family)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(labels, [best_by_family[label] for label in labels])
        ax.axhline(-3.0, color="tab:green", linestyle="--", linewidth=1)
        ax.axhline(-1.0, color="tab:orange", linestyle=":", linewidth=1)
        ax.set_ylabel("Best h10/h15 ATE delta vs H9 (m)")
        ax.set_xticklabels(labels, rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(action_dir / "best_h10_h15_delta_by_family.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            [_to_float(row["action_signal_proxy"]) for row in h10],
            [_to_float(row["ATE_delta_vs_H9"]) for row in h10],
            s=18,
            alpha=0.75,
        )
        ax.set_xlabel("Action signal proxy: raw translation diff")
        ax.set_ylabel("ATE delta vs H9 (m)")
        ax.set_title(f"Spearman={summary.get('corr_action_proxy_vs_ATE_delta')}")
        fig.tight_layout()
        fig.savefig(action_dir / "action_proxy_vs_ate_delta_scatter.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--boundary-csv", action="append", default=[])
    parser.add_argument("--candidate-csv", action="append", default=[])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows: List[Dict[str, object]] = []
    for path in args.boundary_csv:
        rows.extend(_boundary_rows(Path(path)))
    rows.extend(_candidate_rows(Path(path) for path in args.candidate_csv))

    candidate_rows = [row for row in rows if row["row_type"] == "short_rollout_candidate"]
    h10_rows = [row for row in candidate_rows if str(row.get("horizon")) in {"10", "15"}]
    best_ate = min(
        (_to_float(row["ATE_delta_vs_H9"]) for row in h10_rows if math.isfinite(_to_float(row["ATE_delta_vs_H9"]))),
        default=float("nan"),
    )
    best_seg = min(
        (
            _to_float(row["seg_200_300_delta_vs_H9"])
            for row in h10_rows
            if math.isfinite(_to_float(row["seg_200_300_delta_vs_H9"]))
        ),
        default=float("nan"),
    )
    gate_like = any(
        (
            _to_float(row["ATE_delta_vs_H9"]) <= -3.0
            or _to_float(row["seg_200_300_delta_vs_H9"]) <= -5.0
        )
        and (
            not math.isfinite(_to_float(row["seg_400_600_delta_vs_H9"]))
            or _to_float(row["seg_400_600_delta_vs_H9"]) <= 1.0
        )
        for row in h10_rows
    )
    corr_action_ate, n_action_ate = _spearman(
        [_to_float(row["action_signal_proxy"]) for row in candidate_rows],
        [_to_float(row["ATE_delta_vs_H9"]) for row in candidate_rows],
    )
    corr_action_rot, n_action_rot = _spearman(
        [_to_float(row["action_signal_proxy"]) for row in candidate_rows],
        [_to_float(row["Rot"]) for row in candidate_rows],
    )
    corr_action_final, n_action_final = _spearman(
        [_to_float(row["action_signal_proxy"]) for row in candidate_rows],
        [_to_float(row["FinalErr"]) for row in candidate_rows],
    )
    corr_tradeoff, n_tradeoff = _spearman(
        [_to_float(row["seg_200_300_delta_vs_H9"]) for row in candidate_rows],
        [_to_float(row["seg_400_600_delta_vs_H9"]) for row in candidate_rows],
    )
    summary: Dict[str, object] = {
        "row_count": len(rows),
        "candidate_row_count": len(candidate_rows),
        "h10_h15_candidate_row_count": len(h10_rows),
        "best_h10_h15_ATE_delta_vs_H9": best_ate,
        "best_h10_h15_200_300_delta_vs_H9": best_seg,
        "existing_action_gate_like_pass": gate_like,
        "corr_action_proxy_vs_ATE_delta": corr_action_ate,
        "corr_action_proxy_vs_ATE_delta_n": n_action_ate,
        "corr_action_proxy_vs_Rot": corr_action_rot,
        "corr_action_proxy_vs_Rot_n": n_action_rot,
        "corr_action_proxy_vs_FinalErr": corr_action_final,
        "corr_action_proxy_vs_FinalErr_n": n_action_final,
        "corr_200_300_delta_vs_400_600_delta": corr_tradeoff,
        "corr_200_300_delta_vs_400_600_delta_n": n_tradeoff,
        "diagnostic_only": True,
    }

    _write_csv(out_dir / "drift_state_autopsy_rows.csv", rows)
    _write_json(out_dir / "drift_state_autopsy_rows.json", rows)
    _write_json(out_dir / "drift_state_autopsy_summary.json", summary)
    _plot_outputs(out_dir, rows, summary)

    lines = [
        "# ACL2 v19 Drift-State Autopsy",
        "",
        "This is an offline diagnostic over landed artifacts only.",
        "",
        f"- Candidate rows: `{summary['candidate_row_count']}`",
        f"- h10/h15 candidate rows: `{summary['h10_h15_candidate_row_count']}`",
        f"- Best h10/h15 ATE delta vs H9: `{summary['best_h10_h15_ATE_delta_vs_H9']}`",
        f"- Best h10/h15 [200,300) delta vs H9: `{summary['best_h10_h15_200_300_delta_vs_H9']}`",
        f"- Any existing action satisfies v19 gate-like rule: `{str(summary['existing_action_gate_like_pass']).lower()}`",
        f"- Spearman(action proxy, ATE delta): `{summary['corr_action_proxy_vs_ATE_delta']}` n=`{summary['corr_action_proxy_vs_ATE_delta_n']}`",
        f"- Spearman(action proxy, Rot): `{summary['corr_action_proxy_vs_Rot']}` n=`{summary['corr_action_proxy_vs_Rot_n']}`",
        f"- Spearman(action proxy, FinalErr): `{summary['corr_action_proxy_vs_FinalErr']}` n=`{summary['corr_action_proxy_vs_FinalErr_n']}`",
        f"- Spearman([200,300) delta, [400,600) delta): `{summary['corr_200_300_delta_vs_400_600_delta']}` n=`{summary['corr_200_300_delta_vs_400_600_delta_n']}`",
        "",
        "No row in this autopsy counts as deployable success.",
        "",
    ]
    (out_dir / "drift_state_autopsy.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
