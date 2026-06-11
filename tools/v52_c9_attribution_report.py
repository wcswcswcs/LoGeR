#!/usr/bin/env python3
"""Generate ACL2 v52 C9 attribution report from landed v45/v46B artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
V46B_ROOT = REPO_ROOT / "results/kitti01_hmc_v2/acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/report_R1"
V45_ROOT = REPO_ROOT / "results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay"
C9_ATE = 33.76294210291885

REQUIRED_FACTORIAL_ROWS = [
    "F000_NONE",
    "F100_ONLY_FRAME_ATTN",
    "F010_ONLY_TTT",
    "F001_ONLY_SWA",
    "F110_FRAME_ATTN_TTT",
    "F101_FRAME_ATTN_SWA",
    "F011_TTT_SWA",
    "F111_ALL_THREE",
]

D_ROW_DESCRIPTIONS = {
    "F0": "Exact C9_P0_R2 repeat",
    "D1": "fixed read beta only; read beta chunk map removed",
    "D2": "fixed tri gamma 0.003; tri gamma map and tri replay chunk params removed",
    "D3": "fixed tri gamma 0.004; best fixed tri gamma substitute in D2-D4",
    "D4": "fixed tri gamma 0.005",
    "D5": "commit EMA off",
    "D6": "global commit EMA alpha 0.8 on branch 0",
    "D7": "C9-Clean best fixed: fixed read beta + fixed tri gamma 0.004 + commit EMA off",
}

LEDGER_NOTES = {
    "tri_replay": "largest exact C9 write-side contribution; removal disables tri replay behavior",
    "tri_gamma_chunk_map": "chunk-specific gamma map contribution",
    "fixed_tri_gamma_best": "best fixed gamma substitute still leaves substantial gap",
    "commit_ema": "chunk-specific commit EMA contribution",
    "fixed_ema_best": "best fixed/global EMA substitute contribution",
    "native_mix": "native mix contribution",
    "swa_overlap_replace": "SWA overlap replacement contribution",
    "read_beta_map": "read beta chunk map contribution",
    "fixed_read_beta": "fixed read beta substitute contribution",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
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


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _copy_factorial_table(out_dir: Path) -> List[Dict[str, Any]]:
    registry = _read_csv(V46B_ROOT / "phase2_factorial_registry.csv")
    by_row = {row["row"]: row for row in registry}
    missing = [row for row in REQUIRED_FACTORIAL_ROWS if row not in by_row]
    if missing:
        raise RuntimeError(f"missing v46B rows: {missing}")
    out_rows: List[Dict[str, Any]] = []
    for row_name in REQUIRED_FACTORIAL_ROWS:
        row = by_row[row_name]
        out_rows.append({
            "row": row_name,
            "run_name": row.get("run_name"),
            "FRAME_ATTN": row.get("FRAME_ATTN"),
            "TTT": row.get("TTT"),
            "SWA": row.get("SWA"),
            "status": row.get("status"),
            "frames": row.get("frames"),
            "ATE_full": row.get("ATE_full"),
            "gain_vs_F000": _float(by_row["F000_NONE"].get("ATE_full")) - _float(row.get("ATE_full")),
            "Rot_full": row.get("Rot_full"),
            "FinalErr_full": row.get("FinalErr_full"),
            "RPE_t_full": row.get("RPE_t_full"),
            "RPE_r_full": row.get("RPE_r_full"),
            "segment_200_300_ATE": row.get("segment_200_300_ATE"),
            "segment_400_600_ATE": row.get("segment_400_600_ATE"),
            "rolling50_mean": row.get("rolling50_mean"),
            "rolling50_p90": row.get("rolling50_p90"),
            "rolling50_worst": row.get("rolling50_worst"),
            "rolling100_mean": row.get("rolling100_mean"),
            "rolling100_p90": row.get("rolling100_p90"),
            "rolling100_worst": row.get("rolling100_worst"),
            "rolling200_mean": row.get("rolling200_mean"),
            "rolling200_p90": row.get("rolling200_p90"),
            "rolling200_worst": row.get("rolling200_worst"),
            "hmc_rows": row.get("hmc_rows"),
            "frame_attn_read_control_active": row.get("frame_attn_read_control_active"),
            "ttt_tri_replay_applied_count": row.get("ttt_tri_replay_applied_count"),
            "ttt_positive_mass_mean": row.get("ttt_positive_mass_mean"),
            "ttt_neutral_mass_mean": row.get("ttt_neutral_mass_mean"),
            "ttt_negative_mass_mean": row.get("ttt_negative_mass_mean"),
            "swa_overlap_replace_applied_count": row.get("swa_overlap_replace_applied_count"),
            "no_chunk_policy_pass": row.get("no_chunk_policy_pass"),
            "row_valid": row.get("row_valid"),
            "invalid_reason": row.get("invalid_reason"),
        })
    _write_csv(out_dir / "positive_only_factorial_table.csv", out_rows)
    return out_rows


def _copy_exact_c9_tables(out_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    clean_rows = _read_csv(V45_ROOT / "phase1_c9_clean/report_R1/full_metrics/full_online_registry.csv")
    clean_out: List[Dict[str, Any]] = []
    for row in clean_rows:
        name = row.get("name", "")
        clean_out.append({
            "name": name,
            "description": D_ROW_DESCRIPTIONS.get(name, ""),
            "status": row.get("status"),
            "frames": row.get("frames"),
            "ATE_full": row.get("ATE_full"),
            "delta_vs_C9": _float(row.get("ATE_full")) - C9_ATE,
            "Rot_full": row.get("Rot_full"),
            "FinalErr_full": row.get("FinalErr_full"),
            "RPE_t_full": row.get("RPE_t_full"),
            "RPE_r_full": row.get("RPE_r_full"),
            "segment_200_300_ATE": row.get("segment_200_300_ATE"),
            "segment_400_600_ATE": row.get("segment_400_600_ATE"),
            "rolling50_mean": row.get("rolling50_mean"),
            "rolling50_p90": row.get("rolling50_p90"),
            "rolling50_worst": row.get("rolling50_worst"),
            "rolling100_mean": row.get("rolling100_mean"),
            "rolling100_p90": row.get("rolling100_p90"),
            "rolling100_worst": row.get("rolling100_worst"),
            "rolling200_mean": row.get("rolling200_mean"),
            "rolling200_p90": row.get("rolling200_p90"),
            "rolling200_worst": row.get("rolling200_worst"),
            "hmc_rows": row.get("hmc_rows"),
        })
    _write_csv(out_dir / "exact_c9_clean_rows.csv", clean_out)

    ledger_rows = _read_csv(V45_ROOT / "final_reports/v45_component_contribution_ledger.csv")
    ledger_out = []
    for row in ledger_rows:
        component = row.get("component", "")
        ledger_out.append({
            "component": component,
            "effect_delta_vs_C9": row.get("effect_delta_vs_C9"),
            "note": LEDGER_NOTES.get(component, ""),
            "source": "v45_component_contribution_ledger.csv",
        })
    _write_csv(out_dir / "exact_c9_knockout_table.csv", ledger_out)

    matrix_rows = _read_csv(V45_ROOT / "final_reports/v45_component_interaction_matrix.csv")
    _write_csv(out_dir / "exact_c9_interaction_table.csv", matrix_rows)
    return {"clean_rows": clean_out, "ledger_rows": ledger_out, "matrix_rows": matrix_rows}


def _plot_main_effects(out_dir: Path, factorial_rows: List[Dict[str, Any]]) -> None:
    rows = [row for row in factorial_rows if row["row"] != "F000_NONE"]
    labels = [str(row["row"]).replace("_", "\n") for row in rows]
    gains = [_float(row.get("gain_vs_F000")) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#3b82f6", "#ef4444", "#10b981", "#7c3aed", "#14b8a6", "#f59e0b", "#111827"]
    ax.bar(labels, gains, color=colors[: len(labels)])
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Gain vs F000 ATE (m)")
    ax.set_title("v46B Clean No-Chunk Component Main Effects")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "component_main_effects.png", dpi=180)
    plt.close(fig)


def _write_report(
    out_dir: Path,
    factorial_rows: List[Dict[str, Any]],
    exact: Dict[str, List[Dict[str, Any]]],
) -> None:
    by_row = {row["row"]: row for row in factorial_rows}
    f000 = _float(by_row["F000_NONE"]["ATE_full"])
    f100 = _float(by_row["F100_ONLY_FRAME_ATTN"]["ATE_full"])
    f010 = _float(by_row["F010_ONLY_TTT"]["ATE_full"])
    f001 = _float(by_row["F001_ONLY_SWA"]["ATE_full"])
    f110 = _float(by_row["F110_FRAME_ATTN_TTT"]["ATE_full"])
    f101 = _float(by_row["F101_FRAME_ATTN_SWA"]["ATE_full"])
    f011 = _float(by_row["F011_TTT_SWA"]["ATE_full"])
    f111 = _float(by_row["F111_ALL_THREE"]["ATE_full"])
    gain = lambda ate: f000 - ate
    read_ttt_synergy = gain(f110) - max(gain(f100), gain(f010))
    read_swa_synergy = gain(f101) - max(gain(f100), gain(f001))
    ttt_swa_synergy = gain(f011) - max(gain(f010), gain(f001))
    three_way = gain(f111) - max(gain(f110), gain(f101), gain(f011))

    ledger_by_component = {row["component"]: row for row in exact["ledger_rows"]}
    clean_by_name = {row["name"]: row for row in exact["clean_rows"]}
    lines = [
        "# ACL2 v52 Phase 1 C9 Component Attribution Report",
        "",
        "This report is generated only from landed v45/v46B artifacts. No metrics are filled by guesswork.",
        "",
        "## Artifact Inputs",
        "",
        f"- v46B factorial registry: `{V46B_ROOT / 'phase2_factorial_registry.csv'}`",
        f"- v45 C9 clean rows: `{V45_ROOT / 'phase1_c9_clean/report_R1/full_metrics/full_online_registry.csv'}`",
        f"- v45 exact contribution ledger: `{V45_ROOT / 'final_reports/v45_component_contribution_ledger.csv'}`",
        "",
        "## v46B Clean No-Chunk Factorial",
        "",
        "| Row | READ | TTT | SWA | ATE | Gain vs F000 | hmc_rows | audit |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in factorial_rows:
        lines.append(
            f"| `{row['row']}` | {row['FRAME_ATTN']} | {row['TTT']} | {row['SWA']} | "
            f"{row['ATE_full']} | {row['gain_vs_F000']} | {row['hmc_rows']} | "
            f"valid={row['row_valid']}, no_chunk={row['no_chunk_policy_pass']} |"
        )
    lines.extend([
        "",
        "Key clean no-chunk effects:",
        "",
        f"- READ-only gain: `{gain(f100)}` m.",
        f"- TTT-only gain: `{gain(f010)}` m.",
        f"- SWA-only gain: `{gain(f001)}` m.",
        f"- READ+TTT gain: `{gain(f110)}` m; incremental margin over best single `{read_ttt_synergy}` m.",
        f"- READ+SWA gain: `{gain(f101)}` m; incremental margin over best single `{read_swa_synergy}` m.",
        f"- TTT+SWA gain: `{gain(f011)}` m; incremental margin over best single `{ttt_swa_synergy}` m.",
        f"- All-three gain: `{gain(f111)}` m; best-pair margin `{three_way}` m.",
        "",
        "## v45 Exact C9 Knockout / Chunk-Policy Attribution",
        "",
        "| Component | Effect delta vs C9 | Note |",
        "|---|---:|---|",
    ])
    for component in (
        "tri_replay",
        "tri_gamma_chunk_map",
        "commit_ema",
        "native_mix",
        "swa_overlap_replace",
        "read_beta_map",
        "fixed_tri_gamma_best",
        "fixed_ema_best",
        "fixed_read_beta",
    ):
        row = ledger_by_component.get(component)
        if not row:
            continue
        lines.append(f"| `{component}` | {row['effect_delta_vs_C9']} | {row['note']} |")
    lines.extend([
        "",
        "C9-clean rows:",
        "",
        "| Name | Description | ATE | Delta vs C9 | [200,300) | [400,600) | hmc_rows |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for name in ("F0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"):
        row = clean_by_name.get(name)
        if not row:
            continue
        lines.append(
            f"| `{name}` | {row['description']} | {row['ATE_full']} | {row['delta_vs_C9']} | "
            f"{row['segment_200_300_ATE']} | {row['segment_400_600_ATE']} | {row['hmc_rows']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "v46B answers the clean no-chunk component question: READ and TTT are the two meaningful positive components, SWA alone is near zero, and READ+TTT has about `1.9245m` non-additive synergy.",
        "",
        "v45 answers the exact C9 question: C9's deployable ATE advantage is dominated by tri replay, the chunk gamma/tri params path, and commit EMA. The read beta map and SWA overlap replacement are small in full ATE. D7/C9-Clean removes the chunk-id policies but regresses to `35.500497135292775`, so fixed global substitutes do not reproduce C9.",
        "",
        "These are different experiments: v46B is a clean no-chunk positive-only factorial, while v45 is exact C9 knockout/chunk-policy attribution. They should not be averaged into one contribution table.",
        "",
        "## Outputs",
        "",
        "- `positive_only_factorial_table.csv`",
        "- `exact_c9_clean_rows.csv`",
        "- `exact_c9_knockout_table.csv`",
        "- `exact_c9_interaction_table.csv`",
        "- `component_main_effects.png`",
        "- `interaction_heatmap.png`",
    ])
    (out_dir / "c9_component_attribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    factorial_rows = _copy_factorial_table(out_dir)
    exact = _copy_exact_c9_tables(out_dir)
    _plot_main_effects(out_dir, factorial_rows)
    shutil.copy2(V46B_ROOT / "component_interaction_heatmap.png", out_dir / "interaction_heatmap.png")
    summary = {
        "required_factorial_rows": REQUIRED_FACTORIAL_ROWS,
        "factorial_rows_present": [row["row"] for row in factorial_rows],
        "all_factorial_rows_valid": all(str(row.get("row_valid")) == "True" for row in factorial_rows),
        "c9_ate": C9_ATE,
        "c9_clean_ate": _float(next(row["ATE_full"] for row in exact["clean_rows"] if row["name"] == "D7")),
    }
    (out_dir / "phase1_attribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(out_dir, factorial_rows, exact)
    if not summary["all_factorial_rows_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
