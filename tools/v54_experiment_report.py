#!/usr/bin/env python3
"""Generate ACL2 v54 state-conditioned adaptive TTT reports from artifacts."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v47_adaptive_ttt_writer_report import _walk  # noqa: E402
from tools.v53_experiment_report import (  # noqa: E402
    C9_P0_ATE,
    V52_AUTOPSY_DIR,
    _fmt,
    _iter_run_dirs,
    _plot_no_data,
    _read_csv,
    _read_jsonl,
    _safe_float,
    _summarize_runs,
    _write_csv,
    _write_json,
)
from tools.v53_full_sequence_drift_autopsy import (  # noqa: E402
    _load_kitti_gt,
    _load_run_poses,
    _segment_error,
)


DEFAULT_RESULT_ROOT = REPO_ROOT / "results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9"
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
DEFAULT_H35_704 = (
    REPO_ROOT
    / "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/"
    "phase7_layergamma_refine_screen/rollouts/V53_PHASE7_SCREEN_H35_LAYERGAMMAFIX_RHO0075_704F"
)
DEFAULT_H35_FULL = (
    REPO_ROOT
    / "results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/"
    "phase7_layergamma_fix_full/rollouts/V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075"
)


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _augment_segments(rows: Sequence[Dict[str, Any]], gt_path: Path) -> None:
    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(gt_path)
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        pose = _load_run_poses(run_dir, gt_poses, gt_pos)
        if pose.get("pose_status") != "done":
            continue
        frames = pose["frames"]
        aligned_pos = pose["aligned_pos"]
        for key, start, end in (
            ("seg0_000_384", 0, 384),
            ("seg1_384_700", 384, 700),
            ("seg2_700_end", 700, 20000),
        ):
            seg = _segment_error(frames, aligned_pos, gt_pos, start, end)
            for metric_key, value in seg.items():
                row[f"{key}_{metric_key}"] = value


def _is_704_row(row: Mapping[str, Any]) -> bool:
    name = str(row.get("run_name") or "")
    frames = int(row.get("frames") or 0)
    return "704" in name.upper() or (650 <= frames <= 750)


def _is_full_row(row: Mapping[str, Any]) -> bool:
    return bool(row.get("full_kitti01")) or int(row.get("frames") or 0) >= 1000


def _promotion_gate(row: Mapping[str, Any], h35: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if h35 is None:
        return {"promotion_gate_pass": False, "promotion_gate_reason": "missing_h35_704_reference"}
    reasons: List[str] = []
    runtime_projection = _safe_float(row.get("projected_full_wall_time_min"), 999.0)
    if runtime_projection > 28.0:
        reasons.append("projected_runtime_gt_28min")
    ate = _safe_float(row.get("ATE"), 999.0)
    h35_ate = _safe_float(h35.get("ATE"), 999.0)
    if ate > h35_ate - 0.20:
        reasons.append("ATE704_not_0.20m_better_than_H35")
    seg2 = _safe_float(row.get("seg2_700_end_rmse"), 999.0)
    h35_seg2 = _safe_float(h35.get("seg2_700_end_rmse"), 999.0)
    roll100 = _safe_float(row.get("rolling100_p90"), 999.0)
    h35_roll100 = _safe_float(h35.get("rolling100_p90"), 999.0)
    if not (seg2 <= h35_seg2 - 0.50 or roll100 <= h35_roll100 - 0.50):
        reasons.append("seg2_or_rolling100_not_0.50m_better_than_H35")
    if row.get("no_chunk_policy_pass") is not True:
        reasons.append("no_chunk_policy_fail")
    if row.get("manual_percentage_audit_pass") is not True:
        reasons.append("manual_percentage_audit_fail")
    if int(row.get("role_collapse_debug_rows") or 0) != 0:
        reasons.append("role_collapse_nonzero")
    if row.get("status") != "done":
        reasons.append("run_not_done")
    return {
        "promotion_gate_pass": not reasons,
        "promotion_gate_reason": ",".join(reasons) if reasons else "pass",
    }


def _copy_or_no_data(src: Path, dst: Path, title: str, note: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)
    else:
        _plot_no_data(dst, title, note)


def _plot_phase1_artifacts(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_or_no_data(
        V52_AUTOPSY_DIR / "post_zp_delta_ratio_by_chunk.png",
        out_dir / "teacher_student_post_zp_delta_timeline.png",
        "teacher/student post-zp delta timeline",
        f"missing source: {V52_AUTOPSY_DIR / 'post_zp_delta_ratio_by_chunk.png'}",
    )
    _copy_or_no_data(
        V52_AUTOPSY_DIR / "teacher_student_role_mass_timeline.png",
        out_dir / "teacher_student_gamma_timeline.png",
        "teacher/student gamma timeline",
        f"missing source: {V52_AUTOPSY_DIR / 'teacher_student_role_mass_timeline.png'}",
    )

    rows = _read_csv(V52_AUTOPSY_DIR / "delta_norm_ratio_by_layer.csv")
    if rows:
        layers = sorted({int(float(r["layer"])) for r in rows if str(r.get("layer", "")).strip()})
        branches = ["w0", "w1", "w2"]
        data = np.full((len(layers), len(branches)), np.nan, dtype=np.float32)
        for r in rows:
            try:
                li = layers.index(int(float(r.get("layer", "nan"))))
                bi = branches.index(str(r.get("branch")))
            except (ValueError, TypeError):
                continue
            data[li, bi] = _safe_float(r.get("student_over_c9_delta_norm"))
        fig, ax = plt.subplots(figsize=(5.5, 7.0))
        im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=0.5, vmax=1.5)
        ax.set_xticks(range(len(branches)))
        ax.set_xticklabels(branches)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([str(x) for x in layers])
        ax.set_xlabel("branch")
        ax.set_ylabel("layer")
        ax.set_title("student/C9 delta norm by layer/branch")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "teacher_student_layer_branch_heatmap.png", dpi=160)
        plt.close(fig)
    else:
        _plot_no_data(
            out_dir / "teacher_student_layer_branch_heatmap.png",
            "teacher/student layer branch heatmap",
            "delta_norm_ratio_by_layer.csv missing",
        )


def _phase1_rows(h35_704: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for src in _read_csv(V52_AUTOPSY_DIR / "run_overview.csv"):
        rows.append(
            {
                "source": "v52_phase2_autopsy",
                "artifact_status": "present",
                "run": src.get("run"),
                "label": src.get("label"),
                "method": src.get("method"),
                "ATE": src.get("ATE_full"),
                "delta_vs_C9": src.get("delta_vs_C9"),
                "chunk_total_seconds_mean": src.get("chunk_total_seconds_mean"),
                "probe_ttt_write_seconds_mean": src.get("probe_ttt_write_seconds_mean"),
                "risk_source": src.get("risk_source_seen"),
                "role_sources_seen": src.get("role_sources_seen"),
                "positive_mass_mean": src.get("positive_mass_mean"),
                "neutral_mass_mean": src.get("neutral_mass_mean"),
                "negative_mass_mean": src.get("negative_mass_mean"),
                "w0_gamma_mean": src.get("w0_gamma_mean"),
            }
        )
    if h35_704 is not None:
        rows.append(
            {
                "source": "v53_h35_704_reference",
                "artifact_status": "present",
                "run": h35_704.get("run_name"),
                "label": "v53 H35 clean adaptive 704F reference",
                "method": "adaptive_writer_sc_gamma_split + residual_x_dg + layer 0/8/17 rho0.0075",
                "ATE": h35_704.get("ATE"),
                "delta_vs_C9": _safe_float(h35_704.get("ATE")) - C9_P0_ATE,
                "chunk_total_seconds_mean": h35_704.get("chunk_total_seconds_mean"),
                "probe_ttt_write_seconds_mean": h35_704.get("probe_ttt_write_seconds_mean"),
                "risk_source": h35_704.get("risk_source_config"),
                "role_sources_seen": h35_704.get("role_sources_seen"),
                "positive_mass_mean": h35_704.get("ttt_positive_mass_mean"),
                "neutral_mass_mean": h35_704.get("ttt_neutral_mass_mean"),
                "negative_mass_mean": h35_704.get("ttt_negative_mass_mean"),
                "w0_gamma_mean": h35_704.get("adaptive_gamma_mean"),
            }
        )
    rows.append(
        {
            "source": "v52_energy_matched_expected",
            "artifact_status": "missing_or_not_landed",
            "run": "V52_EnergyMatched",
            "label": "v52 EnergyMatched reference",
            "method": "expected by v54 plan, no landed row found in v52 phase2 run_overview.csv",
        }
    )
    return rows


def _write_phase1_report(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# ACL2 v54 Phase 1 teacher/student autopsy",
        "",
        "This file only summarizes landed artifacts. Missing v52 EnergyMatched fields are left empty.",
        "",
        f"Source v52 autopsy directory: `{V52_AUTOPSY_DIR}`",
        "",
        "| source | run | method | ATE | delta vs C9 | risk source | role source | gamma mean |",
        "|---|---|---|---:|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('source')}` | `{row.get('run')}` | {row.get('method') or ''} | "
            f"{_fmt(row.get('ATE'))} | {_fmt(row.get('delta_vs_C9'))} | `{row.get('risk_source') or ''}` | "
            f"`{row.get('role_sources_seen') or ''}` | {_fmt(row.get('w0_gamma_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Design Readout",
            "",
            "- v52/v53 evidence supports the v54 design premise: role split is active and auditable, but post-zp update energy and commit behavior remain mismatched to C9.",
            "- v54 therefore implements only M1/M2 instead of continuing rho/layer/role-threshold sweeps.",
        ]
    )
    (out_dir / "v54_phase1_teacher_student_autopsy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_v54_timelines(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    gamma_rows: List[Dict[str, Any]] = []
    commit_rows: List[Dict[str, Any]] = []
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        if not run_dir.is_dir():
            continue
        for idx, outer in enumerate(_read_jsonl(run_dir / "hmc_state_hash.jsonl")):
            gamma_vals: List[float] = []
            commit_vals: List[float] = []
            for node in _walk(outer):
                if "ttt_tri_replay_state_energy_gamma_mean" in node:
                    gamma_vals.append(_safe_float(node.get("ttt_tri_replay_state_energy_gamma_mean")))
                if "ttt_write_commit_filter_scale_mean" in node:
                    commit_vals.append(_safe_float(node.get("ttt_write_commit_filter_scale_mean")))
            gamma_vals = [v for v in gamma_vals if math.isfinite(v)]
            commit_vals = [v for v in commit_vals if math.isfinite(v)]
            if gamma_vals:
                gamma_rows.append({"run_name": row.get("run_name"), "chunk_index_observed": idx, "state_energy_gamma_mean": float(np.mean(gamma_vals))})
            if commit_vals:
                commit_rows.append({"run_name": row.get("run_name"), "chunk_index_observed": idx, "commit_alpha_mean": float(np.mean(commit_vals))})
        log_path = run_dir / "01.log"
        if log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            matches = list(re.finditer(r"# V2 Chunk\s+(\d+)/\d+", text))
            for pos, match in enumerate(matches):
                chunk_idx = int(match.group(1))
                start = match.start()
                end = matches[pos + 1].start() if pos + 1 < len(matches) else len(text)
                segment = text[start:end]
                gamma_vals = [
                    _safe_float(value)
                    for value in re.findall(r"['\"]ttt_tri_replay_state_energy_gamma_mean['\"]:\s*([-+0-9.eE]+)", segment)
                ]
                gamma_vals = [v for v in gamma_vals if math.isfinite(v)]
                commit_vals = [
                    _safe_float(value)
                    for value in re.findall(r"['\"]ttt_write_commit_filter_scale_mean['\"]:\s*([-+0-9.eE]+)", segment)
                ]
                commit_vals = [v for v in commit_vals if math.isfinite(v)]
                if gamma_vals:
                    gamma_rows.append({
                        "run_name": row.get("run_name"),
                        "chunk_index_observed": chunk_idx,
                        "state_energy_gamma_mean": float(np.mean(gamma_vals)),
                        "source": "01.log",
                    })
                if commit_vals:
                    commit_rows.append({
                        "run_name": row.get("run_name"),
                        "chunk_index_observed": chunk_idx,
                        "commit_alpha_mean": float(np.mean(commit_vals)),
                        "source": "01.log",
                    })
    _write_csv(out_dir / "v54_state_energy_gamma_timeline.csv", gamma_rows)
    _write_csv(out_dir / "v54_commit_alpha_timeline.csv", commit_rows)

    if gamma_rows:
        fig, ax = plt.subplots(figsize=(10, 4))
        for run in sorted({str(r["run_name"]) for r in gamma_rows}):
            pts = [(int(r["chunk_index_observed"]), _safe_float(r["state_energy_gamma_mean"])) for r in gamma_rows if str(r["run_name"]) == run]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=1.1, label=run)
        ax.set_xlabel("observed JSONL row index")
        ax.set_ylabel("state energy gamma mean")
        ax.set_title("v54 state-energy gamma timeline")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "v54_state_energy_gamma_timeline.png", dpi=160)
        plt.close(fig)
    else:
        _plot_no_data(out_dir / "v54_state_energy_gamma_timeline.png", "v54 state-energy gamma timeline", "no v54 gamma debug rows found")

    if commit_rows:
        fig, ax = plt.subplots(figsize=(10, 4))
        for run in sorted({str(r["run_name"]) for r in commit_rows}):
            pts = [(int(r["chunk_index_observed"]), _safe_float(r["commit_alpha_mean"])) for r in commit_rows if str(r["run_name"]) == run]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=1.1, label=run)
        ax.set_xlabel("observed JSONL row index")
        ax.set_ylabel("commit alpha/scale mean")
        ax.set_title("v54 commit alpha timeline")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "teacher_student_commit_delta_timeline.png", dpi=160)
        plt.close(fig)
    else:
        _plot_no_data(out_dir / "teacher_student_commit_delta_timeline.png", "v54 commit alpha timeline", "no commit alpha debug rows found")


def _write_704_report(out_dir: Path, rows_704: Sequence[Mapping[str, Any]], h35: Optional[Mapping[str, Any]]) -> None:
    lines = [
        "# ACL2 v54 704F report",
        "",
        f"H35 reference: `{h35.get('run_name') if h35 else 'missing'}`.",
        "",
        "| run | ATE704 | H35 delta | seg2 rmse | rolling100 p90 | projected full min | chunk mean | TTT mean | no-chunk | manual % | collapse | gate | reason |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|",
    ]
    h35_ate = _safe_float(h35.get("ATE") if h35 else None)
    for row in rows_704:
        lines.append(
            f"| `{row.get('run_name')}` | {_fmt(row.get('ATE'))} | {_fmt(_safe_float(row.get('ATE')) - h35_ate)} | "
            f"{_fmt(row.get('seg2_700_end_rmse'))} | {_fmt(row.get('rolling100_p90'))} | {_fmt(row.get('projected_full_wall_time_min'))} | "
            f"{_fmt(row.get('chunk_total_seconds_mean'))} | {_fmt(row.get('probe_ttt_write_seconds_mean'))} | "
            f"{row.get('no_chunk_policy_pass')} | {row.get('manual_percentage_audit_pass')} | "
            f"{row.get('role_collapse_debug_rows')} | {row.get('promotion_gate_pass')} | `{row.get('promotion_gate_reason')}` |"
        )
    if not rows_704:
        lines.append("| no v54 704F candidate found | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | False | no_data |")
    (out_dir / "v54_704_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_full_report(out_dir: Path, full_rows: Sequence[Mapping[str, Any]], rows_704: Sequence[Mapping[str, Any]]) -> None:
    any_704_pass = any(bool(row.get("promotion_gate_pass")) for row in rows_704)
    lines = [
        "# ACL2 v54 full report",
        "",
        "| run | ATE | delta vs C9 | frames | wall min | chunk mean | TTT mean | progress <=35.30 | soft <=34.60 | close <=34.30 | excellent <=34.06 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in full_rows:
        ate = _safe_float(row.get("ATE"), 999.0)
        lines.append(
            f"| `{row.get('run_name')}` | {_fmt(row.get('ATE'))} | {_fmt(row.get('delta_vs_C9_P0'))} | {row.get('frames')} | "
            f"{_fmt(row.get('wall_time_min'))} | {_fmt(row.get('chunk_total_seconds_mean'))} | {_fmt(row.get('probe_ttt_write_seconds_mean'))} | "
            f"{ate <= 35.30} | {ate <= 34.60} | {ate <= 34.30} | {ate <= 34.06} |"
        )
    if not full_rows:
        reason = "704F gate did not pass; full run was not allowed" if not any_704_pass else "no full artifact found"
        lines.append(f"| no v54 full run | NA | NA | NA | NA | NA | NA | False | False | False | False |")
        lines.extend(["", f"Full status: {reason}."])
    (out_dir / "v54_full_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_failure_routing(out_dir: Path, rows_704: Sequence[Mapping[str, Any]], full_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# ACL2 v54 failure routing",
        "",
        "| run | phase | ATE | gate | reason | prescribed action |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows_704:
        action = "eligible for full" if row.get("promotion_gate_pass") else "analyze missing state variable/layer-branch energy mismatch; do not start full"
        lines.append(
            f"| `{row.get('run_name')}` | `704F` | {_fmt(row.get('ATE'))} | {row.get('promotion_gate_pass')} | "
            f"`{row.get('promotion_gate_reason')}` | {action} |"
        )
    for row in full_rows:
        ate = _safe_float(row.get("ATE"), 999.0)
        action = "freeze as candidate" if ate <= 35.30 else "generate full-vs-704 autopsy; do not claim success"
        lines.append(
            f"| `{row.get('run_name')}` | `full` | {_fmt(row.get('ATE'))} | {ate <= 35.30} | "
            f"`full_progress_gate_{ate <= 35.30}` | {action} |"
        )
    if not rows_704 and not full_rows:
        lines.append("| no v54 run found | NA | NA | False | `no_data` | run M1_704F/M2_704F first |")
    if rows_704 and not any(bool(r.get("promotion_gate_pass")) for r in rows_704):
        lines.extend(
            [
                "",
                "## Routed Diagnosis",
                "",
                "Both 704F candidates failed or are absent. Per v54 plan, the next action-space change should be singular: add a stronger online state variable that separates C9's late/seg2 behavior, rather than sweeping rho/layer/role percentages.",
            ]
        )
    (out_dir / "v54_failure_routing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_code_audit(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# ACL2 v54 code audit",
        "",
        "No synthetic metric values are inserted by this report.",
        "",
        "## Edited Files",
        "",
        "| file | audit note |",
        "|---|---|",
        "| `loger/pipeline/ttt_write_controller.py` | Added v54 state-energy role split, causal energy EMA gamma, and directional commit guard. No absolute chunk-id maps added. |",
        "| `run_pipeline_abc_v2.py` | Added CLI choices for the v54 directional commit guard. |",
        "| `tools/run_v54_state_conditioned_candidate.sh` | Added fixed v54 launcher wrapper over the no-chunk v47 runner. |",
        "| `tools/v54_experiment_report.py` | Added report/gate generation from landed artifacts. |",
        "",
        "## Run Audit",
        "",
        "| run | no-chunk | manual % | role mode | risk source | commit mode |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('run_name')}` | {row.get('no_chunk_policy_pass')} | {row.get('manual_percentage_audit_pass')} | "
            f"`{row.get('role_mode_config')}` | `{row.get('risk_source_config')}` | `{row.get('commit_filter_mode_config')}` |"
        )
    if not rows:
        lines.append("| no v54 run found | NA | NA | NA | NA | NA |")
    (out_dir / "v54_code_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reports(
    out_dir: Path,
    v54_rows: Sequence[Dict[str, Any]],
    h35_704: Optional[Dict[str, Any]],
    h35_full: Optional[Dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = list(v54_rows)
    ref_rows = [r for r in (h35_704, h35_full) if r is not None]
    _write_csv(out_dir / "v54_all_registry.csv", all_rows)
    _write_json(out_dir / "v54_all_registry.json", all_rows)
    _write_csv(out_dir / "v54_reference_registry.csv", ref_rows)

    rows_704 = [dict(row) for row in all_rows if _is_704_row(row)]
    for row in rows_704:
        row.update(_promotion_gate(row, h35_704))
    full_rows = [dict(row) for row in all_rows if _is_full_row(row)]
    _write_csv(out_dir / "v54_704_registry.csv", rows_704)
    _write_csv(out_dir / "v54_full_registry.csv", full_rows)
    _write_704_report(out_dir, rows_704, h35_704)
    _write_full_report(out_dir, full_rows, rows_704)
    _write_failure_routing(out_dir, rows_704, full_rows)
    _write_code_audit(out_dir, all_rows)
    _extract_v54_timelines(all_rows, out_dir)

    phase1 = _phase1_rows(h35_704)
    _write_csv(out_dir / "v54_phase1_teacher_student_autopsy.csv", phase1)
    _write_phase1_report(out_dir, phase1)
    _plot_phase1_artifacts(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--gt", default=str(DEFAULT_GT))
    parser.add_argument("--h35-704", default=str(DEFAULT_H35_704))
    parser.add_argument("--h35-full", default=str(DEFAULT_H35_FULL))
    args = parser.parse_args()

    result_root = Path(args.result_root)
    out_dir = Path(args.out_dir) if args.out_dir else result_root / "report_final"
    rollout_roots = sorted(result_root.glob("phase*/rollouts"))
    run_dirs = _iter_run_dirs(rollout_roots)
    v54_rows = _summarize_runs(run_dirs, Path(args.gt)) if run_dirs else []
    ref_dirs = [Path(args.h35_704), Path(args.h35_full)]
    ref_rows = _summarize_runs([p for p in ref_dirs if p.is_dir()], Path(args.gt))
    _augment_segments(v54_rows, Path(args.gt))
    _augment_segments(ref_rows, Path(args.gt))
    h35_704 = next((row for row in ref_rows if "704" in str(row.get("run_name", "")).upper()), None)
    h35_full = next((row for row in ref_rows if _is_full_row(row)), None)
    _write_reports(out_dir, v54_rows, h35_704, h35_full)
    print(f"Wrote v54 report with {len(v54_rows)} v54 rows to {out_dir}")


if __name__ == "__main__":
    main()
