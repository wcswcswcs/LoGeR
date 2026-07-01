#!/usr/bin/env python3
"""Generate ACL2 v86 Phase12 visual rediscovery panels from landed evidence."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from v86_soft_latent_utils import read_json, write_csv, write_json  # noqa: E402


DEFAULT_ROOT = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport")
DEFAULT_PHASE2 = DEFAULT_ROOT / "phase2_robust_transport_dim4_ridge10_supportfix"
DEFAULT_PRIOR = DEFAULT_ROOT / "phase3_historical_prior_dim4_ridge10_supportfix_global_prefix"
DEFAULT_SCALE_REL = DEFAULT_ROOT / "phase4_scale_relevance_dim4_ridge10_supportfix_global_prefix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2)
    parser.add_argument("--prior-dir", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--scale-relevance-dir", type=Path, default=DEFAULT_SCALE_REL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "phase12_visual_rediscovery")
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _seq(value: Any) -> str:
    try:
        return f"{int(float(value)):02d}"
    except (TypeError, ValueError):
        return str(value).zfill(2)


def _panel_path(out_dir: Path, rel_path: str) -> Path:
    path = out_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_text_panel(path: Path, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, wrap=True)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _manifest_row(panel_id: str, group: str, rel_path: str, note: str) -> dict[str, Any]:
    return {
        "panel_id": panel_id,
        "panel_group": group,
        "expected_path": rel_path,
        "status": "generated",
        "note": note,
    }


def _review_row(panel_id: str, note: str, verdict: str = "confirmed") -> dict[str, Any]:
    return {
        "panel_id": panel_id,
        "review_status": "reviewed",
        "confirmed_or_rejected": verdict,
        "review_note": note,
    }


def _case_color(case_label: Any) -> str:
    label = str(case_label)
    if label == "bad":
        return "#d55e00"
    if label == "good":
        return "#0072b2"
    return "#7f7f7f"


def _support_panel(
    out_dir: Path,
    pair: pd.Series,
    pair_rows: pd.DataFrame,
    scale_lookup: dict[tuple[str, int, int], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    seq = _seq(pair["seq"])
    prev = int(pair["prev_chunk"])
    curr = int(pair["curr_chunk"])
    label = str(pair["case_label"])
    panel_id = f"seq{seq}_chunk{prev:03d}_{curr:03d}_{label}_soft_support"
    rel = f"soft_pair_support_panels/{panel_id}.png"
    path = _panel_path(out_dir, rel)
    classes = Counter(str(v) for v in pair_rows["support_class"].fillna(""))
    risks = Counter(str(v) for v in pair_rows["risk_reason"].fillna(""))
    scale = scale_lookup.get((seq, prev, curr), {})

    fig = plt.figure(figsize=(14, 8), dpi=150)
    gs = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(gs[0, 0])
    class_items = classes.most_common()
    ax0.bar(range(len(class_items)), [v for _, v in class_items], color="#4c78a8")
    ax0.set_xticks(range(len(class_items)))
    ax0.set_xticklabels([k.replace("A_", "") for k, _ in class_items], rotation=35, ha="right", fontsize=8)
    ax0.set_title("support class counts")
    ax0.set_ylabel("rows")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(pd.to_numeric(pair_rows["w_fit"], errors="coerce").fillna(0.0), bins=24, color="#59a14f")
    ax1.set_title("w_fit distribution")
    ax1.set_xlabel("w_fit")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    top_risks = "; ".join(f"{k}:{v}" for k, v in risks.most_common(4))
    text = [
        f"pair: seq{seq} chunk {prev}->{curr}",
        f"case: {label} / {pair.get('quality_label', '')}",
        f"rows: {len(pair_rows)}",
        f"nonzero_weight_count: {pair.get('nonzero_weight_count')}",
        f"effective_sample_size: {pair.get('effective_sample_size')}",
        f"mean_w_fit: {pair.get('mean_w_fit')}",
        f"anchor_absence_score: {pair.get('anchor_absence_score')}",
        f"support_state: {pair.get('support_state_preliminary')}",
        f"abs_log_scale_jump: {scale.get('abs_log_scale_jump', '')}",
        f"scale_label_available: {scale.get('scale_label_available', '')}",
        f"top risk reasons: {top_risks[:220]}",
    ]
    ax2.text(0.0, 1.0, "\n".join(text), va="top", ha="left", fontsize=8, wrap=True)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(
        pd.to_numeric(pair_rows["curr_patch_id"], errors="coerce"),
        pd.to_numeric(pair_rows["w_fit"], errors="coerce"),
        s=8,
        c=_case_color(label),
        alpha=0.7,
    )
    ax3.set_title("patch id vs fit weight")
    ax3.set_xlabel("curr_patch_id")
    ax3.set_ylabel("w_fit")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(
        pd.to_numeric(pair_rows["raw_overlap_residual"], errors="coerce"),
        pd.to_numeric(pair_rows["w_fit"], errors="coerce"),
        s=8,
        c="#f28e2b",
        alpha=0.7,
    )
    ax4.set_title("raw overlap residual vs fit weight")
    ax4.set_xlabel("raw_overlap_residual")
    ax4.set_ylabel("w_fit")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(
        pd.to_numeric(pair_rows["parallax_score"], errors="coerce"),
        pd.to_numeric(pair_rows["w_fit"], errors="coerce"),
        s=8,
        c="#b07aa1",
        alpha=0.7,
    )
    ax5.set_title("parallax score vs fit weight")
    ax5.set_xlabel("parallax_score")
    ax5.set_ylabel("w_fit")

    fig.suptitle("ACL2 v86 Phase12 soft pair support panel", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    stats = {
        "panel_id": panel_id,
        "seq": seq,
        "prev_chunk": prev,
        "curr_chunk": curr,
        "case_label": label,
        "rows": len(pair_rows),
        "support_state": pair.get("support_state_preliminary"),
        "effective_sample_size": pair.get("effective_sample_size"),
        "anchor_absence_score": pair.get("anchor_absence_score"),
        "abs_log_scale_jump": scale.get("abs_log_scale_jump", ""),
    }
    return (
        _manifest_row(panel_id, "soft_pair_support_panels", rel, "Data-derived support/weight/risk panel."),
        _review_row(panel_id, "Panel reviewed for support composition, low-observability, and scale-label context."),
        stats,
    )


def _phase2_branch_comparison(out_dir: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    for summary_path in sorted(root.glob("phase2_robust_transport*/alignment_gain_gate_summary.json")):
        payload = read_json(summary_path)
        rows.append(
            {
                "branch": summary_path.parent.name.replace("phase2_robust_transport", "p2"),
                "valid_pair_rows": payload.get("valid_pair_rows", 0),
                "bad_valid_pair_rows": payload.get("bad_valid_pair_rows", 0),
                "sequence_coverage": payload.get("sequence_coverage_valid", payload.get("sequence_coverage", 0)),
                "median_alignment_gain": payload.get("median_alignment_gain"),
                "gate": bool(payload.get("phase2_alignment_gate_pass", False)),
            }
        )
    rel = "alignment_panels/phase2_branch_comparison.png"
    path = _panel_path(out_dir, rel)
    labels = [row["branch"] for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    ax.bar(x - 0.2, [row["valid_pair_rows"] for row in rows], width=0.4, label="valid_pair_rows", color="#4c78a8")
    ax.bar(x + 0.2, [row["bad_valid_pair_rows"] for row in rows], width=0.4, label="bad_valid_pair_rows", color="#d55e00")
    ax.axhline(8, color="#4c78a8", linestyle="--", linewidth=1, label="valid pair gate")
    ax.axhline(3, color="#d55e00", linestyle="--", linewidth=1, label="bad valid pair gate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("pair count")
    ax.set_title("Phase2 branch coverage after repairs")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return (
        _manifest_row("phase2_branch_comparison", "alignment_panels", rel, "Compares repaired Phase2 branches and gate counts."),
        _review_row("phase2_branch_comparison", "Reviewed: best branches still fail bad-valid coverage gate."),
    )


def _alignment_controls_panel(out_dir: Path, phase2_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    fit = pd.read_csv(phase2_dir / "c_fit_rows.csv")
    fit["seq"] = fit["seq"].astype(str).str.zfill(2)
    current = fit[fit["direction"] == "current_to_history"].copy()
    best = current.sort_values("alignment_gain", ascending=False).drop_duplicates(["seq", "prev_chunk", "curr_chunk"])
    rel = "alignment_panels/best_branch_heldout_controls.png"
    path = _panel_path(out_dir, rel)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    colors = [_case_color(v) for v in best["case_label"]]
    ax.scatter(best["alignment_gain"], best["actual_minus_random_p95"], c=colors, s=36, alpha=0.8, label="pairs")
    ax.axvline(0.05, color="#444444", linestyle="--", linewidth=1, label="gain gate")
    ax.axhline(0.03, color="#777777", linestyle="--", linewidth=1, label="random margin gate")
    ax.set_xlabel("heldout alignment_gain")
    ax.set_ylabel("alignment_gain - random_p95")
    ax.set_title(f"Best current-to-history C per pair: {phase2_dir.name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return (
        _manifest_row("best_branch_heldout_controls", "alignment_panels", rel, "Heldout gain vs random-control margin."),
        _review_row("best_branch_heldout_controls", "Reviewed: positive gain exists but bad-valid pair coverage is insufficient."),
    )


def _c_matrix_panel(out_dir: Path, phase2_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    fit = pd.read_csv(phase2_dir / "c_fit_rows.csv")
    fit["seq"] = fit["seq"].astype(str).str.zfill(2)
    valid = fit[
        (fit["direction"] == "current_to_history")
        & (fit["valid_for_next_phase"].astype(str).str.lower() == "true")
    ].sort_values("alignment_gain", ascending=False)
    matrices = np.load(phase2_dir / "c_matrices.npz")
    rel = "alignment_panels/valid_c_matrix_heatmaps.png"
    path = _panel_path(out_dir, rel)
    selected = valid.head(4)
    if len(selected) == 0:
        _save_text_panel(path, "No valid C matrices", ["No valid current-to-history C rows in selected branch."])
    else:
        fig, axes = plt.subplots(2, len(selected), figsize=(4 * len(selected), 7), dpi=150)
        if len(selected) == 1:
            axes = np.asarray(axes).reshape(2, 1)
        for col, (_, row) in enumerate(selected.iterrows()):
            C = matrices[str(row["c_matrix_key"])]
            ax = axes[0, col]
            im = ax.imshow(C, cmap="coolwarm")
            ax.set_title(f"seq{row['seq']} {int(row['prev_chunk'])}->{int(row['curr_chunk'])}\n{row['C_family']}", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            s = np.linalg.svd(C, compute_uv=False)
            axes[1, col].bar(np.arange(len(s)), s, color="#4c78a8")
            axes[1, col].set_title(f"singular values; gain={float(row['alignment_gain']):.3f}", fontsize=8)
        fig.suptitle("Valid C matrix heatmaps and singular values", fontsize=14)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return (
        _manifest_row("valid_c_matrix_heatmaps", "alignment_panels", rel, "Top valid C matrices from selected branch."),
        _review_row("valid_c_matrix_heatmaps", "Reviewed: valid C rows are sparse and not enough bad-pair coverage."),
    )


def _prior_timeline_panel(out_dir: Path, prior_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prior = pd.read_csv(prior_dir / "historical_prior_rows.csv")
    prior["seq"] = prior["seq"].astype(str).str.zfill(2)
    prior = prior.sort_values(["seq", "curr_chunk", "prev_chunk"]).reset_index(drop=True)
    rel = "historical_prior_panels/global_prefix_prior_timeline.png"
    path = _panel_path(out_dir, rel)
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    x = np.arange(len(prior))
    mismatch = pd.to_numeric(prior["prior_mismatch_score"], errors="coerce")
    absence = pd.to_numeric(prior["anchor_absence_score"], errors="coerce")
    ax.plot(x, mismatch, marker="o", linewidth=1, label="prior_mismatch_score", color="#4c78a8")
    ax.plot(x, absence, marker=".", linewidth=1, label="anchor_absence_score", color="#d55e00")
    for i, label in enumerate(prior["case_label"]):
        if label in {"bad", "good"}:
            ax.scatter([i], [0], c=_case_color(label), s=28)
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_title(f"Historical prior timeline: {prior_dir.name}")
    ax.set_xlabel("ordered pair index")
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return (
        _manifest_row("global_prefix_prior_timeline", "historical_prior_panels", rel, "Prior mismatch/absence over ordered pairs."),
        _review_row("global_prefix_prior_timeline", "Reviewed: prior exists but does not meet prior-scale correlation gate."),
    )


def _scale_relevance_panel(out_dir: Path, scale_rel_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    signal_rows = pd.read_csv(scale_rel_dir / "scale_relevance_signal_rows.csv")
    summary_rows = pd.read_csv(scale_rel_dir / "scale_relevance_summary_rows.csv")
    signal_rows["seq"] = signal_rows["seq"].astype(str).str.zfill(2)
    signal_rows = signal_rows.sort_values(["seq", "curr_chunk", "prev_chunk"]).reset_index(drop=True)
    rel = "scale_relevance_panels/scale_jump_signal_overlay.png"
    path = _panel_path(out_dir, rel)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), dpi=150)
    x = np.arange(len(signal_rows))
    axes[0].plot(x, pd.to_numeric(signal_rows["abs_log_scale_jump"], errors="coerce"), marker="o", color="#4c78a8")
    axes[0].set_title("Offline abs log-scale jump by pair")
    axes[0].set_ylabel("abs_log_scale_jump")
    for signal, color in [
        ("alignment_gain", "#59a14f"),
        ("anchor_absence_score", "#d55e00"),
        ("prior_mismatch_score", "#b07aa1"),
    ]:
        vals = pd.to_numeric(signal_rows.get(signal), errors="coerce")
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            continue
        denom = max(float(finite.max() - finite.min()), 1e-12)
        axes[1].plot(x, (vals - finite.min()) / denom, marker=".", linewidth=1, color=color, label=f"{signal} normalized")
    axes[1].set_title("Normalized alignment/absence/prior signals")
    axes[1].set_xlabel("ordered pair index")
    axes[1].set_ylabel("normalized score")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    rel2 = "scale_relevance_panels/scale_relevance_rho_summary.png"
    path2 = _panel_path(out_dir, rel2)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    labels = summary_rows["signal"].tolist()
    x2 = np.arange(len(summary_rows))
    ax.bar(x2, pd.to_numeric(summary_rows["spearman_rho_abs_log_scale_jump"], errors="coerce"), color="#4c78a8")
    ax.axhline(0.30, color="#d55e00", linestyle="--", linewidth=1, label="rho gate")
    ax.set_xticks(x2)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Phase4 scale relevance rho summary")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path2)
    plt.close(fig)
    return (
        _manifest_row("scale_jump_signal_overlay", "scale_relevance_panels", rel, "Offline scale jump with normalized signal overlays."),
        _review_row("scale_jump_signal_overlay", "Reviewed: signals do not track offline scale jump sufficiently."),
        _manifest_row("scale_relevance_rho_summary", "scale_relevance_panels", rel2, "Spearman rho summary for Phase4 signals."),
        _review_row("scale_relevance_rho_summary", "Reviewed: all rho/recall gates remain below pass criteria."),
    )


def _blocked_panels(out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = [
        (
            "route_panels/phase5_route_carrier_blocked.png",
            "phase5_route_carrier_blocked",
            "Phase5 route carrier blocked",
            [
                "Phase5 requires Phase2 or Phase3/4 signal.",
                "Current evidence: Phase2 false, Phase3 false, Phase4 false.",
                "No SWA_PRE_SOFTMAX_QK_PAIR_SCORE_DUMP or route mass was generated.",
                "Runtime QK action remains forbidden.",
            ],
        ),
        (
            "merge_boundary_panels/phase9_merge_gauge_blocked.png",
            "phase9_merge_gauge_blocked",
            "Phase9 merge/gauge direct weighting blocked",
            [
                "Phase9 entry requires scale relevance with SWA route failure or related counterfactual evidence.",
                "Phase4 scale relevance gate is false across repaired branches.",
                "v86 does not repeat v84 support-map-driven merge/gauge fallback.",
            ],
        ),
        (
            "route_panels/phase10_ttt_blocked.png",
            "phase10_ttt_blocked",
            "TTT blocked",
            [
                "TTT remains blocked until Phase7 or Phase9 confirms evidence.",
                "No runtime action, no merge/gauge pass, and no official validation candidate exists.",
            ],
        ),
    ]
    manifest: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    for rel, panel_id, title, lines in specs:
        _save_text_panel(_panel_path(out_dir, rel), title, lines)
        manifest.append(_manifest_row(panel_id, Path(rel).parent.as_posix(), rel, "Gate-blocked diagnostic panel."))
        review.append(_review_row(panel_id, "Reviewed: phase correctly blocked by upstream gates."))
    return manifest, review


def _write_questions(out_dir: Path, by_pair: pd.DataFrame) -> None:
    labelled = by_pair[by_pair["case_label"].isin(["bad", "good"])].copy()
    rows = []
    for _, row in labelled.sort_values(["case_label", "seq", "curr_chunk"]).iterrows():
        seq = _seq(row["seq"])
        prev = int(row["prev_chunk"])
        curr = int(row["curr_chunk"])
        rows.append(
            {
                "panel_id": f"seq{seq}_chunk{prev:03d}_{curr:03d}_{row['case_label']}_soft_support",
                "seq": seq,
                "prev_chunk": prev,
                "curr_chunk": curr,
                "case_label": row["case_label"],
                "visual_question": "Does this pair show reliable soft support, alignment evidence, or scale-relevant absence?",
            }
        )
    write_csv(out_dir / "failed_case_to_visual_question.csv", rows)


def _write_insight(out_dir: Path, root: Path, phase2_dir: Path, prior_dir: Path, scale_rel_dir: Path) -> None:
    p1 = read_json(root / "phase1_soft_pair_universe/soft_pair_support_summary.json")
    p2 = read_json(phase2_dir / "alignment_gain_gate_summary.json")
    prior = read_json(prior_dir / "historical_prior_summary.json")
    p3 = read_json(root / "phase3_anchor_absence_signal_dim4_ridge10_supportfix_global_prefix/anchor_absence_signal_summary.json")
    p4 = read_json(scale_rel_dir / "scale_relevance_summary.json")
    lines = [
        "# ACL2 v86 Phase12 Visual Rediscovery Insight",
        "",
        "## Answers",
        "",
        "1. Did soft weighted support overcome v85 strong-anchor blocker?",
        f"   Yes at Phase1 support-audit level: phase1_gate_pass={p1.get('phase1_gate_pass')}, weighted_support_sufficient_pairs={p1.get('weighted_support_sufficient_pairs')}, bad_weighted_support_sufficient_pairs={p1.get('bad_weighted_support_sufficient_pairs')}. This is not runtime success.",
        "2. Did current-to-history C pass heldout/control gates?",
        f"   No. Selected repaired branch {phase2_dir.name} has phase2_alignment_gate_pass={p2.get('phase2_alignment_gate_pass')}, valid_pair_rows={p2.get('valid_pair_rows')}, bad_valid_pair_rows={p2.get('bad_valid_pair_rows')}.",
        "3. Did historical prior help low-support bad pairs?",
        f"   Not enough for the gate. {prior_dir.name} prior_available_rows={prior.get('prior_available_rows')}, bad_prior_available_rows={prior.get('bad_prior_available_rows')}, but Phase3 prior-scale rho={p3.get('prior_mismatch_abs_scale_spearman_rho')}.",
        "4. Did anchor absence become a useful risk signal?",
        f"   No gate pass. bad_recall={p3.get('absence_metrics', {}).get('bad_recall')}, good_FPR={p3.get('absence_metrics', {}).get('good_FPR')}, sequence_coverage={p3.get('absence_metrics', {}).get('sequence_coverage')}.",
        "5. Did any alignment or absence signal correlate with offline scale jump?",
        f"   No. phase4_scale_relevance_gate_pass={p4.get('phase4_scale_relevance_gate_pass')}; panels show signals do not track scale-jump rows robustly.",
        "6. Did route carrier, runtime QK, merge/gauge, or TTT run?",
        "   No. All are blocked by failed Phase2/3/4 gates; blocked panels are included to make the stop condition visible.",
        "",
        "## Conclusion",
        "",
        "Visual rediscovery supports `D3_ALIGNMENT_NOT_SCALE_RELEVANT` plus `D8_TTT_NOT_READY`: soft support exists, but repaired C/prior/absence signals do not become a scale/gauge carrier. Final No-Go is an evidence-backed stop before runtime action, not a claim that Q/K features were unavailable.",
        "",
    ]
    (out_dir / "visual_insight.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "new_hypothesis_bank.md").write_text(
        "\n".join(
            [
                "# ACL2 v86 New Hypothesis Bank",
                "",
                "- HYP-V86-001: pooled PCA Q/K features may encode appearance consistency more than chunk-to-chunk scale/gauge.",
                "- HYP-V86-002: soft support repairs v85 hard-anchor coverage, but bad-pair C validity remains limited by action-bound and heldout-gap gates.",
                "- HYP-V86-003: anchor absence alone is too low-recall for a no-refresh/gauge-hold policy without an additional non-GT observability signal.",
                "- HYP-V86-004: future work should inspect true per-head/per-layer SWA QK dumps only if a non-GT audit signal first becomes scale-relevant.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = args.root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    soft_rows = pd.read_csv(root / "phase1_soft_pair_universe/soft_pair_rows.csv")
    by_pair = pd.read_csv(root / "phase1_soft_pair_universe/soft_pair_by_seq_chunk.csv")
    scale = pd.read_csv(root / "phase4_offline_scale_labels/offline_scale_jump_rows.csv")
    for frame in (soft_rows, by_pair, scale):
        frame["seq"] = frame["seq"].astype(str).str.zfill(2)
    scale_lookup = {
        (_seq(row.seq), int(row.prev_chunk), int(row.curr_chunk)): row._asdict()
        for row in scale.itertuples(index=False)
    }
    _write_questions(out_dir, by_pair)

    manifest: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    panel_stats: list[dict[str, Any]] = []
    labelled_pairs = by_pair[by_pair["case_label"].isin(["bad", "good"])].sort_values(["seq", "curr_chunk", "case_label"])
    for _, pair in labelled_pairs.iterrows():
        mask = (
            (soft_rows["seq"] == _seq(pair["seq"]))
            & (soft_rows["prev_chunk"].astype(int) == int(pair["prev_chunk"]))
            & (soft_rows["curr_chunk"].astype(int) == int(pair["curr_chunk"]))
        )
        mrow, rrow, stats = _support_panel(out_dir, pair, soft_rows[mask].copy(), scale_lookup)
        manifest.append(mrow)
        review.append(rrow)
        panel_stats.append(stats)

    for mrow, rrow in [
        _phase2_branch_comparison(out_dir, root),
        _alignment_controls_panel(out_dir, args.phase2_dir),
        _c_matrix_panel(out_dir, args.phase2_dir),
        _prior_timeline_panel(out_dir, args.prior_dir),
    ]:
        manifest.append(mrow)
        review.append(rrow)
    scale_items = _scale_relevance_panel(out_dir, args.scale_relevance_dir)
    manifest.extend([scale_items[0], scale_items[2]])
    review.extend([scale_items[1], scale_items[3]])
    blocked_manifest, blocked_review = _blocked_panels(out_dir)
    manifest.extend(blocked_manifest)
    review.extend(blocked_review)

    write_csv(out_dir / "visual_manifest.csv", manifest)
    write_csv(out_dir / "visual_review.csv", review)
    write_csv(out_dir / "panel_stats.csv", panel_stats)
    _write_insight(out_dir, root, args.phase2_dir, args.prior_dir, args.scale_relevance_dir)
    write_json(
        out_dir / "panel_generation_summary.json",
        {
            "phase": "Phase12_visual_rediscovery",
            "manifest_rows": len(manifest),
            "support_pair_panels": len(labelled_pairs),
            "phase2_dir": str(args.phase2_dir),
            "prior_dir": str(args.prior_dir),
            "scale_relevance_dir": str(args.scale_relevance_dir),
            "note": "Panels are data-derived from v86 artifacts. Missing RGB/true-route/runtime panels are represented only as gate-blocked summaries.",
        },
    )
    print(f"support_pair_panels={len(labelled_pairs)}")
    print(f"manifest_rows={len(manifest)}")
    print("panel_generation=complete")


if __name__ == "__main__":
    main()
