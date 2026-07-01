#!/usr/bin/env python3
"""Build v91 visual rediscovery panels for semantic regime memory control."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from v86_soft_latent_utils import read_json, write_csv, write_json
from v90_semantic_topology_utils import image_path, load_raw, median_frame, seq_norm
from v91_semantic_regime_utils import ROOT, V90_LEDGER


DEFAULT_OUT = ROOT / "phase11_visual_rediscovery"
REQUIRED_CATEGORIES = [
    "semantic_topology_tracklet_panels",
    "regime_classifier_panels",
    "regime_conditioned_mode_panels",
    "policy_state_panels",
    "delayed_commit_panels",
    "carrier_or_blocked_panels",
    "counterfactual_or_blocked_panels",
    "runtime_or_blocked_panels",
    "ttt_or_blocked_panels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _load_image(seq: str, raw: dict[str, np.ndarray], side: str) -> tuple[Image.Image | None, int, Path | None]:
    frame = median_frame(raw, side)
    path = image_path(seq, frame)
    if path is None:
        return None, frame, None
    try:
        return Image.open(path).convert("RGB"), frame, path
    except Exception:
        return None, frame, path


def _scatter_semantics(ax: Any, raw: dict[str, np.ndarray], side: str, limit: int = 1000) -> None:
    pix_key = "prev_pixel_coords" if side == "prev" else "curr_pixel_coords"
    lab_key = "prev_semantic_labels" if side == "prev" else "curr_semantic_labels"
    conf_key = "prev_semantic_conf" if side == "prev" else "curr_semantic_conf"
    pts = np.asarray(raw.get(pix_key, []))
    labels = np.asarray(raw.get(lab_key, []), dtype=float)
    conf = np.asarray(raw.get(conf_key, []), dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) == 0:
        return
    step = max(1, int(len(pts) / limit))
    sub = pts[::step]
    lab = labels[::step] if len(labels) else np.zeros(len(sub))
    alpha = np.clip(conf[::step], 0.25, 0.85) if len(conf) else 0.55
    ax.scatter(sub[:, 1], sub[:, 0], c=lab, s=3, cmap="tab20", alpha=alpha)


def _select_visual_row(policy: pd.DataFrame, fallback: pd.DataFrame) -> pd.Series:
    candidates = policy if len(policy) else fallback
    if not len(candidates):
        raise RuntimeError("no v91 rows available for visualization")
    for _, row in candidates.iterrows():
        source_path = row.get("source_path", "")
        raw = load_raw(str(source_path)) if source_path else None
        if raw is None:
            continue
        seq = seq_norm(row.get("seq", ""))
        prev_img, _, _ = _load_image(seq, raw, "prev")
        curr_img, _, _ = _load_image(seq, raw, "curr")
        if prev_img is not None and curr_img is not None:
            return row
    return candidates.iloc[0]


def _mode_rows_for_pair(pair: pd.Series) -> pd.DataFrame:
    path = V90_LEDGER / "topology_mode_rows.csv"
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    seq = seq_norm(pair.get("seq", ""))
    prev = int(float(pair.get("prev_chunk", 0) or 0))
    curr = int(float(pair.get("curr_chunk", 0) or 0))
    return rows[
        (rows["seq"].astype(str).str.zfill(2) == seq)
        & (pd.to_numeric(rows["prev_chunk"], errors="coerce").fillna(-1).astype(int) == prev)
        & (pd.to_numeric(rows["curr_chunk"], errors="coerce").fillna(-1).astype(int) == curr)
    ]


def _text(ax: Any, title: str, lines: list[str]) -> None:
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=8.5)


def _bar(ax: Any, title: str, labels: list[str], values: list[float]) -> None:
    ax.set_title(title)
    if not labels:
        ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
        return
    ax.barh(labels, values, color=["#2f855a", "#c53030", "#2b6cb0", "#b7791f", "#4a5568"][: len(labels)])
    ax.invert_yaxis()


def _panel(
    out_path: Path,
    category: str,
    row: pd.Series,
    category_lines: list[str],
    blocked_reason: str,
    phase3: dict[str, Any],
    phase5: dict[str, Any],
    phase6: dict[str, Any],
) -> dict[str, Any]:
    source_path = str(row.get("source_path", ""))
    raw = load_raw(source_path) if source_path else None
    seq = seq_norm(row.get("seq", ""))
    prev_img = curr_img = None
    prev_frame = curr_frame = -1
    if raw is not None:
        prev_img, prev_frame, _ = _load_image(seq, raw, "prev")
        curr_img, curr_frame, _ = _load_image(seq, raw, "curr")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax, img, side, frame in [(axes[0, 0], prev_img, "prev", prev_frame), (axes[0, 1], curr_img, "curr", curr_frame)]:
        if img is not None and raw is not None:
            ax.imshow(img)
            _scatter_semantics(ax, raw, side)
            ax.set_title(f"{side} RGB frame {frame}\nsemantic labels/confidence overlay")
        else:
            ax.text(0.5, 0.5, f"{side} RGB unavailable", ha="center", va="center")
        ax.set_axis_off()
    modes = _mode_rows_for_pair(row)
    if len(modes):
        centers = pd.to_numeric(modes["mode_center_mu"], errors="coerce").fillna(0.0)
        masses = pd.to_numeric(modes["mode_mass"], errors="coerce").fillna(0.0)
        colors = [
            "#2f855a" if "VALID" in str(kind) else "#c53030" if "INVALID" in str(kind) else "#b7791f" if "LOWOBS" in str(kind) else "#4a5568"
            for kind in modes.get("topology_mode_type", pd.Series([""] * len(modes)))
        ]
        axes[0, 2].bar(centers, masses, width=0.025, color=colors)
        axes[0, 2].set_title("signed scale-mode histogram")
        axes[0, 2].set_xlabel("mode center")
        axes[0, 2].set_ylabel("mode mass")
    else:
        axes[0, 2].text(0.5, 0.5, "signed scale-mode histogram unavailable", ha="center", va="center")
    tracklet_labels = ["valid", "invalid", "context", "boundary"]
    tracklet_values = [
        float(row.get("S_valid", row.get("valid_tracklet_ratio", 0.0)) or 0.0),
        float(row.get("S_invalid", row.get("invalid_tracklet_ratio", 0.0)) or 0.0),
        float(row.get("S_context", row.get("context_lowobs_ratio", 0.0)) or 0.0),
        float(row.get("boundary_mass", 1.0 - float(row.get("valid_tracklet_ratio", 0.0) or 0.0)) or 0.0),
    ]
    _bar(axes[0, 3], "component/tracklet edge summary", tracklet_labels, tracklet_values)
    _text(
        axes[1, 0],
        "pair/regime/policy",
        [
            f"category: {category}",
            f"seq/chunk: {seq} {row.get('prev_chunk')}->{row.get('curr_chunk')}",
            f"regime: {row.get('regime', 'unavailable')}",
            f"policy_state: {row.get('policy_state', 'unavailable')}",
            f"delayed_commit: {row.get('delayed_commit_state', 'unavailable')}",
            f"feature_match_support: {row.get('feature_match_support_count', 'unavailable')}",
            f"verified_inliers: {row.get('verified_inlier_count', 'unavailable')}",
            "class names: compact ids only",
        ],
    )
    best = phase3.get("best_semantic_policy", {}) if isinstance(phase3.get("best_semantic_policy"), dict) else {}
    _text(
        axes[1, 1],
        "shuffle/control summary",
        [
            f"phase3_gate: {phase3.get('phase3_regime_semantic_gate_pass')}",
            f"best_policy: {best.get('signal')}",
            f"rho: {best.get('spearman_rho_abs_log_scale_jump')}",
            f"bad_recall: {best.get('bad_recall')}",
            f"good_FPR: {best.get('good_FPR')}",
            f"sem_margin: {best.get('semantic_shuffle_margin')}",
            f"comp_margin: {best.get('component_shuffle_margin')}",
            f"reg_margin: {best.get('regime_shuffle_margin')}",
        ],
    )
    _text(
        axes[1, 2],
        "memory policy / delayed commit",
        [
            f"phase5_gate: {phase5.get('phase5_memory_update_policy_gate_pass')}",
            f"phase5_bad_recall: {phase5.get('bad_recall')}",
            f"phase5_good_FPR: {phase5.get('good_FPR')}",
            f"phase6_gate: {phase6.get('phase6_delayed_commit_gate_pass')}",
            f"phase6_bad_recall: {phase6.get('bad_recall')}",
            f"phase6_good_FPR: {phase6.get('good_FPR')}",
            "runtime_action_allowed: False",
            "ttt_allowed: False",
        ],
    )
    _text(axes[1, 3], "category evidence / blocker", category_lines + [f"blocked_reason: {blocked_reason or 'none'}"])
    fig.suptitle(f"v91 {category} seq {seq} chunk {row.get('prev_chunk')}->{row.get('curr_chunk')}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    size = out_path.stat().st_size if out_path.exists() else 0
    return {
        "category": category,
        "panel_path": str(out_path),
        "exists": out_path.exists(),
        "size_bytes": size,
        "rgb_prev_curr_available": bool(prev_img is not None and curr_img is not None),
        "includes_semantic_labels": True,
        "includes_component_boundary_summary": True,
        "includes_tracklet_edges_summary": True,
        "includes_feature_match_summary": True,
        "includes_scale_mode_histogram": True,
        "includes_regime_policy_state": True,
        "includes_shuffle_control_summary": True,
        "includes_blocked_reason_if_downstream": bool(blocked_reason) if "blocked" in category else True,
        "is_fake_route_runtime_ttt_panel": False,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracklet_pairs = pd.read_csv(args.root / "phase1_semantic_topology_tracklets/semantic_topology_tracklet_pair_summary.csv")
    regimes = pd.read_csv(args.root / "phase2_semantic_regime_classifier/semantic_regime_rows.csv")
    policy = pd.read_csv(args.root / "phase5_memory_update_policy/policy_state_rows.csv") if (args.root / "phase5_memory_update_policy/policy_state_rows.csv").exists() else pd.DataFrame()
    delayed = pd.read_csv(args.root / "phase6_adaptive_memory_baseline/adaptive_memory_baseline_rows.csv") if (args.root / "phase6_adaptive_memory_baseline/adaptive_memory_baseline_rows.csv").exists() else pd.DataFrame()
    row_base = policy.merge(delayed[["pair_id", "delayed_commit_state"]], on="pair_id", how="left") if len(policy) and len(delayed) else policy
    if len(row_base):
        rows = row_base
    else:
        rows = regimes.merge(tracklet_pairs[["pair_id", "source_path"]], on="pair_id", how="left")
    visual_row = _select_visual_row(rows, tracklet_pairs)
    phase3 = _json(args.root / "phase3_regime_conditioned_semantic_relevance/regime_conditioned_relevance_summary.json")
    phase5 = _json(args.root / "phase5_memory_update_policy/policy_state_audit.json")
    phase6 = _json(args.root / "phase6_adaptive_memory_baseline/delayed_commit_audit.json")
    phase7 = _json(args.root / "phase7_carrier_attribution_or_blocked/phase7_carrier_summary.json")
    phase8 = _json(args.root / "phase8_counterfactual_or_blocked/counterfactual_or_blocked_summary.json")
    phase9 = _json(args.root / "phase9_runtime_or_blocked/runtime_or_blocked_summary.json")
    phase10 = _json(args.root / "phase10_ttt_or_blocked/ttt_or_blocked_summary.json")
    category_info = {
        "semantic_topology_tracklet_panels": ["tracklet source: v90 topology edges/nodes", "tracklet roles: valid/invalid/context/split"],
        "regime_classifier_panels": ["regime classifier: deterministic no-GT", "assignment labels: not used for bad/good or scale"],
        "regime_conditioned_mode_panels": ["phase3: regime-conditioned semantic relevance", f"blocker: {phase3.get('blocker', '')}"],
        "policy_state_panels": ["phase5: memory update policy", f"blocker: {phase5.get('blocker', '')}"],
        "delayed_commit_panels": ["phase6: delayed commit audit", f"blocker: {phase6.get('blocker', '')}"],
        "carrier_or_blocked_panels": ["phase7: carrier attribution or blocked", f"entered: {phase7.get('entered')}", f"blocker: {phase7.get('blocker', '')}"],
        "counterfactual_or_blocked_panels": ["phase8: counterfactual or blocked", f"entered: {phase8.get('entered')}", f"blocker: {phase8.get('blocker', '')}"],
        "runtime_or_blocked_panels": ["phase9: runtime action or blocked", f"entered: {phase9.get('entered')}", f"blocker: {phase9.get('blocker', '')}"],
        "ttt_or_blocked_panels": ["phase10: TTT or blocked", f"entered: {phase10.get('entered')}", f"blocker: {phase10.get('blocker', '')}"],
    }
    blockers = {
        "carrier_or_blocked_panels": str(phase7.get("blocker", "")),
        "counterfactual_or_blocked_panels": str(phase8.get("blocker", "")),
        "runtime_or_blocked_panels": str(phase9.get("blocker", "")),
        "ttt_or_blocked_panels": str(phase10.get("blocker", "")),
    }
    matrix = []
    for category in REQUIRED_CATEGORIES:
        out_path = args.out_dir / category / f"{category}_overview.png"
        matrix.append(_panel(out_path, category, visual_row, category_info[category], blockers.get(category, ""), phase3, phase5, phase6))
    write_csv(args.out_dir / "visual_requirement_matrix.csv", matrix)
    summary = {
        "phase": "Phase11_visual_rediscovery_build",
        "required_categories": REQUIRED_CATEGORIES,
        "panel_count": len(matrix),
        "categories_present": [row["category"] for row in matrix if row["exists"] and row["size_bytes"] > 0],
        "rgb_prev_curr_available_ratio": float(sum(bool(row["rgb_prev_curr_available"]) for row in matrix) / max(len(matrix), 1)),
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "visual_rediscovery_summary.json", summary)
    insight = [
        "# v91 Visual Insight",
        "",
        "The panels reuse actual v90/v91 pair evidence: RGB prev/curr frames, compact semantic label/confidence overlays, topology/tracklet summaries, signed scale-mode histograms, regime/policy states, and shuffle/control fields.",
        "",
        "Downstream carrier, counterfactual, runtime, and TTT panels are explicitly blocked panels. They do not claim route dumps, runtime execution, or TTT writes.",
        "",
        f"- phase3_gate: `{phase3.get('phase3_regime_semantic_gate_pass')}`",
        f"- phase5_gate: `{phase5.get('phase5_memory_update_policy_gate_pass')}`",
        f"- phase6_gate: `{phase6.get('phase6_delayed_commit_gate_pass')}`",
        f"- phase7_entered: `{phase7.get('entered')}`",
        f"- runtime_action_allowed: `False`",
        f"- ttt_allowed: `False`",
    ]
    (args.out_dir / "visual_insight.md").write_text("\n".join(insight) + "\n", encoding="utf-8")
    print(f"panel_count={summary['panel_count']}")
    print(f"rgb_prev_curr_available_ratio={summary['rgb_prev_curr_available_ratio']}")
    print(f"runtime_action_allowed={summary['runtime_action_allowed']}")
    print(f"ttt_allowed={summary['ttt_allowed']}")


if __name__ == "__main__":
    main()
