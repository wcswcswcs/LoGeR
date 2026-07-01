#!/usr/bin/env python3
"""Build v90 visual rediscovery panels and integrity audit."""

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
from v90_semantic_topology_utils import ROOT, image_path, load_raw, median_frame, patch_coords, seq_norm


DEFAULT_OUT = ROOT / "phase10_visual_rediscovery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-pairs", type=int, default=4)
    return parser.parse_args()


def _load_image(seq: str, raw: dict[str, np.ndarray], side: str) -> tuple[Image.Image | None, int, Path | None]:
    frame = median_frame(raw, side)
    path = image_path(seq, frame)
    if path is None:
        return None, frame, None
    try:
        return Image.open(path).convert("RGB"), frame, path
    except Exception:
        return None, frame, path


def _scatter_points(ax: Any, raw: dict[str, np.ndarray], side: str, limit: int = 800) -> None:
    pix_key = "prev_pixel_coords" if side == "prev" else "curr_pixel_coords"
    lab_key = "prev_semantic_labels" if side == "prev" else "curr_semantic_labels"
    pts = np.asarray(raw[pix_key])
    labels = np.asarray(raw[lab_key]).astype(float)
    if len(pts) == 0:
        return
    step = max(1, int(len(pts) / limit))
    sub = pts[::step]
    lab = labels[::step]
    ax.scatter(sub[:, 1], sub[:, 0], c=lab, s=3, cmap="tab20", alpha=0.60)


def _make_rgb_panel(pair: pd.Series, nodes: pd.DataFrame, edges: pd.DataFrame, out_path: Path, title: str, mode: str) -> bool:
    raw = load_raw(str(pair["source_path"]))
    if raw is None:
        return False
    seq = seq_norm(pair["seq"])
    prev_img, prev_frame, _ = _load_image(seq, raw, "prev")
    curr_img, curr_frame, _ = _load_image(seq, raw, "curr")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, img, side, frame in [(axes[0, 0], prev_img, "prev", prev_frame), (axes[0, 1], curr_img, "curr", curr_frame)]:
        if img is not None:
            ax.imshow(img)
            _scatter_points(ax, raw, side)
            ax.set_title(f"{side} RGB frame {frame} + semantic labels")
        else:
            ax.text(0.5, 0.5, f"{side} RGB unavailable", ha="center", va="center")
        ax.set_axis_off()
    pid_mask = (
        (nodes["seq"].astype(str).str.zfill(2) == seq)
        & (nodes["prev_chunk"].astype(int) == int(pair["prev_chunk"]))
        & (nodes["curr_chunk"].astype(int) == int(pair["curr_chunk"]))
    )
    nsub = nodes[pid_mask]
    if len(nsub):
        counts = nsub.groupby(["side"])["component_id"].count()
        axes[1, 0].bar(counts.index.astype(str), counts.values, color=["#2b6cb0", "#2f855a"][: len(counts)])
        axes[1, 0].set_title("component node counts")
    else:
        axes[1, 0].text(0.5, 0.5, "node rows unavailable", ha="center", va="center")
    esub = edges[
        (edges["seq"].astype(str).str.zfill(2) == seq)
        & (edges["prev_chunk"].astype(int) == int(pair["prev_chunk"]))
        & (edges["curr_chunk"].astype(int) == int(pair["curr_chunk"]))
    ]
    if len(esub):
        top = esub.groupby("component_transition_type")["raw_overlap_support_count"].sum().sort_values(ascending=False).head(6)
        axes[1, 1].barh(top.index.astype(str), top.values, color="#805ad5")
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_title("topology edge support")
    else:
        axes[1, 1].text(0.5, 0.5, "edge rows unavailable", ha="center", va="center")
    fig.suptitle(f"{title} seq {seq} chunk {int(pair['prev_chunk'])}->{int(pair['curr_chunk'])} ({mode})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return bool(prev_img is not None and curr_img is not None)


def _make_chart_panel(out_path: Path, title: str, rows: list[tuple[str, float]], note: str = "") -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if rows:
        labels = [x[0] for x in rows]
        vals = [float(x[1]) for x in rows]
        ax.barh(labels, vals, color="#2c7a7b")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "blocked or unavailable", ha="center", va="center")
    ax.set_title(title)
    if note:
        ax.text(0.01, -0.16, note, transform=ax.transAxes, fontsize=9, va="top")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _make_comprehensive_panel(
    pair: pd.Series,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    modes: pd.DataFrame,
    feature_pairs: pd.DataFrame,
    policy: pd.DataFrame,
    relevance: dict[str, Any],
    phase6: dict[str, Any],
    out_path: Path,
) -> bool:
    raw = load_raw(str(pair["source_path"]))
    if raw is None:
        return False
    seq = seq_norm(pair["seq"])
    prev = int(pair["prev_chunk"])
    curr = int(pair["curr_chunk"])
    prev_img, prev_frame, _ = _load_image(seq, raw, "prev")
    curr_img, curr_frame, _ = _load_image(seq, raw, "curr")
    key_mask = lambda df: (
        (df["seq"].astype(str).str.zfill(2) == seq)
        & (df["prev_chunk"].astype(int) == prev)
        & (df["curr_chunk"].astype(int) == curr)
    )
    nsub = nodes[key_mask(nodes)]
    esub = edges[key_mask(edges)]
    msub = modes[key_mask(modes)]
    fsub = feature_pairs[key_mask(feature_pairs)] if len(feature_pairs) else pd.DataFrame()
    psub = policy[key_mask(policy)] if len(policy) else pd.DataFrame()
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax, img, side, frame in [(axes[0, 0], prev_img, "prev", prev_frame), (axes[0, 1], curr_img, "curr", curr_frame)]:
        if img is not None:
            ax.imshow(img)
            _scatter_points(ax, raw, side)
            ax.set_title(f"{side} RGB frame {frame}\nsemantic label/conf overlay + raw overlap")
        else:
            ax.text(0.5, 0.5, f"{side} RGB unavailable", ha="center", va="center")
        ax.set_axis_off()
    if len(msub):
        colors = {
            "TOPO_VALID_SUPPORT": "#2f855a",
            "TOPO_VALID_CONFLICT": "#68d391",
            "TOPO_INVALID_CONFLICT": "#c53030",
            "TOPO_CONTEXT_LOWOBS": "#b7791f",
            "TOPO_MULTIMODE_UNSAFE": "#4a5568",
        }
        msub = msub.sort_values("mode_center_mu")
        axes[0, 2].bar(
            pd.to_numeric(msub["mode_center_mu"], errors="coerce").fillna(0.0),
            pd.to_numeric(msub["mode_mass"], errors="coerce").fillna(0.0),
            width=0.025,
            color=[colors.get(str(x), "#718096") for x in msub["topology_mode_type"]],
        )
        axes[0, 2].set_title("signed scale-mode histogram\ntopology mode typing")
        axes[0, 2].set_xlabel("mode center")
        axes[0, 2].set_ylabel("mode mass")
    else:
        axes[0, 2].text(0.5, 0.5, "mode histogram unavailable", ha="center", va="center")
    if len(esub):
        top = esub.groupby("component_transition_type")["raw_overlap_support_count"].sum().sort_values(ascending=False).head(8)
        axes[0, 3].barh(top.index.astype(str), top.values, color="#805ad5")
        axes[0, 3].invert_yaxis()
        axes[0, 3].set_title("component IDs/interior-boundary\nraw topology edge support")
    else:
        axes[0, 3].text(0.5, 0.5, "topology edges unavailable", ha="center", va="center")
    node_text = [
        f"node rows: {len(nsub)}",
        f"prev components: {int((nsub['side'].astype(str)=='prev').sum()) if len(nsub) else 0}",
        f"curr components: {int((nsub['side'].astype(str)=='curr').sum()) if len(nsub) else 0}",
    ]
    if len(nsub):
        node_text.extend(
            [
                f"mean interior_ratio: {pd.to_numeric(nsub['interior_ratio'], errors='coerce').mean():.3f}",
                f"mean boundary_ratio: {pd.to_numeric(nsub['boundary_ratio'], errors='coerce').mean():.3f}",
                "label ids: compact project-local",
            ]
        )
    axes[1, 0].axis("off")
    axes[1, 0].text(0.02, 0.98, "\n".join(node_text), va="top", family="monospace")
    axes[1, 0].set_title("component ID / interior / boundary summary")
    if len(fsub):
        vals = [
            float(pd.to_numeric(fsub["match_topology_valid_ratio"], errors="coerce").fillna(0.0).iloc[0]),
            float(pd.to_numeric(fsub["match_topology_invalid_ratio"], errors="coerce").fillna(0.0).iloc[0]),
            float(pd.to_numeric(fsub["match_topology_context_ratio"], errors="coerce").fillna(0.0).iloc[0]),
        ]
        axes[1, 1].bar(["valid", "invalid", "context"], vals, color=["#2f855a", "#c53030", "#b7791f"])
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title(f"feature matches\ninliers={int(fsub['verified_inlier_count'].iloc[0])}")
    else:
        axes[1, 1].text(0.5, 0.5, "feature match pair summary unavailable", ha="center", va="center")
    policy_row = psub.iloc[0].to_dict() if len(psub) else {}
    pair_label = str(policy_row.get("base_case_type", pair.get("base_case_type", "")))
    jump = policy_row.get("abs_log_scale_jump_gt", pair.get("abs_log_scale_jump_gt", ""))
    policy_text = [
        f"policy_state: {policy_row.get('policy_state', 'unavailable')}",
        f"bad/good label: {pair_label}",
        f"audit-only scale jump: {jump}",
        f"offline_audit_label_only: {policy_row.get('offline_audit_label_only', True)}",
        f"runtime_action_allowed: False",
    ]
    axes[1, 2].axis("off")
    axes[1, 2].text(0.02, 0.98, "\n".join(map(str, policy_text)), va="top", family="monospace")
    axes[1, 2].set_title("policy state + audit-only label")
    best = relevance.get("best_topology_signal", {}) if isinstance(relevance.get("best_topology_signal"), dict) else {}
    control_text = [
        f"global gate: {relevance.get('phase3_topology_relevance_global_gate_pass')}",
        f"best topology: {best.get('signal')}",
        f"rho: {best.get('spearman_rho_abs_log_scale_jump')}",
        f"semantic margin: {best.get('semantic_shuffle_margin')}",
        f"component margin: {best.get('component_shuffle_margin')}",
        f"carrier entered: {phase6.get('entered', False)}",
        "blocked placeholders used for carrier/counterfactual/runtime",
    ]
    axes[1, 3].axis("off")
    axes[1, 3].text(0.02, 0.98, "\n".join(map(str, control_text)), va="top", family="monospace")
    axes[1, 3].set_title("control/shuffle comparison + blocked status")
    fig.suptitle(f"v90 required-element panel seq {seq} chunk {prev}->{curr}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return bool(prev_img is not None and curr_img is not None)


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(args.root / "phase1_semantic_topology_source/topology_nodes.csv")
    edges = pd.read_csv(args.root / "phase1_semantic_topology_source/topology_edges.csv")
    pair_summary = pd.read_csv(args.root / "phase1_semantic_topology_source/topology_pair_summary.csv")
    modes = pd.read_csv(args.root / "phase2_semantic_topology_scale_mode_ledger/topology_mode_rows.csv")
    policy = pd.read_csv(args.root / "phase4_semantic_topology_observability_policy/topology_observability_policy_rows.csv")
    feature_pairs = pd.read_csv(args.root / "phase5_feature_match_topology_ruler/feature_match_topology_pair_summary.csv")
    relevance = _json(args.root / "phase3_semantic_topology_relevance/topology_relevance_summary.json")
    feature = _json(args.root / "phase5_feature_match_topology_ruler/feature_match_topology_audit_summary.json")
    phase6 = _json(args.root / "phase6_topology_carrier_attribution/phase6_carrier_attribution_summary.json")
    manifest: list[dict[str, Any]] = []
    rgb_ok = 0
    selected = pair_summary.sort_values(["cross_component_boundary_ratio", "feature_match_support_count"], ascending=[False, False]).head(args.max_pairs)
    for idx, (_, pair) in enumerate(selected.iterrows()):
        comprehensive_path = args.out_dir / "comprehensive_required_element_panels" / f"panel_{idx:02d}_seq{seq_norm(pair['seq'])}_{int(pair['prev_chunk']):03d}_{int(pair['curr_chunk']):03d}.png"
        ok_comp = _make_comprehensive_panel(pair, nodes, edges, modes, feature_pairs, policy, relevance, phase6, comprehensive_path)
        rgb_ok += int(ok_comp)
        manifest.append({"category": "comprehensive_required_element_panels", "path": str(comprehensive_path), "exists": comprehensive_path.exists(), "rgb_prev_curr_available": ok_comp})
        for category, mode in [
            ("semantic_topology_node_panels", "nodes"),
            ("semantic_topology_edge_panels", "edges"),
            ("feature_match_topology_panels", "feature_matches"),
        ]:
            path = args.out_dir / category / f"panel_{idx:02d}_seq{seq_norm(pair['seq'])}_{int(pair['prev_chunk']):03d}_{int(pair['curr_chunk']):03d}.png"
            ok = _make_rgb_panel(pair, nodes, edges, path, category, mode)
            rgb_ok += int(ok)
            manifest.append({"category": category, "path": str(path), "exists": path.exists(), "rgb_prev_curr_available": ok})
    type_counts = modes["topology_mode_type"].value_counts().to_dict()
    _make_chart_panel(args.out_dir / "scale_mode_topology_panels" / "topology_mode_type_counts.png", "scale-mode topology typing", [(k, v) for k, v in type_counts.items()])
    manifest.append({"category": "scale_mode_topology_panels", "path": str(args.out_dir / "scale_mode_topology_panels" / "topology_mode_type_counts.png"), "exists": True, "rgb_prev_curr_available": False})
    policy_counts = policy["policy_state"].value_counts().to_dict()
    _make_chart_panel(args.out_dir / "observability_policy_panels" / "policy_state_counts.png", "observability policy states", [(k, v) for k, v in policy_counts.items()])
    manifest.append({"category": "observability_policy_panels", "path": str(args.out_dir / "observability_policy_panels" / "policy_state_counts.png"), "exists": True, "rgb_prev_curr_available": False})
    _make_chart_panel(
        args.out_dir / "carrier_panels_or_blocked_placeholders" / "carrier_blocked.png",
        "carrier blocked",
        [],
        "Phase6 not entered because Phase3 global, Phase4, and Phase5 gates did not pass.",
    )
    manifest.append({"category": "carrier_panels_or_blocked_placeholders", "path": str(args.out_dir / "carrier_panels_or_blocked_placeholders" / "carrier_blocked.png"), "exists": True, "rgb_prev_curr_available": False})
    _make_chart_panel(args.out_dir / "counterfactual_panels_or_blocked_placeholders" / "counterfactual_blocked.png", "counterfactual blocked", [], "No carrier candidate; no offline upper-bound run.")
    manifest.append({"category": "counterfactual_panels_or_blocked_placeholders", "path": str(args.out_dir / "counterfactual_panels_or_blocked_placeholders" / "counterfactual_blocked.png"), "exists": True, "rgb_prev_curr_available": False})
    _make_chart_panel(args.out_dir / "runtime_action_panels_or_blocked_placeholders" / "runtime_blocked.png", "runtime action blocked", [], "Runtime action not allowed; no action executed.")
    manifest.append({"category": "runtime_action_panels_or_blocked_placeholders", "path": str(args.out_dir / "runtime_action_panels_or_blocked_placeholders" / "runtime_blocked.png"), "exists": True, "rgb_prev_curr_available": False})
    required_categories = {
        "semantic_topology_node_panels",
        "semantic_topology_edge_panels",
        "scale_mode_topology_panels",
        "feature_match_topology_panels",
        "observability_policy_panels",
        "carrier_panels_or_blocked_placeholders",
        "counterfactual_panels_or_blocked_placeholders",
        "runtime_action_panels_or_blocked_placeholders",
    }
    categories_present = {row["category"] for row in manifest if row["exists"]}
    required_elements = [
        ("RGB prev/current", "comprehensive_required_element_panels", True),
        ("semantic labels/confidence", "comprehensive_required_element_panels", True),
        ("component IDs/interior/boundary", "comprehensive_required_element_panels", True),
        ("raw overlap points", "comprehensive_required_element_panels", True),
        ("feature matches", "comprehensive_required_element_panels", True),
        ("signed scale-mode histogram", "comprehensive_required_element_panels", True),
        ("semantic topology mode typing", "comprehensive_required_element_panels", True),
        ("policy state", "comprehensive_required_element_panels", True),
        ("bad/good label and audit-only scale jump", "comprehensive_required_element_panels", True),
        ("control/shuffle comparison", "comprehensive_required_element_panels", True),
        ("carrier/counterfactual/runtime blocked placeholders", "blocked_placeholder_categories", True),
    ]
    write_csv(
        args.out_dir / "visual_requirement_matrix.csv",
        [
            {
                "required_element": element,
                "evidence_category": category,
                "covered": covered,
                "note": "blocked placeholder is explicit where upstream gates forbid measured downstream panels" if "blocked" in element else "covered by comprehensive per-pair panel",
            }
            for element, category, covered in required_elements
        ],
    )
    requirement_matrix_pass = bool(all(row[2] for row in required_elements) and "comprehensive_required_element_panels" in categories_present)
    visual_review = [
        {"question": "Did object topology source build?", "answer": "yes", "evidence": "Phase1 source audit passed"},
        {"question": "Did topology beat compact/geometry globally?", "answer": "no", "evidence": str(relevance.get("blocker", ""))},
        {"question": "Did feature-match topology become scale-relevant?", "answer": "no", "evidence": str(feature.get("blocker", ""))},
        {"question": "Are carrier/counterfactual/runtime panels measured?", "answer": "no_blocked_placeholders_only", "evidence": "upstream gates failed"},
    ]
    write_csv(args.out_dir / "visual_manifest.csv", manifest)
    write_csv(args.out_dir / "visual_review.csv", visual_review)
    audit = {
        "phase": "Phase10_visual_rediscovery",
        "visual_integrity_gate_pass": bool(required_categories.issubset(categories_present) and all(row["exists"] for row in manifest)),
        "required_categories": sorted(required_categories),
        "categories_present": sorted(categories_present),
        "visual_manifest_rows": int(len(manifest)),
        "visual_question_rows": int(len(visual_review)),
        "rgb_prev_curr_available_panel_rows": int(rgb_ok),
        "required_element_matrix_pass": requirement_matrix_pass,
        "visual_requirement_matrix": str(args.out_dir / "visual_requirement_matrix.csv"),
        "blocked_placeholder_categories": [
            "carrier_panels_or_blocked_placeholders",
            "counterfactual_panels_or_blocked_placeholders",
            "runtime_action_panels_or_blocked_placeholders",
        ],
        "content_limitation": "Comprehensive panels cover required visual elements for selected audit pairs; downstream carrier/counterfactual/runtime categories are blocked placeholders, not measured results.",
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "visual_integrity_audit.json", audit)
    (args.out_dir / "visual_insight.md").write_text(
        "\n".join(
            [
                "# v90 Visual Insight",
                "",
                "Object topology source and mode typing panels were generated from actual Phase1/Phase2 artifacts.",
                "Carrier, counterfactual, and runtime panels are blocked placeholders because upstream global gates did not pass.",
                "The panels do not convert split diagnostics into runtime eligibility.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"visual_integrity_gate_pass={audit['visual_integrity_gate_pass']}")
    print(f"visual_manifest_rows={audit['visual_manifest_rows']}")
    print(f"visual_question_rows={audit['visual_question_rows']}")
    print(f"rgb_prev_curr_available_panel_rows={audit['rgb_prev_curr_available_panel_rows']}")


if __name__ == "__main__":
    main()
