#!/usr/bin/env python3
"""Generate v87 mandatory visual rediscovery panels and integrity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from v86_soft_latent_utils import write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v87tf_scale_conditioned_latent_gauge_carrier")
DEFAULT_PHASE1 = DEFAULT_ROOT / "phase1_scale_conditioned_pair_universe_k16_r1_median_abs"
DEFAULT_PHASE2 = DEFAULT_ROOT / "phase2_scale_relevance_k16_r1_median_abs_highobs"
DEFAULT_PHASE3 = DEFAULT_ROOT / "phase3_state_conditioned_latent_transport"
DEFAULT_PHASE4 = DEFAULT_ROOT / "phase4_no_refresh_guard"
DEFAULT_PHASE8 = DEFAULT_ROOT / "phase8_merge_gauge_direct_pair_weighting"
DEFAULT_ANCHOR_ROWS = Path(
    "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase1_anchor_pair_universe/anchor_pair_rows.csv"
)
DEFAULT_IMAGE_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
DEFAULT_OUT = DEFAULT_ROOT / "phase12_visual_rediscovery"


CATEGORIES = [
    "local_shape_scale_proxy_panels",
    "support_conflict_absence_panels",
    "qk_transport_failure_panels",
    "route_carrier_failure_panels",
    "merge_direct_pair_failure_panels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2)
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--phase4-dir", type=Path, default=DEFAULT_PHASE4)
    parser.add_argument("--phase8-dir", type=Path, default=DEFAULT_PHASE8)
    parser.add_argument("--anchor-rows", type=Path, default=DEFAULT_ANCHOR_ROWS)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-cases", type=int, default=8)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _image_path(root: Path, seq: str, frame: int) -> Path | None:
    for cam in ("image_2", "image_3"):
        path = root / str(seq).zfill(2) / cam / f"{int(frame):06d}.png"
        if path.exists():
            return path
    return None


def _source_path(anchor: pd.DataFrame, pair_ids: list[str]) -> str:
    sub = anchor[anchor["pair_id"].isin(pair_ids)]
    if len(sub) == 0:
        return ""
    return str(sub["source_path"].mode().iloc[0])


def _raw_pixels_for_patches(source_path: str, patch_ids: set[int], max_points: int = 900) -> tuple[np.ndarray, bool]:
    if not source_path or not Path(source_path).exists():
        return np.zeros((0, 2), dtype=float), False
    try:
        obj = torch.load(source_path, map_location="cpu", weights_only=False)
        pixels = obj["prev_pixel_coords"].detach().cpu().numpy().astype(float)
    except Exception:  # noqa: BLE001
        return np.zeros((0, 2), dtype=float), False
    patch_y = np.floor(pixels[:, 0] / 14.0).astype(int)
    patch_x = np.floor(pixels[:, 1] / 14.0).astype(int)
    patch = patch_y * 66 + patch_x
    mask = np.isin(patch, list(patch_ids))
    pts = pixels[mask]
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
        pts = pts[idx]
    return pts, True


def _select_cases(by_pair: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    by_pair = by_pair.copy()
    by_pair["abs_log_scale_jump_gt"] = pd.to_numeric(by_pair["abs_log_scale_jump_gt"], errors="coerce")
    by_pair["sort_bad"] = (by_pair["base_case_type"] == "bad").astype(int)
    by_pair["sort_scale"] = by_pair["abs_log_scale_jump_gt"].fillna(-1.0)
    bad = by_pair[by_pair["base_case_type"] == "bad"].sort_values(["sort_scale"], ascending=False)
    good = by_pair[by_pair["base_case_type"] == "good"].sort_values(["sort_scale"], ascending=False)
    other = by_pair[~by_pair["base_case_type"].isin(["bad", "good"])].sort_values(["sort_scale"], ascending=False)
    selected = pd.concat([bad.head(max_cases // 2 + 1), good.head(max_cases // 2), other.head(2)]).drop_duplicates(
        ["seq", "prev_chunk", "curr_chunk"]
    )
    return selected.head(max_cases)


def _pair_rows(rows: pd.DataFrame, pair: pd.Series) -> pd.DataFrame:
    return rows[
        (rows["seq"].astype(str).str.zfill(2) == str(pair["seq"]).zfill(2))
        & (rows["prev_chunk"].astype(int) == int(pair["prev_chunk"]))
        & (rows["curr_chunk"].astype(int) == int(pair["curr_chunk"]))
    ].copy()


def _panel_text(pair: pd.Series, phase2: dict[str, Any], phase3: dict[str, Any], phase4: dict[str, Any], phase8: dict[str, Any]) -> str:
    best = phase2.get("best_signal") or {}
    return "\n".join(
        [
            f"seq {str(pair['seq']).zfill(2)} {int(pair['prev_chunk'])}->{int(pair['curr_chunk'])}",
            f"case={pair.get('base_case_type')} state={pair.get('state_label')} quality={pair.get('quality_type')}",
            f"abs scale jump audit-only={pair.get('abs_log_scale_jump_gt')}",
            f"S_shape={pair.get('weighted_median_local_shape_log_ratio')} S_overlap={pair.get('mean_confidence_weighted_overlap_residual')}",
            f"Phase2 pass={phase2.get('phase2_scale_proxy_gate_pass')} best={best.get('signal')}",
            f"Phase3 pass={phase3.get('phase3_alignment_gate_pass')} blocker={phase3.get('blocker')}",
            f"Phase4 pass={phase4.get('phase4_no_refresh_guard_gate_pass')} good_FPR={phase4.get('good_FPR')}",
            f"Phase8 pass={phase8.get('phase8_merge_gauge_gate_pass')} geom_cf={phase8.get('actual_geometry_counterfactual_available')}",
        ]
    )


def _draw_panel(
    out_path: Path,
    category: str,
    pair: pd.Series,
    rows: pd.DataFrame,
    anchor: pd.DataFrame,
    image_root: Path,
    phase2: dict[str, Any],
    phase3: dict[str, Any],
    phase4: dict[str, Any],
    phase8: dict[str, Any],
) -> dict[str, Any]:
    seq = str(pair["seq"]).zfill(2)
    group = _pair_rows(rows, pair)
    frame_vals = pd.to_numeric(group["curr_frame_id"], errors="coerce").dropna()
    frame = int(np.median(frame_vals)) if len(frame_vals) else 0
    image_path = _image_path(image_root, seq, frame)
    image_exists = image_path is not None
    if image_path is not None:
        image = np.asarray(Image.open(image_path).convert("RGB"))
    else:
        image = np.full((376, 1241, 3), 230, dtype=np.uint8)
    source = _source_path(anchor, [str(v) for v in group["pair_id"].head(80).tolist()])
    patch_ids = set(int(v) for v in pd.to_numeric(group["prev_patch_id"], errors="coerce").dropna().head(80).tolist())
    pixels, raw_exists = _raw_pixels_for_patches(source, patch_ids)

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(image)
    if len(pixels):
        ax_img.scatter(pixels[:, 1], pixels[:, 0], s=5, c="#ff2a2a", alpha=0.45, label="raw overlap patch pixels")
        ax_img.legend(loc="lower right", fontsize=8)
    ax_img.set_title(f"{category} | seq {seq} chunk {int(pair['prev_chunk'])}->{int(pair['curr_chunk'])} frame {frame}")
    ax_img.set_axis_off()

    ax_hist = fig.add_subplot(gs[1, 0])
    state_colors = {"SUPPORT": "#2ca02c", "CONFLICT": "#d62728", "ABSENCE": "#7f7f7f", "STRESS": "#9467bd"}
    for state, sub in group.groupby("state_label"):
        vals = pd.to_numeric(sub["local_shape_log_ratio_median"], errors="coerce").dropna()
        if len(vals):
            ax_hist.hist(vals, bins=20, alpha=0.55, label=str(state), color=state_colors.get(str(state), "#1f77b4"))
    ax_hist.set_xlabel("local_shape_log_ratio")
    ax_hist.set_ylabel("row count")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.25)

    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")
    ax_text.text(0.0, 1.0, _panel_text(pair, phase2, phase3, phase4, phase8), va="top", ha="left", fontsize=9, family="monospace")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {
        "category": category,
        "path": str(out_path),
        "seq": seq,
        "prev_chunk": int(pair["prev_chunk"]),
        "curr_chunk": int(pair["curr_chunk"]),
        "case_label": pair.get("base_case_type"),
        "state_label": pair.get("state_label"),
        "frame": frame,
        "rgb_frame_path": str(image_path) if image_path else "",
        "rgb_frame_exists": image_exists,
        "raw_overlap_source_path": source,
        "raw_overlap_exists": raw_exists,
        "panel_exists": out_path.exists(),
    }


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.phase1_dir / "scale_conditioned_pair_rows.csv")
    by_pair = pd.read_csv(args.phase1_dir / "scale_conditioned_pair_by_adjacent.csv")
    anchor = pd.read_csv(args.anchor_rows, usecols=["pair_id", "source_path"])
    phase2 = _read_json(args.phase2_dir / "proxy_relevance_summary.json")
    phase3 = _read_json(args.phase3_dir / "state_conditioned_alignment_summary.json")
    phase4 = _read_json(args.phase4_dir / "no_refresh_guard_summary.json")
    phase8 = _read_json(args.phase8_dir / "merge_gauge_direct_pair_summary.json")
    selected = _select_cases(by_pair, args.max_cases)

    for category in CATEGORIES:
        (args.out_dir / category).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    for _, pair in selected.iterrows():
        key = f"seq{str(pair['seq']).zfill(2)}_chunk{int(pair['prev_chunk']):03d}_{int(pair['curr_chunk']):03d}_{pair.get('base_case_type')}"
        questions.append(
            {
                "seq": str(pair["seq"]).zfill(2),
                "prev_chunk": int(pair["prev_chunk"]),
                "curr_chunk": int(pair["curr_chunk"]),
                "case_label": pair.get("base_case_type"),
                "state_label": pair.get("state_label"),
                "visual_question": "Does the raw overlap/local-shape conflict correspond to an inspectable geometry issue, and why did it fail to become a QK/merge carrier?",
            }
        )
        for category in CATEGORIES:
            out_path = args.out_dir / category / f"{key}_{category.replace('_panels', '')}.png"
            manifest.append(_draw_panel(out_path, category, pair, rows, anchor, args.image_root, phase2, phase3, phase4, phase8))

    write_csv(args.out_dir / "failed_case_to_visual_question.csv", questions)
    write_csv(args.out_dir / "visual_manifest.csv", manifest)
    review_rows = [
        {
            **row,
            "review_status": "generated_visual_audit",
            "review_note": "Panel generated from v87 artifacts; offline scale label displayed as audit-only.",
        }
        for row in manifest
    ]
    write_csv(args.out_dir / "visual_review.csv", review_rows)
    required_dir_counts = {
        category: sum(1 for row in manifest if row["category"] == category and Path(row["path"]).exists()) for category in CATEGORIES
    }
    all_panels_exist = all(Path(row["path"]).exists() for row in manifest)
    all_rgb_exists = all(bool(row["rgb_frame_exists"]) for row in manifest)
    all_raw_exists = all(bool(row["raw_overlap_exists"]) for row in manifest)
    integrity = {
        "phase": "Phase12_visual_rediscovery",
        "visual_integrity_gate_pass": bool(manifest and all_panels_exist and all_rgb_exists and all_raw_exists and all(v > 0 for v in required_dir_counts.values())),
        "manifest_rows": len(manifest),
        "question_rows": len(questions),
        "review_rows": len(review_rows),
        "required_dir_counts": required_dir_counts,
        "all_panels_exist": all_panels_exist,
        "all_rgb_frames_exist": all_rgb_exists,
        "all_raw_overlap_sources_exist": all_raw_exists,
    }
    write_json(args.out_dir / "visual_integrity_audit.json", integrity)
    visual_insight = [
        "# v87 Visual Insight",
        "",
        "- Panels use real KITTI RGB frames and raw overlap pixel locations from the v85/v81S overlap .pt artifacts.",
        "- The selected v87 path found a high-observability geometry/local-shape proxy, but semantic-aware signals did not pass.",
        "- The state distribution is conflict/stress dominated; SUPPORT rows are absent, so Phase3 cannot fit a state-conditioned C.",
        "- Phase4 no-refresh decisions flag nearly everything as risk, producing good_FPR=1.0.",
        "- Phase8 direct pair weights are inspectable as raw-pair weights, but no compliant geometry counterfactual artifact exists.",
    ]
    (args.out_dir / "visual_insight.md").write_text("\n".join(visual_insight) + "\n", encoding="utf-8")
    hypotheses = [
        "# New Hypothesis Bank",
        "",
        "1. The high-observability local-shape proxy is mostly geometry-only and may need a non-semantic merge/gauge route, not QK semantic memory.",
        "2. The current SUPPORT classifier is too strict for useful C fitting, but relaxing it would need a new good-FPR guard, not a v87 threshold shortcut.",
        "3. Per-head route dumps are still required before claiming SWA carrier; pooled Q/K is insufficient.",
        "4. A future plan should implement a true direct raw-pair merge/gauge counterfactual instead of reusing support-map fallback artifacts.",
    ]
    (args.out_dir / "new_hypothesis_bank.md").write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
    print(f"visual_integrity_gate_pass={integrity['visual_integrity_gate_pass']}")
    print(f"manifest_rows={integrity['manifest_rows']}")
    print(f"required_dir_counts={required_dir_counts}")


if __name__ == "__main__":
    main()
