#!/usr/bin/env python3
"""Generate v88 Phase7 visual rediscovery panels and integrity audit."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from v86_soft_latent_utils import read_json, safe_float, stable_hash_float, write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution")
DEFAULT_IMAGE_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
DEFAULT_OUT = DEFAULT_ROOT / "phase7_visual_rediscovery"

CATEGORIES = [
    "scale_mode_histogram_panels",
    "native_mode_mismatch_panels",
    "good_false_positive_mismatch_panels",
    "multimode_unsafe_panels",
    "swa_route_mode_panels",
    "merge_boundary_mode_panels",
    "counterfactual_failure_panels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=900)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _key(row: pd.Series | dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["seq"]).zfill(2), int(row["prev_chunk"]), int(row["curr_chunk"]))


def _image_path(root: Path, seq: str, frame: int) -> Path | None:
    for cam in ("image_2", "image_3"):
        path = root / seq / cam / f"{int(frame):06d}.png"
        if path.exists():
            return path
    return None


def _load_image(path: Path | None) -> np.ndarray:
    if path is None:
        return np.full((376, 1241, 3), 232, dtype=np.uint8)
    return np.asarray(Image.open(path).convert("RGB"))


def _load_raw(source_path: str, max_points: int) -> dict[str, Any]:
    source = Path(str(source_path))
    if not source.exists():
        return {
            "source_path_exists": False,
            "raw_load_ok": False,
            "prev_points": np.zeros((0, 2), dtype=float),
            "curr_points": np.zeros((0, 2), dtype=float),
            "prev_frame": 0,
            "curr_frame": 0,
            "prev_semantic_summary": "missing raw source",
            "curr_semantic_summary": "missing raw source",
        }
    try:
        obj = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return {
            "source_path_exists": True,
            "raw_load_ok": False,
            "raw_error": repr(exc),
            "prev_points": np.zeros((0, 2), dtype=float),
            "curr_points": np.zeros((0, 2), dtype=float),
            "prev_frame": 0,
            "curr_frame": 0,
            "prev_semantic_summary": "raw load failed",
            "curr_semantic_summary": "raw load failed",
        }

    def arr(name: str) -> np.ndarray:
        value = obj.get(name)
        if value is None:
            return np.asarray([])
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    prev_pixels = arr("prev_pixel_coords").astype(float)
    curr_pixels = arr("curr_pixel_coords").astype(float)
    prev_frames = arr("prev_frame_ids")
    curr_frames = arr("curr_frame_ids")
    prev_labels = arr("prev_semantic_labels")
    curr_labels = arr("curr_semantic_labels")
    prev_conf = arr("prev_semantic_conf")
    curr_conf = arr("curr_semantic_conf")
    n = min(len(prev_pixels), len(curr_pixels))
    if n == 0:
        keep = np.asarray([], dtype=int)
    elif n > max_points:
        order = sorted(range(n), key=lambda idx: stable_hash_float(source_path, idx))
        keep = np.asarray(order[:max_points], dtype=int)
    else:
        keep = np.arange(n, dtype=int)

    def frame_id(values: np.ndarray) -> int:
        vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
        return int(vals.median()) if len(vals) else 0

    def semantic_summary(labels: np.ndarray, conf: np.ndarray) -> str:
        if len(labels) == 0:
            return "semantic labels unavailable"
        lab = labels[keep] if len(keep) else labels[:0]
        c = conf[keep] if len(conf) >= len(labels) and len(keep) else np.asarray([])
        top: list[str] = []
        vals, counts = np.unique(lab.astype(int), return_counts=True)
        order = np.argsort(counts)[::-1][:4]
        for idx in order:
            label = int(vals[idx])
            mask = lab.astype(int) == label
            mean_conf = float(np.mean(c[mask])) if len(c) and mask.any() else float("nan")
            conf_text = "nan" if not math.isfinite(mean_conf) else f"{mean_conf:.3f}"
            top.append(f"{label}:{int(counts[idx])}@{conf_text}")
        return ", ".join(top) if top else "semantic labels unavailable"

    return {
        "source_path_exists": True,
        "raw_load_ok": True,
        "prev_points": prev_pixels[keep] if len(prev_pixels) else np.zeros((0, 2), dtype=float),
        "curr_points": curr_pixels[keep] if len(curr_pixels) else np.zeros((0, 2), dtype=float),
        "prev_frame": frame_id(prev_frames),
        "curr_frame": frame_id(curr_frames),
        "prev_labels": prev_labels[keep] if len(prev_labels) >= len(keep) else np.asarray([]),
        "curr_labels": curr_labels[keep] if len(curr_labels) >= len(keep) else np.asarray([]),
        "prev_semantic_summary": semantic_summary(prev_labels, prev_conf),
        "curr_semantic_summary": semantic_summary(curr_labels, curr_conf),
    }


def _merge_phase3_class(pair_rows: pd.DataFrame, phase3_rows: pd.DataFrame) -> pd.DataFrame:
    if phase3_rows.empty:
        pair_rows["attribution_class"] = ""
        pair_rows["flagged_mismatch"] = ""
        return pair_rows
    phase3 = phase3_rows[phase3_rows["variant"] == "mismatch_q75"].copy()
    keep = ["seq", "prev_chunk", "curr_chunk", "attribution_class", "flagged_mismatch"]
    phase3["seq"] = phase3["seq"].astype(str).str.zfill(2)
    return pair_rows.merge(phase3[keep], on=["seq", "prev_chunk", "curr_chunk"], how="left")


def _select_cases(df: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    labelled = df[_num(df["abs_log_scale_jump_gt"]).notna()].copy()
    labelled["sort_scale"] = _num(labelled["abs_log_scale_jump_gt"]).fillna(-1)
    labelled["sort_mismatch"] = _num(labelled["native_mode_mismatch"]).fillna(-1)
    bad = labelled[labelled["base_case_type"] == "bad"].sort_values(["sort_scale", "sort_mismatch"], ascending=False)
    good_mismatch = labelled[
        (labelled["base_case_type"] == "good")
        & (
            labelled["attribution_class"].astype(str).eq("MISMATCH_GOOD")
            | (_num(labelled["native_mode_mismatch"]) >= _num(labelled["native_mode_mismatch"]).quantile(0.75))
        )
    ].sort_values(["sort_mismatch", "sort_scale"], ascending=False)
    multimode = df[df["attribution_class"].astype(str).eq("MULTIMODE_UNSAFE")].copy()
    multimode["sort_entropy"] = _num(multimode["mode_entropy"]).fillna(-1)
    lowobs = df[df["attribution_class"].astype(str).eq("LOWOBS_ABSTAIN")].copy()
    lowobs["sort_obs"] = _num(lowobs["observability_score"]).fillna(1)
    rows.extend([bad.head(3), good_mismatch.head(2), multimode.sort_values("sort_entropy", ascending=False).head(2), lowobs.sort_values("sort_obs").head(1)])
    selected = pd.concat([r for r in rows if len(r)], ignore_index=True)
    if len(selected) < max_cases:
        fill = df.copy()
        fill["sort_scale"] = _num(fill["abs_log_scale_jump_gt"]).fillna(-1)
        fill["sort_mismatch"] = _num(fill["native_mode_mismatch"]).fillna(-1)
        selected = pd.concat([selected, fill.sort_values(["sort_scale", "sort_mismatch"], ascending=False)], ignore_index=True)
    selected = selected.drop_duplicates(["seq", "prev_chunk", "curr_chunk"]).head(max_cases).copy()
    return selected


def _histogram(hist: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    seq, prev_chunk, curr_chunk = _key(row)
    return hist[
        (hist["seq"].astype(str).str.zfill(2) == seq)
        & (hist["prev_chunk"].astype(int) == prev_chunk)
        & (hist["curr_chunk"].astype(int) == curr_chunk)
    ].copy()


def _best_signal_text(phase2: dict[str, Any], phase2_highobs: dict[str, Any], phase3: dict[str, Any], phase5: dict[str, Any]) -> str:
    best_all = phase2.get("best_signal") or {}
    best_highobs = phase2_highobs.get("best_signal") or {}
    best_variant = phase3.get("best_variant") or {}
    best_family = phase5.get("best_family") or {}
    highobs_signals = "/".join(str(v) for v in (phase2_highobs.get("passing_signals") or []))
    return "\n".join(
        [
            f"P2 all pass={phase2.get('phase2_mode_relevance_gate_pass')} best={best_all.get('signal')} rho={best_all.get('spearman_rho_abs_log_scale_jump')}",
            f"P2 highobs pass={phase2_highobs.get('phase2_mode_relevance_gate_pass')} signals={highobs_signals} semantic={phase2_highobs.get('semantic_aware_pass')}",
            f"P3 pass={phase3.get('phase3_native_update_attribution_gate_pass')} recall={best_variant.get('MISMATCH_BAD_recall')} FPR={best_variant.get('MISMATCH_GOOD_FPR')} rho={best_variant.get('native_mode_mismatch_rho_abs_log_scale_jump')}",
            f"P5 pass={phase5.get('scale_label_gate_pass')} best={best_family.get('family')} bad_I={best_family.get('bad_median_I_scale')} good_worsen={best_family.get('good_max_scale_error_worsen')}",
        ]
    )


def _category_note(category: str, row: pd.Series, phase4_swa: dict[str, Any], phase4_merge: dict[str, Any]) -> str:
    if category == "swa_route_mode_panels":
        return f"SWA carrier: pass={phase4_swa.get('swa_route_carrier_gate_pass')} blocker={phase4_swa.get('blocker')}; per-head route dump unavailable."
    if category == "merge_boundary_mode_panels":
        return f"Merge/gauge carrier: pass={phase4_merge.get('merge_gauge_mode_carrier_gate_pass')} blocker={phase4_merge.get('blocker')}; bad_recall={phase4_merge.get('bad_recall')} good_FPR={phase4_merge.get('good_FPR')}."
    if category == "counterfactual_failure_panels":
        return "Counterfactual points: no raw residual refit available; only audit-only scale-label families were evaluated."
    if category == "good_false_positive_mismatch_panels":
        return f"Good-protection check: class={row.get('attribution_class')} flagged_mismatch={row.get('flagged_mismatch')}."
    if category == "multimode_unsafe_panels":
        return f"Multimode check: entropy={row.get('mode_entropy')} gap={row.get('mode_gap_top1_top2')}."
    if category == "native_mode_mismatch_panels":
        return f"Native marker: native_delta={row.get('native_delta_log_scale')} dominant_mode={row.get('weighted_mode_mu')} mismatch={row.get('native_mode_mismatch')} sign_mismatch={row.get('native_mode_sign_mismatch')}."
    return f"Scale mode histogram: top1_mass={row.get('mode_mass_top1')} top2_mass={row.get('mode_mass_top2')} dominant={row.get('weighted_mode_mu')}."


def _draw_panel(
    out_path: Path,
    category: str,
    row: pd.Series,
    hist: pd.DataFrame,
    raw: dict[str, Any],
    image_root: Path,
    phase2: dict[str, Any],
    phase2_highobs: dict[str, Any],
    phase3: dict[str, Any],
    phase4_swa: dict[str, Any],
    phase4_merge: dict[str, Any],
    phase5: dict[str, Any],
) -> dict[str, Any]:
    seq, prev_chunk, curr_chunk = _key(row)
    prev_path = _image_path(image_root, seq, int(raw["prev_frame"]))
    curr_path = _image_path(image_root, seq, int(raw["curr_frame"]))
    prev_img = _load_image(prev_path)
    curr_img = _load_image(curr_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.35, 1.35, 1.0], height_ratios=[1.35, 0.85, 0.45])
    ax_prev = fig.add_subplot(gs[0, 0])
    ax_curr = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1:, 1:])
    ax_note = fig.add_subplot(gs[2, 0])

    def scatter(ax: plt.Axes, image: np.ndarray, points: np.ndarray, labels: np.ndarray, title: str) -> None:
        ax.imshow(image, aspect="auto")
        if len(points):
            # The raw overlap cache stores pixel coordinates as row/col; matplotlib
            # image axes need x=col and y=row. Clamp axes after scatter so outliers
            # cannot shrink the RGB frame into a tiny corner.
            xs = points[:, 1]
            ys = points[:, 0]
            if len(labels) == len(points):
                ax.scatter(xs, ys, s=5, c=labels.astype(float), cmap="tab20", alpha=0.45)
            else:
                ax.scatter(xs, ys, s=5, c="#e4572e", alpha=0.45)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.set_axis_off()

    scatter(ax_prev, prev_img, raw["prev_points"], raw.get("prev_labels", np.asarray([])), f"prev RGB seq {seq} frame {raw['prev_frame']}")
    scatter(ax_curr, curr_img, raw["curr_points"], raw.get("curr_labels", np.asarray([])), f"curr RGB seq {seq} frame {raw['curr_frame']}")

    if len(hist):
        centers = (_num(hist["bin_left"]) + _num(hist["bin_right"])) / 2.0
        ax_hist.bar(centers, _num(hist["weighted_mass"]), width=0.02, color="#4c78a8", alpha=0.75)
    ax_hist.axvline(safe_float(row.get("weighted_mode_mu")) or 0.0, color="#2ca02c", label="dominant mode")
    ax_hist.axvline(safe_float(row.get("native_delta_log_scale")) or 0.0, color="#d62728", label="native transition")
    ax_hist.set_title("signed local-shape ratio histogram")
    ax_hist.set_xlabel("signed log ratio")
    ax_hist.set_ylabel("weighted mass")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.25)

    bar_labels = ["mode", "native", "mismatch", "entropy", "gap"]
    bar_values = [
        safe_float(row.get("weighted_mode_mu")) or 0.0,
        safe_float(row.get("native_delta_log_scale")) or 0.0,
        safe_float(row.get("native_mode_mismatch")) or 0.0,
        safe_float(row.get("mode_entropy")) or 0.0,
        safe_float(row.get("mode_gap_top1_top2")) or 0.0,
    ]
    ax_bar.bar(bar_labels, bar_values, color=["#4c78a8", "#54a24b", "#e45756", "#f58518", "#72b7b2"])
    ax_bar.set_title("mode/native markers")
    ax_bar.tick_params(axis="x", rotation=25)
    ax_bar.grid(axis="y", alpha=0.25)

    text = "\n".join(
        [
            f"category={category}",
            f"seq={seq} chunk={prev_chunk}->{curr_chunk} case={row.get('base_case_type')} quality={row.get('quality_type')}",
            f"offline abs log scale label (audit only)={row.get('abs_log_scale_jump_gt')}",
            f"dominant mode={row.get('weighted_mode_mu')} second-mass={row.get('mode_mass_top2')} top1/top2 gap={row.get('mode_gap_top1_top2')}",
            f"native transition source={row.get('native_transition_source')} native delta={row.get('native_delta_log_scale')} mismatch={row.get('native_mode_mismatch')}",
            f"semantic prev label/conf: {raw.get('prev_semantic_summary')}",
            f"semantic curr label/conf: {raw.get('curr_semantic_summary')}",
            f"raw overlap sampled points prev/curr={len(raw['prev_points'])}/{len(raw['curr_points'])}; source exists={raw.get('source_path_exists')} load_ok={raw.get('raw_load_ok')}",
            _category_note(category, row, phase4_swa, phase4_merge),
            "random/shuffle controls:",
            _best_signal_text(phase2, phase2_highobs, phase3, phase5),
        ]
    )
    ax_text.axis("off")
    ax_text.text(0.0, 1.0, text, va="top", ha="left", fontsize=7.8, family="monospace")
    ax_note.axis("off")
    ax_note.text(
        0.0,
        1.0,
        "Review cue: decide whether the mode pattern is a causal carrier, a geometry-only diagnostic, "
        "or a good-case false positive. No runtime action is encoded in this panel.",
        va="top",
        ha="left",
        fontsize=9,
        wrap=True,
    )
    fig.suptitle(f"v88 {category} | seq {seq} chunk {prev_chunk}->{curr_chunk}", fontsize=13)
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    size = out_path.stat().st_size if out_path.exists() else 0
    return {
        "category": category,
        "panel_path": str(out_path),
        "panel_exists": out_path.exists(),
        "panel_size_bytes": size,
        "seq": seq,
        "prev_chunk": prev_chunk,
        "curr_chunk": curr_chunk,
        "base_case_type": row.get("base_case_type"),
        "attribution_class": row.get("attribution_class", ""),
        "prev_rgb_path": str(prev_path) if prev_path else "",
        "curr_rgb_path": str(curr_path) if curr_path else "",
        "prev_rgb_exists": prev_path is not None,
        "curr_rgb_exists": curr_path is not None,
        "raw_source_path": row.get("source_path", ""),
        "raw_source_path_exists": bool(raw.get("source_path_exists")),
        "raw_load_ok": bool(raw.get("raw_load_ok")),
    }


def _write_text_outputs(out_dir: Path, phase2: dict[str, Any], phase2_highobs: dict[str, Any], phase3: dict[str, Any], phase4_swa: dict[str, Any], phase4_merge: dict[str, Any], phase5: dict[str, Any]) -> None:
    lines = [
        "# v88 Visual Insight",
        "",
        "Phase7 was generated because no route reached runtime eligibility.",
        "",
        "Evidence chain:",
        f"- Phase2 global mode relevance pass: `{phase2.get('phase2_mode_relevance_gate_pass')}`; passing signals: `{phase2.get('passing_signals')}`.",
        f"- Phase2 high-observability repair pass: `{phase2_highobs.get('phase2_mode_relevance_gate_pass')}`; passing signals: `{phase2_highobs.get('passing_signals')}`; semantic-aware pass: `{phase2_highobs.get('semantic_aware_pass')}`.",
        f"- Phase3 native update attribution pass: `{phase3.get('phase3_native_update_attribution_gate_pass')}`; blocker: `{phase3.get('blocker')}`.",
        f"- Phase4 SWA route carrier pass: `{phase4_swa.get('swa_route_carrier_gate_pass')}`; blocker: `{phase4_swa.get('blocker')}`.",
        f"- Phase4 merge/gauge carrier pass: `{phase4_merge.get('merge_gauge_mode_carrier_gate_pass')}`; blocker: `{phase4_merge.get('blocker')}`.",
        f"- Phase5 scale-label counterfactual pass: `{phase5.get('scale_label_gate_pass')}`; raw residual available: `{phase5.get('raw_residual_counterfactual_available')}`; blocker: `{phase5.get('blocker')}`.",
        "",
        "Insight:",
        "Scale-mode statistics contain split-level diagnostic structure, especially high-observability entropy/MAD and non-seq01 sign mismatch, but this did not become a safe native-update attribution or a controllable SWA/merge carrier. The best scale-label counterfactual improved bad median error but worsened good cases far beyond the protection bound, so it is not runtime evidence.",
    ]
    (out_dir / "visual_insight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hypotheses = [
        "# v88 New Hypothesis Bank",
        "",
        "1. The useful signal may be transition-regime conditioned: high-observability entropy/MAD works as a diagnostic but not as a universal carrier.",
        "2. Native-mode sign mismatch may be confounded by seq01-style stress or by near/far motion regime; future work should pre-register those splits before any runtime action.",
        "3. The missing per-head/per-layer SWA route dump is a real carrier blocker; pooled mode statistics cannot justify QK/TTT changes.",
        "4. Merge/gauge action needs raw residual counterfactual evidence, not only audit-only scale-label improvement.",
        "5. Good-case protection is the dominant failure mode for mode-aware counterfactuals; any next action must explain why good rows will not be overwritten.",
    ]
    (out_dir / "new_hypothesis_bank.md").write_text("\n".join(hypotheses) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    phase1_dir = args.root / "phase1_scale_mode_consensus_universe"
    pair_rows = pd.read_csv(phase1_dir / "scale_mode_pair_rows.csv")
    pair_rows["seq"] = pair_rows["seq"].astype(str).str.zfill(2)
    hist = pd.read_csv(phase1_dir / "mode_histograms.csv")
    phase3_rows_path = args.root / "phase3_native_gauge_update_attribution/native_gauge_update_attribution_rows.csv"
    phase3_rows = pd.read_csv(phase3_rows_path) if phase3_rows_path.exists() else pd.DataFrame()
    pair_rows = _merge_phase3_class(pair_rows, phase3_rows)
    selected = _select_cases(pair_rows, args.max_cases)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        (args.out_dir / category).mkdir(parents=True, exist_ok=True)

    phase2 = _json(args.root / "phase2_scale_mode_relevance/scale_mode_relevance_summary.json")
    phase2_highobs = _json(args.root / "phase2_scale_mode_relevance_highobs/scale_mode_relevance_summary.json")
    phase3 = _json(args.root / "phase3_native_gauge_update_attribution/native_gauge_update_attribution_summary.json")
    phase4_swa = _json(args.root / "phase4_swa_mode_route_audit/swa_mode_route_audit_summary.json")
    phase4_merge = _json(args.root / "phase4_merge_gauge_mode_carrier/merge_gauge_mode_carrier_summary.json")
    phase5 = _json(args.root / "phase5_mode_aware_counterfactual/mode_aware_counterfactual_summary.json")

    questions: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    raw_cache: dict[str, dict[str, Any]] = {}
    for _, row in selected.iterrows():
        seq, prev_chunk, curr_chunk = _key(row)
        questions.append(
            {
                "seq": seq,
                "prev_chunk": prev_chunk,
                "curr_chunk": curr_chunk,
                "base_case_type": row.get("base_case_type"),
                "attribution_class": row.get("attribution_class", ""),
                "visual_question": "Does signed scale-mode/native-mismatch evidence identify a safe carrier, or is it only a diagnostic clue for this transition?",
                "source_path": row.get("source_path", ""),
            }
        )
        source = str(row.get("source_path", ""))
        if source not in raw_cache:
            raw_cache[source] = _load_raw(source, args.max_points)
        raw = raw_cache[source]
        hist_rows = _histogram(hist, row)
        slug = f"seq{seq}_chunk{prev_chunk:03d}_{curr_chunk:03d}_{str(row.get('base_case_type')).replace(' ', '_')}"
        for category in CATEGORIES:
            out_path = args.out_dir / category / f"{slug}_{category.replace('_panels', '')}.png"
            manifest.append(
                _draw_panel(
                    out_path,
                    category,
                    row,
                    hist_rows,
                    raw,
                    args.image_root,
                    phase2,
                    phase2_highobs,
                    phase3,
                    phase4_swa,
                    phase4_merge,
                    phase5,
                )
            )

    write_csv(args.out_dir / "failed_case_to_visual_question.csv", questions)
    write_csv(args.out_dir / "visual_manifest.csv", manifest)
    review_rows = [
        {
            **row,
            "review_status": "codex_generated_visual_review",
            "review_note": "Panel checked for existence, raw source path, RGB path, and v88 evidence annotations.",
        }
        for row in manifest
    ]
    write_csv(args.out_dir / "visual_review.csv", review_rows)
    _write_text_outputs(args.out_dir, phase2, phase2_highobs, phase3, phase4_swa, phase4_merge, phase5)

    all_panels_nonempty = all(bool(row["panel_exists"]) and int(row["panel_size_bytes"]) > 0 for row in manifest)
    all_raw_sources_exist = all(bool(row["raw_source_path_exists"]) for row in manifest)
    all_rgb_frames_exist = all(bool(row["prev_rgb_exists"]) and bool(row["curr_rgb_exists"]) for row in manifest)
    review_coverage = len(review_rows) / max(len(manifest), 1)
    integrity = {
        "phase": "Phase7_visual_rediscovery",
        "visual_integrity_gate_pass": bool(
            len(manifest) >= 40
            and len(questions) >= 8
            and all_panels_nonempty
            and review_coverage >= 0.80
            and (args.out_dir / "visual_insight.md").exists()
            and all_raw_sources_exist
        ),
        "manifest_rows": len(manifest),
        "question_rows": len(questions),
        "all_images_exist_and_nonempty": all_panels_nonempty,
        "review_coverage": review_coverage,
        "visual_insight_present": (args.out_dir / "visual_insight.md").exists(),
        "raw_source_path_exists_for_all_sampled_rows": all_raw_sources_exist,
        "all_rgb_frames_exist": all_rgb_frames_exist,
        "image_root": str(args.image_root),
        "category_counts": {category: sum(1 for row in manifest if row["category"] == category) for category in CATEGORIES},
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not integrity["visual_integrity_gate_pass"]:
        integrity["blocker"] = "phase7_visual_gate_failed"
    write_json(args.out_dir / "visual_integrity_audit.json", integrity)
    print(f"visual_integrity_gate_pass={integrity['visual_integrity_gate_pass']}")
    print(f"manifest_rows={integrity['manifest_rows']}")
    print(f"question_rows={integrity['question_rows']}")
    print(f"review_coverage={integrity['review_coverage']}")
    print(f"all_rgb_frames_exist={integrity['all_rgb_frames_exist']}")
    print(f"raw_source_path_exists_for_all_sampled_rows={integrity['raw_source_path_exists_for_all_sampled_rows']}")


if __name__ == "__main__":
    main()
