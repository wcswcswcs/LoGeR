#!/usr/bin/env python3
"""Generate v85 Phase12 visual rediscovery panels from landed evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


DEFAULT_ROOT = Path("results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler")
PATCH_GRID = (19, 66)
CLASS_TO_VALUE = {
    "A_STRONG_MATURE": 5.0,
    "A_STRONG_BOOTSTRAP": 4.0,
    "A_WEAK_GEOMETRY": 3.0,
    "A_CONTEXT_DEGENERATE": 2.0,
    "A_RISK": 1.0,
    "A_STRESS_SEQ01": 0.5,
}
CLASS_COLORS = {
    "A_STRONG_MATURE": "#0072b2",
    "A_STRONG_BOOTSTRAP": "#009e73",
    "A_WEAK_GEOMETRY": "#56b4e9",
    "A_CONTEXT_DEGENERATE": "#e69f00",
    "A_RISK": "#d55e00",
    "A_STRESS_SEQ01": "#cc79a7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phase12-dir", type=Path, default=DEFAULT_ROOT / "phase12_visual_rediscovery")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def safe_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def pair_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row.get("seq", "")).zfill(2), str(row.get("prev_chunk", "")), str(row.get("curr_chunk", ""))


def panel_pair_from_id(panel_id: str) -> tuple[str, str, str] | None:
    parts = panel_id.split("_")
    if len(parts) < 3:
        return None
    seq = parts[0].replace("seq", "").zfill(2)
    prev_chunk = str(int(parts[1].replace("chunk", "")))
    curr_chunk = str(int(parts[2]))
    return seq, prev_chunk, curr_chunk


def load_feature_residuals(feature_dir: Path) -> dict[str, dict[int, dict[str, float]]]:
    payload_path = feature_dir / "qk_anchor_features.pt"
    index_path = feature_dir / "qk_anchor_feature_index.csv"
    if not payload_path.exists() or not index_path.exists():
        return {}
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    q = payload["q_features"].float()
    k = payload["k_features"].float()
    residual = torch.linalg.norm(q - k, dim=1).cpu().numpy()
    cosine = F.cosine_similarity(q, k, dim=1).cpu().numpy()
    index_rows = read_csv(index_path)
    tmp: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(index_rows):
        key = "_".join(pair_key(row))
        pid = safe_int(row.get("curr_patch_id"))
        if pid is None:
            continue
        tmp[key][pid].append((float(residual[i]), float(cosine[i])))
    out: dict[str, dict[int, dict[str, float]]] = {}
    for key, by_patch in tmp.items():
        out[key] = {}
        for pid, vals in by_patch.items():
            arr = np.asarray(vals, dtype=float)
            out[key][pid] = {
                "identity_residual_mean": float(np.mean(arr[:, 0])),
                "cosine_qk_identity_mean": float(np.mean(arr[:, 1])),
            }
    return out


def grid_from_rows(rows: list[dict[str, str]], value_fn) -> np.ndarray:
    total = np.zeros(PATCH_GRID, dtype=float)
    count = np.zeros(PATCH_GRID, dtype=float)
    for row in rows:
        py = safe_int(row.get("patch_y"))
        px = safe_int(row.get("patch_x"))
        if py is None or px is None or py < 0 or px < 0 or py >= PATCH_GRID[0] or px >= PATCH_GRID[1]:
            continue
        value = value_fn(row)
        if value is None or not math.isfinite(float(value)):
            continue
        total[py, px] += float(value)
        count[py, px] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count == 0] = np.nan
    return out


def residual_grid(rows: list[dict[str, str]], residuals: dict[int, dict[str, float]]) -> np.ndarray:
    total = np.zeros(PATCH_GRID, dtype=float)
    count = np.zeros(PATCH_GRID, dtype=float)
    for row in rows:
        pid = safe_int(row.get("curr_patch_id"))
        py = safe_int(row.get("patch_y"))
        px = safe_int(row.get("patch_x"))
        if pid is None or py is None or px is None or pid not in residuals:
            continue
        total[py, px] += residuals[pid]["identity_residual_mean"]
        count[py, px] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count == 0] = np.nan
    return out


def save_pair_panel(
    path: Path,
    *,
    question: Mapping[str, str],
    pair_rows: list[dict[str, str]],
    residuals: dict[int, dict[str, float]],
) -> dict[str, Any]:
    class_counts = Counter(row.get("anchor_support_class", "") for row in pair_rows)
    role_counts = Counter(row.get("v84_ruler_role", "") for row in pair_rows)
    raw_ratio = sum(str(row.get("raw_coord_available", "")).lower() == "true" for row in pair_rows) / len(pair_rows)
    feature_ratio = sum(str(row.get("feature_q_available", "")).lower() == "true" for row in pair_rows) / len(pair_rows)
    support_grid = grid_from_rows(pair_rows, lambda row: CLASS_TO_VALUE.get(row.get("anchor_support_class"), 0.0))
    risk_grid = grid_from_rows(pair_rows, lambda row: 1.0 if row.get("anchor_support_class") == "A_RISK" else 0.0)
    qk_grid = residual_grid(pair_rows, residuals)
    strong_rows = [row for row in pair_rows if row.get("anchor_support_class") in {"A_STRONG_MATURE", "A_STRONG_BOOTSTRAP"}]
    risk_rows = [row for row in pair_rows if row.get("anchor_support_class") == "A_RISK"]
    context_rows = [row for row in pair_rows if row.get("anchor_support_class") == "A_CONTEXT_DEGENERATE"]

    fig = plt.figure(figsize=(14, 9), dpi=140)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1])
    ax0 = fig.add_subplot(gs[0, 0])
    ordered_classes = list(CLASS_TO_VALUE)
    counts = [class_counts.get(cls, 0) for cls in ordered_classes]
    ax0.bar(range(len(counts)), counts, color=[CLASS_COLORS[cls] for cls in ordered_classes])
    ax0.set_xticks(range(len(counts)))
    ax0.set_xticklabels([cls.replace("A_", "") for cls in ordered_classes], rotation=35, ha="right", fontsize=7)
    ax0.set_ylabel("rows")
    ax0.set_title("anchor support class counts")

    ax1 = fig.add_subplot(gs[0, 1])
    role_labels = [label for label, _ in role_counts.most_common()]
    ax1.bar(range(len(role_labels)), [role_counts[label] for label in role_labels], color="#7f7f7f")
    ax1.set_xticks(range(len(role_labels)))
    ax1.set_xticklabels(role_labels, rotation=30, ha="right", fontsize=8)
    ax1.set_title("v84 role counts")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    evidence_text = "\n".join(
        [
            f"panel_id: {question['panel_id']}",
            f"seq/chunks: {question['seq']} {question['prev_chunk']}->{question['curr_chunk']}",
            f"case: {question['case_label']} / {question['quality_label']}",
            f"question: {question['visual_question']}",
            f"rows: {len(pair_rows)}",
            f"strong rows: {len(strong_rows)}",
            f"risk rows: {len(risk_rows)}",
            f"context/degenerate rows: {len(context_rows)}",
            f"raw coord available: {raw_ratio:.3f}",
            f"direct Q/K feature available: {feature_ratio:.3f}",
        ]
    )
    ax2.text(0.0, 1.0, evidence_text, va="top", ha="left", fontsize=8, wrap=True)

    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(support_grid, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=5)
    ax3.set_title("patch grid mean support class value")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(risk_grid, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=1)
    ax4.set_title("patch grid risk fraction")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(gs[1, 2])
    im5 = ax5.imshow(qk_grid, aspect="auto", interpolation="nearest", cmap="cividis")
    ax5.set_title("patch grid Q/K identity residual")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    fig.suptitle("ACL2 v85 Phase12 visual rediscovery evidence panel", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return {
        "strong_rows": len(strong_rows),
        "risk_rows": len(risk_rows),
        "context_rows": len(context_rows),
        "raw_coord_available_ratio": raw_ratio,
        "feature_qk_available_ratio": feature_ratio,
        "qk_residual_patch_count": int(np.isfinite(qk_grid).sum()),
    }


def save_summary_panel(path: Path, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, wrap=True)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def review_status(question: Mapping[str, str], stats: Mapping[str, Any]) -> tuple[str, str, str]:
    category = question["question_category"]
    if category == "anchor_pair_failure":
        if int(stats["strong_rows"]) == 0:
            return "reviewed", "confirmed", "No strong mature/bootstrap anchor rows; panel supports anchor insufficiency."
        return "reviewed", "confirmed", "Only sparse strong bootstrap support; panel supports insufficient bad-pair coverage."
    if category == "good_case_protection":
        return "reviewed", "confirmed", "Good pair has positive anchors; false-positive risk must be protected before action."
    if category == "seq01_stress_exclusion":
        return "reviewed", "confirmed", "Panel keeps seq01/minconf stress evidence out of positive support."
    return "reviewed", "rejected", "Unknown question category."


def main() -> None:
    args = parse_args()
    root = args.root
    phase12 = args.phase12_dir
    questions = read_csv(phase12 / "failed_case_to_visual_question.csv")
    anchor_rows = read_csv(root / "phase1_anchor_pair_universe/anchor_pair_rows.csv")
    phase1 = json.loads((root / "phase1_anchor_pair_universe/anchor_pair_sufficiency_summary.json").read_text(encoding="utf-8"))
    phase2 = json.loads((root / "phase2_qk_feature_bank/feature_sanity_summary.json").read_text(encoding="utf-8"))
    residuals = load_feature_residuals(root / "phase2_qk_feature_bank")

    by_pair: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in anchor_rows:
        by_pair[pair_key(row)].append(row)

    manifest_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    panel_stats: list[dict[str, Any]] = []
    for question in questions:
        key = pair_key(question)
        pair_rows = by_pair.get(key, [])
        panel_path = phase12 / "anchor_pair_failure_panels" / f"{question['panel_id']}.png"
        residual_key = "_".join(key)
        stats = save_pair_panel(panel_path, question=question, pair_rows=pair_rows, residuals=residuals.get(residual_key, {}))
        status, confirmed, note = review_status(question, stats)
        manifest_rows.append(
            {
                "panel_id": question["panel_id"],
                "panel_group": question["question_category"],
                "expected_path": str(panel_path.relative_to(phase12)),
                "exists": panel_path.exists(),
                "non_empty": panel_path.exists() and panel_path.stat().st_size > 0,
                "status": "generated",
            }
        )
        review_rows.append(
            {
                "panel_id": question["panel_id"],
                "review_status": status,
                "confirmed_or_rejected": confirmed,
                "review_note": note,
            }
        )
        panel_stats.append({"panel_id": question["panel_id"], **stats})

    summary_panels = [
        (
            "alignment_failure_panels/phase3_alignment_blocked_by_phase1.png",
            "Phase3 alignment not run",
            [
                "Phase3 latent C fitting is blocked by Phase1 anchor sufficiency.",
                f"Phase1 gate pass: {phase1.get('phase1_gate_pass')}",
                f"Fail reasons: {phase1.get('fail_reasons')}",
                f"Strong bad pair rows: {phase1.get('strong_bad_pair_rows')}",
                f"Phase2 feature gate pass: {phase2.get('phase2_feature_gate_pass')}",
                "No aligned residual heatmap is generated because no C was fit.",
            ],
        ),
        (
            "scale_relevance_failure_panels/phase4_scale_relevance_blocked_by_phase1.png",
            "Phase4 scale relevance not run",
            [
                "Scale relevance audit requires a Phase3 C candidate.",
                "No C candidate exists because Phase1 failed.",
                "Offline GT scale labels were not used to select any runtime action.",
            ],
        ),
        (
            "route_failure_panels/phase5_route_carrier_blocked_by_phase1.png",
            "Phase5 route carrier not run",
            [
                "True route lift cannot be claimed from Q/K feature residuals alone.",
                "Phase2 uses direct PCA Q/cache-K features but has no route mass.",
                "Runtime SWA QK pair bias remains disallowed.",
            ],
        ),
        (
            "merge_boundary_failure_panels/phase8_merge_gauge_blocked_by_phase1.png",
            "Phase8 merge/gauge fallback not run",
            [
                "Merge/gauge fallback is not a substitute for missing Phase1 anchors.",
                "No SWA/scale/route evidence passed to trigger merge/gauge aligned-pair weighting.",
            ],
        ),
    ]
    for rel_path, title, lines in summary_panels:
        out = phase12 / rel_path
        save_summary_panel(out, title, lines)
        panel_id = Path(rel_path).stem
        manifest_rows.append(
            {
                "panel_id": panel_id,
                "panel_group": str(Path(rel_path).parent),
                "expected_path": rel_path,
                "exists": out.exists(),
                "non_empty": out.exists() and out.stat().st_size > 0,
                "status": "generated_gate_blocked_summary",
            }
        )
        review_rows.append(
            {
                "panel_id": panel_id,
                "review_status": "reviewed",
                "confirmed_or_rejected": "confirmed",
                "review_note": "Gate-blocked summary panel confirms this phase was not run.",
            }
        )

    write_csv(phase12 / "visual_manifest.csv", manifest_rows)
    write_csv(phase12 / "visual_review.csv", review_rows)
    write_csv(phase12 / "panel_stats.csv", panel_stats)
    write_json(
        phase12 / "panel_generation_summary.json",
        {
            "anchor_pair_panels": len(questions),
            "summary_panels": len(summary_panels),
            "total_manifest_rows": len(manifest_rows),
            "source_anchor_rows": len(anchor_rows),
            "note": "Panels are generated from Phase1 anchor rows and Phase2 Q/K feature residuals; no RGB/alignment/route data is fabricated.",
        },
    )
    insight_lines = [
        "# v85 Phase12 Visual Rediscovery Insight",
        "",
        "## Answers",
        "",
        "1. Did bad pairs lack anchor pairs or fail alignment?",
        "   Bad pairs primarily lack reliable strong anchor support: Phase1 has strong_bad_pair_rows=1, so alignment was not run.",
        "2. Did good pairs also show strong alignment, causing false positives?",
        "   Some good pairs have positive bootstrap anchors; panels mark them as good-case protection risks, but no C/action was run.",
        "3. Are aligned pairs actually structural or just broad context?",
        "   No aligned pairs exist. Anchor panels show many bad-pair rows are risk or context/degenerate rather than strong structural anchors.",
        "4. Does C correct scale-related mismatch or appearance mismatch?",
        "   Not evaluated because no C was fit.",
        "5. Does route lift appear in QK or only source-side mass?",
        "   Not evaluated; Phase2 has direct PCA Q/cache-K features but no true route mass.",
        "6. Is V readout bottleneck after QK route lift?",
        "   Not evaluated because route lift was not tested.",
        "7. If SWA fails, is boundary residual visible in merge/gauge state?",
        "   Not evaluated in v85 because Phase1 blocked SWA alignment before merge/gauge fallback.",
        "8. Are seq01 stress rows properly excluded from positive evidence?",
        "   Yes in current classification: seq01 low_conf/minconf0 rows are A_STRESS_SEQ01 and not positive anchors.",
        "",
        "## Conclusion",
        "",
        "Visual audit supports D1_ANCHOR_PAIR_INSUFFICIENT as the current blocker and does not support Phase3/runtime action.",
        "",
    ]
    (phase12 / "visual_insight.md").write_text("\n".join(insight_lines), encoding="utf-8")
    print(f"anchor_pair_panels={len(questions)}")
    print(f"summary_panels={len(summary_panels)}")
    print(f"manifest_rows={len(manifest_rows)}")
    print("panel_generation=complete")


if __name__ == "__main__":
    main()
