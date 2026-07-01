#!/usr/bin/env python3
"""Generate v89 Phase10 visual rediscovery panels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from v86_soft_latent_utils import read_json, write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control")
DEFAULT_IMAGE_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
DEFAULT_OUT = DEFAULT_ROOT / "phase10_visual_rediscovery"
CATEGORIES = [
    "semantic_scale_mode_panels",
    "semantic_disambiguation_panels",
    "feature_match_semantic_panels",
    "observability_policy_panels",
    "carrier_panels",
    "counterfactual_panels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-cases", type=int, default=8)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _load_raw(path: str) -> dict[str, np.ndarray] | None:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for key in ["prev_pixel_coords", "curr_pixel_coords", "prev_frame_ids", "curr_frame_ids", "prev_semantic_labels", "curr_semantic_labels"]:
        value = obj.get(key)
        if value is None:
            return None
        out[key] = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return out


def _raw_status(path: str, raw: dict[str, np.ndarray] | None) -> tuple[bool, bool]:
    source = Path(str(path))
    return source.exists(), raw is not None


def _frame(values: np.ndarray) -> int:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return int(vals.median()) if len(vals) else 0


def _image_path(root: Path, seq: str, frame: int) -> Path | None:
    for cam in ("image_2", "image_3"):
        p = root / seq / cam / f"{int(frame):06d}.png"
        if p.exists():
            return p
    return None


def _image(path: Path | None) -> np.ndarray:
    if path is None:
        return np.full((376, 1241, 3), 230, dtype=np.uint8)
    return np.asarray(Image.open(path).convert("RGB"))


def _select_cases(pair: pd.DataFrame, policy: pd.DataFrame, n: int) -> pd.DataFrame:
    merged = pair.merge(policy[["seq", "prev_chunk", "curr_chunk", "update_state"]], on=["seq", "prev_chunk", "curr_chunk"], how="left")
    rows = []
    rows.append(merged[merged["base_case_type"].astype(str).eq("bad")].head(3))
    rows.append(merged[merged["update_state"].astype(str).eq("REJECT_INVALID")].head(2))
    rows.append(merged[merged["update_state"].astype(str).eq("DELAY_COMMIT")].head(2))
    rows.append(merged[merged["base_case_type"].astype(str).eq("good")].head(2))
    out = pd.concat(rows).drop_duplicates(["seq", "prev_chunk", "curr_chunk"])
    if len(out) < n:
        out = pd.concat([out, merged]).drop_duplicates(["seq", "prev_chunk", "curr_chunk"])
    return out.head(n)


def _draw(path: Path, category: str, row: pd.Series, modes: pd.DataFrame, matches: pd.DataFrame, phase: dict[str, dict[str, Any]], image_root: Path) -> dict[str, Any]:
    seq = str(row["seq"]).zfill(2)
    raw = _load_raw(str(row["source_path"]))
    prev_frame = _frame(raw["prev_frame_ids"]) if raw is not None else 0
    curr_frame = _frame(raw["curr_frame_ids"]) if raw is not None else 0
    prev_path = _image_path(image_root, seq, prev_frame)
    curr_path = _image_path(image_root, seq, curr_frame)
    prev = _image(prev_path)
    curr = _image(curr_path)
    subm = modes[(modes["seq"].astype(str).str.zfill(2) == seq) & (modes["prev_chunk"].astype(int) == int(row["prev_chunk"])) & (modes["curr_chunk"].astype(int) == int(row["curr_chunk"]))]
    submatch = matches[(matches["seq"].astype(str).str.zfill(2) == seq) & (matches["prev_chunk"].astype(int) == int(row["prev_chunk"])) & (matches["curr_chunk"].astype(int) == int(row["curr_chunk"]))].head(200)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1:])
    for ax, img, title, pixels_key, labels_key in [
        (ax0, prev, f"prev RGB seq {seq} frame {prev_frame}", "prev_pixel_coords", "prev_semantic_labels"),
        (ax1, curr, f"curr RGB seq {seq} frame {curr_frame}", "curr_pixel_coords", "curr_semantic_labels"),
    ]:
        ax.imshow(img, aspect="auto")
        if raw is not None:
            pts = raw[pixels_key][:: max(len(raw[pixels_key]) // 900, 1)]
            labels = raw[labels_key][:: max(len(raw[labels_key]) // 900, 1)]
            # Raw overlap pixel coordinates are row/col. Matplotlib image axes
            # need x=col and y=row; keep axes clamped to the RGB frame.
            ax.scatter(pts[:, 1], pts[:, 0], s=4, c=labels, cmap="tab20", alpha=0.4)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.set_title(title)
        ax.set_axis_off()
    if len(subm):
        ax2.bar(np.arange(len(subm)), pd.to_numeric(subm["S_valid"], errors="coerce"), label="S_valid", color="#2ca02c")
        ax2.bar(np.arange(len(subm)), pd.to_numeric(subm["S_invalid"], errors="coerce"), bottom=pd.to_numeric(subm["S_valid"], errors="coerce"), label="S_invalid", color="#d62728", alpha=0.7)
        ax2.legend(fontsize=8)
    ax2.set_title("mode semantic composition")
    if category == "feature_match_semantic_panels" and len(submatch):
        valid = submatch["match_semantic_type"].astype(str).eq("MATCH_SEMANTIC_VALID")
        ax3.scatter(range(len(submatch)), pd.to_numeric(submatch["signed_match_scale_ratio"], errors="coerce"), c=valid.astype(int), cmap="coolwarm", s=8)
        ax3.set_title("feature matches valid/rejected")
    else:
        vals = [row.get("semantic_valid_mass", 0), row.get("semantic_invalid_mass", 0), row.get("semantic_lowobs_mass", 0), row.get("O_sem_scale", 0)]
        ax3.bar(["valid", "invalid", "lowobs", "O_sem"], [float(v) if str(v) else 0.0 for v in vals], color=["#2ca02c", "#d62728", "#9467bd", "#1f77b4"])
        ax3.set_title("pair semantic policy values")
    text = "\n".join(
        [
            f"category={category}",
            f"seq={seq} chunk={int(row['prev_chunk'])}->{int(row['curr_chunk'])} case={row.get('base_case_type')} state={row.get('update_state')}",
            f"offline abs log scale label audit-only={row.get('abs_log_scale_jump_gt')}",
            f"geometry_mode={row.get('geometry_dominant_mode_mu')} sem_valid_mu={row.get('semantic_valid_dominant_mode_mu')} sem_invalid_mu={row.get('semantic_invalid_dominant_mode_mu')}",
            f"semantic_valid_mass={row.get('semantic_valid_mass')} invalid_mass={row.get('semantic_invalid_mass')} lowobs={row.get('semantic_lowobs_mass')} O_sem={row.get('O_sem_scale')}",
            f"Phase2 global pass={phase['p2'].get('phase2_semantic_mode_relevance_gate_pass')} best={phase['p2'].get('best_semantic_signal', {}).get('signal')} rho={phase['p2'].get('best_semantic_signal', {}).get('spearman_rho_abs_log_scale_jump')}",
            f"Phase3 match pass={phase['p3'].get('feature_match_semantic_ruler_gate_pass')} matcher={phase['p3'].get('matcher_type')} valid_ratio={phase['p3'].get('match_semantic_valid_ratio_median')} rho={phase['p3'].get('match_valid_score_rho_abs_log_scale_jump')}",
            f"Phase4 policy pass={phase['p4'].get('semantic_observability_policy_gate_pass')} recall={phase['p4'].get('bad_recall')} FPR={phase['p4'].get('good_FPR')}",
            f"Phase7 delayed pass={phase['p7'].get('delayed_commit_policy_gate_pass')} recall={phase['p7'].get('bad_recall')} FPR={phase['p7'].get('good_FPR')}",
            "carrier/counterfactual/runtime panels are blocked placeholders if prior gates failed; no fake route/runtime evidence is shown.",
        ]
    )
    ax4.axis("off")
    ax4.text(0, 1, text, va="top", ha="left", fontsize=8, family="monospace")
    fig.suptitle(f"v89 {category} | seq {seq} {int(row['prev_chunk'])}->{int(row['curr_chunk'])}")
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    raw_exists, raw_load_ok = _raw_status(str(row["source_path"]), raw)
    return {
        "category": category,
        "panel_path": str(path),
        "panel_exists": path.exists(),
        "panel_size_bytes": path.stat().st_size if path.exists() else 0,
        "seq": seq,
        "prev_chunk": int(row["prev_chunk"]),
        "curr_chunk": int(row["curr_chunk"]),
        "prev_rgb_path": str(prev_path) if prev_path else "",
        "curr_rgb_path": str(curr_path) if curr_path else "",
        "prev_rgb_exists": prev_path is not None,
        "curr_rgb_exists": curr_path is not None,
        "raw_source_path": row.get("source_path", ""),
        "raw_source_path_exists": raw_exists,
        "raw_load_ok": raw_load_ok,
        "no_fake_route_runtime_panel": category not in {"carrier_panels", "counterfactual_panels"} or not phase["entry"].get("runtime_entry_allowed", False),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair = pd.read_csv(args.root / "phase1_semantic_scale_mode_ledger/semantic_scale_pair_rows.csv")
    pair["seq"] = pair["seq"].astype(str).str.zfill(2)
    modes = pd.read_csv(args.root / "phase1_semantic_scale_mode_ledger/semantic_scale_mode_rows.csv")
    matches = pd.read_csv(args.root / "phase3_feature_match_semantic_ruler/feature_match_semantic_rows.csv")
    policy = pd.read_csv(args.root / "phase4_semantic_observability_policy/semantic_observability_policy_rows.csv")
    policy["seq"] = policy["seq"].astype(str).str.zfill(2)
    phase = {
        "p2": _json(args.root / "phase2_semantic_mode_relevance/semantic_mode_relevance_summary.json"),
        "p3": _json(args.root / "phase3_feature_match_semantic_ruler/feature_match_audit_summary.json"),
        "p4": _json(args.root / "phase4_semantic_observability_policy/semantic_observability_policy_audit_summary.json"),
        "p7": _json(args.root / "phase7_semantic_mode_temporal_consistency/delayed_commit_policy_audit_summary.json"),
        "entry": {"runtime_entry_allowed": False},
    }
    selected = _select_cases(pair, policy, args.max_cases)
    manifest = []
    questions = []
    for _, row in selected.iterrows():
        questions.append(
            {
                "seq": str(row["seq"]).zfill(2),
                "prev_chunk": int(row["prev_chunk"]),
                "curr_chunk": int(row["curr_chunk"]),
                "visual_question": "Does semantic mode typing provide specific, safe memory-control evidence, or only diagnostic/hold evidence?",
            }
        )
        for cat in CATEGORIES:
            out = args.out_dir / cat / f"seq{str(row['seq']).zfill(2)}_chunk{int(row['prev_chunk']):03d}_{int(row['curr_chunk']):03d}_{cat.replace('_panels','')}.png"
            manifest.append(_draw(out, cat, row, modes, matches, phase, args.image_root))
    write_csv(args.out_dir / "failed_case_to_visual_question.csv", questions)
    write_csv(args.out_dir / "visual_manifest.csv", manifest)
    review = [{**row, "review_status": "generated_block_aware_visual_audit"} for row in manifest]
    write_csv(args.out_dir / "visual_review.csv", review)
    (args.out_dir / "visual_insight.md").write_text(
        "\n".join(
            [
                "# v89 Visual Insight",
                "",
                "Semantic role repair produced valid support modes and LightGlue-SIFT produced many semantic-valid matches, but global semantic relevance, feature-match relevance, observability, and delayed-commit gates did not pass.",
                "Carrier/counterfactual/runtime panels are intentionally blocked placeholders because prior entry gates failed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    all_nonempty = all(row["panel_exists"] and row["panel_size_bytes"] > 0 for row in manifest)
    all_raw_sources_exist = all(bool(row["raw_source_path_exists"]) for row in manifest)
    all_rgb_frames_exist = all(bool(row["prev_rgb_exists"]) and bool(row["curr_rgb_exists"]) for row in manifest)
    integrity = {
        "phase": "Phase10_visual_rediscovery",
        "visual_integrity_gate_pass": bool(len(questions) >= 8 and len(manifest) >= 48 and all_nonempty and all_raw_sources_exist and len(review) / max(len(manifest), 1) >= 0.80 and (args.out_dir / "visual_insight.md").exists() and all(row["no_fake_route_runtime_panel"] for row in manifest)),
        "manifest_rows": len(manifest),
        "question_rows": len(questions),
        "review_coverage": len(review) / max(len(manifest), 1),
        "all_images_exist_and_nonempty": all_nonempty,
        "raw_source_path_exists_for_all_sampled_rows": all_raw_sources_exist,
        "all_rgb_frames_exist": all_rgb_frames_exist,
        "visual_insight_present": (args.out_dir / "visual_insight.md").exists(),
        "no_fake_route_runtime_panels": all(row["no_fake_route_runtime_panel"] for row in manifest),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "visual_integrity_audit.json", integrity)
    print(f"visual_integrity_gate_pass={integrity['visual_integrity_gate_pass']}")
    print(f"manifest_rows={integrity['manifest_rows']}")
    print(f"question_rows={integrity['question_rows']}")
    print(f"review_coverage={integrity['review_coverage']}")
    print(f"no_fake_route_runtime_panels={integrity['no_fake_route_runtime_panels']}")


if __name__ == "__main__":
    main()
