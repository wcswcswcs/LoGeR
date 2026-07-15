#!/usr/bin/env python3
"""Build ACL2 v118 Stage4-R51 fresh R47-policy validation configs.

R51 applies the post-R47 abstention rule to fresh KITTI 03/04 evidence built in
R49/R50.  It intentionally keeps the intervention form narrow: uniform source
frame value scaling, with token polarity selected by the R47 rule from fresh
internal anchor-read and semantic token/support signals.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from statistics import median, pstdev
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"


def parse_seq_env(name: str, default: str) -> tuple[str, ...]:
    seqs = tuple(part.strip().zfill(2) for part in os.environ.get(name, default).replace(";", ",").split(",") if part.strip())
    return seqs or tuple(part.strip().zfill(2) for part in default.split(",") if part.strip())


STAGE_TAG = os.environ.get("ACL2_V118_FRESH_POLICY_TAG", "r51").strip().lower() or "r51"
STAGE_LABEL = STAGE_TAG.upper()
STAGE = RESULT_ROOT / os.environ.get("ACL2_V118_FRESH_POLICY_STAGE_SLUG", "stage4_r51_lingbot_ar_fresh_r47_policy_validation")
CONFIG_DIR = STAGE / "configs"
RUNTIME = STAGE / "runtime_full_thread8"
ACTION_DIR = RUNTIME / "action_traces"
SUMMARY_DIR = STAGE / "summary"
R49 = RESULT_ROOT / os.environ.get("ACL2_V118_FRESH_POLICY_TRACE_STAGE_SLUG", "stage4_r49_lingbot_ar_fresh_trace_baseline")
R50 = RESULT_ROOT / os.environ.get("ACL2_V118_FRESH_POLICY_TOKEN_STAGE_SLUG", "stage4_r50_lingbot_ar_fresh_support_token_tensors")
WORKSPACE = R49 / "workspace"
SUPPORT = R50 / "summary/stage4_r50_fresh_frame_semantic_support_rows.csv"
TOKEN_ROOT = R50 / "token_semantics"
TRACE_DIR = R49 / "runtime_full"
BENCH = ROOT / "third_party/lingbot-map/benchmark"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
PYTHONPATH = f"{ROOT / 'third_party/lingbot-map'}:{ROOT / 'third_party/lingbot-map/benchmark'}"
SEQS = parse_seq_env("ACL2_V118_FRESH_POLICY_SEQS", "04,03")
SEQ_LABEL = "/".join(SEQS)
DATASET_PREFIX = os.environ.get("ACL2_V118_FRESH_DATASET_PREFIX", "kitti_v118_r49_fresh_seq")
ANCHOR_FRAMES = tuple(range(8))
METHOD_PREFIX = os.environ.get("ACL2_V118_FRESH_POLICY_METHOD_PREFIX", f"lingbot_map_stream_sdpa_v118_{STAGE_TAG}_fresh_r47_policy")
BASELINE_METHOD = os.environ.get("ACL2_V118_FRESH_POLICY_BASELINE_METHOD", "lingbot_map_stream_flashinfer_v118_r49_fresh_trace")
POLICY_RULE = os.environ.get("ACL2_V118_FRESH_POLICY_RULE", "r47").strip().lower() or "r47"
RISK_MIN_CORR = float(os.environ.get("ACL2_V118_FRESH_POLICY_RISK_MIN_CORR", "0.50"))
RISK_MIN_STABLE_TO_WEAK_LOWTRUST = float(os.environ.get("ACL2_V118_FRESH_POLICY_RISK_MIN_STABLE_TO_WEAK_LOWTRUST", "0.20"))
ACTION_DYNAMIC_MIN = float(os.environ.get("ACL2_V118_FRESH_POLICY_ACTION_DYNAMIC_MIN", "0.24"))
NEGATIVE_CORR_RISK_MIN_RATIO = float(os.environ.get("ACL2_V118_FRESH_POLICY_NEGATIVE_CORR_RISK_MIN_RATIO", "0.08"))
NEGATIVE_CORR_RISK_MIN_DYNAMIC = float(os.environ.get("ACL2_V118_FRESH_POLICY_NEGATIVE_CORR_RISK_MIN_DYNAMIC", "0.18"))
STABLE_BOOST_MIN_RATIO = float(os.environ.get("ACL2_V118_FRESH_POLICY_STABLE_BOOST_MIN_RATIO", "0.20"))
STABLE_BOOST_MIN_STABLE_MEAN = float(os.environ.get("ACL2_V118_FRESH_POLICY_STABLE_BOOST_MIN_STABLE_MEAN", "0.15"))
CONTROL_SAFE_NEGATIVE_RATIO_LOW = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_NEGATIVE_RATIO_LOW", "0.05"))
CONTROL_SAFE_NEGATIVE_RATIO_HIGH = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_NEGATIVE_RATIO_HIGH", "0.20"))
CONTROL_SAFE_RISK_MIN_CORR = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_RISK_MIN_CORR", "0.75"))
CONTROL_SAFE_RISK_MIN_RATIO = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_RISK_MIN_RATIO", "0.20"))
CONTROL_SAFE_RISK_MIN_DYNAMIC = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_RISK_MIN_DYNAMIC", "0.20"))
CONTROL_SAFE_V2_STRONG_NEG_CORR = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_STRONG_NEG_CORR", "-0.40"))
CONTROL_SAFE_V2_MID_RATIO_LOW = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_MID_RATIO_LOW", "0.08"))
CONTROL_SAFE_V2_MID_RATIO_HIGH = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_MID_RATIO_HIGH", "0.20"))
CONTROL_SAFE_V2_POSITIVE_LOW_RATIO_MAX = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_POSITIVE_LOW_RATIO_MAX", "0.08"))
CONTROL_SAFE_V2_POSITIVE_LOW_DYNAMIC_MAX = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_POSITIVE_LOW_DYNAMIC_MAX", "0.20"))
CONTROL_SAFE_V2_STABLE_BOOST_CORR_MIN = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_STABLE_BOOST_CORR_MIN", "0.65"))
CONTROL_SAFE_V2_STABLE_BOOST_RATIO_MAX = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_STABLE_BOOST_RATIO_MAX", "0.16"))
CONTROL_SAFE_V2_STABLE_BOOST_DYNAMIC_MIN = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_STABLE_BOOST_DYNAMIC_MIN", "0.24"))
CONTROL_SAFE_V2_MODERATE_REVERSE_CORR_MAX = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_MODERATE_REVERSE_CORR_MAX", "0.50"))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def dataset_name(seq: str) -> str:
    return f"{DATASET_PREFIX}{seq}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_support() -> dict[str, dict[int, dict[str, str]]]:
    out: dict[str, dict[int, dict[str, str]]] = {}
    with SUPPORT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(str(row["seq"]).zfill(2), {})[int(float(row["frame_id"]))] = row
    return out


def parse_role_count(raw: str, role: str) -> int:
    total = 0
    prefix = f"{role}:"
    for part in str(raw or "").split(";"):
        part = part.strip()
        if not part.startswith(prefix):
            continue
        try:
            total += int(part.split(":", 1)[1])
        except ValueError:
            pass
    return total


def semantic_score(row: dict[str, str]) -> float:
    visible = max(1, int(fnum(row.get("visible_track_rows"))))
    stable = parse_role_count(row.get("top_roles", ""), "stable_landmark") / visible
    vegetation = parse_role_count(row.get("top_roles", ""), "vegetation_repetitive") / visible
    weak = parse_role_count(row.get("top_roles", ""), "weak_context") / visible
    dynamic = parse_role_count(row.get("top_roles", ""), "dynamic") / visible
    sky = parse_role_count(row.get("top_roles", ""), "sky_lowobs") / visible
    persistence = fnum(row.get("mean_semantic_persistence_prefix"))
    confidence = fnum(row.get("mean_semantic_confidence_prefix"))
    role_prior = stable + 0.25 * vegetation + 0.10 * weak - 0.35 * dynamic - 0.45 * sky
    return role_prior + 0.20 * persistence + 0.10 * confidence


def normalize(values: dict[int, float], *, invert: bool = False) -> dict[int, float]:
    vals = list(values.values())
    lo = min(vals)
    hi = max(vals)
    if abs(hi - lo) < 1e-12:
        return {frame: 0.5 for frame in values}
    out: dict[int, float] = {}
    for frame, value in values.items():
        norm = (value - lo) / (hi - lo)
        out[frame] = 1.0 - norm if invert else norm
    return out


def corr(xs: list[float], ys: list[float]) -> float:
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def read_anchor_read_stats(seq: str) -> dict[int, dict[str, Any]]:
    path = TRACE_DIR / f"seq{seq}_flashinfer_trace.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    stats: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row_type") != "read":
                continue
            if row.get("memory_family") != "anchor":
                continue
            if row.get("token_type") != "image_patch":
                continue
            frame_raw = row.get("source_frame_id")
            if frame_raw is None:
                continue
            frame = int(frame_raw)
            if frame not in ANCHOR_FRAMES:
                continue
            bucket = stats.setdefault(
                frame,
                {
                    "frame_id": frame,
                    "read_rows": 0,
                    "qk_cosines": [],
                    "qk_softmaxes": [],
                    "qk_ranks": [],
                    "entropies": [],
                },
            )
            bucket["read_rows"] += 1
            bucket["qk_cosines"].append(fnum(row.get("qk_relevance_cosine")))
            bucket["qk_softmaxes"].append(fnum(row.get("qk_relevance_softmax")))
            bucket["qk_ranks"].append(fnum(row.get("qk_relevance_rank")))
            bucket["entropies"].append(fnum(row.get("read_entropy_normalized")))
    missing = [frame for frame in ANCHOR_FRAMES if frame not in stats]
    if missing:
        raise RuntimeError(f"missing anchor read rows for seq {seq}: {missing}")
    for bucket in stats.values():
        n = int(bucket["read_rows"])
        bucket["mean_qk_cosine"] = sum(bucket["qk_cosines"]) / n if n else 0.0
        bucket["mean_qk_softmax"] = sum(bucket["qk_softmaxes"]) / n if n else 0.0
        bucket["mean_qk_rank"] = sum(bucket["qk_ranks"]) / n if n else 0.0
        bucket["mean_read_entropy"] = sum(bucket["entropies"]) / n if n else 0.0
        bucket["qk_cosine_std"] = pstdev(bucket["qk_cosines"]) if n > 1 else 0.0
    return stats


def frame_scores(seq: str, support_by_seq: dict[int, dict[str, str]], internal: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    qk_norm = normalize({frame: fnum(internal[frame]["mean_qk_cosine"]) for frame in ANCHOR_FRAMES})
    softmax_norm = normalize({frame: fnum(internal[frame]["mean_qk_softmax"]) for frame in ANCHOR_FRAMES})
    rank_quality = normalize({frame: fnum(internal[frame]["mean_qk_rank"]) for frame in ANCHOR_FRAMES}, invert=True)
    stability = normalize({frame: -fnum(internal[frame]["qk_cosine_std"]) for frame in ANCHOR_FRAMES})
    sem_norm = normalize({frame: semantic_score(support_by_seq.get(frame, {})) for frame in ANCHOR_FRAMES})
    out: dict[int, dict[str, Any]] = {}
    for frame in ANCHOR_FRAMES:
        internal_score = 0.55 * qk_norm[frame] + 0.30 * softmax_norm[frame] + 0.15 * rank_quality[frame]
        reliability_score = 0.50 * rank_quality[frame] + 0.30 * stability[frame] + 0.20 * softmax_norm[frame]
        row = support_by_seq.get(frame, {})
        out[frame] = {
            "schema": "acl2_v118tf_stage4_r51_fresh_policy_frame_score_row_v1",
            "seq": seq,
            "source_frame": frame,
            "read_rows": int(internal[frame]["read_rows"]),
            "mean_qk_cosine": internal[frame]["mean_qk_cosine"],
            "mean_qk_softmax": internal[frame]["mean_qk_softmax"],
            "mean_qk_rank": internal[frame]["mean_qk_rank"],
            "qk_cosine_std": internal[frame]["qk_cosine_std"],
            "internal_score": internal_score,
            "semantic_score_norm": sem_norm[frame],
            "reliability_score": reliability_score,
            "visible_track_rows": row.get("visible_track_rows", ""),
            "top_roles": row.get("top_roles", ""),
            "best_track_role": row.get("best_track_role", ""),
        }
    return out


def token_stats(seq: str) -> dict[str, Any]:
    filled = np.load(TOKEN_ROOT / f"seq{seq}_filled.npy").astype(bool)
    out: dict[str, Any] = {
        "schema": "acl2_v118tf_stage4_r51_fresh_policy_token_stats_row_v1",
        "seq": seq,
        "token_weight_root": rel(TOKEN_ROOT),
        "filled_mean": float(filled.mean()),
    }
    for channel in ("dynamic", "boundary", "lowtrust", "stable", "weak", "confidence"):
        arr = np.load(TOKEN_ROOT / f"seq{seq}_{channel}.npy")
        vals = arr[filled] if filled.any() else arr.reshape(-1)
        vals = vals.astype("float64", copy=False)
        out[f"{channel}_mean"] = float(vals.mean())
        out[f"{channel}_p90"] = float(np.quantile(vals, 0.90))
        out[f"{channel}_p99"] = float(np.quantile(vals, 0.99))
    out["dynamic_plus_lowtrust_mean"] = out["dynamic_mean"] + out["lowtrust_mean"]
    out["weak_plus_lowtrust_mean"] = out["weak_mean"] + out["lowtrust_mean"]
    out["stable_to_weak_lowtrust"] = out["stable_mean"] / max(1e-9, out["weak_plus_lowtrust_mean"])
    return out


def choose_action(internal_semantic_corr: float, stats: dict[str, Any]) -> tuple[str, str]:
    ratio = fnum(stats["stable_to_weak_lowtrust"])
    dynamic = fnum(stats["dynamic_plus_lowtrust_mean"])
    stable = fnum(stats.get("stable_mean"))
    if POLICY_RULE == "control_safe_boundary_v3":
        if internal_semantic_corr <= CONTROL_SAFE_V2_STRONG_NEG_CORR and CONTROL_SAFE_V2_MID_RATIO_LOW <= ratio < CONTROL_SAFE_V2_MID_RATIO_HIGH:
            return "risk", "strong_negative_corr_mid_ratio_risk_probe"
        if internal_semantic_corr <= 0.0:
            if ratio < CONTROL_SAFE_NEGATIVE_RATIO_LOW:
                return "reverse", "negative_corr_low_ratio_reverse_control_safe"
            if ratio >= CONTROL_SAFE_NEGATIVE_RATIO_HIGH:
                return "reverse", "negative_corr_high_ratio_reverse_control_safe"
            return "abstain", "negative_corr_mid_ratio_control_danger_abstain"
        if (
            internal_semantic_corr >= CONTROL_SAFE_RISK_MIN_CORR
            and ratio >= CONTROL_SAFE_RISK_MIN_RATIO
            and dynamic >= CONTROL_SAFE_RISK_MIN_DYNAMIC
        ):
            return "risk", "positive_corr_high_ratio_dynamic_risk_control_safe"
        if internal_semantic_corr >= 0.65 and ratio < CONTROL_SAFE_V2_POSITIVE_LOW_RATIO_MAX and dynamic < CONTROL_SAFE_V2_POSITIVE_LOW_DYNAMIC_MAX:
            return "risk", "positive_corr_low_ratio_low_dynamic_risk_dev_repair"
        if (
            internal_semantic_corr >= CONTROL_SAFE_V2_STABLE_BOOST_CORR_MIN
            and CONTROL_SAFE_V2_MID_RATIO_LOW <= ratio < CONTROL_SAFE_V2_STABLE_BOOST_RATIO_MAX
            and dynamic >= CONTROL_SAFE_V2_STABLE_BOOST_DYNAMIC_MIN
        ):
            return "risk_only", "positive_corr_mid_ratio_high_dynamic_risk_only_probe"
        if internal_semantic_corr < CONTROL_SAFE_V2_MODERATE_REVERSE_CORR_MAX and ratio >= CONTROL_SAFE_RISK_MIN_RATIO and dynamic >= CONTROL_SAFE_RISK_MIN_DYNAMIC:
            return "reverse", "moderate_positive_corr_high_ratio_reverse_reused_stress"
        return "abstain", "positive_corr_or_ratio_not_control_safe_abstain"
    if POLICY_RULE == "control_safe_boundary_v2":
        if internal_semantic_corr <= CONTROL_SAFE_V2_STRONG_NEG_CORR and CONTROL_SAFE_V2_MID_RATIO_LOW <= ratio < CONTROL_SAFE_V2_MID_RATIO_HIGH:
            return "risk", "strong_negative_corr_mid_ratio_risk_probe"
        if internal_semantic_corr <= 0.0:
            if ratio < CONTROL_SAFE_NEGATIVE_RATIO_LOW:
                return "reverse", "negative_corr_low_ratio_reverse_control_safe"
            if ratio >= CONTROL_SAFE_NEGATIVE_RATIO_HIGH:
                return "reverse", "negative_corr_high_ratio_reverse_control_safe"
            return "abstain", "negative_corr_mid_ratio_control_danger_abstain"
        if (
            internal_semantic_corr >= CONTROL_SAFE_RISK_MIN_CORR
            and ratio >= CONTROL_SAFE_RISK_MIN_RATIO
            and dynamic >= CONTROL_SAFE_RISK_MIN_DYNAMIC
        ):
            return "risk", "positive_corr_high_ratio_dynamic_risk_control_safe"
        if internal_semantic_corr >= 0.65 and ratio < CONTROL_SAFE_V2_POSITIVE_LOW_RATIO_MAX and dynamic < CONTROL_SAFE_V2_POSITIVE_LOW_DYNAMIC_MAX:
            return "risk", "positive_corr_low_ratio_low_dynamic_risk_dev_repair"
        if (
            internal_semantic_corr >= CONTROL_SAFE_V2_STABLE_BOOST_CORR_MIN
            and CONTROL_SAFE_V2_MID_RATIO_LOW <= ratio < CONTROL_SAFE_V2_STABLE_BOOST_RATIO_MAX
            and dynamic >= CONTROL_SAFE_V2_STABLE_BOOST_DYNAMIC_MIN
        ):
            return "stable_boost", "positive_corr_mid_ratio_high_dynamic_stable_boost_probe"
        if internal_semantic_corr < CONTROL_SAFE_V2_MODERATE_REVERSE_CORR_MAX and ratio >= CONTROL_SAFE_RISK_MIN_RATIO and dynamic >= CONTROL_SAFE_RISK_MIN_DYNAMIC:
            return "reverse", "moderate_positive_corr_high_ratio_reverse_reused_stress"
        return "abstain", "positive_corr_or_ratio_not_control_safe_abstain"
    if POLICY_RULE == "control_safe_boundary":
        if internal_semantic_corr <= 0.0:
            if ratio < CONTROL_SAFE_NEGATIVE_RATIO_LOW:
                return "reverse", "negative_corr_low_ratio_reverse_control_safe"
            if ratio >= CONTROL_SAFE_NEGATIVE_RATIO_HIGH:
                return "reverse", "negative_corr_high_ratio_reverse_control_probe"
            return "abstain", "negative_corr_mid_ratio_control_danger_abstain"
        if (
            internal_semantic_corr >= CONTROL_SAFE_RISK_MIN_CORR
            and ratio >= CONTROL_SAFE_RISK_MIN_RATIO
            and dynamic >= CONTROL_SAFE_RISK_MIN_DYNAMIC
        ):
            return "risk", "positive_corr_high_ratio_dynamic_risk_control_safe"
        return "abstain", "positive_corr_or_ratio_not_control_safe_abstain"
    if POLICY_RULE == "stable_dominant_boost":
        if ratio >= STABLE_BOOST_MIN_RATIO and stable >= STABLE_BOOST_MIN_STABLE_MEAN:
            return "stable_boost", "stable_dominant_memory_regime_boost"
        if internal_semantic_corr <= 0.0:
            return "reverse", "nonpositive_internal_semantic_corr"
        if internal_semantic_corr >= RISK_MIN_CORR:
            return "risk", "strong_positive_internal_semantic_corr"
        return "abstain", "no_stable_dominant_or_polarity_regime"
    if POLICY_RULE == "regime_action_sensitivity":
        if internal_semantic_corr <= 0.0:
            if ratio >= NEGATIVE_CORR_RISK_MIN_RATIO or dynamic >= NEGATIVE_CORR_RISK_MIN_DYNAMIC:
                return "risk", "nonpositive_corr_action_sensitive_risk_regime"
            return "reverse", "nonpositive_corr_low_action_sensitivity_reverse_regime"
        if internal_semantic_corr >= RISK_MIN_CORR and ratio >= RISK_MIN_STABLE_TO_WEAK_LOWTRUST:
            return "risk", "positive_corr_stable_guard_risk_regime"
        if internal_semantic_corr >= RISK_MIN_CORR and dynamic >= ACTION_DYNAMIC_MIN:
            return "risk", "positive_corr_dynamic_action_sensitive_risk_regime"
        if internal_semantic_corr >= RISK_MIN_CORR:
            return "abstain", "positive_corr_low_action_sensitivity_abstain_regime"
        if ratio >= 0.20:
            return "reverse", "moderate_positive_corr_high_stable_to_weak_lowtrust"
        return "abstain", "moderate_positive_corr_low_stable_to_weak_lowtrust"
    if internal_semantic_corr <= 0.0:
        return "reverse", "nonpositive_internal_semantic_corr"
    if POLICY_RULE == "stable_guarded_risk":
        if internal_semantic_corr >= RISK_MIN_CORR and ratio >= RISK_MIN_STABLE_TO_WEAK_LOWTRUST:
            return "risk", "strong_positive_internal_semantic_corr_stable_guard_pass"
        if internal_semantic_corr >= RISK_MIN_CORR:
            return "abstain", "strong_positive_internal_semantic_corr_stable_guard_fail"
        if ratio >= 0.20:
            return "reverse", "moderate_positive_corr_high_stable_to_weak_lowtrust"
        return "abstain", "moderate_positive_corr_low_stable_to_weak_lowtrust"
    if POLICY_RULE != "r47":
        raise ValueError(f"unknown ACL2_V118_FRESH_POLICY_RULE={POLICY_RULE!r}")
    if internal_semantic_corr >= 0.50:
        return "risk", "strong_positive_internal_semantic_corr"
    if ratio >= 0.20:
        return "reverse", "moderate_positive_corr_high_stable_to_weak_lowtrust"
    return "abstain", "moderate_positive_corr_low_stable_to_weak_lowtrust"


def policy_rule_lines() -> list[str]:
    if POLICY_RULE == "control_safe_boundary_v3":
        return [
            f"if corr <= {CONTROL_SAFE_V2_STRONG_NEG_CORR:.2f} and {CONTROL_SAFE_V2_MID_RATIO_LOW:.2f} <= stable_to_weak_lowtrust < {CONTROL_SAFE_V2_MID_RATIO_HIGH:.2f}: risk",
            f"elif corr <= 0 and stable_to_weak_lowtrust < {CONTROL_SAFE_NEGATIVE_RATIO_LOW:.2f}: reverse",
            f"elif corr <= 0 and stable_to_weak_lowtrust >= {CONTROL_SAFE_NEGATIVE_RATIO_HIGH:.2f}: reverse",
            "elif corr <= 0: abstain",
            (
                f"elif corr >= {CONTROL_SAFE_RISK_MIN_CORR:.2f} "
                f"and stable_to_weak_lowtrust >= {CONTROL_SAFE_RISK_MIN_RATIO:.2f} "
                f"and dynamic_plus_lowtrust_mean >= {CONTROL_SAFE_RISK_MIN_DYNAMIC:.2f}: risk"
            ),
            f"elif corr >= 0.65 and stable_to_weak_lowtrust < {CONTROL_SAFE_V2_POSITIVE_LOW_RATIO_MAX:.2f} and dynamic_plus_lowtrust_mean < {CONTROL_SAFE_V2_POSITIVE_LOW_DYNAMIC_MAX:.2f}: risk",
            (
                f"elif corr >= {CONTROL_SAFE_V2_STABLE_BOOST_CORR_MIN:.2f} "
                f"and {CONTROL_SAFE_V2_MID_RATIO_LOW:.2f} <= stable_to_weak_lowtrust < {CONTROL_SAFE_V2_STABLE_BOOST_RATIO_MAX:.2f} "
                f"and dynamic_plus_lowtrust_mean >= {CONTROL_SAFE_V2_STABLE_BOOST_DYNAMIC_MIN:.2f}: risk_only"
            ),
            (
                f"elif corr < {CONTROL_SAFE_V2_MODERATE_REVERSE_CORR_MAX:.2f} "
                f"and stable_to_weak_lowtrust >= {CONTROL_SAFE_RISK_MIN_RATIO:.2f} "
                f"and dynamic_plus_lowtrust_mean >= {CONTROL_SAFE_RISK_MIN_DYNAMIC:.2f}: reverse"
            ),
            "else: abstain",
        ]
    if POLICY_RULE == "control_safe_boundary_v2":
        return [
            f"if corr <= {CONTROL_SAFE_V2_STRONG_NEG_CORR:.2f} and {CONTROL_SAFE_V2_MID_RATIO_LOW:.2f} <= stable_to_weak_lowtrust < {CONTROL_SAFE_V2_MID_RATIO_HIGH:.2f}: risk",
            f"elif corr <= 0 and stable_to_weak_lowtrust < {CONTROL_SAFE_NEGATIVE_RATIO_LOW:.2f}: reverse",
            f"elif corr <= 0 and stable_to_weak_lowtrust >= {CONTROL_SAFE_NEGATIVE_RATIO_HIGH:.2f}: reverse",
            "elif corr <= 0: abstain",
            (
                f"elif corr >= {CONTROL_SAFE_RISK_MIN_CORR:.2f} "
                f"and stable_to_weak_lowtrust >= {CONTROL_SAFE_RISK_MIN_RATIO:.2f} "
                f"and dynamic_plus_lowtrust_mean >= {CONTROL_SAFE_RISK_MIN_DYNAMIC:.2f}: risk"
            ),
            f"elif corr >= 0.65 and stable_to_weak_lowtrust < {CONTROL_SAFE_V2_POSITIVE_LOW_RATIO_MAX:.2f} and dynamic_plus_lowtrust_mean < {CONTROL_SAFE_V2_POSITIVE_LOW_DYNAMIC_MAX:.2f}: risk",
            (
                f"elif corr >= {CONTROL_SAFE_V2_STABLE_BOOST_CORR_MIN:.2f} "
                f"and {CONTROL_SAFE_V2_MID_RATIO_LOW:.2f} <= stable_to_weak_lowtrust < {CONTROL_SAFE_V2_STABLE_BOOST_RATIO_MAX:.2f} "
                f"and dynamic_plus_lowtrust_mean >= {CONTROL_SAFE_V2_STABLE_BOOST_DYNAMIC_MIN:.2f}: stable_boost"
            ),
            (
                f"elif corr < {CONTROL_SAFE_V2_MODERATE_REVERSE_CORR_MAX:.2f} "
                f"and stable_to_weak_lowtrust >= {CONTROL_SAFE_RISK_MIN_RATIO:.2f} "
                f"and dynamic_plus_lowtrust_mean >= {CONTROL_SAFE_RISK_MIN_DYNAMIC:.2f}: reverse"
            ),
            "else: abstain",
        ]
    if POLICY_RULE == "control_safe_boundary":
        return [
            f"if corr <= 0 and stable_to_weak_lowtrust < {CONTROL_SAFE_NEGATIVE_RATIO_LOW:.2f}: reverse",
            f"elif corr <= 0 and stable_to_weak_lowtrust >= {CONTROL_SAFE_NEGATIVE_RATIO_HIGH:.2f}: reverse",
            f"elif corr <= 0: abstain",
            (
                f"elif corr >= {CONTROL_SAFE_RISK_MIN_CORR:.2f} "
                f"and stable_to_weak_lowtrust >= {CONTROL_SAFE_RISK_MIN_RATIO:.2f} "
                f"and dynamic_plus_lowtrust_mean >= {CONTROL_SAFE_RISK_MIN_DYNAMIC:.2f}: risk"
            ),
            "else: abstain",
        ]
    if POLICY_RULE == "stable_dominant_boost":
        return [
            f"if stable_to_weak_lowtrust >= {STABLE_BOOST_MIN_RATIO:.2f} and stable_mean >= {STABLE_BOOST_MIN_STABLE_MEAN:.2f}: stable_boost",
            "elif corr <= 0: reverse",
            f"elif corr >= {RISK_MIN_CORR:.2f}: risk",
            "else: abstain",
        ]
    if POLICY_RULE == "regime_action_sensitivity":
        return [
            f"if corr <= 0 and (stable_to_weak_lowtrust >= {NEGATIVE_CORR_RISK_MIN_RATIO:.2f} or dynamic_plus_lowtrust_mean >= {NEGATIVE_CORR_RISK_MIN_DYNAMIC:.2f}): risk",
            "elif corr <= 0: reverse",
            f"elif corr >= {RISK_MIN_CORR:.2f} and stable_to_weak_lowtrust >= {RISK_MIN_STABLE_TO_WEAK_LOWTRUST:.2f}: risk",
            f"elif corr >= {RISK_MIN_CORR:.2f} and dynamic_plus_lowtrust_mean >= {ACTION_DYNAMIC_MIN:.2f}: risk",
            f"elif corr >= {RISK_MIN_CORR:.2f}: abstain",
            "elif stable_to_weak_lowtrust >= 0.20: reverse",
            "else: abstain",
        ]
    if POLICY_RULE == "stable_guarded_risk":
        return [
            "if corr <= 0: reverse",
            f"elif corr >= {RISK_MIN_CORR:.2f} and stable_to_weak_lowtrust >= {RISK_MIN_STABLE_TO_WEAK_LOWTRUST:.2f}: risk",
            f"elif corr >= {RISK_MIN_CORR:.2f}: abstain",
            "elif stable_to_weak_lowtrust >= 0.20: reverse",
            "else: abstain",
        ]
    return [
        "if corr <= 0: reverse",
        "elif corr >= 0.50: risk",
        "elif stable_to_weak_lowtrust >= 0.20: reverse",
        "else: abstain",
    ]


def mode_for_action(action: str) -> str:
    if action == "risk":
        return "risk_suppress_plus_stable_x_frame"
    if action == "risk_only":
        return "risk_suppress_only_x_frame"
    if action == "reverse":
        return "reverse_risk_x_frame"
    if action == "stable_boost":
        return "stable_boost_only_x_frame"
    if action == "random":
        return "same_magnitude_random_logit_x_frame"
    raise ValueError(f"unknown action: {action}")


def method_yaml(policy: str, token_weight_mode: str) -> str:
    weight_map = {str(frame): 1.0 for frame in ANCHOR_FRAMES}
    return "\n".join(
        [
            "model: lingbot_map",
            "env: loger",
            f"_checkpoint: {ROOT / 'third_party/lingbot-map/checkpoints/lingbot-map-long.pt'}",
            "_device: cuda",
            "_use_amp: true",
            "_use_sdpa: true",
            "_image_size: 518",
            "_patch_size: 14",
            "_enable_3d_rope: true",
            "_num_scale_frames: 8",
            "_max_frame_num: 1024",
            "_kv_cache_sliding_window: 64",
            "_kv_cache_scale_frames: 8",
            "_auto_keyframe_threshold: 320",
            "_area_budget: 255000",
            "_align: 14",
            "_mode: streaming",
            "_keyframe_interval: auto",
            "_stage4_action_mode: anchor_source_value_scaling",
            f"_stage4_action_label: {policy}",
            f"_stage4_anchor_source_weight_map: {json.dumps(weight_map, sort_keys=True)}",
            '_stage4_anchor_source_token_roles: ["patch"]',
            "_stage4_anchor_source_query_roles: []",
            '_stage4_anchor_source_context_roles: ["scale_reference_context"]',
            f"_stage4_anchor_source_token_weight_root: {json.dumps(str(TOKEN_ROOT.resolve()))}",
            f"_stage4_anchor_source_token_weight_mode: {json.dumps(token_weight_mode)}",
            "",
        ]
    )


def dataset_yaml(seq: str) -> str:
    return "\n".join(
        [
            "dataset: kitti",
            f"raw_data_root: {ROOT / 'data/kitti/dataset'}",
            "_target_size: [504, 280]",
            f'_sequences: ["{seq}"]',
            "",
        ]
    )


def main_config(seq: str, methods: list[str]) -> str:
    return "\n".join(
        [
            f"workspace: {WORKSPACE}",
            "",
            "evaluation:",
            "  traj:",
            "    enable: true",
            "    vis: true",
            "  auc:",
            "    enable: false",
            "  depth:",
            "    enable: false",
            "  points:",
            "    enable: false",
            "",
            "datasets:",
            f"  - {dataset_name(seq)}",
            "",
            "methods:",
            *[f"  - {method}" for method in methods],
            "",
        ]
    )


def methods_for_seq(seq: str, action: str) -> list[dict[str, Any]]:
    if action == "abstain":
        specs = [
            ("forced_risk_control", "risk", f"{STAGE_LABEL}_FRESH_R47_ABSTAIN_FORCED_RISK_CONTROL"),
            ("forced_reverse_control", "reverse", f"{STAGE_LABEL}_FRESH_R47_ABSTAIN_FORCED_REVERSE_CONTROL"),
            ("forced_random_control", "random", f"{STAGE_LABEL}_FRESH_R47_ABSTAIN_FORCED_RANDOM_CONTROL"),
        ]
    elif action == "stable_boost":
        specs = [
            ("candidate", "stable_boost", f"{STAGE_LABEL}_FRESH_STABLE_BOOST_CANDIDATE"),
            ("token_risk_control", "risk", f"{STAGE_LABEL}_FRESH_STABLE_BOOST_RISK_CONTROL"),
            ("token_risk_only_control", "risk_only", f"{STAGE_LABEL}_FRESH_STABLE_BOOST_RISK_ONLY_CONTROL"),
            ("token_reverse_control", "reverse", f"{STAGE_LABEL}_FRESH_STABLE_BOOST_REVERSE_CONTROL"),
            ("token_random_control", "random", f"{STAGE_LABEL}_FRESH_STABLE_BOOST_RANDOM_CONTROL"),
        ]
    elif action == "risk_only":
        specs = [
            ("candidate", "risk_only", f"{STAGE_LABEL}_FRESH_RISK_ONLY_CANDIDATE"),
            ("token_risk_control", "risk", f"{STAGE_LABEL}_FRESH_RISK_ONLY_PLUS_STABLE_CONTROL"),
            ("token_reverse_control", "reverse", f"{STAGE_LABEL}_FRESH_RISK_ONLY_REVERSE_CONTROL"),
            ("token_stable_boost_control", "stable_boost", f"{STAGE_LABEL}_FRESH_RISK_ONLY_STABLE_BOOST_CONTROL"),
            ("token_random_control", "random", f"{STAGE_LABEL}_FRESH_RISK_ONLY_RANDOM_CONTROL"),
        ]
    elif POLICY_RULE in {"control_safe_boundary", "control_safe_boundary_v2", "control_safe_boundary_v3"}:
        opposite = "reverse" if action == "risk" else "risk"
        specs = [
            ("candidate", action, f"{STAGE_LABEL}_FRESH_CONTROL_SAFE_{action.upper()}_CANDIDATE"),
            ("token_opposite_polarity_control", opposite, f"{STAGE_LABEL}_FRESH_CONTROL_SAFE_{opposite.upper()}_OPPOSITE_CONTROL"),
            ("token_stable_boost_control", "stable_boost", f"{STAGE_LABEL}_FRESH_CONTROL_SAFE_STABLE_BOOST_CONTROL"),
            ("token_random_control", "random", f"{STAGE_LABEL}_FRESH_CONTROL_SAFE_RANDOM_CONTROL"),
        ]
    else:
        opposite = "reverse" if action == "risk" else "risk"
        specs = [
            ("candidate", action, f"{STAGE_LABEL}_FRESH_R47_POLICY_{action.upper()}_CANDIDATE"),
            ("token_opposite_polarity_control", opposite, f"{STAGE_LABEL}_FRESH_R47_POLICY_{opposite.upper()}_OPPOSITE_CONTROL"),
            ("token_random_control", "random", f"{STAGE_LABEL}_FRESH_R47_POLICY_RANDOM_CONTROL"),
        ]
    rows: list[dict[str, Any]] = []
    for role, action_name, policy in specs:
        method = f"{METHOD_PREFIX}_{role}_seq{seq}"
        rows.append(
            {
                "seq": seq,
                "dataset": dataset_name(seq),
                "method": method,
                "role": role,
                "action": action_name,
                "policy": policy,
                "token_weight_mode": mode_for_action(action_name),
            }
        )
    return rows


def run_env(seq: str, method: str, policy: str, gpu: int, action_trace: Path) -> str:
    dataset = dataset_name(seq)
    return (
        f"PATH=/mnt/data/users/chengshun.wang/miniconda3/bin:$PATH "
        f"PYTHONPATH={PYTHONPATH} "
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"ACL2_V105_STAGE4_ACTION_FILE={action_trace.resolve()} "
        f"ACL2_V105_STAGE4_ACTION_LABEL={policy} "
        f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
        f"ACL2_V105_GCA_TRACE_SEQ={seq} "
        f"ACL2_V105_GCA_TRACE_METHOD={method} "
        f"ACL2_V118_LB_PROVENANCE_SEQ={seq}"
    )


def main() -> int:
    if not SUPPORT.is_file():
        raise FileNotFoundError(SUPPORT)
    if not TOKEN_ROOT.is_dir():
        raise FileNotFoundError(TOKEN_ROOT)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "datasets").mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "methods").mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    (RUNTIME / "logs").mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    support = read_support()
    frame_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for seq in SEQS:
        scores = frame_scores(seq, support.get(seq, {}), read_anchor_read_stats(seq))
        stats = token_stats(seq)
        token_rows.append(stats)
        internal_semantic_corr = corr(
            [float(scores[frame]["internal_score"]) for frame in ANCHOR_FRAMES],
            [float(scores[frame]["semantic_score_norm"]) for frame in ANCHOR_FRAMES],
        )
        action, reason = choose_action(internal_semantic_corr, stats)
        frame_rows.extend(scores[frame] | {"internal_semantic_corr": internal_semantic_corr} for frame in ANCHOR_FRAMES)
        seq_methods = methods_for_seq(seq, action)
        method_rows.extend(seq_methods)
        write_text(CONFIG_DIR / "datasets" / f"{dataset_name(seq)}.yaml", dataset_yaml(seq))
        write_text(CONFIG_DIR / f"kitti_lingbot_sdpa_v118_{STAGE_TAG}_fresh_r47_policy_seq{seq}.yaml", main_config(seq, [row["method"] for row in seq_methods]))

        policy_rows.append(
            {
                "schema": "acl2_v118tf_stage4_r51_fresh_r47_policy_row_v1",
                "seq": seq,
                "dataset": dataset_name(seq),
                "policy_action": action,
                "selection_reason": reason,
                "internal_semantic_corr": internal_semantic_corr,
                "stable_to_weak_lowtrust": stats["stable_to_weak_lowtrust"],
                "dynamic_plus_lowtrust_mean": stats["dynamic_plus_lowtrust_mean"],
                "selected_metric_source": "baseline" if action == "abstain" else "candidate_run",
                "candidate_method": next((row["method"] for row in seq_methods if row["role"] == "candidate"), ""),
                "control_methods": ";".join(row["method"] for row in seq_methods if row["role"] != "candidate"),
            }
        )

        for spec in seq_methods:
            method = str(spec["method"])
            write_text(CONFIG_DIR / "methods" / f"{method}.yaml", method_yaml(str(spec["policy"]), str(spec["token_weight_mode"])))
            action_trace = ACTION_DIR / f"{method}_seq{seq}.jsonl"
            config = CONFIG_DIR / f"kitti_lingbot_sdpa_v118_{STAGE_TAG}_fresh_r47_policy_seq{seq}.yaml"
            log = RUNTIME / "logs" / f"run_seq{seq}_{method}.log"
            cleanup_log = RUNTIME / "logs" / f"cleanup_seq{seq}_{method}.log"
            gpu = len([row for row in run_rows if row.get("phase") == "run_worker"]) % 6
            manifest_rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r51_fresh_policy_method_manifest_row_v1",
                    **spec,
                    "branch": "LB-AR",
                    "action_mode": "anchor_source_value_scaling",
                    "source_frames": ";".join(str(frame) for frame in ANCHOR_FRAMES),
                    "source_context_roles": "scale_reference_context",
                    "token_roles": "patch",
                    "query_roles": "all",
                    "uniform_frame_weight": 1.0,
                    "token_weight_root": rel(TOKEN_ROOT),
                    "action_trace": rel(action_trace),
                    "config": rel(config),
                }
            )
            run_rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r51_fresh_policy_run_manifest_row_v1",
                    "phase": "cleanup_trace",
                    "seq": seq,
                    "dataset": dataset_name(seq),
                    "method": method,
                    "gpu": "",
                    "cwd": str(BENCH),
                    "config": str(config.resolve()),
                    "action_trace": rel(action_trace),
                    "log": rel(cleanup_log),
                    "command": f"rm -f {action_trace.resolve()} > {cleanup_log} 2>&1",
                }
            )
            run_rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r51_fresh_policy_run_manifest_row_v1",
                    "phase": "run_worker",
                    "seq": seq,
                    "dataset": dataset_name(seq),
                    "method": method,
                    "gpu": gpu,
                    "cwd": str(BENCH),
                    "config": str(config.resolve()),
                    "action_trace": rel(action_trace),
                    "log": rel(log),
                    "command": (
                        f"{run_env(seq, method, str(spec['policy']), gpu, action_trace)} "
                        f"{CONDA} run -n loger --no-capture-output python run_worker.py "
                        f"--config {config.resolve()} --method {method} --dataset {dataset_name(seq)} "
                        f"--scene {seq} --force > {log} 2>&1"
                    ),
                }
            )
        eval_log = RUNTIME / "logs" / f"evaluate_seq{seq}.log"
        config = CONFIG_DIR / f"kitti_lingbot_sdpa_v118_{STAGE_TAG}_fresh_r47_policy_seq{seq}.yaml"
        run_rows.append(
            {
                "schema": "acl2_v118tf_stage4_r51_fresh_policy_run_manifest_row_v1",
                "phase": "evaluate",
                "seq": seq,
                "dataset": dataset_name(seq),
                "method": ";".join(row["method"] for row in seq_methods),
                "gpu": "",
                "cwd": str(BENCH),
                "config": str(config.resolve()),
                "action_trace": "",
                "log": rel(eval_log),
                "command": (
                    f"PATH=/mnt/data/users/chengshun.wang/miniconda3/bin:$PATH PYTHONPATH={PYTHONPATH} "
                    f"{CONDA} run -n loger --no-capture-output python evaluate.py --config {config.resolve()} "
                    f"--force > {eval_log} 2>&1"
                ),
            }
        )

    write_csv(SUMMARY_DIR / "stage4_r51_fresh_policy_frame_score_rows.csv", frame_rows)
    write_csv(SUMMARY_DIR / "stage4_r51_fresh_policy_token_stats_rows.csv", token_rows)
    write_csv(SUMMARY_DIR / "stage4_r51_fresh_policy_rows.csv", policy_rows)
    write_csv(SUMMARY_DIR / "stage4_r51_fresh_policy_method_manifest.csv", manifest_rows)
    write_csv(STAGE / "run_manifest.csv", run_rows)
    summary = {
        "schema": "acl2_v118tf_stage4_r51_fresh_policy_config_summary_v1",
        f"stage4_{STAGE_TAG}_decision": "FRESH_R47_POLICY_CONFIGS_READY_NOT_RUN",
        "global_goal_achieved": False,
        "sequences": list(SEQS),
        "baseline_method": BASELINE_METHOD,
        "policy_rule_name": POLICY_RULE,
        "workspace": rel(WORKSPACE),
        "policy_rule": policy_rule_lines(),
        "policy_rows": policy_rows,
        "method_count": len(manifest_rows),
        "outputs": {
            "policy_rows": rel(SUMMARY_DIR / "stage4_r51_fresh_policy_rows.csv"),
            "method_manifest": rel(SUMMARY_DIR / "stage4_r51_fresh_policy_method_manifest.csv"),
            "run_manifest": rel(STAGE / "run_manifest.csv"),
        },
        "boundary": (
            f"{STAGE_LABEL} config generation only. It applies the selected policy rule to fresh {SEQ_LABEL} evidence and prepares candidate/control "
            "runs; no fresh policy success is claimed until runtime, evaluation, and action-fidelity summary pass."
        ),
    }
    write_json(STAGE / "config_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
