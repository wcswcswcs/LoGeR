#!/usr/bin/env python3
"""Analyze v96 raw-QK source-marginal carrier localization.

This diagnostic consumes existing raw-QK smoke dumps.  It is deliberately
conservative: the dumps are full-query source marginals, not full pairwise
per-head attention matrices, so this script can localize source-region/layer
mass but cannot claim exact per-head carrier attribution.  When sampled
query-to-source matrices are available, they are reduced to sampled-query
source marginals per head and kept diagnostic-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch


ROOT = Path("results/acl2_v96tf_vggt4d_informed_semantic_gauge_preserving_memory_control")
DEFAULT_INPUT_ROOTS = [
    ROOT / "trackJ_raw_qk_trace_smoke_trackg_head_marginal",
    ROOT / "trackJ_raw_qk_trace_smoke",
    ROOT / "trackJ_raw_qk_trace_smoke_trackg_coverage",
]
OUT_DIR = ROOT / "trackG_read_qk_carrier_localization"
MASK_BANK = ROOT / "trackJ_semantic_region_bank" / "semantic_region_masks.pt"
REGIONS = [
    "WEAK_SCALE_CONTEXT",
    "VEGETATION_REPETITIVE",
    "LOW_OBSERVABILITY",
    "STABLE_ANCHOR",
    "OBJECT_BOUNDARY_BAND",
    "MULTIMODE_CONFLICT",
    "DYNAMIC_OBJECT",
    "UNKNOWN_CONTEXT",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}", "path": str(path)}
    return payload if isinstance(payload, dict) else {"value": payload}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def load_case_label(case_dir: Path) -> str:
    for variant in ("trace_noop", "baseline_noop", "action_trace_probe"):
        summary = read_json(case_dir / variant / "job_summary.json")
        if summary.get("label"):
            return str(summary.get("label"))
    return ""


def region_mask_for(raw: dict[str, Any], case_masks: dict[str, Any], region: str) -> torch.Tensor | None:
    region_masks = case_masks.get("region_token_masks", {}) if isinstance(case_masks, dict) else {}
    mask = region_masks.get(region)
    if not torch.is_tensor(mask):
        return None
    mask = mask.float().reshape(mask.shape[0], -1)
    source_token_count = int(raw.get("source_token_count", 0) or raw["source_attention_before_marginal"].shape[1])
    special_tokens = source_token_count - int(mask.shape[1])
    if special_tokens < 0:
        return None
    prefix = torch.zeros((mask.shape[0], special_tokens), dtype=mask.dtype)
    return torch.cat([prefix, mask], dim=1)


def masked_mass(marginal: torch.Tensor, mask: torch.Tensor | None) -> tuple[float, float, float]:
    if mask is None or tuple(mask.shape) != tuple(marginal.shape):
        return 0.0, 0.0, 0.0
    mass = float((marginal.float() * mask.float()).sum(dim=1).mean().item())
    token_fraction = float(mask.float().mean(dim=1).mean().item())
    enrichment = mass / token_fraction if token_fraction > 0 else 0.0
    return mass, token_fraction, enrichment


def collect_rows(input_roots: list[Path], mask_bank: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for root in input_roots:
        if not root.is_dir():
            continue
        for case_dir in sorted(path for path in root.iterdir() if path.is_dir() and "_" in path.name):
            case_id = case_dir.name
            label = load_case_label(case_dir)
            seq = case_id.split("_")[0]
            chunk = int(case_id.split("_")[-1])
            case_masks = mask_bank.get(case_id, {})
            for variant in ("trace_noop", "action_trace_probe"):
                dump_dir = case_dir / variant / "raw_qk_attention_dumps"
                for dump_path in sorted(dump_dir.glob("*.pt")):
                    raw = torch.load(dump_path, map_location="cpu")
                    layer = int(raw.get("layer", -1))
                    key = (case_id, variant, layer)
                    if key in seen:
                        continue
                    seen.add(key)
                    head_before = raw.get("source_attention_before_head_marginal")
                    head_after = raw.get("source_attention_after_bias_head_marginal")
                    sampled_before = raw.get("attention_before_control")
                    sampled_after = raw.get("attention_after_bias_control")
                    per_head_available = torch.is_tensor(head_before) and head_before.ndim == 3
                    if per_head_available:
                        marginal_pairs = []
                        for head in range(int(head_before.shape[1])):
                            before = head_before[:, head, :].float()
                            after = (
                                head_after[:, head, :].float()
                                if torch.is_tensor(head_after) and head_after.shape == head_before.shape else before
                            )
                            marginal_pairs.append((head, before, after, True))
                    elif torch.is_tensor(sampled_before) and sampled_before.ndim == 4:
                        sampled_after_ok = torch.is_tensor(sampled_after) and sampled_after.shape == sampled_before.shape
                        marginal_pairs = []
                        for head in range(int(sampled_before.shape[1])):
                            before = sampled_before[:, head, :, :].float().mean(dim=1)
                            after = (
                                sampled_after[:, head, :, :].float().mean(dim=1)
                                if sampled_after_ok else before
                            )
                            marginal_pairs.append((head, before, after, True))
                    else:
                        before_raw = raw.get("source_attention_before_marginal")
                        if not torch.is_tensor(before_raw):
                            continue
                        before = before_raw.float()
                        after_raw = raw.get("source_attention_after_bias_marginal")
                        after = after_raw.float() if torch.is_tensor(after_raw) else before
                        marginal_pairs = [(-1, before, after, False)]
                    affected = raw.get("affected_mask")
                    affected_fraction = float(affected.float().mean().item()) if torch.is_tensor(affected) else 0.0
                    for head, before, after, row_per_head in marginal_pairs:
                        row: dict[str, Any] = {
                            "input_root": str(root),
                            "case_id": case_id,
                            "seq": seq,
                            "chunk": chunk,
                            "label": label,
                            "variant": variant,
                            "layer": layer,
                            "head": head,
                            "per_head_source_marginal_available": row_per_head,
                            "raw_dump": str(dump_path),
                            "schema": raw.get("schema", ""),
                            "attention_source": raw.get("attention_source", ""),
                            "pairwise_attention_matrix_stored": raw.get("pairwise_attention_matrix_stored", ""),
                            "sampled_pairwise_attention_matrix_stored": raw.get("sampled_pairwise_attention_matrix_stored", ""),
                            "sampled_query_count": (
                                int(raw.get("query_indices").numel()) if torch.is_tensor(raw.get("query_indices")) else ""
                            ),
                            "query_axis_fully_covered": raw.get("query_axis_fully_covered", ""),
                            "q_shape": "x".join(str(v) for v in raw.get("q_shape", [])),
                            "k_shape": "x".join(str(v) for v in raw.get("k_shape", [])),
                            "source_token_count": raw.get("source_token_count", ""),
                            "affected_token_fraction": affected_fraction,
                            "affected_mass_before": (
                                float((before.float() * affected.float()).sum(dim=-1).mean().item())
                                if torch.is_tensor(affected) else 0.0
                            ),
                            "affected_mass_after": (
                                float((after.float() * affected.float()).sum(dim=-1).mean().item())
                                if torch.is_tensor(affected) else 0.0
                            ),
                        }
                        row["affected_mass_delta"] = row["affected_mass_after"] - row["affected_mass_before"]
                        for region in REGIONS:
                            mask = region_mask_for(raw, case_masks, region)
                            before_mass, token_fraction, before_enrichment = masked_mass(before, mask)
                            after_mass, _, after_enrichment = masked_mass(after, mask)
                            key_prefix = region.lower()
                            row[f"{key_prefix}_mass_before"] = before_mass
                            row[f"{key_prefix}_mass_after"] = after_mass
                            row[f"{key_prefix}_mass_delta"] = after_mass - before_mass
                            row[f"{key_prefix}_token_fraction"] = token_fraction
                            row[f"{key_prefix}_enrichment_before"] = before_enrichment
                            row[f"{key_prefix}_enrichment_after"] = after_enrichment
                        lowstuff_before = (
                            row["weak_scale_context_mass_before"]
                            + row["vegetation_repetitive_mass_before"]
                            + row["low_observability_mass_before"]
                        )
                        stable_before = row["stable_anchor_mass_before"]
                        row["lowstuff_mass_before"] = lowstuff_before
                        row["weak_over_stable_attention_mass"] = row["weak_scale_context_mass_before"] / (stable_before + 1.0e-6)
                        rows.append(row)
    return sorted(rows, key=lambda item: (item["case_id"], item["variant"], int(item["layer"]), int(item["head"])))


def summarize_layers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    layers = sorted({int(row["layer"]) for row in rows})
    heads = sorted({int(row.get("head", -1)) for row in rows})
    for variant in ("trace_noop", "action_trace_probe"):
        for layer in layers:
            for head in heads:
                layer_rows = [
                    row for row in rows
                    if row["variant"] == variant and int(row["layer"]) == layer and int(row.get("head", -1)) == head
                ]
                bad = [row for row in layer_rows if row.get("label") == "READ_LOCAL_BAD"]
                good = [row for row in layer_rows if row.get("label") == "GOOD_CONTROL"]
                if not bad or not good:
                    continue
                bad_delta = median([safe_float(row.get("affected_mass_delta")) for row in bad])
                good_delta = median([safe_float(row.get("affected_mass_delta")) for row in good])
                out.append(
                    {
                        "variant": variant,
                        "layer": layer,
                        "head": head,
                        "per_head_source_marginal_available": any(boolish(row.get("per_head_source_marginal_available")) for row in layer_rows),
                        "bad_case_count": len(bad),
                        "good_case_count": len(good),
                        "bad_seq_coverage": len({row["seq"] for row in bad}),
                        "good_seq_coverage": len({row["seq"] for row in good}),
                        "bad_median_affected_mass_before": median([safe_float(row.get("affected_mass_before")) for row in bad]),
                        "good_median_affected_mass_before": median([safe_float(row.get("affected_mass_before")) for row in good]),
                        "affected_mass_bad_minus_good": median([safe_float(row.get("affected_mass_before")) for row in bad])
                        - median([safe_float(row.get("affected_mass_before")) for row in good]),
                        "bad_median_affected_mass_delta": bad_delta,
                        "good_median_affected_mass_delta": good_delta,
                        "selective_suppression_margin_good_delta_minus_bad_delta": good_delta - bad_delta,
                        "bad_median_weak_mass_before": median([safe_float(row.get("weak_scale_context_mass_before")) for row in bad]),
                        "good_median_weak_mass_before": median([safe_float(row.get("weak_scale_context_mass_before")) for row in good]),
                        "bad_median_stable_mass_before": median([safe_float(row.get("stable_anchor_mass_before")) for row in bad]),
                        "good_median_stable_mass_before": median([safe_float(row.get("stable_anchor_mass_before")) for row in good]),
                        "bad_median_weak_over_stable_attention_mass": median([safe_float(row.get("weak_over_stable_attention_mass")) for row in bad]),
                        "good_median_weak_over_stable_attention_mass": median([safe_float(row.get("weak_over_stable_attention_mass")) for row in good]),
                    }
                )
    return out


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    input_roots = [Path(item) for item in args.input_roots.split(",") if item.strip()]
    mask_bank = torch.load(args.mask_bank, map_location="cpu")
    rows = collect_rows(input_roots, mask_bank)
    layer_rows = summarize_layers(rows)
    trace_layer_rows = [row for row in layer_rows if row["variant"] == "trace_noop"]
    action_layer_rows = [row for row in layer_rows if row["variant"] == "action_trace_probe"]
    bad_cases = sorted({row["case_id"] for row in rows if row.get("label") == "READ_LOCAL_BAD"})
    good_cases = sorted({row["case_id"] for row in rows if row.get("label") == "GOOD_CONTROL"})
    per_head_rows = [row for row in rows if boolish(row.get("per_head_source_marginal_available"))]
    per_head_bad_cases = sorted({row["case_id"] for row in per_head_rows if row.get("label") == "READ_LOCAL_BAD"})
    per_head_good_cases = sorted({row["case_id"] for row in per_head_rows if row.get("label") == "GOOD_CONTROL"})
    per_head_available = bool(per_head_rows)
    per_head_coverage_pass = (
        len(per_head_bad_cases) >= 4
        and len(per_head_good_cases) >= 4
        and len({case.split("_")[0] for case in per_head_bad_cases}) >= 3
    )
    max_affected_margin = max([safe_float(row.get("affected_mass_bad_minus_good")) for row in trace_layer_rows] or [0.0])
    max_selective_suppression = max(
        [safe_float(row.get("selective_suppression_margin_good_delta_minus_bad_delta")) for row in action_layer_rows] or [0.0]
    )
    full_pairwise_available = all(boolish(row.get("pairwise_attention_matrix_stored")) for row in rows) if rows else False
    sampled_pairwise_available = any(boolish(row.get("sampled_pairwise_attention_matrix_stored")) for row in rows)
    coverage_pass = len(bad_cases) >= 4 and len(good_cases) >= 4 and len({case.split("_")[0] for case in bad_cases}) >= 3
    layer_specificity_pass = max_affected_margin >= args.affected_margin_threshold
    action_selectivity_pass = max_selective_suppression >= args.action_selectivity_threshold
    carrier_evidence_available = bool(full_pairwise_available or per_head_coverage_pass)
    carrier_gate_pass = bool(carrier_evidence_available and coverage_pass and layer_specificity_pass and action_selectivity_pass)
    classification = (
        "RAW_QK_PAIRWISE_OR_PER_HEAD_CARRIER_PASS"
        if carrier_gate_pass
        else "RAW_QK_PER_HEAD_SOURCE_MARGINAL_NO_GEOMETRY_SPECIFIC_CARRIER"
        if per_head_available
        else "RAW_QK_SOURCE_MARGINAL_NO_GEOMETRY_SPECIFIC_LAYER_CARRIER"
    )
    summary = {
        "stage": "TrackG_READ_raw_QK_carrier_localization",
        "status": "complete",
        "method_success": False,
        "mechanism_success": False,
        "runtime_action_allowed": False,
        "classification": classification,
        "carrier_localization_gate_pass": carrier_gate_pass,
        "coverage_pass": coverage_pass,
        "full_pairwise_per_head_available": full_pairwise_available,
        "sampled_pairwise_attention_available": sampled_pairwise_available,
        "per_head_source_marginal_available": per_head_available,
        "per_head_coverage_pass": per_head_coverage_pass,
        "source_marginal_only": not full_pairwise_available,
        "layer_specificity_pass": layer_specificity_pass,
        "action_selectivity_pass": action_selectivity_pass,
        "case_counts": {
            "read_bad_cases": len(bad_cases),
            "good_control_cases": len(good_cases),
            "read_bad_seq_coverage": len({case.split("_")[0] for case in bad_cases}),
            "good_control_seq_coverage": len({case.split("_")[0] for case in good_cases}),
            "per_head_read_bad_cases": len(per_head_bad_cases),
            "per_head_good_control_cases": len(per_head_good_cases),
            "per_head_read_bad_seq_coverage": len({case.split("_")[0] for case in per_head_bad_cases}),
            "per_head_good_control_seq_coverage": len({case.split("_")[0] for case in per_head_good_cases}),
        },
        "bad_cases": bad_cases,
        "good_cases": good_cases,
        "layer_count": len({int(row["layer"]) for row in rows}),
        "layers": sorted({int(row["layer"]) for row in rows}),
        "head_count": len({int(row.get("head", -1)) for row in rows if int(row.get("head", -1)) >= 0}),
        "heads": sorted({int(row.get("head", -1)) for row in rows if int(row.get("head", -1)) >= 0}),
        "per_layer_rows": len(layer_rows),
        "per_case_layer_rows": len(rows),
        "max_affected_mass_bad_minus_good": max_affected_margin,
        "max_selective_suppression_margin_good_delta_minus_bad_delta": max_selective_suppression,
        "affected_margin_threshold": args.affected_margin_threshold,
        "action_selectivity_threshold": args.action_selectivity_threshold,
        "input_roots": [str(path) for path in input_roots],
        "mask_bank": str(args.mask_bank),
        "gate_rule": (
            "carrier gate requires full pairwise per-head dumps or per-head source marginal evidence, "
            ">=4 READ bad cases, >=4 good controls, READ bad seq coverage >=3, "
            "affected_mass_bad_minus_good >= threshold, and action suppression selectively stronger on bad cases."
        ),
    }
    write_csv(out / "per_case_layer_region_rows.csv", rows)
    write_csv(out / "layer_group_summary.csv", layer_rows)
    write_csv(out / "rows.csv", layer_rows)
    write_csv(
        out / "gate_checks.csv",
        [
            {"gate": "coverage_pass", "pass": coverage_pass, "value": summary["case_counts"]},
            {"gate": "full_pairwise_per_head_available", "pass": full_pairwise_available, "value": full_pairwise_available},
            {"gate": "sampled_pairwise_attention_available", "pass": sampled_pairwise_available, "value": sampled_pairwise_available},
            {"gate": "per_head_source_marginal_available", "pass": per_head_available, "value": per_head_available},
            {"gate": "per_head_coverage_pass", "pass": per_head_coverage_pass, "value": summary["case_counts"]},
            {"gate": "source_marginal_layer_specificity_pass", "pass": layer_specificity_pass, "value": max_affected_margin},
            {"gate": "action_selectivity_pass", "pass": action_selectivity_pass, "value": max_selective_suppression},
            {"gate": "carrier_localization_gate_pass", "pass": carrier_gate_pass, "value": classification},
            {"gate": "runtime_action_allowed", "pass": False, "value": False},
        ],
    )
    write_json(out / "summary.json", summary)
    interpretation = (
        "Per-head source-marginal raw-QK evidence exposes a candidate READ-bad-specific layer/head/source "
        "carrier under the diagnostic thresholds. This is still not a runtime method: the dump is not a full "
        "pairwise matrix, runtime action is not promoted, and a separate layer/head-scoped action pilot must "
        "pass geometry, good-control, trace-fidelity, and stable-anchor gates before Stage7."
        if carrier_gate_pass else
        "The available raw-QK source marginal does not expose a service-ready READ-bad-specific layer/head/source "
        "carrier. The tested action probe suppresses affected source mass without enough bad-case specificity, "
        "so the trace supports the earlier conclusion that this actuator is not geometry-specific enough."
    )
    write_text(
        out / "carrier_localization_report.md",
        f"""# Track G READ Raw-QK Carrier Localization

Status: diagnostic-only. No runtime action is promoted.

Coverage:

- READ bad cases: `{len(bad_cases)}` / `{','.join(bad_cases)}`
- good controls: `{len(good_cases)}` / `{','.join(good_cases)}`
- layers: `{','.join(str(layer) for layer in summary['layers'])}`

Dump limitation:

- full pairwise per-head matrices available: `{full_pairwise_available}`
- sampled pairwise source-target matrices available: `{sampled_pairwise_available}`
- per-head source marginals available: `{per_head_available}`
- available signal: full-query source marginal attention, with per-head marginals when present

Key result:

- max affected-mass bad-good margin: `{max_affected_margin}`
- max selective action-suppression margin: `{max_selective_suppression}`
- carrier gate pass: `{carrier_gate_pass}`

Interpretation:

{interpretation}
""",
    )
    failure_text = (
        "# Raw-QK Carrier Localization Diagnostic Report\n\n"
        f"Carrier diagnostic pass. Classification: `{classification}`. This is still diagnostic-only: "
        "runtime action remains blocked until a layer/head-scoped action pilot passes geometry, "
        "good-control, trace-fidelity, and stable-anchor gates."
        if carrier_gate_pass else
        "# Raw-QK Carrier Localization Failure Report\n\n"
        f"No carrier pass. Classification: `{classification}`. The dump family is source-marginal/sample-only, "
        "and the observed layer/source-region signal is not READ-bad-specific enough for a runtime action."
    )
    write_text(out / "failure_report.md", failure_text)
    write_text(
        out / "what_would_have_to_be_true_to_pass.md",
        "# What Would Have To Be True To Pass\n\nA future run needs full pairwise per-head raw-QK dumps or equivalent per-head carrier evidence, then a layer/head/region signal that is stronger on READ bad cases than good controls and whose action-probe suppression is selective to READ bad cases before any runtime action can be promoted.",
    )
    write_csv(out / "visual_manifest.csv", [])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", default=",".join(str(path) for path in DEFAULT_INPUT_ROOTS))
    parser.add_argument("--mask-bank", type=Path, default=MASK_BANK)
    parser.add_argument("--output-root", type=Path, default=OUT_DIR)
    parser.add_argument("--affected-margin-threshold", type=float, default=0.005)
    parser.add_argument("--action-selectivity-threshold", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps({k: summary[k] for k in ("status", "classification", "carrier_localization_gate_pass", "runtime_action_allowed")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
