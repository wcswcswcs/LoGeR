from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .v47_common import ROOT, parse_bool, parse_int, read_csv, utc_now, write_csv, write_json


DEFAULT_V62_QUERY_CANDIDATES = "outputs/audit/v62_active_query_refresh/query_candidate_rows.csv"
DEFAULT_V62_NOVELTY_MATERIAL = "outputs/audit/v62_increment_attribution/novelty_material_rows.csv"


@dataclass(frozen=True)
class V63QueryCandidateConfig:
    v62_query_candidate_rows: str | Path = DEFAULT_V62_QUERY_CANDIDATES
    v62_novelty_material_rows: str | Path = DEFAULT_V62_NOVELTY_MATERIAL
    output_root: str | Path = "outputs/audit/v63_query_candidates"
    visualization_root: str | Path = "outputs/audit/v63_visualizations/query_candidates"
    per_type_budget: int = 64
    per_control_budget: int = 64


def build_v63_query_candidates(config: V63QueryCandidateConfig | None = None) -> dict[str, Any]:
    cfg = config or V63QueryCandidateConfig()
    query_rows = read_csv(_project(cfg.v62_query_candidate_rows))
    novelty_rows = read_csv(_project(cfg.v62_novelty_material_rows))
    novelty_by_material = _novelty_by_material(novelty_rows)
    enriched = [_enrich_query_row(row, novelty_by_material.get(row.get("material_node_id", ""), {})) for row in query_rows]
    novelty_enriched = [_enrich_novelty_row(row) for row in novelty_rows]

    method_rows: list[dict[str, Any]] = []
    heldout = _select_rows(
        enriched,
        cfg.per_type_budget,
        lambda row: (
            row.get("candidate_source") in {"bridge_low_support", "update_new_low_support"}
            and parse_bool(row.get("has_K_mat"))
            and row.get("state") == "confirmed"
        ),
        "heldout_recovery",
    )
    method_rows.extend(_method_rows(heldout, "heldout_recovery", "recover low-support K_mat ownership edge with heldout validation"))

    shortcut = _select_rows(
        enriched,
        cfg.per_type_budget,
        lambda row: row.get("candidate_source") == "shared_shortcut_boundary"
        or row.get("state") == "shared"
        or row.get("novelty_type") == "shortcut_shared",
        "shortcut_quarantine",
    )
    method_rows.extend(_method_rows(shortcut, "shortcut_quarantine", "quarantine shared shortcut or undersegmented ownership candidate"))

    unknown = _select_rows(
        novelty_enriched,
        cfg.per_type_budget,
        lambda row: row.get("state") in {"unknown", "tentative"}
        or (
            not parse_bool(row.get("has_K_mat"))
            and not parse_bool(row.get("has_K_mask"))
            and not parse_bool(row.get("has_K_sem"))
        ),
        "unknown_defer",
    )
    method_rows.extend(_method_rows(unknown, "unknown_defer", "safe defer for low-evidence or unknown ownership candidate"))

    decoy = _decoy_rows(heldout, cfg.per_type_budget)
    method_rows.extend(decoy)

    numbered_method_rows = _renumber(method_rows, "v63q")
    control_rows = _control_rows(enriched, novelty_enriched, numbered_method_rows, cfg.per_control_budget)
    numbered_control_rows = _renumber(control_rows, "v63c")
    heldout_edge_rows = _heldout_edge_rows([row for row in numbered_method_rows if row["candidate_type"] == "heldout_recovery"])
    all_candidate_rows = [*numbered_method_rows, *numbered_control_rows]
    method_counts = Counter(row["candidate_type"] for row in method_rows)
    control_counts = Counter(row["control_id"] for row in control_rows)
    gate = {
        "all_required_method_candidate_types_present": all(
            method_counts.get(name, 0) > 0
            for name in ["heldout_recovery", "shortcut_quarantine", "decoy_rejection", "unknown_defer"]
        ),
        "method_candidate_type_counts_balanced": len(set(method_counts.values())) == 1 and len(method_counts) == 4,
        "all_required_controls_present": all(control_counts.get(name, 0) > 0 for name in ["C0_v62_original", "C1_random_matched", "C2_mask_boundary", "C3_semantic_only", "C4_K_mask_only_ablation"]),
        "control_counts_balanced": len(set(control_counts.values())) == 1 and len(control_counts) == 5,
        "heldout_edge_rows_available": len(heldout_edge_rows) == method_counts.get("heldout_recovery", 0),
        "selection_uses_gt_for_prediction": False,
        "diagnostic_labels_not_used_for_query_selection": True,
    }
    gate["pass"] = bool(all(value is True or value is False and key == "selection_uses_gt_for_prediction" for key, value in gate.items()))
    summary = {
        "phase": "v63_query_candidates",
        "created_at": utc_now(),
        "method_status": "candidate_protocol_only_no_D4RT_query_run",
        "method_candidate_count": len(method_rows),
        "method_candidate_type_counts": dict(method_counts),
        "baseline_control_count": len(control_rows),
        "baseline_control_counts": dict(control_counts),
        "heldout_edge_count": len(heldout_edge_rows),
        "per_type_budget": int(cfg.per_type_budget),
        "per_control_budget": int(cfg.per_control_budget),
        "selection_policy": (
            "Deterministic balanced selection from v62 query candidates and novelty material rows using state, support count, "
            "K_mat/K_mask/K_sem flags, and stable non-GT history pairing. Diagnostic labels are not used for query selection."
        ),
        "candidate_types": {
            "heldout_recovery": "K_mat confirmed low-support bridge/update candidates; heldout edge rows created for future validation.",
            "shortcut_quarantine": "shared shortcut/undersegmentation candidates expected to be quarantined if D4RT material evidence supports it.",
            "decoy_rejection": "deterministic same-scene decoy history pairing, built without GT labels.",
            "unknown_defer": "unknown/tentative or no-K evidence rows where safe defer is the desired action.",
        },
        "controls": {
            "C0_v62_original": "original v62 query candidates sampled deterministically",
            "C1_random_matched": "hash-stable scene-matched random control",
            "C2_mask_boundary": "material boundary source control",
            "C3_semantic_only": "K_sem without K_mat/K_mask control",
            "C4_K_mask_only_ablation": "K_mask-positive rows scored later with K_mask-only ablation",
        },
        "input_paths": {
            "v62_query_candidate_rows": _rel(cfg.v62_query_candidate_rows),
            "v62_novelty_material_rows": _rel(cfg.v62_novelty_material_rows),
        },
        "not_used_for_query_selection": ["diagnostic_expected_history_id", "diagnostic_exact_match", "diagnostic_contains_expected"],
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
        "gate": gate,
    }
    return {
        "summary": summary,
        "query_candidate_rows": all_candidate_rows,
        "heldout_edge_rows": heldout_edge_rows,
    }


def write_v63_query_candidates(result: dict[str, Any], config: V63QueryCandidateConfig | None = None) -> dict[str, str]:
    cfg = config or V63QueryCandidateConfig()
    output_root = _project(cfg.output_root)
    visual_root = _project(cfg.visualization_root)
    output_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "query_candidate_summary.json"
    rows_path = output_root / "query_candidate_rows.csv"
    heldout_path = output_root / "heldout_edge_rows.csv"
    write_json(summary_path, result["summary"])
    write_csv(rows_path, result["query_candidate_rows"])
    write_csv(heldout_path, result["heldout_edge_rows"])
    visuals = _write_visualizations(result, visual_root)
    return {
        "query_candidate_summary": _rel(summary_path),
        "query_candidate_rows": _rel(rows_path),
        "heldout_edge_rows": _rel(heldout_path),
        **visuals,
    }


def _method_rows(rows: list[dict[str, Any]], candidate_type: str, reason: str) -> list[dict[str, Any]]:
    return [
        _base_output_row(
            row,
            row_role="method_candidate",
            candidate_type=candidate_type,
            selection_reason=reason,
            control_id="",
            query_history_id=row.get("candidate_history_id", ""),
        )
        for row in rows
    ]


def _decoy_rows(rows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    by_scene: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        history = row.get("candidate_history_id", "")
        if history and history not in by_scene[row.get("scene", "")]:
            by_scene[row.get("scene", "")].append(history)
    out: list[dict[str, Any]] = []
    for row in _stable_sort(rows, "decoy_rejection"):
        histories = by_scene.get(row.get("scene", ""), [])
        decoy = _shifted_history(row.get("candidate_history_id", ""), histories)
        if not decoy:
            continue
        out.append(
            _base_output_row(
                row,
                row_role="method_candidate",
                candidate_type="decoy_rejection",
                selection_reason="same-scene deterministic decoy history pairing for rejection validation",
                control_id="",
                query_history_id=decoy,
                extra={
                    "decoy_source_history_id": row.get("candidate_history_id", ""),
                    "decoy_history_id": decoy,
                    "history_pairing_policy": "same_scene_stable_cyclic_shift",
                },
            )
        )
        if len(out) >= budget:
            break
    return out


def _control_rows(
    enriched: list[dict[str, Any]],
    novelty_enriched: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    budget: int,
) -> list[dict[str, Any]]:
    method_material_ids = {row.get("material_node_id", "") for row in method_rows}
    non_method = [row for row in enriched if row.get("material_node_id", "") not in method_material_ids]
    non_method_novelty = [row for row in novelty_enriched if row.get("material_node_id", "") not in method_material_ids]
    controls: list[dict[str, Any]] = []
    control_specs = [
        ("C0_v62_original", enriched, lambda row: bool(row.get("query_candidate_id")) and _has_support(row), "original v62 query candidate control"),
        ("C1_random_matched", non_method_novelty, lambda row: _has_support(row), "hash-stable non-method random matched control with runnable support observation"),
        ("C2_mask_boundary", non_method, lambda row: parse_bool(row.get("has_material_boundary_source")) and _has_support(row), "mask-boundary/material-boundary source control"),
        ("C3_semantic_only", novelty_enriched, lambda row: parse_bool(row.get("has_K_sem")) and not parse_bool(row.get("has_K_mat")) and not parse_bool(row.get("has_K_mask")) and _has_support(row), "semantic-only K_sem control"),
        ("C4_K_mask_only_ablation", novelty_enriched, lambda row: parse_bool(row.get("has_K_mask")) and _has_support(row), "K_mask-positive row reserved for K_mask-only ablation control"),
    ]
    for control_id, pool, predicate, reason in control_specs:
        selected = _select_rows(pool, budget, predicate, control_id)
        controls.extend(
            _base_output_row(
                row,
                row_role="baseline_control",
                candidate_type="control",
                selection_reason=reason,
                control_id=control_id,
                query_history_id=row.get("candidate_history_id", ""),
            )
            for row in selected
        )
    return controls


def _heldout_edge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    heldout_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        source_frame = _source_frame(row)
        heldout_rows.append(
            {
                "heldout_edge_id": f"v63h_{idx:08d}",
                "v63_candidate_id": row.get("v63_candidate_id", ""),
                "material_node_id": row.get("material_node_id", ""),
                "scene": row.get("scene", ""),
                "component_id": row.get("component_id", ""),
                "source_query_candidate_id": row.get("source_query_candidate_id", ""),
                "candidate_history_id": row.get("candidate_history_id", ""),
                "heldout_history_id": row.get("query_history_id", ""),
                "support_observation_ids_json": row.get("support_observation_ids_json", "[]"),
                "source_frame_id": source_frame,
                "future_window_start_frame": "" if source_frame is None else source_frame + 1,
                "heldout_policy": "remove_candidate_history_edge_for_validation_only",
                "heldout_edge_status": "protocol_spec_row_not_applied_until_phase5_6",
                "uses_gt_for_prediction": False,
            }
        )
    return heldout_rows


def _base_output_row(
    row: dict[str, Any],
    *,
    row_role: str,
    candidate_type: str,
    selection_reason: str,
    control_id: str,
    query_history_id: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "v63_candidate_id": "",
        "row_role": row_role,
        "candidate_type": candidate_type,
        "control_id": control_id,
        "source_query_candidate_id": row.get("query_candidate_id", ""),
        "material_node_id": row.get("material_node_id", ""),
        "scene": row.get("scene", ""),
        "component_id": row.get("component_id", ""),
        "candidate_source": row.get("candidate_source", ""),
        "state": row.get("state", ""),
        "novelty_type": row.get("novelty_type", ""),
        "support_observation_count": parse_int(row.get("support_observation_count")),
        "has_material_boundary_source": parse_bool(row.get("has_material_boundary_source")),
        "has_K_mat": parse_bool(row.get("has_K_mat")),
        "has_K_mask": parse_bool(row.get("has_K_mask")),
        "has_K_sem": parse_bool(row.get("has_K_sem")),
        "candidate_history_id": row.get("candidate_history_id", ""),
        "predicted_history_id": row.get("predicted_history_id", ""),
        "query_history_id": query_history_id,
        "support_observation_ids_json": row.get("support_observation_ids_json", "[]"),
        "selection_score": _selection_score(row, candidate_type or control_id),
        "selection_reason": selection_reason,
        "selection_inputs": "state|support_observation_count|K_flags|candidate_source|candidate_history_id",
        "not_used_for_query_selection": "diagnostic_expected_history_id|diagnostic_exact_match|diagnostic_contains_expected",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
    }
    if extra:
        payload.update(extra)
    return payload


def _renumber(rows: Iterable[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        new_row = dict(row)
        new_row["v63_candidate_id"] = f"{prefix}_{idx:08d}"
        out.append(new_row)
    return out


def _select_rows(rows: list[dict[str, Any]], budget: int, predicate: Any, salt: str) -> list[dict[str, Any]]:
    return _stable_sort([row for row in rows if predicate(row)], salt)[:budget]


def _stable_sort(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (_selection_score(row, salt), _stable_unit(_stable_key(row, salt))), reverse=True)


def _selection_score(row: dict[str, Any], salt: str) -> float:
    support = min(parse_int(row.get("support_observation_count")) / 8.0, 1.0)
    score = 0.20 * support
    score += 0.35 if parse_bool(row.get("has_K_mat")) else 0.0
    score += 0.20 if parse_bool(row.get("has_K_sem")) else 0.0
    score += 0.15 if parse_bool(row.get("has_K_mask")) else 0.0
    if row.get("candidate_source") == "shared_shortcut_boundary" or row.get("state") == "shared":
        score += 0.20
    if row.get("state") in {"unknown", "tentative"}:
        score += 0.25
    if "random" in salt:
        score = _stable_unit(_stable_key(row, salt))
    if "decoy" in salt:
        score += 0.05 * _stable_unit(_stable_key(row, salt))
    return float(score)


def _enrich_query_row(row: dict[str, str], novelty: dict[str, str]) -> dict[str, Any]:
    enriched = dict(row)
    for key in [
        "candidate_history_id",
        "predicted_history_id",
        "support_observation_ids_json",
        "has_K_mat",
        "has_K_mask",
        "has_K_sem",
        "has_K_underseg",
    ]:
        if key not in enriched or enriched.get(key, "") == "":
            enriched[key] = novelty.get(key, "")
    return enriched


def _enrich_novelty_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "query_candidate_id": f"novelty:{row.get('material_node_id', '')}",
        "material_node_id": row.get("material_node_id", ""),
        "scene": row.get("scene", ""),
        "component_id": row.get("component_id", ""),
        "candidate_source": f"novelty_{row.get('novelty_type', '')}",
        "state": row.get("state", ""),
        "novelty_type": row.get("novelty_type", ""),
        "support_observation_count": row.get("support_observation_count", ""),
        "has_material_boundary_source": row.get("novelty_type") in {"bridge_overlap", "shortcut_shared", "update_new"},
        "candidate_history_id": row.get("candidate_history_id", ""),
        "predicted_history_id": row.get("predicted_history_id", ""),
        "support_observation_ids_json": row.get("support_observation_ids_json", "[]"),
        "has_K_mat": row.get("has_K_mat", ""),
        "has_K_mask": row.get("has_K_mask", ""),
        "has_K_sem": row.get("has_K_sem", ""),
        "has_K_underseg": row.get("has_K_underseg", ""),
    }


def _novelty_by_material(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        material = row.get("material_node_id", "")
        if material and material not in out:
            out[material] = row
    return out


def _shifted_history(current: str, histories: list[str]) -> str:
    if not current or len(histories) < 2:
        return ""
    ordered = sorted(histories)
    try:
        idx = ordered.index(current)
    except ValueError:
        idx = int(_stable_unit(current) * len(ordered)) % len(ordered)
    for offset in range(1, len(ordered)):
        candidate = ordered[(idx + offset) % len(ordered)]
        if candidate != current:
            return candidate
    return ""


def _source_frame(row: dict[str, Any]) -> int | None:
    try:
        observations = json.loads(row.get("support_observation_ids_json") or "[]")
    except json.JSONDecodeError:
        observations = []
    frames: list[int] = []
    for token in observations:
        parts = str(token).split(":")
        if len(parts) == 4 and parts[0] == "m":
            try:
                frames.append(int(parts[2]))
            except ValueError:
                pass
    return min(frames) if frames else None


def _has_support(row: dict[str, Any]) -> bool:
    try:
        observations = json.loads(row.get("support_observation_ids_json") or "[]")
    except json.JSONDecodeError:
        return False
    return bool(observations)


def _stable_key(row: dict[str, Any], salt: str) -> str:
    return "|".join(
        [
            salt,
            str(row.get("material_node_id", "")),
            str(row.get("candidate_history_id", "")),
            str(row.get("scene", "")),
            str(row.get("component_id", "")),
        ]
    )


def _stable_unit(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "big")
    return raw / float((1 << 64) - 1)


def _write_visualizations(result: dict[str, Any], root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    summary = result["summary"]
    method_path = root / "candidate_type_counts.png"
    _write_bar_png(method_path, "v63 method candidate types", summary["method_candidate_type_counts"])
    paths["candidate_type_counts"] = _rel(method_path)
    control_path = root / "baseline_control_counts.png"
    _write_bar_png(control_path, "v63 baseline controls", summary["baseline_control_counts"])
    paths["baseline_control_counts"] = _rel(control_path)
    return paths


def _write_bar_png(path: Path, title: str, counts: dict[str, int]) -> None:
    labels = list(counts)
    values = [int(counts[label]) for label in labels]
    width = max(900, 160 * max(1, len(labels)))
    height = 520
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(image, title, (36, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (32, 36, 44), 2, cv2.LINE_AA)
    max_value = max(values) if values else 1
    chart_x, chart_y, chart_h = 80, 120, 280
    bar_w = min(100, max(48, int((width - 180) / max(1, len(labels)) * 0.52)))
    step = int((width - 180) / max(1, len(labels)))
    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = chart_x + idx * step + max(0, (step - bar_w) // 2)
        x1 = x0 + bar_w
        y1 = chart_y + chart_h
        y0 = int(y1 - (value / max_value) * chart_h)
        color = (70 + 23 * idx % 160, 118 + 31 * idx % 100, 210 - 19 * idx % 130)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(image, (x0, y0), (x1, y1), (48, 52, 60), 1)
        cv2.putText(image, str(value), (x0, max(chart_y + 24, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (32, 36, 44), 1, cv2.LINE_AA)
        short = label.replace("_", " ")
        cv2.putText(image, short[:22], (x0 - 25, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 66, 74), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)
