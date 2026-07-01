from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .v47_common import ROOT, parse_bool, parse_float, read_json, safe_mean, utc_now, write_csv, write_json


DEFAULT_GRAPH_ROOT = "outputs/audit/v59_phase1_graph"
DEFAULT_EXPLANATION_ROOT = "outputs/audit/v58_counterfactual_explanation_dino_full_repair6"
DEFAULT_HISTORY_SEMANTIC_ROWS = "outputs/audit/v58_semantic_memory_dino_full_repair2/history_semantic_rows.csv"
DEFAULT_SCAN_ROOT = "data/scannet/scans"


@dataclass(frozen=True)
class V59PathConfig:
    graph_root: str | Path = DEFAULT_GRAPH_ROOT
    explanation_root: str | Path = DEFAULT_EXPLANATION_ROOT
    history_semantic_rows_path: str | Path = DEFAULT_HISTORY_SEMANTIC_ROWS
    scan_root: str | Path = DEFAULT_SCAN_ROOT
    output_root: str | Path = "outputs/audit/v59_phase2_paths"
    visualization_root: str | Path = "outputs/audit/v59_visualizations/phase2"
    primary_variant: str = "E6_counterfactual_semantic_material_underseg"
    semantic_pairwise_baseline_variant: str = "E1_semantic_only"
    require_material_path: bool = True
    reject_exclusion_paths: bool = True
    shortcut_multisignal_min: int = 0
    deferred_shortcuts_remain_tentative: bool = False
    shortcut_min_posterior: float = 0.0
    shortcut_min_margin: float = 0.0
    shortcut_reject_exclusion: bool = False


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _iter_csv(path: str | Path) -> Iterable[dict[str, str]]:
    with _project(path).open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _parse_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def build_v59_manifold_paths(config: V59PathConfig | None = None) -> dict[str, Any]:
    cfg = config or V59PathConfig()
    graph_root = _project(cfg.graph_root)
    graph_summary = read_json(graph_root / "graph_summary.json")
    edge_rows = list(_iter_csv(graph_root / "edge_rows.csv"))
    selected_rows = _load_selected_rows(cfg.explanation_root, cfg.primary_variant)
    baseline_rows = _load_selected_rows(cfg.explanation_root, cfg.semantic_pairwise_baseline_variant)
    category_payload = _load_history_categories(cfg.history_semantic_rows_path, cfg.scan_root)

    semantic_histories_by_mask: dict[str, set[str]] = defaultdict(set)
    material_components_by_mask: dict[str, set[str]] = defaultdict(set)
    material_histories_by_component: dict[str, set[str]] = defaultdict(set)
    underseg_histories_by_mask: dict[str, set[str]] = defaultdict(set)
    exclusion_by_mask: dict[str, int] = defaultdict(int)
    edge_type_counter: Counter[str] = Counter()

    for row in edge_rows:
        edge_type = str(row.get("edge_type") or "")
        edge_type_counter[edge_type] += 1
        src = str(row.get("src_node_id") or "")
        dst = str(row.get("dst_node_id") or "")
        if edge_type == "semantic_compatibility" and src.startswith("m:") and dst.startswith("s:"):
            obs_id = src[2:]
            history_id = _history_from_semantic_node(dst)
            if history_id:
                semantic_histories_by_mask[obs_id].add(history_id)
        elif edge_type == "mask_support" and src.startswith("m:") and dst.startswith("a:"):
            material_components_by_mask[src[2:]].add(dst[2:])
        elif edge_type == "material_continuity" and src.startswith("a:") and dst.startswith("h:"):
            material_histories_by_component[src[2:]].add(dst[2:])
        elif edge_type == "underseg_bridge" and src.startswith("m:") and dst.startswith("h:"):
            underseg_histories_by_mask[src[2:]].add(dst[2:])
        elif edge_type == "exclusion" and src.startswith("m:"):
            exclusion_by_mask[src[2:]] += 1

    path_rows: list[dict[str, Any]] = []
    shortcut_rows: list[dict[str, Any]] = []
    accepted_lengths: list[float] = []
    accepted_path_edge_types: Counter[str] = Counter()
    accepted_count = 0
    accepted_correct = 0
    part_to_core_count = 0
    part_to_core_correct = 0
    diagnostic_positive_count = 0
    false_shortcut_count = 0

    for obs_id, row in selected_rows.items():
        explanation_type = str(row.get("explanation_type") or "")
        decision_state = str(row.get("decision_state") or "")
        expected_type = str(row.get("diagnostic_expected_type") or "")
        expected_histories = set(_parse_list(row.get("diagnostic_expected_history_ids_json")))
        selected_history = str(row.get("history_id") or "")
        candidate_histories = set(_parse_list(row.get("candidate_history_ids_json"))) or {selected_history}
        if expected_type in {"assign_to_existing", "partial_of_existing"} and expected_histories:
            diagnostic_positive_count += 1

        material_histories = set()
        for component_id in material_components_by_mask.get(obs_id, set()):
            material_histories.update(material_histories_by_component.get(component_id, set()))
        has_exclusion = exclusion_by_mask.get(obs_id, 0) > 0
        semantic_histories = semantic_histories_by_mask.get(obs_id, set())
        raw_shortcut_candidate = (
            explanation_type == "underseg_mixture"
            or len(candidate_histories) > 1
            or obs_id in underseg_histories_by_mask
        )
        semantic_multimodal = len(semantic_histories | candidate_histories) > 1
        material_competing = len(material_histories) > 1
        shortcut_signal_count = int(semantic_multimodal) + int(material_competing) + int(has_exclusion)
        deferred_shortcut_suppressed = (
            cfg.deferred_shortcuts_remain_tentative
            and decision_state == "defer_to_active_query"
            and not has_exclusion
        )
        shortcut_score_ok = (
            parse_float(row.get("posterior"), 0.0) >= cfg.shortcut_min_posterior
            and parse_float(row.get("posterior_top1_margin"), 0.0) >= cfg.shortcut_min_margin
        )
        shortcut_exclusion_ok = not (cfg.shortcut_reject_exclusion and has_exclusion)
        is_shortcut_candidate = (
            raw_shortcut_candidate
            and not deferred_shortcut_suppressed
            and shortcut_signal_count >= cfg.shortcut_multisignal_min
            and shortcut_score_ok
            and shortcut_exclusion_ok
        )
        target_history = selected_history
        has_semantic_path = target_history in semantic_histories
        has_material_path = target_history in material_histories
        accepted = (
            bool(target_history)
            and not is_shortcut_candidate
            and has_semantic_path
            and (has_material_path or not cfg.require_material_path)
            and (not has_exclusion or not cfg.reject_exclusion_paths)
        )
        path_length = 3 if has_semantic_path and has_material_path else 2 if has_semantic_path or has_material_path else None
        diagnostic_correct = target_history in expected_histories if expected_histories else parse_bool(row.get("diagnostic_correct"))
        if accepted:
            accepted_count += 1
            accepted_lengths.append(float(path_length or 0.0))
            accepted_path_edge_types.update(["semantic_compatibility", "mask_support", "material_continuity"])
            if diagnostic_correct:
                accepted_correct += 1
            else:
                false_shortcut_count += 1
            if explanation_type in {"assign_to_existing", "partial_of_existing"}:
                part_to_core_count += 1
                if diagnostic_correct:
                    part_to_core_correct += 1

        if is_shortcut_candidate:
            shortcut_histories = sorted(candidate_histories | underseg_histories_by_mask.get(obs_id, set()))
            shortcut_correct = parse_bool(row.get("diagnostic_correct"))
            shortcut_rows.append(
                {
                    "observation_id": obs_id,
                    "scene": row.get("scene"),
                    "frame_id": row.get("frame_id"),
                    "mask_id": row.get("mask_id"),
                    "shortcut_histories_json": json.dumps(shortcut_histories, sort_keys=True),
                    "explanation_type": explanation_type,
                    "quarantine_recommended": True,
                    "shortcut_quarantine_correct_diagnostic": shortcut_correct,
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": parse_bool(row.get("uses_gt_for_diagnostic_labels")),
                }
            )

        path_rows.append(
            {
                "observation_id": obs_id,
                "scene": row.get("scene"),
                "frame_id": row.get("frame_id"),
                "mask_id": row.get("mask_id"),
                "target_history_id": target_history,
                "explanation_type": explanation_type,
                "expected_histories_json": json.dumps(sorted(expected_histories), sort_keys=True),
                "has_semantic_path": has_semantic_path,
                "has_material_path": has_material_path,
                "has_exclusion": has_exclusion,
                "is_shortcut_candidate": is_shortcut_candidate,
                "raw_shortcut_candidate": raw_shortcut_candidate,
                "deferred_shortcut_suppressed": deferred_shortcut_suppressed,
                "semantic_multimodal": semantic_multimodal,
                "material_competing": material_competing,
                "shortcut_signal_count": shortcut_signal_count,
                "shortcut_score_ok": shortcut_score_ok,
                "shortcut_exclusion_ok": shortcut_exclusion_ok,
                "accepted_path": accepted,
                "path_length": path_length,
                "diagnostic_correct": diagnostic_correct,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": parse_bool(row.get("uses_gt_for_diagnostic_labels")),
            }
        )

    baseline = _semantic_pairwise_baseline(baseline_rows)
    path_precision = accepted_correct / accepted_count if accepted_count else None
    part_precision = part_to_core_correct / part_to_core_count if part_to_core_count else None
    recall_proxy = accepted_correct / diagnostic_positive_count if diagnostic_positive_count else None
    shortcut_correct = sum(1 for row in shortcut_rows if row["shortcut_quarantine_correct_diagnostic"])
    shortcut_precision = shortcut_correct / len(shortcut_rows) if shortcut_rows else None
    false_path_rate_proxy = (accepted_count - accepted_correct) / accepted_count if accepted_count else None
    same_category = _same_category_diagnostics(path_rows, baseline_rows, category_payload["history_label_exact"])
    same_category_metric_available = bool(same_category["metric_available"])
    same_category_false_path_rate = same_category["method_same_category_false_path_rate"]
    same_category_gate_pass = bool(same_category["gate_pass"])

    gate = {
        "path_precision_diagnostic_ge_0_80": path_precision is not None and path_precision >= 0.80,
        "part_to_core_path_precision_ge_0_80": part_precision is not None and part_precision >= 0.80,
        "shortcut_quarantine_precision_ge_0_75": shortcut_precision is not None and shortcut_precision >= 0.75,
        "same_category_false_path_rate_metric_available": same_category_metric_available,
        "same_category_false_path_rate_le_semantic_pairwise_baseline_minus_0_05": same_category_gate_pass,
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v59_phase2_manifold_paths",
        "created_at": utc_now(),
        "primary_variant": cfg.primary_variant,
        "graph_summary_path": _rel(graph_root / "graph_summary.json"),
        "selected_observation_count": len(selected_rows),
        "accepted_path_count": accepted_count,
        "path_precision_diagnostic": path_precision,
        "path_recall_proxy": recall_proxy,
        "mean_path_length": safe_mean(accepted_lengths),
        "path_edge_type_distribution": dict(accepted_path_edge_types),
        "false_shortcut_count": false_shortcut_count,
        "shortcut_quarantine_count": len(shortcut_rows),
        "shortcut_quarantine_precision": shortcut_precision,
        "part_to_core_path_precision": part_precision,
        "same_category_false_path_rate": same_category_false_path_rate,
        "same_category_metric_available": same_category_metric_available,
        "same_category_metric_note": (
            "Diagnostic only: history dominant_gt_diagnostic is mapped to ScanNet aggregation labels with exact ids. "
            "Exact/minus-one ambiguity is reported separately and not used for prediction."
        ),
        "same_category_method_pair_count": same_category["method_same_category_pair_count"],
        "same_category_method_false_count": same_category["method_same_category_false_count"],
        "same_category_baseline_pair_count": same_category["baseline_same_category_pair_count"],
        "same_category_baseline_false_count": same_category["baseline_same_category_false_count"],
        "same_category_baseline_false_path_rate": same_category["baseline_same_category_false_path_rate"],
        "same_category_required_max_rate": same_category["required_max_rate"],
        "same_category_history_label_coverage": category_payload["coverage"],
        "same_category_history_label_ambiguity": category_payload["ambiguity"],
        "same_category_false_path_rate_proxy": false_path_rate_proxy,
        "semantic_pairwise_baseline_false_path_rate_proxy": baseline["false_path_rate_proxy"],
        "semantic_pairwise_baseline_selected_count": baseline["selected_count"],
        "semantic_pairwise_baseline_false_count": baseline["false_count"],
        "paths_with_both_semantic_and_material_rate": _both_evidence_rate(path_rows),
        "require_material_path": cfg.require_material_path,
        "reject_exclusion_paths": cfg.reject_exclusion_paths,
        "shortcut_multisignal_min": cfg.shortcut_multisignal_min,
        "deferred_shortcuts_remain_tentative": cfg.deferred_shortcuts_remain_tentative,
        "shortcut_min_posterior": cfg.shortcut_min_posterior,
        "shortcut_min_margin": cfg.shortcut_min_margin,
        "shortcut_reject_exclusion": cfg.shortcut_reject_exclusion,
        "graph_phase1_gate_pass": bool((graph_summary.get("gate") or {}).get("pass")),
        "gate": gate,
        "stop_rule": (
            "Do not run full embedding as method if path precision < 0.80 or if required same-category gate is unavailable/failing."
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "diagnostic_label_sources": [
            "v58_counterfactual_explanation diagnostic_expected_history_ids_json and diagnostic_correct",
            "no diagnostic labels used for path acceptance",
        ],
        "input_paths": {
            "graph_summary": _rel(graph_root / "graph_summary.json"),
            "node_rows": _rel(graph_root / "node_rows.csv"),
            "edge_rows": _rel(graph_root / "edge_rows.csv"),
            "explanation_rows": _rel(Path(cfg.explanation_root) / "explanation_rows.csv"),
            "history_semantic_rows": _rel(cfg.history_semantic_rows_path),
            "scan_root": _rel(cfg.scan_root),
        },
    }
    metric_rows = [
        {"metric": "accepted_path_count", "value": accepted_count},
        {"metric": "path_precision_diagnostic", "value": path_precision},
        {"metric": "path_recall_proxy", "value": recall_proxy},
        {"metric": "mean_path_length", "value": summary["mean_path_length"]},
        {"metric": "false_shortcut_count", "value": false_shortcut_count},
        {"metric": "shortcut_quarantine_count", "value": len(shortcut_rows)},
        {"metric": "shortcut_quarantine_precision", "value": shortcut_precision},
        {"metric": "part_to_core_path_precision", "value": part_precision},
        {"metric": "same_category_false_path_rate", "value": same_category_false_path_rate},
        {"metric": "same_category_baseline_false_path_rate", "value": same_category["baseline_same_category_false_path_rate"]},
        {"metric": "same_category_required_max_rate", "value": same_category["required_max_rate"]},
        {"metric": "same_category_false_path_rate_proxy", "value": false_path_rate_proxy},
        {"metric": "semantic_pairwise_baseline_false_path_rate_proxy", "value": baseline["false_path_rate_proxy"]},
    ]
    return {
        "summary": summary,
        "path_rows": path_rows,
        "shortcut_rows": shortcut_rows,
        "path_metric_rows": metric_rows,
    }


def write_v59_manifold_paths(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "path_summary": root / "path_summary.json",
        "path_rows": root / "path_rows.csv",
        "shortcut_rows": root / "shortcut_rows.csv",
        "path_metric_rows": root / "path_metric_rows.csv",
    }
    write_json(paths["path_summary"], result["summary"])
    write_csv(paths["path_rows"], result["path_rows"])
    write_csv(paths["shortcut_rows"], result["shortcut_rows"])
    write_csv(paths["path_metric_rows"], result["path_metric_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def write_v59_path_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = root / "part_to_whole_path_story_overview.png"
        labels = ["accepted", "shortcut quarantine", "false shortcut"]
        values = [
            summary["accepted_path_count"],
            summary["shortcut_quarantine_count"],
            summary["false_shortcut_count"],
        ]
        fig, ax = plt.subplots(figsize=(7.6, 4.0))
        ax.bar(labels, values, color=["#52796F", "#B56576", "#E76F51"])
        ax.set_title("v59 Phase2 path and shortcut counts")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

        metric_path = root / "false_shortcut_story_overview.png"
        labels = ["path precision", "part precision", "shortcut precision", "proxy false path"]
        values = [
            summary["path_precision_diagnostic"] or 0.0,
            summary["part_to_core_path_precision"] or 0.0,
            summary["shortcut_quarantine_precision"] or 0.0,
            summary["same_category_false_path_rate_proxy"] or 0.0,
        ]
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
        ax.bar(labels, values, color=["#2A9D8F", "#457B9D", "#6D597A", "#E9C46A"])
        ax.axhline(0.80, color="#2A9D8F", linestyle="--", linewidth=1)
        ax.axhline(0.75, color="#6D597A", linestyle=":", linewidth=1)
        ax.set_ylim(0.0, 1.05)
        ax.set_title("v59 Phase2 precision diagnostics")
        ax.tick_params(axis="x", labelrotation=15)
        fig.tight_layout()
        fig.savefig(metric_path, dpi=160)
        plt.close(fig)
        return {
            "path_story": _rel(path),
            "false_shortcut_story": _rel(metric_path),
            "visualization_status": "created",
        }
    except Exception as exc:  # pragma: no cover - optional visualization backend
        error_path = root / "v59_phase2_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _load_selected_rows(explanation_root: str | Path, variant: str) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for row in _iter_csv(Path(explanation_root) / "explanation_rows.csv"):
        if str(row.get("variant") or "") != variant:
            continue
        if not parse_bool(row.get("is_selected")):
            continue
        obs_id = str(row.get("observation_id") or "")
        if obs_id:
            selected[obs_id] = row
    return selected


def _history_from_semantic_node(node_id: str) -> str:
    if not node_id.startswith("s:"):
        return ""
    body = node_id[2:]
    if ":mode" not in body:
        return ""
    return body.rsplit(":mode", 1)[0]


def _semantic_pairwise_baseline(rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    selected_count = 0
    false_count = 0
    for row in rows.values():
        explanation_type = str(row.get("explanation_type") or "")
        if explanation_type not in {"assign_to_existing", "partial_of_existing"}:
            continue
        selected_count += 1
        expected = set(_parse_list(row.get("diagnostic_expected_history_ids_json")))
        history_id = str(row.get("history_id") or "")
        correct = history_id in expected if expected else parse_bool(row.get("diagnostic_correct"))
        if not correct:
            false_count += 1
    return {
        "selected_count": selected_count,
        "false_count": false_count,
        "false_path_rate_proxy": false_count / selected_count if selected_count else None,
    }


def _load_history_categories(history_semantic_rows_path: str | Path, scan_root: str | Path) -> dict[str, Any]:
    history_gt: dict[str, tuple[str, str]] = {}
    for row in _iter_csv(history_semantic_rows_path):
        history_id = str(row.get("history_id") or "")
        scene = str(row.get("scene") or "")
        gt = str(row.get("dominant_gt_diagnostic") or "")
        if history_id and history_id not in history_gt:
            history_gt[history_id] = (scene, gt)

    scene_maps: dict[str, tuple[dict[str, str], dict[str, str]]] = {}

    def maps(scene: str) -> tuple[dict[str, str], dict[str, str]]:
        if scene in scene_maps:
            return scene_maps[scene]
        path = _project(scan_root) / scene / f"{scene}.aggregation.json"
        exact: dict[str, str] = {}
        minus_one: dict[str, str] = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for group in data.get("segGroups", []):
                idx = int(group.get("id", group.get("objectId", -1)))
                label = str(group.get("label", ""))
                exact[str(idx)] = label
                minus_one[str(idx + 1)] = label
        scene_maps[scene] = (exact, minus_one)
        return scene_maps[scene]

    labels_exact: dict[str, str] = {}
    coverage = Counter()
    ambiguity = Counter()
    for history_id, (scene, gt) in history_gt.items():
        coverage["history_count"] += 1
        exact, minus_one = maps(scene)
        exact_label = exact.get(gt)
        minus_label = minus_one.get(gt)
        if exact_label:
            labels_exact[history_id] = exact_label
            coverage["exact_label_count"] += 1
        if minus_label:
            coverage["minus_one_label_count"] += 1
        if exact_label and minus_label and exact_label == minus_label:
            ambiguity["exact_minus_one_agree_count"] += 1
        elif exact_label or minus_label:
            ambiguity["exact_minus_one_disagree_or_single_count"] += 1
        else:
            ambiguity["missing_label_count"] += 1
    return {
        "history_label_exact": labels_exact,
        "coverage": dict(coverage),
        "ambiguity": dict(ambiguity),
    }


def _same_category_diagnostics(
    path_rows: list[dict[str, Any]],
    baseline_rows: dict[str, dict[str, str]],
    history_label_exact: dict[str, str],
) -> dict[str, Any]:
    method_pairs = 0
    method_false = 0
    for row in path_rows:
        if not row.get("accepted_path"):
            continue
        pred = str(row.get("target_history_id") or "")
        expected = _parse_list(row.get("expected_histories_json"))
        if not expected:
            continue
        exp = expected[0]
        pred_label = history_label_exact.get(pred)
        exp_label = history_label_exact.get(exp)
        if not pred_label or not exp_label or pred_label != exp_label or pred == exp:
            continue
        method_pairs += 1
        if not bool(row.get("diagnostic_correct")):
            method_false += 1

    baseline_pairs = 0
    baseline_false = 0
    for row in baseline_rows.values():
        explanation_type = str(row.get("explanation_type") or "")
        if explanation_type not in {"assign_to_existing", "partial_of_existing"}:
            continue
        pred = str(row.get("history_id") or "")
        expected = _parse_list(row.get("diagnostic_expected_history_ids_json"))
        if not expected:
            continue
        exp = expected[0]
        pred_label = history_label_exact.get(pred)
        exp_label = history_label_exact.get(exp)
        if not pred_label or not exp_label or pred_label != exp_label or pred == exp:
            continue
        baseline_pairs += 1
        correct = pred in set(expected)
        if not correct:
            baseline_false += 1

    method_rate = method_false / method_pairs if method_pairs else None
    baseline_rate = baseline_false / baseline_pairs if baseline_pairs else None
    required = baseline_rate - 0.05 if baseline_rate is not None else None
    gate_pass = method_rate is not None and required is not None and method_rate <= required
    return {
        "metric_available": method_rate is not None and baseline_rate is not None,
        "method_same_category_pair_count": method_pairs,
        "method_same_category_false_count": method_false,
        "method_same_category_false_path_rate": method_rate,
        "baseline_same_category_pair_count": baseline_pairs,
        "baseline_same_category_false_count": baseline_false,
        "baseline_same_category_false_path_rate": baseline_rate,
        "required_max_rate": required,
        "gate_pass": gate_pass,
    }


def _both_evidence_rate(path_rows: list[dict[str, Any]]) -> float | None:
    accepted = [row for row in path_rows if row.get("accepted_path")]
    if not accepted:
        return None
    both = sum(1 for row in accepted if row.get("has_semantic_path") and row.get("has_material_path"))
    return both / len(accepted)
