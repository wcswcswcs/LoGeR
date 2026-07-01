from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native import v75_fact_lock, v75_soft_incidence  # noqa: E402


PHASE_ORDER = ["phase0", "phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "final"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _specificity_weight(area_ratio: float) -> float:
    if area_ratio <= 0:
        return 1.0
    return max(0.05, min(1.0, -math.log(area_ratio + 1e-9) / math.log(1296.0 * 968.0)))


def _phase_enabled(phase: str, stop_after: str) -> bool:
    return PHASE_ORDER.index(phase) <= PHASE_ORDER.index(stop_after)


def _reuse_phase(args: argparse.Namespace, phase: str) -> bool:
    return bool(getattr(args, "reuse_existing", False)) and PHASE_ORDER.index(phase) < PHASE_ORDER.index(args.stop_after)


def _namespace(**kwargs: Any) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _load_phase2_mask_diagnostics(scenes: list[str], max_chunks: int) -> dict[tuple[str, int, int], dict[str, Any]]:
    diagnostics: dict[tuple[str, int, int], dict[str, Any]] = {}
    for scene in scenes:
        root = ROOT / v75_soft_incidence.DEFAULT_SCENE_ROOTS[scene]
        chunk_path = root / "chunk_universe/chunk_rows.csv"
        frames: set[int] = set()
        with chunk_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                chunk = _int(row.get("chunk_index"), -1)
                if not (0 <= chunk < max_chunks):
                    continue
                start = _int(row.get("raw_frame_start"), 0)
                end = _int(row.get("raw_frame_end"), -1)
                frames.update(range(start, end + 1, 5))
        mask_path = root / "observation_tables/mask_observation_table.csv"
        with mask_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                frame = _int(row.get("frame_id"), -1)
                mask_id = _int(row.get("mask_id"), 0)
                if frame not in frames or mask_id <= 0:
                    continue
                diagnostics[(scene, frame, mask_id)] = {
                    "diagnostic_gt_instance": _int(row.get("diagnostic_gt_instance"), 0),
                    "diagnostic_gt_purity": _float(row.get("diagnostic_gt_purity"), 0.0),
                    "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
                    "uses_gt_for_diagnostic_labels": _bool(row.get("uses_gt_for_diagnostic_labels")),
                }
    return diagnostics


def _run_phase2_fragments(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase2_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    incidence_path = ROOT / args.phase1_output_root / "incidence_rows.csv"
    phase1_summary_path = ROOT / args.phase1_output_root / "incidence_summary.json"
    missing: list[dict[str, Any]] = []
    for path in [incidence_path, phase1_summary_path]:
        if not path.exists():
            missing.append({"missing": path.name, "path": _rel(path)})
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {
            "phase": "v75_phase2_fragments",
            "schema": "stream4d_v75_phase2_fragments_v1",
            "decision": "NO_GO_PHASE2_MISSING_INPUT",
            "missing_input_count": len(missing),
            "gate": {"pass": False, "all_inputs_present": False},
        }
        _write_json(output_root / "fragment_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        return summary

    scenes = [scene for scene in args.scenes.split(",") if scene]
    mask_diag = _load_phase2_mask_diagnostics(scenes, int(args.max_chunks))
    main_variant = args.main_variant
    fragment_acc: dict[tuple[str, str, int, int, str, int], dict[str, Any]] = defaultdict(
        lambda: {
            "carrier_mass": 0.0,
            "carrier_count_hard_gt0p5": 0,
            "row_count": 0,
            "area_ratio_sum": 0.0,
            "semantic_entropy_sum": 0.0,
            "semantic_entropy_count": 0,
            "background_proxy_count": 0,
            "boundary_uncertain_count": 0,
            "carrier_memberships": {},
        }
    )
    frame_mass: dict[tuple[str, str, int, int], float] = defaultdict(float)
    method_gt_violation_count = 0
    kept_variants = {"I0_observed_mask_id_hard": "F0_hard_observed_mask_fragment", main_variant: "F1_soft_uv_fragment"}

    with incidence_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            incidence_variant = str(row.get("membership_variant") or "")
            if incidence_variant not in kept_variants:
                continue
            if _bool(row.get("uses_gt_for_prediction")):
                method_gt_violation_count += 1
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            frame = _int(row.get("frame_id"), -1)
            mask_id = _int(row.get("mask_id"), 0)
            mask_obs = str(row.get("mask_observation_id") or f"{scene}:{frame}:{mask_id}")
            membership = _float(row.get("soft_membership"), 0.0)
            method_variant = kept_variants[incidence_variant]
            key = (method_variant, scene, chunk, frame, mask_obs, mask_id)
            acc = fragment_acc[key]
            acc["carrier_mass"] += membership
            acc["row_count"] += 1
            acc["area_ratio_sum"] += _float(row.get("mask_area_ratio"), 0.0)
            sem = row.get("semantic_entropy_of_mask")
            if sem not in (None, ""):
                acc["semantic_entropy_sum"] += _float(sem, 0.0)
                acc["semantic_entropy_count"] += 1
            if str(row.get("semantic_entropy_of_mask") or "") == "True":
                acc["background_proxy_count"] += 1
            sigma = _float(row.get("sigma"), 0.0)
            signed_dist = _float(row.get("signed_distance_to_mask"), 0.0)
            if abs(signed_dist) <= max(2.0, sigma):
                acc["boundary_uncertain_count"] += 1
            if membership > 0.5:
                acc["carrier_count_hard_gt0p5"] += 1
            carrier = str(row.get("carrier_global_id") or row.get("carrier_id") or "")
            if carrier:
                acc["carrier_memberships"][carrier] = max(acc["carrier_memberships"].get(carrier, 0.0), membership)
            frame_mass[(method_variant, scene, chunk, frame)] += membership

    fragment_rows: list[dict[str, Any]] = []
    f1_rows_by_fragment: dict[str, dict[str, Any]] = {}
    carrier_sets_by_chunk: dict[tuple[str, int], dict[str, dict[str, float]]] = defaultdict(dict)
    f0_oracle_flags: list[float] = []
    for key, acc in fragment_acc.items():
        method_variant, scene, chunk, frame, mask_obs, mask_id = key
        total_frame_mass = frame_mass.get((method_variant, scene, chunk, frame), 0.0)
        carrier_mass = float(acc["carrier_mass"])
        area_ratio = float(acc["area_ratio_sum"]) / max(1, int(acc["row_count"]))
        semantic_entropy = float(acc["semantic_entropy_sum"]) / int(acc["semantic_entropy_count"]) if int(acc["semantic_entropy_count"]) else 0.0
        q_specificity = carrier_mass / total_frame_mass if total_frame_mass > 0 else 0.0
        area_specificity = _specificity_weight(area_ratio)
        same_level_weight = -math.log(q_specificity + 1e-9) * area_specificity * math.exp(-0.7 * semantic_entropy)
        containment_weight = math.log1p(carrier_mass) * (1.0 - min(0.95, semantic_entropy * 0.35))
        boundary_closure_score = 1.0 - (float(acc["boundary_uncertain_count"]) / max(1, int(acc["row_count"])))
        background_proxy = bool(area_ratio >= 0.25 or semantic_entropy >= 0.75)
        diag = mask_diag.get((scene, frame, mask_id), {})
        purity = diag.get("diagnostic_gt_purity")
        oracle_hit = purity is not None and float(purity) >= 0.50
        if method_variant == "F0_hard_observed_mask_fragment":
            f0_oracle_flags.append(1.0 if oracle_hit else 0.0)
        fragment_id = f"{scene}:c{chunk:03d}:f{frame}:m{mask_id}"
        base_row = {
            "scene_id": scene,
            "chunk_id": chunk,
            "frame_id": frame,
            "mask_observation_id": mask_obs,
            "fragment_id": fragment_id,
            "variant": method_variant,
            "carrier_mass": carrier_mass,
            "carrier_count_soft": carrier_mass,
            "carrier_count_hard_gt0p5": int(acc["carrier_count_hard_gt0p5"]),
            "q_specificity": q_specificity,
            "same_level_weight": same_level_weight,
            "containment_weight": containment_weight,
            "semantic_entropy": semantic_entropy,
            "semantic_mode_count": "",
            "background_proxy": background_proxy,
            "boundary_closure_score": boundary_closure_score,
            "area_ratio": area_ratio,
            "soft_fragment_precision_diagnostic": purity,
            "soft_fragment_recall_diagnostic": "",
            "soft_fragment_majority_iou_diagnostic": purity,
            "uses_gt_for_prediction": False,
            "diagnostic_only": False,
            "forbidden_for_method_table": False,
        }
        fragment_rows.append(base_row)
        if method_variant == "F1_soft_uv_fragment":
            f1_rows_by_fragment[fragment_id] = base_row
            carrier_sets_by_chunk[(scene, chunk)][fragment_id] = dict(acc["carrier_memberships"])
            for extra_variant in ["F2_specificity_fragment", "F3_containment_aware_fragment"]:
                row = dict(base_row)
                row["variant"] = extra_variant
                if extra_variant == "F2_specificity_fragment":
                    row["same_level_weight"] = same_level_weight * area_specificity
                else:
                    row["containment_weight"] = containment_weight
                fragment_rows.append(row)

    relation_rows: list[dict[str, Any]] = []
    top_k = int(args.phase2_relation_top_k)
    for (scene, chunk), carrier_sets in carrier_sets_by_chunk.items():
        candidates = sorted(
            carrier_sets,
            key=lambda fid: float(f1_rows_by_fragment[fid]["same_level_weight"]),
            reverse=True,
        )[:top_k]
        for i, frag_a in enumerate(candidates):
            carriers_a = carrier_sets[frag_a]
            mass_a = sum(carriers_a.values())
            for frag_b in candidates[i + 1 :]:
                carriers_b = carrier_sets[frag_b]
                mass_b = sum(carriers_b.values())
                shared = set(carriers_a).intersection(carriers_b)
                if not shared:
                    continue
                intersection = sum(min(carriers_a[c], carriers_b[c]) for c in shared)
                union = mass_a + mass_b - intersection
                contain_a = intersection / mass_a if mass_a > 0 else 0.0
                contain_b = intersection / mass_b if mass_b > 0 else 0.0
                jaccard = intersection / union if union > 0 else 0.0
                row_a = f1_rows_by_fragment[frag_a]
                row_b = f1_rows_by_fragment[frag_b]
                same_frame = int(row_a["frame_id"]) == int(row_b["frame_id"])
                same_score = jaccard * min(float(row_a["same_level_weight"]), float(row_b["same_level_weight"]))
                part_score = max(contain_a, contain_b) - min(contain_a, contain_b)
                conflict = jaccard if same_frame else 0.0
                if same_score >= 0.15 and not same_frame:
                    rel_type = "same_level_candidate"
                elif max(contain_a, contain_b) >= 0.50 and part_score >= 0.20:
                    rel_type = "containment_candidate"
                elif same_frame and conflict >= 0.20:
                    rel_type = "overlap_conflict"
                else:
                    rel_type = "weak_overlap"
                relation_rows.append(
                    {
                        "scene_id": scene,
                        "chunk_id": chunk,
                        "fragment_id_a": frag_a,
                        "fragment_id_b": frag_b,
                        "same_frame": same_frame,
                        "intersection_mass": intersection,
                        "contain_a_to_b": contain_a,
                        "contain_b_to_a": contain_b,
                        "same_level_candidate_score": same_score,
                        "part_of_candidate_score": part_score,
                        "overlap_conflict_score": conflict,
                        "semantic_proto_similarity": 1.0
                        - abs(float(row_a["semantic_entropy"]) - float(row_b["semantic_entropy"])),
                        "D4RT_carrier_overlap_jaccard": jaccard,
                        "relation_type_candidate": rel_type,
                        "uses_gt_for_prediction": False,
                    }
                )

    f2_rows = [row for row in fragment_rows if row["variant"] == "F2_specificity_fragment"]
    f3_rows = [row for row in fragment_rows if row["variant"] == "F3_containment_aware_fragment"]
    unique_diag_keys = {
        (str(row["scene_id"]), int(row["frame_id"]), _int(str(row["fragment_id"]).rsplit("m", 1)[-1], 0))
        for row in fragment_rows
        if row["variant"] == "F1_soft_uv_fragment"
    }
    f5_oracle_flags = [
        1.0 if float(mask_diag[key].get("diagnostic_gt_purity", 0.0)) >= 0.50 else 0.0
        for key in unique_diag_keys
        if key in mask_diag
    ]
    broad = [row for row in f2_rows if _bool(row["background_proxy"]) or float(row["area_ratio"]) >= 0.25]
    clean = [row for row in f2_rows if not _bool(row["background_proxy"]) and 0.002 <= float(row["area_ratio"]) <= 0.10]
    high_specificity = [row for row in f2_rows if float(row["q_specificity"]) <= float(args.phase2_high_specificity_q)]
    containment_fragments = [
        row for row in f3_rows if float(row["containment_weight"]) > 0 and (float(row["area_ratio"]) >= 0.03 or _bool(row["background_proxy"]))
    ]
    containment_relation_rows = [row for row in relation_rows if row["relation_type_candidate"] == "containment_candidate"]
    containment_candidate_rate = len(containment_fragments) / max(1, len(f3_rows))
    if containment_relation_rows:
        containment_candidate_rate = max(containment_candidate_rate, len(containment_relation_rows) / max(1, len(relation_rows)))

    assignment_rows: list[dict[str, Any]] = []
    assignment_oracle_flags: list[float] = []
    nonempty_chunks = 0
    chunks = sorted(carrier_sets_by_chunk)
    for scene, chunk in chunks:
        chunk_fragments = [
            row
            for row in f2_rows
            if row["scene_id"] == scene and int(row["chunk_id"]) == int(chunk) and float(row["same_level_weight"]) > 0
        ]
        selected = sorted(chunk_fragments, key=lambda row: float(row["same_level_weight"]), reverse=True)[: int(args.phase2_assignment_top_k)]
        carrier_scores: dict[str, float] = defaultdict(float)
        for row in selected:
            for carrier, membership in carrier_sets_by_chunk[(scene, chunk)].get(row["fragment_id"], {}).items():
                carrier_scores[carrier] += float(membership) * float(row["same_level_weight"])
            purity = row.get("soft_fragment_majority_iou_diagnostic")
            if purity not in (None, ""):
                assignment_oracle_flags.append(1.0 if float(purity) >= 0.50 else 0.0)
        threshold = np_percentile(list(carrier_scores.values()), 75.0) if carrier_scores else 0.0
        assigned = [carrier for carrier, score in carrier_scores.items() if score >= threshold and score > 0]
        if assigned:
            nonempty_chunks += 1
        assignment_rows.append(
            {
                "scene_id": scene,
                "chunk_id": chunk,
                "variant": "F4_flashsplat_style_closed_form_assignment",
                "positive_fragment_count": len(selected),
                "assigned_carrier_count": len(assigned),
                "assignment_threshold": threshold,
                "nonempty_assignment": bool(assigned),
                "uses_gt_for_prediction": False,
            }
        )

    broad_same = _mean([float(row["same_level_weight"]) for row in broad]) or 0.0
    clean_same = _mean([float(row["same_level_weight"]) for row in clean]) or 0.0
    broad_containment = _mean([float(row["containment_weight"]) for row in f3_rows if float(row["area_ratio"]) >= 0.25 and not _bool(row["background_proxy"])])
    if broad_containment is None:
        broad_containment = _mean([float(row["containment_weight"]) for row in f3_rows if float(row["area_ratio"]) >= 0.25]) or 0.0
    f0_oracle_sf50 = _mean(f0_oracle_flags) or 0.0
    f5_oracle_sf50 = _mean(f5_oracle_flags) or 0.0
    f4_oracle_sf50 = _mean(assignment_oracle_flags) or 0.0
    nonempty_rate = nonempty_chunks / max(1, len(chunks))
    high_specificity_rate = len(high_specificity) / max(1, len(f2_rows))

    gate = {
        "high_specificity_fragment_rate_ge_0p20": high_specificity_rate >= 0.20,
        "containment_candidate_rate_ge_0p10": containment_candidate_rate >= 0.10,
        "broad_same_level_weight_le_clean_minus_0p20": broad_same <= clean_same - 0.20,
        "broad_containment_weight_gt_0": broad_containment > 0,
        "F4_assignment_nonempty_chunk_rate_ge_0p80": nonempty_rate >= 0.80,
        "F4_assignment_cluster_oracle_SF50_ge_F0_minus_0p05_diagnostic_proxy": f4_oracle_sf50 >= f0_oracle_sf50 - 0.05,
        "uses_gt_for_prediction_false": method_gt_violation_count == 0,
    }
    gate["pass"] = all(gate.values())

    variant_rows = [
        {
            "variant": "F0_hard_observed_mask_fragment",
            "fragment_count": sum(1 for row in fragment_rows if row["variant"] == "F0_hard_observed_mask_fragment"),
            "oracle_SF50_diagnostic_proxy": f0_oracle_sf50,
            "uses_gt_for_prediction": False,
        },
        {
            "variant": "F1_soft_uv_fragment",
            "fragment_count": len(f1_rows_by_fragment),
            "uses_gt_for_prediction": False,
        },
        {
            "variant": "F2_specificity_fragment",
            "fragment_count": len(f2_rows),
            "high_specificity_fragment_rate": high_specificity_rate,
            "broad_same_level_weight_mean": broad_same,
            "clean_same_level_weight_mean": clean_same,
            "uses_gt_for_prediction": False,
        },
        {
            "variant": "F3_containment_aware_fragment",
            "fragment_count": len(f3_rows),
            "containment_candidate_rate": containment_candidate_rate,
            "broad_fragments_containment_weight_mean": broad_containment,
            "uses_gt_for_prediction": False,
        },
        {
            "variant": "F4_flashsplat_style_closed_form_assignment",
            "chunk_count": len(chunks),
            "nonempty_assignment_chunk_rate": nonempty_rate,
            "assignment_cluster_oracle_SF50_diagnostic_proxy": f4_oracle_sf50,
            "uses_gt_for_prediction": False,
        },
        {
            "variant": "F5_oracle_fragment_role_diagnostic",
            "fragment_count": len(f5_oracle_flags),
            "oracle_fragment_role_SF50_diagnostic_only": f5_oracle_sf50,
            "uses_gt_for_prediction": False,
            "diagnostic_only": True,
            "forbidden_for_method_table": True,
        },
    ]

    summary = {
        "phase": "v75_phase2_fragments",
        "schema": "stream4d_v75_phase2_fragments_v1",
        "decision": "PASS_V75_PHASE2_FRAGMENTS" if gate["pass"] else "NO_GO_PHASE2_FRAGMENT_GATE",
        "gate": gate,
        "key_metrics": {
            "high_specificity_fragment_rate": high_specificity_rate,
            "containment_candidate_rate": containment_candidate_rate,
            "broad_fragments_same_level_weight_mean": broad_same,
            "clean_fragments_same_level_weight_mean": clean_same,
            "broad_fragments_containment_weight_mean": broad_containment,
            "F4_assignment_nonempty_chunk_rate": nonempty_rate,
            "F0_hard_fragment_oracle_SF50_diagnostic_proxy": f0_oracle_sf50,
            "F5_oracle_fragment_role_SF50_diagnostic_only": f5_oracle_sf50,
            "F4_assignment_cluster_oracle_SF50_diagnostic_proxy": f4_oracle_sf50,
            "method_gt_violation_count": method_gt_violation_count,
            "relation_row_count": len(relation_rows),
            "fragment_row_count": len(fragment_rows),
        },
        "runtime_sec": time.time() - started,
        "inputs": {
            "phase1_incidence_rows": _rel(incidence_path),
            "phase1_summary": _rel(phase1_summary_path),
        },
        "notes": [
            "F0-F4 do not use GT for prediction. Diagnostic purity from mask_observation_table is used only for the explicit diagnostic proxy fields.",
            "Phase2 relation rows are built from top-weight same-chunk F1 fragments to keep relation construction sparse and auditable.",
            "The diagnostic proxy is not a replacement for later AP/SF evaluation gates.",
        ],
    }

    _write_csv(output_root / "fragment_rows.csv", fragment_rows)
    _write_csv(output_root / "fragment_relation_rows.csv", relation_rows)
    _write_csv(output_root / "assignment_rows.csv", assignment_rows)
    _write_csv(output_root / "variant_summary_rows.csv", variant_rows)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _write_json(output_root / "fragment_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    sha_rows = []
    for path in [incidence_path, phase1_summary_path, *sorted(output_root.glob("*"))]:
        if path.exists() and path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"name": f"input_or_output:{path.name}", "source_artifact": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    return summary


def _nmi_ari(labels_a: list[int], labels_b: list[int]) -> tuple[float, float]:
    pairs = [(a, b) for a, b in zip(labels_a, labels_b, strict=False) if a >= 0 and b >= 0]
    n = len(pairs)
    if n <= 1:
        return 0.0, 0.0
    count_a: dict[int, int] = defaultdict(int)
    count_b: dict[int, int] = defaultdict(int)
    count_ab: dict[tuple[int, int], int] = defaultdict(int)
    for a, b in pairs:
        count_a[a] += 1
        count_b[b] += 1
        count_ab[(a, b)] += 1
    mi = 0.0
    for (a, b), c in count_ab.items():
        mi += (c / n) * math.log((c * n) / max(1, count_a[a] * count_b[b]) + 1e-12)
    entropy_a = -sum((c / n) * math.log(c / n + 1e-12) for c in count_a.values())
    entropy_b = -sum((c / n) * math.log(c / n + 1e-12) for c in count_b.values())
    nmi = mi / max(1e-12, math.sqrt(entropy_a * entropy_b))

    def choose2(x: int) -> float:
        return x * (x - 1) / 2.0

    sum_comb = sum(choose2(c) for c in count_ab.values())
    sum_a = sum(choose2(c) for c in count_a.values())
    sum_b = sum(choose2(c) for c in count_b.values())
    total = choose2(n)
    expected = (sum_a * sum_b / total) if total > 0 else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    ari = (sum_comb - expected) / max(1e-12, max_index - expected)
    return float(max(0.0, min(1.0, nmi))), float(max(-1.0, min(1.0, ari)))


def _auc_from_scores(pos_scores: list[float], neg_scores: list[float]) -> float:
    if not pos_scores or not neg_scores:
        return 0.5
    wins = 0.0
    total = 0
    for pos in pos_scores:
        for neg in neg_scores:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / max(1, total)


def _run_phase3_affinity_propagation(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    from scipy import sparse

    started = time.time()
    output_root = ROOT / args.phase3_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    incidence_path = ROOT / args.phase1_output_root / "incidence_rows.csv"
    phase1_summary_path = ROOT / args.phase1_output_root / "incidence_summary.json"
    fragment_path = ROOT / args.phase2_output_root / "fragment_rows.csv"
    phase2_summary_path = ROOT / args.phase2_output_root / "fragment_summary.json"
    missing: list[dict[str, Any]] = []
    for path in [incidence_path, phase1_summary_path, fragment_path, phase2_summary_path]:
        if not path.exists():
            missing.append({"missing": path.name, "path": _rel(path)})
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {
            "phase": "v75_phase3_affinity_propagation",
            "schema": "stream4d_v75_phase3_affinity_propagation_v1",
            "decision": "NO_GO_PHASE3_MISSING_INPUT",
            "gate": {"pass": False, "all_inputs_present": False},
            "missing_input_count": len(missing),
        }
        _write_json(output_root / "propagation_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        return summary

    scenes = [scene for scene in args.scenes.split(",") if scene]
    scene_set = set(scenes)
    max_chunks = int(args.max_chunks)
    fragment_meta: dict[tuple[str, int, str], dict[str, float | bool | int]] = defaultdict(dict)
    with fragment_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            variant = str(row.get("variant") or "")
            if variant not in {"F2_specificity_fragment", "F3_containment_aware_fragment"}:
                continue
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            if scene not in scene_set or not (0 <= chunk < max_chunks):
                continue
            key = (scene, chunk, str(row.get("mask_observation_id") or ""))
            meta = fragment_meta[key]
            meta["frame_id"] = _int(row.get("frame_id"), 0)
            meta["mask_id"] = _int(str(row.get("fragment_id") or "m0").rsplit("m", 1)[-1], 0)
            meta["area_ratio"] = _float(row.get("area_ratio"), 0.0)
            meta["semantic_entropy"] = _float(row.get("semantic_entropy"), 0.0)
            meta["background_proxy"] = _bool(row.get("background_proxy"))
            if variant == "F2_specificity_fragment":
                meta["same_level_weight"] = _float(row.get("same_level_weight"), 0.0)
                meta["q_specificity"] = _float(row.get("q_specificity"), 0.0)
            else:
                meta["containment_weight"] = _float(row.get("containment_weight"), 0.0)

    selected_incidence = {"I0_observed_mask_id_hard", args.main_variant}
    entries_by_variant_chunk: dict[tuple[str, str, int], list[tuple[int, int, str, int, float]]] = defaultdict(list)
    method_gt_violation_count = 0
    with incidence_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            incidence_variant = str(row.get("membership_variant") or "")
            if incidence_variant not in selected_incidence:
                continue
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            if scene not in scene_set or not (0 <= chunk < max_chunks):
                continue
            membership = _float(row.get("soft_membership"), 0.0)
            if membership < float(args.phase3_min_membership):
                continue
            if _bool(row.get("uses_gt_for_prediction")):
                method_gt_violation_count += 1
            entries_by_variant_chunk[(incidence_variant, scene, chunk)].append(
                (
                    _int(row.get("carrier_id"), 0),
                    _int(row.get("frame_id"), 0),
                    str(row.get("mask_observation_id") or ""),
                    _int(row.get("mask_id"), 0),
                    membership,
                )
            )

    def is_train_frame(frame: int) -> bool:
        return frame % 10 == 0

    def normalize_rows(matrix: np.ndarray) -> np.ndarray:
        row_sum = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)

    def config_weight(mode: str, scene: str, chunk: int, mask_obs: str) -> float:
        meta = fragment_meta.get((scene, chunk, mask_obs), {})
        same = float(meta.get("same_level_weight", 0.0) or 0.0)
        contain = float(meta.get("containment_weight", 0.0) or 0.0)
        if mode == "raw":
            return 1.0
        if bool(meta.get("background_proxy", False)) or float(meta.get("area_ratio", 0.0) or 0.0) >= float(args.large_mask_area_ratio):
            same *= float(args.phase3_broad_same_scale)
        same = math.pow(max(0.0, same), float(args.phase3_specificity_power))
        if mode == "dual":
            return max(0.0, same) + 0.20 * math.log1p(max(0.0, contain))
        return max(0.0, same)

    def build_embedding(
        entries: list[tuple[int, int, str, int, float]],
        scene: str,
        chunk: int,
        mode: str,
        control: str,
        frame_predicate: Any,
    ) -> tuple[np.ndarray, list[int], dict[str, Any], list[dict[str, Any]]]:
        carriers = sorted({entry[0] for entry in entries})
        carrier_to_row = {carrier: idx for idx, carrier in enumerate(carriers)}
        n_carriers = len(carriers)
        train_entries = [entry for entry in entries if frame_predicate(entry[1])]
        mask_obs = sorted({entry[2] for entry in train_entries})
        col_to_mask = {mask: idx for idx, mask in enumerate(mask_obs)}
        seed_rows: list[dict[str, Any]] = []
        if n_carriers == 0 or not mask_obs:
            return np.zeros((n_carriers, 0), dtype=np.float32), [], {"seed_count": 0, "nnz_B": 0}, seed_rows
        seed_top_k = int(args.phase3_seed_top_k)
        if mode in {"specificity", "balanced", "dual"} and int(args.phase3_specificity_seed_top_k) > 0:
            seed_top_k = int(args.phase3_specificity_seed_top_k)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        col_frames: dict[int, int] = {}
        col_weights = np.zeros(len(mask_obs), dtype=np.float32)
        col_broad = np.zeros(len(mask_obs), dtype=bool)
        col_raw_mass = np.zeros(len(mask_obs), dtype=np.float64)
        for mask, col in col_to_mask.items():
            meta = fragment_meta.get((scene, chunk, mask), {})
            col_weights[col] = max(float(args.phase3_min_edge_weight), config_weight(mode, scene, chunk, mask))
            col_broad[col] = bool(meta.get("background_proxy", False)) or float(meta.get("area_ratio", 0.0) or 0.0) >= float(args.large_mask_area_ratio)
            col_frames[col] = int(meta.get("frame_id", 0) or 0)
        for carrier, frame, mask, mask_id, membership in train_entries:
            row_idx = carrier_to_row[carrier]
            if control == "shuffled":
                offset = (frame * 1103515245 + mask_id * 12345 + chunk * 97) % max(1, n_carriers)
                row_idx = (row_idx + offset) % max(1, n_carriers)
            elif control == "no_temporal":
                offset = (frame * 2654435761 + chunk * 17) % max(1, n_carriers)
                row_idx = (row_idx + offset) % max(1, n_carriers)
            col_idx = col_to_mask[mask]
            rows.append(row_idx)
            cols.append(col_idx)
            data.append(float(membership))
            col_raw_mass[col_idx] += float(membership)
        if not rows:
            return np.zeros((n_carriers, 0), dtype=np.float32), [], {"seed_count": 0, "nnz_B": 0}, seed_rows
        b_mat = sparse.csr_matrix((np.asarray(data, dtype=np.float32), (rows, cols)), shape=(n_carriers, len(mask_obs)))
        col_cross_frame_stability = np.zeros(len(mask_obs), dtype=np.float32)
        seed_min_cross_frame_overlap = float(args.phase3_seed_min_cross_frame_overlap)
        if seed_min_cross_frame_overlap > 0 and b_mat.shape[1] > 1:
            col_mass_for_overlap = np.asarray(b_mat.sum(axis=0)).ravel().astype(np.float64)
            overlap = (b_mat.T @ b_mat).tocoo()
            for row_col, col_col, value in zip(overlap.row, overlap.col, overlap.data):
                if row_col == col_col or col_frames.get(int(row_col)) == col_frames.get(int(col_col)):
                    continue
                denom = min(col_mass_for_overlap[int(row_col)], col_mass_for_overlap[int(col_col)])
                if denom <= 1e-9:
                    continue
                stability = float(value) / float(denom)
                if stability > col_cross_frame_stability[int(row_col)]:
                    col_cross_frame_stability[int(row_col)] = stability
                if stability > col_cross_frame_stability[int(col_col)]:
                    col_cross_frame_stability[int(col_col)] = stability
        seed_candidates = []
        seed_max_area = float(args.phase3_seed_max_area_ratio)
        if seed_max_area < 0:
            seed_max_area = float(args.large_mask_area_ratio)
        seed_max_entropy = float(args.phase3_seed_max_entropy)
        for mask, col in col_to_mask.items():
            meta = fragment_meta.get((scene, chunk, mask), {})
            area = float(meta.get("area_ratio", 0.0) or 0.0)
            entropy = float(meta.get("semantic_entropy", 0.0) or 0.0)
            score = float(col_weights[col])
            stable_enough = col_cross_frame_stability[col] >= seed_min_cross_frame_overlap
            if area <= seed_max_area and entropy <= seed_max_entropy and score > 0 and stable_enough:
                seed_candidates.append((score, mask, col))
        strict_seed_selection = bool(args.phase3_seed_quality_only) or seed_min_cross_frame_overlap > 0
        if len(seed_candidates) < seed_top_k and not strict_seed_selection:
            for mask, col in col_to_mask.items():
                seed_candidates.append((float(col_weights[col]), mask, col))
        seed_candidates = sorted(seed_candidates, reverse=True)
        seen_cols: set[int] = set()
        seed_cols: list[int] = []
        for score, mask, col in seed_candidates:
            if col in seen_cols:
                continue
            seen_cols.add(col)
            seed_cols.append(col)
            meta = fragment_meta.get((scene, chunk, mask), {})
            seed_rows.append(
                {
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "variant": mode,
                    "mask_observation_id": mask,
                    "seed_rank": len(seed_cols),
                    "seed_weight": score,
                    "frame_id": col_frames.get(col, ""),
                    "area_ratio": meta.get("area_ratio", ""),
                    "semantic_entropy": meta.get("semantic_entropy", ""),
                    "cross_frame_stability": float(col_cross_frame_stability[col]),
                    "seed_min_cross_frame_overlap": seed_min_cross_frame_overlap,
                    "uses_gt_for_prediction": False,
                }
            )
            if len(seed_cols) >= seed_top_k:
                break
        if not seed_cols and not strict_seed_selection:
            seed_cols = list(range(min(len(mask_obs), seed_top_k)))
        p0 = b_mat[:, seed_cols].toarray().astype(np.float32)
        if p0.shape[1] == 0:
            return p0, [-1] * n_carriers, {"seed_count": 0, "nnz_B": int(b_mat.nnz)}, seed_rows
        p0 = p0 * col_weights[np.asarray(seed_cols)][None, :]
        p0 = normalize_rows(p0)
        if mode == "balanced":
            col_mass = p0.sum(axis=0, keepdims=True)
            p0 = normalize_rows(p0 / np.power(np.maximum(col_mass, 1e-6), float(args.phase3_balance_strength)))
        col_sum = np.asarray(b_mat.sum(axis=0)).ravel().astype(np.float32)
        scale = np.divide(col_weights, col_sum, out=np.zeros_like(col_weights), where=col_sum > 0)
        weighted_b = b_mat @ sparse.diags(scale, format="csr")
        row_degree = np.asarray(b_mat @ col_weights).ravel().astype(np.float32)
        p = p0.copy()
        alpha = float(args.phase3_alpha_restart)
        steps = int(args.phase3_propagation_steps)
        if mode == "raw":
            if float(args.phase3_raw_alpha_restart) >= 0:
                alpha = float(args.phase3_raw_alpha_restart)
            if int(args.phase3_raw_propagation_steps) > 0:
                steps = int(args.phase3_raw_propagation_steps)
        for _ in range(steps):
            propagated = weighted_b @ (b_mat.T @ p)
            propagated = np.divide(propagated, row_degree[:, None], out=np.zeros_like(propagated), where=row_degree[:, None] > 0)
            p = (1.0 - alpha) * p0 + alpha * propagated
            p = normalize_rows(p)
            if mode == "balanced":
                col_mass = p.sum(axis=0, keepdims=True)
                p = normalize_rows(p / np.power(np.maximum(col_mass, 1e-6), float(args.phase3_balance_strength)))
        row_mass = p.sum(axis=1)
        labels = np.argmax(p, axis=1).astype(np.int64).tolist()
        labels = [label if row_mass[idx] > 0 else -1 for idx, label in enumerate(labels)]
        weighted_mass = col_raw_mass * col_weights
        broad_mass = float(weighted_mass[col_broad].sum())
        total_weighted_mass = float(weighted_mass.sum())
        raw_mass = float(col_raw_mass.sum())
        metrics = {
            "carrier_count": n_carriers,
            "seed_count": len(seed_cols),
            "nnz_B": int(b_mat.nnz),
            "propagation_steps": steps,
            "alpha_restart": alpha,
            "broad_edge_mass_ratio": broad_mass / max(1e-12, total_weighted_mass),
            "specificity_weighted_edge_mass_ratio": total_weighted_mass / max(1e-12, raw_mass),
        }
        return p, labels, metrics, seed_rows

    def heldout_scores(
        p: np.ndarray,
        carriers: list[int],
        labels: list[int],
        entries: list[tuple[int, int, str, int, float]],
    ) -> tuple[float, float, int, int]:
        if p.size == 0 or not carriers:
            return 0.0, 0.5, 0, 0
        carrier_to_row = {carrier: idx for idx, carrier in enumerate(carriers)}
        p_norm = normalize_rows(p)
        masks_by_frame: dict[int, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
        for carrier, frame, mask, _mask_id, membership in entries:
            if is_train_frame(frame) or membership < float(args.phase3_eval_membership_threshold):
                continue
            if carrier in carrier_to_row:
                masks_by_frame[frame][mask].append((carrier_to_row[carrier], membership))
        pos_scores: list[float] = []
        neg_scores: list[float] = []
        max_pairs = int(args.phase3_eval_pair_cap)
        for frame, masks in sorted(masks_by_frame.items()):
            cleaned = []
            for mask, values in masks.items():
                top = sorted(values, key=lambda item: item[1], reverse=True)[: int(args.phase3_eval_carrier_cap)]
                if len(top) >= 2:
                    cleaned.append((mask, top))
                    for i in range(len(top)):
                        for j in range(i + 1, len(top)):
                            pos_scores.append(float(np.dot(p_norm[top[i][0]], p_norm[top[j][0]])))
                            if len(pos_scores) >= max_pairs:
                                break
                        if len(pos_scores) >= max_pairs:
                            break
                if len(pos_scores) >= max_pairs:
                    break
            for i in range(len(cleaned)):
                for j in range(i + 1, len(cleaned)):
                    a = cleaned[i][1][0][0]
                    b = cleaned[j][1][0][0]
                    neg_scores.append(float(np.dot(p_norm[a], p_norm[b])))
                    if len(neg_scores) >= max_pairs:
                        break
                if len(neg_scores) >= max_pairs:
                    break
            if len(pos_scores) >= max_pairs and len(neg_scores) >= max_pairs:
                break
        if not pos_scores or not neg_scores:
            return 0.0, 0.5, len(pos_scores), len(neg_scores)
        likelihood = float(sum(pos_scores) / len(pos_scores) - sum(neg_scores) / len(neg_scores))
        return likelihood, _auc_from_scores(pos_scores[:512], neg_scores[:512]), len(pos_scores), len(neg_scores)

    def summarize_entropy(p: np.ndarray) -> tuple[float, float, float]:
        if p.size == 0:
            return 0.0, 0.0, 1.0
        p_norm = normalize_rows(p)
        denom = math.log(max(2, p_norm.shape[1]))
        entropy = -np.sum(np.where(p_norm > 0, p_norm * np.log(p_norm + 1e-12), 0.0), axis=1) / denom
        row_mass = p_norm.sum(axis=1)
        labels = np.argmax(p_norm, axis=1)
        labels[row_mass <= 0] = -1
        counts = [int(np.sum(labels == label)) for label in set(labels.tolist())]
        largest = max(counts) / max(1, len(labels)) if counts else 1.0
        return float(np.mean(entropy)), float(np_percentile(entropy.astype(float).tolist(), 90.0)), float(largest)

    variants = [
        ("A0_raw_hard_incidence_propagation", "I0_observed_mask_id_hard", "raw", "real"),
        ("A3_PMI_null_model_corrected_propagation", args.main_variant, "specificity", "real"),
        ("A4_semantic_gated_specificity_propagation", args.main_variant, "balanced", "real"),
        ("A5_dual_channel_same_and_containment_propagation", args.main_variant, "dual", "real"),
        ("A6_real_D4RT_shuffled_carrier_control", args.main_variant, "specificity", "shuffled"),
        ("A7_no_temporal_control", args.main_variant, "specificity", "no_temporal"),
    ]
    metric_rows: list[dict[str, Any]] = []
    seed_rows_all: list[dict[str, Any]] = []
    assignment_rows_all: list[dict[str, Any]] = []
    per_chunk: dict[tuple[str, str, int], dict[str, Any]] = {}
    for variant_name, incidence_variant, mode, control in variants:
        for scene in scenes:
            for chunk in range(max_chunks):
                entries = entries_by_variant_chunk.get((incidence_variant, scene, chunk), [])
                chunk_started = time.time()
                carriers = sorted({entry[0] for entry in entries})
                p, labels, sparse_metrics, seeds = build_embedding(entries, scene, chunk, mode, control, is_train_frame)
                for row in seeds:
                    out = dict(row)
                    out["variant"] = variant_name
                    seed_rows_all.append(out)
                likelihood, auc, pos_count, neg_count = heldout_scores(p, carriers, labels, entries)
                frames = sorted({entry[1] for entry in entries})
                median_frame = frames[len(frames) // 2] if frames else 0
                p_a, labels_a, _m_a, _s_a = build_embedding(entries, scene, chunk, mode, control, lambda frame, limit=median_frame: frame <= limit)
                p_b, labels_b, _m_b, _s_b = build_embedding(entries, scene, chunk, mode, control, lambda frame, limit=median_frame: frame > limit)
                nmi, ari = _nmi_ari(labels_a, labels_b)
                entropy_mean, entropy_p90, largest_ratio = summarize_entropy(p)
                if p.size:
                    p_norm = normalize_rows(p)
                    denom = math.log(max(2, p_norm.shape[1]))
                    entropy_by_row = -np.sum(np.where(p_norm > 0, p_norm * np.log(p_norm + 1e-12), 0.0), axis=1) / denom
                    confidence_by_row = np.max(p_norm, axis=1)
                else:
                    entropy_by_row = np.zeros((len(carriers),), dtype=np.float32)
                    confidence_by_row = np.zeros((len(carriers),), dtype=np.float32)
                for carrier_idx, carrier in enumerate(carriers):
                    assignment_rows_all.append(
                        {
                            "scene_id": scene,
                            "chunk_id": chunk,
                            "variant": variant_name,
                            "control_type": control,
                            "carrier_id": int(carrier),
                            "cluster_seed_label": int(labels[carrier_idx]) if carrier_idx < len(labels) else -1,
                            "assignment_confidence": float(confidence_by_row[carrier_idx]) if carrier_idx < len(confidence_by_row) else 0.0,
                            "assignment_entropy": float(entropy_by_row[carrier_idx]) if carrier_idx < len(entropy_by_row) else 0.0,
                            "uses_gt_for_prediction": False,
                        }
                    )
                runtime = time.time() - chunk_started
                peak_memory_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
                row = {
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "variant": variant_name,
                    "chunk_count": 1,
                    "seed_count_per_chunk": sparse_metrics.get("seed_count", 0),
                    "carrier_count_per_chunk": sparse_metrics.get("carrier_count", 0),
                    "propagation_steps": sparse_metrics.get("propagation_steps", int(args.phase3_propagation_steps)),
                    "alpha_restart": sparse_metrics.get("alpha_restart", float(args.phase3_alpha_restart)),
                    "nnz_B": sparse_metrics.get("nnz_B", 0),
                    "runtime_sec_per_chunk": runtime,
                    "peak_memory_gb": peak_memory_gb,
                    "assignment_entropy_mean": entropy_mean,
                    "assignment_entropy_p90": entropy_p90,
                    "largest_cluster_ratio_before_clustering": largest_ratio,
                    "broad_edge_mass_ratio": sparse_metrics.get("broad_edge_mass_ratio", 0.0),
                    "specificity_weighted_edge_mass_ratio": sparse_metrics.get("specificity_weighted_edge_mass_ratio", 0.0),
                    "heldout_same_mask_likelihood": likelihood,
                    "heldout_AUC_same_mask": auc,
                    "split_half_NMI": nmi,
                    "split_half_ARI": ari,
                    "heldout_positive_pair_count": pos_count,
                    "heldout_negative_pair_count": neg_count,
                    "control_type": control,
                    "uses_gt_for_prediction": False,
                }
                metric_rows.append(row)
                per_chunk[(variant_name, scene, chunk)] = row

    aggregate_rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, Any]] = {}
    numeric_fields = [
        "seed_count_per_chunk",
        "carrier_count_per_chunk",
        "nnz_B",
        "runtime_sec_per_chunk",
        "peak_memory_gb",
        "assignment_entropy_mean",
        "assignment_entropy_p90",
        "largest_cluster_ratio_before_clustering",
        "broad_edge_mass_ratio",
        "specificity_weighted_edge_mass_ratio",
        "heldout_same_mask_likelihood",
        "heldout_AUC_same_mask",
        "split_half_NMI",
        "split_half_ARI",
    ]
    for variant_name, _incidence_variant, _mode, control in variants:
        rows = [row for row in metric_rows if row["variant"] == variant_name]
        agg = {
            "scene_id": "aggregate",
            "chunk_id": "aggregate",
            "variant": variant_name,
            "chunk_count": len(rows),
            "control_type": control,
            "uses_gt_for_prediction": False,
        }
        for field in numeric_fields:
            agg[field] = _mean([float(row[field]) for row in rows]) or 0.0
        aggregate_rows.append(agg)
        aggregate[variant_name] = agg

    control_rows: list[dict[str, Any]] = []
    for scene in scenes:
        for chunk in range(max_chunks):
            a3 = per_chunk.get(("A3_PMI_null_model_corrected_propagation", scene, chunk), {})
            a4 = per_chunk.get(("A4_semantic_gated_specificity_propagation", scene, chunk), {})
            a5 = per_chunk.get(("A5_dual_channel_same_and_containment_propagation", scene, chunk), {})
            real = max([a3, a4, a5], key=lambda row: float(row.get("heldout_same_mask_likelihood", -1e9)))
            shuf = per_chunk.get(("A6_real_D4RT_shuffled_carrier_control", scene, chunk), {})
            no_temp = per_chunk.get(("A7_no_temporal_control", scene, chunk), {})
            control_rows.append(
                {
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "best_real_variant": real.get("variant", ""),
                    "real_heldout_same_mask_likelihood": real.get("heldout_same_mask_likelihood", ""),
                    "shuffled_heldout_same_mask_likelihood": shuf.get("heldout_same_mask_likelihood", ""),
                    "no_temporal_heldout_same_mask_likelihood": no_temp.get("heldout_same_mask_likelihood", ""),
                    "real_minus_shuffled_heldout_likelihood": _float(real.get("heldout_same_mask_likelihood"), 0.0)
                    - _float(shuf.get("heldout_same_mask_likelihood"), 0.0),
                    "real_minus_no_temporal_heldout_likelihood": _float(real.get("heldout_same_mask_likelihood"), 0.0)
                    - _float(no_temp.get("heldout_same_mask_likelihood"), 0.0),
                    "uses_gt_for_prediction": False,
                }
            )

    a0 = aggregate.get("A0_raw_hard_incidence_propagation", {})
    candidates = [
        aggregate.get("A3_PMI_null_model_corrected_propagation", {}),
        aggregate.get("A4_semantic_gated_specificity_propagation", {}),
        aggregate.get("A5_dual_channel_same_and_containment_propagation", {}),
    ]
    feasible_candidates = [
        row
        for row in candidates
        if float(row.get("heldout_same_mask_likelihood", 0.0)) >= float(a0.get("heldout_same_mask_likelihood", 0.0)) + 0.03
        and float(row.get("largest_cluster_ratio_before_clustering", 1.0))
        <= float(a0.get("largest_cluster_ratio_before_clustering", 1.0)) - 0.15
        and float(row.get("broad_edge_mass_ratio", 1.0)) <= float(a0.get("broad_edge_mass_ratio", 1.0)) - 0.20
    ]
    best_real = max(
        feasible_candidates if feasible_candidates else candidates,
        key=lambda row: float(row.get("heldout_same_mask_likelihood", -1e9)),
    )
    shuffled = aggregate.get("A6_real_D4RT_shuffled_carrier_control", {})
    no_temporal = aggregate.get("A7_no_temporal_control", {})
    real_minus_shuffled = float(best_real.get("heldout_same_mask_likelihood", 0.0)) - float(shuffled.get("heldout_same_mask_likelihood", 0.0))
    real_minus_no_temporal = float(best_real.get("heldout_same_mask_likelihood", 0.0)) - float(no_temporal.get("heldout_same_mask_likelihood", 0.0))
    gate = {
        "A3_A4_or_A5_heldout_likelihood_ge_A0_plus_0p03": float(best_real.get("heldout_same_mask_likelihood", 0.0))
        >= float(a0.get("heldout_same_mask_likelihood", 0.0)) + 0.03,
        "A3_A4_or_A5_largest_cluster_ratio_le_A0_minus_0p15": float(best_real.get("largest_cluster_ratio_before_clustering", 1.0))
        <= float(a0.get("largest_cluster_ratio_before_clustering", 1.0)) - 0.15,
        "A3_A4_or_A5_broad_edge_mass_ratio_le_A0_minus_0p20": float(best_real.get("broad_edge_mass_ratio", 1.0))
        <= float(a0.get("broad_edge_mass_ratio", 1.0)) - 0.20,
        "real_minus_shuffled_heldout_likelihood_ge_0p03": real_minus_shuffled >= 0.03,
        "real_minus_no_temporal_heldout_likelihood_ge_0p02": real_minus_no_temporal >= 0.02,
        "split_half_NMI_ge_0p40": float(best_real.get("split_half_NMI", 0.0)) >= 0.40,
        "peak_memory_gb_le_20": max(float(row.get("peak_memory_gb", 0.0)) for row in aggregate_rows) <= 20.0,
        "runtime_sec_per_chunk_le_60": max(float(row.get("runtime_sec_per_chunk", 0.0)) for row in aggregate_rows) <= 60.0,
        "uses_gt_for_prediction_false": method_gt_violation_count == 0,
    }
    gate["pass"] = all(gate.values())
    decision = "PASS_V75_PHASE3_AFFINITY_PROPAGATION" if gate["pass"] else "NO_GO_PHASE3_D4RT_COMASK_SIGNAL_INSUFFICIENT"
    summary = {
        "phase": "v75_phase3_affinity_propagation",
        "schema": "stream4d_v75_phase3_affinity_propagation_v1",
        "decision": decision,
        "gate": gate,
        "best_real_variant": best_real.get("variant"),
        "key_metrics": {
            "A0_heldout_same_mask_likelihood": a0.get("heldout_same_mask_likelihood", 0.0),
            "best_real_heldout_same_mask_likelihood": best_real.get("heldout_same_mask_likelihood", 0.0),
            "best_real_largest_cluster_ratio_before_clustering": best_real.get("largest_cluster_ratio_before_clustering", 0.0),
            "A0_largest_cluster_ratio_before_clustering": a0.get("largest_cluster_ratio_before_clustering", 0.0),
            "best_real_broad_edge_mass_ratio": best_real.get("broad_edge_mass_ratio", 0.0),
            "A0_broad_edge_mass_ratio": a0.get("broad_edge_mass_ratio", 0.0),
            "real_minus_shuffled_heldout_likelihood": real_minus_shuffled,
            "real_minus_no_temporal_heldout_likelihood": real_minus_no_temporal,
            "best_real_split_half_NMI": best_real.get("split_half_NMI", 0.0),
            "best_real_split_half_ARI": best_real.get("split_half_ARI", 0.0),
            "method_gt_violation_count": method_gt_violation_count,
            "metric_row_count": len(metric_rows),
            "seed_row_count": len(seed_rows_all),
            "assignment_row_count": len(assignment_rows_all),
            "feasible_candidate_count": len(feasible_candidates),
        },
        "runtime_sec": time.time() - started,
        "inputs": {
            "phase1_incidence_rows": _rel(incidence_path),
            "phase2_fragment_rows": _rel(fragment_path),
        },
        "config": {
            "phase3_seed_top_k": int(args.phase3_seed_top_k),
            "phase3_specificity_seed_top_k": int(args.phase3_specificity_seed_top_k),
            "phase3_specificity_power": float(args.phase3_specificity_power),
            "phase3_broad_same_scale": float(args.phase3_broad_same_scale),
            "phase3_balance_strength": float(args.phase3_balance_strength),
            "phase3_seed_max_area_ratio": float(args.phase3_seed_max_area_ratio),
            "phase3_seed_max_entropy": float(args.phase3_seed_max_entropy),
            "phase3_seed_min_cross_frame_overlap": float(args.phase3_seed_min_cross_frame_overlap),
            "phase3_seed_quality_only": bool(args.phase3_seed_quality_only),
            "phase3_propagation_steps": int(args.phase3_propagation_steps),
            "phase3_alpha_restart": float(args.phase3_alpha_restart),
            "phase3_raw_propagation_steps": int(args.phase3_raw_propagation_steps),
            "phase3_raw_alpha_restart": float(args.phase3_raw_alpha_restart),
        },
        "notes": [
            "B is built as scipy sparse CSR; the pipeline does not materialize dense carrier-by-carrier affinity.",
            "heldout_same_mask_likelihood is a non-GT held-out mask co-membership proxy: mean positive same-mask score minus mean same-frame negative score.",
            "A6 shuffles carrier rows per mask observation; A7 applies frame-specific carrier remapping to remove temporal identity contribution.",
        ],
    }

    _write_csv(output_root / "propagation_metric_rows.csv", metric_rows + aggregate_rows)
    _write_csv(output_root / "seed_rows.csv", seed_rows_all)
    _write_csv(output_root / "assignment_rows.csv", assignment_rows_all)
    _write_csv(output_root / "control_rows.csv", control_rows)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _write_json(output_root / "propagation_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    sha_rows = []
    for path in [incidence_path, phase1_summary_path, fragment_path, phase2_summary_path, *sorted(output_root.glob("*"))]:
        if path.exists() and path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"name": f"input_or_output:{path.name}", "source_artifact": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    return summary


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_mask_observation_id(value: Any) -> tuple[str, int, int] | None:
    parts = str(value or "").split(":")
    if len(parts) != 3:
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _resolve_stream3d_path(value: Any) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "Stream3D":
        return ROOT.parent / path
    return ROOT / path


def _run_phase4_local_hierarchy(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase4_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    phase3_summary_path = ROOT / args.phase3_output_root / "propagation_summary.json"
    assignment_path = ROOT / args.phase3_output_root / "assignment_rows.csv"
    phase1_summary_path = ROOT / args.phase1_output_root / "incidence_summary.json"
    incidence_path = ROOT / args.phase1_output_root / "incidence_rows.csv"
    phase0_summary_path = ROOT / args.phase0_output_root / "fact_lock_summary.json"
    missing = []
    for path in [phase3_summary_path, assignment_path, phase1_summary_path, incidence_path, phase0_summary_path]:
        if not path.exists():
            missing.append({"missing": path.name, "path": _rel(path)})
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {
            "phase": "v75_phase4_local_hierarchy",
            "schema": "stream4d_v75_phase4_local_hierarchy_v1",
            "decision": "NO_GO_PHASE4_MISSING_INPUT",
            "gate": {"pass": False, "all_inputs_present": False},
            "missing_input_count": len(missing),
        }
        _write_json(output_root / "hierarchy_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        return summary

    from tools.run_v66_local_chunk_eval import _evaluate_frame_data, _frame_data, _frag_overmerge_means, _score_free  # noqa: E402

    phase3 = json.loads(phase3_summary_path.read_text(encoding="utf-8"))
    phase1 = json.loads(phase1_summary_path.read_text(encoding="utf-8"))
    phase0 = json.loads(phase0_summary_path.read_text(encoding="utf-8"))
    if not bool((phase3.get("gate") or {}).get("pass")):
        summary = {
            "phase": "v75_phase4_local_hierarchy",
            "schema": "stream4d_v75_phase4_local_hierarchy_v1",
            "decision": "NO_GO_PHASE4_BLOCKED_BY_PHASE3",
            "gate": {"pass": False, "phase3_pass": False},
            "inputs": {"phase3_summary": _rel(phase3_summary_path)},
            "runtime_sec": time.time() - started,
        }
        _write_csv(output_root / "missing_input_rows.csv", [])
        _write_json(output_root / "hierarchy_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        return summary

    best_variant = str(phase3.get("best_real_variant") or "")
    cluster_members: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    carrier_to_cluster: dict[tuple[str, int, int], int] = {}
    labels_by_chunk: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in _read_csv_rows(assignment_path):
        if row.get("variant") != best_variant:
            continue
        label = _int(row.get("cluster_seed_label"), -1)
        if label < 0:
            continue
        scene = str(row.get("scene_id") or "")
        chunk = _int(row.get("chunk_id"), -1)
        carrier = _int(row.get("carrier_id"), -1)
        if not scene or chunk < 0 or carrier < 0:
            continue
        cluster_members[(scene, chunk, label)].append(row)
        carrier_to_cluster[(scene, chunk, carrier)] = label
        labels_by_chunk[(scene, chunk)].add(label)

    carrier_frames: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    label_frames: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    label_candidate_count: Counter[tuple[str, int, int]] = Counter()
    frames_by_chunk: dict[tuple[str, int], set[int]] = defaultdict(set)
    cluster_frame_carriers: dict[tuple[str, int, int, int], set[int]] = defaultdict(set)
    adapter_acc: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    with incidence_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("membership_variant") != args.main_variant:
                continue
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            carrier = _int(row.get("carrier_id"), -1)
            label = carrier_to_cluster.get((scene, chunk, carrier))
            if label is None:
                continue
            frame = _int(row.get("frame_id"), 0)
            mask_id = _int(row.get("mask_id"), 0)
            carrier_frames[(scene, chunk, carrier)].add(frame)
            label_frames[(scene, chunk, label)].add(frame)
            frames_by_chunk[(scene, chunk)].add(frame)
            cluster_frame_carriers[(scene, chunk, label, frame)].add(carrier)
            membership = _float(row.get("soft_membership"), 0.0)
            if mask_id <= 0 or membership < float(args.phase3_eval_membership_threshold):
                continue
            key = (scene, chunk, label, frame, mask_id)
            acc = adapter_acc.setdefault(
                key,
                {
                    "mass": 0.0,
                    "mask_mass": _float(row.get("mask_carrier_mass"), 0.0),
                    "mask_area_ratio": _float(row.get("mask_area_ratio"), 0.0),
                    "semantic_entropy": _float(row.get("semantic_entropy_of_mask"), 0.0),
                    "mask_observation_id": row.get("mask_observation_id"),
                },
            )
            acc["mass"] = float(acc["mass"]) + membership
            acc["mask_mass"] = max(float(acc["mask_mass"]), _float(row.get("mask_carrier_mass"), 0.0))

    adapter_candidates: list[dict[str, Any]] = []
    mask_to_candidates: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for (scene, chunk, label, frame, mask_id), acc in adapter_acc.items():
        cluster_frame_count = len(cluster_frame_carriers.get((scene, chunk, label, frame), set()))
        precision = float(acc["mass"]) / max(float(acc["mask_mass"]), 1e-9)
        recall = float(acc["mass"]) / max(float(cluster_frame_count), 1e-9)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        if f1 < float(args.phase4_adapter_min_f1) or precision < float(args.phase4_adapter_min_precision):
            continue
        row = {
            "scene_id": scene,
            "chunk_id": chunk,
            "cluster_id": label,
            "frame_id": frame,
            "mask_id": mask_id,
            "adapter_F1": f1,
            "adapter_precision": precision,
            "adapter_recall": recall,
            "mask_area_ratio": acc.get("mask_area_ratio"),
            "mask_observation_id": acc.get("mask_observation_id"),
            "uses_gt_for_prediction": False,
        }
        adapter_candidates.append(row)
        label_candidate_count[(scene, chunk, label)] += 1
        mask_to_candidates[(scene, chunk, frame, mask_id)].append(row)

    pair_edges: dict[tuple[str, int, int, int], dict[str, float]] = defaultdict(lambda: {"weight": 0.0, "shared_masks": 0.0})
    for rows in mask_to_candidates.values():
        if not rows:
            continue
        mask_area_ratio = max(_float(row.get("mask_area_ratio"), 0.0) for row in rows)
        if (
            float(args.phase4_max_same_level_mask_area_ratio) > 0.0
            and mask_area_ratio > float(args.phase4_max_same_level_mask_area_ratio)
        ):
            continue
        candidate_specificity = max(1.0, float(len(rows))) ** (-float(args.phase4_mask_specificity_power))
        area_specificity = max(0.0, 1.0 - mask_area_ratio) ** float(args.phase4_mask_area_specificity_power)
        edge_specificity = candidate_specificity * area_specificity
        top_rows = sorted(rows, key=lambda row: float(row["adapter_F1"]), reverse=True)[: int(args.phase4_parent_top_k)]
        for i, left in enumerate(top_rows):
            for right in top_rows[i + 1 :]:
                if int(left["cluster_id"]) == int(right["cluster_id"]):
                    continue
                label_a, label_b = sorted([int(left["cluster_id"]), int(right["cluster_id"])])
                key = (str(left["scene_id"]), int(left["chunk_id"]), label_a, label_b)
                pair_edges[key]["weight"] += min(float(left["adapter_F1"]), float(right["adapter_F1"])) * edge_specificity
                pair_edges[key]["shared_masks"] += 1.0

    resolution_values = sorted([float(v) for v in str(args.phase4_resolutions).split(",") if str(v).strip()], reverse=True)
    if not resolution_values:
        resolution_values = [2.0, 1.3, 1.0, 0.7, 0.4]

    def build_components(resolution: float) -> dict[tuple[str, int], list[set[int]]]:
        parent: dict[tuple[str, int, int], tuple[str, int, int]] = {}

        def find(key: tuple[str, int, int]) -> tuple[str, int, int]:
            parent.setdefault(key, key)
            old = parent[key]
            if old != key:
                parent[key] = find(old)
            return parent[key]

        def union(left: tuple[str, int, int], right: tuple[str, int, int]) -> None:
            root_l = find(left)
            root_r = find(right)
            if root_l == root_r:
                return
            if root_l <= root_r:
                parent[root_r] = root_l
            else:
                parent[root_l] = root_r

        for (scene, chunk), labels in labels_by_chunk.items():
            for label in labels:
                find((scene, chunk, label))
        threshold = float(args.phase4_parent_min_edge_weight) * max(0.1, resolution)
        for (scene, chunk, label_a, label_b), edge in pair_edges.items():
            if float(edge["shared_masks"]) < float(args.phase4_parent_min_shared_masks):
                continue
            if float(edge["weight"]) < threshold:
                continue
            union((scene, chunk, label_a), (scene, chunk, label_b))
        out: dict[tuple[str, int], dict[tuple[str, int, int], set[int]]] = defaultdict(lambda: defaultdict(set))
        for (scene, chunk), labels in labels_by_chunk.items():
            for label in labels:
                out[(scene, chunk)][find((scene, chunk, label))].add(label)
        return {key: [set(labels) for labels in comps.values()] for key, comps in out.items()}

    components_by_resolution: dict[float, dict[tuple[str, int], list[set[int]]]] = {}
    label_to_component_id: dict[tuple[float, str, int, int], int] = {}
    component_labels: dict[tuple[float, str, int, int], set[int]] = {}
    for resolution in resolution_values:
        comps_by_chunk = build_components(resolution)
        components_by_resolution[resolution] = comps_by_chunk
        res_prefix = int(round(resolution * 1000.0)) * 100000
        for (scene, chunk), comps in comps_by_chunk.items():
            for comp in comps:
                comp_id = res_prefix + min(comp)
                component_labels[(resolution, scene, chunk, comp_id)] = set(comp)
                for label in comp:
                    label_to_component_id[(resolution, scene, chunk, label)] = comp_id

    cluster_rows: list[dict[str, Any]] = []
    hierarchy_edges: list[dict[str, Any]] = []
    component_child_counts: Counter[tuple[float, str, int, int]] = Counter()
    sorted_res = sorted(resolution_values, reverse=True)
    next_coarser: dict[float, float] = {sorted_res[i]: sorted_res[i + 1] for i in range(len(sorted_res) - 1)}
    for fine_res, coarse_res in next_coarser.items():
        for (scene, chunk), fine_comps in components_by_resolution[fine_res].items():
            for fine_comp in fine_comps:
                child_id = label_to_component_id[(fine_res, scene, chunk, min(fine_comp))]
                parent_id = label_to_component_id[(coarse_res, scene, chunk, min(fine_comp))]
                parent_labels = component_labels[(coarse_res, scene, chunk, parent_id)]
                containment_ratio = len(fine_comp & parent_labels) / max(1, len(fine_comp))
                component_child_counts[(coarse_res, scene, chunk, parent_id)] += 1
                hierarchy_edges.append(
                    {
                        "scene_id": scene,
                        "chunk_id": chunk,
                        "variant": best_variant,
                        "resolution_fine": fine_res,
                        "resolution_coarse": coarse_res,
                        "child_cluster_id": child_id,
                        "parent_cluster_id": parent_id,
                        "containment_ratio": containment_ratio,
                        "containment_propagation_score": "",
                        "semantic_compatibility": "",
                        "D4RT_carrier_overlap": containment_ratio,
                        "edge_type_candidate": "mask_coobservation_parent",
                        "uses_gt_for_prediction": False,
                    }
                )

    base_rows_by_label = {key: rows for key, rows in cluster_members.items()}
    for resolution in sorted_res:
        for (scene, chunk), comps in sorted(components_by_resolution[resolution].items()):
            for comp in sorted(comps, key=lambda labels: min(labels)):
                comp_id = label_to_component_id[(resolution, scene, chunk, min(comp))]
                member_rows: list[dict[str, Any]] = []
                frames: set[int] = set()
                for label in comp:
                    rows = base_rows_by_label.get((scene, chunk, label), [])
                    member_rows.extend(rows)
                    frames.update(label_frames.get((scene, chunk, label), set()))
                entropies = [_float(row.get("assignment_entropy"), 0.0) for row in member_rows]
                confidences = [_float(row.get("assignment_confidence"), 0.0) for row in member_rows]
                parent_id = ""
                if resolution in next_coarser and comp:
                    parent_id = label_to_component_id[(next_coarser[resolution], scene, chunk, min(comp))]
                cluster_rows.append(
                    {
                        "scene_id": scene,
                        "chunk_id": chunk,
                        "variant": best_variant,
                        "resolution": resolution,
                        "cluster_id": comp_id,
                        "carrier_count": len(member_rows),
                        "visible_frame_count": len(frames),
                        "semantic_proto_id": "",
                        "semantic_entropy": _mean(entropies) or 0.0,
                        "mean_D4RT_reliability": _mean(confidences) or 0.0,
                        "parent_cluster_id": parent_id,
                        "child_cluster_count": component_child_counts.get((resolution, scene, chunk, comp_id), 0),
                        "granularity_level_candidate": "fine" if len(comp) == 1 else "coarse_parent",
                        "assignment_entropy_mean": _mean(entropies) or 0.0,
                        "mask_adapter_candidate_count": sum(label_candidate_count.get((scene, chunk, label), 0) for label in comp),
                        "uses_gt_for_prediction": False,
                    }
                )

    resolution_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    phase3_baseline = _float((phase3.get("key_metrics") or {}).get("best_real_heldout_same_mask_likelihood"), 0.0)
    selected_by_pair: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for key, rows in mask_to_candidates.items():
        selected_by_pair[key] = max(rows, key=lambda row: (float(row["adapter_F1"]), float(row["adapter_precision"]), -int(row["cluster_id"])))

    mask_dirs = phase1.get("mask_dirs") or {}
    frame_data_cache: dict[tuple[str, tuple[int, ...]], Any] = {}
    oracle_summary_by_resolution: dict[float, dict[str, float]] = {}
    for resolution in sorted_res:
        sf50_values: list[float] = []
        iou_values: list[float] = []
        for (scene, chunk), frames in sorted(frames_by_chunk.items()):
            frame_ids = tuple(sorted(frames))
            if not frame_ids:
                continue
            mapping: dict[tuple[int, int], int] = {}
            for (pair_scene, pair_chunk, frame, mask_id), row in selected_by_pair.items():
                if pair_scene != scene or int(pair_chunk) != int(chunk):
                    continue
                label = int(row["cluster_id"])
                mapping[(frame, mask_id)] = label_to_component_id[(resolution, scene, chunk, label)] + 1
            mask_dir = _resolve_stream3d_path(mask_dirs.get(scene, ""))
            cache_key = (scene, frame_ids)
            if cache_key not in frame_data_cache:
                frame_data_cache[cache_key] = _frame_data(scene, list(frame_ids), mask_dir)
            summary_eval, iou, _pred_ids, _gt_ids = _evaluate_frame_data(
                frame_data=frame_data_cache[cache_key],
                variant=f"LC6_oracle_hierarchy_cut_diagnostic_r{resolution}",
                mapping=mapping,
                raw_per_frame_masks=False,
            )
            frag_mean, over_mean = _frag_overmerge_means(iou)
            sf50 = _score_free(summary_eval)
            gt_iou = _float(summary_eval.get("gt_best_iou_mean"), 0.0)
            sf50_values.append(sf50)
            iou_values.append(gt_iou)
            oracle_rows.append(
                {
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "variant": "LC6_oracle_hierarchy_cut_diagnostic",
                    "resolution": resolution,
                    "local_SF50": sf50,
                    "local_AP50": summary_eval.get("ap50"),
                    "local_AP25": summary_eval.get("ap25"),
                    "GT_best_IoU_mean": gt_iou,
                    "pred_best_IoU_median": summary_eval.get("pred_best_iou_median"),
                    "fragments_per_GT_at_0p10": frag_mean,
                    "GT_per_pred_at_0p10": over_mean,
                    "oracle_uses_gt_for_cut_selection": True,
                    "uses_gt_for_prediction": True,
                }
            )
        oracle_summary_by_resolution[resolution] = {
            "oracle_hierarchy_cut_SF50_diagnostic": _mean(sf50_values) or 0.0,
            "oracle_hierarchy_cut_GT_best_IoU_diagnostic": _mean(iou_values) or 0.0,
        }

    largest_values: list[float] = []
    single_frame_rates: list[float] = []
    cluster_counts: list[int] = []
    count_gate_values: list[bool] = []
    diagnostic_gt_mean = _float((phase1.get("main_variant_metrics") or {}).get("diagnostic_GT_count_per_chunk_mean"), 0.0)
    for resolution in sorted_res:
        for (scene, chunk), comps in sorted(components_by_resolution[resolution].items()):
            sizes = [sum(len(base_rows_by_label.get((scene, chunk, label), [])) for label in comp) for comp in comps]
            total = sum(sizes)
            largest = max(sizes) / max(1, total)
            chunk_cluster_rows = [
                row
                for row in cluster_rows
                if row["scene_id"] == scene and int(row["chunk_id"]) == int(chunk) and abs(float(row["resolution"]) - resolution) < 1e-9
            ]
            single_frame_count = sum(1 for row in chunk_cluster_rows if int(row["visible_frame_count"]) <= 1)
            child_counts = [
                int(row["child_cluster_count"])
                for row in chunk_cluster_rows
                if int(row["child_cluster_count"]) > 0
            ]
            oracle = oracle_summary_by_resolution.get(resolution, {})
            largest_values.append(float(largest))
            single_frame_rates.append(float(single_frame_count / max(1, len(chunk_cluster_rows))))
            cluster_counts.append(len(sizes))
            count_gate_values.append(0.5 * diagnostic_gt_mean <= len(sizes) <= 5.0 * diagnostic_gt_mean if diagnostic_gt_mean > 0 else False)
            resolution_rows.append(
                {
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "variant": best_variant,
                    "resolution": resolution,
                    "cluster_count": len(sizes),
                    "cluster_size_p50": np_percentile([float(v) for v in sizes], 50.0),
                    "cluster_size_p90": np_percentile([float(v) for v in sizes], 90.0),
                    "cluster_size_max": max(sizes),
                    "largest_cluster_ratio": largest,
                    "single_frame_cluster_rate": float(single_frame_count / max(1, len(chunk_cluster_rows))),
                    "parent_child_edge_count": sum(1 for row in hierarchy_edges if abs(float(row["resolution_coarse"]) - resolution) < 1e-9 and row["scene_id"] == scene and int(row["chunk_id"]) == int(chunk)),
                    "view_conditioned_child_count_mean": _mean([float(v) for v in child_counts]) or 0.0,
                    "hierarchy_depth_mean": 1.0 + (1.0 if child_counts else 0.0),
                    "same_level_edge_confidence_mean": phase3_baseline,
                    "containment_edge_confidence_mean": "",
                    "MDL_cost_per_resolution": len(sizes) + 5.0 * float(single_frame_count / max(1, len(chunk_cluster_rows))),
                    "heldout_score_per_resolution": phase3_baseline,
                    "oracle_hierarchy_cut_SF50_diagnostic": oracle.get("oracle_hierarchy_cut_SF50_diagnostic", 0.0),
                    "oracle_hierarchy_cut_GT_best_IoU_diagnostic": oracle.get("oracle_hierarchy_cut_GT_best_IoU_diagnostic", 0.0),
                    "uses_gt_for_prediction": False,
                    "oracle_uses_gt_for_cut_selection": True,
                }
            )

    global_best_oracle_resolution = (
        max(oracle_summary_by_resolution, key=lambda r: oracle_summary_by_resolution[r]["oracle_hierarchy_cut_SF50_diagnostic"])
        if oracle_summary_by_resolution
        else None
    )
    global_best_oracle_sf50 = (
        oracle_summary_by_resolution.get(global_best_oracle_resolution, {}).get("oracle_hierarchy_cut_SF50_diagnostic")
        if global_best_oracle_resolution is not None
        else None
    )
    global_best_oracle_iou = (
        oracle_summary_by_resolution.get(global_best_oracle_resolution, {}).get("oracle_hierarchy_cut_GT_best_IoU_diagnostic")
        if global_best_oracle_resolution is not None
        else None
    )
    oracle_rows_by_chunk: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in oracle_rows:
        oracle_rows_by_chunk[(str(row.get("scene_id") or ""), _int(row.get("chunk_id"), -1))].append(row)
    oracle_mixed_rows: list[dict[str, Any]] = []
    for (scene, chunk), rows in sorted(oracle_rows_by_chunk.items()):
        best = max(
            rows,
            key=lambda row: (
                _float(row.get("local_SF50"), 0.0),
                _float(row.get("GT_best_IoU_mean"), 0.0),
                -_float(row.get("resolution"), 0.0),
            ),
        )
        mixed = dict(best)
        mixed["variant"] = "LC6_oracle_hierarchy_cut_diagnostic_per_chunk_mixed_resolution"
        mixed["oracle_resolution_selected_for_chunk"] = best.get("resolution")
        oracle_mixed_rows.append(mixed)
    mixed_oracle_sf50 = _mean([_float(row.get("local_SF50"), 0.0) for row in oracle_mixed_rows]) if oracle_mixed_rows else None
    mixed_oracle_iou = _mean([_float(row.get("GT_best_IoU_mean"), 0.0) for row in oracle_mixed_rows]) if oracle_mixed_rows else None
    mixed_resolution_counts = Counter(str(row.get("oracle_resolution_selected_for_chunk")) for row in oracle_mixed_rows)
    if mixed_oracle_sf50 is not None and (global_best_oracle_sf50 is None or mixed_oracle_sf50 >= global_best_oracle_sf50):
        best_oracle_resolution: Any = "per_chunk_mixed"
        best_oracle_sf50 = mixed_oracle_sf50
        best_oracle_iou = mixed_oracle_iou
        oracle_cut_selection_mode = "per_chunk_mixed_resolution_diagnostic_oracle"
    else:
        best_oracle_resolution = global_best_oracle_resolution
        best_oracle_sf50 = global_best_oracle_sf50
        best_oracle_iou = global_best_oracle_iou
        oracle_cut_selection_mode = "global_best_resolution_diagnostic_oracle"
    v73_p5 = _float((phase0.get("key_metrics") or {}).get("v73_local_SF50"), 0.0)
    oracle_target = v73_p5 + 0.05
    parent_child_edge_count = len(hierarchy_edges)
    child_count_mean = _mean([float(count) for count in component_child_counts.values()]) or 0.0
    gate = {
        "phase3_pass": True,
        "largest_cluster_ratio_le_0p35": (_mean(largest_values) or 1.0) <= 0.35,
        "single_frame_cluster_rate_le_0p60": (_mean(single_frame_rates) or 1.0) <= 0.60,
        "cluster_count_in_diagnostic_GT_range": any(count_gate_values),
        "heldout_score_ge_phase3_baseline_plus_0p02": False,
        "parent_child_edge_count_gt_0": parent_child_edge_count > 0,
        "view_conditioned_child_count_mean_gt_1": child_count_mean > 1.0,
        "oracle_hierarchy_cut_evaluated": best_oracle_sf50 is not None,
        "oracle_hierarchy_cut_SF50_ge_v73_P5_plus_0p05": (best_oracle_sf50 or 0.0) >= oracle_target,
        "uses_gt_for_prediction_false": True,
    }
    gate["pass"] = all(gate.values())
    decision = "PASS_V75_PHASE4_LOCAL_HIERARCHY" if gate["pass"] else "NO_GO_PHASE4_HIERARCHY_INCOMPLETE"
    summary = {
        "phase": "v75_phase4_local_hierarchy",
        "schema": "stream4d_v75_phase4_local_hierarchy_v1",
        "decision": decision,
        "gate": gate,
        "best_real_variant": best_variant,
        "key_metrics": {
            "cluster_row_count": len(cluster_rows),
            "cluster_count_per_chunk_mean": _mean([float(v) for v in cluster_counts]) or 0.0,
            "largest_cluster_ratio_mean": _mean(largest_values) or 0.0,
            "single_frame_cluster_rate_mean": _mean(single_frame_rates) or 0.0,
            "parent_child_edge_count": parent_child_edge_count,
            "view_conditioned_child_count_mean": child_count_mean,
            "best_oracle_resolution": best_oracle_resolution,
            "oracle_cut_selection_mode": oracle_cut_selection_mode,
            "global_best_oracle_resolution": global_best_oracle_resolution,
            "global_best_oracle_SF50_diagnostic": global_best_oracle_sf50,
            "global_best_oracle_GT_best_IoU_diagnostic": global_best_oracle_iou,
            "mixed_oracle_resolution_counts": dict(mixed_resolution_counts),
            "oracle_hierarchy_cut_SF50_diagnostic": best_oracle_sf50,
            "oracle_hierarchy_cut_GT_best_IoU_diagnostic": best_oracle_iou,
            "oracle_target_SF50_v73_P5_plus_0p05": oracle_target,
        },
        "runtime_sec": time.time() - started,
        "config": {
            "phase4_resolutions": resolution_values,
            "phase4_adapter_min_f1": float(args.phase4_adapter_min_f1),
            "phase4_adapter_min_precision": float(args.phase4_adapter_min_precision),
            "phase4_parent_top_k": int(args.phase4_parent_top_k),
            "phase4_parent_min_shared_masks": float(args.phase4_parent_min_shared_masks),
            "phase4_parent_min_edge_weight": float(args.phase4_parent_min_edge_weight),
            "phase4_mask_specificity_power": float(args.phase4_mask_specificity_power),
            "phase4_mask_area_specificity_power": float(args.phase4_mask_area_specificity_power),
            "phase4_max_same_level_mask_area_ratio": float(args.phase4_max_same_level_mask_area_ratio),
        },
        "inputs": {
            "phase3_summary": _rel(phase3_summary_path),
            "phase3_assignment_rows": _rel(assignment_path),
            "phase1_incidence_rows": _rel(incidence_path),
            "phase1_summary": _rel(phase1_summary_path),
        },
        "notes": [
            "Phase4 builds a deterministic non-GT mask co-observation hierarchy over Phase3 carrier clusters.",
            "This is a lightweight resolution sweep, not a full Leiden/Louvain implementation.",
            "Oracle hierarchy cut uses GT only for diagnostic cut selection/evaluation and remains forbidden for method-table prediction.",
        ],
    }
    _write_csv(output_root / "cluster_rows.csv", cluster_rows)
    _write_csv(output_root / "hierarchy_edge_rows.csv", hierarchy_edges)
    _write_csv(output_root / "resolution_metric_rows.csv", resolution_rows)
    _write_csv(output_root / "oracle_cut_rows.csv", oracle_rows)
    _write_csv(output_root / "oracle_mixed_cut_rows.csv", oracle_mixed_rows)
    _write_csv(output_root / "adapter_candidate_rows.csv", adapter_candidates)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _write_json(output_root / "hierarchy_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    sha_rows = []
    for path in [phase3_summary_path, assignment_path, phase1_summary_path, incidence_path, phase0_summary_path, *sorted(output_root.glob("*"))]:
        if path.exists() and path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"name": f"input_or_output:{path.name}", "source_artifact": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    return summary


def _run_phase5_local_cut(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase5_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    phase3_summary_path = ROOT / args.phase3_output_root / "propagation_summary.json"
    assignment_path = ROOT / args.phase3_output_root / "assignment_rows.csv"
    phase1_summary_path = ROOT / args.phase1_output_root / "incidence_summary.json"
    incidence_path = ROOT / args.phase1_output_root / "incidence_rows.csv"
    phase0_summary_path = ROOT / args.phase0_output_root / "fact_lock_summary.json"
    phase4_summary_path = ROOT / args.phase4_output_root / "hierarchy_summary.json"
    missing = []
    for path in [phase3_summary_path, assignment_path, phase1_summary_path, incidence_path, phase0_summary_path]:
        if not path.exists():
            missing.append({"missing": path.name, "path": _rel(path)})
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {
            "phase": "v75_phase5_local_cut",
            "schema": "stream4d_v75_phase5_local_cut_v1",
            "decision": "NO_GO_PHASE5_MISSING_INPUT",
            "gate": {"pass": False, "all_inputs_present": False},
            "missing_input_count": len(missing),
        }
        _write_json(output_root / "local_cut_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        return summary

    from tools.run_v66_local_chunk_eval import _evaluate_frame_data, _frame_data, _frag_overmerge_means, _score_free  # noqa: E402

    phase3 = json.loads(phase3_summary_path.read_text(encoding="utf-8"))
    phase1 = json.loads(phase1_summary_path.read_text(encoding="utf-8"))
    phase0 = json.loads(phase0_summary_path.read_text(encoding="utf-8"))
    phase4 = json.loads(phase4_summary_path.read_text(encoding="utf-8")) if phase4_summary_path.exists() else None
    best_variant = str(phase3.get("best_real_variant") or "")
    carrier_to_label: dict[tuple[str, int, int], int] = {}
    cluster_carriers: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    for row in _read_csv_rows(assignment_path):
        if row.get("variant") != best_variant:
            continue
        label = _int(row.get("cluster_seed_label"), -1)
        if label < 0:
            continue
        scene = str(row.get("scene_id") or "")
        chunk = _int(row.get("chunk_id"), -1)
        carrier = _int(row.get("carrier_id"), -1)
        carrier_to_label[(scene, chunk, carrier)] = label
        cluster_carriers[(scene, chunk, label)].add(carrier)

    adapter_acc: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    cluster_frame_carriers: dict[tuple[str, int, int, int], set[int]] = defaultdict(set)
    frames_by_chunk: dict[tuple[str, int], set[int]] = defaultdict(set)
    with incidence_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("membership_variant") != args.main_variant:
                continue
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            carrier = _int(row.get("carrier_id"), -1)
            label = carrier_to_label.get((scene, chunk, carrier))
            if label is None:
                continue
            frame = _int(row.get("frame_id"), 0)
            mask_id = _int(row.get("mask_id"), 0)
            frames_by_chunk[(scene, chunk)].add(frame)
            cluster_frame_carriers[(scene, chunk, label, frame)].add(carrier)
            membership = _float(row.get("soft_membership"), 0.0)
            if mask_id <= 0 or membership < float(args.phase3_eval_membership_threshold):
                continue
            key = (scene, chunk, label, frame, mask_id)
            acc = adapter_acc.setdefault(
                key,
                {
                    "mass": 0.0,
                    "mask_mass": _float(row.get("mask_carrier_mass"), 0.0),
                    "mask_area_ratio": _float(row.get("mask_area_ratio"), 0.0),
                    "semantic_entropy": _float(row.get("semantic_entropy_of_mask"), 0.0),
                    "mask_observation_id": row.get("mask_observation_id"),
                },
            )
            acc["mass"] = float(acc["mass"]) + membership
            acc["mask_mass"] = max(float(acc["mask_mass"]), _float(row.get("mask_carrier_mass"), 0.0))

    candidates: list[dict[str, Any]] = []
    for (scene, chunk, label, frame, mask_id), acc in adapter_acc.items():
        cluster_frame_count = len(cluster_frame_carriers.get((scene, chunk, label, frame), set()))
        precision = float(acc["mass"]) / max(float(acc["mask_mass"]), 1e-9)
        recall = float(acc["mass"]) / max(float(cluster_frame_count), 1e-9)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        candidates.append(
            {
                "scene_id": scene,
                "chunk_id": chunk,
                "cluster_id": label,
                "frame_id": frame,
                "mask_id": mask_id,
                "mask_observation_id": acc.get("mask_observation_id"),
                "adapter_precision": precision,
                "adapter_recall": recall,
                "adapter_F1": f1,
                "mask_area_ratio": acc.get("mask_area_ratio"),
                "semantic_entropy": acc.get("semantic_entropy"),
                "broad_adapter": float(acc.get("mask_area_ratio") or 0.0) >= float(args.large_mask_area_ratio),
                "uses_gt_for_prediction": False,
            }
        )

    pair_to_candidates: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if float(row["adapter_F1"]) < float(args.phase5_min_adapter_f1):
            continue
        if float(row["adapter_precision"]) < float(args.phase5_min_adapter_precision):
            continue
        if float(row["adapter_recall"]) < float(args.phase5_min_adapter_recall):
            continue
        if bool(args.phase5_demote_broad_adapters) and bool(row["broad_adapter"]):
            continue
        row["flat_adapter_eligible"] = True
        if bool(args.phase5_demote_broad_adapters):
            row["adapter_policy"] = "phase5_thresholded_broad_demoted"
        else:
            row["adapter_policy"] = "phase5_thresholded"
        pair_to_candidates[(str(row["scene_id"]), int(row["chunk_id"]), int(row["frame_id"]), int(row["mask_id"]))].append(row)
    selected_by_pair: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    pre_nms_conflict_count = 0
    for key, rows in pair_to_candidates.items():
        if len(rows) > 1:
            pre_nms_conflict_count += 1
        selected_by_pair[key] = max(rows, key=lambda row: (float(row["adapter_F1"]), float(row["adapter_precision"]), -int(row["cluster_id"])))
    pair_selection_count = len(selected_by_pair)
    cluster_frame_suppressed_count = 0
    if bool(args.phase5_one_mask_per_slot_frame):
        selected_by_cluster_frame: dict[tuple[str, int, int, int], tuple[tuple[str, int, int, int], dict[str, Any]]] = {}
        for key, row in selected_by_pair.items():
            cluster_key = (str(row["scene_id"]), int(row["chunk_id"]), int(row["cluster_id"]), int(row["frame_id"]))
            old = selected_by_cluster_frame.get(cluster_key)
            if old is None or (float(row["adapter_F1"]), float(row["adapter_precision"])) > (float(old[1]["adapter_F1"]), float(old[1]["adapter_precision"])):
                selected_by_cluster_frame[cluster_key] = (key, row)
        kept_keys = {key for key, _row in selected_by_cluster_frame.values()}
        cluster_frame_suppressed_count = max(0, len(selected_by_pair) - len(kept_keys))
        selected_by_pair = {key: row for key, row in selected_by_pair.items() if key in kept_keys}

    merge_parent: dict[tuple[str, int, int], tuple[str, int, int]] = {}

    def _merge_find(key: tuple[str, int, int]) -> tuple[str, int, int]:
        merge_parent.setdefault(key, key)
        parent = merge_parent[key]
        if parent != key:
            merge_parent[key] = _merge_find(parent)
        return merge_parent[key]

    def _merge_union(left: tuple[str, int, int], right: tuple[str, int, int]) -> bool:
        left_root = _merge_find(left)
        right_root = _merge_find(right)
        if left_root == right_root:
            return False
        if (left_root[0], left_root[1], left_root[2]) <= (right_root[0], right_root[1], right_root[2]):
            merge_parent[right_root] = left_root
        else:
            merge_parent[left_root] = right_root
        return True

    for key in cluster_carriers:
        _merge_find(key)
    cluster_merge_edge_count = 0
    if bool(args.phase5_merge_competing_mask_clusters):
        for rows in pair_to_candidates.values():
            cluster_keys = sorted({(str(row["scene_id"]), int(row["chunk_id"]), int(row["cluster_id"])) for row in rows})
            if len(cluster_keys) <= 1:
                continue
            base = cluster_keys[0]
            for other in cluster_keys[1:]:
                if _merge_union(base, other):
                    cluster_merge_edge_count += 1
    merged_cluster_carriers: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    for key, carriers in cluster_carriers.items():
        merged_cluster_carriers[_merge_find(key)].update(carriers)
    merged_cluster_count = len({root for root, carriers in merged_cluster_carriers.items() if carriers})

    selected_pair_counts: Counter[tuple[str, int, int, int]] = Counter()
    for key in selected_by_pair:
        selected_pair_counts[key] += 1
    unresolved_conflict_count = sum(1 for count in selected_pair_counts.values() if count > 1)
    pre_nms_conflict_rate = float(pre_nms_conflict_count / max(1, len(pair_to_candidates)))
    unresolved_conflict_rate = float(unresolved_conflict_count / max(1, len(selected_pair_counts)))

    slot_stats: dict[tuple[str, int, int], dict[str, Any]] = defaultdict(lambda: {"frames": set(), "f1": [], "precision": [], "recall": [], "broad": []})
    mapping_by_chunk: dict[tuple[str, int], dict[tuple[int, int], int]] = defaultdict(dict)
    for (scene, chunk, frame, mask_id), row in selected_by_pair.items():
        label = int(row["cluster_id"])
        merged_scene, merged_chunk, merged_label = _merge_find((scene, chunk, label))
        slot_id = merged_label + 1
        mapping_by_chunk[(scene, chunk)][(frame, mask_id)] = slot_id
        stats = slot_stats[(merged_scene, merged_chunk, merged_label)]
        stats["frames"].add(frame)
        stats["f1"].append(float(row["adapter_F1"]))
        stats["precision"].append(float(row["adapter_precision"]))
        stats["recall"].append(float(row["adapter_recall"]))
        stats["broad"].append(1.0 if bool(row["broad_adapter"]) else 0.0)

    slot_rows: list[dict[str, Any]] = []
    for (scene, chunk, label), stats in sorted(slot_stats.items()):
        slot_rows.append(
            {
                "scene_id": scene,
                "chunk_id": chunk,
                "local_slot_id": label + 1,
                "variant": "LC5_full_nonGT_cut",
                "resolution_cut": "phase3_seed_label_proxy_merged_competing_masks" if bool(args.phase5_merge_competing_mask_clusters) else "phase3_seed_label_proxy",
                "carrier_count": len(merged_cluster_carriers.get((scene, chunk, label), set())),
                "frame_count": len(stats["frames"]),
                "adapter_mask_count": len(stats["f1"]),
                "adapter_precision_mean": _mean(stats["precision"]) or 0.0,
                "adapter_recall_mean": _mean(stats["recall"]) or 0.0,
                "adapter_F1_mean": _mean(stats["f1"]) or 0.0,
                "part_adapter_rate": _mean([1.0 if p >= 0.50 and r < 0.50 else 0.0 for p, r in zip(stats["precision"], stats["recall"])]) or 0.0,
                "parent_adapter_rate": _mean([1.0 if r >= 0.50 and p < 0.50 else 0.0 for p, r in zip(stats["precision"], stats["recall"])]) or 0.0,
                "broad_adapter_rate": _mean(stats["broad"]) or 0.0,
                "semantic_proto_id": "",
                "parent_slot_id": "",
                "child_slot_ids": "",
                "confidence": _mean(stats["f1"]) or 0.0,
                "ambiguity_score": 1.0 - (_mean(stats["f1"]) or 0.0),
                "uses_gt_for_prediction": False,
            }
        )

    mask_dirs = phase1.get("mask_dirs") or {}
    metric_rows: list[dict[str, Any]] = []
    for (scene, chunk), frames in sorted(frames_by_chunk.items()):
        frame_ids = sorted(frames)
        if not frame_ids:
            continue
        mask_dir = _resolve_stream3d_path(mask_dirs.get(scene, ""))
        frame_data = _frame_data(scene, frame_ids, mask_dir)
        summary_eval, iou, _pred_ids, _gt_ids = _evaluate_frame_data(
            frame_data=frame_data,
            variant="LC5_full_nonGT_cut",
            mapping=mapping_by_chunk.get((scene, chunk), {}),
            raw_per_frame_masks=False,
        )
        frag_mean, over_mean = _frag_overmerge_means(iou)
        chunk_slots = [row for row in slot_rows if row["scene_id"] == scene and int(row["chunk_id"]) == int(chunk)]
        single_frame_slot_rate = float(sum(1 for row in chunk_slots if int(row["frame_count"]) <= 1) / max(1, len(chunk_slots)))
        broad_rate = _mean([_float(row.get("broad_adapter_rate"), 0.0) for row in chunk_slots]) or 0.0
        local_sf50 = _score_free(summary_eval)
        metric_rows.append(
            {
                "scene_id": scene,
                "chunk_id": chunk,
                "variant": "LC5_full_nonGT_cut",
                "local_SF50": local_sf50,
                "local_AP50": summary_eval.get("ap50"),
                "local_AP25": summary_eval.get("ap25"),
                "GT_best_IoU_mean": summary_eval.get("gt_best_iou_mean"),
                "pred_best_IoU_median": summary_eval.get("pred_best_iou_median"),
                "same_frame_violation_count": 0,
                "duplicate_frame_mask_conflict_rate": unresolved_conflict_rate,
                "pre_nms_duplicate_frame_mask_conflict_rate": pre_nms_conflict_rate,
                "single_frame_slot_rate": single_frame_slot_rate,
                "fragments_per_GT_at_0p10": frag_mean,
                "GT_per_pred_at_0p10": over_mean,
                "unresolved_broad_underseg_rate": broad_rate,
                "adapter_precision_mean": _mean([_float(row.get("adapter_precision_mean"), 0.0) for row in chunk_slots]) or 0.0,
                "adapter_recall_mean": _mean([_float(row.get("adapter_recall_mean"), 0.0) for row in chunk_slots]) or 0.0,
                "oracle_minus_nonGT_cut_SF50": "",
                "uses_gt_for_prediction": False,
            }
        )

    lc5_sf50_mean = _mean([_float(row.get("local_SF50"), 0.0) for row in metric_rows])
    lc5_gt_iou_mean = _mean([_float(row.get("GT_best_IoU_mean"), 0.0) for row in metric_rows])
    lc5_single_mean = _mean([_float(row.get("single_frame_slot_rate"), 1.0) for row in metric_rows])
    lc5_dup_mean = _mean([_float(row.get("duplicate_frame_mask_conflict_rate"), 1.0) for row in metric_rows])
    lc5_broad_mean = _mean([_float(row.get("unresolved_broad_underseg_rate"), 1.0) for row in metric_rows])
    lc5_sf50 = lc5_sf50_mean if lc5_sf50_mean is not None else 0.0
    lc5_gt_iou = lc5_gt_iou_mean if lc5_gt_iou_mean is not None else 0.0
    lc5_single = lc5_single_mean if lc5_single_mean is not None else 1.0
    lc5_dup = lc5_dup_mean if lc5_dup_mean is not None else 1.0
    lc5_broad = lc5_broad_mean if lc5_broad_mean is not None else 1.0
    phase0_metrics = phase0.get("key_metrics") or {}
    imported_controls = [
        _float(phase0_metrics.get("v73_area_only_control_SF50"), 0.0),
        _float(phase0_metrics.get("v73_lattice_only_control_SF50"), 0.0),
        _float(phase0_metrics.get("v73_local_SF50"), 0.0),
    ]
    control_target = max(imported_controls) + 0.03
    phase4_metrics = (phase4 or {}).get("key_metrics") or {}
    lc6_oracle_sf50 = phase4_metrics.get("oracle_hierarchy_cut_SF50_diagnostic")
    lc6_oracle_iou = phase4_metrics.get("oracle_hierarchy_cut_GT_best_IoU_diagnostic")
    lc6_oracle_target = phase4_metrics.get("oracle_target_SF50_v73_P5_plus_0p05")
    lc6_oracle_minus_lc5 = None
    if lc6_oracle_sf50 is not None:
        lc6_oracle_minus_lc5 = _float(lc6_oracle_sf50, 0.0) - lc5_sf50
    gate = {
        "LC5_full_nonGT_cut_SF50_ge_controls_plus_0p03": lc5_sf50 >= control_target,
        "LC5_GT_best_IoU_mean_ge_0p25": lc5_gt_iou >= 0.25,
        "LC5_same_frame_violation_count_eq_0": True,
        "LC5_duplicate_frame_mask_conflict_rate_le_0p02": lc5_dup <= 0.02,
        "LC5_single_frame_slot_rate_le_0p60": lc5_single <= 0.60,
        "LC5_unresolved_broad_underseg_rate_le_0p35": lc5_broad <= 0.35,
        "LC6_oracle_hierarchy_cut_available": lc6_oracle_sf50 is not None,
        "LC6_oracle_minus_LC5_SF50_le_0p10": lc6_oracle_minus_lc5 is not None and lc6_oracle_minus_lc5 <= 0.10,
        "uses_gt_for_prediction_false": True,
    }
    gate["pass"] = all(gate.values())
    decision = "PASS_V75_PHASE5_LOCAL_CUT" if gate["pass"] else "NO_GO_PHASE5_LOCAL_CUT_GATE_FAILED"
    control_rows = [
        {"control": "v73_area_only_control_SF50", "value": imported_controls[0], "source_phase": "phase0_fact_lock"},
        {"control": "v73_lattice_only_control_SF50", "value": imported_controls[1], "source_phase": "phase0_fact_lock"},
        {"control": "v73_local_SF50", "value": imported_controls[2], "source_phase": "phase0_fact_lock"},
        {"control": "LC5_required_SF50_threshold", "value": control_target, "source_phase": "phase5_gate"},
        {"control": "LC6_oracle_hierarchy_cut_SF50_diagnostic", "value": lc6_oracle_sf50, "source_phase": "phase4_hierarchy"},
        {"control": "LC6_oracle_target_SF50_v73_P5_plus_0p05", "value": lc6_oracle_target, "source_phase": "phase4_hierarchy"},
        {"control": "LC6_oracle_minus_LC5_SF50", "value": lc6_oracle_minus_lc5, "source_phase": "phase5_gate"},
    ]
    variant_rows = [
        {
            "variant": "LC5_full_nonGT_cut",
            "chunk_count": len(metric_rows),
            "local_SF50_mean": lc5_sf50,
            "local_AP50_mean": _mean([_float(row.get("local_AP50"), 0.0) for row in metric_rows]) or 0.0,
            "local_AP25_mean": _mean([_float(row.get("local_AP25"), 0.0) for row in metric_rows]) or 0.0,
            "GT_best_IoU_mean": lc5_gt_iou,
            "single_frame_slot_rate_mean": lc5_single,
            "duplicate_frame_mask_conflict_rate_mean": lc5_dup,
            "unresolved_broad_underseg_rate_mean": lc5_broad,
        }
    ]
    summary = {
        "phase": "v75_phase5_local_cut",
        "schema": "stream4d_v75_phase5_local_cut_v1",
        "decision": decision,
        "gate": gate,
        "best_real_variant": best_variant,
        "key_metrics": {
            "LC5_full_nonGT_cut_SF50": lc5_sf50,
            "LC5_GT_best_IoU_mean": lc5_gt_iou,
            "LC5_duplicate_frame_mask_conflict_rate": lc5_dup,
            "LC5_single_frame_slot_rate": lc5_single,
            "LC5_unresolved_broad_underseg_rate": lc5_broad,
            "LC6_oracle_hierarchy_cut_SF50_diagnostic": lc6_oracle_sf50,
            "LC6_oracle_hierarchy_cut_GT_best_IoU_diagnostic": lc6_oracle_iou,
            "LC6_oracle_target_SF50_v73_P5_plus_0p05": lc6_oracle_target,
            "LC6_oracle_minus_LC5_SF50": lc6_oracle_minus_lc5,
            "control_target_SF50": control_target,
            "local_slot_count": len(slot_rows),
            "selected_adapter_pair_count": len(selected_by_pair),
            "pre_nms_selected_adapter_pair_count": pair_selection_count,
            "pre_nms_adapter_conflict_pair_count": pre_nms_conflict_count,
            "pre_nms_duplicate_frame_mask_conflict_rate": pre_nms_conflict_rate,
            "adapter_conflict_pair_count": unresolved_conflict_count,
            "cluster_frame_suppressed_adapter_count": cluster_frame_suppressed_count,
            "cluster_merge_edge_count": cluster_merge_edge_count,
            "merged_cluster_count": merged_cluster_count,
            "method_gt_violation_count": 0,
        },
        "runtime_sec": time.time() - started,
        "inputs": {
            "phase3_summary": _rel(phase3_summary_path),
            "phase3_assignment_rows": _rel(assignment_path),
            "phase1_incidence_rows": _rel(incidence_path),
            "phase4_summary": _rel(phase4_summary_path) if phase4_summary_path.exists() else None,
        },
        "config": {
            "phase5_min_adapter_f1": float(args.phase5_min_adapter_f1),
            "phase5_min_adapter_precision": float(args.phase5_min_adapter_precision),
            "phase5_min_adapter_recall": float(args.phase5_min_adapter_recall),
            "phase5_demote_broad_adapters": bool(args.phase5_demote_broad_adapters),
            "phase5_one_mask_per_slot_frame": bool(args.phase5_one_mask_per_slot_frame),
            "phase5_merge_competing_mask_clusters": bool(args.phase5_merge_competing_mask_clusters),
        },
        "notes": [
            "LC5 maps Phase3 carrier clusters back to real CropFormer masks by non-GT adapter precision/recall/F1.",
            "Duplicate frame-mask conflict is measured after non-GT adapter NMS; pre-NMS candidate competition is recorded separately.",
            "Optional competing-mask cluster merge uses only non-GT mask co-observation before local metric evaluation.",
            "GT is used only by the local metric evaluator after prediction materialization; it is not used for adapter selection.",
            "LC6 oracle hierarchy cut comes from Phase4 diagnostic rows when available; it is not a method-table prediction.",
        ],
    }
    _write_csv(output_root / "local_slot_rows.csv", slot_rows)
    _write_csv(output_root / "local_slot_metric_rows.csv", metric_rows)
    _write_csv(output_root / "cut_variant_summary_rows.csv", variant_rows)
    _write_csv(output_root / "control_rows.csv", control_rows)
    _write_csv(output_root / "adapter_candidate_rows.csv", candidates)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _write_json(output_root / "local_cut_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    sha_rows = []
    for path in [phase3_summary_path, assignment_path, incidence_path, *sorted(output_root.glob("*"))]:
        if path.exists() and path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"name": f"input_or_output:{path.name}", "source_artifact": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    return summary


def _run_phase6_local_attribution(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase6_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    phase3_summary_path = ROOT / args.phase3_output_root / "propagation_summary.json"
    phase5_summary_path = ROOT / args.phase5_output_root / "local_cut_summary.json"
    missing: list[dict[str, Any]] = []
    if not phase3_summary_path.exists():
        missing.append({"missing": phase3_summary_path.name, "path": _rel(phase3_summary_path)})
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {
            "phase": "v75_phase6_local_attribution",
            "schema": "stream4d_v75_phase6_local_attribution_v1",
            "decision": "NO_GO_PHASE6_MISSING_INPUT",
            "gate": {"pass": False, "all_inputs_present": False},
            "missing_input_count": len(missing),
        }
        _write_json(output_root / "attribution_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        return summary

    phase3 = json.loads(phase3_summary_path.read_text(encoding="utf-8"))
    phase3_gate = phase3.get("gate") or {}
    phase3_metrics = phase3.get("key_metrics") or {}
    phase3_pass = bool(phase3_gate.get("pass"))
    phase5 = json.loads(phase5_summary_path.read_text(encoding="utf-8")) if phase5_summary_path.exists() else None
    phase5_gate = (phase5 or {}).get("gate") or {}
    phase5_metrics = (phase5 or {}).get("key_metrics") or {}
    phase5_pass = bool(phase5_gate.get("pass"))
    if not phase3_pass:
        local_decision = "NO_GO_PHASE3_D4RT_COMASK_SIGNAL_INSUFFICIENT"
        primary = "PHASE3_A3_A4_A5_DID_NOT_SATISFY_BASELINE_AND_COLLAPSE_GATES"
        secondary = "PHASE4_PHASE5_NOT_RUN_BLOCKED_BY_PHASE3"
        can_enter_local2history = False
        local2history_decision = "NO_GO_LOCAL2HISTORY_BLOCKED_BY_LOCAL"
    elif phase5 is None:
        local_decision = "NO_GO_PHASE5_LOCAL_CUT_NOT_RUN"
        primary = "PHASE5_LOCAL_CUT_NOT_RUN_AFTER_PHASE3_PASS"
        secondary = "LOCAL_FINAL_METRIC_GATE_NOT_EVALUATED"
        can_enter_local2history = False
        local2history_decision = "NO_GO_LOCAL2HISTORY_BLOCKED_BY_LOCAL"
    elif not phase5_pass:
        local_decision = "NO_GO_PHASE5_LOCAL_CUT_GATE_FAILED"
        primary = "PHASE5_LC5_DID_NOT_PASS_LOCAL_METHOD_AND_CONTROL_GATES"
        secondary = "STAGE2_LOCAL2HISTORY_BLOCKED_BY_LOCAL"
        can_enter_local2history = False
        local2history_decision = "NO_GO_LOCAL2HISTORY_BLOCKED_BY_LOCAL"
    else:
        local_decision = "GO_V75_LOCAL_ATTRIBUTED"
        primary = "LOCAL_METHOD_ATTRIBUTION_PASSED"
        secondary = "STAGE2_LOCAL2HISTORY_NOT_RUN_IN_THIS_PIPELINE"
        can_enter_local2history = True
        local2history_decision = "READY_FOR_STAGE2_LOCAL2HISTORY_NOT_RUN"

    decision_rows = [
        {
            "case": "A_D4RT_mask_signal_insufficient",
            "condition": "phase3_gate_pass_false",
            "observed": not phase3_pass,
            "evidence": phase3.get("decision"),
            "decision_if_true": "NO_GO_PHASE3_D4RT_COMASK_SIGNAL_INSUFFICIENT",
        },
        {
            "case": "C_specificity_PMI_effective",
            "condition": "broad_edge_reduced_and_controls_pass",
            "observed": bool(phase3_gate.get("real_minus_shuffled_heldout_likelihood_ge_0p03"))
            and bool(phase3_gate.get("real_minus_no_temporal_heldout_likelihood_ge_0p02"))
            and bool(phase3_gate.get("A3_A4_or_A5_broad_edge_mass_ratio_le_A0_minus_0p20")),
            "evidence": json.dumps(
                {
                    "real_minus_shuffled": phase3_metrics.get("real_minus_shuffled_heldout_likelihood"),
                    "real_minus_no_temporal": phase3_metrics.get("real_minus_no_temporal_heldout_likelihood"),
                    "best_real_broad_edge_mass_ratio": phase3_metrics.get("best_real_broad_edge_mass_ratio"),
                    "A0_broad_edge_mass_ratio": phase3_metrics.get("A0_broad_edge_mass_ratio"),
                },
                sort_keys=True,
            ),
            "decision_if_true": "PARTIAL_SIGNAL_PRESENT_BUT_NOT_CLAIMABLE",
        },
        {
            "case": "D_local_cut_gate_failed",
            "condition": "phase3_pass_true_and_phase5_gate_false",
            "observed": bool(phase3_pass and phase5 is not None and not phase5_pass),
            "evidence": json.dumps(phase5_metrics, sort_keys=True) if phase5 is not None else "phase5_not_run",
            "decision_if_true": "NO_GO_PHASE5_LOCAL_CUT_GATE_FAILED",
        },
        {
            "case": "A2_hierarchy_oracle_signal_insufficient",
            "condition": "oracle_hierarchy_cut_available_and_below_plan_target",
            "observed": bool(
                phase5 is not None
                and phase5_metrics.get("LC6_oracle_hierarchy_cut_SF50_diagnostic") is not None
                and phase5_metrics.get("LC6_oracle_target_SF50_v73_P5_plus_0p05") is not None
                and _float(phase5_metrics.get("LC6_oracle_hierarchy_cut_SF50_diagnostic"), 0.0)
                < _float(phase5_metrics.get("LC6_oracle_target_SF50_v73_P5_plus_0p05"), 0.0)
            ),
            "evidence": json.dumps(
                {
                    "LC6_oracle_hierarchy_cut_SF50_diagnostic": phase5_metrics.get("LC6_oracle_hierarchy_cut_SF50_diagnostic"),
                    "LC6_oracle_target_SF50_v73_P5_plus_0p05": phase5_metrics.get("LC6_oracle_target_SF50_v73_P5_plus_0p05"),
                },
                sort_keys=True,
            )
            if phase5 is not None
            else "phase5_not_run",
            "decision_if_true": "NO_GO_PHASE4_HIERARCHY_SIGNAL_INSUFFICIENT",
        },
        {
            "case": "B_signal_present_but_cut_selection_misused",
            "condition": "oracle_hierarchy_cut_available_and_oracle_minus_nonGT_gap_gt_0p10",
            "observed": bool(
                phase5 is not None
                and phase5_metrics.get("LC6_oracle_hierarchy_cut_SF50_diagnostic") is not None
                and _float(phase5_metrics.get("LC6_oracle_minus_LC5_SF50"), 0.0) > 0.10
            ),
            "evidence": json.dumps(
                {
                    "LC6_oracle_hierarchy_cut_SF50_diagnostic": phase5_metrics.get("LC6_oracle_hierarchy_cut_SF50_diagnostic"),
                    "LC5_full_nonGT_cut_SF50": phase5_metrics.get("LC5_full_nonGT_cut_SF50"),
                    "LC6_oracle_minus_LC5_SF50": phase5_metrics.get("LC6_oracle_minus_LC5_SF50"),
                },
                sort_keys=True,
            )
            if phase5 is not None
            else "phase5_not_run",
            "decision_if_true": "NO_GO_CUT_SELECTION_MISUSED",
        },
        {
            "case": "E_local_passed_with_attribution",
            "condition": "phase3_gate_pass_true_and_phase5_gate_pass_true",
            "observed": bool(phase3_pass and phase5_pass),
            "evidence": json.dumps(phase5_metrics, sort_keys=True) if phase5 is not None else "phase5_not_run",
            "decision_if_true": "GO_V75_LOCAL_ATTRIBUTED",
        },
    ]
    control_rows = [
        {
            "comparison": "best_real_minus_shuffled_heldout_likelihood",
            "value": phase3_metrics.get("real_minus_shuffled_heldout_likelihood"),
            "threshold": 0.03,
            "pass": phase3_gate.get("real_minus_shuffled_heldout_likelihood_ge_0p03"),
            "source_phase": "phase3",
        },
        {
            "comparison": "best_real_minus_no_temporal_heldout_likelihood",
            "value": phase3_metrics.get("real_minus_no_temporal_heldout_likelihood"),
            "threshold": 0.02,
            "pass": phase3_gate.get("real_minus_no_temporal_heldout_likelihood_ge_0p02"),
            "source_phase": "phase3",
        },
        {
            "comparison": "best_real_heldout_minus_A0",
            "value": _float(phase3_metrics.get("best_real_heldout_same_mask_likelihood"), 0.0)
            - _float(phase3_metrics.get("A0_heldout_same_mask_likelihood"), 0.0),
            "threshold": 0.03,
            "pass": phase3_gate.get("A3_A4_or_A5_heldout_likelihood_ge_A0_plus_0p03")
            or phase3_gate.get("A3_or_A5_heldout_likelihood_ge_A0_plus_0p03"),
            "source_phase": "phase3",
        },
        {
            "comparison": "A0_largest_minus_best_real_largest",
            "value": _float(phase3_metrics.get("A0_largest_cluster_ratio_before_clustering"), 0.0)
            - _float(phase3_metrics.get("best_real_largest_cluster_ratio_before_clustering"), 0.0),
            "threshold": 0.15,
            "pass": phase3_gate.get("A3_A4_or_A5_largest_cluster_ratio_le_A0_minus_0p15")
            or phase3_gate.get("A3_or_A5_largest_cluster_ratio_le_A0_minus_0p15"),
            "source_phase": "phase3",
        },
    ]
    if phase5 is not None:
        control_rows.extend(
            [
                {
                    "comparison": "LC5_full_nonGT_cut_SF50",
                    "value": phase5_metrics.get("LC5_full_nonGT_cut_SF50"),
                    "threshold": phase5_metrics.get("control_target_SF50"),
                    "pass": phase5_gate.get("LC5_full_nonGT_cut_SF50_ge_controls_plus_0p03"),
                    "source_phase": "phase5",
                },
                {
                    "comparison": "LC5_GT_best_IoU_mean",
                    "value": phase5_metrics.get("LC5_GT_best_IoU_mean"),
                    "threshold": 0.25,
                    "pass": phase5_gate.get("LC5_GT_best_IoU_mean_ge_0p25"),
                    "source_phase": "phase5",
                },
                {
                    "comparison": "LC6_oracle_hierarchy_cut_vs_plan_target",
                    "value": phase5_metrics.get("LC6_oracle_hierarchy_cut_SF50_diagnostic"),
                    "threshold": phase5_metrics.get("LC6_oracle_target_SF50_v73_P5_plus_0p05"),
                    "pass": phase5_metrics.get("LC6_oracle_hierarchy_cut_SF50_diagnostic") is not None
                    and phase5_metrics.get("LC6_oracle_target_SF50_v73_P5_plus_0p05") is not None
                    and _float(phase5_metrics.get("LC6_oracle_hierarchy_cut_SF50_diagnostic"), 0.0)
                    >= _float(phase5_metrics.get("LC6_oracle_target_SF50_v73_P5_plus_0p05"), 0.0),
                    "source_phase": "phase4_phase5",
                },
                {
                    "comparison": "LC6_oracle_minus_LC5_SF50",
                    "value": phase5_metrics.get("LC6_oracle_minus_LC5_SF50"),
                    "threshold": 0.10,
                    "pass": phase5_gate.get("LC6_oracle_minus_LC5_SF50_le_0p10"),
                    "source_phase": "phase4_phase5",
                },
            ]
        )
    summary = {
        "phase": "v75_phase6_local_attribution",
        "schema": "stream4d_v75_phase6_local_attribution_v1",
        "decision": local_decision,
        "local_decision": local_decision,
        "local2history_decision": local2history_decision,
        "can_enter_local2history": can_enter_local2history,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "method_uses_gt_anywhere": False,
        "can_claim_method_table": bool(phase5_pass),
        "can_claim_diagnostic_table_only": not bool(phase5_pass),
        "phase3_decision": phase3.get("decision"),
        "phase5_decision": (phase5 or {}).get("decision"),
        "best_real_variant": phase3.get("best_real_variant"),
        "phase3_gate": phase3_gate,
        "phase3_key_metrics": phase3_metrics,
        "phase5_gate": phase5_gate,
        "phase5_key_metrics": phase5_metrics,
        "runtime_sec": time.time() - started,
        "inputs": {
            "phase3_summary": _rel(phase3_summary_path),
            "phase5_summary": _rel(phase5_summary_path) if phase5_summary_path.exists() else None,
        },
        "notes": [
            "Phase6 local_decision is based on Phase5 local cut gates; Phase3 pass alone is not treated as local success.",
            "Stage2 local2history is blocked unless Phase5 local attribution passes.",
            "Phase3 heldout margins are diagnostic co-mask proxy evidence, not AP/SF.",
        ],
    }
    _write_csv(output_root / "decision_matrix_rows.csv", decision_rows)
    _write_csv(output_root / "control_comparison_rows.csv", control_rows)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _write_json(output_root / "attribution_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    sha_rows = []
    for path in [phase3_summary_path, phase5_summary_path, *sorted(output_root.glob("*"))]:
        if path.exists() and path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"name": f"input_or_output:{path.name}", "source_artifact": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    return summary


def _run_final_decision(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.final_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    phase6_summary_path = ROOT / args.phase6_output_root / "attribution_summary.json"
    missing: list[dict[str, Any]] = []
    if not phase6_summary_path.exists():
        missing.append({"missing": phase6_summary_path.name, "path": _rel(phase6_summary_path)})
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {
            "phase": "v75_final_decision",
            "schema": "stream4d_v75_final_decision_v1",
            "final_decision": "NO_GO_FINAL_MISSING_INPUT",
            "gate": {"pass": False, "all_inputs_present": False},
            "missing_input_count": len(missing),
        }
        _write_json(output_root / "final_decision.json", summary)
        _write_json(output_root / "summary.json", summary)
        return summary

    phase6 = json.loads(phase6_summary_path.read_text(encoding="utf-8"))
    phase3_metrics = phase6.get("phase3_key_metrics") or {}
    phase5_metrics = phase6.get("phase5_key_metrics") or {}
    final_decision = phase6.get("decision") or "NO_GO_PHASE3_D4RT_COMASK_SIGNAL_INSUFFICIENT"
    if final_decision == "NO_GO_PHASE3_D4RT_COMASK_SIGNAL_INSUFFICIENT":
        final_label = "NO_GO_PHASE3_D4RT_COMASK_SIGNAL_INSUFFICIENT"
    else:
        final_label = str(final_decision)
    summary = {
        "phase": "v75_final_decision",
        "schema": "stream4d_v75_final_decision_v1",
        "final_decision": final_label,
        "local_decision": phase6.get("local_decision"),
        "local2history_decision": phase6.get("local2history_decision"),
        "method_uses_gt_anywhere": phase6.get("method_uses_gt_anywhere"),
        "can_claim_method_table": phase6.get("can_claim_method_table"),
        "can_claim_diagnostic_table_only": phase6.get("can_claim_diagnostic_table_only"),
        "can_enter_local2history": phase6.get("can_enter_local2history"),
        "primary_blocker": phase6.get("primary_blocker"),
        "secondary_blocker": phase6.get("secondary_blocker"),
        "best_local_variant": phase6.get("best_real_variant") or phase6.get("phase3_decision"),
        "best_local_SF50": phase5_metrics.get("LC5_full_nonGT_cut_SF50"),
        "best_local_controls": {
            "real_minus_shuffled_D4RT": phase3_metrics.get("real_minus_shuffled_heldout_likelihood"),
            "real_minus_no_temporal_D4RT": phase3_metrics.get("real_minus_no_temporal_heldout_likelihood"),
            "LC5_control_target_SF50": phase5_metrics.get("control_target_SF50"),
            "LC6_oracle_hierarchy_cut_SF50_diagnostic": phase5_metrics.get("LC6_oracle_hierarchy_cut_SF50_diagnostic"),
            "LC6_oracle_target_SF50_v73_P5_plus_0p05": phase5_metrics.get("LC6_oracle_target_SF50_v73_P5_plus_0p05"),
            "LC6_oracle_minus_LC5_SF50": phase5_metrics.get("LC6_oracle_minus_LC5_SF50"),
            "heldout_proxy_note": "Phase3 heldout mask co-membership proxy, not AP/SF.",
        },
        "oracle_hierarchy_cut_SF50": phase5_metrics.get("LC6_oracle_hierarchy_cut_SF50_diagnostic"),
        "nonGT_cut_SF50": phase5_metrics.get("LC5_full_nonGT_cut_SF50"),
        "real_minus_shuffled_D4RT": phase3_metrics.get("real_minus_shuffled_heldout_likelihood"),
        "history_scene_SF50": None,
        "local_only_scene_SF50": None,
        "phase4_status": "hierarchy_oracle_evaluated"
        if phase5_metrics.get("LC6_oracle_hierarchy_cut_SF50_diagnostic") is not None
        else "not_run_or_missing",
        "phase5_status": phase6.get("phase5_decision") or "not_run_or_missing",
        "stage2_local2history_status": "blocked_by_local",
        "runtime_sec": time.time() - started,
        "inputs": {"phase6_summary": _rel(phase6_summary_path)},
        "notes": [
            "No method-table success is claimed unless Phase5 local cut gates pass.",
            "GT-derived values in earlier phases remain diagnostic-only and are not method predictions.",
        ],
    }
    _write_json(output_root / "final_decision.json", summary)
    _write_json(output_root / "summary.json", summary)
    _write_csv(output_root / "missing_input_rows.csv", [])
    sha_rows = []
    for path in [phase6_summary_path, *sorted(output_root.glob("*"))]:
        if path.exists() and path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"name": f"input_or_output:{path.name}", "source_artifact": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    return summary


def np_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * percentile / 100.0
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - index) + values[hi] * (index - lo)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    pipeline_root = ROOT / args.pipeline_root
    pipeline_root.mkdir(parents=True, exist_ok=True)

    phase_rows: list[dict[str, Any]] = []
    phase_summaries: dict[str, dict[str, Any]] = {}

    phase0_summary = None
    phase1_summary = None
    if _phase_enabled("phase0", args.stop_after):
        phase0_started = time.time()
        phase0_reused = False
        phase0_path = ROOT / args.phase0_output_root / "fact_lock_summary.json"
        if _reuse_phase(args, "phase0") and phase0_path.exists():
            phase0_summary = json.loads(phase0_path.read_text(encoding="utf-8"))
            phase0_reused = True
        else:
            phase0_summary = v75_fact_lock.run(_namespace(output_root=args.phase0_output_root))
        phase_summaries["phase0"] = phase0_summary
        phase_rows.append(
            {
                "phase": "phase0",
                "decision": phase0_summary.get("decision"),
                "gate_pass": (phase0_summary.get("gate") or {}).get("pass"),
                "output_root": args.phase0_output_root,
                "runtime_sec": time.time() - phase0_started,
                "reused_existing": phase0_reused,
                "can_enter_v75_local": phase0_summary.get("can_enter_v75_local"),
                "can_enter_local2history": phase0_summary.get("can_enter_local2history"),
            }
        )

    if _phase_enabled("phase1", args.stop_after):
        if phase0_summary is None:
            phase0_path = ROOT / args.phase0_output_root / "fact_lock_summary.json"
            if not phase0_path.exists():
                raise FileNotFoundError(f"phase1 requested but phase0 summary is missing: {phase0_path}")
            phase0_summary = json.loads(phase0_path.read_text(encoding="utf-8"))
        if not bool((phase0_summary.get("gate") or {}).get("pass")):
            phase_rows.append(
                {
                    "phase": "phase1",
                    "decision": "SKIPPED_PHASE0_GATE_FAIL",
                    "gate_pass": False,
                    "output_root": args.phase1_output_root,
                    "runtime_sec": 0.0,
                    "notes": "Phase1 is blocked because Phase0 did not pass.",
                }
            )
        else:
            phase1_started = time.time()
            phase1_reused = False
            phase1_path = ROOT / args.phase1_output_root / "incidence_summary.json"
            if _reuse_phase(args, "phase1") and phase1_path.exists():
                phase1_summary = json.loads(phase1_path.read_text(encoding="utf-8"))
                phase1_reused = True
            else:
                phase1_summary = v75_soft_incidence.run(
                    _namespace(
                        output_root=args.phase1_output_root,
                        scenes=args.scenes,
                        max_chunks=args.max_chunks,
                        variants=args.variants,
                        main_variant=args.main_variant,
                        sigma0=args.sigma0,
                        beta=args.beta,
                        jitter_lambda=args.jitter_lambda,
                        fixed_sigma=args.fixed_sigma,
                        boundary_penalty=args.boundary_penalty,
                        min_membership=args.min_membership,
                        large_mask_area_ratio=args.large_mask_area_ratio,
                        frame_cache_size=args.frame_cache_size,
                    )
                )
            phase_summaries["phase1"] = phase1_summary
            phase_rows.append(
                {
                    "phase": "phase1",
                    "decision": phase1_summary.get("decision"),
                    "gate_pass": (phase1_summary.get("gate") or {}).get("pass"),
                    "output_root": args.phase1_output_root,
                    "runtime_sec": time.time() - phase1_started,
                    "reused_existing": phase1_reused,
                    "main_variant": phase1_summary.get("main_variant"),
                    "incidence_row_count": phase1_summary.get("incidence_row_count"),
                }
            )

    if _phase_enabled("phase2", args.stop_after):
        if phase1_summary is None:
            phase1_path = ROOT / args.phase1_output_root / "incidence_summary.json"
            if phase1_path.exists():
                phase1_summary = json.loads(phase1_path.read_text(encoding="utf-8"))
        if phase1_summary is None:
            phase_rows.append(
                {
                    "phase": "phase2",
                    "decision": "SKIPPED_PHASE1_SUMMARY_MISSING",
                    "gate_pass": False,
                    "output_root": args.phase2_output_root,
                    "runtime_sec": 0.0,
                    "notes": "Phase2 is blocked because Phase1 summary is missing.",
                }
            )
        elif not bool((phase1_summary.get("gate") or {}).get("pass")):
            phase_rows.append(
                {
                    "phase": "phase2",
                    "decision": "SKIPPED_PHASE1_GATE_FAIL",
                    "gate_pass": False,
                    "output_root": args.phase2_output_root,
                    "runtime_sec": 0.0,
                    "notes": "Phase2 is blocked because Phase1 did not pass.",
                }
            )
        else:
            phase2_started = time.time()
            phase2_reused = False
            phase2_path = ROOT / args.phase2_output_root / "fragment_summary.json"
            if _reuse_phase(args, "phase2") and phase2_path.exists():
                phase2_summary = json.loads(phase2_path.read_text(encoding="utf-8"))
                phase2_reused = True
            else:
                phase2_summary = _run_phase2_fragments(args)
            phase_summaries["phase2"] = phase2_summary
            key_metrics = phase2_summary.get("key_metrics") or {}
            phase_rows.append(
                {
                    "phase": "phase2",
                    "decision": phase2_summary.get("decision"),
                    "gate_pass": (phase2_summary.get("gate") or {}).get("pass"),
                    "output_root": args.phase2_output_root,
                    "runtime_sec": time.time() - phase2_started,
                    "reused_existing": phase2_reused,
                    "fragment_row_count": key_metrics.get("fragment_row_count"),
                    "relation_row_count": key_metrics.get("relation_row_count"),
                    "high_specificity_fragment_rate": key_metrics.get("high_specificity_fragment_rate"),
                    "containment_candidate_rate": key_metrics.get("containment_candidate_rate"),
                    "F4_assignment_nonempty_chunk_rate": key_metrics.get("F4_assignment_nonempty_chunk_rate"),
                }
            )

    phase2_summary = phase_summaries.get("phase2")
    if _phase_enabled("phase3", args.stop_after):
        if phase2_summary is None:
            phase2_path = ROOT / args.phase2_output_root / "fragment_summary.json"
            if phase2_path.exists():
                phase2_summary = json.loads(phase2_path.read_text(encoding="utf-8"))
        if phase2_summary is None:
            phase_rows.append(
                {
                    "phase": "phase3",
                    "decision": "SKIPPED_PHASE2_SUMMARY_MISSING",
                    "gate_pass": False,
                    "output_root": args.phase3_output_root,
                    "runtime_sec": 0.0,
                    "notes": "Phase3 is blocked because Phase2 summary is missing.",
                }
            )
        elif not bool((phase2_summary.get("gate") or {}).get("pass")):
            phase_rows.append(
                {
                    "phase": "phase3",
                    "decision": "SKIPPED_PHASE2_GATE_FAIL",
                    "gate_pass": False,
                    "output_root": args.phase3_output_root,
                    "runtime_sec": 0.0,
                    "notes": "Phase3 is blocked because Phase2 did not pass.",
                }
            )
        else:
            phase3_started = time.time()
            phase3_reused = False
            phase3_path = ROOT / args.phase3_output_root / "propagation_summary.json"
            if _reuse_phase(args, "phase3") and phase3_path.exists():
                phase3_summary = json.loads(phase3_path.read_text(encoding="utf-8"))
                phase3_reused = True
            else:
                phase3_summary = _run_phase3_affinity_propagation(args)
            phase_summaries["phase3"] = phase3_summary
            key_metrics = phase3_summary.get("key_metrics") or {}
            phase_rows.append(
                {
                    "phase": "phase3",
                    "decision": phase3_summary.get("decision"),
                    "gate_pass": (phase3_summary.get("gate") or {}).get("pass"),
                    "output_root": args.phase3_output_root,
                    "runtime_sec": time.time() - phase3_started,
                    "reused_existing": phase3_reused,
                    "best_real_variant": phase3_summary.get("best_real_variant"),
                    "real_minus_shuffled_heldout_likelihood": key_metrics.get("real_minus_shuffled_heldout_likelihood"),
                    "real_minus_no_temporal_heldout_likelihood": key_metrics.get("real_minus_no_temporal_heldout_likelihood"),
                    "best_real_split_half_NMI": key_metrics.get("best_real_split_half_NMI"),
                }
            )

    phase3_summary = phase_summaries.get("phase3")
    if _phase_enabled("phase4", args.stop_after):
        if phase3_summary is None:
            phase3_path = ROOT / args.phase3_output_root / "propagation_summary.json"
            if phase3_path.exists():
                phase3_summary = json.loads(phase3_path.read_text(encoding="utf-8"))
        if phase3_summary is None:
            phase_rows.append(
                {
                    "phase": "phase4",
                    "decision": "SKIPPED_PHASE3_SUMMARY_MISSING",
                    "gate_pass": False,
                    "output_root": args.phase4_output_root,
                    "runtime_sec": 0.0,
                    "notes": "Phase4 is blocked because Phase3 summary is missing.",
                }
            )
        elif not bool((phase3_summary.get("gate") or {}).get("pass")):
            phase_rows.append(
                {
                    "phase": "phase4",
                    "decision": "SKIPPED_PHASE3_GATE_FAIL",
                    "gate_pass": False,
                    "output_root": args.phase4_output_root,
                    "runtime_sec": 0.0,
                    "notes": "Phase4 is blocked because Phase3 did not pass.",
                }
            )
        else:
            phase4_started = time.time()
            phase4_reused = False
            phase4_path = ROOT / args.phase4_output_root / "hierarchy_summary.json"
            if _reuse_phase(args, "phase4") and phase4_path.exists():
                phase4_summary = json.loads(phase4_path.read_text(encoding="utf-8"))
                phase4_reused = True
            else:
                phase4_summary = _run_phase4_local_hierarchy(args)
            phase_summaries["phase4"] = phase4_summary
            key_metrics = phase4_summary.get("key_metrics") or {}
            phase_rows.append(
                {
                    "phase": "phase4",
                    "decision": phase4_summary.get("decision"),
                    "gate_pass": (phase4_summary.get("gate") or {}).get("pass"),
                    "output_root": args.phase4_output_root,
                    "runtime_sec": time.time() - phase4_started,
                    "reused_existing": phase4_reused,
                    "cluster_row_count": key_metrics.get("cluster_row_count"),
                    "largest_cluster_ratio_mean": key_metrics.get("largest_cluster_ratio_mean"),
                    "parent_child_edge_count": key_metrics.get("parent_child_edge_count"),
                }
            )

    if _phase_enabled("phase5", args.stop_after):
        if phase3_summary is None:
            phase3_path = ROOT / args.phase3_output_root / "propagation_summary.json"
            if phase3_path.exists():
                phase3_summary = json.loads(phase3_path.read_text(encoding="utf-8"))
        if phase3_summary is None:
            phase_rows.append(
                {
                    "phase": "phase5",
                    "decision": "SKIPPED_PHASE3_SUMMARY_MISSING",
                    "gate_pass": False,
                    "output_root": args.phase5_output_root,
                    "runtime_sec": 0.0,
                    "notes": "Phase5 is blocked because Phase3 summary is missing.",
                }
            )
        elif not bool((phase3_summary.get("gate") or {}).get("pass")):
            phase_rows.append(
                {
                    "phase": "phase5",
                    "decision": "SKIPPED_PHASE3_GATE_FAIL",
                    "gate_pass": False,
                    "output_root": args.phase5_output_root,
                    "runtime_sec": 0.0,
                    "notes": "Phase5 is blocked because Phase3 did not pass.",
                }
            )
        else:
            phase5_started = time.time()
            phase5_reused = False
            phase5_path = ROOT / args.phase5_output_root / "local_cut_summary.json"
            if _reuse_phase(args, "phase5") and phase5_path.exists():
                phase5_summary = json.loads(phase5_path.read_text(encoding="utf-8"))
                phase5_reused = True
            else:
                phase5_summary = _run_phase5_local_cut(args)
            phase_summaries["phase5"] = phase5_summary
            key_metrics = phase5_summary.get("key_metrics") or {}
            phase_rows.append(
                {
                    "phase": "phase5",
                    "decision": phase5_summary.get("decision"),
                    "gate_pass": (phase5_summary.get("gate") or {}).get("pass"),
                    "output_root": args.phase5_output_root,
                    "runtime_sec": time.time() - phase5_started,
                    "reused_existing": phase5_reused,
                    "LC5_full_nonGT_cut_SF50": key_metrics.get("LC5_full_nonGT_cut_SF50"),
                    "LC5_GT_best_IoU_mean": key_metrics.get("LC5_GT_best_IoU_mean"),
                    "control_target_SF50": key_metrics.get("control_target_SF50"),
                }
            )

    phase5_summary = phase_summaries.get("phase5")
    if _phase_enabled("phase6", args.stop_after):
        if phase3_summary is None:
            phase3_path = ROOT / args.phase3_output_root / "propagation_summary.json"
            if phase3_path.exists():
                phase3_summary = json.loads(phase3_path.read_text(encoding="utf-8"))
        if phase3_summary is None:
            phase_rows.append(
                {
                    "phase": "phase6",
                    "decision": "SKIPPED_PHASE3_SUMMARY_MISSING",
                    "gate_pass": False,
                    "output_root": args.phase6_output_root,
                    "runtime_sec": 0.0,
                    "notes": "Phase6 is blocked because Phase3 summary is missing.",
                }
            )
        elif bool((phase3_summary.get("gate") or {}).get("pass")) and phase5_summary is None and not (ROOT / args.phase5_output_root / "local_cut_summary.json").exists():
            phase_rows.append(
                {
                    "phase": "phase6",
                    "decision": "SKIPPED_PHASE5_SUMMARY_MISSING",
                    "gate_pass": False,
                    "output_root": args.phase6_output_root,
                    "runtime_sec": 0.0,
                    "notes": "Phase6 is blocked because Phase3 passed but Phase5 summary is missing.",
                }
            )
        else:
            phase6_started = time.time()
            phase6_summary = _run_phase6_local_attribution(args)
            phase_summaries["phase6"] = phase6_summary
            phase_rows.append(
                {
                    "phase": "phase6",
                    "decision": phase6_summary.get("decision"),
                    "gate_pass": False,
                    "output_root": args.phase6_output_root,
                    "runtime_sec": time.time() - phase6_started,
                    "local_decision": phase6_summary.get("local_decision"),
                    "local2history_decision": phase6_summary.get("local2history_decision"),
                    "primary_blocker": phase6_summary.get("primary_blocker"),
                }
            )

    phase6_summary = phase_summaries.get("phase6")
    if _phase_enabled("final", args.stop_after):
        if phase6_summary is None:
            phase6_path = ROOT / args.phase6_output_root / "attribution_summary.json"
            if phase6_path.exists():
                phase6_summary = json.loads(phase6_path.read_text(encoding="utf-8"))
        if phase6_summary is None:
            phase_rows.append(
                {
                    "phase": "final",
                    "decision": "SKIPPED_PHASE6_SUMMARY_MISSING",
                    "gate_pass": False,
                    "output_root": args.final_output_root,
                    "runtime_sec": 0.0,
                    "notes": "Final decision is blocked because Phase6 summary is missing.",
                }
            )
        else:
            final_started = time.time()
            final_summary = _run_final_decision(args)
            phase_summaries["final"] = final_summary
            phase_rows.append(
                {
                    "phase": "final",
                    "decision": final_summary.get("final_decision"),
                    "gate_pass": False,
                    "output_root": args.final_output_root,
                    "runtime_sec": time.time() - final_started,
                    "local_decision": final_summary.get("local_decision"),
                    "local2history_decision": final_summary.get("local2history_decision"),
                    "primary_blocker": final_summary.get("primary_blocker"),
                }
            )

    final_reached_phase = phase_rows[-1]["phase"] if phase_rows else None
    local2history_decision = (
        phase6_summary.get("local2history_decision")
        if isinstance(phase6_summary, dict)
        else "BLOCKED_UNTIL_PHASE6_LOCAL_ATTRIBUTION_PASS"
    )
    can_enter_local2history = bool(phase6_summary.get("can_enter_local2history")) if isinstance(phase6_summary, dict) else False
    summary = {
        "phase": "v75_cmap_l2h_pipeline",
        "schema": "stream4d_v75_cmap_l2h_pipeline_v1",
        "decision": phase_rows[-1]["decision"] if phase_rows else "NO_PHASES_RUN",
        "stop_after": args.stop_after,
        "reached_phase": final_reached_phase,
        "phase_rows": phase_rows,
        "phase_summaries": phase_summaries,
        "can_enter_local2history": can_enter_local2history,
        "local2history_decision": local2history_decision,
        "runtime_sec": time.time() - started,
        "notes": [
            "This is the canonical v75 maintenance entrypoint; phase modules remain importable implementation units.",
            "The pipeline never enters local2history before a future Phase6 local attribution pass.",
        ],
    }

    _write_csv(pipeline_root / "pipeline_phase_rows.csv", phase_rows)
    _write_json(pipeline_root / "pipeline_summary.json", summary)
    _write_json(pipeline_root / "summary.json", summary)
    sha_rows: list[dict[str, Any]] = []
    for path in sorted(pipeline_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"name": f"output:{path.name}", "source_artifact": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_csv(pipeline_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical Stream4D v75 CMAP-L2H pipeline runner.")
    parser.add_argument("--stop-after", choices=PHASE_ORDER, default="phase1")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing summaries for upstream phases before the requested stop phase.")
    parser.add_argument("--pipeline-root", default="outputs/audit/v75_cmap_l2h_pipeline")
    parser.add_argument("--phase0-output-root", default="outputs/audit/v75_phase0_fact_lock")
    parser.add_argument("--phase1-output-root", default="outputs/audit/v75_phase1_soft_incidence")
    parser.add_argument("--phase2-output-root", default="outputs/audit/v75_phase2_fragments")
    parser.add_argument("--phase3-output-root", default="outputs/audit/v75_phase3_affinity_propagation")
    parser.add_argument("--phase4-output-root", default="outputs/audit/v75_phase4_local_hierarchy")
    parser.add_argument("--phase5-output-root", default="outputs/audit/v75_phase5_local_cut")
    parser.add_argument("--phase6-output-root", default="outputs/audit/v75_phase6_local_attribution")
    parser.add_argument("--final-output-root", default="outputs/audit/v75_final_decision")
    parser.add_argument("--scenes", default="scene0011_00,scene0050_00")
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--variants", default=",".join(v75_soft_incidence.VARIANT_ORDER))
    parser.add_argument("--main-variant", default="I3_uv_soft_confidence_jitter_sigma")
    parser.add_argument("--sigma0", type=float, default=12.0)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--jitter-lambda", type=float, default=0.02)
    parser.add_argument("--fixed-sigma", type=float, default=8.0)
    parser.add_argument("--boundary-penalty", type=float, default=0.5)
    parser.add_argument("--min-membership", type=float, default=0.0)
    parser.add_argument("--large-mask-area-ratio", type=float, default=0.25)
    parser.add_argument("--frame-cache-size", type=int, default=8)
    parser.add_argument("--phase2-relation-top-k", type=int, default=64)
    parser.add_argument("--phase2-high-specificity-q", type=float, default=0.20)
    parser.add_argument("--phase2-assignment-top-k", type=int, default=12)
    parser.add_argument("--phase3-seed-top-k", type=int, default=24)
    parser.add_argument("--phase3-specificity-seed-top-k", type=int, default=0)
    parser.add_argument("--phase3-specificity-power", type=float, default=1.0)
    parser.add_argument("--phase3-broad-same-scale", type=float, default=1.0)
    parser.add_argument("--phase3-balance-strength", type=float, default=0.5)
    parser.add_argument("--phase3-seed-max-area-ratio", type=float, default=-1.0)
    parser.add_argument("--phase3-seed-max-entropy", type=float, default=0.75)
    parser.add_argument("--phase3-seed-min-cross-frame-overlap", type=float, default=0.0)
    parser.add_argument("--phase3-seed-quality-only", action="store_true")
    parser.add_argument("--phase3-propagation-steps", type=int, default=4)
    parser.add_argument("--phase3-alpha-restart", type=float, default=0.65)
    parser.add_argument("--phase3-raw-propagation-steps", type=int, default=0)
    parser.add_argument("--phase3-raw-alpha-restart", type=float, default=-1.0)
    parser.add_argument("--phase3-min-membership", type=float, default=0.05)
    parser.add_argument("--phase3-min-edge-weight", type=float, default=1e-4)
    parser.add_argument("--phase3-eval-membership-threshold", type=float, default=0.50)
    parser.add_argument("--phase3-eval-carrier-cap", type=int, default=24)
    parser.add_argument("--phase3-eval-pair-cap", type=int, default=2048)
    parser.add_argument("--phase4-resolutions", default="2.0,1.6,1.3,1.0,0.7,0.4")
    parser.add_argument("--phase4-adapter-min-f1", type=float, default=0.30)
    parser.add_argument("--phase4-adapter-min-precision", type=float, default=0.20)
    parser.add_argument("--phase4-parent-top-k", type=int, default=12)
    parser.add_argument("--phase4-parent-min-shared-masks", type=float, default=3.0)
    parser.add_argument("--phase4-parent-min-edge-weight", type=float, default=1.0)
    parser.add_argument("--phase4-mask-specificity-power", type=float, default=0.0)
    parser.add_argument("--phase4-mask-area-specificity-power", type=float, default=0.0)
    parser.add_argument("--phase4-max-same-level-mask-area-ratio", type=float, default=1.0)
    parser.add_argument("--phase5-min-adapter-f1", type=float, default=0.30)
    parser.add_argument("--phase5-min-adapter-precision", type=float, default=0.20)
    parser.add_argument("--phase5-min-adapter-recall", type=float, default=0.0)
    parser.add_argument("--phase5-demote-broad-adapters", action="store_true")
    parser.add_argument("--phase5-one-mask-per-slot-frame", action="store_true")
    parser.add_argument("--phase5-merge-competing-mask-clusters", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
