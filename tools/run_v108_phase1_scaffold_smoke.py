#!/usr/bin/env python3
"""Smoke-test the Stream4D v108 package scaffold.

This is not a Phase1 parity experiment. It only verifies that the v108 package
boundaries are importable and that core policy guards work without a GPU.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Stream3D.stream4d_v108.artifacts import ArtifactWriter
from Stream3D.stream4d_v108.diagnostics import DiagnosticMetricPolicy
from Stream3D.stream4d_v108.gap_hypothesis_graph import (
    GapCandidateMask,
    GapHypothesis,
    GapHypothesisGraph,
    GapSeed,
)
from Stream3D.stream4d_v108.geometry_capsule import GeometryAnchor, GeometrySupport, ProjectedAnchor
from Stream3D.stream4d_v108.growth_repair import GrowthRepairPlanner
from Stream3D.stream4d_v108.lifecycle import LifecycleState, LifecycleStateMachine
from Stream3D.stream4d_v108.masklet_watcher import MaskletObservation, MaskletWatcher
from Stream3D.stream4d_v108.reactivation import prompt_set_from_projected_anchors
from Stream3D.stream4d_v108.transaction_manager import (
    Plane,
    Sam2MemoryMutationRequest,
    TransactionManager,
)
from Stream3D.stream4d_v108.video_export import CasebookItem


def build_smoke(output_root: Path) -> dict[str, Any]:
    writer = ArtifactWriter(output_root)

    graph = GapHypothesisGraph()
    seed = GapSeed(
        seed_id="seed_000",
        frame_id=4500,
        component_id="component_000",
        uv=(512.0, 384.0),
        distance_to_component_edge_px=24.0,
        distance_to_image_edge_px=80.0,
    )
    graph.add_seed(seed)
    candidate = GapCandidateMask(
        candidate_id="candidate_000",
        frame_id=4500,
        seed_id=seed.seed_id,
        area_px=12000,
        bbox_area_fraction=0.08,
        touches_image_edge=False,
        sam2_multimask_index=0,
    )
    graph.add_candidate(candidate)
    support = GeometrySupport(
        depth_valid_fraction=0.82,
        depth_compactness=0.71,
        plane_like_support=0.44,
        anchor_conflict_count=0,
    )
    hypothesis = GapHypothesis(
        hypothesis_id="hypothesis_000",
        frame_id=4500,
        component_id="component_000",
        candidate_ids=(candidate.candidate_id,),
        geometry_support=support,
        existing_anchor_conflict=False,
        output_allowed=True,
        memory_admission_allowed=False,
        reason="output probation smoke; durable memory intentionally withheld",
    )
    graph.add_hypothesis(hypothesis)

    txm = TransactionManager()
    output_tx = txm.propose(
        frame_id=4500,
        global_object_id=900001,
        plane=Plane.OUTPUT,
        action="emit_probation_mask",
        evidence={"hypothesis_id": hypothesis.hypothesis_id},
    )
    output_tx = txm.apply_output_only(output_tx, "current-frame output only")
    memory_request = Sam2MemoryMutationRequest(
        frame_id=4500,
        global_object_id=900001,
        sam2_runtime_object_id=None,
        mutation="defer_add_new_points_or_box",
        prompt_count=0,
    )
    memory_tx = txm.propose(
        frame_id=4500,
        global_object_id=900001,
        plane=Plane.SAM2_MEMORY,
        action="defer_memory_admission",
        evidence=memory_request.as_transaction_evidence(),
    )
    memory_tx = txm.reject(memory_tx, "not enough evidence for durable memory")

    wrong_plane_guard_triggered = False
    try:
        txm.apply_output_only(memory_tx, "should fail")
    except ValueError:
        wrong_plane_guard_triggered = True

    policy = DiagnosticMetricPolicy()
    metric_row = policy.label_metric("foreground_iou", 0.3118481859832104, "p34 artifact smoke")
    acceptance_guard_triggered = False
    try:
        policy.assert_no_auto_acceptance({"status": "FINAL_ACCEPTED"})
    except ValueError:
        acceptance_guard_triggered = True

    anchor = GeometryAnchor(
        global_object_id=900001,
        source_frame_id=4495,
        point_xyz=(1.0, 2.0, 3.0),
        source_uv=(500.0, 380.0),
        depth_m=3.2,
        distance_to_mask_edge_px=12.0,
    )
    projected = ProjectedAnchor(
        anchor=anchor,
        target_frame_id=4500,
        target_uv=(512.0, 384.0),
        target_depth_m=3.25,
        depth_residual_m=0.05,
        in_frame=True,
        occluded=False,
    )
    prompt_set = prompt_set_from_projected_anchors(4500, 900001, [projected], [])

    watcher = MaskletWatcher(growth_ratio_alert=1.5)
    watcher.observe(MaskletObservation(4495, 900001, 1000, 0.02, True, False))
    growth_alert = watcher.observe(MaskletObservation(4500, 900001, 2500, 0.05, True, False))
    repair = GrowthRepairPlanner().suggest(growth_alert, has_anchor_conflict=False) if growth_alert else None

    lifecycle_event = LifecycleStateMachine().transition(
        frame_id=4500,
        global_object_id=900001,
        from_state=LifecycleState.DORMANT,
        to_state=LifecycleState.OUTPUT_PROBATION,
        reason="gap hypothesis output without memory admission",
    )

    casebook = CasebookItem(
        case_id="case_gap_hypothesis_smoke",
        frame_id=4500,
        category="largest_gap_component_with_one_seed",
        artifact_paths=("gap_hypothesis_rows.jsonl",),
        visible_facts=("single interior seed produced one probation candidate",),
        possible_error_classes=("insufficient physical support",),
    )
    casebook.assert_pending()

    writer.write_jsonl(
        "gap_hypothesis_rows.jsonl",
        [hypothesis],
        "stream4d_v108_gap_hypothesis_rows_v1",
    )
    writer.write_jsonl(
        "transaction_rows.jsonl",
        txm.transactions,
        "stream4d_v108_transaction_rows_v1",
    )
    writer.write_json(
        "casebook_manifest.json",
        {"schema_version": "stream4d_v108_casebook_manifest_v1", "items": [casebook]},
        "stream4d_v108_casebook_manifest_v1",
    )

    summary = {
        "schema_version": "stream4d_v108_phase1_scaffold_smoke_v1",
        "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "v108_experiment_run": False,
        "gpu_used": False,
        "parity_experiment_run": False,
        "wrong_plane_guard_triggered": wrong_plane_guard_triggered,
        "acceptance_guard_triggered": acceptance_guard_triggered,
        "output_memory_plane_separated": output_tx.plane is Plane.OUTPUT and memory_tx.plane is Plane.SAM2_MEMORY,
        "memory_admission_deferred": memory_tx.reason == "not enough evidence for durable memory",
        "metric_row": metric_row,
        "prompt_positive_count": prompt_set.positive_count,
        "prompt_negative_count": prompt_set.negative_count,
        "growth_repair_action": repair.action if repair else None,
        "lifecycle_to_state": lifecycle_event.to_state.value,
        "casebook_status": casebook.review_status.value,
        "artifact_manifest": writer.manifest(),
    }
    writer.write_json("scaffold_smoke_summary.json", summary, "stream4d_v108_phase1_scaffold_smoke_v1")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root. Defaults to Stream3D/outputs/audit/v108_phase1_scaffold_smoke_<UTC timestamp>.",
    )
    args = parser.parse_args()
    if args.output_root:
        output_root = REPO_ROOT / args.output_root
    else:
        tag = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "Stream3D" / "outputs" / "audit" / f"v108_phase1_scaffold_smoke_{tag}"
    output_root.mkdir(parents=True, exist_ok=True)
    summary = build_smoke(output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    required = [
        summary["wrong_plane_guard_triggered"],
        summary["acceptance_guard_triggered"],
        summary["output_memory_plane_separated"],
        summary["memory_admission_deferred"],
        summary["casebook_status"] == "USER_REVIEW_PENDING",
    ]
    return 0 if all(required) else 2


if __name__ == "__main__":
    raise SystemExit(main())
