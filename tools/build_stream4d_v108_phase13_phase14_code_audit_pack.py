#!/usr/bin/env python3
"""Build a compact Stream4D v108 user-review guard audit package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "code_audit_pack"


EXACT_FILES = [
    "Stream3D/geometry_provider/lingbot_map_provider.py",
    "Stream3D/stream4d/lingbot_map_stream3d_geometry_adapter.py",
    "tools/audit_v106_sam2_rolling_state.py",
    "tools/build_v108_phase13_scaling_summary.py",
    "tools/build_v108_phase14_ablation_summary.py",
    "tools/build_v108_phase14_anchor_durability_audit.py",
    "tools/build_v108_phase19_review_preflight_packet.py",
    "tools/build_v108_phase23_current_hash_probe.py",
    "tools/build_v108_phase24_duplicate_manifest_probe.py",
    "tools/build_v108_phase25_unexpected_key_probe.py",
    "tools/build_v108_phase26_partial_acceptance_probe.py",
    "tools/build_v108_phase27_attestation_probe.py",
    "tools/run_v108_phase28_policy_attestation_probe.py",
    "tools/run_v108_phase30_policy_flag_flow_audit.py",
    "tools/run_v108_phase31_review_manifest_verifier.py",
    "tools/build_v108_phase32_user_review_handoff.py",
    "tools/run_v108_phase33_completion_audit.py",
    "tools/build_stream4d_v108_phase13_phase14_code_audit_pack.py",
    "tools/run_v106_stateful_sam2_rolling_scene_stream.py",
    "tools/run_v107_phase5_prompt_capsule_visibility_probe.py",
    "tools/run_v107_phase7_lingbot_sam2_prompt_benchmark.py",
    "tools/run_v107_phase8_g3_rolling_scheduler_smoke.py",
    "tools/run_v107_phase8_sam2_live_state_reactivation_probe.py",
    "tools/run_v108_phase0_fact_lock.py",
    "tools/run_v108_phase1_label_parity_check.py",
    "tools/run_v108_phase2_online_event_shadow.py",
    "tools/run_v108_phase3_appearance_benchmark.py",
    "tools/run_v108_phase4_gap_multiseed_sam2_shadow.py",
    "tools/run_v108_phase6_probation_watcher_shadow.py",
    "tools/run_v108_phase7_visual_admission_audit.py",
    "tools/run_v108_phase8_sparse_transaction_shadow.py",
    "tools/run_v108_phase9_growth_repair_shadow.py",
    "tools/run_v108_phase10_2d_reactivation_shadow.py",
    "tools/run_v108_phase11_lingbot_geometry_reactivation_shadow.py",
    "tools/run_v108_phase12_full_online.py",
    "tools/run_v108_phase12_integration_readiness.py",
    "tools/run_v108_phase20_user_review_activation_preflight.py",
    "tools/run_v108_phase21_user_review_guard_controls.py",
    "docs/stream4d_v106_sequential_streaming_sam2_objectlet_plan.md",
    "docs/stream4d_v106_执行日志.md",
    "docs/stream4d_v106_实验结果复盘.md",
    "docs/stream4d_v107_recoverability_aware_lifecycle_memory_plan.md",
    "docs/stream4d_v107_执行日志.md",
    "docs/stream4d_v107_实验结果复盘.md",
    "docs/stream4d_v108_dualplane_lifecycle_physical_gap_plan.md",
    "docs/stream4d_v108_执行日志.md",
    "docs/stream4d_v108_实验结果复盘.md",
    "Stream3D/outputs/audit/v108_phase13_scaling_scene0011_90_180_360_475_20260714_2310/phase13_scaling_summary.json",
    "Stream3D/outputs/audit/v108_phase13_scaling_scene0011_90_180_360_475_20260714_2310/phase13_scaling_rows.csv",
    "Stream3D/outputs/audit/v108_phase13_scaling_scene0011_90_180_360_475_20260714_2310/phase13_scaling_summary.md",
    "Stream3D/outputs/audit/v108_phase14_ablation_summary_scene0011_full90_20260714_2342/phase14_ablation_summary.json",
    "Stream3D/outputs/audit/v108_phase14_ablation_summary_scene0011_full90_20260714_2342/phase14_ablation_rows.csv",
    "Stream3D/outputs/audit/v108_phase14_ablation_summary_scene0011_full90_20260714_2342/phase14_ablation_summary.md",
    "Stream3D/outputs/audit/v108_phase14r_ablation_summary_scene0011_full90_20260715_0038/phase14_ablation_summary.json",
    "Stream3D/outputs/audit/v108_phase14r_ablation_summary_scene0011_full90_20260715_0038/phase14_ablation_rows.csv",
    "Stream3D/outputs/audit/v108_phase14r_ablation_summary_scene0011_full90_20260715_0038/phase14_ablation_summary.md",
    "Stream3D/outputs/audit/v108_phase14r_ablation_summary_scene0050_full99_20260715_0120/phase14_ablation_summary.json",
    "Stream3D/outputs/audit/v108_phase14r_ablation_summary_scene0050_full99_20260715_0120/phase14_ablation_rows.csv",
    "Stream3D/outputs/audit/v108_phase14r_ablation_summary_scene0050_full99_20260715_0120/phase14_ablation_summary.md",
    "Stream3D/outputs/audit/v108_phase14r_anchor_durability_scene0050_full99_20260715_0135/anchor_durability_summary.json",
    "Stream3D/outputs/audit/v108_phase14r_anchor_durability_scene0050_full99_20260715_0135/anchor_durability_rows.csv",
    "Stream3D/outputs/audit/v108_phase14r_anchor_durability_scene0050_full99_20260715_0135/anchor_durability_rows.json",
    "Stream3D/outputs/audit/v108_phase14r_anchor_durability_scene0050_full99_20260715_0135/anchor_durability_summary.md",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0050_pilot_baseline_20260715_0135/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0050_pilot_baseline_20260715_0135/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0050_pilot_baseline_20260715_0135/v107_online_runner/g3_scheduler_summary.json",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0050_pilot_baseline_20260715_0135/v107_online_runner/g3_scheduler_records.csv",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0050_pilot_random_geometry_20260715_0135/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0050_pilot_random_geometry_20260715_0135/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0050_pilot_random_geometry_20260715_0135/v107_online_runner/g3_scheduler_summary.json",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0050_pilot_random_geometry_20260715_0135/v107_online_runner/g3_scheduler_records.csv",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0030_pilot_baseline_20260715_0140/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0030_pilot_baseline_20260715_0140/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0030_pilot_baseline_20260715_0140/v107_online_runner/g3_scheduler_summary.json",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0030_pilot_baseline_20260715_0140/v107_online_runner/g3_scheduler_records.csv",
    "Stream3D/outputs/audit/v108_phase16_physical_ready_scene0011_full90_baseline_20260715_0146/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase16_physical_ready_scene0011_full90_baseline_20260715_0146/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase16_physical_ready_scene0011_full90_baseline_20260715_0146/v107_online_runner/g3_scheduler_summary.json",
    "Stream3D/outputs/audit/v108_phase16_physical_ready_scene0011_full90_baseline_20260715_0146/v107_online_runner/g3_scheduler_records.csv",
    "Stream3D/outputs/audit/v108_phase16_physical_ready_scene0011_full90_baseline_20260715_0146/v107_online_runner/highres_probation_visuals/event000_probation_G1_pos_f5_ref5_live4.jpg",
    "Stream3D/outputs/audit/v108_phase16_physical_ready_scene0011_full90_baseline_20260715_0146/v107_online_runner/highres_event_visuals/event000_G1_pos_all_prompts_confirm_f10_ref5_live4.jpg",
    "Stream3D/outputs/audit/v108_phase16_physical_ready_scene0030_full90_baseline_20260715_0146/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase16_physical_ready_scene0030_full90_baseline_20260715_0146/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase16_physical_ready_scene0030_full90_baseline_20260715_0146/v107_online_runner/g3_scheduler_summary.json",
    "Stream3D/outputs/audit/v108_phase16_physical_ready_scene0030_full90_baseline_20260715_0146/v107_online_runner/g3_scheduler_records.csv",
    "Stream3D/outputs/audit/v108_phase16_physical_ready_scene0030_full90_baseline_20260715_0146/v107_online_runner/highres_probation_visuals/event003_probation_G1_pos_f5_ref7_live6.jpg",
    "Stream3D/outputs/audit/v108_phase16_physical_ready_scene0030_full90_baseline_20260715_0146/v107_online_runner/highres_event_visuals/event003_G1_pos_all_prompts_confirm_f10_ref7_live6.jpg",
    "Stream3D/outputs/audit/v108_phase16b_visual_consistency_scene0011_full90_baseline_20260715_0210/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase16b_visual_consistency_scene0011_full90_baseline_20260715_0210/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase16b_visual_consistency_scene0011_full90_baseline_20260715_0210/phase12_visual_review_index.csv",
    "Stream3D/outputs/audit/v108_phase16b_visual_consistency_scene0011_full90_baseline_20260715_0210/phase12_casebook/casebook.md",
    "Stream3D/outputs/audit/v108_phase16b_visual_consistency_scene0011_full90_baseline_20260715_0210/phase12_casebook/casebook_manifest.json",
    "Stream3D/outputs/audit/v108_phase16b_visual_consistency_scene0011_full90_baseline_20260715_0210/phase12_highres_visual_review/phase12_case_00_event000_f000005_live0004.png",
    "Stream3D/outputs/audit/v108_phase16b_visual_consistency_scene0030_full90_baseline_20260715_0210/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase16b_visual_consistency_scene0030_full90_baseline_20260715_0210/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase16b_visual_consistency_scene0030_full90_baseline_20260715_0210/phase12_visual_review_index.csv",
    "Stream3D/outputs/audit/v108_phase16b_visual_consistency_scene0030_full90_baseline_20260715_0210/phase12_casebook/casebook.md",
    "Stream3D/outputs/audit/v108_phase16b_visual_consistency_scene0030_full90_baseline_20260715_0210/phase12_casebook/casebook_manifest.json",
    "Stream3D/outputs/audit/v108_phase16b_visual_consistency_scene0030_full90_baseline_20260715_0210/phase12_highres_visual_review/phase12_case_00_event003_f000005_live0006.png",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0011_full90_baseline_20260715_0220/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0011_full90_baseline_20260715_0220/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0011_full90_baseline_20260715_0220/phase12_visual_review_index.csv",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0011_full90_baseline_20260715_0220/phase12_casebook/casebook.md",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0011_full90_baseline_20260715_0220/phase12_casebook/casebook_manifest.json",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0011_full90_baseline_20260715_0220/phase12_highres_visual_review/phase12_case_00_event000_f000005_live0004.png",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0030_full90_baseline_20260715_0220/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0030_full90_baseline_20260715_0220/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0030_full90_baseline_20260715_0220/phase12_visual_review_index.csv",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0030_full90_baseline_20260715_0220/phase12_casebook/casebook.md",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0030_full90_baseline_20260715_0220/phase12_casebook/casebook_manifest.json",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0030_full90_baseline_20260715_0220/phase12_highres_visual_review/phase12_case_00_event003_f000005_live0006.png",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_baseline_20260715_0220/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_baseline_20260715_0220/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_baseline_20260715_0220/phase12_visual_review_index.csv",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_baseline_20260715_0220/phase12_casebook/casebook.md",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_baseline_20260715_0220/phase12_casebook/casebook_manifest.json",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_baseline_20260715_0220/phase12_highres_visual_review/phase12_case_00_event006_f004460_live0006.png",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_baseline_20260715_0220/phase12_highres_visual_review/phase12_case_01_event008_f004460_live0003.png",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_baseline_20260715_0220/phase12_highres_visual_review/phase12_case_02_event012_f004470_live0006.png",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_baseline_20260715_0220/phase12_highres_visual_review/phase12_case_03_event013_f004470_live0003.png",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_random_geometry_20260715_0220/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_random_geometry_20260715_0220/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_random_geometry_20260715_0220/phase12_visual_review_index.csv",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_random_geometry_20260715_0220/phase12_casebook/casebook.md",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_random_geometry_20260715_0220/phase12_casebook/casebook_manifest.json",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_random_geometry_20260715_0220/phase12_highres_visual_review/phase12_case_00_event018_f004490_live60000.png",
    "Stream3D/outputs/audit/v108_phase16c_label_decode_scene0050_pilot_random_geometry_20260715_0220/phase12_highres_visual_review/phase12_case_01_event023_f004495_live60000.png",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0050_pilot_baseline_20260715_0135/v107_online_runner/highres_probation_visuals/event006_probation_G2_pos_neg_f4460_ref141_live6.jpg",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0050_pilot_baseline_20260715_0135/v107_online_runner/highres_probation_visuals/event012_probation_G2_pos_neg_f4470_ref141_live6.jpg",
    "Stream3D/outputs/audit/v108_phase15_physical_ready_scene0050_pilot_random_geometry_20260715_0135/v107_online_runner/highres_probation_visuals/event018_probation_G2_pos_neg_f4490_ref130_live60000.jpg",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_baseline_20260715_0235/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_baseline_20260715_0235/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_baseline_20260715_0235/phase12_visual_review_index.csv",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_baseline_20260715_0235/phase12_casebook/casebook.md",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_baseline_20260715_0235/phase12_casebook/casebook_manifest.json",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_baseline_20260715_0235/v107_online_runner/g3_scheduler_summary.json",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_baseline_20260715_0235/v107_online_runner/g3_scheduler_records.csv",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_baseline_20260715_0235/phase12_highres_visual_review/phase12_case_01_event008_f004460_live0084.png",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_baseline_20260715_0235/phase12_highres_visual_review/phase12_case_03_event013_f004470_live0084.png",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_baseline_20260715_0235/v107_online_runner/highres_probation_visuals/event008_probation_G1_pos_f4460_ref129_live84.jpg",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_baseline_20260715_0235/v107_online_runner/highres_probation_visuals/event013_probation_G1_pos_f4470_ref129_live84.jpg",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_baseline_20260715_0235/v107_online_runner/highres_event_visuals/event008_G1_pos_all_prompts_confirm_f4465_ref129_live84.jpg",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_baseline_20260715_0235/v107_online_runner/highres_event_visuals/event013_G1_pos_all_prompts_confirm_f4475_ref129_live84.jpg",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_random_geometry_20260715_0235/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_random_geometry_20260715_0235/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_random_geometry_20260715_0235/phase12_visual_review_index.csv",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_random_geometry_20260715_0235/phase12_casebook/casebook.md",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_random_geometry_20260715_0235/phase12_casebook/casebook_manifest.json",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_random_geometry_20260715_0235/v107_online_runner/g3_scheduler_summary.json",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_random_geometry_20260715_0235/v107_online_runner/g3_scheduler_records.csv",
    "Stream3D/outputs/audit/v108_phase17_object_candidate_scene0050_full99_random_geometry_20260715_0235/phase12_highres_visual_review/phase12_case_00_event020_f004490_live60000.png",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_baseline_20260715_0240/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_baseline_20260715_0240/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_baseline_20260715_0240/phase12_visual_review_index.csv",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_baseline_20260715_0240/phase12_lifecycle_admission_rows.csv",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_baseline_20260715_0240/phase12_transaction_boundary_rows.csv",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_baseline_20260715_0240/phase12_casebook/casebook.md",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_baseline_20260715_0240/phase12_casebook/casebook_manifest.json",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_baseline_20260715_0240/phase12_highres_visual_review/phase12_case_01_event008_f004460_live0084.png",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_baseline_20260715_0240/phase12_highres_visual_review/phase12_case_03_event013_f004470_live0084.png",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_random_geometry_20260715_0240/phase12_full_online_summary.json",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_random_geometry_20260715_0240/phase12_visual_review_index.json",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_random_geometry_20260715_0240/phase12_visual_review_index.csv",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_random_geometry_20260715_0240/phase12_lifecycle_admission_rows.csv",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_random_geometry_20260715_0240/phase12_transaction_boundary_rows.csv",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_random_geometry_20260715_0240/phase12_casebook/casebook.md",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_random_geometry_20260715_0240/phase12_casebook/casebook_manifest.json",
    "Stream3D/outputs/audit/v108_phase18d_temporal_physical_readiness_scene0050_full99_random_geometry_20260715_0240/phase12_highres_visual_review/phase12_case_00_event020_f004490_live60000.png",
]


GLOB_FILES = [
    "Stream3D/stream4d_v108/*.py",
    "configs/v106/*.yaml",
    "Stream3D/outputs/audit/v108_phase19_review_preflight_scene0050_full99_20260715_0250/*",
    "Stream3D/outputs/audit/v108_phase19_review_preflight_scene0050_full99_20260715_0250/review_images/*_review_contact_sheet.png",
    "Stream3D/outputs/audit/v108_phase20_user_review_activation_preflight_scene0050_full99_20260715_0305/*",
    "Stream3D/outputs/audit/v108_phase20_user_review_activation_preflight_pending_manifest_scene0050_full99_20260715_0305/*",
    "Stream3D/outputs/audit/v108_phase21_user_review_guard_controls_scene0050_full99_20260715_0312/*",
    "Stream3D/outputs/audit/v108_phase21_user_review_guard_controls_scene0050_full99_20260715_0312/control_manifests/*",
    "Stream3D/outputs/audit/v108_phase21_user_review_guard_controls_scene0050_full99_20260715_0312/phase20_control_runs/*/*",
    "Stream3D/outputs/audit/v108_phase22_manifest_level_guard_pre_repair_probe_scene0050_full99_20260715_0318/*",
    "Stream3D/outputs/audit/v108_phase22_manifest_level_guard_pre_repair_probe_scene0050_full99_20260715_0318/phase20_pre_repair_probe/*",
    "Stream3D/outputs/audit/v108_phase22_manifest_level_guard_repair_scene0050_full99_20260715_0318/post_repair_not_real_probe/*",
    "Stream3D/outputs/audit/v108_phase22_manifest_level_guard_controls_scene0050_full99_20260715_0318/*",
    "Stream3D/outputs/audit/v108_phase22_manifest_level_guard_controls_scene0050_full99_20260715_0318/control_manifests/*",
    "Stream3D/outputs/audit/v108_phase22_manifest_level_guard_controls_scene0050_full99_20260715_0318/phase20_control_runs/*/*",
    "Stream3D/outputs/audit/v108_phase23_evidence_current_hash_pre_repair_probe_scene0050_full99_20260715_0330/*",
    "Stream3D/outputs/audit/v108_phase23_evidence_current_hash_pre_repair_probe_scene0050_full99_20260715_0330/phase19_tampered_current_hash/*",
    "Stream3D/outputs/audit/v108_phase23_evidence_current_hash_pre_repair_probe_scene0050_full99_20260715_0330/phase19_tampered_current_hash/review_images/*",
    "Stream3D/outputs/audit/v108_phase23_evidence_current_hash_pre_repair_probe_scene0050_full99_20260715_0330/phase20_pre_repair_probe/*",
    "Stream3D/outputs/audit/v108_phase23_evidence_current_hash_repair_scene0050_full99_20260715_0334/post_repair_current_hash_probe/*",
    "Stream3D/outputs/audit/v108_phase23_evidence_current_hash_template_check_scene0050_full99_20260715_0334/*",
    "Stream3D/outputs/audit/v108_phase23_evidence_current_hash_guard_controls_scene0050_full99_20260715_0334/*",
    "Stream3D/outputs/audit/v108_phase23_evidence_current_hash_guard_controls_scene0050_full99_20260715_0334/control_manifests/*",
    "Stream3D/outputs/audit/v108_phase23_evidence_current_hash_guard_controls_scene0050_full99_20260715_0334/control_phase19_roots/*/*",
    "Stream3D/outputs/audit/v108_phase23_evidence_current_hash_guard_controls_scene0050_full99_20260715_0334/control_phase19_roots/*/review_images/*",
    "Stream3D/outputs/audit/v108_phase23_evidence_current_hash_guard_controls_scene0050_full99_20260715_0334/phase20_control_runs/*/*",
    "Stream3D/outputs/audit/v108_phase24_duplicate_manifest_pre_repair_probe_scene0050_full99_20260715_0345/*",
    "Stream3D/outputs/audit/v108_phase24_duplicate_manifest_pre_repair_probe_scene0050_full99_20260715_0345/phase20_pre_repair_probe/*",
    "Stream3D/outputs/audit/v108_phase24_duplicate_manifest_repair_scene0050_full99_20260715_0345/post_repair_duplicate_probe/*",
    "Stream3D/outputs/audit/v108_phase24_duplicate_manifest_template_check_scene0050_full99_20260715_0345/*",
    "Stream3D/outputs/audit/v108_phase24_duplicate_manifest_guard_controls_scene0050_full99_20260715_0345/*",
    "Stream3D/outputs/audit/v108_phase24_duplicate_manifest_guard_controls_scene0050_full99_20260715_0345/control_manifests/*",
    "Stream3D/outputs/audit/v108_phase24_duplicate_manifest_guard_controls_scene0050_full99_20260715_0345/control_phase19_roots/*/*",
    "Stream3D/outputs/audit/v108_phase24_duplicate_manifest_guard_controls_scene0050_full99_20260715_0345/control_phase19_roots/*/review_images/*",
    "Stream3D/outputs/audit/v108_phase24_duplicate_manifest_guard_controls_scene0050_full99_20260715_0345/phase20_control_runs/*/*",
    "Stream3D/outputs/audit/v108_phase25_unexpected_manifest_key_pre_repair_probe_scene0050_full99_20260715_0355/*",
    "Stream3D/outputs/audit/v108_phase25_unexpected_manifest_key_pre_repair_probe_scene0050_full99_20260715_0355/phase20_pre_repair_probe/*",
    "Stream3D/outputs/audit/v108_phase25_unexpected_manifest_key_repair_scene0050_full99_20260715_0355/post_repair_unexpected_key_probe/*",
    "Stream3D/outputs/audit/v108_phase25_unexpected_manifest_key_template_check_scene0050_full99_20260715_0355/*",
    "Stream3D/outputs/audit/v108_phase25_unexpected_manifest_key_guard_controls_scene0050_full99_20260715_0355/*",
    "Stream3D/outputs/audit/v108_phase25_unexpected_manifest_key_guard_controls_scene0050_full99_20260715_0355/control_manifests/*",
    "Stream3D/outputs/audit/v108_phase25_unexpected_manifest_key_guard_controls_scene0050_full99_20260715_0355/control_phase19_roots/*/*",
    "Stream3D/outputs/audit/v108_phase25_unexpected_manifest_key_guard_controls_scene0050_full99_20260715_0355/control_phase19_roots/*/review_images/*",
    "Stream3D/outputs/audit/v108_phase25_unexpected_manifest_key_guard_controls_scene0050_full99_20260715_0355/phase20_control_runs/*/*",
    "Stream3D/outputs/audit/v108_phase26_partial_acceptance_pre_repair_probe_scene0050_full99_20260715_0405/*",
    "Stream3D/outputs/audit/v108_phase26_partial_acceptance_pre_repair_probe_scene0050_full99_20260715_0405/phase20_pre_repair_missing_second/*",
    "Stream3D/outputs/audit/v108_phase26_partial_acceptance_pre_repair_probe_scene0050_full99_20260715_0405/phase20_pre_repair_second_pending/*",
    "Stream3D/outputs/audit/v108_phase26_partial_acceptance_repair_scene0050_full99_20260715_0405/post_repair_missing_second/*",
    "Stream3D/outputs/audit/v108_phase26_partial_acceptance_repair_scene0050_full99_20260715_0405/post_repair_second_pending/*",
    "Stream3D/outputs/audit/v108_phase26_partial_acceptance_guard_controls_scene0050_full99_20260715_0405/*",
    "Stream3D/outputs/audit/v108_phase26_partial_acceptance_guard_controls_scene0050_full99_20260715_0405/control_manifests/*",
    "Stream3D/outputs/audit/v108_phase26_partial_acceptance_guard_controls_scene0050_full99_20260715_0405/control_phase19_roots/*/*",
    "Stream3D/outputs/audit/v108_phase26_partial_acceptance_guard_controls_scene0050_full99_20260715_0405/control_phase19_roots/*/review_images/*",
    "Stream3D/outputs/audit/v108_phase26_partial_acceptance_guard_controls_scene0050_full99_20260715_0405/phase20_control_runs/*/*",
    "Stream3D/outputs/audit/v108_phase26_partial_acceptance_template_check_scene0050_full99_20260715_0405/*",
    "Stream3D/outputs/audit/v108_phase26_partial_acceptance_allaccepted_schema_check_scene0050_full99_20260715_0405/*",
    "Stream3D/outputs/audit/v108_phase26_partial_acceptance_allaccepted_schema_check_scene0050_full99_20260715_0405/phase20_allaccepted_schema_check/*",
    "Stream3D/outputs/audit/v108_phase27_explicit_attestation_pre_repair_probe_scene0050_full99_20260715_0415/*",
    "Stream3D/outputs/audit/v108_phase27_explicit_attestation_pre_repair_probe_scene0050_full99_20260715_0415/phase20_pre_repair_missing_attestation/*",
    "Stream3D/outputs/audit/v108_phase27_explicit_attestation_pre_repair_probe_scene0050_full99_20260715_0415/phase20_pre_repair_schema_check_provenance/*",
    "Stream3D/outputs/audit/v108_phase27_explicit_attestation_repair_scene0050_full99_20260715_0415/post_repair_missing_attestation/*",
    "Stream3D/outputs/audit/v108_phase27_explicit_attestation_repair_scene0050_full99_20260715_0415/post_repair_schema_check_provenance/*",
    "Stream3D/outputs/audit/v108_phase27_explicit_attestation_template_check_scene0050_full99_20260715_0415/*",
    "Stream3D/outputs/audit/v108_phase27_explicit_attestation_guard_controls_scene0050_full99_20260715_0415/*",
    "Stream3D/outputs/audit/v108_phase27_explicit_attestation_guard_controls_scene0050_full99_20260715_0415/control_manifests/*",
    "Stream3D/outputs/audit/v108_phase27_explicit_attestation_guard_controls_scene0050_full99_20260715_0415/control_phase19_roots/*/*",
    "Stream3D/outputs/audit/v108_phase27_explicit_attestation_guard_controls_scene0050_full99_20260715_0415/control_phase19_roots/*/review_images/*",
    "Stream3D/outputs/audit/v108_phase27_explicit_attestation_guard_controls_scene0050_full99_20260715_0415/phase20_control_runs/*/*",
    "Stream3D/outputs/audit/v108_phase28_policy_attestation_pre_repair_scene0050_full99_20260715_0425/*",
    "Stream3D/outputs/audit/v108_phase28_policy_attestation_repair_scene0050_full99_20260715_0425/*",
    "Stream3D/outputs/audit/v108_phase28_policy_attestation_scaffold_smoke_20260715_0425/*",
    "Stream3D/outputs/audit/v108_phase28_policy_attestation_guard_controls_regression_scene0050_full99_20260715_0425/*",
    "Stream3D/outputs/audit/v108_phase28_policy_attestation_guard_controls_regression_scene0050_full99_20260715_0425/control_manifests/*",
    "Stream3D/outputs/audit/v108_phase28_policy_attestation_guard_controls_regression_scene0050_full99_20260715_0425/control_phase19_roots/*/*",
    "Stream3D/outputs/audit/v108_phase28_policy_attestation_guard_controls_regression_scene0050_full99_20260715_0425/control_phase19_roots/*/review_images/*",
    "Stream3D/outputs/audit/v108_phase28_policy_attestation_guard_controls_regression_scene0050_full99_20260715_0425/phase20_control_runs/*/*",
    "Stream3D/outputs/audit/v108_phase29_attestation_export_template_check_scene0050_full99_20260715_0435/*",
    "Stream3D/outputs/audit/v108_phase29_attestation_export_missing_attestation_scene0050_full99_20260715_0435/*",
    "Stream3D/outputs/audit/v108_phase29_attestation_export_schema_provenance_scene0050_full99_20260715_0435/*",
    "Stream3D/outputs/audit/v108_phase29_attestation_export_guard_controls_scene0050_full99_20260715_0435/*",
    "Stream3D/outputs/audit/v108_phase29_attestation_export_guard_controls_scene0050_full99_20260715_0435/control_manifests/*",
    "Stream3D/outputs/audit/v108_phase29_attestation_export_guard_controls_scene0050_full99_20260715_0435/control_phase19_roots/*/*",
    "Stream3D/outputs/audit/v108_phase29_attestation_export_guard_controls_scene0050_full99_20260715_0435/control_phase19_roots/*/review_images/*",
    "Stream3D/outputs/audit/v108_phase29_attestation_export_guard_controls_scene0050_full99_20260715_0435/phase20_control_runs/*/*",
    "Stream3D/outputs/audit/v108_phase30_policy_flag_flow_audit_20260715_0445/*",
    "Stream3D/outputs/audit/v108_phase31_review_manifest_verifier_scene0050_full99_20260715_0445/*",
    "Stream3D/outputs/audit/v108_phase31_review_manifest_verifier_scene0050_full99_20260715_0445/*/*",
    "Stream3D/outputs/audit/v108_phase31_resume_recheck_scene0050_full99_20260715_0515/*",
    "Stream3D/outputs/audit/v108_phase31_resume_recheck_scene0050_full99_20260715_0515/*/*",
    "Stream3D/outputs/audit/v108_phase32_user_review_handoff_scene0050_full99_20260715_0525/*",
    "Stream3D/outputs/audit/v108_phase32_user_review_handoff_scene0050_full99_20260715_0525/*/*",
    "Stream3D/outputs/audit/v108_phase33_completion_audit_scene0050_full99_20260715_0520/*",
    "Stream3D/outputs/audit/v108_phase33_completion_audit_scene0050_full99_20260715_0520/*/*",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_sources() -> tuple[list[Path], list[str]]:
    files: set[Path] = set()
    missing: list[str] = []
    for rel in EXACT_FILES:
        path = ROOT / rel
        if path.is_file():
            files.add(path)
        else:
            missing.append(rel)
    for pattern in GLOB_FILES:
        matches = sorted(ROOT.glob(pattern))
        if not matches:
            missing.append(pattern)
        for path in matches:
            if path.is_file():
                files.add(path)
    return sorted(files), missing


def copy_payload(files: Iterable[Path], pack_dir: Path) -> list[str]:
    rels: list[str] = []
    for src in files:
        rel = src.relative_to(ROOT)
        dst = pack_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rels.append(rel.as_posix())
    return sorted(rels)


def write_manifest(pack_dir: Path, rels: list[str]) -> None:
    filelist = pack_dir / "FILELIST.txt"
    filelist.write_text("\n".join(rels) + "\n", encoding="utf-8")
    manifest = pack_dir / "MANIFEST.sha256"
    lines = []
    for rel in rels:
        lines.append(f"{sha256_file(pack_dir / rel)}  {rel}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_zip(pack_dir: Path, zip_path: Path) -> list[str]:
    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(pack_dir.rglob("*")):
            if not path.is_file():
                continue
            arc = f"{pack_dir.name}/{path.relative_to(pack_dir).as_posix()}"
            zf.write(path, arc)
            entries.append(arc)
    return entries


def validate(pack_dir: Path, zip_path: Path, payload_rels: list[str]) -> dict[str, object]:
    with zipfile.ZipFile(zip_path) as zf:
        bad_member = zf.testzip()
        zip_entries = sorted(x for x in zf.namelist() if not x.endswith("/"))
    expected = sorted(f"{pack_dir.name}/{rel}" for rel in [*payload_rels, "FILELIST.txt", "MANIFEST.sha256"])
    missing = sorted(set(expected) - set(zip_entries))
    extra = sorted(set(zip_entries) - set(expected))
    return {
        "zip_test_bad_member": bad_member,
        "payload_file_count": len(payload_rels),
        "zip_entry_count": len(zip_entries),
        "entry_parity_ok": not missing and not extra,
        "missing_from_zip": missing,
        "extra_in_zip": extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="stream4d_v108_phase33_completion_audit_core_code_20260715_0520")
    args = parser.parse_args()

    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    pack_dir = PACK_ROOT / args.tag
    zip_path = PACK_ROOT / f"{args.tag}.zip"
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    if zip_path.exists():
        zip_path.unlink()
    pack_dir.mkdir(parents=True)

    files, missing = iter_sources()
    rels = copy_payload(files, pack_dir)
    write_manifest(pack_dir, rels)
    make_zip(pack_dir, zip_path)
    validation = validate(pack_dir, zip_path, rels)

    sha_sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sha_sidecar.write_text(f"{sha256_file(zip_path)}  {zip_path.name}\n", encoding="utf-8")

    summary = {
        "schema": "stream4d_v108_phase33_completion_audit_core_code_audit_pack_v1",
        "tag": args.tag,
        "pack_dir": pack_dir.relative_to(ROOT).as_posix(),
        "zip": zip_path.relative_to(ROOT).as_posix(),
        "zip_sha256": sha256_file(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "missing_requested_sources": missing,
        "validation": validation,
        "note": (
            "Source-focused audit package with Phase18d readiness, Phase19 shadow transaction "
            "preflight artifacts, Phase20 guarded user-review activation preflight artifacts, "
            "Phase21/22 file-level user-review guard controls including manifest-level "
            "provenance guards, and Phase23 current contact-sheet file-hash mismatch guards. "
            "Phase24 adds duplicate visual-review-key rejection and controls. Phase25 adds "
            "unexpected visual-review-key scope rejection and controls. Phase26 blocks partial "
            "accepted coverage from emitting transaction preflight requests. Phase27 requires "
            "an explicit top-level user visual-attestation object bound to the current ready "
            "contact-sheet fingerprint before accepted rows can emit transaction preflight requests. "
            "Phase28 propagates the attestation guard into lower-level lifecycle/growth policies so "
            "accepted status strings alone cannot make durable memory look allowed. Phase29 exports "
            "attestation verification and policy-ready attestation fields separately so lower-level "
            "policies cannot mistake a valid attestation object on a blocked/partial manifest for "
            "a final verified activation. Phase30 audits policy callsites so raw user_attestation_verified "
            "or constant true values cannot be silently passed into lifecycle/growth policy helpers. Phase31 adds "
            "a read-only pre-Phase20 review-manifest verifier that reports whether a manifest may be passed into "
            "Phase20 without constructing transaction requests or applying SAM2 memory. Phase32 adds "
            "a read-only user-review handoff packet with copied current contact sheets, a pending/template "
            "manifest, HTML/Markdown review surfaces, and a Phase31 fail-closed verification of that pending "
            "handoff manifest. Phase33 adds a read-only completion-boundary audit that joins Phase19, "
            "Phase30, Phase31, and Phase32 evidence and reports NOT_ACHIEVED while explicit user visual "
            "acceptance is absent. Metrics are "
            "diagnostic only; durable memory acceptance still requires explicit user visual "
            "acceptance and is not inferred by this package."
        ),
    }
    summary_path = zip_path.with_suffix(zip_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
