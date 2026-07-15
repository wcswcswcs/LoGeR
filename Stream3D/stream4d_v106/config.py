from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class RunConfig:
    version: str = "v106"
    scene_stream_sequential: bool = True
    same_scene_chunk_parallelism: int = 1
    allow_different_scene_parallelism: bool = True
    deterministic: bool = True
    seed: int = 0
    cache_mode: str = "readwrite"


@dataclass
class DataConfig:
    chunk_size: int = 32
    overlap: int = 3
    frame_stride: int = 5
    target_height: int = 240
    target_width: int = 320


@dataclass
class SAM2Config:
    checkpoint: str = "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg: str = "Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    image_autocast_dtype: str = "bfloat16"
    video_autocast_dtype: str = "bfloat16"
    model_load_once_per_scene: bool = True


@dataclass
class LocalExactConfig:
    baseline_x_stage1_enabled: bool = True
    baseline_x_stage2_enabled: bool = True
    later_chunk_full_reinitialize: bool = False
    exact_gap_birth_enabled: bool = True
    birth_choice_policy: str = "smallest_valid_mask_per_point"
    repair_choice_policy: str = "max_candidate_support_valid_mask_per_point"


@dataclass
class HandoffConfig:
    policy: str = "best_quality_overlap"
    overlap_prompt_count: int = 1
    use_mask_logits_if_available: bool = True
    runtime_global_id_decoupled: bool = True


@dataclass
class IdentityConfig:
    occlusion_memory_max_chunks: int = 4
    tentative_reappearance_max_chunks: int = 1
    one_to_one_assignment: bool = True
    co_visible_cannot_link: bool = True


@dataclass
class AccelerationConfig:
    shared_frame_feature_bank: bool = True
    video_feature_storage: str = "cuda"
    video_gpu_hot_window: int = 8
    video_state_template_reuse: bool = True
    model_reuse_across_chunks: bool = True


@dataclass
class SpecGapConfig:
    enabled: bool = False
    inherited_only_proxy: bool = True
    max_anchors_per_tube: int = 1
    max_births_per_chunk: int | None = None
    final_exact_residual_rounds: int = 1


@dataclass
class LingBotConfig:
    enabled: bool = True
    mode: str = "shadow"
    affects_mask: bool = False
    affects_gap: bool = False
    affects_identity: bool = False
    affects_gate: bool = False
    include_in_core_latency: bool = False


@dataclass
class EvaluationConfig:
    identity_metrics_enabled: bool = True
    gt_objectlet_diagnostics_enabled: bool = True
    mv_ap_enabled: bool = True
    mv_ap_is_gate: bool = False


@dataclass
class VideoConfig:
    export_mp4: bool = True
    sheets_per_chunk: int = 4
    frames_per_sheet: int = 8
    export_boundary_casebook: bool = True


@dataclass
class Phase2ArtifactConfig:
    enabled: bool = True
    scene_id: str = "scene0050_00"
    b2_summary: str = "Stream3D/outputs/audit/v106_phase2_b2_phase5_exact_scene0050_start4160_20260712_161052/phase5_frozen_birth_replay_summary.json"
    b2_labels_dir: str = "Stream3D/outputs/audit/v106_phase2_b2_phase5_exact_scene0050_start4160_20260712_161052/labels"
    b2_birth_records: str = "Stream3D/outputs/audit/v105_baseline_x_gapadaptive_scene0050_start4160_f32_20260712/birth_bank/birth_records.json"
    b3_summary: str = "Stream3D/outputs/audit/v106_phase2_b3_phase5_featurebank_scene0050_start4160_20260712_161052/phase5_frozen_birth_replay_summary.json"
    b3_labels_dir: str = "Stream3D/outputs/audit/v106_phase2_b3_phase5_featurebank_scene0050_start4160_20260712_161052/labels"
    b4_summary: str = "Stream3D/outputs/audit/v106_phase2_b4_phase5_state_template_scene0050_start4160_20260712_161622/phase5_frozen_birth_replay_summary.json"
    b4_labels_dir: str = "Stream3D/outputs/audit/v106_phase2_b4_phase5_state_template_scene0050_start4160_20260712_161622/labels"
    x0_summary: str = "Stream3D/outputs/audit/v105_repair_min_scene0050_x0_20260712/baseline_x_sam2_twostage_sam2/summary.json"
    x1_summary: str = "Stream3D/outputs/audit/v105_scene0050_fullscene_frame0seed_promptrepair_full931_20260712/x1seed_scene0050_start4160_f32/baseline_x_gapadaptive_sam2/summary.json"


@dataclass
class Phase3HandoffConfig:
    enabled: bool = True
    scene_id: str = "scene0050_00"
    c0_summary: str = "Stream3D/outputs/audit/v106_phase2_b4_phase5_state_template_scene0050_start4160_20260712_161622/phase5_frozen_birth_replay_summary.json"
    c0_labels_dir: str = "Stream3D/outputs/audit/v106_phase2_b4_phase5_state_template_scene0050_start4160_20260712_161622/labels"
    c0_chunk_index: int = 0
    c1_chunk_index: int = 1
    c0_frame_start: int = 4160
    c1_frame_start: int = 4305
    frame_stride: int = 5
    frame_count: int = 32
    overlap: int = 3
    reference_c1_summary: str = ""
    h0_replay_summary: str = ""
    h1_replay_summary: str = ""
    h2_replay_summary: str = ""
    h4_replay_summary: str = ""
    min_ccoc: float = 0.85
    min_hir: float = 0.95
    min_hcr: float = 0.90
    max_cfr: float = 0.05
    max_cmr: float = 0.05
    max_bfmr: float = 0.05
    fragment_overlap_fraction_threshold: float = 0.01
    merge_overlap_fraction_threshold: float = 0.01
    endpoint_drift_area_ratio_min: float = 0.75
    endpoint_drift_area_ratio_max: float = 1.33


@dataclass
class Phase4SceneStateConfig:
    enabled: bool = True
    scene_id: str = "scene0050_00"
    c0_summary: str = "Stream3D/outputs/audit/v106_phase2_b4_phase5_state_template_scene0050_start4160_20260712_161622/phase5_frozen_birth_replay_summary.json"
    c1_summary: str = "Stream3D/outputs/audit/v106_phase3_handoff_h2_earliestprep_20260712_163845/phase3/H2_best_plus_one_correction_phase5_replay/phase5_frozen_birth_replay_summary.json"
    c2_summary: str = "Stream3D/outputs/audit/v106_phase4_c1c2_handoff_prep_20260712_164826/boundary_c1_c2/H2_best_plus_one_correction_phase5_replay/phase5_frozen_birth_replay_summary.json"
    c0_to_c1_handoff_package: str = "Stream3D/outputs/audit/v106_phase3_handoff_h2_earliestprep_20260712_163845/phase3/H2_best_plus_one_correction/handoff_package.json"
    c1_to_c2_handoff_package: str = "Stream3D/outputs/audit/v106_phase4_c1c2_handoff_prep_20260712_164826/boundary_c1_c2/H2_best_plus_one_correction/handoff_package.json"
    c0_to_c1_gate_summary: str = "Stream3D/outputs/audit/v106_phase3_handoff_gate_h2_metricfix_20260712_164156/phase3/phase3_gate_summary.json"
    c1_to_c2_gate_summary: str = "Stream3D/outputs/audit/v106_phase4_c1c2_handoff_gate_20260712_164913/boundary_c1_c2/phase3_gate_summary.json"
    min_ccoc: float = 0.85
    min_hir: float = 0.95
    min_hcr: float = 0.90
    max_cfr: float = 0.05
    max_cmr: float = 0.05
    max_bfmr: float = 0.05


@dataclass
class Phase5RepairBirthDeferConfig:
    enabled: bool = True
    scene_id: str = "scene0050_00"
    frame_start: int = 4450
    frame_stride: int = 5
    frame_count: int = 32
    selected_chunk_indices: str = ""
    inherited_birth_records: str = "Stream3D/outputs/audit/v106_phase4_c1c2_handoff_prep_20260712_164826/boundary_c1_c2/H2_best_plus_one_correction/birth_records.json"
    inherited_replay_summary: str = "Stream3D/outputs/audit/v106_phase4_c1c2_handoff_prep_20260712_164826/boundary_c1_c2/H2_best_plus_one_correction_phase5_replay/phase5_frozen_birth_replay_summary.json"
    reference_birth_records: str = "Stream3D/outputs/audit/v106_phase4_c2_reference_baseline_x_scene0050_start4450_20260712_164629/birth_bank/birth_records.json"
    reference_summary: str = "Stream3D/outputs/audit/v106_phase4_c2_reference_baseline_x_scene0050_start4450_20260712_164629/baseline_x_gapadaptive_sam2/baseline_x_gapadaptive_sam2/summary.json"
    r0_replay_summary: str = ""
    r1_replay_summary: str = ""
    r2_replay_summary: str = ""
    min_area: int = 16
    repair_overlap_coeff: float = 0.55
    duplicate_suppress_overlap_coeff: float = 0.95
    duplicate_suppress_area_ratio_min: float = 0.90
    birth_max_overlap_coeff: float = 0.25
    min_persistence_frames: int = 2
    large_persistent_area: int = 4096
    duplicate_window_frames: int = 3
    duplicate_overlap_threshold: float = 0.55
    required_drr_relative_reduction: float = 0.30
    metric_epsilon: float = 1.0e-9
    fragment_overlap_fraction_threshold: float = 0.01
    merge_overlap_fraction_threshold: float = 0.01


@dataclass
class Phase6OcclusionReappearanceConfig:
    enabled: bool = True
    scene_id: str = "scene0050_00"
    current_chunk_index: int = 2
    previous_chunk_index: int = 1
    c0_summary: str = "Stream3D/outputs/audit/v106_phase2_b4_phase5_state_template_scene0050_start4160_20260712_161622/phase5_frozen_birth_replay_summary.json"
    c1_summary: str = "Stream3D/outputs/audit/v106_phase3_handoff_h2_earliestprep_20260712_163845/phase3/H2_best_plus_one_correction_phase5_replay/phase5_frozen_birth_replay_summary.json"
    scene_state: str = "Stream3D/outputs/audit/v106_phase4_scene_state_gate_20260712_165549/phase4/scene_stream_state.json"
    phase5_birth_records: str = "Stream3D/outputs/audit/v106_phase5_dupsuppress_schedule_20260712_172217/phase5/R1_repair_vs_birth/birth_records.json"
    phase5_classification_records: str = "Stream3D/outputs/audit/v106_phase5_dupsuppress_schedule_20260712_172217/phase5/R1_repair_vs_birth/classification_records.json"
    phase5_replay_summary: str = "Stream3D/outputs/audit/v106_phase5_dupsuppress_schedule_20260712_172217/phase5/R1_repair_vs_birth_phase5_replay/phase5_frozen_birth_replay_summary.json"
    reference_summary: str = "Stream3D/outputs/audit/v106_phase4_c2_reference_baseline_x_scene0050_start4450_20260712_164629/baseline_x_gapadaptive_sam2/baseline_x_gapadaptive_sam2/summary.json"
    rgb_root: str = "Stream3D/data/scannet/processed"
    sam2_baseline_config: str = "configs/v105/baseline_chunk_table/baseline_x_gapadaptive_sam2.generated.yaml"
    sam2_descriptor_cache: str = ""
    use_sam2_descriptors: bool = True
    sam2_descriptor_layer: str = "vision_features"
    sam2_descriptor_model_dtype: str = "bfloat16"
    tau_confirm: float = 0.72
    tau_margin: float = 0.03
    tentative_tau_confirm: float = 0.64
    max_occlusion_age_chunks: int = 4
    metric_epsilon: float = 1.0e-9
    fragment_overlap_fraction_threshold: float = 0.01
    merge_overlap_fraction_threshold: float = 0.01


@dataclass
class Phase7SpecGapParallelConfig:
    enabled: bool = True
    scene_id: str = "scene0050_00"
    integrated_phase0_6_run_summary: str = "Stream3D/outputs/audit/v106_phase0123456_integrated_gate_20260712_175448/run_summary.json"
    phase6_gate_records: str = "Stream3D/outputs/audit/v106_phase0123456_integrated_gate_20260712_175448/phase6/gate_records.json"
    exact_b6_replay_summary: str = "Stream3D/outputs/audit/v106_phase5_dupsuppress_schedule_20260712_172217/phase5/R1_repair_vs_birth_phase5_replay/phase5_frozen_birth_replay_summary.json"
    exact_b6_gate_records: str = "Stream3D/outputs/audit/v106_phase5_repair_birth_defer_gate_dupsuppress_20260712_172451/phase5/gate_records.json"
    candidate_birth_records: str = "Stream3D/outputs/audit/v106_phase5_dupsuppress_schedule_20260712_172217/phase5/R1_repair_vs_birth/birth_records.json"
    candidate_classification_records: str = "Stream3D/outputs/audit/v106_phase5_dupsuppress_schedule_20260712_172217/phase5/R1_repair_vs_birth/classification_records.json"
    s1_filtered_replay_summary: str = ""
    s1_filtered_metric_summary: str = ""
    s2_filtered_replay_summary: str = ""
    s2_filtered_metric_summary: str = ""
    s1_real_replay_summary: str = ""
    s1_real_metric_summary: str = ""
    s1_real_birth_records: str = ""
    s1_real_birth_summary: str = ""
    s2_real_replay_summary: str = ""
    s2_real_metric_summary: str = ""
    s2_real_birth_records: str = ""
    s2_real_birth_summary: str = ""
    large_gap_area_threshold: int = 4096
    min_persistence_frames: int = 2
    max_peak_vram_mb: float = 21000.0
    max_cmr_delta: float = 0.005
    max_drr_delta: float = 0.01
    min_ttp_delta: float = -0.005
    min_asa_delta: float = -0.005
    min_coverage_delta: float = -0.010


@dataclass
class Phase8LingBotShadowConfig:
    enabled: bool = True
    scene_id: str = "scene0050_00"
    frame_start: int = 4450
    frame_stride: int = 5
    expected_frame_count: int = 32
    main_replay_summary: str = "Stream3D/outputs/audit/v106_phase5_dupsuppress_schedule_20260712_172217/phase5/R1_repair_vs_birth_phase5_replay/phase5_frozen_birth_replay_summary.json"
    real_lingbot_stream_summary: str = "Stream3D/outputs/audit/v106_phase8_lingbot_real_stream_contract_20260712_1850_scene0050_c2/lingbot_stream_contract_summary.json"


@dataclass
class Phase9FullDevConfig:
    enabled: bool = True
    required_scenes_csv: str = "scene0011_00,scene0050_00"
    required_variants_csv: str = "B0,B1,B4,B6,B7"
    required_cache_mode: str = "write_only_verified_no_read"
    dev_chain_summaries_csv: str = ""
    coverage_diagnostics_csv: str = ""
    preflight_chunk_summaries_csv: str = "Stream3D/outputs/audit/v106_phase9_scene0011_chunk0_b4_replay_preflight_20260712_1917/phase5_frozen_birth_replay_summary.json"
    integrated_phase0_6_run_summary: str = "Stream3D/outputs/audit/v106_phase0123456_integrated_gate_20260712_175448/run_summary.json"
    phase2_b4_summary: str = "Stream3D/outputs/audit/v106_phase2_b4_phase5_state_template_scene0050_start4160_20260712_161622/phase5_frozen_birth_replay_summary.json"
    phase5_b6_replay_summary: str = "Stream3D/outputs/audit/v106_phase5_dupsuppress_schedule_20260712_172217/phase5/R1_repair_vs_birth_phase5_replay/phase5_frozen_birth_replay_summary.json"
    phase5_b6_gate_records: str = "Stream3D/outputs/audit/v106_phase5_repair_birth_defer_gate_dupsuppress_20260712_172451/phase5/gate_records.json"
    phase7_final_decision: str = "Stream3D/outputs/audit/v106_phase7_final_residual_probe_20260712_1837/phase7_final_residual_decision.json"
    phase8_gate_summary: str = "Stream3D/outputs/audit/v106_phase8_lingbot_shadow_gate_20260712_1853/phase8/gate_summary.json"


@dataclass
class Phase9SceneLoopSmokeConfig:
    enabled: bool = False
    scene_id: str = "scene0011_00"
    c0_summary: str = "Stream3D/outputs/audit/v106_phase9_scene0011_chunk0_b4_replay_preflight_20260712_1917/phase5_frozen_birth_replay_summary.json"
    c0_labels_dir: str = "Stream3D/outputs/audit/v106_phase9_scene0011_chunk0_b4_replay_preflight_20260712_1917/labels"
    c0_chunk_index: int = 0
    c1_chunk_index: int = 1
    c0_frame_start: int = 0
    c1_frame_start: int = 145
    frame_stride: int = 5
    frame_count: int = 32
    overlap: int = 3
    reference_c1_summary: str = ""
    handoff_replay_variant: str = "H4_endpoint_drift_correction"
    execute_handoff_replay: bool = False
    h2_replay_summary: str = ""
    h4_replay_summary: str = ""
    execute_h2_replay: bool = False
    replay_gpu: str = "6"
    sam2_replay_config: str = "configs/v105/baseline_chunk_table/baseline_x_gapadaptive_sam2.generated.yaml"
    replay_output_root: str = ""
    video_feature_bank_storage_device: str = "cuda"
    video_gpu_hot_window: int = 8
    reuse_video_state_template: bool = True
    duplicate_window_frames: int = 3
    duplicate_overlap_threshold: float = 0.55
    phase5_repair_birth_defer_enabled: bool = False
    phase5_variant: str = "R1_repair_vs_birth"
    phase5_reference_birth_records: str = ""
    phase5_reference_summary: str = ""
    phase5_inherited_replay_summary: str = ""
    phase5_min_area: int = 16
    phase5_repair_overlap_coeff: float = 0.55
    phase5_duplicate_suppress_overlap_coeff: float = 0.95
    phase5_duplicate_suppress_area_ratio_min: float = 0.90
    phase5_birth_max_overlap_coeff: float = 0.25
    phase5_min_persistence_frames: int = 1
    phase5_large_persistent_area: int = 4096
    phase5_defer_new_births_until_non_overlap: bool = True
    phase5_overlap_birth_keep_min_area: int = 0
    phase5_preserve_overlap_inherited_masks: bool = True
    phase5_selected_chunk_indices: str = ""
    phase5_min_birth_mask_area: int = 0
    phase5_min_output_mask_area: int = 0
    phase5_min_output_component_area: int = 0
    handoff_drift_filter_enabled: bool = False
    handoff_drift_growth_threshold: float = 1.35
    handoff_drift_min_probe_area: int = 20000
    handoff_drift_probe_frame_count: int = 2
    residual_repair_enabled: bool = False
    residual_gpu: str = "6"
    residual_selected_chunk_indices: str = ""
    residual_input_role: str = "phase9_scene_loop_residual_gap_temporal_age_gate"
    residual_source: str = "v106_phase9_scene_loop_residual_gap_temporal_age_gate"
    residual_model_dtype: str = "bf16"
    residual_mode: str = "gap"
    residual_min_uncovered_ratio: float = 0.0
    residual_start_chunk_index: int = 0
    residual_frame_step: int = 1
    residual_max_residual_frames: int = -1
    residual_max_birth_mask_area: int = 0
    residual_max_birth_mask_area_ratio: float = 0.25
    residual_max_birth_mask_uncovered_ratio: float = 0.0
    residual_repair_birth_defer_mode: str = "overlap"
    residual_repair_overlap_coeff: float = 0.55
    residual_duplicate_suppress_overlap_coeff: float = 0.90
    residual_birth_max_overlap_coeff: float = 0.25
    residual_repair_birth_defer_min_area: int = 8192
    residual_ambiguous_overlap_action: str = "defer"
    residual_temporal_repair_mode: str = "mask_overlap"
    residual_temporal_min_overlap: float = 0.10
    residual_temporal_min_area: int = 1024
    residual_temporal_max_area_ratio: float = 128.0
    residual_temporal_window_chunks: int = 0
    residual_temporal_min_target_age_chunks: int = 2
    residual_temporal_young_match_action: str = "noise"
    min_ccoc: float = 0.85
    min_hir: float = 0.95
    min_hcr: float = 0.90
    max_cfr: float = 0.05
    max_cmr: float = 0.05
    max_bfmr: float = 0.05
    fragment_overlap_fraction_threshold: float = 0.01
    merge_overlap_fraction_threshold: float = 0.01
    endpoint_drift_area_ratio_min: float = 0.75
    endpoint_drift_area_ratio_max: float = 1.33


@dataclass
class Phase10HoldoutConfig:
    enabled: bool = True
    split_name: str = "holdout"
    phase9_summary: str = ""
    forbid_holdout_parameter_callback: bool = True


@dataclass
class V106Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sam2: SAM2Config = field(default_factory=SAM2Config)
    local_exact: LocalExactConfig = field(default_factory=LocalExactConfig)
    handoff: HandoffConfig = field(default_factory=HandoffConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    acceleration: AccelerationConfig = field(default_factory=AccelerationConfig)
    specgap: SpecGapConfig = field(default_factory=SpecGapConfig)
    lingbot: LingBotConfig = field(default_factory=LingBotConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    phase2_artifact_replay: Phase2ArtifactConfig = field(default_factory=Phase2ArtifactConfig)
    phase3_handoff: Phase3HandoffConfig = field(default_factory=Phase3HandoffConfig)
    phase4_scene_state: Phase4SceneStateConfig = field(default_factory=Phase4SceneStateConfig)
    phase5_repair_birth_defer: Phase5RepairBirthDeferConfig = field(default_factory=Phase5RepairBirthDeferConfig)
    phase6_occlusion_reappearance: Phase6OcclusionReappearanceConfig = field(default_factory=Phase6OcclusionReappearanceConfig)
    phase7_specgap_parallel: Phase7SpecGapParallelConfig = field(default_factory=Phase7SpecGapParallelConfig)
    phase8_lingbot_shadow: Phase8LingBotShadowConfig = field(default_factory=Phase8LingBotShadowConfig)
    phase9_full_dev: Phase9FullDevConfig = field(default_factory=Phase9FullDevConfig)
    phase9_scene_loop_smoke: Phase9SceneLoopSmokeConfig = field(default_factory=Phase9SceneLoopSmokeConfig)
    phase10_holdout: Phase10HoldoutConfig = field(default_factory=Phase10HoldoutConfig)


def _parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw in {"true", "True"}:
        return True
    if raw in {"false", "False"}:
        return False
    if raw in {"null", "None", "~"}:
        return None
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _minimal_yaml_load(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def load_mapping(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except Exception:
        return _minimal_yaml_load(text)


def _section(cls: type, payload: Dict[str, Any], key: str) -> Any:
    section = payload.get(key, {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise TypeError(f"Config section {key!r} must be a mapping")
    return cls(**section)


def load_config(path: Path) -> V106Config:
    payload = load_mapping(path)
    return V106Config(
        run=_section(RunConfig, payload, "run"),
        data=_section(DataConfig, payload, "data"),
        sam2=_section(SAM2Config, payload, "sam2"),
        local_exact=_section(LocalExactConfig, payload, "local_exact"),
        handoff=_section(HandoffConfig, payload, "handoff"),
        identity=_section(IdentityConfig, payload, "identity"),
        acceleration=_section(AccelerationConfig, payload, "acceleration"),
        specgap=_section(SpecGapConfig, payload, "specgap"),
        lingbot=_section(LingBotConfig, payload, "lingbot"),
        evaluation=_section(EvaluationConfig, payload, "evaluation"),
        video=_section(VideoConfig, payload, "video"),
        phase2_artifact_replay=_section(Phase2ArtifactConfig, payload, "phase2_artifact_replay"),
        phase3_handoff=_section(Phase3HandoffConfig, payload, "phase3_handoff"),
        phase4_scene_state=_section(Phase4SceneStateConfig, payload, "phase4_scene_state"),
        phase5_repair_birth_defer=_section(
            Phase5RepairBirthDeferConfig, payload, "phase5_repair_birth_defer"
        ),
        phase6_occlusion_reappearance=_section(
            Phase6OcclusionReappearanceConfig, payload, "phase6_occlusion_reappearance"
        ),
        phase7_specgap_parallel=_section(
            Phase7SpecGapParallelConfig, payload, "phase7_specgap_parallel"
        ),
        phase8_lingbot_shadow=_section(
            Phase8LingBotShadowConfig, payload, "phase8_lingbot_shadow"
        ),
        phase9_full_dev=_section(
            Phase9FullDevConfig, payload, "phase9_full_dev"
        ),
        phase9_scene_loop_smoke=_section(
            Phase9SceneLoopSmokeConfig, payload, "phase9_scene_loop_smoke"
        ),
        phase10_holdout=_section(
            Phase10HoldoutConfig, payload, "phase10_holdout"
        ),
    )


def config_contract_records(config: V106Config) -> Dict[str, Any]:
    checks = [
        {
            "name": "version_is_v106",
            "passes": config.run.version == "v106",
            "actual": config.run.version,
            "expected": "v106",
        },
        {
            "name": "scene_stream_sequential",
            "passes": config.run.scene_stream_sequential is True,
            "actual": config.run.scene_stream_sequential,
            "expected": True,
        },
        {
            "name": "same_scene_chunk_parallelism_is_one",
            "passes": config.run.same_scene_chunk_parallelism == 1,
            "actual": config.run.same_scene_chunk_parallelism,
            "expected": 1,
        },
        {
            "name": "later_chunk_full_reinitialize_forbidden",
            "passes": config.local_exact.later_chunk_full_reinitialize is False,
            "actual": config.local_exact.later_chunk_full_reinitialize,
            "expected": False,
        },
        {
            "name": "sam2_load_once_per_scene",
            "passes": config.sam2.model_load_once_per_scene is True,
            "actual": config.sam2.model_load_once_per_scene,
            "expected": True,
        },
        {
            "name": "sam2_image_dtype_bfloat16",
            "passes": config.sam2.image_autocast_dtype == "bfloat16",
            "actual": config.sam2.image_autocast_dtype,
            "expected": "bfloat16",
        },
        {
            "name": "sam2_video_dtype_bfloat16",
            "passes": config.sam2.video_autocast_dtype == "bfloat16",
            "actual": config.sam2.video_autocast_dtype,
            "expected": "bfloat16",
        },
        {
            "name": "lingbot_shadow_only",
            "passes": (
                config.lingbot.mode == "shadow"
                and not config.lingbot.affects_mask
                and not config.lingbot.affects_gap
                and not config.lingbot.affects_identity
                and not config.lingbot.affects_gate
            ),
            "actual": {
                "mode": config.lingbot.mode,
                "affects_mask": config.lingbot.affects_mask,
                "affects_gap": config.lingbot.affects_gap,
                "affects_identity": config.lingbot.affects_identity,
                "affects_gate": config.lingbot.affects_gate,
            },
            "expected": "shadow provider with no core-method effect",
        },
    ]
    return {"passes": all(c["passes"] for c in checks), "checks": checks}
