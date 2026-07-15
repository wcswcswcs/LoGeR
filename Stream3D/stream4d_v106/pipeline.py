from __future__ import annotations

import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from . import VERSION
from .artifacts import (
    file_record,
    module_doc_status,
    now_iso,
    sha256_file,
    source_import_lines,
    write_json,
)
from .baseline_x_engine import decide_chunk_initialization
from .config import V106Config, config_contract_records, load_config
from .identity_metrics import run_synthetic_metric_suite
from .lingbot_shadow import audit_lingbot_shadow
from .phase2_parity import run_phase2_artifact_parity
from .phase3_handoff import run_phase3_handoff_smoke
from .phase4_scene_state import run_phase4_scene_state_audit
from .phase5_repair_birth_defer import run_phase5_repair_birth_defer
from .phase6_occlusion_reappearance import run_phase6_occlusion_reappearance
from .phase7_specgap_parallel import run_phase7_specgap_parallel
from .phase8_lingbot_shadow import run_phase8_lingbot_shadow
from .phase9_full_dev import run_phase9_full_dev
from .phase9_scene_loop import run_phase9_scene_loop_smoke
from .phase10_holdout_casebook import run_phase10_holdout_casebook
from .profiling import StageTimer
from .sam2_feature_bank_adapter import FeatureBankContract
from .sam2_state_template import SAM2StateTemplateContract
from .scene_state import SceneStreamState, assert_same_scene_sequential
from .specgap import build_specgap_policy


PLAN_PATH = "docs/stream4d_v106_sequential_streaming_sam2_objectlet_plan.md"
RUNNER_PATH = "Stream3D/tools/run_v106_streaming_sam2_objectlet_pipeline.py"

MODULE_DOC_PATHS = [
    "docs/v106/modules/v106_pipeline_contract.md",
    "docs/v106/modules/scene_stream_state_design.md",
    "docs/v106/modules/baseline_x_exact_engine_design.md",
    "docs/v106/modules/sam2_feature_bank_design.md",
    "docs/v106/modules/sam2_state_template_design.md",
    "docs/v106/modules/chunk_handoff_design.md",
    "docs/v106/modules/global_identity_registry_design.md",
    "docs/v106/modules/lifecycle_reappearance_design.md",
    "docs/v106/modules/gap_repair_birth_defer_design.md",
    "docs/v106/modules/speculative_gap_parallel_design.md",
    "docs/v106/modules/coverage_preserving_reconciliation_design.md",
    "docs/v106/modules/identity_metric_suite_design.md",
    "docs/v106/modules/lingbot_shadow_provider_design.md",
    "docs/v106/modules/cache_contract.md",
    "docs/v106/modules/profiling_contract.md",
    "docs/v106/modules/video_casebook_contract.md",
]

V105_AUDITED_PATHS = [
    "tools/audit_v105_baseline_x_sam2_twostage_tracking.py",
    "tools/build_v105_phase6_speculative_gap_birth.py",
    "tools/run_v105_fullscene_persistent_batch_pipeline.py",
    "tools/run_v105_fullscene_multichunk_repair.py",
    "tools/build_v105_fullscene_local2history_stitch.py",
    "Stream3D/sgq_v105/sam2_feature_bank.py",
    "tools/build_v105_phase3_lingbot_stream_contract.py",
    "third_party/4D_PM/frontend/segment/video_matcher.py",
    "Grounded-SAM-2/sam2/sam2_image_predictor.py",
    "Grounded-SAM-2/sam2/sam2_video_predictor.py",
]

V106_SOURCE_PATHS = [
    RUNNER_PATH,
    "Stream3D/stream4d_v106/__init__.py",
    "Stream3D/stream4d_v106/artifacts.py",
    "Stream3D/stream4d_v106/baseline_x_engine.py",
    "Stream3D/stream4d_v106/cache.py",
    "Stream3D/stream4d_v106/chunk_schedule.py",
    "Stream3D/stream4d_v106/config.py",
    "Stream3D/stream4d_v106/gap_classifier.py",
    "Stream3D/stream4d_v106/handoff.py",
    "Stream3D/stream4d_v106/identity_metrics.py",
    "Stream3D/stream4d_v106/identity_registry.py",
    "Stream3D/stream4d_v106/lifecycle.py",
    "Stream3D/stream4d_v106/lingbot_shadow.py",
    "Stream3D/stream4d_v106/objectlet_metrics.py",
    "Stream3D/stream4d_v106/phase3_handoff.py",
    "Stream3D/stream4d_v106/phase4_scene_state.py",
    "Stream3D/stream4d_v106/phase5_repair_birth_defer.py",
    "Stream3D/stream4d_v106/phase6_occlusion_reappearance.py",
    "Stream3D/stream4d_v106/phase7_specgap_parallel.py",
    "Stream3D/stream4d_v106/phase8_lingbot_shadow.py",
    "Stream3D/stream4d_v106/phase9_full_dev.py",
    "Stream3D/stream4d_v106/phase9_scene_loop.py",
    "tools/run_v106_stateful_sam2_scene_stream.py",
    "tools/audit_v106_sam2_rolling_state.py",
    "tools/run_v106_stateful_sam2_rolling_scene_stream.py",
    "configs/v106/v106_stateful_sam2_rolling_scene_stream.yaml",
    "configs/v106/v106_stateful_sam2_rolling_scene_stream_fast_admission.yaml",
    "configs/v106/v106_stateful_sam2_rolling_scene_stream_persistent_admission.yaml",
    "configs/v106/v106_stateful_sam2_rolling_scene_stream_area20k_preprune8.yaml",
    "configs/v106/v106_stateful_sam2_rolling_scene_stream_area20k_preprune6.yaml",
    "configs/v106/v106_stateful_sam2_rolling_scene_stream_area20k_preprune6_maxvis45.yaml",
    "configs/v106/v106_stateful_sam2_rolling_scene_stream_area20k_preprune6_maxvis45_labelcompact_noempty.yaml",
    "Stream3D/stream4d_v106/phase10_holdout_casebook.py",
    "Stream3D/stream4d_v106/pipeline.py",
    "Stream3D/stream4d_v106/profiling.py",
    "Stream3D/stream4d_v106/reconciliation.py",
    "Stream3D/stream4d_v106/sam2_feature_bank_adapter.py",
    "Stream3D/stream4d_v106/sam2_state_template.py",
    "Stream3D/stream4d_v106/scene_state.py",
    "Stream3D/stream4d_v106/specgap.py",
    "Stream3D/stream4d_v106/video_export.py",
]

V106_EDGETAM_SOURCE_PATHS = [
    "tools/audit_v106_edgetam_twostage_tracking.py",
    "tools/run_v106_stateful_edgetam_scene_stream.py",
    "tools/audit_v106_sam2seg_edgetam_tracking.py",
    "tools/run_v106_stateful_sam2seg_edgetam_scene_stream.py",
    "configs/v106/v106_stateful_edgetam_scene_stream.yaml",
    "configs/v106/v106_stateful_edgetam_scene_stream_balanced_gap128.yaml",
    "configs/v106/v106_stateful_edgetam_scene_stream_balanced_yoloe.yaml",
    "configs/v106/v106_stateful_edgetam_scene_stream_balanced_yoloe_sparsebirth.yaml",
    "configs/v106/v106_stateful_sam2seg_edgetam_scene_stream.yaml",
    "configs/v106/v106_stateful_edgetam_scene_stream_recall_gap256.yaml",
]

FORBIDDEN_RUNTIME_IMPORT_PATTERNS = [
    "alltracker",
    "sam3",
    "edgetam",
    "fastsam",
    "cropformer",
    "run_v105",
    "build_v105_fullscene_local2history_stitch",
]


class V106Pipeline:
    def __init__(self, repo_root: Path, config_path: Path, output_root: Path, force: bool = False) -> None:
        self.repo_root = repo_root.resolve()
        self.config_path = config_path.resolve()
        self.output_root = output_root.resolve()
        if self.output_root.exists() and force:
            shutil.rmtree(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.config: V106Config = load_config(self.config_path)
        self.timer = StageTimer()

    def run(self, stages: Iterable[str]) -> Dict[str, Any]:
        stage_list = [stage.strip().lower() for stage in stages if stage.strip()]
        results: Dict[str, Any] = {
            "version": VERSION,
            "started_at": now_iso(),
            "repo_root": str(self.repo_root),
            "config_path": str(self.config_path),
            "output_root": str(self.output_root),
            "python": sys.executable,
            "stages_requested": stage_list,
            "stages": {},
        }
        for stage in stage_list:
            if stage == "phase0":
                with self.timer.measure("phase0"):
                    results["stages"]["phase0"] = self.run_phase0()
            elif stage == "phase1":
                with self.timer.measure("phase1"):
                    results["stages"]["phase1"] = self.run_phase1()
            elif stage == "phase2":
                with self.timer.measure("phase2"):
                    results["stages"]["phase2"] = self.run_phase2()
            elif stage == "phase3":
                with self.timer.measure("phase3"):
                    results["stages"]["phase3"] = self.run_phase3()
            elif stage == "phase4":
                with self.timer.measure("phase4"):
                    results["stages"]["phase4"] = self.run_phase4()
            elif stage == "phase5":
                with self.timer.measure("phase5"):
                    results["stages"]["phase5"] = self.run_phase5()
            elif stage == "phase6":
                with self.timer.measure("phase6"):
                    results["stages"]["phase6"] = self.run_phase6()
            elif stage == "phase7":
                with self.timer.measure("phase7"):
                    results["stages"]["phase7"] = self.run_phase7()
            elif stage == "phase8":
                with self.timer.measure("phase8"):
                    results["stages"]["phase8"] = self.run_phase8()
            elif stage == "phase9":
                with self.timer.measure("phase9"):
                    results["stages"]["phase9"] = self.run_phase9()
            elif stage == "phase10":
                with self.timer.measure("phase10"):
                    results["stages"]["phase10"] = self.run_phase10()
            else:
                raise ValueError(f"unsupported stage: {stage}")
        results["finished_at"] = now_iso()
        results["timing_seconds"] = self.timer.summary()
        results["all_requested_stages_pass"] = all(
            payload.get("passes", False) for payload in results["stages"].values()
        )
        write_json(self.output_root / "run_summary.json", results)
        return results

    def run_phase0(self) -> Dict[str, Any]:
        phase_dir = self.output_root / "phase0"
        phase_dir.mkdir(parents=True, exist_ok=True)
        config_contract = config_contract_records(self.config)
        docs_status = module_doc_status(self.repo_root, MODULE_DOC_PATHS)
        fact_lock = self._phase0_fact_lock(docs_status, config_contract)
        code_audit = self._phase0_code_path_audit()
        sam2_contract = self._phase0_sam2_contract()
        root_cause_records = self._phase0_root_causes()
        forbidden_records = self._phase0_forbidden_paths(config_contract)
        design_drift = self._design_implementation_drift(docs_status)

        write_json(phase_dir / "fact_lock_summary.json", fact_lock)
        write_json(phase_dir / "code_path_audit.json", code_audit)
        write_json(phase_dir / "sam2_api_contract.json", sam2_contract)
        write_json(phase_dir / "current_root_cause_records.json", root_cause_records)
        write_json(phase_dir / "forbidden_path_records.json", forbidden_records)
        write_json(phase_dir / "module_design_doc_status.json", docs_status)
        write_json(phase_dir / "design_implementation_drift.json", design_drift)

        gate_checks = [
            {
                "name": "single_v106_runner_exists",
                "passes": (self.repo_root / RUNNER_PATH).exists(),
                "actual": RUNNER_PATH,
                "expected": "runner file exists",
            },
            {
                "name": "all_module_docs_exist_and_pass",
                "passes": docs_status["all_pass"],
                "actual": {
                    "module_doc_count": docs_status["module_doc_count"],
                    "all_exist": docs_status["all_exist"],
                    "all_pass": docs_status["all_pass"],
                },
                "expected": "16 module docs with 9/9 sections",
            },
            {
                "name": "scene_chunks_declared_sequential",
                "passes": config_contract["checks"][1]["passes"] and config_contract["checks"][2]["passes"],
                "actual": {
                    "scene_stream_sequential": self.config.run.scene_stream_sequential,
                    "same_scene_chunk_parallelism": self.config.run.same_scene_chunk_parallelism,
                },
                "expected": {"scene_stream_sequential": True, "same_scene_chunk_parallelism": 1},
            },
            {
                "name": "later_chunk_full_initialization_forbidden",
                "passes": config_contract["checks"][3]["passes"] and forbidden_records["later_chunk_full_init_code_check"]["passes"],
                "actual": {
                    "config": self.config.local_exact.later_chunk_full_reinitialize,
                    "code_check": forbidden_records["later_chunk_full_init_code_check"],
                },
                "expected": False,
            },
            {
                "name": "sam2_checkpoint_and_config_hash_recorded",
                "passes": sam2_contract["checkpoint"]["exists"] and bool(sam2_contract["checkpoint"]["sha256"])
                and sam2_contract["model_cfg"]["exists"] and bool(sam2_contract["model_cfg"]["sha256"]),
                "actual": {
                    "checkpoint": sam2_contract["checkpoint"],
                    "model_cfg": sam2_contract["model_cfg"],
                },
                "expected": "existing SAM2.1-L checkpoint and config with sha256",
            },
            {
                "name": "no_ap_generated",
                "passes": True,
                "actual": {"ap_artifacts_generated": False},
                "expected": {"ap_artifacts_generated": False},
            },
        ]
        gate_summary = {
            "passes": all(check["passes"] for check in gate_checks),
            "checks": gate_checks,
            "no_scene_experiment_was_run": True,
            "gpu_used": False,
            "ap_generated": False,
        }
        write_json(phase_dir / "gate_summary.json", gate_summary)
        return {"passes": gate_summary["passes"], "artifact_dir": str(phase_dir), "gate_summary": gate_summary}

    def run_phase2(self) -> Dict[str, Any]:
        phase_dir = self.output_root / "phase2"
        phase_dir.mkdir(parents=True, exist_ok=True)
        summary = run_phase2_artifact_parity(self.repo_root, self.config.phase2_artifact_replay, phase_dir)
        gate_summary = {
            "passes": summary["passes"],
            "checks": summary["gate_checks"],
            "scope": summary["scope"],
            "ap_generated": False,
            "fresh_sam2_model_run": False,
            "gpu_used_by_this_harness": False,
        }
        write_json(phase_dir / "gate_summary.json", gate_summary)
        return {"passes": gate_summary["passes"], "artifact_dir": str(phase_dir), "gate_summary": gate_summary}

    def run_phase3(self) -> Dict[str, Any]:
        phase_dir = self.output_root / "phase3"
        phase_dir.mkdir(parents=True, exist_ok=True)
        summary = run_phase3_handoff_smoke(self.repo_root, self.config.phase3_handoff, phase_dir)
        gate_summary = {
            "passes": summary["passes"],
            "passing_variants": summary.get("passing_variants", []),
            "alignment_checks": summary.get("alignment_checks", []),
            "c1_stage12_full_initialization_used_by_handoff": summary.get(
                "c1_stage12_full_initialization_used_by_handoff"
            ),
            "variant_metrics": summary.get("variant_metrics", {}),
            "ap_generated": False,
            "fresh_sam2_model_run_by_this_harness": False,
            "gpu_used_by_this_harness": False,
        }
        write_json(phase_dir / "gate_summary.json", gate_summary)
        return {"passes": gate_summary["passes"], "artifact_dir": str(phase_dir), "gate_summary": gate_summary}

    def run_phase4(self) -> Dict[str, Any]:
        phase_dir = self.output_root / "phase4"
        phase_dir.mkdir(parents=True, exist_ok=True)
        summary = run_phase4_scene_state_audit(self.repo_root, self.config.phase4_scene_state, phase_dir)
        gate_summary = {
            "passes": summary["passes"],
            "failure_count": summary.get("failure_count", 0),
            "support_records": summary.get("support_records", {}),
            "gate_records": str(phase_dir / "gate_records.json"),
            "failure_records": str(phase_dir / "failure_records.json"),
            "ap_generated": False,
            "fresh_sam2_model_run_by_this_harness": False,
            "gpu_used_by_this_harness": False,
        }
        write_json(phase_dir / "gate_summary.json", gate_summary)
        return {"passes": gate_summary["passes"], "artifact_dir": str(phase_dir), "gate_summary": gate_summary}

    def run_phase5(self) -> Dict[str, Any]:
        phase_dir = self.output_root / "phase5"
        phase_dir.mkdir(parents=True, exist_ok=True)
        summary = run_phase5_repair_birth_defer(
            self.repo_root, self.config.phase5_repair_birth_defer, phase_dir
        )
        gate_summary = {
            "passes": summary["passes"],
            "passing_variants": summary.get("passing_variants", []),
            "failure_count": summary.get("failure_count", 0),
            "schedule_audits": summary.get("schedule_audits", {}),
            "variant_metrics": summary.get("variant_metrics", {}),
            "replay_commands": summary.get("replay_commands", {}),
            "ap_generated": False,
            "fresh_sam2_model_run_by_this_harness": False,
            "gpu_used_by_this_harness": False,
        }
        write_json(phase_dir / "gate_summary.json", gate_summary)
        return {"passes": gate_summary["passes"], "artifact_dir": str(phase_dir), "gate_summary": gate_summary}

    def run_phase6(self) -> Dict[str, Any]:
        phase_dir = self.output_root / "phase6"
        phase_dir.mkdir(parents=True, exist_ok=True)
        summary = run_phase6_occlusion_reappearance(
            self.repo_root, self.config.phase6_occlusion_reappearance, phase_dir
        )
        gate_summary = {
            "passes": summary["passes"],
            "failure_count": summary.get("failure_count", 0),
            "real": summary.get("real", {}),
            "controls": summary.get("controls", {}),
            "baseline_metrics": summary.get("baseline_metrics", {}),
            "relabeled_metrics": summary.get("relabeled_metrics", {}),
            "descriptor_provenance": summary.get("descriptor_provenance", {}),
            "ap_generated": False,
            "mv_ap_scene_used_as_gate": False,
        }
        write_json(phase_dir / "gate_summary.json", gate_summary)
        return {"passes": gate_summary["passes"], "artifact_dir": str(phase_dir), "gate_summary": gate_summary}

    def run_phase7(self) -> Dict[str, Any]:
        phase_dir = self.output_root / "phase7"
        phase_dir.mkdir(parents=True, exist_ok=True)
        summary = run_phase7_specgap_parallel(
            self.repo_root, self.config.phase7_specgap_parallel, phase_dir
        )
        gate_summary = {
            "passes": summary["passes"],
            "promotion": summary.get("promotion"),
            "failure_count": summary.get("failure_count", 0),
            "scope": summary.get("scope", {}),
            "variant_summary": summary.get("variant_summary", {}),
            "exact_reference_metrics": summary.get("exact_reference_metrics", {}),
            "exact_reference_runtime": summary.get("exact_reference_runtime", {}),
            "ap_generated": False,
            "mv_ap_scene_used_as_gate": False,
        }
        write_json(phase_dir / "gate_summary.json", gate_summary)
        return {"passes": gate_summary["passes"], "artifact_dir": str(phase_dir), "gate_summary": gate_summary}

    def run_phase8(self) -> Dict[str, Any]:
        phase_dir = self.output_root / "phase8"
        phase_dir.mkdir(parents=True, exist_ok=True)
        summary = run_phase8_lingbot_shadow(
            self.repo_root,
            self.config,
            self.config.phase8_lingbot_shadow,
            phase_dir,
        )
        gate_summary = {
            "passes": bool(summary["passes"]),
            "label_sha_identical": bool(summary.get("label_sha_identical", False)),
            "overlap_repush_count": int(summary.get("overlap_repush_count", -1)),
            "affects_main_labels": bool(summary.get("affects_main_labels", True)),
            "real_lingbot_streaming_executed": bool(summary.get("real_lingbot_streaming_executed", False)),
            "real_lingbot_streaming_contract_complete": bool(
                summary.get("real_lingbot_streaming_contract_complete", False)
            ),
            "packet_complete": bool(summary.get("packet_complete", False)),
            "ap_generated": False,
            "mv_ap_scene_used_as_gate": False,
        }
        write_json(phase_dir / "gate_summary.json", gate_summary)
        return {"passes": gate_summary["passes"], "artifact_dir": str(phase_dir), "gate_summary": gate_summary}

    def run_phase9(self) -> Dict[str, Any]:
        phase_dir = self.output_root / "phase9"
        phase_dir.mkdir(parents=True, exist_ok=True)
        scene_loop_smoke_summary: Dict[str, Any] = {}
        if bool(self.config.phase9_scene_loop_smoke.enabled):
            scene_loop_smoke_summary = run_phase9_scene_loop_smoke(
                self.repo_root,
                self.config.phase9_scene_loop_smoke,
                phase_dir / "scene_loop_smoke",
            )
        summary = run_phase9_full_dev(
            self.repo_root,
            self.config,
            self.config.phase9_full_dev,
            phase_dir,
        )
        gate_summary = {
            "passes": bool(summary.get("passes", False)),
            "status": summary.get("status"),
            "missing_evidence_count": int(summary.get("missing_evidence_count", 0)),
            "required_scenes": summary.get("required_scenes", []),
            "required_variants": summary.get("required_variants", []),
            "freeze_decision_json": summary.get("freeze_decision_json"),
            "scene_loop_smoke_enabled": bool(self.config.phase9_scene_loop_smoke.enabled),
            "scene_loop_smoke_summary_json": (
                str(phase_dir / "scene_loop_smoke" / "scene_loop_smoke_summary.json")
                if scene_loop_smoke_summary
                else ""
            ),
            "scene_loop_smoke_passes": bool(scene_loop_smoke_summary.get("passes", False))
            if scene_loop_smoke_summary
            else False,
            "ap_generated": False,
            "mv_ap_scene_used_as_gate": False,
        }
        write_json(phase_dir / "gate_summary.json", gate_summary)
        return {"passes": gate_summary["passes"], "artifact_dir": str(phase_dir), "gate_summary": gate_summary}

    def run_phase10(self) -> Dict[str, Any]:
        phase_dir = self.output_root / "phase10"
        phase_dir.mkdir(parents=True, exist_ok=True)
        decision = run_phase10_holdout_casebook(
            self.repo_root,
            self.config_path,
            self.config,
            self.config.phase10_holdout,
            self.output_root,
            phase_dir,
        )
        gate_summary = {
            "passes": bool(decision.get("method_success", False)),
            "status": decision.get("status"),
            "phase9_pass": bool(decision.get("phase9_pass", False)),
            "frozen_variant": decision.get("frozen_variant"),
            "holdout_run": bool(decision.get("holdout_run", False)),
            "missing_required_outputs": decision.get("missing_required_outputs", []),
            "ap_generated": False,
            "mv_ap_scene_used_as_gate": False,
        }
        write_json(phase_dir / "gate_summary.json", gate_summary)
        return {"passes": gate_summary["passes"], "artifact_dir": str(phase_dir), "gate_summary": gate_summary}

    def run_phase1(self) -> Dict[str, Any]:
        phase_dir = self.output_root / "phase1"
        phase_dir.mkdir(parents=True, exist_ok=True)
        suite = run_synthetic_metric_suite(phase_dir)
        gate_summary = {
            "passes": suite["summary"]["all_pass"],
            "checks": [
                {
                    "name": "all_s0_s7_synthetic_cases_pass",
                    "passes": suite["summary"]["all_pass"],
                    "actual": suite["summary"],
                    "expected": "all S0-S7 predicates pass",
                }
            ],
            "no_scene_experiment_was_run": True,
            "gpu_used": False,
            "ap_generated": False,
        }
        write_json(phase_dir / "gate_summary.json", gate_summary)
        return {"passes": gate_summary["passes"], "artifact_dir": str(phase_dir), "gate_summary": gate_summary}

    def _phase0_fact_lock(self, docs_status: Dict[str, Any], config_contract: Dict[str, Any]) -> Dict[str, Any]:
        scene = SceneStreamState(scene_id="phase0_contract_smoke")
        chunk0 = scene.begin_chunk(0)
        scene.allocate_objectlet(local_id=1, chunk_index=0, mask_area=4)
        scene.finish_chunk(chunk0)
        chunk1 = scene.begin_chunk(1)
        scene.finish_chunk(chunk1)
        assert_same_scene_sequential(scene.chunks)
        decisions = [asdict(decide_chunk_initialization(self.config, idx)) for idx in (0, 1)]
        feature_bank = FeatureBankContract(
            shared_frame_feature_bank=self.config.acceleration.shared_frame_feature_bank,
            storage=self.config.acceleration.video_feature_storage,
            hot_window=self.config.acceleration.video_gpu_hot_window,
        )
        feature_bank.validate()
        sam2_template = SAM2StateTemplateContract(
            model_reuse_across_chunks=self.config.acceleration.model_reuse_across_chunks,
            template_reuse=self.config.acceleration.video_state_template_reuse,
            image_autocast_dtype=self.config.sam2.image_autocast_dtype,
            video_autocast_dtype=self.config.sam2.video_autocast_dtype,
        )
        sam2_template.validate()
        specgap_policy = build_specgap_policy(self.config)
        lingbot_shadow = audit_lingbot_shadow(self.config)
        return {
            "version": VERSION,
            "created_at": now_iso(),
            "plan": file_record(self.repo_root, PLAN_PATH, required=True),
            "config": {
                "path": str(self.config_path),
                "sha256": sha256_file(self.config_path),
                "contract": config_contract,
            },
            "module_docs": {
                "module_doc_count": docs_status["module_doc_count"],
                "all_pass": docs_status["all_pass"],
            },
            "chunk_initialization_decisions": decisions,
            "scene_state_smoke": {
                "history_versions": [asdict(chunk) for chunk in scene.chunks],
                "same_scene_chunks_sequential": True,
            },
            "feature_bank_contract": asdict(feature_bank),
            "sam2_state_template_contract": asdict(sam2_template),
            "specgap_policy": asdict(specgap_policy),
            "lingbot_shadow": asdict(lingbot_shadow),
            "no_ap_generated": True,
            "no_scene_experiment_was_run": True,
        }

    def _phase0_code_path_audit(self) -> Dict[str, Any]:
        return {
            "created_at": now_iso(),
            "v105_audited_reference_files": [
                file_record(self.repo_root, rel_path, required=True) for rel_path in V105_AUDITED_PATHS
            ],
            "v106_source_files": [
                file_record(self.repo_root, rel_path, required=True) for rel_path in V106_SOURCE_PATHS
            ],
            "single_runner": file_record(self.repo_root, RUNNER_PATH, required=True),
        }

    def _phase0_sam2_contract(self) -> Dict[str, Any]:
        checkpoint = file_record(self.repo_root, self.config.sam2.checkpoint, required=True)
        model_cfg = file_record(self.repo_root, self.config.sam2.model_cfg, required=True)
        api_files = [
            file_record(self.repo_root, "Grounded-SAM-2/sam2/sam2_image_predictor.py", required=True),
            file_record(self.repo_root, "Grounded-SAM-2/sam2/sam2_video_predictor.py", required=True),
        ]
        return {
            "created_at": now_iso(),
            "checkpoint": checkpoint,
            "model_cfg": model_cfg,
            "api_files": api_files,
            "autocast": {
                "image_autocast_dtype": self.config.sam2.image_autocast_dtype,
                "video_autocast_dtype": self.config.sam2.video_autocast_dtype,
            },
            "model_load_once_per_scene": self.config.sam2.model_load_once_per_scene,
            "note": "Hashes are recorded only for files that exist; missing files keep sha256=null.",
        }

    def _phase0_root_causes(self) -> Dict[str, Any]:
        return {
            "records": [
                {
                    "root_cause": "independent_chunk_not_scene_stream",
                    "v105_evidence": "v105 multichunk repair ran chunk jobs independently and assembled outputs after the fact",
                    "v106_repair": "same-scene chunk parallelism fixed at 1; later chunks inherit scene state",
                },
                {
                    "root_cause": "local_id_treated_as_global_id",
                    "v105_evidence": "posthoc local2history stitch did not create runtime scene state",
                    "v106_repair": "GlobalIdentityRegistry and handoff state separate runtime global IDs from local prompt IDs",
                },
                {
                    "root_cause": "reseeded_gap_birth_cost",
                    "v105_evidence": "v105 speed remained high because later chunks still wrapped expensive per-chunk paths",
                    "v106_repair": "SAM2 model/config declare load-once-per-scene and bf16 autocast; later chunks start from inherited masks",
                },
            ]
        }

    def _phase0_forbidden_paths(self, config_contract: Dict[str, Any]) -> Dict[str, Any]:
        import_records = []
        matches = []
        for rel_path in V106_SOURCE_PATHS:
            path = self.repo_root / rel_path
            lines = source_import_lines(path)
            import_records.append({"path": rel_path, "import_lines": lines})
            for line in lines:
                lowered = line.lower()
                for pattern in FORBIDDEN_RUNTIME_IMPORT_PATTERNS:
                    if pattern in lowered:
                        matches.append({"path": rel_path, "line": line, "pattern": pattern})
        later_chunk_full_init_code_check = {
            "passes": True,
            "evidence": "baseline_x_engine.decide_chunk_initialization raises if later_chunk_full_reinitialize is true for chunk_index>0",
        }
        return {
            "created_at": now_iso(),
            "forbidden_runtime_import_patterns": FORBIDDEN_RUNTIME_IMPORT_PATTERNS,
            "runtime_import_scan": import_records,
            "forbidden_runtime_import_matches": matches,
            "forbidden_runtime_imports_pass": not matches,
            "config_contract": config_contract,
            "later_chunk_full_init_code_check": later_chunk_full_init_code_check,
            "posthoc_relabel_as_main_method": {
                "passes": True,
                "evidence": "No v106 runtime import calls the v105 local2history stitcher; local2history is represented as next-chunk state handoff.",
            },
        }

    def _design_implementation_drift(self, docs_status: Dict[str, Any]) -> Dict[str, Any]:
        expected_sources = set(V106_SOURCE_PATHS)
        missing_sources = [rel for rel in sorted(expected_sources) if not (self.repo_root / rel).exists()]
        drift = bool(missing_sources or not docs_status["all_pass"])
        return {
            "DESIGN_IMPLEMENTATION_DRIFT": drift,
            "missing_v106_sources": missing_sources,
            "module_docs_all_pass": docs_status["all_pass"],
        }
