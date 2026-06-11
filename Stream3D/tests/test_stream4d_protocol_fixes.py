from __future__ import annotations

import unittest
import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools.oracle_candidate_upper_bound import run as run_oracle_candidate_upper_bound
from stream4d.carrier_sampler import CarrierSampler
from stream4d.local_4d_filter import Local4DFilter, LocalProposal
from stream4d.mask_evidence import MaskObservation
from stream4d.object_memory_v2 import ObjectMemory4DV2
from stream4d.reliable_densifier import apply_wta_to_records
from stream4d.rescore_scannet import validate_args, verify_object_dict_prediction_alignment
from stream4d.export_scannet import score_export_record
from tools.d4rt_geometry_diagnostic import fit_sim3_umeyama
from tools.fuse_prediction_configs import _external_support_signal
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.scan_reportable_configs import scan_configs


def _observation(frame_id: int, carrier_ids: list[int], weights: list[float]) -> MaskObservation:
    return MaskObservation(
        frame_id=frame_id,
        frame_local=0,
        mask_id=1,
        carrier_ids=np.asarray(carrier_ids, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float32),
        uv_norm=np.zeros((len(carrier_ids), 2), dtype=np.float32),
        bbox_xyxy=(0, 0, 1, 1),
        area=max(len(carrier_ids), 1),
    )


class Stream4DProtocolFixTests(unittest.TestCase):
    def test_local_filter_carrier_weights_follow_carrier_id_order(self) -> None:
        obs = _observation(frame_id=7, carrier_ids=[303, 101, 202], weights=[0.3, 0.1, 0.2])

        weights = Local4DFilter._carrier_weights([obs])

        self.assertAlmostEqual(weights[(7, 303)], 0.3, places=6)
        self.assertAlmostEqual(weights[(7, 101)], 0.1, places=6)
        self.assertAlmostEqual(weights[(7, 202)], 0.2, places=6)

    def test_carrier_sampler_uses_actual_sample_count_for_all_fields(self) -> None:
        masks = np.zeros((1, 4, 4), dtype=np.int64)
        masks[0, 1, 1] = 5
        masks[0, 1, 2] = 5
        sampler = CarrierSampler(
            max_points_per_mask=8,
            min_points_per_mask=4,
            min_mask_area=1,
            strategy="uniform_mask_pixels",
        )

        sources = sampler.sample(masks=masks, frame_ids=[42])

        self.assertEqual(sources.carrier_id.shape[0], 2)
        self.assertEqual(sources.src_frame.shape[0], 2)
        self.assertEqual(sources.src_frame_global.shape[0], 2)
        self.assertEqual(sources.src_xy.shape[0], 2)
        self.assertEqual(sources.src_uv.shape[0], 2)
        self.assertEqual(sources.src_mask_id.shape[0], 2)

    def test_rescore_alignment_detects_column_mismatch(self) -> None:
        pred_masks = np.zeros((6, 2), dtype=bool)
        pred_masks[[0, 2], 0] = True
        pred_masks[[3, 4], 1] = True
        object_items = [
            (0, {"point_ids": np.asarray([0, 2], dtype=np.int64)}),
            (1, {"point_ids": np.asarray([3, 5], dtype=np.int64)}),
        ]

        alignment = verify_object_dict_prediction_alignment(pred_masks, object_items, threshold=0.99)

        self.assertTrue(alignment["alignment_checked"])
        self.assertEqual(alignment["alignment_failed_instances"], 1)
        self.assertLess(alignment["alignment_min_iou"], 1.0)

    def test_fixed_path_requires_config_name(self) -> None:
        args = argparse.Namespace(pre_points_policy="fixed_path", fixed_pre_points_config="")

        with self.assertRaisesRegex(ValueError, "--fixed-pre-points-config"):
            validate_args(args)

    def test_reliable_densifier_wta_keeps_highest_reliability_owner(self) -> None:
        records = [
            {"object_id": 0, "point_ids": {1, 2}, "reliability": 0.5},
            {"object_id": 1, "point_ids": {2, 3}, "reliability": 1.0},
        ]

        reassigned, diag = apply_wta_to_records(records)

        self.assertEqual(reassigned[0]["point_ids"], {1})
        self.assertEqual(reassigned[1]["point_ids"], {2, 3})
        self.assertEqual(diag["densify_wta_conflict_points"], 1.0)

    def test_reliable_densifier_wta_recomputes_area_sensitive_scores(self) -> None:
        records = [
            {
                "object_id": 0,
                "point_ids": {1, 2},
                "area_score": 2.0,
                "score": 2.0,
                "observations": 2.0,
                "reliability": 2.0,
                "dense_quality": 4.0,
                "selection_quality": 6.0,
            },
            {
                "object_id": 1,
                "point_ids": {2, 3, 4, 5},
                "area_score": 4.0,
                "score": 4.0,
                "observations": 3.0,
                "reliability": 9.0,
                "dense_quality": 8.0,
                "selection_quality": 12.0,
            },
        ]

        reassigned, _ = apply_wta_to_records(records)

        self.assertEqual(reassigned[0]["point_ids"], {1})
        self.assertEqual(reassigned[0]["area_score"], 1.0)
        self.assertEqual(reassigned[0]["score"], 1.0)
        self.assertAlmostEqual(reassigned[0]["reliability"], 2.0)
        self.assertAlmostEqual(reassigned[0]["dense_quality"], 4.0 / np.sqrt(2.0))
        self.assertEqual(reassigned[1]["point_ids"], {2, 3, 4, 5})
        self.assertEqual(reassigned[1]["area_score"], 4.0)

    def test_export_observation_score_uses_observation_field(self) -> None:
        record = {
            "point_ids": {1, 2, 3, 4},
            "area_score": 4.0,
            "observations": 3.0,
            "reliability": 6.0,
        }

        self.assertEqual(score_export_record(record, "observations"), 3.0)
        self.assertEqual(score_export_record(record, "area"), 4.0)
        self.assertEqual(score_export_record(record, "reliability"), 6.0)

    def test_fusion_external_support_detection(self) -> None:
        self.assertTrue(_external_support_signal("scannet"))
        self.assertTrue(_external_support_signal("scannet_on_stream4d_32f_probe5"))
        self.assertTrue(_external_support_signal("some_stream3d_baseline"))
        self.assertFalse(_external_support_signal("stream4d_scannet_32f_ioc075_fixmem"))
        self.assertFalse(_external_support_signal(""))

    def test_d4rt_geometry_sim3_fit_recovers_known_transform(self) -> None:
        source = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 2.0, 3.0],
            ],
            dtype=np.float64,
        )
        theta = np.deg2rad(20.0)
        rotation = np.asarray(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        scale = 2.5
        translation = np.asarray([0.4, -1.2, 3.0], dtype=np.float64)
        target = scale * (source @ rotation.T) + translation

        fit = fit_sim3_umeyama(source, target)

        self.assertAlmostEqual(fit["scale"], scale, places=6)
        self.assertLess(float(np.max(fit["residual"])), 1e-9)
        self.assertAlmostEqual(fit["rotation_det"], 1.0, places=6)

    def test_memory_v2_enforces_one_to_one_matching(self) -> None:
        memory = ObjectMemory4DV2(
            history_match_threshold=0.3,
            carrier_weight=1.0,
            appearance_weight=0.0,
            geometry_weight=0.0,
            motion_weight=0.0,
            conflict_weight=0.0,
        )
        memory.update(
            [
                LocalProposal(
                    proposal_id=0,
                    observation_indices=[0],
                    carrier_ids={1, 2, 3},
                    frame_support={0: {1, 2, 3}},
                    mask_observations=[(0, 1, 1.0)],
                )
            ],
            window_index=0,
        )

        diag = memory.update(
            [
                LocalProposal(
                    proposal_id=0,
                    observation_indices=[0],
                    carrier_ids={1, 2},
                    frame_support={1: {1, 2}},
                    mask_observations=[(1, 2, 1.0)],
                ),
                LocalProposal(
                    proposal_id=1,
                    observation_indices=[1],
                    carrier_ids={1, 3},
                    frame_support={1: {1, 3}},
                    mask_observations=[(1, 3, 1.0)],
                ),
            ],
            window_index=1,
        )

        self.assertEqual(diag["num_matched"], 1.0)
        self.assertEqual(diag["num_created"], 1.0)
        self.assertEqual(diag["num_objects"], 2.0)

    def test_memory_v2_missing_appearance_does_not_match(self) -> None:
        memory = ObjectMemory4DV2(
            history_match_threshold=0.1,
            carrier_weight=0.0,
            appearance_weight=1.0,
            geometry_weight=0.0,
            motion_weight=0.0,
            conflict_weight=0.0,
        )
        memory.update(
            [
                LocalProposal(
                    proposal_id=0,
                    observation_indices=[0],
                    carrier_ids={1},
                    frame_support={0: {1}},
                    mask_observations=[(0, 1, 1.0)],
                )
            ],
            window_index=0,
        )

        diag = memory.update(
            [
                LocalProposal(
                    proposal_id=1,
                    observation_indices=[1],
                    carrier_ids={2},
                    frame_support={1: {2}},
                    mask_observations=[(1, 2, 1.0)],
                )
            ],
            window_index=1,
        )

        self.assertEqual(diag["num_matched"], 0.0)
        self.assertEqual(diag["num_created"], 1.0)

    def test_oracle_output_config_requires_oracle_name(self) -> None:
        with TemporaryDirectory() as tmp:
            seq_list = Path(tmp) / "seq.txt"
            seq_list.write_text("scene0000_00\n", encoding="utf-8")
            args = argparse.Namespace(
                root=tmp,
                seq_list=str(seq_list),
                pred_config="method_pool",
                pre_points_config="method_support",
                output_config="method_not_allowed",
                pred_suffix="class_agnostic",
                min_select_iou=0.25,
                summary_root="outputs/oracle_candidate_upper_bound",
            )

            with self.assertRaisesRegex(ValueError, "must contain 'oracle'"):
                run_oracle_candidate_upper_bound(args)

    def test_evaluator_rejects_uses_gt_manifest_without_oracle_name(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_dir = root / "method_config_class_agnostic"
            pred_dir.mkdir(parents=True)
            (pred_dir / "config_manifest.json").write_text(
                json.dumps({"uses_gt": True, "is_diagnostic_only": True}),
                encoding="utf-8",
            )
            cmd = [
                sys.executable,
                "-m",
                "evaluation.evaluate",
                "--pred_path",
                str(pred_dir),
                "--gt_path",
                str(root / "gt"),
                "--dataset",
                "scannet",
                "--tmp_root",
                str(root / "tmp"),
                "--tmp_config",
                "method_config",
                "--no_class",
            ]

            proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("uses_gt=true", proc.stderr + proc.stdout)

    def test_v10_manifest_defaults_include_explicit_gt_and_geometry_fields(self) -> None:
        manifest = build_prediction_manifest(
            output_config="unit_config",
            is_method_result=True,
            is_diagnostic_only=False,
            uses_gt=False,
            pre_points_policy="recompute",
            support_policy="unit",
            extra={"eval_policy": "own_recompute_paper_style", "support_source": "own"},
        )

        self.assertFalse(manifest["uses_gt_for_prediction"])
        self.assertFalse(manifest["uses_gt_for_diagnostic"])
        self.assertEqual(manifest["eval_policy"], "own_recompute_paper_style")
        self.assertEqual(manifest["support_source"], "own")
        self.assertEqual(manifest["geometry_source"], "rgbd_eval_bridge")

    def test_v10_scanner_rejects_diagnostic_gt_as_method_result(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_prediction_manifest(
                root=root,
                output_config="bad_method",
                is_method_result=True,
                is_diagnostic_only=False,
                uses_gt=False,
                gt_usage="none",
                pre_points_policy="recompute",
                support_policy="unit",
                extra={
                    "eval_policy": "own_recompute_paper_style",
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic": True,
                },
            )
            write_prediction_manifest("bad_method", manifest, root=root)

            payload = scan_configs(root=root, configs=["bad_method"])

        self.assertEqual(payload["summary"]["num_uses_gt_for_diagnostic_and_method_result"], 1)
        self.assertEqual(payload["summary"]["num_suspicious_configs"], 1)


if __name__ == "__main__":
    unittest.main()
