import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.run_v22_direct_reconstruction_benchmark import (
    VARIANTS,
    VariantSpec,
    _apply_variant_xyz_to_windows,
    _estimate_ref0_local_scale,
    _estimate_ref0_pose_trajectory_scale,
    _spec_outputs_world,
    _transform_xyz_hypothesis,
)
from tools.diagnose_v22_ref0_trajectory_scale import (
    _rotation_error_deg,
    _translation_direction_error_deg,
)
from tools.diagnose_v22_ref0_trajectory_policy_sweep import (
    AnchorPolicy,
    _apply_source_policy,
)
from tools.diagnose_v22_ref0_scale_convention import (
    _median_positive,
    _safe_ratio,
)
from tools.diagnose_v22_ref0_intrinsics_proxy import _estimate_intrinsics_params_from_query_geometry
from tools.diagnose_v22_loger_scale_proxy import (
    _candidate_error_rows,
    _sample_pointmap_uv,
)
from tools.diagnose_v22_loss_scale_invariance import (
    _loss_space_l1,
    _metric_l1,
)
from tools.diagnose_v22_target_scale_observability import (
    _leave_one_scene_out_predictions,
    _safe_absrel_summary,
    _spearman_corr,
)
from tools.diagnose_v22_opend4rt_scale_metadata import (
    _extract_model_output_keys,
    _find_scale_like_keys,
    _has_explicit_scale_head,
    _loss_has_independent_mean_depth_normalization,
)
from tools.diagnose_v22_self_supervised_scale_sensitivity import (
    _best_scale,
    _sweep_scale_metrics,
)
from tools.diagnose_v22_scale_anchor_tolerance import _relative_scale_error, _scale_fit


class V22DirectReconstructionTests(unittest.TestCase):
    def test_signed_log1p_transform_matches_loss_space_hypothesis(self):
        xyz = np.array([[[-2.0, -0.5, 0.0], [0.5, 2.0, 10.0]]], dtype=np.float32)
        expected = np.sign(xyz) * np.log1p(np.abs(xyz))

        actual = _transform_xyz_hypothesis(xyz, "signed_log1p")

        np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=1e-6, atol=1e-6)

    def test_apply_variant_xyz_uses_xyz_local_from_npz(self):
        xyz_ref = np.full((1, 2, 3), 9.0, dtype=np.float32)
        xyz_local = np.array([[[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]], dtype=np.float32)
        expected = np.sign(xyz_local) * np.log1p(np.abs(xyz_local))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "carriers_window000.npz"
            np.savez(path, xyz_ref=xyz_ref, xyz_local=xyz_local)
            window = SimpleNamespace(path=path, xyz=np.zeros_like(xyz_ref))
            spec = VariantSpec(
                name="T",
                label="test",
                cache_root="unused",
                provider_mode="raw",
                xyz_field="xyz_local",
                xyz_transform="signed_log1p",
            )

            _apply_variant_xyz_to_windows([window], spec)

            np.testing.assert_allclose(window.xyz, expected.astype(np.float32), rtol=1e-6, atol=1e-6)

    def test_eval_sim3_xyz_local_variants_are_registered(self):
        variants = {spec.name: spec for spec in VARIANTS}

        self.assertEqual(variants["R20"].provider_mode, "eval_sim3")
        self.assertEqual(variants["R20"].xyz_field, "xyz_local")
        self.assertEqual(variants["R20"].xyz_transform, "raw")
        self.assertEqual(variants["R21"].provider_mode, "eval_sim3")
        self.assertEqual(variants["R21"].xyz_field, "xyz_local")
        self.assertEqual(variants["R21"].xyz_transform, "signed_log1p")

    def test_ref0_pose_variants_are_registered_as_world_outputs(self):
        variants = {spec.name: spec for spec in VARIANTS}

        self.assertEqual(variants["R22"].provider_mode, "ref0_pose")
        self.assertTrue(_spec_outputs_world(variants["R22"]))
        self.assertEqual(variants["R23"].provider_mode, "ref0_pose_scale")
        self.assertTrue(_spec_outputs_world(variants["R23"]))

    def test_ref0_local_scale_variants_are_registered_as_world_outputs(self):
        variants = {spec.name: spec for spec in VARIANTS}

        self.assertEqual(variants["R24"].provider_mode, "ref0_pose_scale_local_median_norm")
        self.assertTrue(_spec_outputs_world(variants["R24"]))
        self.assertEqual(variants["R25"].provider_mode, "ref0_pose_scale_local_rms_norm")
        self.assertTrue(_spec_outputs_world(variants["R25"]))
        self.assertEqual(variants["R26"].provider_mode, "ref0_pose_scale_source_z")
        self.assertTrue(_spec_outputs_world(variants["R26"]))
        self.assertEqual(variants["R27"].provider_mode, "ref0_pose_scale_pose_trajectory")
        self.assertTrue(_spec_outputs_world(variants["R27"]))

    def test_ref0_local_scale_estimators_use_xyz_local_without_gt_depth(self):
        xyz_ref = np.array(
            [
                [[1.0, 0.0, 2.0], [0.0, 2.0, 4.0], [3.0, 0.0, 6.0], [0.0, 4.0, 8.0]],
                [[2.0, 0.0, 6.0], [0.0, 4.0, 8.0], [5.0, 0.0, 10.0], [0.0, 6.0, 12.0]],
            ],
            dtype=np.float32,
        )
        xyz_local = xyz_ref * 2.0
        src_frame = np.array([0, 1, 0, 1], dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "carriers_window000.npz"
            np.savez(path, xyz_ref=xyz_ref, xyz_local=xyz_local, src_frame=src_frame)
            window = SimpleNamespace(
                path=path,
                valid=np.ones(xyz_ref.shape[:2], dtype=bool),
                visibility=np.ones(xyz_ref.shape[:2], dtype=np.float32),
                confidence=np.ones(xyz_ref.shape[:2], dtype=np.float32),
            )

            for mode in ("local_median_norm", "local_rms_norm", "source_z"):
                scale, diag = _estimate_ref0_local_scale(window, mode=mode, max_anchors=16)

                self.assertEqual(diag["ref0_local_scale_status"], "ok")
                self.assertAlmostEqual(scale, 2.0, places=6)

    def test_ref0_pose_trajectory_scale_uses_pose_baseline_without_depth(self):
        xyz_ref = np.array(
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        xyz_local = xyz_ref.copy()
        xyz_local[1, :, 0] += 1.0
        xyz_local[2, :, 0] += 2.0

        class FakeStream:
            def load_pose(self, frame_id):
                pose = np.eye(4, dtype=np.float64)
                pose[0, 3] = 2.0 * float(frame_id)
                return pose

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "carriers_window000.npz"
            np.savez(path, xyz_local=xyz_local)
            window = SimpleNamespace(
                path=path,
                frame_ids=[0, 1, 2],
                xyz=xyz_ref,
                uv=np.full(xyz_ref.shape[:2] + (2,), 0.5, dtype=np.float32),
                valid=np.ones(xyz_ref.shape[:2], dtype=bool),
                visibility=np.ones(xyz_ref.shape[:2], dtype=np.float32),
                confidence=np.ones(xyz_ref.shape[:2], dtype=np.float32),
            )

            scale, diag = _estimate_ref0_pose_trajectory_scale(FakeStream(), window, max_anchors=16)

            self.assertEqual(diag["ref0_trajectory_scale_status"], "ok")
            self.assertAlmostEqual(scale, 2.0, places=6)

    def test_ref0_trajectory_direction_helpers_report_degrees(self):
        rot_z_90 = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.assertAlmostEqual(_rotation_error_deg(np.eye(3), rot_z_90), 90.0, places=6)
        self.assertAlmostEqual(
            _translation_direction_error_deg(
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
            ),
            90.0,
            places=6,
        )

    def test_ref0_trajectory_source_policy_masks(self):
        base_ok = np.array([True, True, True, False, True], dtype=bool)
        src_frame = np.array([0, 2, 3, 0, 2], dtype=np.int64)

        np.testing.assert_array_equal(
            _apply_source_policy(base_ok, src_frame, 2, AnchorPolicy("ref", source_mode="ref_source")),
            np.array([True, False, False, False, False], dtype=bool),
        )
        np.testing.assert_array_equal(
            _apply_source_policy(base_ok, src_frame, 2, AnchorPolicy("target", source_mode="target_source")),
            np.array([False, True, False, False, True], dtype=bool),
        )
        np.testing.assert_array_equal(
            _apply_source_policy(base_ok, src_frame, 2, AnchorPolicy("nonref", source_mode="nonref_source")),
            np.array([False, True, True, False, True], dtype=bool),
        )

    def test_ref0_scale_convention_helpers_ignore_invalid_values(self):
        values = np.array([np.nan, -1.0, 0.0, 2.0, 4.0], dtype=np.float64)

        self.assertAlmostEqual(_median_positive(values), 3.0, places=6)
        self.assertAlmostEqual(_safe_ratio(6.0, 3.0), 2.0, places=6)
        self.assertIsNone(_safe_ratio(1.0, 0.0))

    def test_query_intrinsics_estimator_is_uniform_scale_invariant(self):
        image_hw = (101, 101)
        uv = np.array([[0.70, 0.75], [0.80, 0.90]], dtype=np.float64)
        xyz = np.array(
            [
                [2.0, 3.125, 10.0],
                [1.5, 2.5, 5.0],
            ],
            dtype=np.float64,
        )

        params, diag = _estimate_intrinsics_params_from_query_geometry(xyz, uv, image_hw)
        scaled_params, scaled_diag = _estimate_intrinsics_params_from_query_geometry(xyz * 2.5, uv, image_hw)

        self.assertEqual(diag["fx_count"], 2)
        self.assertEqual(diag["fy_count"], 2)
        self.assertEqual(scaled_diag["fx_count"], 2)
        self.assertAlmostEqual(float(params[0]), 100.0, places=6)
        self.assertAlmostEqual(float(params[1]), 80.0, places=6)
        np.testing.assert_allclose(scaled_params[:2], params[:2], rtol=1e-6, atol=1e-6)

    def test_loger_scale_proxy_uv_sampler_uses_normalized_coordinates(self):
        pointmap = np.zeros((3, 5, 3), dtype=np.float32)
        pointmap[0, 0] = [1.0, 2.0, 3.0]
        pointmap[2, 4] = [4.0, 5.0, 6.0]
        uv = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]], dtype=np.float64)

        sampled, ok = _sample_pointmap_uv(pointmap, uv)

        np.testing.assert_array_equal(ok, np.array([True, True, False]))
        np.testing.assert_allclose(sampled[:2], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_loger_scale_proxy_candidate_rows_mark_gt_controls(self):
        summaries = [
            {
                "scene": "scene_test",
                "loger_z_over_d4rt_local_z_median_median_abs_rel_vs_eval_scale": 0.1,
                "loger_z_over_d4rt_local_z_median_median": 0.9,
                "scannet_depth_over_d4rt_local_z_median_median_abs_rel_vs_eval_scale": 0.0,
                "scannet_depth_over_d4rt_local_z_median_median": 1.0,
            }
        ]

        rows = {row["candidate"]: row for row in _candidate_error_rows(summaries)}

        self.assertFalse(rows["loger_z_over_d4rt_local_z_median"]["uses_scannet_depth_for_proxy"])
        self.assertTrue(rows["scannet_depth_over_d4rt_local_z_median"]["uses_scannet_depth_for_proxy"])

    def test_d4rt_loss_space_l1_is_uniform_pred_scale_invariant(self):
        pred = np.array(
            [
                [0.5, -0.25, 1.0],
                [1.0, 0.5, 2.0],
                [-0.25, 0.75, 3.0],
            ],
            dtype=np.float64,
        )
        gt = np.array(
            [
                [0.6, -0.2, 2.0],
                [1.2, 0.4, 4.0],
                [-0.3, 0.8, 6.0],
            ],
            dtype=np.float64,
        )

        low = _loss_space_l1(pred * 0.25, gt, normalize_depth=True, transform_log=True)
        high = _loss_space_l1(pred * 4.0, gt, normalize_depth=True, transform_log=True)

        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertAlmostEqual(float(low), float(high), places=12)

    def test_metric_l1_changes_under_uniform_pred_scale(self):
        pred = np.array([[0.5, -0.25, 1.0], [1.0, 0.5, 2.0]], dtype=np.float64)
        gt = np.array([[1.0, -0.5, 2.0], [2.0, 1.0, 4.0]], dtype=np.float64)

        unscaled = _metric_l1(pred, gt)
        matched = _metric_l1(pred * 2.0, gt)

        self.assertIsNotNone(unscaled)
        self.assertIsNotNone(matched)
        self.assertGreater(float(unscaled), 0.0)
        self.assertAlmostEqual(float(matched), 0.0, places=12)

    def test_target_scale_observability_absrel_summary(self):
        summary = _safe_absrel_summary(
            np.array([2.0, 4.0, np.nan], dtype=np.float64),
            np.array([1.0, 8.0, 2.0], dtype=np.float64),
        )

        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(float(summary["mean_absrel"]), 0.75, places=6)
        self.assertAlmostEqual(float(summary["median_absrel"]), 0.75, places=6)
        self.assertAlmostEqual(float(summary["max_absrel"]), 1.0, places=6)

    def test_target_scale_observability_loo_linear_predicts_synthetic_scale(self):
        rows = []
        for scene_index, scene in enumerate(["scene_a", "scene_b", "scene_c"]):
            for x in [1.0, 2.0, 3.0]:
                value = x + float(scene_index)
                rows.append(
                    {
                        "scene": scene,
                        "window_index": 0,
                        "frame_id": int(value),
                        "local_idx": int(value),
                        "feature": value,
                        "target_depth_over_local_z_median": 1.0 + 2.0 * value,
                    }
                )

        preds = _leave_one_scene_out_predictions(
            rows,
            label_key="target_depth_over_local_z_median",
            feature_keys=["feature"],
        )
        summary = _safe_absrel_summary(
            np.array([row["predicted_scale"] for row in preds], dtype=np.float64),
            np.array([row["target_scale"] for row in preds], dtype=np.float64),
        )

        self.assertEqual(summary["count"], len(rows))
        self.assertLess(float(summary["mean_absrel"]), 1e-10)

    def test_target_scale_observability_spearman_handles_monotonic_ties(self):
        corr = _spearman_corr(
            np.array([1.0, 2.0, 2.0, 3.0], dtype=np.float64),
            np.array([10.0, 20.0, 20.0, 30.0], dtype=np.float64),
        )

        self.assertIsNotNone(corr)
        self.assertAlmostEqual(float(corr), 1.0, places=6)

    def test_self_supervised_scale_sweep_marks_only_gt_depth_as_scale_sensitive(self):
        image_hw = (101, 101)
        intrinsics = np.array(
            [
                [100.0, 0.0, 50.0],
                [0.0, 80.0, 50.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        xyz = np.array(
            [
                [1.0, 1.0, 2.0],
                [2.0, 1.5, 4.0],
                [1.5, 2.5, 5.0],
                [3.0, 2.0, 8.0],
            ],
            dtype=np.float64,
        )
        uv_px = np.column_stack(
            [
                intrinsics[0, 0] * xyz[:, 0] / xyz[:, 2] + intrinsics[0, 2],
                intrinsics[1, 1] * xyz[:, 1] / xyz[:, 2] + intrinsics[1, 2],
            ]
        )
        uv = np.column_stack(
            [
                uv_px[:, 0] / float(image_hw[1] - 1),
                uv_px[:, 1] / float(image_hw[0] - 1),
            ]
        )
        target_depth = xyz[:, 2] * 2.0

        rows = _sweep_scale_metrics(
            xyz_local=xyz,
            uv=uv,
            target_depth=target_depth,
            intrinsics=intrinsics,
            image_hw=image_hw,
            pred_scales=[0.5, 1.0, 2.0],
        )

        self.assertLess(max(row["uv_reprojection_median_px"] for row in rows), 1e-10)
        self.assertLess(max(row["normalized_z_l1"] for row in rows), 1e-12)
        self.assertLess(max(abs(float(row["depth_rank_spearman"]) - 1.0) for row in rows), 1e-12)
        self.assertEqual(_best_scale(rows, "gt_depth_absrel"), 2.0)
        self.assertLess(float([row for row in rows if row["pred_scale"] == 2.0][0]["gt_depth_absrel"]), 1e-12)
        self.assertGreater(float([row for row in rows if row["pred_scale"] == 1.0][0]["gt_depth_absrel"]), 0.0)

    def test_scale_anchor_tolerance_helpers_scale_without_mutating_fit(self):
        fit = {
            "scale": 2.0,
            "rotation": np.eye(3, dtype=np.float64),
            "translation": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        }

        scaled = _scale_fit(fit, 0.75)

        self.assertIsNotNone(scaled)
        self.assertAlmostEqual(float(scaled["scale"]), 1.5, places=6)
        self.assertAlmostEqual(float(fit["scale"]), 2.0, places=6)
        self.assertAlmostEqual(_relative_scale_error(0.75), 0.25, places=6)
        self.assertAlmostEqual(_relative_scale_error(1.25), 0.25, places=6)

    def test_opend4rt_scale_metadata_key_detector_ignores_xyz_keys(self):
        keys = ["xyz_3d", "tracks_xyz_local", "tracks_uv_norm", "confidence", "target_mean_depth_scale"]

        self.assertEqual(_find_scale_like_keys(keys), ["target_mean_depth_scale"])

    def test_opend4rt_scale_metadata_detects_explicit_scale_head(self):
        text = '''
        return {
            "xyz_3d": self.xyz_head(decoded_queries),
            "uv_2d": self.uv_head(decoded_queries),
            "mean_depth_scale": self.scale_head(decoded_queries),
        }
        '''
        output_keys = _extract_model_output_keys(text)

        self.assertEqual(output_keys, ["mean_depth_scale", "uv_2d", "xyz_3d"])
        self.assertTrue(_has_explicit_scale_head(output_keys))

    def test_opend4rt_scale_metadata_detects_independent_loss_normalization(self):
        text = """
        scale = masked_mean_per_sample(depth, mask).clamp_min(1e-6).unsqueeze(-1)
        out = out / scale
        pred = self._xyz_preprocess(pred, m, use_norm, use_log)
        gt = self._xyz_preprocess(gt, m, use_norm, use_log)
        """

        self.assertTrue(_loss_has_independent_mean_depth_normalization(text))


if __name__ == "__main__":
    unittest.main()
