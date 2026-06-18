from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from tools.run_v36_external_downstream_assignment import RegionNode
from tools.run_v36_external_mask_source import _save_mask_npz
from tools.run_v37_same_frame_oracle_rgb_split import _boundary_split, _dino_part_compactness, _dino_split
from tools.run_v37_learned_pair_solver import _score_eval_rows
from tools.run_v37_4d_if_allowed import _finalize_4d_summary_rows
from tools.rescore_prediction_scores import rescore_sequence
from tools.run_v37_temporal_curriculum import (
    _aggregate_stage_rows,
    _drop_rgb_incoherent_components,
    _filter_edges_by_rgb,
    _frame_rank_map,
    _isolate_rgb_outlier_nodes,
    _labels_for_components_margin_unknown,
    _select_best_stage,
    _singleton_rgb_incoherent_components,
    _split_components_by_rgb,
    _temporal_delta,
)


class _DummyPairModel:
    def predict_proba(self, _features):
        import numpy as np

        return np.asarray([
            [0.1, 0.9],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.8, 0.2],
        ], dtype=np.float32)


class V37TemporalCurriculumTests(unittest.TestCase):
    def test_dino_split_separates_distinct_patch_token_regions(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        grid = np.zeros((4, 4, 2), dtype=np.float32)
        grid[:, :2, 0] = 1.0
        grid[:, 2:, 1] = 1.0
        args = Namespace(
            dino_min_split_area=1,
            dino_min_patch_fraction=0.1,
            dino_min_token_count=2,
            dino_min_tokens_per_child=1,
            dino_max_splits=2,
            dino_large_area=9999,
            dino_spatial_weight=0.0,
            dino_kmeans_iterations=4,
            dino_max_kmeans_tokens=0,
            dino_min_center_distance=0.1,
            dino_guarded_max_child_count=3,
            dino_guarded_min_child_fraction=0.2,
            min_child_area=4,
        )
        parts = _dino_split(mask, grid, args, force_k=2, guarded=True)
        self.assertEqual(len(parts), 2)
        self.assertEqual(sorted(int(part.sum()) for part in parts), [32, 32])

    def test_dino_compactness_prefers_single_token_region(self) -> None:
        grid = np.zeros((4, 4, 2), dtype=np.float32)
        grid[:, :2, 0] = 1.0
        grid[:, 2:, 1] = 1.0
        args = Namespace(dino_min_patch_fraction=0.1)
        pure = np.zeros((8, 8), dtype=bool)
        pure[:, :4] = True
        mixed = np.ones((8, 8), dtype=bool)
        self.assertGreater(_dino_part_compactness(pure, grid, args), _dino_part_compactness(mixed, grid, args))

    def test_boundary_split_cuts_strong_internal_image_edge(self) -> None:
        mask = np.ones((16, 16), dtype=bool)
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[:, 8:] = 255
        args = Namespace(
            boundary_min_split_area=1,
            boundary_gradient_quantile=0.85,
            boundary_min_gradient=0.01,
            boundary_edge_dilate=0,
            boundary_min_child_fraction=0.10,
            boundary_min_core_coverage=0.50,
            boundary_max_child_count=4,
            min_child_area=16,
        )
        parts = _boundary_split(mask, rgb, args, variant="boundary_edgecut_q85_split")
        self.assertEqual(len(parts), 2)
        self.assertEqual(sorted(int(part.sum()) for part in parts), [96, 96])

    def test_uncompressed_external_mask_npz_remains_loadable(self) -> None:
        mask = np.asarray([[True, False], [False, True]], dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            _save_mask_npz(Path(tmp), "unit", 7, [mask], [0.75], compressed=False)
            with np.load(Path(tmp) / "unit_frame000007_masks.npz") as data:
                self.assertTrue(np.array_equal(np.asarray(data["masks"], dtype=bool), mask[None, ...]))
                self.assertAlmostEqual(float(np.asarray(data["scores"])[0]), 0.75)

    def test_score_eval_rows_uses_true_pair_f1(self) -> None:
        rows = [
            {"diagnostic_same_GT": True, "shared_d4rt_tube_count": 1, "shared_d4rt_jaccard": 0.5, "delta_t": 1, "rgb_similarity": 0.9},
            {"diagnostic_same_GT": True, "shared_d4rt_tube_count": 1, "shared_d4rt_jaccard": 0.5, "delta_t": 1, "rgb_similarity": 0.8},
            {"diagnostic_same_GT": False, "shared_d4rt_tube_count": 0, "shared_d4rt_jaccard": 0.0, "delta_t": 3, "rgb_similarity": 0.2},
            {"diagnostic_same_GT": False, "shared_d4rt_tube_count": 0, "shared_d4rt_jaccard": 0.0, "delta_t": 3, "rgb_similarity": 0.1},
        ]
        summary = _score_eval_rows(_DummyPairModel(), rows)
        self.assertEqual(summary["test_pair_count"], 4)
        self.assertAlmostEqual(summary["test_F1"], 1.0)

    def test_aggregate_stage_rows_offsets_true_labels_across_scenes(self) -> None:
        rows = [
            {
                "scene": "scene_a",
                "stage": "S0",
                "_labels_true": [1, 1],
                "_labels_pred": [0, 0],
                "labeled_tube_count": 2,
                "unknown_tube_ratio": 0.0,
                "masklet_temporal_span_mean": 1.0,
                "same_frame_cannot_link_violations": 0,
                "accepted_edges": 0,
                "candidate_edges": 0,
                "rejected_same_frame_conflict": 0,
            },
            {
                "scene": "scene_b",
                "stage": "S0",
                "_labels_true": [1, 1],
                "_labels_pred": [0, 0],
                "labeled_tube_count": 2,
                "unknown_tube_ratio": 0.0,
                "masklet_temporal_span_mean": 1.0,
                "same_frame_cannot_link_violations": 0,
                "accepted_edges": 0,
                "candidate_edges": 0,
                "rejected_same_frame_conflict": 0,
            },
        ]
        summary = _aggregate_stage_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["ARI"], 1.0)
        self.assertAlmostEqual(summary[0]["purity"], 1.0)
        self.assertAlmostEqual(summary[0]["completeness"], 1.0)

    def test_temporal_delta_uses_available_mask_frame_rank_not_raw_frame_id(self) -> None:
        labels_by_frame = {
            0: np.asarray([1, 1], dtype=np.int32),
            10: np.asarray([1, 1], dtype=np.int32),
            20: np.asarray([1, 1], dtype=np.int32),
        }
        frame_rank = _frame_rank_map(labels_by_frame)
        nodes = [
            RegionNode(0, "scene", "src", "mode", 0, 0, 10),
            RegionNode(1, "scene", "src", "mode", 1, 10, 10),
            RegionNode(2, "scene", "src", "mode", 2, 20, 10),
        ]
        self.assertEqual(_temporal_delta(nodes, 0, 1, frame_rank), 1)
        self.assertEqual(_temporal_delta(nodes, 0, 2, frame_rank), 2)
        self.assertNotEqual(_temporal_delta(nodes, 0, 1, frame_rank), 10)

    def test_rescore_constant_control_does_not_default_to_area_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_dir = root / "data" / "prediction" / "input_class_agnostic"
            pred_dir.mkdir(parents=True)
            masks = np.zeros((8, 3), dtype=bool)
            masks[:2, 0] = True
            masks[:4, 1] = True
            masks[:6, 2] = True
            np.savez_compressed(
                pred_dir / "scene0000_00.npz",
                pred_masks=masks,
                pred_score=np.asarray([0.2, 0.4, 0.9], dtype=np.float32),
                pred_classes=np.asarray([0, 0, 0], dtype=np.int32),
            )
            args = Namespace(
                root=str(root),
                input_config="input",
                output_config="output",
                pred_suffix="_class_agnostic",
                min_area=3,
                max_area=0,
                score_feature="none",
                base_score_mode="constant",
                constant_score=1.0,
                tiebreaker_weight=0.0,
                diagnostic_only=True,
                forbidden_for_method_table=True,
                uses_rgbd_for_prediction=True,
                uses_pose_for_prediction=True,
                uses_scannet_mesh_for_prediction=True,
                alignment_source="unit_eval_bridge",
                alignment_used_for_prediction=True,
                eval_policy="unit_constant_score_control",
            )
            row = rescore_sequence(args, "scene0000_00")
            self.assertEqual(row["num_instances_removed_by_area"], 1.0)
            with np.load(root / "data" / "prediction" / "output_class_agnostic" / "scene0000_00.npz") as data:
                self.assertTrue(np.allclose(data["pred_score"], 1.0))
                self.assertEqual(data["pred_masks"].shape[1], 2)
            manifest = (root / "data" / "prediction" / "output_class_agnostic" / "config_manifest.json").read_text()
            self.assertIn('"base_score_mode": "constant"', manifest)
            self.assertIn('"score_feature": "none"', manifest)
            self.assertIn('"forbidden_for_method_table": true', manifest)

    def test_filter_edges_by_rgb_counts_missing_and_dissimilar_edges(self) -> None:
        edges = [
            (1.0, 2, 0.5, 1, 0, 1),
            (0.5, 1, 0.2, 9, 1, 2),
            (0.2, 1, 0.1, 9, 2, 3),
        ]
        diagnostics = {
            0: {"rgb_mean": [10.0, 10.0, 10.0]},
            1: {"rgb_mean": [11.0, 10.0, 10.0]},
            2: {"rgb_mean": [250.0, 250.0, 250.0]},
        }
        kept, rejected = _filter_edges_by_rgb(edges, diagnostics, min_rgb_similarity=0.95)
        self.assertEqual(kept, [edges[0]])
        self.assertEqual(rejected, 2)

    def test_split_components_by_rgb_refines_mixed_component(self) -> None:
        nodes = [
            RegionNode(0, "scene", "src", "mode", 0, 0, 10),
            RegionNode(1, "scene", "src", "mode", 1, 0, 10),
            RegionNode(2, "scene", "src", "mode", 2, 0, 10),
        ]
        diagnostics = {
            0: {"rgb_mean": [10.0, 10.0, 10.0]},
            1: {"rgb_mean": [12.0, 10.0, 10.0]},
            2: {"rgb_mean": [240.0, 240.0, 240.0]},
        }
        refined, info = _split_components_by_rgb(nodes, [[0, 1, 2]], diagnostics, min_rgb_similarity=0.95)
        self.assertEqual(sorted(sorted(part) for part in refined), [[0, 1], [2]])
        self.assertEqual(info["rgb_split_components"], 1)
        self.assertEqual(info["rgb_split_new_components"], 1)

    def test_drop_rgb_incoherent_components_marks_ambiguous_component_unknown(self) -> None:
        nodes = [
            RegionNode(0, "scene", "src", "mode", 0, 0, 10),
            RegionNode(1, "scene", "src", "mode", 1, 0, 10),
            RegionNode(2, "scene", "src", "mode", 2, 0, 10),
        ]
        diagnostics = {
            0: {"rgb_mean": [10.0, 10.0, 10.0]},
            1: {"rgb_mean": [12.0, 10.0, 10.0]},
            2: {"rgb_mean": [240.0, 240.0, 240.0]},
        }
        kept, info = _drop_rgb_incoherent_components(
            nodes,
            [[0, 1], [0, 2]],
            diagnostics,
            min_pairwise_similarity=0.95,
        )
        self.assertEqual(kept, [[0, 1]])
        self.assertEqual(info["rgb_unknown_components"], 1)
        self.assertEqual(info["rgb_unknown_nodes"], 2)

    def test_isolate_rgb_outlier_nodes_keeps_core_component(self) -> None:
        nodes = [
            RegionNode(0, "scene", "src", "mode", 0, 0, 10),
            RegionNode(1, "scene", "src", "mode", 1, 0, 10),
            RegionNode(2, "scene", "src", "mode", 2, 0, 10),
        ]
        diagnostics = {
            0: {"rgb_mean": [10.0, 10.0, 10.0]},
            1: {"rgb_mean": [12.0, 10.0, 10.0]},
            2: {"rgb_mean": [240.0, 240.0, 240.0]},
        }
        refined, info = _isolate_rgb_outlier_nodes(
            nodes,
            [[0, 1, 2]],
            diagnostics,
            min_center_similarity=0.95,
        )
        self.assertEqual(sorted(sorted(part) for part in refined), [[0, 1], [2]])
        self.assertEqual(info["rgb_outlier_components"], 1)
        self.assertEqual(info["rgb_outlier_nodes"], 1)

    def test_singleton_rgb_incoherent_components_keeps_coverage(self) -> None:
        nodes = [
            RegionNode(0, "scene", "src", "mode", 0, 0, 10),
            RegionNode(1, "scene", "src", "mode", 1, 0, 10),
            RegionNode(2, "scene", "src", "mode", 2, 0, 10),
        ]
        diagnostics = {
            0: {"rgb_mean": [10.0, 10.0, 10.0]},
            1: {"rgb_mean": [12.0, 10.0, 10.0]},
            2: {"rgb_mean": [240.0, 240.0, 240.0]},
        }
        refined, info = _singleton_rgb_incoherent_components(
            nodes,
            [[0, 1, 2]],
            diagnostics,
            min_center_similarity=0.95,
        )
        self.assertEqual(sorted(sorted(part) for part in refined), [[0], [1], [2]])
        self.assertEqual(info["rgb_singleton_components"], 1)
        self.assertEqual(info["rgb_singleton_nodes"], 3)

    def test_labels_for_components_margin_unknown_rejects_low_margin_tube(self) -> None:
        labels, unknown, info = _labels_for_components_margin_unknown(
            [[0], [1]],
            {
                10: {0: 5, 1: 4},
                11: {0: 8, 1: 1},
            },
            {10: 9, 11: 9},
            {10: 100, 11: 101},
            min_support=1,
            min_fraction=0.05,
            min_margin_fraction=0.20,
        )
        self.assertNotEqual(labels[10], 0)
        self.assertEqual(labels[11], 0)
        self.assertAlmostEqual(unknown, 0.5)
        self.assertEqual(info["tube_margin_unknown"], 1)

    def test_select_best_stage_prefers_gate_pass_over_higher_ari_fail(self) -> None:
        best, selection = _select_best_stage([
            {"stage": "E4d_rgb090_chain_dt1_component_split", "ARI": 0.4386, "pass_3D_gate": False},
            {
                "stage": "F31_rgb090_component_split_adaptive_density010_frac060",
                "ARI": 0.4261,
                "pass_3D_gate": True,
            },
        ])
        self.assertEqual(best["stage"], "F31_rgb090_component_split_adaptive_density010_frac060")
        self.assertEqual(selection["passing_stage_count"], 1)
        self.assertEqual(selection["selection_policy"], "max_ARI_among_pass_3D_gate_candidates")

    def test_finalize_4d_summary_prefers_gate_pass_over_higher_ari_fail(self) -> None:
        rows = [
            {
                "variant": "I3_span_fail_high_ari",
                "4D_ARI": 0.60,
                "pass_4D_local_ARI_tolerance": True,
                "pass_4D_purity_tolerance": True,
                "pass_4D_temporal_span": False,
            },
            {
                "variant": "I2_gate_pass_lower_ari",
                "4D_ARI": 0.45,
                "pass_4D_local_ARI_tolerance": True,
                "pass_4D_purity_tolerance": True,
                "pass_4D_temporal_span": True,
            },
            {
                "variant": "I5_no_temporal_control",
                "4D_ARI": 0.10,
                "pass_4D_local_ARI_tolerance": False,
                "pass_4D_purity_tolerance": True,
                "pass_4D_temporal_span": False,
            },
        ]
        passing, best, selection_policy = _finalize_4d_summary_rows(rows)
        self.assertFalse(rows[0]["pass_4D_gate"])
        self.assertTrue(rows[1]["pass_4D_gate"])
        self.assertFalse(rows[2]["real_wins_controls"])
        self.assertEqual([row["variant"] for row in passing], ["I2_gate_pass_lower_ari"])
        self.assertEqual(best["variant"], "I2_gate_pass_lower_ari")
        self.assertEqual(selection_policy, "max_4D_ARI_among_passing_variants")


if __name__ == "__main__":
    unittest.main()
