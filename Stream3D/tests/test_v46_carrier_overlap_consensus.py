from __future__ import annotations

import unittest

import numpy as np

from tools.run_v46_raw_carrier_incidence_repair import MaskNode, WindowTrace
from tools.run_v46_supporter_quality_raw_repair import _supporter_fanout_for_variant, _weighted_view_consensus


class V46CarrierOverlapConsensusTest(unittest.TestCase):
    def test_common_wide_mask_without_carrier_overlap_is_downweighted(self) -> None:
        window = WindowTrace(
            window_index=0,
            path="synthetic",
            frame_ids=[1],
            carrier_ids=np.arange(8),
            visible=np.ones((1, 8), dtype=bool),
            labels_by_frame={1: np.array([9, 9, 9, 9, 9, 9, 9, 9], dtype=np.int32)},
        )
        left = MaskNode(
            node_id=1,
            scene="scene_test",
            frame_id=0,
            mask_id=1,
            area=100,
            inc_by_window={0: {0, 1, 2, 3}},
        )
        right = MaskNode(
            node_id=2,
            scene="scene_test",
            frame_id=2,
            mask_id=2,
            area=100,
            inc_by_window={0: {4, 5, 6, 7}},
        )
        quality = {(1, 9): {"Q5_split_outside_fragment_soft": 1.0}}

        ungated, *_ = _weighted_view_consensus(
            left,
            right,
            {0: window},
            quality,
            variant="Q5_split_outside_fragment_soft",
            min_visible_carriers=1,
            observer_frame_mode="all",
            near_endpoint_frame_gap=10,
        )
        overlap_gated, *_ = _weighted_view_consensus(
            left,
            right,
            {0: window},
            quality,
            variant="Q5_split_outside_fragment_soft_carrier_overlap",
            min_visible_carriers=1,
            observer_frame_mode="all",
            near_endpoint_frame_gap=10,
        )

        self.assertEqual(ungated, 1.0)
        self.assertEqual(overlap_gated, 0.0)

        soft25, *_ = _weighted_view_consensus(
            left,
            right,
            {0: window},
            quality,
            variant="Q5_split_outside_fragment_soft_carrier_overlap_soft25",
            min_visible_carriers=1,
            observer_frame_mode="all",
            near_endpoint_frame_gap=10,
        )
        soft50, *_ = _weighted_view_consensus(
            left,
            right,
            {0: window},
            quality,
            variant="Q5_split_outside_fragment_soft_carrier_overlap_soft50",
            min_visible_carriers=1,
            observer_frame_mode="all",
            near_endpoint_frame_gap=10,
        )

        self.assertEqual(soft25, 0.75)
        self.assertEqual(soft50, 0.5)

    def test_common_mask_with_shared_carriers_keeps_fractional_overlap(self) -> None:
        window = WindowTrace(
            window_index=0,
            path="synthetic",
            frame_ids=[1],
            carrier_ids=np.arange(6),
            visible=np.ones((1, 6), dtype=bool),
            labels_by_frame={1: np.array([9, 9, 9, 9, 9, 9], dtype=np.int32)},
        )
        left = MaskNode(
            node_id=1,
            scene="scene_test",
            frame_id=0,
            mask_id=1,
            area=100,
            inc_by_window={0: {0, 1, 2, 3}},
        )
        right = MaskNode(
            node_id=2,
            scene="scene_test",
            frame_id=2,
            mask_id=2,
            area=100,
            inc_by_window={0: {2, 3, 4, 5}},
        )
        quality = {(1, 9): {"Q5_split_outside_fragment_soft": 1.0}}

        score, *_ = _weighted_view_consensus(
            left,
            right,
            {0: window},
            quality,
            variant="Q5_split_outside_fragment_soft_carrier_overlap",
            min_visible_carriers=1,
            observer_frame_mode="all",
            near_endpoint_frame_gap=10,
        )

        self.assertEqual(score, 0.5)

    def test_hubsoft_downweights_reused_supporter(self) -> None:
        window = WindowTrace(
            window_index=0,
            path="synthetic",
            frame_ids=[1],
            carrier_ids=np.arange(8),
            visible=np.ones((1, 8), dtype=bool),
            labels_by_frame={1: np.array([9, 9, 9, 9, 9, 9, 9, 9], dtype=np.int32)},
        )
        left = MaskNode(
            node_id=1,
            scene="scene_test",
            frame_id=0,
            mask_id=1,
            area=100,
            inc_by_window={0: {0, 1, 2, 3}},
        )
        right = MaskNode(
            node_id=2,
            scene="scene_test",
            frame_id=2,
            mask_id=2,
            area=100,
            inc_by_window={0: {4, 5, 6, 7}},
        )
        quality = {(1, 9): {"Q5_split_outside_fragment_soft": 1.0}}

        score, *_ = _weighted_view_consensus(
            left,
            right,
            {0: window},
            quality,
            variant="Q5_split_outside_fragment_soft_hubsoft_q005",
            min_visible_carriers=1,
            observer_frame_mode="all",
            near_endpoint_frame_gap=10,
            supporter_fanout={(1, 9): 4},
        )

        self.assertAlmostEqual(score, 0.5)

    def test_hubcap_only_downweights_extreme_reuse(self) -> None:
        window = WindowTrace(
            window_index=0,
            path="synthetic",
            frame_ids=[1],
            carrier_ids=np.arange(8),
            visible=np.ones((1, 8), dtype=bool),
            labels_by_frame={1: np.array([9, 9, 9, 9, 9, 9, 9, 9], dtype=np.int32)},
        )
        left = MaskNode(
            node_id=1,
            scene="scene_test",
            frame_id=0,
            mask_id=1,
            area=100,
            inc_by_window={0: {0, 1, 2, 3}},
        )
        right = MaskNode(
            node_id=2,
            scene="scene_test",
            frame_id=2,
            mask_id=2,
            area=100,
            inc_by_window={0: {4, 5, 6, 7}},
        )
        quality = {(1, 9): {"Q5_split_outside_fragment_soft": 1.0}}

        low_fanout_score, *_ = _weighted_view_consensus(
            left,
            right,
            {0: window},
            quality,
            variant="Q5_split_outside_fragment_soft_hubcap32_q020",
            min_visible_carriers=1,
            observer_frame_mode="all",
            near_endpoint_frame_gap=10,
            supporter_fanout={(1, 9): 4},
        )
        high_fanout_score, *_ = _weighted_view_consensus(
            left,
            right,
            {0: window},
            quality,
            variant="Q5_split_outside_fragment_soft_hubcap32_q020",
            min_visible_carriers=1,
            observer_frame_mode="all",
            near_endpoint_frame_gap=10,
            supporter_fanout={(1, 9): 128},
        )

        self.assertEqual(low_fanout_score, 1.0)
        self.assertAlmostEqual(high_fanout_score, 0.5)

    def test_hubsoft_fanout_counts_distinct_edges(self) -> None:
        window = WindowTrace(
            window_index=0,
            path="synthetic",
            frame_ids=[1],
            carrier_ids=np.arange(6),
            visible=np.ones((1, 6), dtype=bool),
            labels_by_frame={1: np.array([9, 9, 9, 9, 9, 9], dtype=np.int32)},
        )
        nodes = [
            MaskNode(
                node_id=1,
                scene="scene_test",
                frame_id=0,
                mask_id=1,
                area=100,
                inc_by_window={0: {0, 1}},
            ),
            MaskNode(
                node_id=2,
                scene="scene_test",
                frame_id=2,
                mask_id=2,
                area=100,
                inc_by_window={0: {2, 3}},
            ),
            MaskNode(
                node_id=3,
                scene="scene_test",
                frame_id=4,
                mask_id=3,
                area=100,
                inc_by_window={0: {4, 5}},
            ),
        ]
        quality = {(1, 9): {"Q5_split_outside_fragment_soft": 1.0}}

        fanout = _supporter_fanout_for_variant(
            capped_nodes=nodes,
            windows_by_index={0: window},
            quality_by_key=quality,
            variant="Q5_split_outside_fragment_soft_hubsoft_q005",
            min_visible_carriers=1,
            observer_frame_mode="all",
            near_endpoint_frame_gap=10,
        )

        self.assertEqual(fanout[(1, 9)], 3)


if __name__ == "__main__":
    unittest.main()
