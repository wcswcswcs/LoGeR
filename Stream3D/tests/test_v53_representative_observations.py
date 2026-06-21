from __future__ import annotations

import unittest

from stream4d_native.v53_representative_observations import _select_greedy


class V53RepresentativeObservationTest(unittest.TestCase):
    def test_greedy_selects_masks_by_new_component_gain(self) -> None:
        mask_ids = ["m1", "m2", "m3"]
        universe = {"a", "b", "c"}
        components_by_mask = {"m1": {"a", "b"}, "m2": {"b"}, "m3": {"c"}}
        meta = {
            "m1": {"scene": "s", "frame_id": 0, "R0_visible_tau0.05_component_count": 2},
            "m2": {"scene": "s", "frame_id": 1, "R0_visible_tau0.05_component_count": 1},
            "m3": {"scene": "s", "frame_id": 2, "R0_visible_tau0.05_component_count": 1},
        }
        selected, progress = _select_greedy(
            mask_ids=mask_ids,
            universe=universe,
            components_by_mask=components_by_mask,
            meta_by_mask=meta,
            max_selected=2,
            component_count_key="R0_visible_tau0.05_component_count",
        )
        self.assertEqual(selected, ["m1", "m3"])
        self.assertAlmostEqual(progress[-1]["component_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
