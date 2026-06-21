from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from stream4d_native.v55_history_update import (
    _component_accumulation_pass,
    _component_support_gate_pass,
    _load_list,
    _objectlet_frame_mask_component_counters,
    _objectlet_frame_mask_counters,
    _objectlet_frame_projection_stats,
)


class V55HistoryUpdateTest(unittest.TestCase):
    def test_load_list_handles_empty_and_bad_values(self) -> None:
        self.assertEqual(_load_list(""), [])
        self.assertEqual(_load_list("{bad"), [])

    def test_load_list_converts_items_to_strings(self) -> None:
        self.assertEqual(_load_list('["a", 3]'), ["a", "3"])

    def test_objectlet_frame_mask_counters_filters_needed_ids_and_empty_masks(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "native_rows.csv"
            path.write_text(
                "objectlet_id,frame_id,observed_mask_id,uv_x,uv_y\n"
                "obj-a,1,5,0.1,0.2\n"
                "obj-a,1,5,0.3,0.4\n"
                "obj-b,1,5,0.5,0.6\n"
                "obj-a,2,0,0.7,0.8\n",
                encoding="utf-8",
            )
            self.assertEqual(_objectlet_frame_mask_counters(path, {"obj-a"}), {"obj-a": Counter({(1, 5): 2})})

    def test_objectlet_frame_mask_component_counters_filter_and_count_components(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "native_rows.csv"
            path.write_text(
                "objectlet_id,frame_id,observed_mask_id,component_id\n"
                "obj-a,1,5,c1\n"
                "obj-a,1,5,c1\n"
                "obj-a,1,5,c2\n"
                "obj-b,1,5,c3\n"
                "obj-a,2,0,c4\n"
                "obj-a,2,6,\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _objectlet_frame_mask_component_counters(path, {"obj-a"})["obj-a"][(1, 5)],
                Counter({"c1": 2, "c2": 1}),
            )

    def test_component_accumulation_pass_requires_support_masks_and_frames(self) -> None:
        stats = {"support": 3, "masks": {"m1", "m2"}, "frames": {10, 20}}
        self.assertTrue(_component_accumulation_pass(stats, min_support=3, min_masks=2, min_frames=2))
        self.assertFalse(_component_accumulation_pass(stats, min_support=4, min_masks=2, min_frames=2))
        self.assertFalse(_component_accumulation_pass(stats, min_support=3, min_masks=3, min_frames=2))
        self.assertFalse(_component_accumulation_pass(stats, min_support=3, min_masks=2, min_frames=3))

    def test_component_support_gate_pass_checks_rank_support_and_dominance(self) -> None:
        meta = {"selected_rank": 3, "W_visible": 0.75, "R_mask": 0.12, "is_dominant_component": True}
        self.assertTrue(
            _component_support_gate_pass(
                meta,
                max_selected_rank=3,
                min_w_visible=0.7,
                min_r_mask=0.1,
                require_dominant=True,
            )
        )
        self.assertFalse(
            _component_support_gate_pass(
                meta,
                max_selected_rank=2,
                min_w_visible=0.7,
                min_r_mask=0.1,
                require_dominant=True,
            )
        )
        self.assertFalse(
            _component_support_gate_pass(
                {**meta, "is_dominant_component": False},
                max_selected_rank=3,
                min_w_visible=0.7,
                min_r_mask=0.1,
                require_dominant=True,
            )
        )
        self.assertFalse(
            _component_support_gate_pass(
                None,
                max_selected_rank=3,
                min_w_visible=0.7,
                min_r_mask=0.1,
                require_dominant=True,
            )
        )

    def test_objectlet_frame_projection_stats_builds_bbox_and_centroid_sums(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "native_rows.csv"
            path.write_text(
                "objectlet_id,frame_id,observed_mask_id,uv_x,uv_y\n"
                "obj-a,1,5,0.1,0.2\n"
                "obj-a,1,6,0.3,0.5\n"
                "obj-b,1,5,0.9,0.9\n",
                encoding="utf-8",
            )
            stats = _objectlet_frame_projection_stats(path, {"obj-a"})
            self.assertEqual(stats["obj-a"][1]["n"], 2.0)
            self.assertEqual(stats["obj-a"][1]["sx"], 0.4)
            self.assertEqual(stats["obj-a"][1]["minx"], 0.1)
            self.assertEqual(stats["obj-a"][1]["maxy"], 0.5)


if __name__ == "__main__":
    unittest.main()
