from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from stream4d_native.v64r2_dynamic_env import build_v64r2_dynamic_env


class V64R2DynamicEnvTest(unittest.TestCase):
    def test_declared_masks_do_not_count_when_files_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dynamic-replica" / "v2"
            split = root / "valid"
            scene = split / "seq_source_left"
            (scene / "images").mkdir(parents=True)
            (scene / "trajectories").mkdir()
            (scene / "images" / "seq_source_left-0000.png").write_bytes(b"fake")
            (scene / "trajectories" / "000000.pth").write_bytes(b"fake")
            ann = [
                {
                    "sequence_name": "seq",
                    "frame_number": 0,
                    "image": {"path": "seq_source_left/images/seq_source_left-0000.png"},
                    "mask": {"path": "seq_source_left/masks/seq_source_left_0000.png"},
                    "depth": {"path": "seq_source_left/depths/seq_source_left_0000.geometric.png"},
                    "trajectories": {"path": "seq_source_left/trajectories/000000.pth"},
                    "instance_id_map_path": "seq_source_left/instance_id_maps/seq_source_left_0000.png",
                    "instance_ids": ["obj_a"],
                    "camera_name": "left",
                    "viewpoint": {"R": [], "T": [], "focal_length": [], "principal_point": []},
                }
            ]
            (split / "frame_annotations_valid.json").write_text(json.dumps(ann), encoding="utf-8")
            payload = build_v64r2_dynamic_env(data_root=root, split="valid")
            summary = payload["summary"]
            self.assertEqual(summary["dyn_level_label"], "DYN_LEVEL_1")
            self.assertTrue(summary["rgb_frames_exist"])
            self.assertTrue(summary["object_ids_declared_in_annotations"])
            self.assertFalse(summary["object_ids_exist"])
            self.assertFalse(summary["instance_masks_exist"])
            self.assertFalse(summary["can_report_official_object_tracking"])


if __name__ == "__main__":
    unittest.main()
