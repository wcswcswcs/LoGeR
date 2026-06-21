from __future__ import annotations

import unittest

from stream4d_native.v49_mosaic_stage1 import build_shared_observation


class TestV49SharedObservationNoMerge(unittest.TestCase):
    def test_shared_observation_is_never_identity_edge(self) -> None:
        payload = build_shared_observation()
        self.assertTrue(payload["gate"]["shared_observation_no_identity_merge"])
        self.assertIn("does not create identity edges", payload["note"])


if __name__ == "__main__":
    unittest.main()
