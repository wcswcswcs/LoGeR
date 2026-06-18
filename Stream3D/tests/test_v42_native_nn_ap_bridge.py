from __future__ import annotations

import unittest

from tools.run_v42_native_nn_ap_bridge import _parse_radii


class V42NativeNnApBridgeTests(unittest.TestCase):
    def test_parse_radii(self) -> None:
        self.assertEqual(_parse_radii("0.02, 0.1,,0.5"), [0.02, 0.1, 0.5])


if __name__ == "__main__":
    unittest.main()
