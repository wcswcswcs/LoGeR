from __future__ import annotations

import unittest

from stream4d_native.v64r2_scannet_exporters import build_v64r2_ap_contract


class V64R2ScanNetExportersTest(unittest.TestCase):
    def test_ap_smoke_can_run_with_diagnostic_bridge_when_method_safe_blocked(self) -> None:
        payload = build_v64r2_ap_contract()
        gate = payload["summary"]["gate"]
        self.assertTrue(gate["diagnostic_bridge_available"])
        self.assertTrue(gate["ap_smoke_can_run"])


if __name__ == "__main__":
    unittest.main()
