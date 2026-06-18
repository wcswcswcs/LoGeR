from __future__ import annotations

import unittest

from stream4d_native.v37_object_field_adapter import _metric_deltas


class V432AdapterParityTest(unittest.TestCase):
    def test_metric_deltas_are_zero_for_identity_adapter(self) -> None:
        metrics = {
            "4D_ARI": 0.42599481039581194,
            "4D_purity": 0.8673519940549913,
            "4D_completeness": 0.5056972999752292,
            "temporal_span_mean": 1.702673104336451,
            "scene0081_ARI": 0.20073910315166837,
        }
        deltas = _metric_deltas(metrics, metrics)
        self.assertEqual(deltas["delta_4D_ARI"], 0.0)
        self.assertEqual(deltas["delta_4D_purity"], 0.0)
        self.assertEqual(deltas["delta_4D_completeness"], 0.0)


if __name__ == "__main__":
    unittest.main()
