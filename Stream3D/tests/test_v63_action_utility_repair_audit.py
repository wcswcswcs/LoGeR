from __future__ import annotations

import unittest

from stream4d_native.v63_action_utility_repair_audit import (
    _counter_from_string,
    _temporal_required_no_temporal_utility,
)


class TestV63ActionUtilityRepairAudit(unittest.TestCase):
    def test_counter_from_json_string(self) -> None:
        self.assertEqual(_counter_from_string('{"defer": 64}'), {"defer": 64})

    def test_temporal_required_no_temporal_does_not_confirm_source_only(self) -> None:
        confirm_score = _temporal_required_no_temporal_utility({"planned_action": "confirm"})
        defer_score = _temporal_required_no_temporal_utility({"planned_action": "defer"})
        self.assertLess(confirm_score, defer_score)


if __name__ == "__main__":
    unittest.main()

