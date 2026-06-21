from __future__ import annotations

import unittest

from stream4d_native.v55_anchor_birth import _load_list


class V55AnchorBirthTest(unittest.TestCase):
    def test_load_list_rejects_bad_json(self) -> None:
        self.assertEqual(_load_list("not json"), [])

    def test_load_list_parses_json_array_as_strings(self) -> None:
        self.assertEqual(_load_list('["c1", 2]'), ["c1", "2"])


if __name__ == "__main__":
    unittest.main()
