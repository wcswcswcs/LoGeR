from __future__ import annotations

import unittest

from stream4d_native.v44_typed_mask_assembly import infer_roles


class V44MaskRoleGraphTest(unittest.TestCase):
    def test_large_boundary_mask_can_be_marked_mixed(self) -> None:
        tokens = [
            {"token_id": 1, "area": 10, "boundary_contrast": 0.1},
            {"token_id": 2, "area": 1000, "boundary_contrast": 0.9},
            {"token_id": 3, "area": 500, "boundary_contrast": 0.3},
        ]
        roles = infer_roles(tokens, [], require_two_signal_mixed=False)
        self.assertEqual(roles[2]["role"], "mixed")


if __name__ == "__main__":
    unittest.main()
