from __future__ import annotations

import unittest

from stream4d_native.stage1_scale_aware_typed_assembly import infer_roles_v45


class V45RoleLatticeTest(unittest.TestCase):
    def test_role_lattice_marks_high_conflict_large_mask_mixed(self) -> None:
        tokens = [
            {"token_id": 1, "area": 10000, "boundary_contrast": 0.9},
            {"token_id": 2, "area": 100, "boundary_contrast": 0.1},
            {"token_id": 3, "area": 120, "boundary_contrast": 0.1},
        ]
        edges = [
            {"token_i": 1, "token_j": 2, "same_frame_cannot_link": "True", "semantic_affinity": 0.1, "object_affinity": 0.1},
            {"token_i": 1, "token_j": 3, "same_frame_cannot_link": "True", "semantic_affinity": 0.1, "object_affinity": 0.1},
        ]
        roles = infer_roles_v45(tokens, edges, profile="R4_full_role_lattice")
        self.assertEqual(roles[1]["role"], "mixed")


if __name__ == "__main__":
    unittest.main()

