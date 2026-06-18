from __future__ import annotations

import unittest

from stream4d_native.object_field import ObjectBirthConstraintError, ObjectFieldCandidate


class V41NoD4RTTubeBirthTests(unittest.TestCase):
    def test_object_field_candidate_rejects_d4rt_tube_birth(self) -> None:
        candidate = ObjectFieldCandidate(
            candidate_id=7,
            semantic_masklet_ids=(),
            material_tube_ids=(3,),
            score=1.0,
            birth_source="d4rt_tube",
        )
        with self.assertRaises(ObjectBirthConstraintError):
            candidate.validate_birth()

    def test_semantic_masklet_birth_is_allowed(self) -> None:
        candidate = ObjectFieldCandidate(
            candidate_id=8,
            semantic_masklet_ids=(1,),
            material_tube_ids=(3,),
            score=1.0,
            birth_source="semantic_masklet",
        )
        candidate.validate_birth()


if __name__ == "__main__":
    unittest.main()

