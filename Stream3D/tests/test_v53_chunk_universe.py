from __future__ import annotations

import unittest

from stream4d_native.v53_chunk_universe import _chunk_windows


class V53ChunkUniverseTest(unittest.TestCase):
    def test_chunk_windows_use_observation_rank_not_raw_frame_gap(self) -> None:
        frames = [0, 10, 30, 31, 90]
        windows = _chunk_windows(frames, chunk_size=3, chunk_stride=2)
        self.assertEqual(windows[0], (0, 2, [0, 10, 30]))
        self.assertEqual(windows[1], (2, 4, [30, 31, 90]))


if __name__ == "__main__":
    unittest.main()
