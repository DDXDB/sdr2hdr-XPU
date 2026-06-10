import unittest

import numpy as np

from scripts.prepare_data import DEGRADATION_PROFILES, tone_map_hdr_linear_to_sdr_linear


class PrepareDataTests(unittest.TestCase):
    def test_tone_map_hdr_linear_to_sdr_linear_returns_bounded_frame(self) -> None:
        frame = np.full((8, 8, 3), 0.75, dtype=np.float32)
        out = tone_map_hdr_linear_to_sdr_linear(frame)
        self.assertEqual(out.shape, frame.shape)
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))

    def test_degradation_profiles_produce_distinct_sdr_variants(self) -> None:
        frame = np.full((8, 8, 3), [0.35, 0.50, 0.80], dtype=np.float32)
        natural = tone_map_hdr_linear_to_sdr_linear(frame, profile_name="natural", variant_seed=1)
        clipped = tone_map_hdr_linear_to_sdr_linear(frame, profile_name="clipped", variant_seed=1)
        self.assertEqual(set(DEGRADATION_PROFILES), {"natural", "clipped", "compressed", "night"})
        self.assertGreater(float(np.mean(np.abs(natural - clipped))), 1e-4)


if __name__ == "__main__":
    unittest.main()
