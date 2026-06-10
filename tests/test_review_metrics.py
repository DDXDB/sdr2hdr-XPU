import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sdr2hdr.review import compute_frame_metrics, save_metrics_report


class ReviewMetricsTests(unittest.TestCase):
    def test_compute_frame_metrics_reports_gain_and_clip(self) -> None:
        sdr = np.full((4, 4, 3), 32, dtype=np.uint8)
        hdr = np.full((4, 4, 3), 255, dtype=np.uint8)
        metrics = compute_frame_metrics(sdr, hdr)
        self.assertGreater(metrics["mean_luma_gain"], 0.0)
        self.assertGreater(metrics["hdr_highlight_clip_ratio"], metrics["sdr_highlight_clip_ratio"])

    def test_save_metrics_report_writes_summary(self) -> None:
        metrics = [
            {"time_sec": 0.0, "mean_luma_gain": 0.1, "hdr_highlight_clip_ratio": 0.2, "sdr_highlight_clip_ratio": 0.1, "hdr_shadow_detail_std": 0.05, "sdr_shadow_detail_std": 0.02, "mean_chroma_delta": 0.03},
            {"time_sec": 1.0, "mean_luma_gain": 0.3, "hdr_highlight_clip_ratio": 0.4, "sdr_highlight_clip_ratio": 0.1, "hdr_shadow_detail_std": 0.07, "sdr_shadow_detail_std": 0.02, "mean_chroma_delta": 0.05},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            save_metrics_report(metrics, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertIn("temporal_luma_delta_mean", payload)
        self.assertEqual(len(payload["frames"]), 2)


if __name__ == "__main__":
    unittest.main()
