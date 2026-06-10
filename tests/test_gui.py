from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace

from sdr2hdr.gui import (
    SDR2HDRGUI,
    build_backend_options,
    build_encoder_options,
    filter_models_for_backend,
    format_ai_strength,
    list_available_models,
)


class GuiTests(unittest.TestCase):
    def test_build_encoder_options_for_windows(self) -> None:
        options = build_encoder_options("Windows")
        self.assertIn("libx265", options)
        self.assertIn("hevc_nvenc", options)
        self.assertNotIn("hevc_videotoolbox", options)

    def test_build_encoder_options_for_macos(self) -> None:
        options = build_encoder_options("Darwin")
        self.assertIn("libx265", options)
        self.assertIn("hevc_videotoolbox", options)
        self.assertNotIn("hevc_nvenc", options)

    def test_build_backend_options_for_windows(self) -> None:
        options = build_backend_options("Windows")
        self.assertIn("auto", options)
        self.assertIn("cuda", options)
        self.assertIn("numpy", options)
        self.assertNotIn("directml", options)

    def test_build_backend_options_for_macos(self) -> None:
        options = build_backend_options("Darwin")
        self.assertIn("auto", options)
        self.assertIn("mps", options)
        self.assertIn("numpy", options)
        self.assertNotIn("cuda", options)

    def test_format_ai_strength(self) -> None:
        self.assertEqual(format_ai_strength(0.25), "0.25")
        self.assertEqual(format_ai_strength(0.2), "0.20")

    def test_list_available_models_returns_pt_only(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.pt").write_bytes(b"pt")
            (root / "b.onnx").write_bytes(b"onnx")
            (root / "notes.txt").write_text("x", encoding="utf-8")
            models = list_available_models(root)
        self.assertEqual([path.name for path in models], ["a.pt"])

    def test_filter_models_for_backend_returns_pt_only(self) -> None:
        models = [Path("a.pt"), Path("b.onnx")]
        filtered = filter_models_for_backend(models, "auto", "Windows")
        self.assertEqual(filtered, [Path("a.pt")])

    def test_build_request_includes_hdr_style(self) -> None:
        app = SDR2HDRGUI.__new__(SDR2HDRGUI)
        app.input_var = SimpleNamespace(get=lambda: "in.mp4")
        app.output_var = SimpleNamespace(get=lambda: "out.mp4")
        app.preset_var = SimpleNamespace(get=lambda: "portrait")
        app.hdr_style_var = SimpleNamespace(get=lambda: "natural")
        app.model_path_var = SimpleNamespace(get=lambda: "models\\model.pt")
        app.ai_strength_var = SimpleNamespace(get=lambda: 0.25)
        app._selected_encoder = lambda: "libx265"
        app._selected_x265_mode = lambda: "balanced"
        app._selected_backend = lambda: "auto"

        request = SDR2HDRGUI._build_request(app)

        self.assertEqual(request.hdr_style, "natural")
        self.assertEqual(request.model_path, "models\\model.pt")


if __name__ == "__main__":
    unittest.main()
