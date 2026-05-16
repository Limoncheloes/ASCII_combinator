from __future__ import annotations

import json
import platform
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

from ascii_combinator.bg_mode import BgMode, SoftBgConfig
from ascii_combinator.compositor import Compositor
from ascii_combinator.layers.brightness import BrightnessLayer
from ascii_combinator.layers.diagonal import DiagonalLayer
from ascii_combinator.layers.sobel_x import SobelXLayer
from ascii_combinator.layers.sobel_y import SobelYLayer
from ascii_combinator.profiles.monochrome import MonochromeProfile
from ascii_combinator.renderer import Renderer

RESULTS_DIR = Path(__file__).parent / "results"

_progress: dict[str, dict] = {}

_LAYER_REGISTRY = {
    "brightness": BrightnessLayer,
    "sobel_x": SobelXLayer,
    "sobel_y": SobelYLayer,
    "diagonal": DiagonalLayer,
}

_PROFILE_REGISTRY = {"monochrome": MonochromeProfile}


def _output_path(stem: str, params: dict, ext: str) -> Path:
    """Build deterministic result filename from params."""
    w = params.get("width", 80)
    f = params.get("font_size", 12)
    bg = params.get("bg_mode", "keep")
    name = f"result_w{w}_f{f}"
    if bg != "keep":
        name += f"_bg-{bg}"
    return RESULTS_DIR / stem / f"{name}.{ext}"


def _convert_image(image: Image.Image, params: dict) -> Image.Image:
    """Run ASCII pipeline on a PIL image, return result PIL image."""
    font_size = params.get("font_size", 12)
    cell_w = max(int(font_size * 0.6), 1)
    cell_h = max(font_size, 1)
    width = params.get("width")
    num_cols = width if width else max(image.width // cell_w, 10)
    num_rows = max(1, int(num_cols * cell_w / cell_h * image.height / image.width))

    layer_names = params.get("layers", list(_LAYER_REGISTRY))
    threshold = params.get("threshold", 0.15)
    layers = [_LAYER_REGISTRY[n](threshold=threshold) for n in layer_names if n in _LAYER_REGISTRY]
    charmap_list = [layer.process(image, num_rows, num_cols) for layer in layers]

    bg_mode = BgMode(params.get("bg_mode", "keep"))
    soft_cfg = (
        SoftBgConfig(opacity=params.get("bg_opacity", 0.25), chars=params.get("bg_chars", ".,"))
        if bg_mode == BgMode.SOFT else None
    )
    charmap = Compositor().composite(charmap_list, mask=None, bg_mode=bg_mode, soft_cfg=soft_cfg)

    profile_name = params.get("profile", "monochrome")
    profile = _PROFILE_REGISTRY.get(profile_name, MonochromeProfile)()
    return Renderer().render(
        charmap, profile,
        font_size=font_size,
        jitter=params.get("jitter", 1),
    )


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = testing

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/results/<path:filename>")
    def serve_result(filename: str):
        return send_from_directory(RESULTS_DIR, filename)

    @app.post("/api/convert")
    def api_convert():
        if "file" not in request.files:
            return jsonify({"error": "Файл не передан"}), 400

        f = request.files["file"]
        params_raw = request.form.get("params", "{}")
        mode = request.form.get("mode", "image")

        try:
            params = json.loads(params_raw)
        except json.JSONDecodeError:
            return jsonify({"error": "Неверный формат params"}), 400

        layer_names = params.get("layers", [])
        if not layer_names:
            return jsonify({"error": "Выбери хотя бы один слой"}), 400

        # Sanitize filename to prevent path traversal
        safe_name = Path(f.filename).name if f.filename else "upload"
        stem = Path(safe_name).stem or "upload"
        out_dir = RESULTS_DIR / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save source file with sanitized name
        src_dest = out_dir / safe_name
        if not src_dest.exists():
            f.stream.seek(0)
            src_dest.write_bytes(f.stream.read())
            f.stream.seek(0)

        if mode == "image":
            try:
                image = Image.open(f.stream)
                result_img = _convert_image(image, params)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

            out_path = _output_path(stem, params, "png")
            result_img.save(out_path)
            result_url = "/results/" + str(out_path.relative_to(RESULTS_DIR))
            return jsonify({"result_url": result_url, "result_path": str(out_path)})

        # Video: handled in Task 6
        task_id = uuid.uuid4().hex
        return jsonify({"task_id": task_id, "result_path": str(out_dir)})

    @app.get("/api/progress/<task_id>")
    def api_progress(task_id: str):
        return jsonify({"error": "not implemented"}), 501

    @app.post("/api/open-folder")
    def api_open_folder():
        return jsonify({"error": "not implemented"}), 501

    return app


if __name__ == "__main__":
    import webbrowser
    app = create_app()
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
