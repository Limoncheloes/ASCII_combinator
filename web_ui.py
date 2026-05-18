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
from werkzeug.utils import secure_filename

from ascii_combinator.bg_mode import BgMode, SoftBgConfig
from ascii_combinator.compositor import Compositor
from ascii_combinator.layers.brightness import BrightnessLayer
from ascii_combinator.layers.diagonal import DiagonalLayer
from ascii_combinator.layers.sobel_x import SobelXLayer
from ascii_combinator.layers.sobel_y import SobelYLayer
from ascii_combinator.profiles.monochrome import MonochromeProfile
from ascii_combinator.renderer import Renderer
from ascii_combinator.video import FrameExtractor, FrameProcessor, VideoAssembler, VideoConfig

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


def _run_video_task(
    task_id: str,
    video_src: Path,
    out_dir: Path,
    params: dict,
) -> None:
    """Background thread: extract frames, render, assemble. Updates _progress."""
    try:
        extractor = FrameExtractor()
        assembler = VideoAssembler()
        processor = FrameProcessor()
        fps = float(params.get("fps") or 10.0)
        frame_step = params.get("frame_step")
        if frame_step:
            frame_step = int(frame_step)

        config = VideoConfig(
            width=params.get("width"),
            profile_name=params.get("profile", "monochrome"),
            layer_names=params.get("layers", ["brightness"]),
            jitter=int(params.get("jitter", 1)),
            threshold=float(params.get("threshold", 0.15)),
            font_size=int(params.get("font_size", 12)),
            bg_mode=BgMode(params.get("bg_mode", "keep")),
            soft_cfg=(
                SoftBgConfig(
                    opacity=float(params.get("bg_opacity", 0.25)),
                    chars=params.get("bg_chars", ".,"),
                )
                if params.get("bg_mode") == "soft" else None
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_in_s, \
             tempfile.TemporaryDirectory() as tmp_out_s:
            tmp_in = Path(tmp_in_s)
            tmp_out = Path(tmp_out_s)

            frames_in = extractor.extract(video_src, tmp_in, fps, frame_step)
            total = len(frames_in)
            _progress[task_id].update({"total": total})

            # Preview mode: render only first frame as PNG and return early
            if params.get("preview"):
                preview_out = out_dir / "preview.png"
                processor.process(frames_in[0], preview_out, config)
                _progress[task_id].update({
                    "done": True,
                    "preview_url": "/results/" + str(preview_out.relative_to(RESULTS_DIR)),
                    "result_url": None,
                })
                return

            # Render frames sequentially for fine-grained progress
            frames_out = []
            for i, frame_in in enumerate(frames_in):
                frame_out = tmp_out / frame_in.name
                processor.process(frame_in, frame_out, config)
                frames_out.append(frame_out)
                _progress[task_id]["frame"] = i + 1
                if i == 0:
                    # First frame -> preview PNG saved to out_dir
                    preview_out = out_dir / "preview.png"
                    shutil.copy(frame_out, preview_out)
                    _progress[task_id]["preview_url"] = (
                        "/results/" + str(preview_out.relative_to(RESULTS_DIR))
                    )

            # Assemble MP4
            w = params.get("width", 80)
            f = params.get("font_size", 12)
            mp4_out = out_dir / f"result_w{w}_f{f}.mp4"
            assembler.assemble_mp4(tmp_out, mp4_out, fps)

            gif_url = None
            if params.get("gif"):
                gif_out = mp4_out.with_suffix(".gif")
                gif_fps = float(params.get("gif_fps") or fps)
                assembler.assemble_gif(mp4_out, gif_out, gif_fps)
                gif_url = "/results/" + str(gif_out.relative_to(RESULTS_DIR))

            mp4_url = "/results/" + str(mp4_out.relative_to(RESULTS_DIR))
            _progress[task_id].update({
                "done": True,
                "result_url": mp4_url,
                "gif_url": gif_url,
            })

    except Exception as e:
        _progress[task_id].update({"done": True, "error": str(e)})


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
        safe_name = secure_filename(f.filename or "upload") or "upload"
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

        if mode == "video":
            task_id = uuid.uuid4().hex
            _progress[task_id] = {"frame": 0, "total": 0, "done": False}
            w = params.get("width", 80)
            f_size = params.get("font_size", 12)
            mp4_name = f"result_w{w}_f{f_size}.mp4"
            result_path = str(out_dir / mp4_name)
            src_path = out_dir / safe_name
            t = threading.Thread(
                target=_run_video_task,
                args=(task_id, src_path, out_dir, params),
                daemon=True,
            )
            t.start()
            return jsonify({"task_id": task_id, "result_path": result_path})

        return jsonify({"error": "Unknown mode"}), 400

    @app.get("/api/progress/<task_id>")
    def api_progress(task_id: str):
        import time

        def generate():
            while True:
                info = _progress.get(task_id, {"done": True, "error": "Unknown task"})
                yield f"data: {json.dumps(info)}\n\n"
                if info.get("done"):
                    break
                time.sleep(0.5)

        return app.response_class(generate(), mimetype="text/event-stream")

    @app.post("/api/open-folder")
    def api_open_folder():
        data = request.get_json(silent=True) or {}
        path = data.get("path", "")
        p = Path(path)
        folder = str(p.parent if p.suffix else path)
        try:
            resolved = Path(folder).resolve()
            if not resolved.is_relative_to(RESULTS_DIR.resolve()):
                return jsonify({"error": "Path outside results"}), 400
        except Exception:
            return jsonify({"error": "Invalid path"}), 400
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", folder])
        elif system == "Windows":
            subprocess.Popen(["explorer", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
        return jsonify({"ok": True})

    return app


if __name__ == "__main__":
    import webbrowser
    app = create_app()
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
