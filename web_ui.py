from __future__ import annotations

import platform
import subprocess
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

RESULTS_DIR = Path(__file__).parent / "results"

_progress: dict[str, dict] = {}


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
        return jsonify({"error": "not implemented"}), 501

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
