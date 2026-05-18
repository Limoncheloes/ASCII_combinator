import io
import json
import subprocess as _subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from web_ui import create_app, RESULTS_DIR


@pytest.fixture
def client():
    app = create_app(testing=True)
    with app.test_client() as c:
        yield c


def test_index_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"ASCII Combinator" in resp.data


def _make_jpg_bytes(w: int = 32, h: int = 32) -> bytes:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[h // 2, :, :] = 120
    arr[:, w // 2, :] = 80
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


_DEFAULT_PARAMS = {
    "width": 20, "font_size": 8, "jitter": 0, "threshold": 0.15,
    "layers": ["brightness"], "bg_mode": "keep",
    "bg_opacity": 0.25, "bg_chars": ".,", "profile": "monochrome",
}


def test_convert_image_returns_result_url(client, tmp_path):
    data = {
        "file": (io.BytesIO(_make_jpg_bytes()), "test.jpg", "image/jpeg"),
        "params": json.dumps(_DEFAULT_PARAMS),
        "mode": "image",
    }
    resp = client.post("/api/convert", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "result_url" in body
    assert "result_path" in body
    assert body["result_url"].startswith("/results/")


def test_convert_image_saves_file(client, tmp_path):
    data = {
        "file": (io.BytesIO(_make_jpg_bytes()), "myimg.jpg", "image/jpeg"),
        "params": json.dumps(_DEFAULT_PARAMS),
        "mode": "image",
    }
    resp = client.post("/api/convert", data=data, content_type="multipart/form-data")
    body = resp.get_json()
    result_file = Path(body["result_path"])
    assert result_file.exists()
    assert result_file.suffix == ".png"


def test_convert_image_no_file_returns_400(client):
    resp = client.post("/api/convert", data={"mode": "image", "params": "{}"})
    assert resp.status_code == 400


def test_convert_image_no_layers_returns_400(client):
    params = {**_DEFAULT_PARAMS, "layers": []}
    data = {
        "file": (io.BytesIO(_make_jpg_bytes()), "x.jpg", "image/jpeg"),
        "params": json.dumps(params),
        "mode": "image",
    }
    resp = client.post("/api/convert", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_open_folder_calls_platform_command(client):
    valid_path = str(RESULTS_DIR / "horse" / "result.png")
    with patch("web_ui.subprocess.Popen") as mock_popen:
        resp = client.post(
            "/api/open-folder",
            data=json.dumps({"path": valid_path}),
            content_type="application/json",
        )
    assert resp.status_code == 200
    mock_popen.assert_called_once()
    args, _ = mock_popen.call_args
    assert any("horse" in str(a) for a in args[0])


def test_serve_result_existing_file(client, tmp_path):
    # Create a test PNG in results dir
    stem_dir = RESULTS_DIR / "_test_serve"
    stem_dir.mkdir(parents=True, exist_ok=True)
    test_file = stem_dir / "result.png"
    Image.new("RGB", (10, 10)).save(test_file)

    resp = client.get("/results/_test_serve/result.png")
    assert resp.status_code == 200
    assert resp.content_type.startswith("image/")
    test_file.unlink()
    stem_dir.rmdir()


def _make_test_mp4(path) -> bytes:
    """Create a tiny 0.3s mp4 and return its bytes."""
    from pathlib import Path
    img_path = Path(path).parent / "_f.png"
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[16, :, :] = 100
    Image.fromarray(arr).save(img_path)
    _subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-i", str(img_path),
         "-t", "0.3", "-r", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(path)],
        capture_output=True, check=True,
    )
    img_path.unlink(missing_ok=True)
    return Path(path).read_bytes()


_VIDEO_PARAMS = {
    "width": 20, "font_size": 8, "jitter": 0, "threshold": 0.15,
    "layers": ["brightness"], "bg_mode": "keep",
    "bg_opacity": 0.25, "bg_chars": ".,", "profile": "monochrome",
    "fps": 5, "workers": 1, "frame_step": None,
    "preview": False, "gif": False, "gif_fps": 5,
}


def test_convert_video_returns_task_id(client, tmp_path):
    mp4 = tmp_path / "test.mp4"
    _make_test_mp4(mp4)
    data = {
        "file": (io.BytesIO(mp4.read_bytes()), "test.mp4", "video/mp4"),
        "params": json.dumps(_VIDEO_PARAMS),
        "mode": "video",
    }
    resp = client.post("/api/convert", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "task_id" in body
    assert len(body["task_id"]) == 32  # uuid4 hex


def test_progress_unknown_task_returns_done(client):
    # Unknown task_id should stream a done=true event (graceful)
    resp = client.get("/api/progress/unknowntask123")
    assert resp.status_code == 200
    assert b"data:" in resp.data
