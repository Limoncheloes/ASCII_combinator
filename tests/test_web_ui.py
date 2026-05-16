import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from web_ui import create_app


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
