import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200


def test_upload_real_image_and_get():
    with open("tests/assets/test.jpg", "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        r = client.post("/api/images", files=files)

    assert r.status_code == 200
    body = r.json()

    assert "status" in body
    assert "data" in body
    assert "error" in body
    assert "image_id" in body["data"]

    image_id = body["data"]["image_id"]

    r2 = client.get(f"/api/images/{image_id}")
    assert r2.status_code == 200
    j = r2.json()

    assert j["data"]["image_id"] == image_id


def test_invalid_type():
    files = {"file": ("a.txt", b"hello", "text/plain")}
    r = client.post("/api/images", files=files)

    assert r.status_code == 400