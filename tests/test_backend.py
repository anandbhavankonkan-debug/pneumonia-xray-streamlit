from pathlib import Path
import json

import numpy as np
import pytest

from inference_backend import normalize_image, validate_upload


def test_manifest_contract():
    manifest_path = Path(__file__).parents[1] / "models" / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["input_height"] == 224
    assert manifest["input_width"] == 224
    assert manifest["input_channels"] == 3
    assert manifest["class_names_in_order"] == [
        "Normal",
        "Pneumonia",
        "Not Normal No Lung Opacity",
    ]


def test_normalization_range():
    image = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    normalized = normalize_image(image)
    assert normalized.dtype == np.float32
    assert float(normalized.min()) == pytest.approx(0.0)
    assert float(normalized.max()) == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["xray.dcm", "xray.png", "xray.jpg", "xray.jpeg"])
def test_supported_extensions(name):
    assert validate_upload(name, b"test") == Path(name).suffix.lower()


def test_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_upload("xray.gif", b"test")


def test_rejects_empty_upload():
    with pytest.raises(ValueError, match="empty"):
        validate_upload("xray.png", b"")
