"Prediction backend for the Pneumonia X-ray Streamlit app."

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pydicom
import tensorflow as tf


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "fine_tuned_vgg16.keras"
MANIFEST_PATH = MODEL_DIR / "model_manifest.json"
ALLOWED_EXTENSIONS = {".dcm", ".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@tf.keras.utils.register_keras_serializable(package="ChestXrayProject")
class VGG16Preprocessing(tf.keras.layers.Layer):
    "Convert normalized RGB images to VGG16 ImageNet input space."

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32) * 255.0
        return tf.keras.applications.vgg16.preprocess_input(inputs)

    def get_config(self):
        return super().get_config()


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Model manifest not found: {MANIFEST_PATH}")
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = {
        "input_height", "input_width", "input_channels",
        "class_names_in_order", "sha256",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Manifest keys missing: {sorted(missing)}")
    return manifest


def load_prediction_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Serialized model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"VGG16Preprocessing": VGG16Preprocessing},
        compile=False,
    )


def validate_upload(file_name: str, file_bytes: bytes) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Upload DICOM, PNG, JPG or JPEG."
        )
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded file exceeds the 20 MB limit.")
    return suffix


def _dicom_to_grayscale(file_bytes: bytes) -> np.ndarray:
    dicom = pydicom.dcmread(BytesIO(file_bytes))
    image = dicom.pixel_array.astype(np.float32)
    slope = float(getattr(dicom, "RescaleSlope", 1.0))
    intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
    image = image * slope + intercept
    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image
    return image


def _standard_image_to_grayscale(file_bytes: bytes) -> np.ndarray:
    with Image.open(BytesIO(file_bytes)) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def decode_grayscale(file_name: str, file_bytes: bytes) -> np.ndarray:
    suffix = validate_upload(file_name, file_bytes)
    if suffix == ".dcm":
        image = _dicom_to_grayscale(file_bytes)
    else:
        image = _standard_image_to_grayscale(file_bytes)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D X-ray image; received {image.shape}.")
    return image.astype(np.float32, copy=False)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    minimum = float(image.min())
    maximum = float(image.max())
    if maximum > minimum:
        image = (image - minimum) / (maximum - minimum)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def preprocess_upload(
    file_name: str,
    file_bytes: bytes,
    manifest: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    grayscale = decode_grayscale(file_name, file_bytes)
    normalized = normalize_image(grayscale)
    height = int(manifest["input_height"])
    width = int(manifest["input_width"])
    one_channel = tf.image.resize(
        normalized[..., np.newaxis],
        size=(height, width),
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    ).numpy()
    one_channel = np.clip(one_channel, 0.0, 1.0).astype(np.float32)
    three_channel = np.repeat(one_channel, repeats=3, axis=-1)
    batch = three_channel[np.newaxis, ...].astype(np.float32)
    expected = (1, height, width, int(manifest["input_channels"]))
    if batch.shape != expected:
        raise ValueError(f"Expected model input {expected}; received {batch.shape}.")
    return batch, normalized


def predict_xray(
    model,
    manifest: dict[str, Any],
    file_name: str,
    file_bytes: bytes,
) -> dict[str, Any]:
    batch, display_image = preprocess_upload(file_name, file_bytes, manifest)
    probabilities = np.asarray(model.predict(batch, verbose=0)[0], dtype=float)
    class_names = list(manifest["class_names_in_order"])
    if probabilities.shape != (len(class_names),):
        raise ValueError("Model output does not match the manifest class order.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned a non-finite probability.")
    predicted_index = int(np.argmax(probabilities))
    return {
        "predicted_class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            name: float(probabilities[index])
            for index, name in enumerate(class_names)
        },
        "display_image": display_image,
    }
