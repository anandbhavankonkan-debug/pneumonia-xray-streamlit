# Pneumonia X-ray Classification — Streamlit Deployment

Academic deployment of the validation-selected Fine-tuned VGG16 model for three-class chest X-ray classification.

## Classes

1. Normal
2. Pneumonia
3. Not Normal No Lung Opacity

## Features

- Upload DICOM, PNG, JPG or JPEG chest X-rays
- Matching grayscale, normalization and resize preprocessing
- Predicted class, confidence and all class probabilities
- Streamlit frontend with a separate inference backend
- Docker packaging and GitHub Codespaces support
- Automated backend tests

## Important limitation

This is an academic proof of concept, not a medical device. It must not provide a clinical diagnosis or replace assessment by a qualified clinician.

## Repository structure

```text
.
├── app.py
├── inference_backend.py
├── models/
│   ├── fine_tuned_vgg16.keras
│   └── model_manifest.json
├── tests/test_backend.py
├── .streamlit/config.toml
├── .devcontainer/devcontainer.json
├── requirements.txt
├── requirements-dev.txt
└── Dockerfile
```

## Model storage

The 111.77 MB Keras model is tracked with Git LFS because it exceeds GitHub's normal 100 MB file limit. Install Git LFS and run `git lfs pull` after cloning outside Codespaces.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Test

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Docker

```bash
docker build -t pneumonia-xray-streamlit:1.0 .
docker run --rm -p 8501:8501 pneumonia-xray-streamlit:1.0
```

Open `http://localhost:8501` and upload a supported X-ray image.

## GitHub Codespaces

Create a Codespace from the repository. The development container installs dependencies and pulls the LFS model. Run `streamlit run app.py`; port 8501 is configured for automatic public forwarding.

## Model contract

- Input: normalized grayscale X-ray repeated into 3 channels, shape 224 × 224 × 3
- Output: three softmax probabilities in the class order stored in `models/model_manifest.json`
- Selected model: Fine-tuned VGG16
- Intended use: academic demonstration only
