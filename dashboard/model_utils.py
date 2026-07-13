"""Preprocessing, model loading, and inference shared by the Streamlit dashboard.

Preprocessing here mirrors `final_nb_deliverable3_ml.ipynb` exactly:
- Logistic Regression / Scratch CNN expect 128x128 grayscale, LANCZOS-resized, scaled to [0, 1].
- MobileNetV2 expects 128x128 grayscale stacked into 3 channels, left in [0, 255]
  (its `preprocess_input` rescaling is baked into the saved model graph).

Weights can live in `models/` (default) or be pulled from a public URL base via
`MODEL_WEIGHTS_BASE_URL`, e.g. a GitHub raw path or Hugging Face / S3 bucket.
"""
import json
import os
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
from PIL import Image

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (128, 128)
MODEL_ARTIFACTS = (
    'logistic_regression.joblib',
    'scratch_cnn.keras',
    'mobilenetv2.keras',
    'modified_cnn.keras',
    'metadata.json',
)
DEFAULT_WEIGHTS_BASE_URL = (
    'https://raw.githubusercontent.com/keremyjelly/tumor-classification/main/models'
)


def resize_grayscale(image):
    """128x128 LANCZOS-resized grayscale array, float32 in [0, 255]."""
    return np.asarray(image.convert('L').resize(IMG_SIZE, Image.Resampling.LANCZOS), dtype=np.float32)


def preprocess_for_logistic_regression(image):
    return (resize_grayscale(image) / 255.0).reshape(1, -1)


def preprocess_for_scratch_cnn(image):
    return (resize_grayscale(image) / 255.0)[None, ..., None]


def preprocess_for_mobilenet(image):
    gray = resize_grayscale(image)
    return np.stack([gray, gray, gray], axis=-1)[None, ...]


def _softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def ensure_model_artifacts(models_dir, base_url=None):
    """Download any missing weight files from a public URL base into `models_dir`."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    base_url = (base_url or os.environ.get('MODEL_WEIGHTS_BASE_URL') or '').rstrip('/')
    if not base_url:
        return models_dir

    import urllib.request

    for name in MODEL_ARTIFACTS:
        dest = models_dir / name
        if dest.exists():
            continue
        url = urljoin(f'{base_url}/', name)
        urllib.request.urlretrieve(url, dest)
    return models_dir


def load_models(models_dir):
    """Load whichever exported artifacts exist. Returns (models, metadata)."""
    models_dir = ensure_model_artifacts(Path(models_dir))
    models = {}

    lr_path = models_dir / 'logistic_regression.joblib'
    if lr_path.exists():
        import joblib
        models['Logistic Regression'] = {'kind': 'sklearn', 'model': joblib.load(lr_path)}

    scratch_path = models_dir / 'scratch_cnn.keras'
    if scratch_path.exists():
        from tensorflow import keras
        models['Scratch CNN'] = {'kind': 'scratch_cnn', 'model': keras.models.load_model(scratch_path)}

    mobilenet_path = models_dir / 'mobilenetv2.keras'
    if mobilenet_path.exists():
        from tensorflow import keras
        models['MobileNetV2 Transfer'] = {'kind': 'mobilenetv2', 'model': keras.models.load_model(mobilenet_path)}

    modified_path = models_dir / 'modified_cnn.keras'
    if modified_path.exists():
        from tensorflow import keras
        models['Modified CNN'] = {'kind': 'scratch_cnn', 'model': keras.models.load_model(modified_path)}

    metadata_path = models_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None

    return models, metadata


def predict_one(entry, image):
    """Return a probability vector over CLASSES for one loaded model entry."""
    kind, model = entry['kind'], entry['model']
    if kind == 'sklearn':
        return model.predict_proba(preprocess_for_logistic_regression(image))[0]
    if kind == 'scratch_cnn':
        return model.predict(preprocess_for_scratch_cnn(image), verbose=0)[0]
    if kind == 'mobilenetv2':
        logits = model.predict(preprocess_for_mobilenet(image), verbose=0)[0]
        return _softmax(logits)
    raise ValueError(f'Unknown model kind: {kind}')


def predict_all(models, image):
    """Run every loaded model on `image`. Returns (per_model_probabilities, consensus_probabilities)."""
    per_model = {name: predict_one(entry, image) for name, entry in models.items()}
    consensus = np.mean(list(per_model.values()), axis=0) if per_model else None
    return per_model, consensus
