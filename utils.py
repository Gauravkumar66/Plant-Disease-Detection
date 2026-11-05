import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models/plant_diseaes_prediction_model.h5"
CLASS_FILE_PATH = BASE_DIR / "class_indeces.json"


def load_plant_model():
    """Load the trained Keras model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    return load_model(MODEL_PATH, compile=False)


def load_class_names():
    """
    Load class mapping from JSON.

    Supports:
    - {"0": "Apple___Apple_scab", ...}
    - {"Apple___Apple_scab": 0, ...}
    """
    if not CLASS_FILE_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping file not found: {CLASS_FILE_PATH.name}"
        )

    with open(CLASS_FILE_PATH, "r", encoding="utf-8") as file:
        class_mapping = json.load(file)

    if not isinstance(class_mapping, dict):
        raise ValueError("Class mapping JSON must contain a dictionary.")

    # Case 1: index -> class name
    if all(str(key).isdigit() for key in class_mapping.keys()):
        class_names = {
            int(index): name
            for index, name in class_mapping.items()
        }

    # Case 2: class name -> index
    elif all(
        isinstance(value, int) or str(value).isdigit()
        for value in class_mapping.values()
    ):
        class_names = {
            int(index): name
            for name, index in class_mapping.items()
        }

    else:
        raise ValueError(
            "Unsupported class mapping format. "
            "Expected index-to-name or name-to-index mapping."
        )

    return [
        class_names[index]
        for index in sorted(class_names.keys())
    ]


def preprocess_image(image: Image.Image):
    """Convert image to the model's expected input format."""
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(model, image, class_names):
    """Return top 3 predictions."""
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image, verbose=0)[0]

    if len(predictions) != len(class_names):
        raise ValueError(
            f"Model returned {len(predictions)} predictions, "
            f"but mapping contains {len(class_names)} classes."
        )

    top_indices = np.argsort(predictions)[::-1][:3]

    results = []

    for index in top_indices:
        results.append(
            {
                "class_name": class_names[int(index)],
                "confidence": float(predictions[index]),
            }
        )

    return results