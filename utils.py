import os
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download


BASE_DIR = Path(__file__).resolve().parent

CLASS_FILE_PATH = BASE_DIR / "class_indeces.json"

# Change this to your actual Hugging Face username/repository.
HF_REPO_ID = "gauravkumar8923/plant-disease-model"
MODEL_FILENAME = "plant_diseaes_prediction_model.h5"


@st.cache_resource
def load_plant_model():
    """Download and load the trained model."""

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
    )

    return load_model(model_path, compile=False)


@st.cache_data
def load_class_names():
    """Load the class index mapping."""

    if not CLASS_FILE_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping file not found: {CLASS_FILE_PATH.name}"
        )

    with open(CLASS_FILE_PATH, "r", encoding="utf-8") as file:
        class_mapping = json.load(file)

    if not isinstance(class_mapping, dict):
        raise ValueError("Class mapping JSON must contain a dictionary.")

    # index -> class name
    if all(str(key).isdigit() for key in class_mapping.keys()):
        index_to_class = {
            int(index): name
            for index, name in class_mapping.items()
        }

    # class name -> index
    elif all(
        isinstance(value, int) or str(value).isdigit()
        for value in class_mapping.values()
    ):
        index_to_class = {
            int(index): name
            for name, index in class_mapping.items()
        }

    else:
        raise ValueError("Unsupported class mapping JSON format.")

    return [
        index_to_class[index]
        for index in sorted(index_to_class.keys())
    ]


def preprocess_image(image: Image.Image):
    """Preprocess image according to the notebook."""

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(model, image, class_names):
    """Predict the top three classes."""

    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)[0]

    if len(predictions) != len(class_names):
        raise ValueError(
            f"Model outputs {len(predictions)} classes, "
            f"but class mapping contains {len(class_names)}."
        )

    top_indices = np.argsort(predictions)[::-1][:3]

    return [
        {
            "class_name": class_names[int(index)],
            "confidence": float(predictions[index]),
        }
        for index in top_indices
    ]