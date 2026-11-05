# Plant Disease Detection

A Streamlit application that predicts plant leaf diseases using a
trained deep learning model.

## Features

- Upload a leaf image
- Image preprocessing at 224 × 224 RGB
- Pixel normalization using 1/255
- Plant disease prediction
- Top 3 predictions with confidence scores

## Project Files

- `app.py` - Streamlit user interface
- `utils.py` - Model loading, preprocessing, and prediction
- `plant_diseaes_prediction_model.h5` - Trained Keras model
- `class_indeces.json` - Class index mapping

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt