# Plant Disease Detection

A machine learning model that predicts plant diseases from leaf images using a Convolutional Neural Network (CNN).

**Technologies:** Python, CNN, TensorFlow, Image Processing, NumPy, Matplotlib

## Project structure

```
plant-disease-detection/
├── train.py            # Builds, trains and evaluates the CNN
├── predict.py          # Predicts the disease for a single leaf image
├── requirements.txt    # Python dependencies
└── dataset/            # Put your images here (one folder per class)
    ├── Healthy/
    ├── Disease_A/
    └── Disease_B/
```

## Setup

```bash
pip install -r requirements.txt
```

Download a plant leaf dataset (for example PlantVillage) and place it in `dataset/`, with one sub-folder per class.

## Usage

Train the model:

```bash
python train.py
```

This saves `plant_model.keras`, `class_names.json` and `training_curves.png`.

Predict on an image:

```bash
python predict.py path/to/leaf.jpg
```

## How it works

1. Images are resized to 128x128 and normalised to the 0-1 range.
2. A CNN with three convolution/pooling blocks, dropout and a dense classifier learns the disease classes.
3. Accuracy and loss curves are plotted with Matplotlib so the model can be refined.
