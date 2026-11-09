"""Predict the disease for a single leaf image."""
import json
import sys

import numpy as np
import tensorflow as tf

IMG_SIZE = (128, 128)


def main(image_path):
    model = tf.keras.models.load_model("plant_model.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
    probs = model.predict(arr)[0]
    best = int(np.argmax(probs))
    print(f"Prediction: {class_names[best]} ({probs[best] * 100:.1f}% confidence)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/leaf.jpg")
        sys.exit(1)
    main(sys.argv[1])
