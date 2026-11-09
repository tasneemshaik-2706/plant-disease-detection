"""Train a CNN to classify plant leaf diseases."""
import json

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = "dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15


def load_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    class_names = train_ds.class_names
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=IMG_SIZE + (3,)),
            layers.Rescaling(1.0 / 255),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("training_curves.png")


def main():
    train_ds, val_ds, class_names = load_data()
    model = build_model(len(class_names))
    model.summary()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    model.save("plant_model.keras")
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)
    plot_history(history)
    print("Saved plant_model.keras, class_names.json and training_curves.png")


if __name__ == "__main__":
    main()
