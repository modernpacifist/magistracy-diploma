import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from PIL import Image as pil
from PIL.Image import Image
from shutil import move, copy
from vkyn import *
import concurrent.futures
from tqdm.notebook import tqdm
import os

tf.get_logger().setLevel("ERROR")

# VaskaDataSetPath = "H:/Datasets/MagistracyDiploma/Images/vaska"
# VaskaImagesLess02 = "H:/Datasets/MagistracyDiploma/Images/vaska_bad"
VaskaDataSetPath = "H:/Datasets/MagistracyDiploma/Images/vaska_temp"
VaskaImagesLess02 = "H:/Datasets/MagistracyDiploma/Images/vaska_bad_temp"

SEED = 1337
IMAGE_SIZE = (300, 300) # EfficientNetB3
BATCH_SIZE = 128


def build_model(input_shape, num_classes: int, dropout_rate: float):
    inputs = tf.keras.layers.Input(shape=input_shape)
    cnn = tf.keras.applications.EfficientNetB3(
        input_tensor=inputs,
        include_top=False,
        weights="imagenet",
    )

    # Freeze the pretrained weights
    cnn.trainable = False

    # Rebuild top
    avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(cnn.output)
    x = tf.keras.layers.BatchNormalization()(avg_pool)

    # Dropout
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)

    # Output
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes
    outputs = tf.keras.layers.Dense(units, activation=activation, name="pred")(x)

    return tf.keras.Model(inputs, outputs, name="VKYN01"), (inputs, cnn, avg_pool, outputs)


MODEL, _ = build_model(
    input_shape=IMAGE_SIZE + (3,),
    num_classes=2,
    dropout_rate=0.3,
)

MODEL.load_weights("./weights/vkyn01-tf-efficientnetb3_5_epochs_dropout_03_learningrate1e-3_epochs.keras")


def process_image(path: Path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def predict_image(model, path: Path) -> dict[str, float]:
    img_array = process_image(path)
    predictions = model.predict(img_array, verbose=0)
    score = predictions[0][0]
    return {
        "y": score,
        "n": 1 - score,
    }


def process_file_prediction(file_path):
    try:
        p = predict_image(MODEL, file_path)
        return (file_path, p)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return (file_path, None)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return (file_path, None)


if __name__ == "__main__":
    # Collect all file paths
    file_paths = list(Path(VaskaDataSetPath).glob("*.jpg"))

    # Use ProcessPoolExecutor instead of ThreadPoolExecutor
    # Set max_workers to a reasonable number based on your CPU cores
    num_workers = os.cpu_count()  # Default to 4 if cpu_count returns None
    print(num_workers)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        file_predictions = list(tqdm(
            executor.map(process_file_prediction, file_paths), 
            total=len(file_paths)
        ))

    for i in file_predictions:
        if i[1]["n"] > 0.65:
            print(i[0], i[1])
            move(i[0], VaskaImagesLess02)

    # for f in Path(VaskaDataSetPath).glob("*.jpg"):
    #     p = predict_image(MODEL, f)
    #     if p["n"] > 0.7:
    #         print(f.name, p)
    #         move(f, VaskaImagesLess02)
