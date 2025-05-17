from argparse import ArgumentParser, Namespace
from typing import Iterator, Protocol, Any
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path


def build_vkyn00():
    input_shape = (300, 300, 3)
    num_classes = 2
    dropout_rate = 0.5
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    cnn = tf.keras.applications.EfficientNetB3(
        input_tensor=x,
        include_top=False,
        weights="imagenet",
    )

    # Freeze the pretrained weights
    cnn.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(cnn.output)
    x = tf.keras.layers.BatchNormalization()(x)

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

    model = tf.keras.Model(inputs, outputs, name="VKYN00")
    model.load_weights("weights/vkyn00-tf-efficientnetb3.h5")
    return model


def build_vkyn01():
    input_shape = (300, 300, 3)
    num_classes = 2
    dropout_rate = 0.5
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    cnn = tf.keras.applications.EfficientNetB3(
        input_tensor=x,
        include_top=False,
        weights="imagenet",
    )

    # Freeze the pretrained weights
    cnn.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(cnn.output)
    x = tf.keras.layers.BatchNormalization()(x)

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

    model = tf.keras.Model(inputs, outputs, name="VKYN01")
    model.load_weights("weights/vkyn01-tf-efficientnetb3.h5")
    return model


def build_model(input_shape, num_classes: int, dropout_rate: float):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    cnn = tf.keras.applications.EfficientNetB3(
        input_tensor=x,
        include_top=False,
        weights="imagenet",
    )

    # Freeze the pretrained weights
    cnn.trainable = False

    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(cnn.output)
    x = tf.keras.layers.BatchNormalization()(x)

    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes
    x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(units, activation=activation, name="pred")(x)

    return tf.keras.Model(inputs, outputs, name="VKYN")


class ChunkIter:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, items: Iterator):
        while True:
            chunk = []
            for _ in range(self.size):
                try:
                    x = next(items)
                    chunk.append(x)
                except StopIteration:
                    if len(chunk):
                        yield chunk
                    return None
            if len(chunk):
                yield chunk
            else:
                return None


class VkynOutput(Protocol):
    def __init__(self):
        ...

    def __call__(self, path: Path, prediction) -> Any:
        ...


class VkynDictOutput:
    def __init__(self):
        self.filename_field = "filename"
        self.score_field = "score"

    def __call__(self, path: Path, prediction):
        score = prediction[0]
        return {
            self.filename_field: path.name,
            self.score_field: score,
        }


class VkynFeaturesOutput:
    def __init__(self):
        ...

    def __call__(self, filename: str, prediction):
        return prediction.numpy()


class VkynPredictor:
    def __init__(self, model, image_size: tuple[int, int], batch_size: int) -> None:
        self.model = model
        self.image_size = image_size
        self.batch_size = batch_size
        self.output: VkynOutput = VkynDictOutput()

    def __call__(self, files: list[Path]):
        total = self.calc_batches(files)
        result = []
        split = ChunkIter(size=self.batch_size)
        for chunk in tqdm(split(iter(files)), total=total):
            filenames = []
            images = []
            for f in chunk:
                img = tf.keras.preprocessing.image.load_img(
                    f, target_size=self.image_size,
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                # filenames.append(f.name)
                filenames.append(f)
            batch = tf.stack(images, axis=0)
            predictions = self.model.predict(batch, verbose=0)
            if predictions is None:
                raise Exception("Failed to get predictions")
            for i, filename in enumerate(filenames):
                result.append(
                    self.output(filename, predictions[i])
                )
        return result

    def calc_batches(self, items: list) -> int:
        return int(np.ceil(len(items) / self.batch_size))


class Options(Namespace):
    labeled: bool
    csv: Path
    base: Path
    weights: Path
    score_column: str
    image_size: int
    batch_size: int


def get_args() -> Options:
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        required=True,
        dest="csv",
        type=Path,
        help="Path to CSV file",
    )
    parser.add_argument(
        "--labeled",
        dest="labeled",
        action="store_true",
        help="Use y/b subdirs or not",
    )
    parser.add_argument(
        "--weights",
        required=True,
        dest="weights",
        type=Path,
        help="Path to h5 file",
    )
    parser.add_argument(
        "--base",
        required=True,
        dest="base",
        type=Path,
        help="Path to base image folder",
    )
    parser.add_argument(
        "--score-column",
        required=False,
        dest="score_column",
        default="score",
        type=str,
        help="Name of score column",
    )
    parser.add_argument(
        "--image-size",
        required=False,
        dest="image_size",
        default=300,
        type=int,
        help="width/height of image",
    )
    parser.add_argument(
        "--batch",
        required=False,
        dest="batch_size",
        default=64,
        type=int,
        help="Batch size",
    )
    return parser.parse_args(namespace=Options())


def create_path_to_image(base: Path, filename: str, label: str | None) -> Path:
    if label:
        return base / label / filename
    return base / filename


def has_score(row) -> bool:
    if "score" not in row:
        return False
    return row["score"] is not None


def run_predict(options: Options):
    image_size = (options.image_size, options.image_size)

    # Create an instance of model
    model = build_model(
        input_shape=image_size + (3,),
        num_classes=2,
        dropout_rate=0.5,
    )

    # Load weights in Keras format(h5)
    model.load_weights(options.weights)
    # Load whole model from disk in TensorFlow format (saved_model.pb + variables/)
    # model = tf.keras.models.load_model("vkyn_model")

    df = pd.read_csv(options.csv, index_col="filename")
    paths = [
        create_path_to_image(
            base=options.base,
            label=row["label"] if options.labeled else None,
            filename=str(filename),
        )
        for filename, row in df.iterrows()
        # if has_score(row)
    ]
    print(f"Found {len(paths)} to infer")

    if not len(paths):
        exit()

    vkyn = VkynPredictor(
        model=model,
        image_size=image_size,
        batch_size=options.batch_size,
    )
    prediction = vkyn(paths)
    result = pd.DataFrame(prediction)
    result.set_index("filename", inplace=True)

    df[options.score_column] = result["score"]
    df.to_csv(options.csv)


if __name__ == "__main__":
    options = get_args()

    print(f"CSV File: {options.csv}")
    print(f"Model weights: {options.weights}")
    print(f"Image size: {options.image_size}")
    print(f"Batch size: {options.batch_size}")

    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    tf.get_logger().setLevel("ERROR")

    try:
        run_predict(options)
    except KeyboardInterrupt:
        sys.exit()
