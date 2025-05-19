import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from shutil import move, copy
import concurrent.futures
from tqdm.notebook import tqdm
import numpy as np
import os

tf.get_logger().setLevel("ERROR")

VaskaDataSetPath = "H:/Datasets/MagistracyDiploma/Images/vaska_clean_bak"
VaskaImagesDistributedPath = "H:/Datasets/MagistracyDiploma/Images/vaska_distributed_probability_07"

SEED = 1337
IMAGE_SIZE = (300, 300) # EfficientNetB5
BATCH_SIZE = 128
NUM_CLASSES = 13
DROPOUT_RATE = 0.3
CLASS_NAMES = [
    'activity',
    'animal',
    'city',
    'indoor',
    'kebab',
    'nature',
    'object',
    'other',
    'outdoor',
    'painting',
    'people',
    'transport',
    'vegetable'
]


for class_name in CLASS_NAMES:
    class_dir = Path(VaskaImagesDistributedPath) / class_name
    class_dir.mkdir(exist_ok=True, parents=True)
    print(f"Created directory: {class_dir}")


def _build_model(input_shape, num_classes: int, dropout_rate: float):
    inputs = tf.keras.layers.Input(shape=input_shape)
    cnn = tf.keras.applications.EfficientNetB5(
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


MODEL, _ = _build_model(
    input_shape=IMAGE_SIZE + (3,),
    num_classes=NUM_CLASSES,
    dropout_rate=DROPOUT_RATE,
)

MODEL.load_weights("./weights/tf-efficientnetb5_multiclass_50epoch_dropout04_learningrate1e-4_epochs.keras")


def process_image(path: Path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def predict_image(model, path: Path):
    img_array = process_image(path)
    predictions = model.predict(img_array, verbose=1)

    # Get the class with highest probability
    predicted_class_idx = np.argmax(predictions[0])
    predicted_probability = predictions[0][predicted_class_idx]
    
    result = {
        "class_index": int(predicted_class_idx),
        "probability": float(predicted_probability),
        "class_name": CLASS_NAMES[predicted_class_idx]
    }

    return result


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


def process_batch(batch_files):
    """Process a batch of files, move qualifying files, and return results."""
    results = []
    moved_count = 0
    
    for file_path in batch_files:
        file_prediction = process_file_prediction(file_path)
        results.append(file_prediction)
        
        # Move files that meet the criteria right in the batch, only if the probability is greater than 0.5
        if file_prediction[1] is not None and file_prediction[1]["probability"] > 0.7:
            print(file_prediction[0], file_prediction[1])
            if os.path.exists(file_prediction[0]):
                move(file_prediction[0], f"{VaskaImagesDistributedPath}/{file_prediction[1]['class_name']}")
                moved_count += 1
            else:
                print(f"File {file_prediction[0]} no longer exists, skipping move operation")
    
    return results, moved_count


def split_into_batches(items, num_batches):
    """Split a list into specified number of roughly equal-sized batches."""
    batch_size = len(items) // num_batches
    if batch_size == 0:
        batch_size = 1
    
    batches = []
    for i in range(0, len(items), batch_size):
        # Make sure not to exceed the list bounds
        batch = items[i:i + batch_size]
        if batch:  # Only add non-empty batches
            batches.append(batch)
    
    return batches


if __name__ == "__main__":
    # Use ProcessPoolExecutor instead of ThreadPoolExecutor
    # Set max_workers to a reasonable number based on your CPU cores
    num_workers = os.cpu_count() or 4  # Default to 4 if cpu_count returns None
    print(f"Using {num_workers} workers")

    # Collect all file paths
    file_paths = list(Path(VaskaDataSetPath).glob("*.jpg"))
    print(f"Found {len(file_paths)} files to process")
    
    # Split files into batches equal to the number of workers
    batches = split_into_batches(file_paths, num_workers)
    print(f"Split into {len(batches)} batches")

    # Process batches in parallel
    file_predictions = []
    total_moved = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        batch_results = list(tqdm(
            executor.map(process_batch, batches),
            total=len(batches),
            desc="Processing batches"
        ))
        
        # Flatten the results and sum up moved counts
        for batch_result, moved_count in batch_results:
            file_predictions.extend(batch_result)
            total_moved += moved_count
    
    print(f"Moved {total_moved} files to {VaskaImagesDistributedPath}")

    # for f in Path(VaskaDataSetPath).glob("*.jpg"):
    #     p = predict_image(MODEL, f)
    #     if p["n"] > 0.7:
    #         print(f.name, p)
    #         move(f, VaskaImagesLess02)
