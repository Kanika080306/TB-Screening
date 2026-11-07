import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

IMAGE_DIR = Path(r"...")
MASK_DIR = Path(r"...")
OUTPUT_DIR = Path(r"...")
IMAGE_SIZE = 224
RANDOM_STATE = 42
EPOCHS = 20
BATCH_SIZE = 8
EARLY_STOPPING_PATIENCE = 3

OPTIMIZER_OPTIONS = {
    "Adam": tf.keras.optimizers.Adam,
    "SGD": tf.keras.optimizers.SGD,
    "RMSprop": tf.keras.optimizers.RMSprop
}
LEARNING_RATES = [0.0001, 0.001]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

print("TensorFlow Version:", tf.__version__)

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")
if not MASK_DIR.exists():
    raise FileNotFoundError(f"Mask folder not found: {MASK_DIR}")

all_image_paths = sorted(IMAGE_DIR.glob("*.png"))
mask_lookup = {p.stem: p for p in MASK_DIR.rglob("*.png")}

processed_images, processed_masks = [], []
processed_paths, skipped_images = [], []

for img_path in all_image_paths:
    mask_path = mask_lookup.get(img_path.stem)

    if mask_path is None:
        skipped_images.append(str(img_path))
        continue

    try:
        img = Image.open(img_path).convert("L")
        mask_img = Image.open(mask_path).convert("L")

        img_array = np.asarray(img, dtype=np.float32)
        mask_array = np.asarray(mask_img)

        if img_array.shape != mask_array.shape:
            print("Skipping image because image/mask sizes differ:", img_path.name)
            skipped_images.append(str(img_path))
            continue

        # Create and apply binary lung mask
        binary_mask = (mask_array > 0).astype(np.float32)
        segmented = (img_array * binary_mask) / 255.0

        # Resize while preserving aspect ratio
        height, width = segmented.shape
        scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)

        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        resized_img = Image.fromarray(segmented).resize(
            (new_width, new_height), Image.Resampling.BILINEAR
        )
        resized_mask = Image.fromarray(
            (binary_mask * 255).astype(np.uint8)
        ).resize((new_width, new_height), Image.Resampling.NEAREST)

        # Create 224 x 224 canvas
        canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        mask_canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        x_offset = (IMAGE_SIZE - new_width) // 2
        y_offset = (IMAGE_SIZE - new_height) // 2

        canvas[y_offset:y_offset + new_height,
               x_offset:x_offset + new_width] = np.asarray(
                   resized_img, dtype=np.float32
               )

        mask_canvas[y_offset:y_offset + new_height,
                    x_offset:x_offset + new_width] = np.asarray(
                        resized_mask, dtype=np.uint8
                    )

        processed_images.append(canvas)
        processed_masks.append(mask_canvas)
        processed_paths.append(str(img_path))

    except Exception as exc:
        print(f"Skipping {img_path.name}: {exc}")
        skipped_images.append(str(img_path))

if not processed_images:
    raise RuntimeError(
        "No images were processed. Check folder paths and matching image/mask filenames."
    )

processed_images = np.asarray(processed_images, dtype=np.float32)
processed_masks = np.asarray(processed_masks, dtype=np.uint8)
processed_paths = np.asarray(processed_paths)

print("\nPreprocessing Summary")
print("Images processed:", len(processed_images))
print("Masks processed:", len(processed_masks))
print("Images skipped:", len(skipped_images))
print("Image array shape:", processed_images.shape)
print("Mask array shape:", processed_masks.shape)
print("Pixel range:", float(processed_images.min()), "to",
      float(processed_images.max()))

if not (len(processed_images) == len(processed_masks) == len(processed_paths)):
    raise ValueError("Processed images, masks, and paths have mismatched counts.")

# Visualize one preprocessed image and mask
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(processed_images[0], cmap="gray", vmin=0, vmax=1)
plt.title("Preprocessed X-ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(processed_masks[0], cmap="gray")
plt.title("Lung Mask")
plt.axis("off")

plt.tight_layout()
plt.show()

# Filename ending in _0 = Normal
# Filename ending in _1 = Abnormal / TB-positive

labels_list, valid_paths, valid_indices = [], [], []

for index, path_string in enumerate(processed_paths):
    suffix = Path(path_string).stem.split("_")[-1]

    if suffix in ("0", "1"):
        valid_paths.append(path_string)
        labels_list.append(int(suffix))
        valid_indices.append(index)
    else:
        print("Skipping file with unrecognized label:",
              Path(path_string).name)

processed_images = processed_images[valid_indices]
processed_masks = processed_masks[valid_indices]

X_paths = np.asarray(valid_paths)
y_labels = np.asarray(labels_list, dtype=np.int32)

if len(X_paths) < 10:
    raise ValueError(f"Too few labelled images found: {len(X_paths)}")

if len(np.unique(y_labels)) != 2:
    raise ValueError("Both classes (0 and 1) must be present.")

print("\nDataset Summary")
print("Total labelled images:", len(X_paths))
print("Normal:", int(np.sum(y_labels == 0)))
print("Abnormal / TB-positive:", int(np.sum(y_labels == 1)))

# 80% development and 20% testing
X_dev_paths, X_test_paths, y_dev, y_test = train_test_split(
    X_paths, y_labels, test_size=0.20,
    random_state=RANDOM_STATE, stratify=y_labels
)

# Split development data into training and validation
X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    X_dev_paths, y_dev, test_size=0.125,
    random_state=RANDOM_STATE, stratify=y_dev
)

y_train = np.asarray(y_train, dtype=np.float32)
y_val = np.asarray(y_val, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int32)

# Map image paths to array positions
path_to_index = {path: i for i, path in enumerate(X_paths)}

train_indices = [path_to_index[p] for p in X_train_paths]
val_indices = [path_to_index[p] for p in X_val_paths]
test_indices = [path_to_index[p] for p in X_test_paths]

# Prepare image arrays
X_train_img = processed_images[train_indices][..., np.newaxis]
X_val_img = processed_images[val_indices][..., np.newaxis]
X_test_img = processed_images[test_indices][..., np.newaxis]

print("\nDataset Split")
print("Training:", len(X_train_img), "images,", len(y_train), "labels")
print("Validation:", len(X_val_img), "images,", len(y_val), "labels")
print("Testing:", len(X_test_img), "images,", len(y_test), "labels")

print("\nImage Shapes")
print("Training:", X_train_img.shape)
print("Validation:", X_val_img.shape)
print("Testing:", X_test_img.shape)

if len(X_train_img) != len(y_train):
    raise ValueError("Training image/label count mismatch.")
if len(X_val_img) != len(y_val):
    raise ValueError("Validation image/label count mismatch.")
if len(X_test_img) != len(y_test):
    raise ValueError("Testing image/label count mismatch.")

print("\nClass Counts")
for split_name, labels in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Testing", y_test)
]:
    print(
        f"{split_name}: Normal={int(np.sum(labels == 0))}, "
        f"Abnormal={int(np.sum(labels == 1))}"
    )

classes = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train.astype(np.int32)
)

class_weight_dict = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("\nClass Weights:")
print(class_weight_dict)

def build_model():
    base_model = tf.keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling="avg"
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Grayscale input
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

    # Convert grayscale to RGB
    x = layers.Concatenate()([inputs, inputs, inputs])

    # DenseNet preprocessing
    x = tf.keras.applications.densenet.preprocess_input(x * 255.0)

    # Feature extraction
    x = base_model(x, training=False)

    # Dropout and binary classification
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

experiment_results = []
best_val_loss = float("inf")
best_configuration = None

for optimizer_name, optimizer_class in OPTIMIZER_OPTIONS.items():
    for learning_rate in LEARNING_RATES:

        experiment_name = (
            f"{optimizer_name}_lr_{learning_rate}"
        ).replace(".", "_")

        checkpoint_path = EXPERIMENT_DIR / f"{experiment_name}_best.keras"

        print("\n" + "=" * 60)
        print("Experiment:", experiment_name)
        print("Optimizer:", optimizer_name)
        print("Learning Rate:", learning_rate)
        print("=" * 60)

        # Reset TensorFlow state and random seed
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(RANDOM_STATE)

        # Build and compile model
        experiment_model = build_model()

        experiment_model.compile(
            optimizer=optimizer_class(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )

        # Save best epoch
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            verbose=1
        )

        # Train model
        history = experiment_model.fit(
            X_train_img,
            y_train,
            validation_data=(X_val_img, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping, checkpoint],
            class_weight=class_weight_dict,
            verbose=1
        )

        # Find best epoch
        best_epoch_index = int(np.argmin(history.history["val_loss"]))

        current_val_loss = float(
            history.history["val_loss"][best_epoch_index]
        )
        current_val_accuracy = float(
            history.history["val_accuracy"][best_epoch_index]
        )

        result = {
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "best_epoch": best_epoch_index + 1,
            "val_accuracy": current_val_accuracy,
            "val_loss": current_val_loss,
            "checkpoint_path": str(checkpoint_path)
        }

        experiment_results.append(result)

        print("Best Validation Loss:", current_val_loss)
        print("Best Validation Accuracy:", current_val_accuracy)

        # Save the best experiment
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_configuration = result.copy()

            shutil.copy2(checkpoint_path, BEST_MODEL_PATH)
            print("New best model saved:", BEST_MODEL_PATH)

        del experiment_model
        tf.keras.backend.clear_session()

if not experiment_results or best_configuration is None:
    raise RuntimeError("No experiment completed successfully.")

results_df = pd.DataFrame(experiment_results).sort_values(
    "val_loss", ascending=True
)

results_df.to_csv(RESULTS_PATH, index=False)

print("\n" + "=" * 60)
print("OPTIMIZER COMPARISON RESULTS")
print("=" * 60)

print(
    results_df.drop(columns=["checkpoint_path"]).to_string(index=False)
)

print("\nBest Configuration")
print("Optimizer:", best_configuration["optimizer"])
print("Learning Rate:", best_configuration["learning_rate"])
print("Best Epoch:", best_configuration["best_epoch"])
print("Validation Accuracy:", best_configuration["val_accuracy"])
print("Validation Loss:", best_configuration["val_loss"])

print("\nBest Model Path:", BEST_MODEL_PATH)
print("Results CSV:", RESULTS_PATH)

print("\nLoading best experimental model...")
best_model = load_model(BEST_MODEL_PATH)

print("Generating predictions on test data...")

test_probabilities = best_model.predict(
    X_test_img,
    batch_size=BATCH_SIZE,
    verbose=0
).ravel()

# Convert probabilities to binary predictions
y_pred = (test_probabilities >= 0.5).astype(np.int32)

# Accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", round(float(test_accuracy), 4))

# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Abnormal/TB-positive"],
        zero_division=0
    )
)

# Confusion matrix
print(
    "Confusion Matrix (rows=true labels, columns=predicted labels):"
)

print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# ROC-AUC
if len(np.unique(y_test)) == 2:
    try:
        test_auc = roc_auc_score(y_test, test_probabilities)
        print("Test ROC-AUC:", round(float(test_auc), 4))
    except ValueError as exc:
        print("Could not calculate ROC-AUC:", exc)
else:
    print("ROC-AUC not calculated because the test set contains only one class.")

print("\nTraining and evaluation completed.")


