import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf

# Download and extract the sample Cats vs Dogs dataset once.
data_dir = Path("/content/cats_and_dogs_filtered")
if not (data_dir / "train").is_dir():
    url = "https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip"
    zip_path = tf.keras.utils.get_file(
        fname="cats_and_dogs_filtered.zip",
        origin=url
    )
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall("/content")

train_dir = data_dir / "train"
test_dir = data_dir / "validation"  # separate held-out images

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Use 20% of training images for validation; reserve test_dir for final evaluation.
train_ds = tf.keras.utils.image_dataset_from_directory(
    str(train_dir),
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    str(train_dir),
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    str(test_dir),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Class order:", train_ds.class_names)  # cats=0, dogs=1

# CNN with exactly three convolutional layers.
model = tf.keras.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

# Predict on held-out test images. Sigmoid probability >= 0.5 means "dog".
y_true = np.concatenate([labels.numpy().ravel() for _, labels in test_ds]).astype(int)
dog_probability = model.predict(test_ds, verbose=0).ravel()
y_pred = (dog_probability >= 0.5).astype(int)

# Confusion-matrix rows are actual classes; columns are predicted classes.
cm = np.zeros((2, 2), dtype=int)
np.add.at(cm, (y_true, y_pred), 1)
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
precision = tp / max(tp + fp, 1)  # dog predictions that are correct
recall = tp / max(tp + fn, 1)     # actual dogs correctly found
f1 = 2 * precision * recall / max(precision + recall, 1e-12)

print(f"\nAccuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f} (dog positive)")
print(f"Recall:    {recall:.3f} (dog positive)")
print(f"F1-score:  {f1:.3f}")
print("Confusion matrix [rows=true cats/dogs, columns=predicted cats/dogs]:")
print(cm)