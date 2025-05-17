import numpy as np
import cv2 as cv
import os
import tensorflow as tf
from tensorflow.keras import layers, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utlis import prepare_image

def load_dataset(data_path="mnist_dataset"):
    """Load images and labels from directory structure"""
    images, labels = [], []
    for class_id in range(10):  # Expecting 0-9 subdirectories
        class_dir = os.path.join(data_path, str(class_id))
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                resized_img =prepare_image(img)
                images.append(resized_img)
                labels.append(class_id)

    return np.array(images), np.array(labels)

# Load and prepare dataset
X, y = load_dataset()
X = X[..., np.newaxis]  # Add channel dimension

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train
)

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1
)

# Model architecture
def build_cnn_model(input_shape=(28,28, 1), num_classes=10):
    model = tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_cnn_model()
model.summary()

# Training configuration
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6)

# Training process
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1)

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Save model
model.save("digit_classifier.keras")
print("Model saved as digit_classifier.keras")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()