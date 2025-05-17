import os
import numpy as np
import cv2
from tensorflow.keras.datasets import mnist

# Create folders
os.makedirs("mnist_dataset", exist_ok=True)
for i in range(10):
    os.makedirs(f"mnist_dataset/{i}", exist_ok=True)

# Load MNIST
(train_images, train_labels), _ = mnist.load_data()

# Save as individual images
for i, (img, label) in enumerate(zip(train_images, train_labels)):
    img = cv2.resize(img, (32, 32))
    cv2.imwrite(f"mnist_dataset/{label}/{i}.png", img)

print(f"Saved {len(train_images)} images to mnist_dataset/")