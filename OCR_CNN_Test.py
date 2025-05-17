import numpy as np
import cv2 as cv
import pickle
import os
from tensorflow.keras.models import load_model

model = load_model("digit_classifier.keras")
# Initialize camera
cap = cv.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Load model
model_path = os.path.join(os.getcwd(), "best_model.p")
print(f"Looking for model at: {model_path}")

if not os.path.exists(model_path):
    print("ERROR: Model not found! Please run OCR_CNN_Trainning.py first")
    exit()

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {str(e)}")
    exit()


def preprocess(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img / 255.0
    return img.reshape(1, 32, 28, 1)  # Directly reshape for model input


while True:
    success, frame = cap.read()
    if not success:
        break

    # Process and predict
    processed = preprocess(cv.resize(frame, (32, 32)))
    predictions = model.predict(processed, verbose=0)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Display results
    cv.putText(frame, f"Digit: {predicted_class}", (50, 50),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, f"Confidence: {confidence:.2f}", (50, 100),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("Digit Classifier", frame)

    # Exit on 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()