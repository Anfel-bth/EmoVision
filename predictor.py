import json
import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model("emotion_cnn.keras")
labels = json.load(open("labels.json"))
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def analyze(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))
    results = []
    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
        inp = (face.astype("float32") / 255.0)[None, ..., None]
        p = model(inp, training=False).numpy()[0]
        results.append({
            "box": (x, y, w, h),
            "emotion": labels[int(p.argmax())],
            "confidence": float(p.max()),
            "scores": {l: float(s) for l, s in zip(labels, p)},
        })
    return results