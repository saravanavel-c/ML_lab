import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

model = load_model("face_model.h5")

import pickle

with open("label_dict.pkl", "rb") as f:
    label_dict = pickle.load(f)

# Reverse dictionary for prediction
label_dict = {v: k for k, v in label_dict.items()}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = face.reshape(1, 64, 64)

        prediction = model.predict(face)
        class_index = np.argmax(prediction)
        name = label_dict[class_index]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,0), 2)

    cv2.imshow("ANN Face Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()