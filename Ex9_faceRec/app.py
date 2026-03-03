import streamlit as st
import cv2
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

st.title("🧠 ANN Face Recognition System")

# ---------------------------
# 1️⃣ Collect Face Images
# ---------------------------
st.header("Step 1: Collect Face Images")

person_name = st.text_input("Enter Person Name")

if st.button("Start Camera & Collect Images"):

    if person_name.strip() == "":
        st.warning("Please enter a name first.")
    else:
        save_path = f"dataset/{person_name}"
        os.makedirs(save_path, exist_ok=True)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        cap = cv2.VideoCapture(0)
        count = 0

        message_placeholder = st.empty()   # 👈 For dynamic messages

        st.info("Press 's' to save image, 'q' to quit camera")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (64, 64))
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

                key = cv2.waitKey(1)

                if key == ord('s'):
                    cv2.imwrite(f"{save_path}/{count}.jpg", face)
                    count += 1

                    # 👇 Show capture message in Streamlit
                    message_placeholder.success(
                        f"📸 Image {count} captured successfully!"
                    )

            cv2.imshow("Collect Faces", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        st.success("Image collection completed!")
# ---------------------------
# 2️⃣ Train Model
# ---------------------------
st.header("Step 2: Train ANN Model")

if st.button("Train Model"):

    data = []
    labels = []
    path = "dataset"

    for person in os.listdir(path):
        person_path = os.path.join(path, person)
        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64, 64))
            image = image / 255.0
            data.append(image)
            labels.append(person)

    data = np.array(data)
    labels = np.array(labels)

    unique_labels = np.unique(labels)
    label_dict = {label: i for i, label in enumerate(unique_labels)}

    encoded_labels = np.array([label_dict[label] for label in labels])
    encoded_labels = to_categorical(encoded_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        data, encoded_labels, test_size=0.2, random_state=42
    )

    model = Sequential()
    model.add(Flatten(input_shape=(64, 64)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(unique_labels), activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=40, batch_size=4, verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    model.save("face_model.h5")

    with open("label_dict.pkl", "wb") as f:
        pickle.dump(label_dict, f)

    st.success(f"Model Trained Successfully! Accuracy: {accuracy:.2f}")

# ---------------------------
# 3️⃣ Live Recognition
# ---------------------------
st.header("Step 3: Real-Time Face Recognition")

if st.button("Start Live Recognition"):

    model = load_model("face_model.h5")

    with open("label_dict.pkl", "rb") as f:
        label_dict = pickle.load(f)

    label_dict = {v: k for k, v in label_dict.items()}

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    st.info("Press 'q' to quit camera")

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

            prediction = model.predict(face, verbose=0)
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