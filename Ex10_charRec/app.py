import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from streamlit_drawable_canvas import st_canvas
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# -------------------------------
# Load MNIST Dataset
# -------------------------------
@st.cache_resource
def load_and_train_model():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data / 255.0   # normalize to 0-1
    y = mnist.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MLPClassifier(
    hidden_layer_sizes=(256,128),
    activation='relu',
    solver='adam',
    max_iter=50,
    batch_size=128,
    learning_rate_init=0.001,
    random_state=42,
    verbose=True
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test

model, X_test, y_test = load_and_train_model()

# -------------------------------
# Evaluation
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Handwritten Digit Recognition (MNIST + MLP)")

st.subheader("Draw a Digit (0-9)")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=5,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

st.markdown("---")
if st.button("Predict Digit"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data[:, :, 0]

        if np.sum(img) < 1000:
            st.warning("Please draw a digit first.")
            st.stop()

        # Invert (MNIST = white digit on black)
        img = 255 - img

        # Crop to bounding box
        coords = cv2.findNonZero(img.astype(np.uint8))
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

        # Resize so longest side = 20 pixels
        if h > w:
            new_h = 20
            new_w = int(w * (20 / h))
        else:
            new_w = 20
            new_h = int(h * (20 / w))

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create blank 28x28 image
        final_img = np.zeros((28, 28))

        # Center the digit
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img

        # Normalize
        final_img = final_img / 255.0
        final_img = final_img.reshape(1, -1)

        prediction = model.predict(final_img)
        probabilities = model.predict_proba(final_img)

        st.success(f"Predicted Digit: {prediction[0]}")

        fig, ax = plt.subplots()
        ax.bar(range(10), probabilities[0])
        ax.set_xticks(range(10))
        st.pyplot(fig)
st.subheader("Test Prediction")

# Select image index
index = st.slider("Select a test image index", 0, len(X_test)-1, 0)

# Get image
sample_image = X_test.iloc[index].values.reshape(28,28)
true_label = y_test.iloc[index]

# Display image
st.image(sample_image, width=280, caption=f"Actual Label: {true_label}")

# Predict
if st.button("Predict the Image"):
    prediction = model.predict(X_test.iloc[index].values.reshape(1,-1))
    probabilities = model.predict_proba(X_test.iloc[index].values.reshape(1,-1))

    st.success(f"Predicted Digit: {prediction[0]}")

    fig4, ax4 = plt.subplots()
    ax4.bar(range(10), probabilities[0])
    ax4.set_xticks(range(10))
    ax4.set_title("Prediction Probabilities")
    st.pyplot(fig4)


# -------------------------------
# Model Evaluation Display
# -------------------------------
st.markdown("---")
st.subheader("Model Evaluation on Test Data")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

fig2, ax2 = plt.subplots()
ax2.imshow(cm)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
ax2.set_title("Confusion Matrix")
st.pyplot(fig2)

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))