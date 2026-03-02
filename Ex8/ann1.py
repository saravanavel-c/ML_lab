import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load dataset
data = []
labels = []
path = "dataset/"

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

# Encode labels
unique_labels = np.unique(labels)
label_dict = {label: i for i, label in enumerate(unique_labels)}
encoded_labels = np.array([label_dict[label] for label in labels])
encoded_labels = to_categorical(encoded_labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data, encoded_labels, test_size=0.2, random_state=42
)

# Build ANN model
model = Sequential()
model.add(Flatten(input_shape=(64,64)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(unique_labels), activation='softmax'))

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)