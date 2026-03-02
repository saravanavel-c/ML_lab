import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("../Datasets/car_evaluation.csv", header=None)
df.columns = ['buying','maint','doors','persons','lug_boot','safety','class']

# Encode dataset
encoders = {}

for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("class", axis=1)
y = df["class"]

# Train model
rf = RandomForestClassifier(
    n_estimators=3,
    criterion='entropy',
    random_state=0
)

rf.fit(X, y)

# -----------------------------
# UI
# -----------------------------
st.title("🚗 Car Evaluation Prediction App")
st.write("Predict car acceptability using Random Forest")

buying = st.selectbox("Buying Price", ['vhigh','high','med','low'])
maint = st.selectbox("Maintenance Cost", ['vhigh','high','med','low'])
doors = st.selectbox("Number of Doors", ['2','3','4','5more'])
persons = st.selectbox("Capacity", ['2','4','more'])
lug_boot = st.selectbox("Luggage Boot Size", ['small','med','big'])
safety = st.selectbox("Safety", ['low','med','high'])

if st.button("Predict"):

    user_data = pd.DataFrame([[buying, maint, doors, persons, lug_boot, safety]],
                             columns=['buying','maint','doors','persons','lug_boot','safety'])

    for col in user_data.columns:
        user_data[col] = encoders[col].transform(user_data[col])

    prediction = rf.predict(user_data)
    final_prediction = encoders['class'].inverse_transform(prediction)

    st.success(f"Predicted Car Evaluation: {final_prediction[0]}")