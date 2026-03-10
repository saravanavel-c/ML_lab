import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


st.title("Heart Disease Diagnosis using Bayesian Network")


# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():

    df = pd.read_csv("../Datasets/heart_disease_uci.csv")

    df["heart_disease"] = (df["num"] > 0).astype(int)

    df = df[
        ["age","sex","cp","trestbps","chol",
         "exang","oldpeak","thal","heart_disease"]
    ]

    df = df.dropna()

    # Discretization
    df["age"] = pd.cut(df["age"], bins=3, labels=["young","middle","old"])
    df["chol"] = pd.cut(df["chol"], bins=3, labels=["low","medium","high"])
    df["trestbps"] = pd.cut(df["trestbps"], bins=3, labels=["low","medium","high"])
    df["oldpeak"] = pd.cut(df["oldpeak"], bins=2, labels=["low","high"])

    return df


df = load_data()


# -----------------------------
# Sample Dataset
# -----------------------------
st.subheader("Sample Dataset")
st.dataframe(df.head())


# -----------------------------
# Train Test Split
# -----------------------------
train, test = train_test_split(df, test_size=0.2, random_state=42)


# -----------------------------
# Bayesian Network Structure
# -----------------------------
model = DiscreteBayesianNetwork([

    ("age","chol"),
    ("age","trestbps"),

    ("cp","exang"),
    ("exang","heart_disease"),

    ("chol","heart_disease"),
    ("oldpeak","heart_disease")

])


model.fit(train, estimator=BayesianEstimator, prior_type="BDeu")

inference = VariableElimination(model)


# -----------------------------
# User Input
# -----------------------------
st.subheader("Enter Patient Medical Details")

age = st.selectbox("Age Group", df["age"].unique())
cp = st.selectbox("Chest Pain Type", df["cp"].unique())
chol = st.selectbox("Cholesterol Level", df["chol"].unique())
trestbps = st.selectbox("Blood Pressure", df["trestbps"].unique())
exang = st.selectbox("Exercise Angina", df["exang"].unique())
oldpeak = st.selectbox("Oldpeak Level", df["oldpeak"].unique())


if st.button("Diagnose Patient"):

    evidence = {
        "age": age,
        "cp": cp,
        "chol": chol,
        "trestbps": trestbps,
        "exang": exang,
        "oldpeak": oldpeak
    }

    result = inference.query(["heart_disease"], evidence=evidence)

    prob = float(result.values[1])

    if prob > 0.5:
        st.error(f"High Risk of Heart Disease (Probability {prob:.2f})")
    else:
        st.success(f"Low Risk of Heart Disease (Probability {prob:.2f})")


# -----------------------------
# Bayesian Network Graph
# -----------------------------
st.subheader("Bayesian Network Structure")

G = nx.DiGraph()
G.add_edges_from(model.edges())

fig, ax = plt.subplots()

pos = nx.kamada_kawai_layout(G)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightblue",
    node_size=2500,
    font_size=10,
    ax=ax
)

st.pyplot(fig)


# -----------------------------
# Evaluation Metrics
# -----------------------------
st.subheader("Model Evaluation")

features = ["age","cp","chol","trestbps","exang","oldpeak","thal"]

y_true = test["heart_disease"]
y_pred = []

for _, row in test.iterrows():

    evidence = {col: row[col] for col in features}

    q = inference.query(["heart_disease"], evidence=evidence)

    pred = np.argmax(q.values)

    y_pred.append(pred)


acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

st.write("Accuracy:", round(acc,3))


fig2, ax2 = plt.subplots()

ax2.imshow(cm, cmap="Blues")

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        ax2.text(j,i,cm[i,j],ha="center",va="center")

st.pyplot(fig2)