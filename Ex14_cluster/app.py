import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

st.set_page_config(page_title="K-Means Iris Clustering")

st.title("K-Means Clustering - Iris Dataset")

# -----------------------------
# Load Dataset
# -----------------------------

iris = load_iris()

X = iris.data
feature_names = iris.feature_names

df = pd.DataFrame(X, columns=feature_names)

st.subheader("Dataset Preview")

st.dataframe(df.head())

# -----------------------------
# Elbow Method
# -----------------------------

st.subheader("Elbow Method")

wcss = []

for i in range(1,11):
    model = KMeans(n_clusters=i, random_state=42, n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)

fig1, ax1 = plt.subplots()

ax1.plot(range(1,11), wcss, marker='o')

ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("WCSS")
ax1.set_title("Elbow Method")

st.pyplot(fig1)

# -----------------------------
# Train Model
# -----------------------------

st.subheader("Train K-Means")

k = st.slider("Select Number of Clusters", 2, 10, 3)

model = KMeans(n_clusters=k, random_state=42, n_init=10)

clusters = model.fit_predict(X)

df["Cluster"] = clusters

st.success("Model trained successfully!")

# -----------------------------
# Visualization
# -----------------------------

st.subheader("Cluster Visualization")

fig2, ax2 = plt.subplots()

scatter = ax2.scatter(
    X[:,0],
    X[:,2],
    c=clusters
)

ax2.set_xlabel("Sepal Length")
ax2.set_ylabel("Petal Length")

st.pyplot(fig2)

# -----------------------------
# Cluster Centers
# -----------------------------

st.subheader("Cluster Centers")

centers = pd.DataFrame(
    model.cluster_centers_,
    columns=feature_names
)

st.dataframe(centers)

# -----------------------------
# User Input Prediction
# -----------------------------

st.subheader("Predict Cluster for New Flower")

sepal_length = st.number_input("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.number_input("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.number_input("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.number_input("Petal Width", 0.1, 2.5, 0.2)

if st.button("Find Cluster"):

    sample = np.array([[sepal_length,
                        sepal_width,
                        petal_length,
                        petal_width]])

    cluster = model.predict(sample)

    st.success(f"The flower belongs to Cluster {cluster[0]}")