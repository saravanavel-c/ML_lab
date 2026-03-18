import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

st.title("Locally Weighted Regression (LWR) - House Price Prediction")
# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("../Datasets/kc_house_data.csv")

    # take smaller sample to avoid heavy computation
    df = df.sample(400, random_state=42)

    return df

with st.spinner("Loading dataset..."):
    df = load_data()

st.success("Dataset Loaded")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Feature Selection
# -----------------------------
X = df["sqft_living"].values
y = df["price"].values

# normalize
X_norm = (X - X.min()) / (X.max() - X.min())

# bandwidth
tau = 0.1


# -----------------------------
# LWR Function
# -----------------------------
def lwr_predict(x_query, X, y, tau):

    weights = np.exp(-(X - x_query)**2 / (2 * tau**2))

    X_design = np.vstack([np.ones(len(X)), X]).T
    W = np.diag(weights)

    theta = np.linalg.pinv(X_design.T @ W @ X_design) @ X_design.T @ W @ y

    return theta[0] + theta[1] * x_query


# -----------------------------
# Training (LWR fitting)
# -----------------------------
with st.spinner("Loading model..."):

    # create smooth grid instead of full dataset
    X_grid = np.linspace(X_norm.min(), X_norm.max(), 100)

    y_pred_curve = np.array([
        lwr_predict(x, X_norm, y, tau) for x in X_grid
    ])

st.success("Model Ready")

# -----------------------------
# Evaluation
# -----------------------------
y_pred_eval = np.array([
    lwr_predict(x, X_norm, y, tau) for x in X_norm[:200]
])

mse = mean_squared_error(y[:200], y_pred_eval)
r2 = r2_score(y[:200], y_pred_eval)

st.subheader("Evaluation Metrics")

col1, col2 = st.columns(2)

col1.metric("Mean Squared Error", f"{mse:,.2f}")
col2.metric("R² Score", f"{r2:.3f}")

# -----------------------------
# Graph
# -----------------------------
st.subheader("LWR Regression Curve")

fig, ax = plt.subplots()

ax.scatter(X_norm, y, alpha=0.4, label="Data Points")
ax.plot(X_grid, y_pred_curve, color="red", linewidth=2, label="LWR Curve")

ax.set_xlabel("Normalized sqft_living")
ax.set_ylabel("Price")

ax.legend()

st.pyplot(fig)

# -----------------------------
# Prediction
# -----------------------------
st.subheader("Predict House Price")

sqft_input = st.number_input(
    "Enter sqft_living",
    min_value=int(df["sqft_living"].min()),
    max_value=int(df["sqft_living"].max()),
    value=2000
)

sqft_norm = (sqft_input - df["sqft_living"].min()) / (
    df["sqft_living"].max() - df["sqft_living"].min()
)

predicted_price = lwr_predict(sqft_norm, X_norm, y, tau)

st.success(f"Predicted House Price: ${predicted_price:,.2f}")