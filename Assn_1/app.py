# app.py

import streamlit as st
from linear_regression import train_model as train_lr, predict as predict_lr
from decision_tree import train_model as train_dt, predict as predict_dt
from kmeans_clustering import train_model as train_km, predict as predict_km
from pca_analysis import train_model as train_pca, transform as transform_pca

st.title("⚡ Electricity Usage ML System")

st.sidebar.header("Enter Input Data")

temp = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 30.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 60.0)
size = st.sidebar.number_input("Household Size", 1, 10, 4)
ac_hours = st.sidebar.number_input("AC Usage Hours", 0.0, 24.0, 5.0)
city = "Delhi"

if st.sidebar.button("Predict"):

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Train supervised models
    lr_model, lr_le = train_lr()
    dt_model, dt_le = train_dt()

    # Predict usage
    predicted_usage = predict_lr(lr_model, lr_le, temp, humidity, size, ac_hours, city)
    usage_category = predict_dt(dt_model, dt_le, temp, humidity, size, ac_hours, city)

    # Load full dataset for clustering visualization
    df = pd.read_csv("dataset.csv")

    features = df[['temperature_celsius', 'humidity_percent',
                   'ac_usage_hours', 'electricity_units_kwh']]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    # Train KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_data)

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # Transform user input
    user_scaled = scaler.transform([[temp, humidity, ac_hours, predicted_usage]])
    user_cluster = kmeans.predict(user_scaled)
    user_pca = pca.transform(user_scaled)

    # Display results
    st.subheader("Results")
    st.write("🔹 Predicted Electricity Usage (kWh):", round(predicted_usage, 2))
    st.write("🔹 Usage Category:", usage_category)
    st.write("🔹 Assigned Cluster:", int(user_cluster[0]))

    # Explain PCA
    st.write("🔹 PCA Components (Reduced 2D Representation):")
    st.write("Principal Component 1:", round(user_pca[0][0], 3))
    st.write("Principal Component 2:", round(user_pca[0][1], 3))

    # Plot clusters
    fig, ax = plt.subplots()

    for i in range(3):
        ax.scatter(
            pca_data[df['cluster'] == i, 0],
            pca_data[df['cluster'] == i, 1],
            label=f"Cluster {i}"
        )

    # Highlight user input
    ax.scatter(user_pca[0][0], user_pca[0][1],
               color='black', marker='X', s=200,
               label="Your Input")

    ax.set_title("KMeans Clusters (Visualized using PCA)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()

    st.pyplot(fig)