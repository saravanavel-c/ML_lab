# kmeans_clustering.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def train_model():
    df = pd.read_csv("dataset.csv")

    features = df[['temperature_celsius', 'humidity_percent',
                   'ac_usage_hours', 'electricity_units_kwh']]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=3, random_state=42)
    model.fit(scaled)

    return model, scaler

def predict(model, scaler, temp, humidity, ac_hours, usage):
    scaled_input = scaler.transform([[temp, humidity, ac_hours, usage]])
    cluster = model.predict(scaled_input)
    return cluster[0]