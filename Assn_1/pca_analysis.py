# pca_analysis.py

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def train_model():
    df = pd.read_csv("dataset.csv")

    features = df[['temperature_celsius', 'humidity_percent',
                   'ac_usage_hours', 'electricity_units_kwh']]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    pca.fit(scaled)

    return pca, scaler

def transform(pca, scaler, temp, humidity, ac_hours, usage):
    scaled_input = scaler.transform([[temp, humidity, ac_hours, usage]])
    components = pca.transform(scaled_input)
    return components