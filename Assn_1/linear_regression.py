# linear_regression.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train_model():
    df = pd.read_csv("dataset.csv")

    le = LabelEncoder()
    df['city'] = le.fit_transform(df['city'])

    X = df[['temperature_celsius', 'humidity_percent',
            'household_size', 'ac_usage_hours', 'city']]
    y = df['electricity_units_kwh']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, le

def predict(model, le, temp, humidity, size, ac_hours, city):
    city_encoded = le.transform([city])[0]
    prediction = model.predict([[temp, humidity, size, ac_hours, city_encoded]])
    return prediction[0]