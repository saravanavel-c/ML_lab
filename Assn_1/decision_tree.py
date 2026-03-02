# decision_tree.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_model():
    df = pd.read_csv("dataset.csv")

    le = LabelEncoder()
    df['city'] = le.fit_transform(df['city'])

    threshold = df['electricity_units_kwh'].mean()
    df['usage_label'] = (df['electricity_units_kwh'] > threshold).astype(int)

    X = df[['temperature_celsius', 'humidity_percent',
            'household_size', 'ac_usage_hours', 'city']]
    y = df['usage_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    return model, le, acc, prec, rec

def predict(model, le, temp, humidity, size, ac_hours, city):
    city_encoded = le.transform([city])[0]
    prediction = model.predict([[temp, humidity, size, ac_hours, city_encoded]])
    return "High Usage" if prediction[0] == 1 else "Low Usage"