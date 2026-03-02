# ⚡ Electricity Usage ML System

## 📌 Project Description

This project analyzes and predicts household electricity consumption using Machine Learning techniques.

It applies:

* **Linear Regression** → Predict daily electricity usage (kWh)
* **Decision Tree** → Classify usage as High or Low
* **KMeans Clustering** → Identify electricity consumption patterns
* **PCA (Principal Component Analysis)** → Reduce dimensionality and visualize clusters

The system is deployed using **Streamlit**, allowing users to enter input values and view predictions along with cluster visualization.

---

## 🛠️ Requirements

* Python 3.10+
* pandas
* numpy
* scikit-learn
* streamlit
* matplotlib

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

1. Navigate to the project folder:

```bash
cd Assn_1
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open the browser at:

```
http://localhost:8501
```

4. Enter input values in the sidebar and click **Predict**.

---

## 🎯 Output

The system displays:

* Predicted Electricity Usage (kWh)
* Usage Category (High/Low)
* Assigned Consumption Pattern Cluster
* PCA-based Cluster Visualization Graph