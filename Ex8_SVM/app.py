# ============================================
# Email Spam Detection using SVM (Streamlit)
# ============================================

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Page Configuration
# -------------------------------
st.set_page_config(page_title="Spam Detection App", page_icon="📧")

st.title("📧 Email Spam Detection using SVM")
st.write("This app classifies emails as Spam or Not Spam using Support Vector Machine.")

# -------------------------------
# 2. Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("../Datasets/spam.csv", encoding="latin-1")
    data = data.iloc[:, :2]
    data.columns = ['label', 'text']
    data['label'] = data['label'].astype(str).str.lower().str.strip()
    data['label'] = data['label'].map({'spam': 1, 'ham': 0})
    data = data.dropna(subset=['label'])
    return data

data = load_data()

st.write("### Dataset Preview")
st.dataframe(data.head())

# -------------------------------
# 3. Train Model
# -------------------------------
X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.95,
    min_df=5,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

svm_model = LinearSVC(C=1.0, max_iter=5000)
svm_model.fit(X_train_tfidf, y_train)

accuracy = accuracy_score(y_test, svm_model.predict(X_test_tfidf))

st.success(f"Model trained successfully! Accuracy: {accuracy:.4f}")

# -------------------------------
# 4. User Input Section
# -------------------------------
st.write("## 🔍 Test New Email")

user_input = st.text_area("Enter Email Text Here:")

if st.button("Predict"):
    if user_input.strip() != "":
        input_tfidf = vectorizer.transform([user_input])
        prediction = svm_model.predict(input_tfidf)[0]

        if prediction == 1:
            st.error("🚨 This is a Spam Email!")
        else:
            st.success("✅ This is NOT a Spam Email.")
    else:
        st.warning("Please enter some text.")

# -------------------------------
# 5. End
# -------------------------------
st.write("----")