import streamlit as st
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer using joblib
model = joblib.load("ai_text.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict(text):
    text_length = len(text.split())
    text_tfidf = vectorizer.transform([text]).toarray()
    text_combined = np.hstack([text_tfidf, np.array([[text_length]])])
    prediction = model.predict(text_combined)[0]
    return "AI-generated" if prediction == 1 else "Human-written"

# Streamlit UI
st.title("AI Text Detector")
user_input = st.text_area("Enter text to check if it's AI-generated, minimum text length of 80:")

if st.button("Predict"):
    if user_input.strip():
        result = predict(user_input)
        st.write(f"### Prediction: {result}")
    else:
        st.write("Please enter some text to analyze.")
