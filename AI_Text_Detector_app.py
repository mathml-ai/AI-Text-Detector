import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from io import BytesIO

# GitHub URLs for model and vectorizer
MODEL = "ai_detector.joblib"
VECTORIZER_URL = "tfidf_vectorizer.joblib"

# Function to download and load joblib files
@st.cache_resource
def load_joblib(url):
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))

# Load model and vectorizer
model = load_joblib(MODEL_URL)
vectorizer = load_joblib(VECTORIZER_URL)

# Function to clean text (implement _text_cleaning here)
def _text_cleaning(df, text_column, cleaned_column):
    df[cleaned_column] = df[text_column].str.lower()  # Example cleaning, replace with your own
    return df

# Streamlit UI
st.title("AI Text Detector")

# User input
test_text = st.text_area("Enter text:", "")

if st.button("Predict"):
    if test_text:
        # Create DataFrame
        test_df = pd.DataFrame([test_text], columns=['text'])
        
        # Text cleaning
        test_df = _text_cleaning(test_df, 'text', 'cleaned_text')

        # TF-IDF transformation
        test_tfidf = vectorizer.transform(test_df['cleaned_text']).toarray()

        # Compute text length feature
        test_text_length = len(test_text)

        # Combine TF-IDF vector with text_length
        test_features = np.append(test_tfidf, test_text_length).reshape(1, -1)

        # Make predictions
        prediction = model.predict(test_features)[0]
        prediction_proba = model.predict_proba(test_features)[:, 1][0]

        # Display results
        st.subheader("Prediction Result")
        st.write(f"**Prediction:** {'Positive' if prediction == 1 else 'Negative'}")
        st.write(f"**Prediction Probability:** {prediction_proba:.4f}")
    else:
        st.warning("Please enter text to analyze.")

