# AI Text Detector

## Overview
This project is an AI-based text classifier that predicts whether a given text is AI-generated or human-written. The model is deployed using Streamlit, allowing users to input text and receive predictions instantly.

## How It Works
1. **Preprocessing:**
   - The dataset contains cleaned text along with text length.
   - The text is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency).
   - The text length is added as an additional feature.

2. **Model Training:**
   - An XGBoost classifier is trained on the TF-IDF features and text length.
   - The dataset is split using stratified sampling to maintain class balance.
   - The model achieves high accuracy in distinguishing AI-generated and human-written text.

3. **Deployment:**
   - The trained model and TF-IDF vectorizer are saved as `.pkl` files.
   - A Streamlit-based UI allows users to enter text for classification.
   - The text is transformed using the saved TF-IDF vectorizer and passed to the model for prediction.

## Findings from Analysis
To analyze the distinction between AI-generated and human-written text, a **Word Cloud** was generated for both categories. The analysis revealed:
   - **AI-generated text** tends to have a more structured, repetitive, and formal vocabulary.
   - **Human-written text** shows more variation, creativity, and informal language.
   - Certain keywords and phrases were significantly more common in AI-generated text compared to human-written text.

This clear distinction in word usage allowed the XGBoost model to effectively classify the text based on feature patterns.

Since we were not interested in sentiments and more on distribution of words, I went with tf-idf embeddings. 


## Running the Project
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Future Improvements
- Enhance feature engineering by adding more linguistic and semantic features.
- Experiment with deep learning models for improved classification.
- Optimize model deployment for real-time use cases.

This project demonstrates a robust approach to AI-generated text detection using ML techniques.

