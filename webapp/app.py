import streamlit as st
import pandas as pd
import torch
import sys
from pathlib import Path
import joblib
import string
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import json
import spacy
from collections import Counter
from utils.constants import CATEGORY_MAPPING # You shouldn’t have to change this unless you placed constants elsewhere


sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

@st.cache_resource
def get_nlp_models():
    # Download VADER lexicon if not already present and only if you included sentiment analysis as a feature
    try:
        nltk.data.find("vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    # Load spaCy model (disable unused components for speed). This will be used to tokenize our reviews
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    analyzer = SentimentIntensityAnalyzer()

    return nlp, analyzer

@st.cache_resource
def get_xgb_model():
    # 1. Create a Path object to your saved model (i.e. model_name.pkl)
    best_model_path = Path(__file__).resolve().parent.parent / "model" / "best_review_classifier.pkl"

    # 2. Load the model using joblib
    best_model = joblib.load(best_model_path)

    return {"best_model": best_model}

def extract_features(text, rating=5.0, include_pos=False, include_sentiment=False):
    from feature_extraction import extract_features as fe_extract_features

    nlp, sia = get_nlp_models()

    # Create a DataFrame with the input text and rating
    df = pd.DataFrame([{"cleaned_text": text, "rating": rating}])

    # Extract features using the feature extraction function
    df_features = fe_extract_features(df, include_pos=include_pos, include_sentiment=include_sentiment)

    return df_features

def prepare_features_for_prediction(text, category="unknown", rating=5.0):
    # Extract features from the input text
    df_features = extract_features(text, rating=rating, include_pos=True, include_sentiment=True).drop(columns=["cleaned_text"])

    # Map category to its corresponding feature value
    category_feature = CATEGORY_MAPPING.get(category)

    # Load the feature names your model expects
    sys.path.append(str(Path(__file__).resolve().parent.parent / "model/"))
    with open("../model/best_feature_names.json", "r") as f:
        feature_names = json.load(f)
    
    # Add category features df_features
    for i, cat in enumerate(feature_names[-10:]):
        df_features[cat] = 1 if cat == category_feature else 0

    # Check for all expected features and put them in the right order
    for feature in feature_names:
        if feature not in df_features.columns:
            df_features[feature] = 0.0

    # Return features in the exact order the model expects
    return df_features[feature_names]

def xgb_predict(text, model, category="unknown", rating=5.0):
    features = prepare_features_for_prediction(text, category=category, rating=rating)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = probabilities[prediction]
    
    label = "Human" if prediction == 1 else "AI"

    return label, confidence, probabilities.tolist()