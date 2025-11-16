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
    except:
        nltk.download("vader_lexicon", quiet=True)

    # Load spaCy model (disable unused components for speed). This will be used to tokenize our reviews
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.add_pipe("sentencizer")

    analyzer = SentimentIntensityAnalyzer()

    return nlp, analyzer

@st.cache_resource
def get_xgb_model():
    # 1. Create a Path object to your saved model (i.e. model_name.pkl)
    best_model_path = Path(__file__).resolve().parent.parent / "model" / "best_review_classifier.pkl"

    # 2. Load the model using joblib
    best_model = joblib.load(best_model_path)

    return {"best_model": best_model}

def extract_features(text, rating=5.0, include_pos=True, include_sentiment=True):
    from feature_extraction import extract_features as fe_extract_features

    nlp, sia = get_nlp_models()

    # Create a DataFrame with the input text and rating
    df = pd.DataFrame([{"cleaned_text": text, "rating": rating}])

    # Extract features using the feature extraction function
    df_features = fe_extract_features(df, nlp, sia, include_pos=include_pos, include_sentiment=include_sentiment)

    return df_features

def prepare_features_for_prediction(text, category="unknown", rating=5.0):
    # Extract features from the input text
    df_features = extract_features(text, rating=rating, include_pos=True, include_sentiment=True).drop(columns=["cleaned_text"])

    # Map category to its corresponding feature value
    category_feature = CATEGORY_MAPPING.get(category)

    # Load the feature names your model expects
    best_feature_names_path = Path(__file__).resolve().parent.parent / "model" / "best_feature_names.json"
    sys.path.append(str(Path(__file__).resolve().parent.parent / "model/"))
    with open(best_feature_names_path, "r") as f:
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


def main():
    st.set_page_config(page_title="Amazon Review Analyzer", page_icon="🤖", layout="wide")

    st.title("Amazon Review Analyzer")
    st.header("**Check if an Amazon review is written by a human or AI!**")

    with st.spinner("Loading XGBoost model..."):
        model_dict = get_xgb_model()
    if model_dict is None:
        st.error("Failed to load XGBoost model. Please check if model files exist")
        return
    st.success("XGBoost model loaded successfully!")
    
    input_col, output_col = st.columns(2)
    with input_col:
        input_container = st.container()
        with input_container:
            input_review = st.text_area("Enter your Amazon review text below:", height=240)
            category = st.selectbox("Select the product category:", options=list(CATEGORY_MAPPING.keys()))
            rating = st.slider("Select the review rating (1-5):", min_value=1.0, max_value=5.0, value=5.0, step=0.5)
            analyze_button = st.button("Analyze Review")

    with output_col:
        label = None
        confidence = None

        if analyze_button and input_review.strip():
            with st.spinner("Analyzing review..."):
                try:
                    dataset_category = CATEGORY_MAPPING[category]
                    label, confidence, probabilities = xgb_predict(
                        text=input_review,
                        model=model_dict["best_model"],
                        category=dataset_category,
                        rating=rating
                    )
                    confidence = (confidence * 100).round(2)
                    if label == "Human":
                        st.success("Human!")
                    else:
                        st.error("AI!")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.exception(e)
        elif analyze_button and not input_review.strip():
            st.warning("Please enter a review to analyze!")
        
        prediction_container = st.container()
        with prediction_container:
            st.header("**Results:**")
            st.write("Label:", label)
            st.write("Confidence:", confidence, " %")

        features_container = st.container()
        with features_container:
            st.subheader("**Important Features:**")

            if input_review.strip():
                features = extract_features(text=input_review, rating=rating, include_pos=True, include_sentiment=True)

                first_features_col, second_features_col = st.columns(2)
                with first_features_col:
                    st.write("1. Average Word Length: ", features["avg_word_length"].values[0].round(2))
                    st.write("2. Ratio of Determiner POS: ", features["POSR_DET"].values[0].round(2) if "POSR_DET" in features.columns else 0.0)
                    st.write("3. Number of Verb POS: ", features["POS_VERB"].values[0] if "POS_VERB" in features.columns else 0.0)
                    st.write("4. Number of Adposition POS: ", features["POS_ADP"].values[0] if "POS_ADP" in features.columns else 0.0)
                    st.write("5. Ratio of Adposition POS: ", features["POSR_ADP"].values[0].round(2) if "POSR_ADP" in features.columns else 0.0)
                with second_features_col:
                    st.write("6. Unique Words Percentage: ", features["unique_words_pct"].values[0].round(2) * 100, "%")
                    st.write("7. Ratio of Auxiliary POS: ", features["POSR_AUX"].values[0].round(2) if "POSR_AUX" in features.columns else 0.0)
                    st.write("8. Number of Noun POS: ", features["POS_NOUN"].values[0] if "POS_NOUN" in features.columns else 0.0)
                    st.write("9. Average Sentence Length: ", features["avg_sentence_length"].values[0].round(2))
                    st.write("10. Number of Words: ", features["num_words"].values[0])

    data_context = st.container()
    with data_context:
        dataset = pd.read_csv(Path(__file__).resolve().parent.parent / "fake-reviews.csv")
        datafeatures = pd.read_json(Path(__file__).resolve().parent.parent / "model" / "best_feature_names.json")
        metadata = pd.read_json(Path(__file__).resolve().parent.parent / "model" / "best_model_metadata.json")
        
        st.header("**Data Context:**")
        st.write("The XGBoost model was trained/tested on a dataset of Amazon reviews containing both human-written (OR) and AI-generated (CG) reviews.")
        st.write("**Dataset:**")
        st.write(pd.concat([dataset.head(5), dataset.tail(5)], ignore_index=False))
        st.write("Feature extraction yielded 59 data features used by the model (sentence structure, all POS tags, and sentiment analysis). This would be trimmed to 57 features after applying an importance threshold.")
        st.write("**Data Features:**")
        st.write(datafeatures.T)
        st.write("After getting the initial results, GridCV hyperparameter tuning was performed to optimize the model's performance.")
        st.write("**Hyperparameters:**")
        st.write(metadata["best_params"])
        st.write("After testing different values, an importance threshold of 0.002 was applied to select the most impactful features for the final model. The final model achieved an AUC score of 0.9656 on the test data.")

if __name__ == "__main__":
    main()