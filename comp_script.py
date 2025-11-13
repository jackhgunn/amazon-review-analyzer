import pandas as pd
import joblib
import sys
from pathlib import Path
import json
import string
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import spacy
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import os

sys.path.append(str(Path(__file__).resolve().parent / "webapp"))
from utils.constants import (
    CATEGORY_MAPPING,
)

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from preprocess import preprocess_text

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "model", "best_review_classifier.pkl"
)


sys.path.append(str(Path(__file__).resolve().parent / "src"))

def load_nltk_and_spacy():
    # Download VADER lexicon if not already present and only if you included sentiment analysis as a feature
    try:
        nltk.data.find("vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    # Load spaCy model (disable unused components for speed). This will be used to tokenize our reviews
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    analyzer = SentimentIntensityAnalyzer()

    return nlp, analyzer

def get_xgb_model():
    # 1. Create a Path object to your saved model (i.e. model_name.pkl)
    best_model_path = Path(__file__).resolve().parent.parent / "model" / "best_review_classifier.pkl"

    # 2. Load the model using joblib
    best_model = joblib.load(best_model_path)

    return {"best_model": best_model}

def extract_features(text, rating=5.0, include_pos=True, include_sentiment=True):
    from feature_extraction import extract_features as fe_extract_features

    nlp, sia = load_nltk_and_spacy()

    # Create a DataFrame with the input text and rating
    df = pd.DataFrame([{"cleaned_text": text, "rating": rating}])

    # Extract features using the feature extraction function
    df_features = fe_extract_features(df, include_pos=include_pos, include_sentiment=include_sentiment)

    return df_features

def prepare_features(text, feature_names, category="unknown", rating=5.0):
    # Extract features from the input text
    df_features = extract_features(text, rating=rating, include_pos=True, include_sentiment=True).drop(columns=["cleaned_text"])

    # Map category to its corresponding feature value
    category_feature = CATEGORY_MAPPING.get(category)
    
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
    features = prepare_features(text, category=category, rating=rating)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = probabilities[prediction]
    
    label = "Human" if prediction == 1 else "AI"

    return label, confidence, probabilities.tolist()


def load_test_data():
    """Load the hidden test set"""
    # In actual competition, this would load from a file
    # Until the competition, you will just get one sample
    test_data = pd.DataFrame(
        [
            {
                "category": "Kindle_Store_5",
                "rating": 3.0,
                "label": "CG",  # AI-generated
                "text_": """Eva is on her to find a way to escape her abusive mother. When she meets a handsome stranger she doesn't know is the man she wants but the man she wants is strong and handsome. She is in love with him and will do anything to get her happily ever after.

I loved this book! I can't wait to see how the next book comes out!I received a free copy of this book from the author for an honest review.

This book was so good. I loved it. I loved the character development. I loved the relationship between the two main characters. The romance was real, and the story flowed at a great pace. It was a fun read. I would definitely recommend this book to anyone.I received this book for an honest review.  This is my first book by this author and I was very happy to see it.  I love the characters and the plot.  I will definitely be looking for more by this author.  This is a great series and I look forward to reading more from this author""",
            }
        ]
    )

    # Convert labels: CG (AI) = 0, OR (Human) = 1
    test_data["label_numeric"] = test_data["label"].map({"CG": 0, "OR": 1})

    return test_data


def main():
    print("=" * 60)
    print("Amazon Review Classification Competition")
    print("Model Evaluation Script")
    print("=" * 60)
    print()

    # Load student's model
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        print("Please save your trained model as 'model_name.pkl'")
        return

    print("Loading your model...")
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return

    feature_names_path = "model/best_feature_names.json"
    if not Path(feature_names_path).exists():
        print(f"❌ Error: Feature names file not found at {feature_names_path}")
        return

    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

    # Load NLP models
    print("\nLoading NLP models...")
    nlp, analyzer = load_nltk_and_spacy()
    print("✅ NLP models loaded!")

    # Load test data
    print("\nLoading test data...")
    test_df = load_test_data()
    print(f"✅ Loaded {len(test_df)} test samples")

    # Extract features for all test samples
    print("\nExtracting features...")
    feature_dfs = []
    for idx, row in test_df.iterrows():
        features = extract_features(row["text_"], row["rating"], nlp, analyzer)
        features = prepare_features(features, feature_names, row["category"])
        feature_dfs.append(features)

    X_test = pd.concat(feature_dfs, ignore_index=True)
    y_test = test_df["label_numeric"].values

    # Make predictions
    print("Making predictions...")
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"❌ Error making predictions: {str(e)}")
        return

    # Calculate metrics
    print("\n" + "=" * 60)
    print("📈 RESULTS")
    print("=" * 60)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")

    print(f"\n🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🎯 Recall:    {recall:.4f}")
    print(f"🎯 F1 Score:  {f1:.4f}")

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["AI (CG)", "Human (OR)"]))

    print("\n" + "=" * 60)
    print(f"🏆 Your model achieved {accuracy*100:.2f}% accuracy!")
    print("=" * 60)


if __name__ == "__main__":
    main()
