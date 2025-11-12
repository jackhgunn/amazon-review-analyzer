import pandas as pd
from pathlib import Path # To manipulate file paths easier as objects rather than just strings
import sys

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from preprocess import preprocess_text
from feature_extraction import extract_features

dataset_path = Path(__file__).resolve().parent / "fake-reviews.csv"
dataset_df = pd.read_csv(dataset_path)

dataset_df["cleaned_text"] = dataset_df["text_"].apply(preprocess_text)
POS_Tagging = True
Sentiment_Analysis = True
dataset_df = extract_features(dataset_df, include_pos=POS_Tagging, include_sentiment=Sentiment_Analysis)

processed_dataset_path = Path(__file__).resolve().parent / "processed-fake-reviews.csv"
dataset_df.to_csv(processed_dataset_path, index=False)