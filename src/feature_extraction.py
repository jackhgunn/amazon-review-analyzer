import pandas as pd
import string
import spacy # Tokenize text so we can count each POS
from collections import Counter # To keep a dictionary of the count of each POS

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def extract_features(df, include_pos=False):
    df["char_length"] = df["cleaned_text"].apply(len)
    df["word_count"] = df["cleaned_text"].apply(lambda x: len(x.strip().split()))
    df["punctuation_ct"] = df["cleaned_text"].apply(lambda x: sum(1 for char in x if char in string.punctuation))
    df["is_extreme_star"] = df["rating"].isin([1.0, 5.0])
    
    # Add this right before returning the dataframe in extract_features
    if include_pos:
        df = add_pos_features(df)
    
    return df

POS_WHITELIST = {"VERB", "NOUN", "ADV"}

def pos_counts(text):
    doc = nlp(text) # tokenizes the text
    
    return Counter(token.pos_ for token in doc if token.pos_ in POS_WHITELIST) # count the number of each pos in the given tokenized text. We're only doing this for the whitelisted POS

def add_pos_features(df):
    pos_data = df["cleaned_text"].apply(pos_counts)
    pos_df = pd.DataFrame(list(pos_data)).fillna(0) # fill null counts with 0
    pos_df.index = df.index # align columns with original df (dataframe)
    
    return pd.concat([df, pos_df], axis=1)
