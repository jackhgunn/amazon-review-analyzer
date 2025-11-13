import re
import string
import pandas as pd
import spacy
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer

#nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
#nlp.add_pipe("sentencizer")

#sia = SentimentIntensityAnalyzer()

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def _doc_tokens_sentences(text, nlp):
    text = text if isinstance(text, str) else ""
    doc = nlp(text)

    tokens = [t for t in doc if not t.is_space]
    words = WORD_RE.findall(text)
    sents = [s for s in doc.sents if s.text.strip()]
    return doc, tokens, words, sents, text

def _pos_counter(doc):
    return Counter(t.pos_ for t in doc if not t.is_space)

def extract_features(df, nlp, sia, include_pos=True, include_sentiment=True):
    df = df.copy()

    processed = df["cleaned_text"].apply(lambda x: _doc_tokens_sentences(x, nlp))

    df["_doc"]   = processed.apply(lambda x: x[0])
    df["_toks"]  = processed.apply(lambda x: x[1])
    df["_words"] = processed.apply(lambda x: x[2])
    df["_sents"] = processed.apply(lambda x: x[3])
    df["_raw"]   = processed.apply(lambda x: x[4])

    df["num_words"] = df["_words"].apply(len)
    df["num_chars"] = df["_raw"].str.replace(r"\s+", "", regex=True).str.len()
    df["unique_words_pct"] = df.apply(
        lambda r: (len(set(w.casefold() for w in r["_words"])) / r["num_words"]) if r["num_words"] else 0.0,
        axis=1
    )
    df["avg_word_length"] = df.apply(
        lambda r: (sum(len(w) for w in r["_words"]) / r["num_words"]) if r["num_words"] else 0.0,
        axis=1
    )
    df["avg_sentence_length"] = df.apply(
        lambda r: (sum(len(WORD_RE.findall(s.text)) for s in r["_sents"]) / len(r["_sents"]))
                  if len(r["_sents"]) else 0.0,
        axis=1
    )
    df["punctuation_pct"] = df.apply(
        lambda r: (sum(ch in string.punctuation for ch in r["_raw"]) / r["num_chars"]) if r["num_chars"] else 0.0,
        axis=1
    )
    df["num_exclamation_marks"] = df["_raw"].str.count("!")
    df["num_question_marks"]    = df["_raw"].str.count(r"\?")
    df["contraction_pct"] = df.apply(
        lambda r: (sum("'" in w[1:-1] for w in r["_words"]) / r["num_words"]) if r["num_words"] else 0.0,
        axis=1
    )
    df["is_extreme_star"] = df["rating"].isin([1.0, 5.0])

    # POS features
    if include_pos:
        pos_series = df["_doc"].apply(_pos_counter)
        pos_df = pd.DataFrame(list(pos_series)).fillna(0).astype(int)
        pos_df.index = df.index
        
        ratio_df = pos_df.div(df["num_words"].replace(0, pd.NA), axis=0).fillna(0)

        pos_df = pos_df.add_prefix("POS_")
        ratio_df = ratio_df.add_prefix("POSR_")
        df = pd.concat([df, pos_df, ratio_df], axis=1)

    # Sentiment features
    if include_sentiment:
        sent = df["_raw"].apply(sia.polarity_scores)
        df["sent_neg"] = sent.apply(lambda d: d["neg"])
        df["sent_neu"] = sent.apply(lambda d: d["neu"])
        df["sent_pos"] = sent.apply(lambda d: d["pos"])
        df["sent_compound"] = sent.apply(lambda d: d["compound"])

    df.drop(columns=["_doc", "_toks", "_words", "_sents", "_raw"], inplace=True)

    return df