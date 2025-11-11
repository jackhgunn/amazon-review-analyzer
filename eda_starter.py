import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (make sure the path is correct)
df = pd.read_csv("fake-reviews.csv")

# Quick look at the first rows
print(df.head())

# Count missing values in each column
print(df.isnull().sum())

# Character length of each review
df["char_length"] = df["text_"].apply(len)

# Word count of each review
df["word_count"] = df["text_"].str.split().apply(len)

df["sentence_count"] = df["text_"].str.split(r'[.!?]').apply(lambda x: len([s for s in x if s.strip()]))
df["average_sentence_length"] = df.apply(lambda row: row["char_length"] / row["sentence_count"] if row["sentence_count"] > 0 else 0, axis=1)


"""
sns.boxplot(x="char_length", y="label", data=df)
plt.title("Character Length of Reviews by Label")
plt.xlabel("Character Length")
plt.ylabel("Review Label")
plt.show()
"""

sns.boxplot(x="average_sentence_length", y="label", data=df)
plt.title("Avg. Sentence Length of Reviews by Label")
plt.xlabel("Characters per Sentence")
plt.ylabel("Review Label")
plt.show()