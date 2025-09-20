import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("fake-reviews.csv")

print(df.head())
print(df.isnull().sum())

df["char_length"] = df["text_"].apply(len)
df["word_count"] = df["text_"].str.split().apply(len)

# ChatGPT code for average sentence length
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