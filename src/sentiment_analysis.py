import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

os.makedirs("output", exist_ok=True)

# Load dataset
df = pd.read_csv(
    "data/Amazon_Reviews.csv",
    engine="python"
)

# Dataset Information
print("\n===== DATASET INFORMATION =====")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# STEP 9 GOES HERE
def rating_category(rating):

    if "5" in str(rating) or "4" in str(rating):
        return "Positive"

    elif "3" in str(rating):
        return "Neutral"

    else:
        return "Negative"

df["Expected Sentiment"] = (
    df["Rating"]
    .astype(str)
    .apply(rating_category)
)

# Remove null values
df = df.dropna()

# Sentiment Analysis Function
def analyze_sentiment(text):

    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive", polarity

    elif polarity < 0:
        return "Negative", polarity

    else:
        return "Neutral", polarity

# Apply sentiment analysis
results = df["Review Text"].apply(analyze_sentiment)

df["Sentiment"] = results.apply(lambda x: x[0])
df["Polarity"] = results.apply(lambda x: x[1])

# Save results
df.to_csv(
    "output/sentiment_results.csv",
    index=False
)

print(df.head())

print("\nAnalysis Complete")

summary = df["Sentiment"].value_counts()

print("\nSummary")

print(summary)

summary.to_csv(
    "output/sentiment_summary.csv"
)

plt.figure(figsize=(8,8))

summary.plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Sentiment Distribution")

plt.ylabel("")

plt.savefig(
    "output/pie_chart.png"
)

plt.show()

plt.figure(figsize=(8,5))

summary.plot(
    kind="bar"
)

plt.title(
    "Sentiment Analysis Results"
)

plt.xlabel("Sentiment")

plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    "output/bar_chart.png"
)

plt.show()

all_text = " ".join(
    df["Review Text"].astype(str)
)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(all_text)

plt.figure(figsize=(10,5))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Most Frequent Words")

plt.savefig(
    "output/wordcloud.png"
)

plt.show()

total_reviews = len(df)

positive = len(
    df[df["Sentiment"]=="Positive"]
)

negative = len(
    df[df["Sentiment"]=="Negative"]
)

neutral = len(
    df[df["Sentiment"]=="Neutral"]
)

print("\n===== PROJECT SUMMARY =====")

print(f"Total Reviews: {total_reviews}")

print(f"Positive Reviews: {positive}")

print(f"Negative Reviews: {negative}")

print(f"Neutral Reviews: {neutral}")


print()

print("\n===== SENTIMENT SUMMARY =====")

summary = df["Sentiment"].value_counts()

print(summary)

accuracy = (
    df["Expected Sentiment"]
    == df["Sentiment"]
).mean() * 100

print(f"Accuracy: {accuracy:.2f}%")
