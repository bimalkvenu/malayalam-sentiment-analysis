import pandas as pd
from textblob import TextBlob

# Load the CSV file
df = pd.read_csv("d:/LPU/4th sem/CAP457_ BIGDATA (LABORATORY)/MABSA with English translation.csv")
print(df.columns)

# Define function to get sentiment from TextBlob
def get_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment function to English comments
df['Sentiment'] = df['English Translation (Approximate) '].apply(get_sentiment)

# Save the updated file
df.to_csv("d:/LPU/4th sem/CAP457_ BIGDATA (LABORATORY)/MABSA_with_sentiment_output.csv", index=False)

print("Sentiment analysis complete. File saved as MABSA_with_sentiment_output.csv")
