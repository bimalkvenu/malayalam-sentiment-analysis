import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

# Load the dataset
df = pd.read_csv("D:/LPU/4th sem/CAP457_ BIGDATA (LABORATORY)/Project/MABSA4000.csv")

# Clean data: drop empty rows
df.dropna(subset=['Review', 'Sentiment'], inplace=True)

# Print a sample and class distribution
print(df.head())
print(df['Sentiment'].value_counts())
print(df.columns)

# Features and labels
X = df['Review']
y = df['Sentiment']

# Vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Rule-based neutral word fallback (very basic)
neutral_words = [
    "വായിച്ചു", "എഴുന്നേറ്റു", "പോയി", "വന്നു", "കണ്ടു", "കൈരളി", "നിലവാരത്തിൽ", "തീർത്തു",
    "വെച്ചിരിക്കുന്നു", "ഉണ്ടായിരുന്നു", "തുറന്നു", "ഇന്നാണ്", "കഴിഞ്ഞു", "വായിക്കുന്നു"
]

def detect_neutral_fallback(text):
    for word in neutral_words:
        if word in text:
            return True
    return False

# Sentiment Prediction Loop
while True:
    user_input = input("\nEnter a Malayalam sentence to analyze sentiment (or type 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        break
    if not user_input:
        print("Please enter a non-empty sentence.")
        continue

    # Check if neutral keywords are dominant
    if detect_neutral_fallback(user_input):
        print("Predicted Sentiment: neutral (rule-based fallback)")
    else:
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)
        print("Predicted Sentiment:", prediction[0])
