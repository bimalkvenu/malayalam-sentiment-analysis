import os
import pandas as pd
import numpy as np
import joblib
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

# ============ STEP 1: Train and Save Model if not already present ============

model_path = 'malayalam_sentiment_model.pkl'
vectorizer_path = 'vectorizer.pkl'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    print("🔁 Training model...")

    # Load and clean data
    df = pd.read_csv("D:/LPU/4th sem/CAP457_ BIGDATA (LABORATORY)/Project/MABSA4000.csv")
    df.dropna(subset=['Review', 'Sentiment'], inplace=True)

    X = df['Review']
    y = df['Sentiment']

    # Vectorization
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # Model training
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("✅ Model and vectorizer saved!")

else:
    print("✅ Model already exists, loading...")

# ============ STEP 2: Load model and vectorizer ============

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# ============ STEP 3: Rule-based Fallback ============

neutral_words = [
    "വായിച്ചു", "എഴുന്നേറ്റു", "പോയി", "വന്നു", "കണ്ടു", "കൈരളി", "നിലവാരത്തിൽ", "തീർത്തു",
    "വെച്ചിരിക്കുന്നു", "ഉണ്ടായിരുന്നു", "തുറന്നു", "ഇന്നാണ്", "കഴിഞ്ഞു", "വായിക്കുന്നു"
]

def clean_text(text):
    # Remove symbols, digits, etc.
    text = re.sub(r'[^\u0D00-\u0D7F\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_neutral_fallback(text):
    return any(word in text for word in neutral_words)

# ============ STEP 4: GUI ============

def analyze_sentiment():
    user_input = input_text.get("1.0", tk.END).strip()

    if not user_input:
        messagebox.showwarning("Warning", "Please enter a sentence.")
        return

    cleaned_input = clean_text(user_input)

    # Rule-based check
    if detect_neutral_fallback(cleaned_input):
        result_label.config(text="Predicted Sentiment: Neutral (Rule-Based)", fg="blue")
        return

    # ML-based prediction
    user_vector = vectorizer.transform([cleaned_input])
    prediction = model.predict(user_vector)[0]
    result_label.config(text=f"Predicted Sentiment: {prediction.capitalize()}", fg="green")

# GUI layout
window = tk.Tk()
window.title("Malayalam Sentiment Analyzer")
window.geometry("500x300")
window.resizable(False, False)

title_label = tk.Label(window, text="Malayalam Sentiment Analyzer", font=("Arial", 16))
title_label.pack(pady=10)

input_text = tk.Text(window, height=5, width=50, font=("Arial", 12))
input_text.pack(pady=10)

analyze_button = tk.Button(window, text="Analyze Sentiment", font=("Arial", 12), command=analyze_sentiment)
analyze_button.pack(pady=5)

result_label = tk.Label(window, text="", font=("Arial", 14))
result_label.pack(pady=10)

window.mainloop()
