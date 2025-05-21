import tkinter as tk
from tkinter import messagebox
import joblib
import re
import numpy as np

# Load the trained model and vectorizer
model = joblib.load('malayalam_sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Rule-based neutral word list
neutral_words = [
    "വായിച്ചു", "എഴുന്നേറ്റു", "പോയി", "വന്നു", "കണ്ടു", "കൈരളി", "നിലവാരത്തിൽ", "തീർത്തു",
    "വെച്ചിരിക്കുന്നു", "ഉണ്ടായിരുന്നു", "തുറന്നു", "ഇന്നാണ്", "കഴിഞ്ഞു", "വായിക്കുന്നു"
]

def clean_text(text):
    # Remove punctuation, digits, and extra spaces
    text = re.sub(r'[^\u0D00-\u0D7F\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_neutral_fallback(text):
    for word in neutral_words:
        if word in text:
            return True
    return False

def analyze_sentiment():
    user_input = input_text.get("1.0", tk.END).strip()

    if not user_input:
        messagebox.showwarning("Warning", "Please enter a sentence.")
        return

    cleaned_input = clean_text(user_input)

    # Rule-based fallback
    if detect_neutral_fallback(cleaned_input):
        result_label.config(text="Predicted Sentiment: Neutral (Rule-Based)", fg="blue")
        return

    user_vector = vectorizer.transform([cleaned_input])
    probs = model.predict_proba(user_vector)
    max_prob = np.max(probs)

    if max_prob < 0.5:
        result_label.config(text="Predicted Sentiment: Neutral (Low Confidence)", fg="blue")
    else:
        prediction = model.predict(user_vector)[0]
        result_label.config(text=f"Predicted Sentiment: {prediction.capitalize()}", fg="green")

# GUI Layout
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
