from flask import Flask, render_template, request
import joblib
import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Load and clean data
DATA_PATH = r"D:/LPU/4th sem/CAP457_ BIGDATA (LABORATORY)/Project/MABSA4000.csv"

NEGATIVE_KEYWORDS = ["ദു:ഖകരമായിരുന്നു", "വഷളായിരുന്ന", "മോശം", "ചീത്ത", "നിരാശ", "അപര്യാപ്തം", "ഇല്ല", "ലഭിച്ചില്ല", "കണ്ടില്ല", "ഉള്ളില്ല"]
NEUTRAL_KEYWORDS = ["വായിച്ചു", "എഴുന്നേറ്റു", "പോയി", "വന്നു", "കണ്ടു", "ക്ലാസിൽ", "ഇന്ന്", "കഴിഞ്ഞു"]

def clean_text(text):
    txt = re.sub(r"[^\u0D00-\u0D7F0-9\s]", "", str(text))
    return re.sub(r"\s+", " ", txt).strip()

def detect_negative_fallback(txt):
    txt = txt.lower()
    if re.search(r"\b\w*ഇല്ല\b", txt):
        return True
    return any(kw in txt for kw in NEGATIVE_KEYWORDS)

def detect_neutral_fallback(txt):
    txt = txt.lower()
    return any(kw in txt for kw in NEUTRAL_KEYWORDS)

df = pd.read_csv(DATA_PATH)
df.dropna(subset=['Review','Sentiment'], inplace=True)
df['clean'] = df['Review'].apply(clean_text)

# Vectorizer & model setup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.naive_bayes import MultinomialNB

def augment_text(text, times=2, p=0.7):
    import random
    words = text.split()
    aug_texts = []
    for _ in range(times):
        new = [
            (random.choice(words) if (w in words and random.random()<p) else w)
            for w in words
        ]
        aug_texts.append(" ".join(new))
    return aug_texts

augmented = []
for _, row in df.iterrows():
    for aug in augment_text(row['clean']):
        augmented.append({'clean': aug, 'Sentiment': row['Sentiment']})
df_aug = pd.DataFrame(augmented)

pos = df_aug[df_aug['Sentiment']=='positive']
neu = df_aug[df_aug['Sentiment']=='neutral']
neg = df_aug[df_aug['Sentiment']=='negative']

neu_up = resample(neu, replace=True, n_samples=len(pos), random_state=42)
neg_up = resample(neg, replace=True, n_samples=len(pos), random_state=42)

data_bal = pd.concat([pos, neu_up, neg_up], ignore_index=True)
X_texts = data_bal['clean'].tolist()
y = data_bal['Sentiment'].tolist()

X_train, _, y_train, _ = train_test_split(X_texts, y, test_size=0.2, stratify=y)

vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=5, max_df=0.8)
X_train_tfidf = vectorizer.fit_transform(X_train)

clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# 2. Flask web app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    text = ""
    if request.method == "POST":
        text = request.form["input_text"].strip()
        clean_inp = clean_text(text)

        if detect_negative_fallback(clean_inp):
            prediction = "negative (fallback)"
        elif detect_neutral_fallback(clean_inp):
            prediction = "neutral (fallback)"
        else:
            vec = vectorizer.transform([clean_inp])
            prediction = clf.predict(vec)[0]

    return render_template("index.html", prediction=prediction, input_text=text)

if __name__ == "__main__":
    app.run(debug=True)
