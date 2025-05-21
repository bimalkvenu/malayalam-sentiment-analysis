import os
import pandas as pd
import numpy as np
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import resample

# 1. CONFIG: absolute path to your CSV
DATA_PATH = r"D:/LPU/4th sem/CAP457_ BIGDATA (LABORATORY)/Project/MABSA4000.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

# 2. KEYWORD LISTS
NEGATIVE_KEYWORDS = [
    "ദു:ഖകരമായിരുന്നു", "വഷളായിരുന്ന", "മോശം", "ചീത്ത",
    "നിരാശ", "അപര്യാപ്തം", "ഇല്ല", "ലഭിച്ചില്ല", "കണ്ടില്ല", "ഉള്ളില്ല"
]
NEUTRAL_KEYWORDS = [
    "വായിച്ചു", "എഴുന്നേറ്റു", "പോയി", "വന്നു", "കണ്ടു",
    "ക്ലാസിൽ", "ഇന്ന്", "കഴിഞ്ഞു"
]

# 3. TEXT CLEANING
def clean_text(text):
    # Keep only Malayalam unicode range, digits, spaces
    txt = re.sub(r"[^\u0D00-\u0D7F0-9\s]", "", str(text))
    return re.sub(r"\s+", " ", txt).strip()

# 4. NEGATIVE & NEUTRAL FALLBACK
def detect_negative_fallback(txt):
    txt = txt.lower()
    # catch any word ending with 'ഇല്ല'
    if re.search(r"\b\w*ഇല്ല\b", txt):
        return True
    return any(kw in txt for kw in NEGATIVE_KEYWORDS)

def detect_neutral_fallback(txt):
    txt = txt.lower()
    return any(kw in txt for kw in NEUTRAL_KEYWORDS)

# 5. DATA LOADING & PREP
df = pd.read_csv(DATA_PATH, encoding='utf-8')
df.dropna(subset=['Review','Sentiment'], inplace=True)
df['clean'] = df['Review'].apply(clean_text)

# 6. SIMPLE AUGMENTATION (optional—you can expand this)
def augment_text(text, times=2, p=0.7):
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

# 7. BALANCE DATA: upsample neutral & negative to match positive
pos = df_aug[df_aug['Sentiment']=='positive']
neu = df_aug[df_aug['Sentiment']=='neutral']
neg = df_aug[df_aug['Sentiment']=='negative']

neu_up = resample(neu, replace=True, n_samples=len(pos), random_state=42)
neg_up = resample(neg, replace=True, n_samples=len(pos), random_state=42)

data_bal = pd.concat([pos, neu_up, neg_up], ignore_index=True)
X_texts = data_bal['clean'].tolist()
y = data_bal['Sentiment'].tolist()

# 8. TRAIN – TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_texts, y, test_size=0.2, random_state=42, stratify=y
)

# 9. VECTORIZE (unigrams + bigrams + trigrams)
vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=5, max_df=0.8)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# 10. MODEL TRAINING
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# 11. EVALUATION
y_pred = clf.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 12. INTERACTIVE LOOP
print("\nType 'exit' to quit.")
while True:
    inp = input("\nEnter Malayalam text: ").strip()
    if inp.lower() == 'exit':
        break
    if not inp:
        print("Please enter something.")
        continue

    clean_inp = clean_text(inp)
    if detect_negative_fallback(clean_inp):
        pred = 'negative'
    elif detect_neutral_fallback(clean_inp):
        pred = 'neutral'
    else:
        vec = vectorizer.transform([clean_inp])
        pred = clf.predict(vec)[0]

    print(f"→ {pred}")
