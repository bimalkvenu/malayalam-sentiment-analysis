Malayalam Sentiment Analysis using NLP 🗣️🧠  


A state-of-the-art sentiment analysis toolkit for Malayalam text, leveraging NLP and machine learning to classify emotions in social media content, reviews, and more.

---

🌟 Features

- Pre-trained models for Malayalam sentiment (Naive Bayes + TF-IDF)
- Custom dataset with annotated Malayalam tweets/comments
- Hybrid approach combining ML + rule-based fallback methods
- Easy-to-use API for real-time predictions (Flask-based)
- Evaluation metrics for model performance (F1-score, accuracy)

---


🚀 Quick Start
Predict sentiment on sample text:

from predict import analyze_sentiment
text = "ഈ പ്രോജക്ട് വളരെ നല്ലതാണ്!"  # "This project is great!"
result = analyze_sentiment(text)
print(result)  # Output: {'text': '...', 'sentiment': 'positive', 'confidence': 0.92}

Train your own model:
from train import train_model

train_model(data_path="data/malayalam_reviews.csv")

📂 Project Structure
├── data/                    # Annotated datasets
│   ├── malayalam_tweets.csv
│   └── preprocessed/
├── models/                  # Saved models
│   └── svm.pkl
├── notebooks/               # Jupyter notebooks
│   ├── EDA.ipynb
│   └── Model_Training.ipynb
├── src/
│   ├── preprocess.py        # Text cleaning
│   ├── train.py             # Model training
│   └── predict.py           # Inference
├── app.py                   # Flask API
└── requirements.txt



📊 Results
Model	Accuracy	F1-Score
Naive Bayes (TF-IDF)	82.4%	0.81
Lexicon-based	76.5%	0.72

🛠️ Tech Stack
NLP Libraries: NLTK, spaCy, HuggingFace (optional)

ML Frameworks: Scikit-learn

Embeddings: TF-IDF, N-grams

Deployment: Flask (local server or demo)
