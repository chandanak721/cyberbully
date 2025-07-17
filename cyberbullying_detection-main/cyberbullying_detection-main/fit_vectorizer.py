import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("dataset.csv")  # Use full path if needed
texts = df['headline'].astype(str)

tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(texts)

# ✅ Correct: Save only the object, not a dictionary
with open("tfidfvectoizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Vectorizer saved")
