import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle

# 1. Load dataset
df = pd.read_csv("cyberbullying_tweets.csv")

# 2. Clean dataset: Convert labels into 0 and 1
df['label'] = df['cyberbullying_type'].apply(lambda x: 0 if x == 'not_cyberbullying' else 1)

# 3. Input and output
X = df['tweet_text']
y = df['label']

# 4. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 6. Model Training
model = LinearSVC()
model.fit(X_train, y_train)

# 7. Save model and vectorizer
pickle.dump(vectorizer, open("tfidfvectoizer.pkl", "wb"))
pickle.dump(model, open("LinearSVCTuned.pkl", "wb"))

print("✅ Training complete. Model and vectorizer saved.")
