import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("documents_clean.csv")

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(df["text"])

model = LogisticRegression(max_iter=1000)
model.fit(X, df["category"])

joblib.dump(vectorizer, "vectorizer.joblib")
joblib.dump(model, "model.joblib")

print("Saved final model and vectorizer")