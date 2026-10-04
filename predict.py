import joblib
import csv
import os
from datetime import datetime
from explain import explain_prediction

vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")

THRESHOLD = 0.60
LOG_FILE = "flagged_documents.csv"

def log_flagged_document(text, category, confidence):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "text", "predicted_category", "confidence"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            text, category, f"{confidence:.2f}",
        ])

def classify_document(text):
    X = vectorizer.transform([text])
    probabilities = model.predict_proba(X)[0]
    best = probabilities.argmax()
    category = model.classes_[best]
    confidence = probabilities[best]

    _, top_words, matched_words = explain_prediction(text)

    if confidence < THRESHOLD:
        status = "NEEDS MANUAL REVIEW"
        log_flagged_document(text, category, confidence)
    else:
        status = "auto-classified"

    return category, confidence, status, matched_words