import joblib
import numpy as np

vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")

feature_names = vectorizer.get_feature_names_out()

def explain_prediction(text, top_n=8):
    X = vectorizer.transform([text])
    predicted_category = model.predict(X)[0]

    class_index = list(model.classes_).index(predicted_category)
    class_weights = model.coef_[class_index]

    top_indices = np.argsort(class_weights)[::-1][:top_n]
    top_words_for_class = [feature_names[i] for i in top_indices]

    words_in_this_document = text.lower().split()
    matched_words = [word for word in top_words_for_class if word in words_in_this_document]

    return predicted_category, top_words_for_class, matched_words

if __name__ == "__main__":
    sample = "The Receiving Party shall keep all Confidential Information secret and shall not disclose it to any third party without prior written consent."
    category, top_words, matched = explain_prediction(sample)
    print("Predicted category:", category)
    print("Top words the model associates with this category:", top_words)
    print("Which of those words actually appear in this document:", matched)