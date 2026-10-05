# *AI Document Risk & Compliance Classifier*

An AI system that reads a legal/contract document, classifies it into a risk or compliance category, explains *why* it made that decision, and flags documents it isn't confident about for manual human review.

## What it does 📠

1. Takes a legal document (uploaded as a file or pasted as text) as input.
2. Classifies it into one of 6 categories: **Confidentiality, Indemnifications, Compliance With Laws, Taxes, Litigations, Insurances**.
3. Shows a confidence score (how sure the model is about its answer).
4. Explains the decision by listing the key words/phrases in the document that led the model to that classification.
5. If the model's confidence is below 60%, the document is automatically flagged as "NEEDS MANUAL REVIEW" and logged to a CSV audit file instead of being silently trusted.
6. A dashboard shows every flagged document so a human reviewer can see what the AI wasn't sure about.

## Why this matters 🤔

In real compliance and legal workflows, blindly trusting an AI's classification is risky — a wrong label on a legal document could have real consequences. This project doesn't just classify; it's built to be honest about its own uncertainty and to explain its reasoning, so a human stays in the loop exactly when they're needed.

## Project structure 🧬

- | File | What it does |
- |---|---|
- | `download_data.py` | Downloads the LEDGAR legal-document dataset from Hugging Face |
- | `clean_data.py` | Filters the dataset down to 6 target categories and cleans the text |
- | `train_model.py` | Trains a TF-IDF + Logistic Regression classifier and evaluates it |
- | `save_model.py` | Retrains on the full cleaned dataset and saves the final model files |
- | `explain.py` | Explains a prediction by showing which words influenced it |
- | `predict.py` | Combines prediction + confidence check + explanation + logging into one function |
- | `app.py` | Streamlit dashboard — upload a document, see its classification, explanation, and the flagged-documents log |
- | `documents_clean.csv` | The cleaned dataset used for training |
- | `vectorizer.joblib` / `model.joblib` | The saved, trained model files |
- | `flagged_documents.csv` | Auto-generated log of low-confidence documents that needed manual review |

## How to run it 🏃🏼‍♀️

1. Install the required libraries:

pip install datasets pandas scikit-learn joblib streamlit

2. Download and prepare the data:

python download_data.py
python clean_data.py

3. Train and save the model:

python train_model.py
python save_model.py

4. Launch the dashboard:

streamlit run app.py

5. Upload a document or paste text, and see the classification, confidence, and explanation.

## Approach ☝🏼

- **Dataset**: LEDGAR (a real-world dataset of legal contract clauses), filtered down to 6 common risk/compliance categories.
- **Model**: Text is converted into numbers using **TF-IDF** (which measures how important each word is to a document), then classified using **Logistic Regression**.
- **Explainability**: Since Logistic Regression is a linear model, each word has a learned "weight" showing how strongly it pushes the model toward a particular category. For any prediction, the app finds the words in the document that have the highest weight for the predicted category — this is how it explains its own reasoning in plain terms, rather than being a black box.
- **Confidence & human-in-the-loop**: The model outputs a probability for its prediction. If that probability is below 60%, the document is flagged instead of being confidently (and possibly wrongly) labeled. Flagged documents are logged to `flagged_documents.csv` with their text, predicted category, and confidence score, and shown on a live dashboard table.

## Evaluation results 

The model was evaluated on a held-out 20% test split (stratified, so each category is fairly represented):

- **Overall accuracy: 98%**
- High precision and recall across all 6 categories (see the full `classification_report` output in `train_model.py`'s run)

## A real technical challenge we ran into ⛓️

While downloading the dataset, we hit this error:

HfUriError: Invalid HF URI... Repository id must be 'namespace/name', got 'lex_glue'

This happened because Hugging Face changed its rules — dataset names now need an "owner/name" format instead of just a plain name. We fixed it by changing the dataset identifier from `"lex_glue"` to `"coastalcph/lex_glue"` (adding the correct namespace), after which the download worked correctly.

## Honest limitations 😥

- The model is trained only on the 6 categories chosen for this project, not the full range of legal clause types that exist in the real world.
- It works on English-language text only.
- TF-IDF + Logistic Regression is a simpler, more interpretable model choice rather than a large transformer-based LLM — this was a deliberate trade-off to keep the explainability straightforward and the model lightweight, at some cost to handling more nuanced or ambiguous phrasing.
- The 60% confidence threshold for flagging was chosen manually, not tuned against a formal cost/benefit analysis of false positives vs. false negatives.

## Possible next steps 🪜

- Expand to more categories of legal/compliance documents.
- Try a transformer-based model (like BERT) for potentially higher accuracy on harder, more ambiguous clauses.
- Let reviewers submit corrections for flagged documents, and use that feedback to retrain and improve the model over time.

## 🤣 *What I Learnt and the Challenges that I faced while making this project* 🤣
Firstly, I installed pandas scikit-learn , matplotlib joblib and created download_data.py with the help of my AI assistant. We extracted the data from the dataset. There was code missing in that file ,so we wrote a one line code to loaded dataset from ledger. We got 60000 downloaded documents ,we divided that into 6 categories and wrote 'clean_data' to clean that raw data that we extracted. After cleaning the data by categorizing them , we added TF-IDF to extract unique word from the cleaned file. By using the unique words we wrote 'train_model' to train the model using TF-IDF. Because the dataset was perfect dataset for training a model I didn't got any error so far. I saved the model which has been trained by the unique word. In the next step we wrote 'explain.py' to show "why & which word influenced "  whenever the AI model predicts or decision. I with the help of my AI assistant developed 'predict.py' code to show how confident is the AI model sure about it's prediction/decision. At the end we developed 'app.py' to open the model in the web page and run the output. This project was easy because I worked on AI-support-ticket-triage and was similar to that project so, that was the reason why I had minimal error while execution of this project

# 🙇🏼‍♀️ Thankyou for Reading 🙇🏼‍♀️
