import streamlit as st
import pandas as pd
import os
from predict import classify_document, LOG_FILE

st.title("AI Document Risk & Compliance Classifier")
st.write("Upload a document or paste its text below to classify it by compliance category, with an explanation and a manual-review flag for low-confidence cases.")

uploaded_file = st.file_uploader("Upload a document (.txt)", type=["txt"])

if uploaded_file is not None:
    document_text = uploaded_file.read().decode("utf-8")
    st.text_area("Document text", value=document_text, height=250)
else:
    document_text = st.text_area("Document text", height=250, placeholder="The Receiving Party shall keep all Confidential Information secret...")

if st.button("Classify document"):
    if document_text.strip() == "":
        st.warning("Please paste or upload a document first.")
    else:
        category, confidence, status, matched_words = classify_document(document_text)

        st.subheader(f"Predicted category: {category}")
        st.write(f"**Confidence:** {confidence*100:.1f}%")

        if status == "auto-classified":
            st.success("Auto-classified (high confidence)")
        else:
            st.warning("NEEDS MANUAL REVIEW (low confidence)")

        if matched_words:
            st.write("**Why:** this document contains words strongly associated with this category:")
            st.write(", ".join(matched_words))
        else:
            st.write("**Why:** no strongly distinctive words were found; classification is less certain.")

st.divider()
st.subheader("Flagged documents (needing manual review)")

if os.path.exists(LOG_FILE):
    flagged_df = pd.read_csv(LOG_FILE)
    st.dataframe(flagged_df)
else:
    st.info("No documents have been flagged for manual review yet.")