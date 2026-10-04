from datasets import load_dataset
import pandas as pd

dataset = load_dataset("coastalcph/lex_glue", "ledgar", split="train")

texts = dataset["text"]
label_ids = dataset["label"]
label_names = dataset.features["label"].names

labels = [label_names[i] for i in label_ids]

df = pd.DataFrame({"text": texts, "category": labels})
df.to_csv("documents_raw.csv", index=False)

print("Saved", len(df), "documents")
print(df["category"].value_counts().head(20))