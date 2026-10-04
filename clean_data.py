import pandas as pd

df = pd.read_csv("documents_raw.csv")

categories_to_keep = [
    "Confidentiality",
    "Indemnifications",
    "Compliance With Laws",
    "Taxes",
    "Litigations",
    "Insurances",
]

df = df[df["category"].isin(categories_to_keep)]

df["text"] = df["text"].str.strip().str.replace(r"\s+", " ", regex=True)

df.to_csv("documents_clean.csv", index=False)

print("Kept", len(df), "documents across", df["category"].nunique(), "categories")
print(df["category"].value_counts())