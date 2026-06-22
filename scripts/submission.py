# submission.py — Champion auf test.csv anwenden
# Ausfuehren in conda env: python scripts/submission.py

import sys
sys.path.append("src")

import pandas as pd
import joblib
from preprocessing import clean_text, tokenize, remove_stopwords, apply_stemming

# Champion laden
model = joblib.load("models/champion_logreg.joblib")
vectorizer = joblib.load("models/champion_vectorizer.joblib")

# Testdaten laden
test = pd.read_csv("data/raw/test.csv", keep_default_na=False)
test[["keyword", "location"]] = test[["keyword", "location"]].replace("", float("nan"))

print(f"Test-Shape: {test.shape}")
print(f"Fehlende keyword: {test['keyword'].isnull().sum()}")
print(f"Fehlende text: {test['text'].isnull().sum()}")

# Preprocessing (identisch zu Training)
def preprocess(text):
    return " ".join(apply_stemming(
        remove_stopwords(tokenize(clean_text(str(text))))))

test["text_stemmed"] = test["text"].apply(preprocess)

# Vektorisierung + Vorhersage
X_test = vectorizer.transform(test["text_stemmed"])
test["target"] = model.predict(X_test)
test["prob"] = model.predict_proba(X_test)[:, 1]

# Submission-Format: id + target
submission = test[["id", "target"]].copy()
submission.to_csv("submission/submission.csv", index=False)

print(f"\nSubmission gespeichert: submission/submission.csv")
print(f"Shape: {submission.shape}")
print(f"\nKlassenverteilung Vorhersagen:")
print(submission["target"].value_counts())
print(f"Disaster-Rate: {submission['target'].mean()*100:.1f}%")

# Stichprobe
print("\nStichprobe (5 Zeilen):")
print(test[["id", "text", "prob", "target"]].head(5).to_string())