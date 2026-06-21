# =============================================================================
# sanity_check.py – Phase 0: Verifikation, dass alle Kaggle-Dateien korrekt
# lesbar sind und der erwarteten Struktur entsprechen. Kein EDA – folgt in Phase 1.
# =============================================================================

import pandas as pd

# --- train.csv ---------------------------------------------------------------
train = pd.read_csv("data/raw/train.csv")
print("=== train.csv ===")
print("Shape:", train.shape)
print("Spalten:", list(train.columns))
print("Fehlende Werte pro Spalte:\n", train.isnull().sum())
print("Zielvariable – Klassenverteilung:\n", train["target"].value_counts())

# --- test.csv ------------------------------------------------------------
test = pd.read_csv("data/raw/test.csv")
print("\n=== test.csv ===")
print("Shape:", test.shape)
print("Spalten:", list(test.columns))
print("Fehlende Werte pro Spalte:\n", test.isnull().sum())

# --- sample_submission.csv ----------------------------------------------
sample_sub = pd.read_csv("data/raw/sample_submission.csv")
print("\n=== sample_submission.csv ===")
print("Shape:", sample_sub.shape)
print("Spalten:", list(sample_sub.columns))
print(sample_sub.head())

# --- Konsistenz-Check: passt test.csv zu sample_submission.csv? ----------
assert len(test) == len(sample_sub), "Zeilenanzahl test.csv != sample_submission.csv!"
assert set(test["id"]) == set(sample_sub["id"]), "IDs stimmen nicht überein!"
print("\nKonsistenz-Check bestanden: test.csv und sample_submission.csv passen zusammen.")