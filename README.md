# Disaster Tweets NLP-Klassifikation
**MIST Term 10 — NLP Klassifikationsprojekt**

Binaere Klassifikation von Tweets: Disaster (1) vs. Kein Disaster (0).
Datensatz: Kaggle "NLP Getting Started" Competition.

---

## Ergebnisse (Standardprojekt)

| Modell | F1 Macro | ROC-AUC | MCC |
|---|---|---|---|
| Dummy Baseline (uniform) | 0.468 | 0.500 | 0.000 |
| TF-IDF + LogReg (Champion) | **0.7775** | **0.8487** | **0.5551** |
| TF-IDF + LinearSVC (getunt) | 0.7807 | 0.8424 | 0.5682 |
| Human-Level (geschaetzt) | ~0.800 | — | — |
| BERTweet SOTA (realistisch) | 0.850 | 0.930 | — |

**Champion:** LogReg_balanced, TF-IDF Config C (text_stemmed, min_df=2)
**Evaluierung:** 5-Fold Stratified CV, Out-of-Fold, Bootstrap-KI 95%

---

## Projektstruktur
_myMIST-NLP-disaster-tweets-Term10/

├── data/

│   ├── raw/                    # Originaldaten (unveraendert)

│   │   ├── train.csv

│   │   ├── test.csv

│   │   └── sample_submission.csv

│   └── processed/

│       ├── train_clean.csv     # Nach Phase 01 (7.434 Zeilen)

│       └── train_preprocessed.csv  # Nach Phase 02 (6.801 Zeilen)

├── notebooks/

│   ├── 01_eda.ipynb            # EDA & Datenqualitaet

│   ├── 02_preprocessing.ipynb  # Text-Bereinigung & Normalisierung

│   ├── 03_vectorization.ipynb  # BoW, TF-IDF, LSA, Ablationsplan

│   ├── 04_baseline.ipynb       # Baseline-Modelle (20 Kombinationen)

│   ├── 05_tuning.ipynb         # Grid Search, Nested CV, Threshold

│   └── 06_error_analysis.ipynb # Fehleranalyse & SOTA-Abstand

├── src/

│   ├── preprocessing.py        # Wiederverwendbare Preprocessing-Module

│   └── viz_config.py           # Store44-Farbschema (zentralisiert)

├── models/

│   ├── champion_logreg.joblib  # Finales Champion-Modell

│   ├── champion_vectorizer.joblib  # Finaler Vectorizer

│   ├── champion_metadata.json  # Parameter & CV-Metriken

│   └── registry.md             # Modell-Registry

├── reports/

│   ├── figures/                # Alle Visualisierungen (Store44-Stil)

│   ├── tables/                 # Alle CSV-Auswertungen

│   ├── errors/                 # Fehler-Exports (Phase 06)

│   └── findings.md             # Vollstaendige Befund-Dokumentation

├── submission/

│   └── submission.csv          # Kaggle-Submission

├── scripts/

│   └── sanity_check.py         # Datensatz-Validierung

└── requirements.txt

---

## Reproduktion

**Voraussetzungen:**
```bash
conda create -n disaster-tweets python=3.11
conda activate disaster-tweets
pip install -r requirements.txt
```

**Reihenfolge der Notebooks:**
01_eda → 02_preprocessing → 03_vectorization →
04_baseline → 05_tuning → 06_error_analysis

**Champion-Modell direkt verwenden:**
```python
import joblib
from src.preprocessing import clean_text, tokenize, remove_stopwords, apply_stemming

model = joblib.load("models/champion_logreg.joblib")
vectorizer = joblib.load("models/champion_vectorizer.joblib")

tweet = "forest fire near La Ronge Sask Canada"
cleaned = " ".join(apply_stemming(
    remove_stopwords(tokenize(clean_text(tweet)))))
X = vectorizer.transform([cleaned])
prob = model.predict_proba(X)[0][1]
pred = "DISASTER" if prob >= 0.5 else "KEIN DISASTER"
print(f"P(Disaster)={prob:.3f} → {pred}")
```

---

## Methodik-Uebersicht

| Phase | Inhalt | Key-Entscheidung |
|---|---|---|
| 01 EDA | Struktur, Duplikate, Klassenbalance, Chi² | Konflikte auf Roh-Text entfernt |
| 02 Preprocessing | Cleaning, Tokenisierung, Stemming/Lemma | Bereinigung auch auf text_clean-Basis |
| 03 Vektorisierung | BoW, TF-IDF, LSA, Ablationsplan | Config C (Stemmed) als Hauptkandidat |
| 04 Baseline | 20 Kombinationen, Bootstrap-KI, OvF-Gap | LogReg_balanced als Champion-Kandidat |
| 05 Tuning | Grid Search, Nested CV, Threshold, FI | Default-Parameter optimal fuer LogReg |
| 06 Fehleranalyse | Fehlertypen, Konfidenz, LC, SOTA | 91,1% FN ohne Disaster-Vokabular |

---

## SOTA-Abstand & Grenzen

Unser Champion (F1=0.777) liegt ~2,3 Punkte unter geschaetztem
Human-Level (~0.800) und ~7,3 Punkte unter BERTweet-SOTA (0.850).

**Hauptursache:** TF-IDF ist wortbasiert — 91,1% der verpassten
Disasters enthalten kein klassisches Disaster-Wort. Das Modell
kann Kontext, Metaphern und Ironie nicht erkennen.

**Naechster Schritt (Bonus-Branch):** Transformer-Modelle
(DistilBERT, BERTweet) auf Kaggle GPU (2×T4).

---

## Technische Details

| Parameter | Wert |
|---|---|
| Python | 3.11 |
| Conda Environment | disaster-tweets |
| Hauptbibliotheken | scikit-learn, nltk, pandas, numpy, matplotlib |
| Notebook-Hygiene | nbstripout (saubere Diffs) |
| Visualisierungen | Store44-Farbschema (#1C1C1C, #F5A623, #6CB4E4) |
| Versionierung | Conventional Commits |

---

*MIST Term 10 | GitHub: AwaTekoete |
Solo-Projekt | Standardprojekt abgeschlossen*
