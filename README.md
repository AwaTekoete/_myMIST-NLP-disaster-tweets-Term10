# Disaster Tweets NLP-Klassifikation

Binäre Textklassifikation: Unterscheidung zwischen Tweets über reale Katastrophen
und Tweets, die nur metaphorisch/umgangssprachlich Katastrophen-Vokabular nutzen.

**Kurs:** MIST Term 10 | **Wettbewerb:** [Kaggle – NLP Getting Started](https://www.kaggle.com/c/nlp-getting-started)

---

## Projektziel

Lernprinzip: **Lernen → Verstehen → Dokumentieren.** State-of-the-Art (F1 ≈ 0.83–0.85,
realistischer BERT-Klasse-Referenzwert, da Kaggle-Leaderboard-Top-Scores teils durch
Test-Set-Leakage verzerrt sind) dient als Referenzpunkt, nicht als reines Ziel.
Fokus liegt auf dem Verständnis der Grenzen des gewählten Ansatzes (Datensatz, Modell,
Ressourcen) und was nötig wäre, um den Abstand zu schließen.

---

## Projektstruktur
data/

├── raw/            # Originaldaten (train.csv) – nicht versioniert, siehe unten

└── processed/       # Transformierte Daten
notebooks/            # 01_eda … 06_error_analysis (Reihenfolge = Bearbeitungsphasen)

src/                  # Wiederverwendbare Module (Preprocessing, Features, Modelle, Viz)

models/               # Lokal gespeicherte Baseline-Modelle (.pkl/.joblib) – nicht versioniert

reports/

├── figures/          # Plots (Store44-Farbschema), Name = <notebook-nr>_<slug>.png

├── tables/           # CSV-Exports: Metriken, Modellvergleiche

└── errors/           # Fehlklassifizierte Beispiele als CSV

submission/           # Finales Colab-Abgabe-Notebook
---

## Setup

```powershell
# Repository klonen
git clone https://github.com/AwaTekoete/_myMIST-NLP-disaster-tweets-Term10.git
cd _myMIST-NLP-disaster-tweets-Term10

# Conda-Environment erstellen & aktivieren
conda create -n disaster-tweets python=3.11 -y
conda activate disaster-tweets

# Abhängigkeiten installieren
pip install -r requirements.txt

# Notebook-Diff-Filter aktivieren (lokal pro Klon erforderlich)
nbstripout --install
```

**Datensatz:** `train.csv` ist nicht im Repository enthalten (siehe `.gitignore`).
Download über die [Kaggle-Wettbewerbsseite](https://www.kaggle.com/c/nlp-getting-started/data)
und in `data/raw/` ablegen.

---

## Phasen

| # | Notebook | Inhalt |
|---|----------|--------|
| 0 | – | Setup (dieses Dokument) |
| 1 | `01_eda.ipynb` | Datenqualität & explorative Analyse |
| 2 | `02_preprocessing.ipynb` | Textbereinigung & Normalisierung |
| 3 | `03_vectorization.ipynb` | BoW / TF-IDF / n-Gramme / LSA |
| 4 | `04_baseline.ipynb` | Logistic Regression, Naive Bayes, Linear SVC + K-Fold CV |
| 5 | `05_tuning.ipynb` | Hyperparameter-Tuning (Grid Search) |
| 6 | `06_error_analysis.ipynb` | Fehleranalyse & SOTA-Abstand |
| 7 | `submission/` | Colab-Konsolidierung (Abgabe-Notebook) |
| 8 | Bonus-Branch | Dichte/gelernte Embeddings, ggf. Transformer-Finetuning |

---

## Ergebnisse

*Wird nach Abschluss der Modellierungsphasen ergänzt (Phase 4–6).*

---

## Präsentation

Google-Slides-Link: *folgt nach Projektabschluss*
Stil-Referenz: `Praesentationsstil_Store44.md`

---

*Erik Gerst | MIST Term 10 | 2026*