# findings.md — Disaster Tweets NLP-Klassifikation (MIST Term 10)

## Projekt-Uebersicht

**Datensatz:** Kaggle "NLP Getting Started" (Disaster Tweets)
**Aufgabe:** Binaere Klassifikation (Disaster=1 / Kein Disaster=0)
**Champion:** LogReg_balanced, TF-IDF Config C (text_stemmed, min_df=2)
**Finale Performance:** F1=0.7775 | ROC-AUC=0.8487 | MCC=0.5551

---

## Phase 01 — EDA & Datenqualitaet

- Rohdaten: 7.613 Zeilen, 5 Spalten
- Klassenbalance: 57% / 43% (moderat unbalanciert)
- 18 Konflikt-Gruppen auf Roh-Text-Ebene (55 Zeilen) entfernt
- 92 harmlose Duplikate entfernt
- Nach Notebook 01: 7.434 Zeilen (-179)
- URL-Praesenz: 66,4% Disaster vs. 41,4% kein Disaster → has_url Feature
- keyword: Kardinalitaets-Ratio 0.029 → Feature-Kandidat
- location: Ratio 0.658, 84,3% Einzelvorkommen → nicht verwendet
- Dummy-Baseline: F1=0.468 (uniform random)

## Phase 02 — Preprocessing

- Zusaetzliche Bereinigung auf text_clean-Basis: 633 Zeilen entfernt
  (196 Konflikt-Zeilen + 437 harmlose Duplikate nach Normalisierung)
- Gesamt entfernt: 812 Zeilen (-10,7%)
- Finale Datenmenge: 6.801 Zeilen
- Vier Text-Varianten: text_clean, text_no_stopwords,
  text_stemmed, text_lemmatized
- Mojibake-Korrektur: vollstaendiger Bereich [\x80-\xff]
- Zahlen → NUM-Platzhalter (bestaetigt als starkes Feature in Phase 5)

## Phase 03 — Vektorisierung

- Config C (TF-IDF, text_stemmed, min_df=2) konsistent beste Variante
- Config D (LSA, 100 Komponenten): nur 13,8% erklaerte Varianz
  → strukturell ungeeignet fuer kurze Tweets
- Silhouette-Score: 0.005 (keine globale Trennung im Vektorraum)
- t-SNE zeigt interpretierbaren URL-Gradienten
- keyword OHE: 88% Redundanz mit Fliesstext → nicht als Feature

## Phase 04 — Baseline-Modelle

- 20 Kombinationen (4 Configs × 3 Modelle × class_weight)
- Alle F1-Unterschiede statistisch nicht signifikant (< 0.54× Std)
- Fehler-Profile: NB konservativ (P1=0.825, R1=0.636),
  LogReg ausgewogen (P1=0.741, R1=0.728)
- LinearSVC: groesstes Overfitting (Gap=0.181)
- keyword als Feature: -0.003 F1, Std verdoppelt → verworfen

## Phase 05 — Hyperparameter-Tuning

- LogReg: Default-Parameter optimal (Delta +0.0007)
- MultinomialNB: alpha=0.5 optimal (Delta -0.0013 durch Nested CV)
- LinearSVC: groesster Gewinn +0.0126, Gap 0.181 → 0.106 (-42%)
- Champion: LogReg_balanced (ROC-AUC 0.8487, ausgewogenes Profil,
  beste Erklaerbarkeit, statistisch gleichwertig mit anderen)
- Threshold: t=0.50 (Standard) / t=0.54 (F1/MCC-optimiert)
- Feature Importance: "fire" staerkstes Feature (Perm. Imp.=0.010)
  — viele schwache Signale, kein Killer-Feature (Ratio Top1/Top20: 7×)

## Phase 06 — Fehleranalyse & SOTA-Abstand

- FNR=27,2% | FPR=17,4%
- 91,1% der FN haben kein klassisches Disaster-Vokabular
  → strukturelle Grenze von TF-IDF, nicht durch Tuning behebbar
- 819 Tweets (12%) im Threshold-Bereich 0.45-0.55: nur 54,9% korrekt
- Learning Curves: CV-Kurve noch steigend, kein Plateau
  → mehr Daten wuerden helfen, aber TF-IDF-Obergrenze ~0.82-0.83
- Champion liegt nur ~0.023 unter geschaetztem Human-Level (~0.800)

### F1-Limitatoren (Engineering-Manager-Perspektive)

| Rang | Limitor | Erwarteter Gewinn | Aufwand |
|---|---|---|---|
| 1 | Kontext-Blindheit (TF-IDF) | +0.05-0.07 | Hoch (Transformer) |
| 2 | Datenmenge | +0.01-0.05 | Sehr hoch |
| 3 | Label-Rauschen | +0.005-0.015 | Mittel |
| 4 | Stemming-Verluste | +0.005-0.010 | Mittel |
| 5 | keyword-Kodierung | +0.002-0.008 | Niedrig |
| 6 | Hyperparameter | ~0.000-0.003 | Minimal |

### SOTA-Referenz

| Modell | F1 |
|---|---|
| Unser Champion | 0.777 |
| Human-Level (geschaetzt) | ~0.800 |
| BERT-base (realistisch) | 0.840 |
| BERTweet (realistisch) | 0.850 |
| Kaggle Top* (*Leakage) | 0.890 |

---

## Offene Punkte / Bonus-Kandidaten (Phase 8)

- DistilBERT, BERT-base, RoBERTa-base, BERTweet fine-tuned
  (GPU: Kaggle 2×T4 oder Colab)
- Few-Shot mit GPT/Claude (kein Training, Vergleichspunkt)
- Datensatz-Variante als Hyperparameter: bereinigt (6.801)
  vs. original (7.613)
- Label Smoothing / Mehrheits-Voting
- Disaster-Rate-Kodierung von keyword (statt OHE)
- 3D-t-SNE auf Word2Vec-Embeddings