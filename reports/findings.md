# Findings & Entscheidungen

Laufende Dokumentation von Erkenntnissen und darauf basierenden Entscheidungen
während des Projekts. Reihenfolge = chronologisch.

---

## Phase 0 — Datenquelle: Kaggle-Vollversion vs. Schul-Version

**Ausgangslage:** Zwei Versionen des Trainingsdatensatzes verfügbar:
- Schul-Version: 776 KB, 2 Spalten (`text`, `target`)
- Kaggle-Vollversion: 964 KB, 5 Spalten (`id`, `keyword`, `location`, `text`, `target`)

**Untersuchung:** Direkter Vergleich der Spalten `text` und `target` zwischen
beiden Versionen (sortierter Abgleich, 7.613 Zeilen je Version).

**Ergebnis:** `text` und `target` sind in beiden Versionen zu 100 % identisch.
Die Schul-Version ist ein reines Spalten-Subset derselben Daten, keine inhaltlich
abweichende Version.

**Zusätzlicher Befund (Kaggle-Vollversion):** Fehlende Werte in den zusätzlichen
Spalten:

| Spalte | train.csv | test.csv | Anteil |
|---|---|---|---|
| `keyword` | 61 / 7.613 | 26 / 3.263 | ~0,8 % |
| `location` | 2.533 / 7.613 | 1.105 / 3.263 | ~33 % |

Fehlquoten zwischen Train- und Test-Set nahezu identisch — Hinweis auf
konsistenten Erhebungsprozess, kein Verteilungsproblem zwischen den Sets.
`location` ist zusätzlich ein Twitter-Freitextfeld; auch vorhandene Werte
sind potenziell unzuverlässig (z. B. nicht-geografische Angaben) – inhaltliche
Prüfung folgt in Phase 1.

**Entscheidung:** Kaggle-Vollversion (5 Spalten) wird als primäre Datenquelle
verwendet. `train_schulversion.csv` bleibt als Fallback-Referenz erhalten.

**Begründung:**
- Datenidentität bei `text`/`target` bedeutet keinen Verlust gegenüber der
  Schul-Version, nur zusätzliche Optionalität
- `keyword`/`location` ermöglichen zusätzliche, datengetriebene
  Qualitätsanalyse (Mutual Information, Missingness-Pattern) statt einer
  vorab getroffenen Annahme über deren Nutzlosigkeit
- `id`-Spalte erforderlich für Kaggle-kompatibles Submission-Format