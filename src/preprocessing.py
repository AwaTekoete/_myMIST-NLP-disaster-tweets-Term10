# =============================================================================
# preprocessing.py - Text-Cleaning, Tokenisierung, Stopwords, Stemming/Lemma
# =============================================================================
# Zentrales Modul fuer alle Preprocessing-Schritte aus Notebook 02.
# Wird von Notebooks und spaeteren Phasen importiert (kein Code-Duplikat).

import re
import html

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Einmalige Initialisierung auf Modul-Ebene (nicht bei jedem Funktionsaufruf neu)
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Bereinigt rohen Tweet-Text: HTML-Entities, URLs, Mojibake, Zahlen, Lowercase.

    Begruendung der Schritte siehe Notebook 02, Zelle 03/04.
    """
    text = html.unescape(text)
    text = re.sub(r"http[s]?://\S+", "", text)             # URLs entfernen
    text = re.sub(r"[\x80-\xff]", "", text)                 # Mojibake entfernen
    text = re.sub(r"\d+(?:[.,]\d+)*", "NUM", text)           # Zahlen -> Platzhalter
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Wort-Tokenisierung (nur Buchstaben-Tokens, siehe Notebook 02 Zelle 06)."""
    return re.findall(r"\b[a-z]+\b", text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Entfernt englische Stopwords (NLTK-Standardliste)."""
    return [t for t in tokens if t not in STOP_WORDS]


def apply_stemming(tokens: list[str]) -> list[str]:
    """Porter-Stemming (aggressiv, oft keine echten Woerter)."""
    return [STEMMER.stem(t) for t in tokens]


def apply_lemmatization(tokens: list[str]) -> list[str]:
    """WordNet-Lemmatisierung (konservativ, lesbare Wortformen)."""
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess_pipeline(text: str) -> dict:
    """
    Fuehrt die vollstaendige Preprocessing-Kette fuer einen einzelnen Text aus
    und gibt alle Zwischen-/Endvarianten als Dictionary zurueck (entspricht
    den Spalten aus Notebook 02, Zelle 14).
    """
    text_clean = clean_text(text)
    tokens = tokenize(text_clean)
    tokens_no_stopwords = remove_stopwords(tokens)
    tokens_stemmed = apply_stemming(tokens_no_stopwords)
    tokens_lemmatized = apply_lemmatization(tokens_no_stopwords)

    return {
        "text_clean": text_clean,
        "text_no_stopwords": " ".join(tokens_no_stopwords),
        "text_stemmed": " ".join(tokens_stemmed),
        "text_lemmatized": " ".join(tokens_lemmatized),
    }
