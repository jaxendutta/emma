"""
emma.cluster
------------
Topic modelling and clustering on MedQA questions using BERTopic.

Purpose in the EMMA pipeline:
  The classifier (notebook 02) assigns a broad specialty label to each
  MedQA question. Clustering goes one level deeper: within each specialty,
  BERTopic discovers latent topic structure from the question text itself,
  without any labels. These topics are used at inference time to retrieve
  the most topically similar questions from the question bank, providing
  few-shot context for the LLM.

Following A2 methodology:
  - Primary metric  : Cohen's kappa (agreement between topic and specialty label)
  - Secondary metrics: Silhouette score, Topic Coherence C_v
  - A2 champion baselines:
      TF-IDF + GMM/EM      (kappa = 0.418, frequency-based champion)
      PubMedBERT + Spectral (state-of-the-art champion)
  These are reproduced on MedQA as starting points, then BERTopic is
  evaluated as the primary method.

Data-driven philosophy:
  No topic count is hardcoded. BERTopic's HDBSCAN component determines
  the number of topics automatically from the data. nr_topics is used
  only for post-hoc reduction if the initial count is unwieldy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import silhouette_score as _silhouette_score
from sklearn.preprocessing import LabelEncoder


# ── Coherence ─────────────────────────────────────────────────────────────────

def compute_coherence_cv(
    topic_words: list[list[str]],
    texts: list[str],
    topn: int = 10,
) -> float:
    """
    Compute mean C_v topic coherence using gensim.

    C_v combines indirect confirmation measures with NPMI and cosine
    similarity -- the standard metric from A2.

    Parameters
    ----------
    topic_words : list of word lists, one per topic (from BERTopic)
    texts       : tokenised documents (list of word lists or raw strings)
    topn        : number of top words per topic to evaluate

    Returns
    -------
    Mean C_v score across all non-outlier topics.
    """
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.corpora import Dictionary

    # Tokenise if raw strings passed
    if texts and isinstance(texts[0], str):
        tokenised = [t.lower().split() for t in texts]
    else:
        tokenised = texts

    dictionary = Dictionary(tokenised)
    corpus     = [dictionary.doc2bow(doc) for doc in tokenised]

    cm = CoherenceModel(
        topics=topic_words,
        texts=tokenised,
        dictionary=dictionary,
        coherence="c_v",
        topn=topn,
    )
    return cm.get_coherence()


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_topics(
    topic_labels: np.ndarray | list[int],
    specialty_labels: np.ndarray | list[str],
    embeddings: np.ndarray | None = None,
    topic_words: list[list[str]] | None = None,
    texts: list[str] | None = None,
) -> dict:
    """
    Compute all three A2 evaluation metrics for a clustering result.

    Parameters
    ----------
    topic_labels     : BERTopic topic assignments (int, -1 = outlier)
    specialty_labels : ground-truth specialty from notebook 02 classifier
    embeddings       : sentence embeddings for Silhouette (optional)
    topic_words      : word lists per topic for C_v coherence (optional)
    texts            : raw question strings for C_v coherence (optional)

    Returns
    -------
    dict with keys: kappa, silhouette, coherence_cv
    None values for metrics that could not be computed.
    """
    topics  = np.array(topic_labels)
    specs   = np.array(specialty_labels)

    # Filter outliers (topic = -1) for all metrics
    mask    = topics != -1
    n_total = len(topics)
    n_valid = mask.sum()
    n_out   = n_total - n_valid

    results: dict = {
        "n_topics":    int(topics[mask].max() + 1) if n_valid > 0 else 0,
        "n_outliers":  int(n_out),
        "outlier_pct": round(100 * n_out / n_total, 1),
        "kappa":       None,
        "silhouette":  None,
        "coherence_cv": None,
    }

    if n_valid < 2:
        return results

    # Cohen's kappa: agreement between topic id and specialty label
    le = LabelEncoder()
    spec_enc = le.fit_transform(specs[mask])
    results["kappa"] = round(float(cohen_kappa_score(topics[mask], spec_enc)), 4)

    # Silhouette score on embeddings
    if embeddings is not None and len(np.unique(topics[mask])) > 1:
        try:
            results["silhouette"] = round(
                float(_silhouette_score(embeddings[mask], topics[mask], metric="cosine")),
                4,
            )
        except Exception:
            pass

    # C_v coherence
    if topic_words and texts:
        try:
            results["coherence_cv"] = round(
                float(compute_coherence_cv(topic_words, texts)), 4
            )
        except Exception:
            pass

    return results


def format_eval(results: dict, name: str = "") -> str:
    """Format evaluate_topics() results as a readable string."""
    lines = [f"{name}" if name else ""]
    lines += [
        f"  Topics         : {results['n_topics']}",
        f"  Outliers       : {results['n_outliers']:,}  ({results['outlier_pct']}%)",
        f"  Cohen's kappa  : {results['kappa']}",
        f"  Silhouette     : {results['silhouette']}",
        f"  Coherence C_v  : {results['coherence_cv']}",
    ]
    return "\n".join(lines)


# ── BERTopic helpers ──────────────────────────────────────────────────────────

def get_topic_words(topic_model, n_words: int = 10) -> list[list[str]]:
    """
    Extract top-n words per topic from a fitted BERTopic model.

    Excludes the outlier topic (-1).
    """
    words = []
    for topic_id in sorted(topic_model.get_topics().keys()):
        if topic_id == -1:
            continue
        topic_terms = topic_model.get_topic(topic_id)
        if topic_terms:
            words.append([w for w, _ in topic_terms[:n_words]])
    return words


def topic_specialty_alignment(
    topics: list[int],
    specialties: list[str],
    topic_model,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    For each discovered topic, show the dominant specialty labels.

    Useful for qualitative validation: a well-formed topic should be
    dominated by one or two specialties, not uniformly distributed.

    Returns a DataFrame with one row per topic showing:
        topic_id, top_words, dominant_specialty, specialty_pct, size
    """
    df = pd.DataFrame({"topic": topics, "specialty": specialties})
    df = df[df["topic"] != -1]

    rows = []
    for tid, grp in df.groupby("topic"):
        spec_counts = grp["specialty"].value_counts()
        top_spec    = spec_counts.index[0]
        top_pct     = round(100 * spec_counts.iloc[0] / len(grp), 1)

        topic_terms = topic_model.get_topic(tid)
        top_words   = ", ".join(w for w, _ in topic_terms[:5]) if topic_terms else ""

        rows.append({
            "topic_id":          int(tid),
            "top_words":         top_words,
            "dominant_specialty": top_spec,
            "specialty_pct":     top_pct,
            "size":              len(grp),
        })

    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
