"""
emma.classify
-------------
Specialty classifier: given a medical question, predict which medical
subject it belongs to (Pharmacology, Medicine, Surgery, etc.).

Trained on MedMCQA which provides gold subject_name labels for 182k
questions. The trained model is then applied to MedQA questions (which
have no subject labels) to produce specialty routing tags for the RAG
pipeline.

Following the A1 methodology: a full grid of feature representations x
classifiers is evaluated via 10-fold stratified cross-validation, with
weighted F1 and Cohen's kappa as primary metrics. The champion config
is selected empirically on this corpus -- not assumed from A1, since
MedMCQA is a different dataset.

Feature types evaluated:
  BOW, TF-IDF Bigrams, Word2Vec, Sentence Embeddings (MiniLM / Qwen3)

Classifiers evaluated:
  LinearSVC, Naive Bayes (MNB), Random Forest

Omitted with justification (see notebook section 4 for full rationale):
  LR, SGD, KNN, XGBoost, CatBoost, LightGBM, DistilBERT,
  TF (no IDF), TF-IDF unigrams, Trigrams, Doc2Vec, LDA, FastText

Evaluation: 10-fold stratified CV on stratified 20k sample,
weighted F1 + Cohen's kappa. Champion retrained on full corpus.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from tqdm import tqdm

# ── Subject normalisation ─────────────────────────────────────────────────────
# MedMCQA has 21 raw subject_name values. We collapse rare / overlapping ones
# into 13 canonical specialties for cleaner classification.

# ── Subject collapse rules ────────────────────────────────────────────────────
# These are explicit design decisions: subjects too small or too semantically
# similar to neighbouring classes to train reliably on their own are merged.
# The full subject list is NOT hardcoded here — it is read from the data.
# Any subject not listed in COLLAPSE_RULES passes through unchanged.

COLLAPSE_RULES: dict[str, str] = {
    "Medicine":                     "Internal Medicine",
    "Social & Preventive Medicine": "Public Health",
    "Forensic Medicine":            "Public Health",
    "Gynaecology & Obstetrics":     "Obstetrics & Gynaecology",
    "Skin":                         "Dermatology",
}

# Populated at runtime by build_subject_map() after loading the data.
SUBJECT_MAP: dict[str, str] = {}


def build_subject_map(subject_series: "pd.Series") -> dict[str, str]:
    """
    Build the full subject -> canonical specialty mapping from the data.

    Reads all unique subject_name values from the DataFrame, applies
    COLLAPSE_RULES for subjects that need renaming or collapsing, and
    passes everything else through unchanged.

    Call this once after loading MedMCQA:
        SUBJECT_MAP.update(build_subject_map(mcqa["subject_name"]))

    Parameters
    ----------
    subject_series : raw subject_name column from load_medmcqa()

    Returns
    -------
    dict mapping every observed subject_name -> canonical specialty string
    """
    mapping = {}
    for subject in subject_series.unique():
        if not isinstance(subject, str) or subject.strip() == "":
            mapping[subject] = "Unknown"
        else:
            mapping[subject] = COLLAPSE_RULES.get(subject, subject)
    return mapping

RANDOM_SEED = 42

CLASSIFIERS = {
    "LinearSVC": lambda: LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_SEED),
    "MNB":       lambda: MultinomialNB(),
    "RF":        lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED),
}


# ── Label helpers ─────────────────────────────────────────────────────────────

def normalise_subjects(series: pd.Series) -> pd.Series:
    """
    Map raw MedMCQA subject_name values to canonical specialty labels.

    Requires SUBJECT_MAP to be populated first via build_subject_map().
    Subjects not in the map fall back to "Unknown".
    """
    if not SUBJECT_MAP:
        raise RuntimeError(
            "SUBJECT_MAP is empty. Call build_subject_map() first:\n"
            "  from src.classify import SUBJECT_MAP, build_subject_map\n"
            "  SUBJECT_MAP.update(build_subject_map(mcqa['subject_name']))"
        )
    return series.map(SUBJECT_MAP).fillna("Unknown")


def get_label_encoder(labels: pd.Series) -> LabelEncoder:
    """Fit and return a LabelEncoder on canonical specialty labels."""
    le = LabelEncoder()
    # Cast to plain str to avoid np.str_ showing up in printouts
    le.fit(sorted(str(v) for v in labels.unique()))
    return le


# ── Pipeline builders ─────────────────────────────────────────────────────────

def _make_clf(name: str):
    return CLASSIFIERS[name]()


def build_bow_pipeline(
    classifier: str = "LinearSVC",
    binary: bool = True,
) -> Pipeline:
    """Binary bag-of-words + classifier."""
    return Pipeline([
        ("vec", CountVectorizer(
            max_features=50_000, binary=binary,
            strip_accents="unicode", min_df=2,
        )),
        ("clf", _make_clf(classifier)),
    ])


def build_tf_pipeline(classifier: str = "LinearSVC") -> Pipeline:
    """Term-frequency (no IDF) + classifier."""
    return Pipeline([
        ("vec", TfidfVectorizer(
            max_features=50_000, use_idf=False, sublinear_tf=False,
            strip_accents="unicode", min_df=2,
        )),
        ("clf", _make_clf(classifier)),
    ])


def build_tfidf_pipeline(
    classifier: str = "LinearSVC",
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 1),
) -> Pipeline:
    """TF-IDF + classifier. ngram_range=(1,2) adds bigrams."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            min_df=2,
        )),
        ("clf", _make_clf(classifier)),
    ])


def train_word2vec(
    texts: list[str],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
):
    """
    Train a Word2Vec model on the given texts.

    Returns the fitted gensim Word2Vec model.
    """
    from gensim.models import Word2Vec

    tokenised = [t.lower().split() for t in texts]
    return Word2Vec(
        sentences=tokenised,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=RANDOM_SEED,
    )


def embed_word2vec(model, texts: list[str]) -> np.ndarray:
    """Mean-pool Word2Vec word vectors for each document."""
    vecs = []
    for text in texts:
        tokens = text.lower().split()
        token_vecs = [model.wv[t] for t in tokens if t in model.wv]
        vecs.append(
            np.mean(token_vecs, axis=0) if token_vecs
            else np.zeros(model.vector_size)
        )
    return np.array(vecs, dtype=np.float32)


def build_embedding_clf(classifier: str = "LinearSVC"):
    """Return an unfitted classifier for use with pre-computed embeddings."""
    return _make_clf(classifier)


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_pipeline(
    pipeline,
    X,
    y: np.ndarray,
    n_splits: int = 10,
    config_label: str = "",
    config_num: int = 0,
    config_total: int = 0,
) -> dict:
    """
    10-fold stratified cross-validation with verbose per-fold progress.

    Prints a clear header before starting and one line per fold showing
    fold number, F1, kappa, elapsed time, and running mean so far.
    Designed for transparency — you always know exactly where you are.

    Returns a dict with:
        mean_f1, std_f1, mean_kappa, std_kappa, fold_f1, fold_kappa
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    n_samples = len(y)

    prefix = f"[{config_num}/{config_total}]" if config_total else ""
    print(f"{prefix} {config_label}  ({n_samples:,} samples, {n_splits} folds)")

    fold_f1:    list[float] = []
    fold_kappa: list[float] = []
    t_config = time.time()

    bar_fmt = "    {l_bar}{bar}| {n:,}/{total:,} folds [{elapsed}<{remaining}]"
    pbar = tqdm(total=n_splits, unit=" fold", bar_format=bar_fmt, leave=False)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()

        if hasattr(X, "iloc"):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_tr, X_val = X[train_idx], X[val_idx]

        y_tr, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)

        f1    = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        kappa = cohen_kappa_score(y_val, y_pred)

        fold_f1.append(f1)
        fold_kappa.append(kappa)

        elapsed  = time.time() - t0
        run_f1   = float(np.mean(fold_f1))
        run_kap  = float(np.mean(fold_kappa))
        pbar.update(1)
        tqdm.write(
            f"    Fold {fold+1:2d}/{n_splits} | "
            f"F1={f1:.4f}  kappa={kappa:.4f}  ({elapsed:.1f}s) | "
            f"running mean -> F1={run_f1:.4f}  kappa={run_kap:.4f}"
        )

    pbar.close()
    total_elapsed = time.time() - t_config
    mean_f1   = float(np.mean(fold_f1))
    mean_kap  = float(np.mean(fold_kappa))
    std_f1    = float(np.std(fold_f1))
    std_kap   = float(np.std(fold_kappa))
    print(
        f"    Done  -> mean F1={mean_f1:.4f} +/- {std_f1:.4f} | "
        f"mean kappa={mean_kap:.4f} +/- {std_kap:.4f}  "
        f"({total_elapsed/60:.1f} min total)\n"
    )

    return {
        "mean_f1":    mean_f1,
        "std_f1":     std_f1,
        "mean_kappa": mean_kap,
        "std_kappa":  std_kap,
        "fold_f1":    fold_f1,
        "fold_kappa": fold_kappa,
    }


def run_grid(
    configs: list[dict],
    y: np.ndarray,
    n_splits: int = 10,
) -> pd.DataFrame:
    """
    Run a full feature x classifier grid with 10-fold CV.

    Each entry in configs is a dict with keys:
        name     : display name for the results row
        pipeline : a fittable pipeline or EmbeddingPipeline wrapper
        X        : feature matrix (pd.Series for text, np.ndarray for embeddings)
        corpus   : optional string describing the corpus size

    Prints a numbered header per config and per-fold lines for full
    transparency. Returns a DataFrame with one row per config.
    """
    n_total = len(configs)
    t_grid  = time.time()
    print(f"=== Feature x Classifier Grid: {n_total} configs x {n_splits} folds "
          f"= {n_total * n_splits} fits ===\n")

    rows = []
    for i, cfg in enumerate(configs, 1):
        # Use the config's own y if provided (embedding configs use y_sample)
        cfg_y = cfg.get("y", y)
        cv = cross_validate_pipeline(
            cfg["pipeline"], cfg["X"], cfg_y,
            n_splits=n_splits,
            config_label=cfg["name"],
            config_num=i,
            config_total=n_total,
        )
        rows.append({
            "Configuration": cfg["name"],
            "Corpus":        cfg.get("corpus", f"{len(cfg_y):,}"),
            "Weighted F1":   f"{cv['mean_f1']:.4f} +/- {cv['std_f1']:.4f}",
            "Cohen's kappa": f"{cv['mean_kappa']:.4f} +/- {cv['std_kappa']:.4f}",
            "_cv":           cv,
        })

    total_min = (time.time() - t_grid) / 60
    print(f"=== Grid complete in {total_min:.1f} min ===")
    return pd.DataFrame(rows)


def format_cv_results(results: dict, name: str) -> str:
    """Format cross-validation results as a readable string."""
    return (
        f"{name}\n"
        f"  Weighted F1  : {results['mean_f1']:.4f} +/- {results['std_f1']:.4f}\n"
        f"  Cohen's kappa: {results['mean_kappa']:.4f} +/- {results['std_kappa']:.4f}"
    )


# ── Train final model ─────────────────────────────────────────────────────────

def train_classifier(
    X_train,
    y_train: np.ndarray,
    pipeline=None,
) -> Pipeline:
    """
    Train the final classifier on the full training set.

    If pipeline is None, defaults to TF-IDF bigrams + LinearSVC.
    """
    if pipeline is None:
        pipeline = build_tfidf_pipeline(ngram_range=(1, 2))
    pipeline.fit(X_train, y_train)
    return pipeline


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_specialty(
    questions: pd.Series | list[str],
    pipeline,
    label_encoder: LabelEncoder,
) -> pd.Series:
    """
    Predict specialty labels for a series of questions.

    Returns a Series of decoded string labels (e.g. "Pharmacology").
    """
    if isinstance(questions, list):
        questions = pd.Series(questions)
    encoded = pipeline.predict(questions)
    return pd.Series(label_encoder.inverse_transform(encoded), index=questions.index)


# ── Evaluation helpers ────────────────────────────────────────────────────────

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
) -> None:
    """Print sklearn classification report with decoded class names."""
    print(classification_report(
        y_true, y_pred, target_names=label_encoder.classes_, zero_division=0
    ))


def inter_category_similarity(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    n_per_class: int = 200,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Compute pairwise cosine similarity between TF-IDF category centroids.

    Replicates the A1 corpus characterisation step.
    High mean off-diagonal similarity (>0.85) -> TF-IDF likely dominates.
    Lower similarity -> embeddings may be competitive.

    Returns a symmetric DataFrame of shape (n_classes, n_classes).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    rng     = np.random.default_rng(seed)
    classes = sorted(df[label_col].unique())

    samples = []
    for cls in classes:
        idx    = df[df[label_col] == cls].index
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        samples.append(df.loc[chosen, text_col])

    vectorizer = TfidfVectorizer(max_features=20_000, sublinear_tf=True, min_df=2)
    vectorizer.fit(pd.concat(samples))

    centroids = normalize(np.vstack([
        np.asarray(vectorizer.transform(s).mean(axis=0)) for s in samples
    ]))
    sim = cosine_similarity(centroids)

    return pd.DataFrame(sim, index=classes, columns=classes)
