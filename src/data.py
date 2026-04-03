"""
src.data
─────────
Loaders for all three data sources:
  - MedQA USMLE   (JSONL, questions/US/4_options/)
  - MedMCQA       (Parquet, train/validation/test)
  - Medical textbooks (plain .txt, textbooks/en/)

All functions return plain pandas DataFrames so notebooks stay simple.
Paths are anchored to the repo root (the directory containing pyproject.toml),
so they work correctly regardless of where the notebook or script is run from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd

# ── Repo root detection ───────────────────────────────────────────────────────
# Walk up from this file (src/data.py) until we find pyproject.toml.
# This makes every path work regardless of cwd — notebooks, scripts, or shell.

def _find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(
        "Could not find repo root (no pyproject.toml found in any parent directory).\n"
        f"Starting search from: {here}"
    )

REPO_ROOT = _find_repo_root()

# ── Default data paths (all relative to repo root) ───────────────────────────

DATA_DIR     = REPO_ROOT / "data"
MEDQA_DIR    = DATA_DIR / "MedQA-USMLE" / "questions" / "US" / "4_options"
MEDMCQA_DIR  = DATA_DIR / "MedMCQA"
TEXTBOOK_DIR = DATA_DIR / "MedQA-USMLE" / "textbooks" / "en"


# ── MedQA USMLE ──────────────────────────────────────────────────────────────

def load_medqa(
    split: Literal["train", "dev", "test"] = "train",
    data_dir: Path | str | None = None,
    n: int | None = None,
) -> pd.DataFrame:
    """
    Load MedQA USMLE 4-option questions.

    Returns a DataFrame with columns:
        question     : str        — the clinical vignette
        options      : dict       — {"A": ..., "B": ..., "C": ..., "D": ...}
        answer_idx   : str        — correct option key ("A" / "B" / "C" / "D")
        answer       : str        — correct option text (convenience column)
        meta_phrases : list | None — MetaMap phrases if available

    Parameters
    ----------
    split    : "train" | "dev" | "test"
    data_dir : override the default MEDQA_DIR
    n        : if set, return only the first n rows (quick exploration)
    """
    base = Path(data_dir) if data_dir else MEDQA_DIR
    path = base / f"phrases_no_exclude_{split}.jsonl"

    if not path.exists():
        raise FileNotFoundError(
            f"MedQA file not found: {path}\n"
            f"Repo root detected as: {REPO_ROOT}\n"
            f"Expected layout: <repo>/data/MedQA-USMLE/questions/US/4_options/"
        )

    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            obj = json.loads(line)
            rows.append({
                "question":     obj["question"],
                "options":      obj["options"],
                "answer_idx":   obj["answer_idx"],
                "answer":       obj["options"][obj["answer_idx"]],
                "meta_phrases": obj.get("metamap_phrases"),
            })

    df = pd.DataFrame(rows)
    print(f"[MedQA] Loaded {len(df):,} rows  (split={split})")
    return df


def load_medqa_all(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load and concatenate all three MedQA splits with a 'split' column."""
    parts = []
    for split in ("train", "dev", "test"):
        df = load_medqa(split=split, data_dir=data_dir)
        df["split"] = split
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


# ── MedMCQA ──────────────────────────────────────────────────────────────────

def load_medmcqa(
    split: Literal["train", "validation", "test"] = "train",
    data_dir: Path | str | None = None,
    n: int | None = None,
) -> pd.DataFrame:
    """
    Load MedMCQA from Parquet files (HuggingFace format).

    Returns a DataFrame with columns:
        question     : str  — the question stem
        opa / opb / opc / opd : str  — the four answer options
        cop          : int  — correct option index (0=A, 1=B, 2=C, 3=D)
        answer       : str  — correct option text (convenience column)
        subject_name : str  — medical subject (pharmacology, anatomy, etc.)
        topic_name   : str  — finer topic label

    Parameters
    ----------
    split    : "train" | "validation" | "test"
    data_dir : override the default MEDMCQA_DIR
    n        : if set, return only the first n rows
    """
    base = Path(data_dir) if data_dir else MEDMCQA_DIR
    path = base / f"{split}-00000-of-00001.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"MedMCQA file not found: {path}\n"
            f"Repo root detected as: {REPO_ROOT}\n"
            f"Expected layout: <repo>/data/MedMCQA/{split}-00000-of-00001.parquet"
        )

    df = pd.read_parquet(path)
    if n is not None:
        df = df.head(n)

    # Add plain-text answer column for convenience
    option_cols = ["opa", "opb", "opc", "opd"]
    df["answer"] = df.apply(
        lambda row: row[option_cols[int(row["cop"])]]
        if pd.notna(row["cop"]) else None,
        axis=1,
    )

    print(f"[MedMCQA] Loaded {len(df):,} rows  (split={split})")
    return df


# ── Medical textbooks ─────────────────────────────────────────────────────────

# Friendly display names for the 18 USMLE textbooks
TEXTBOOK_NAMES = {
    "Anatomy_Gray":            "Gray's Anatomy",
    "Biochemistry_Lippincott": "Lippincott Biochemistry",
    "Cell_Biology_Alberts":    "Alberts Cell Biology",
    "First_Aid_Step1":         "First Aid Step 1",
    "First_Aid_Step2":         "First Aid Step 2",
    "Gynecology_Novak":        "Novak's Gynecology",
    "Histology_Ross":          "Ross Histology",
    "Immunology_Janeway":      "Janeway's Immunology",
    "InternalMed_Harrison":    "Harrison's Internal Medicine",
    "Neurology_Adams":         "Adams Neurology",
    "Obstentrics_Williams":    "Williams Obstetrics",
    "Pathology_Robbins":       "Robbins Pathology",
    "Pathoma_Husain":          "Pathoma (Husain)",
    "Pediatrics_Nelson":       "Nelson Pediatrics",
    "Pharmacology_Katzung":    "Katzung Pharmacology",
    "Physiology_Levy":         "Levy Physiology",
    "Psichiatry_DSM-5":        "DSM-5 Psychiatry",
    "Surgery_Schwartz":        "Schwartz Surgery",
}


def load_textbook(stem: str, textbook_dir: Path | str | None = None) -> str:
    """
    Load a single textbook as a raw string.

    Parameters
    ----------
    stem : filename without .txt  (e.g. "InternalMed_Harrison")
    """
    base = Path(textbook_dir) if textbook_dir else TEXTBOOK_DIR
    path = base / f"{stem}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Textbook not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    friendly = TEXTBOOK_NAMES.get(stem, stem)
    print(f"[Textbook] Loaded '{friendly}'  ({len(text):,} chars)")
    return text


def load_all_textbooks(textbook_dir: Path | str | None = None) -> dict[str, str]:
    """
    Load all 18 English textbooks into a dict keyed by stem.

    Returns
    -------
    { "InternalMed_Harrison": "<full text>", ... }
    """
    base = Path(textbook_dir) if textbook_dir else TEXTBOOK_DIR
    books = {}
    for path in sorted(base.glob("*.txt")):
        stem = path.stem
        books[stem] = path.read_text(encoding="utf-8", errors="replace")
        friendly = TEXTBOOK_NAMES.get(stem, stem)
        print(f"  ✓  {friendly}  ({len(books[stem]):,} chars)")
    print(f"\n[Textbooks] Loaded {len(books)} books total")
    return books


def textbooks_as_dataframe(textbook_dir: Path | str | None = None) -> pd.DataFrame:
    """
    Return textbooks as a DataFrame — useful for EDA and clustering notebooks.

    Columns: stem | friendly_name | text | char_count
    """
    books = load_all_textbooks(textbook_dir)
    rows = [
        {
            "stem":          stem,
            "friendly_name": TEXTBOOK_NAMES.get(stem, stem),
            "text":          text,
            "char_count":    len(text),
        }
        for stem, text in books.items()
    ]
    return pd.DataFrame(rows)
