"""
emma.vectorstore
────────────────
Builds and queries the FAISS vector index over the 18 medical textbooks.

Two main responsibilities:
  1. BUILD  – chunk all textbooks, embed with a biomedical sentence-transformer,
               persist the FAISS index + metadata to disk.
  2. QUERY  – load the index and retrieve the top-k most relevant chunks
               for any natural-language query.

Typical usage
─────────────
Build once (takes ~10–30 min depending on hardware):

    from vectorstore import build_index
    build_index()          # writes to models/vectorstore/

Query at inference time:

    from vectorstore import load_index, search
    index, meta = load_index()
    results = search("What is the mechanism of septic shock?", index, meta, k=5)
    for r in results:
        print(r["score"], r["book"], r["text"][:200])
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Repo root detection (same approach as data.py) ────────────────────────────

def _find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not find repo root (no pyproject.toml found).")

REPO_ROOT = _find_repo_root()

# ── Default paths ─────────────────────────────────────────────────────────────

TEXTBOOK_DIR  = REPO_ROOT / "data" / "MedQA-USMLE" / "textbooks" / "en"
VECTORSTORE_DIR = REPO_ROOT / "models" / "vectorstore"

# ── Embedding model ───────────────────────────────────────────────────────────
# PubMedBERT fine-tuned for semantic similarity — domain-appropriate for
# medical text. Falls back to MiniLM if the biomedical model is unavailable.

BIOMEDICAL_MODEL = "Qwen/Qwen3-Embedding-0.6B"
FALLBACK_MODEL   = "all-MiniLM-L6-v2"

# Instruction prefix for queries (improves retrieval by 1-5%)
# Documents are embedded WITHOUT a prefix; only queries use this
QUERY_INSTRUCTION = "Instruct: Given a medical question, retrieve relevant textbook passages that answer it\nQuery: "

# ── Chunking parameters ───────────────────────────────────────────────────────

CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 50    # words of overlap between consecutive chunks


# ── Friendly book names (mirrors data.py) ────────────────────────────────────

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


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping word-level chunks.

    Returns a list of chunk strings. Each chunk is approximately
    `chunk_size` words, with `overlap` words shared with the next chunk.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap  # slide window with overlap
    return chunks


def chunk_all_textbooks(
    textbook_dir: Path | str | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Load and chunk all 18 textbooks.

    Returns
    -------
    texts : list[str]
        All chunk strings in order.
    metadata : list[dict]
        Parallel list with one dict per chunk:
        { "book": stem, "friendly_name": str, "chunk_idx": int }
    """
    base = Path(textbook_dir) if textbook_dir else TEXTBOOK_DIR
    texts: list[str] = []
    metadata: list[dict] = []

    for path in tqdm(sorted(base.glob("*.txt")), desc="Chunking textbooks"):
        stem = path.stem
        friendly = TEXTBOOK_NAMES.get(stem, stem)
        raw = path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(raw, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadata.append({
                "book":          stem,
                "friendly_name": friendly,
                "chunk_idx":     idx,
                "char_start":    None,   # approximate — not tracked at word level
            })

    print(f"\n[Vectorstore] Total chunks: {len(texts):,} across {len(set(m['book'] for m in metadata))} books")
    return texts, metadata


# ── Embedding ─────────────────────────────────────────────────────────────────

def load_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    """
    Load the sentence-transformer embedding model.

    Tries the biomedical model first; falls back to MiniLM if unavailable.
    """
    target = model_name or BIOMEDICAL_MODEL
    try:
        import torch
        is_cuda = torch.cuda.is_available()

        if is_cuda:
            # Free any cached memory from previous runs before loading
            torch.cuda.empty_cache()
            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
            total_gb = torch.cuda.mem_get_info()[1] / 1024**3
            print(f"[Vectorstore] GPU memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
            if free_gb < 1.5:
                raise RuntimeError(
                    f"Only {free_gb:.1f} GB free on GPU. "
                    "Run the GPU cleanup cell above, or do Runtime → Restart session "
                    "and re-run the setup cells before building."
                )
            elif free_gb < 4.0:
                print(f"[Vectorstore] ⚠️  Only {free_gb:.1f} GB free — proceeding but watch for OOM")

        # Load in float16 on GPU, float32 on CPU
        # Use device_map="cuda" so weights are streamed directly onto GPU in fp16
        # rather than loaded in fp32 on CPU then moved (which doubles peak memory)
        if is_cuda:
            model = SentenceTransformer(
                target,
                model_kwargs={
                    "torch_dtype": torch.float16,
                    "device_map": "cuda",
                },
                tokenizer_kwargs={"padding_side": "left"},
            )
        else:
            model = SentenceTransformer(target)

        device = "cuda" if is_cuda else "cpu"
        dtype = next(model.parameters()).dtype
        print(f"[Vectorstore] Embedding model: {target}  (dtype={dtype}, device={device})")
        return model
    except RuntimeError as e:
        # Re-raise memory errors with a clear message — don't silently fall back
        if "out of memory" in str(e).lower() or "free" in str(e).lower():
            raise
        print(f"[Vectorstore] Could not load {target}: {e}")
        print(f"[Vectorstore] Falling back to {FALLBACK_MODEL}")
        import torch
        torch.cuda.empty_cache()
        return SentenceTransformer(FALLBACK_MODEL)
    except Exception as e:
        print(f"[Vectorstore] Could not load {target}: {e}")
        print(f"[Vectorstore] Falling back to {FALLBACK_MODEL}")
        return SentenceTransformer(FALLBACK_MODEL)


def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 16,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed a list of text strings into a float32 numpy array.

    Shape: (len(texts), embedding_dim)
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity via dot product
    )
    return embeddings.astype(np.float32)


# ── FAISS index ───────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS inner-product (cosine) index from normalised embeddings.

    Since embeddings are L2-normalised, inner product == cosine similarity.
    IndexFlatIP is exact (no approximation) — fine for ~200k chunks.
    """
    if embeddings.ndim < 2 or embeddings.shape[0] == 0:
        raise ValueError(
            f"embeddings array is empty (shape={embeddings.shape}). "
            "No textbook chunks were found — check that TEXTBOOK_DIR exists "
            f"and contains .txt files:\n  {TEXTBOOK_DIR}"
        )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[Vectorstore] FAISS index built: {index.ntotal:,} vectors, dim={dim}")
    return index


# ── Persist & load ────────────────────────────────────────────────────────────

def save_index(
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    out_dir: Path | str | None = None,
    model_name: str = BIOMEDICAL_MODEL,
) -> None:
    """
    Persist the FAISS index and metadata to disk.

    Writes:
        models/vectorstore/index.faiss
        models/vectorstore/metadata.pkl
        models/vectorstore/config.json
    """
    out = Path(out_dir) if out_dir else VECTORSTORE_DIR
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out / "index.faiss"))

    with open(out / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    config = {
        "model_name":  model_name,
        "chunk_size":  CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_chunks":  len(metadata),
        "num_books":   len(set(m["book"] for m in metadata)),
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[Vectorstore] Saved to {out}")
    print(f"  index.faiss  ({index.ntotal:,} vectors)")
    print(f"  metadata.pkl ({len(metadata):,} entries)")
    print(f"  config.json  {config}")


def load_index(
    index_dir: Path | str | None = None,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Load the FAISS index and metadata from disk.

    Returns
    -------
    index    : faiss.IndexFlatIP
    metadata : list[dict]  — parallel to the index vectors
    """
    src = Path(index_dir) if index_dir else VECTORSTORE_DIR
    index_path = src / "index.faiss"
    meta_path  = src / "metadata.pkl"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}.\n"
            "Run build_index() first to generate it."
        )

    index = faiss.read_index(str(index_path))

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"[Vectorstore] Loaded index: {index.ntotal:,} vectors from {src}")
    return index, metadata


# ── Search ────────────────────────────────────────────────────────────────────

def search(
    query: str,
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    texts: list[str],
    model: SentenceTransformer,
    k: int = 5,
) -> list[dict]:
    """
    Retrieve the top-k most relevant textbook chunks for a query.

    Parameters
    ----------
    query    : natural-language question
    index    : loaded FAISS index
    metadata : parallel metadata list
    texts    : parallel list of chunk strings
    model    : the same embedding model used to build the index
    k        : number of results to return

    Returns
    -------
    List of dicts, each with keys:
        rank, score, book, friendly_name, chunk_idx, text
    """
    # Prepend instruction prefix to queries (Qwen3-Embedding recommendation)
    # Documents are embedded without prefix; only queries use this instruction
    prefixed_query = QUERY_INSTRUCTION + query
    query_vec = model.encode(
        [prefixed_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(query_vec, k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:
            continue  # FAISS returns -1 for empty slots
        meta = metadata[idx]
        results.append({
            "rank":          rank + 1,
            "score":         float(score),
            "book":          meta["book"],
            "friendly_name": meta["friendly_name"],
            "chunk_idx":     meta["chunk_idx"],
            "text":          texts[idx],
        })
    return results


# ── Top-level build function ──────────────────────────────────────────────────

def build_index(
    textbook_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    model_name: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    batch_size: int = 16,
) -> None:
    """
    Full pipeline: chunk → embed → index → save.

    This is the function to call from notebook 01_vectorstore_build.ipynb.
    Depending on hardware, this takes 10–45 minutes.
    """
    import os
    # Reduce CUDA memory fragmentation (helps on T4 with 15GB VRAM)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    print("=" * 60)
    print(" EMMA vectorstore build")
    print("=" * 60)

    # 1. Chunk
    texts, metadata = chunk_all_textbooks(
        textbook_dir=textbook_dir,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # 2. Embed
    model = load_embedding_model(model_name)
    print(f"\n[Vectorstore] Embedding {len(texts):,} chunks (batch_size={batch_size})…")
    embeddings = embed_texts(texts, model, batch_size=batch_size)

    # 3. Index
    index = build_faiss_index(embeddings)

    # 4. Save
    save_index_with_texts(
        index=index,
        metadata=metadata,
        out_dir=out_dir,
        model_name=model_name or BIOMEDICAL_MODEL,
        texts=texts,
    )

    print("\n[Vectorstore] Build complete.")


# ── Extended save/load that also persists the raw chunk texts ─────────────────
# Using these avoids re-chunking on every load.

def save_index_with_texts(
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    texts: list[str],
    out_dir: Path | str | None = None,
    model_name: str = BIOMEDICAL_MODEL,
) -> None:
    """
    Like save_index(), but also saves the chunk text strings to
    models/vectorstore/texts.pkl so they don't need to be re-generated.
    """
    save_index(index, metadata, out_dir=out_dir, model_name=model_name)
    out = Path(out_dir) if out_dir else VECTORSTORE_DIR
    with open(out / "texts.pkl", "wb") as f:
        pickle.dump(texts, f)
    print(f"  texts.pkl    ({len(texts):,} chunks)")


def load_index_with_texts(
    index_dir: Path | str | None = None,
) -> tuple[faiss.IndexFlatIP, list[dict], list[str]]:
    """
    Load the FAISS index, metadata, and raw chunk texts from disk.

    Returns
    -------
    index    : faiss.IndexFlatIP
    metadata : list[dict]
    texts    : list[str]
    """
    index, metadata = load_index(index_dir)
    src = Path(index_dir) if index_dir else VECTORSTORE_DIR
    texts_path = src / "texts.pkl"

    if not texts_path.exists():
        raise FileNotFoundError(
            f"texts.pkl not found at {texts_path}.\n"
            "Re-run build_index() (it now saves texts automatically)."
        )

    with open(texts_path, "rb") as f:
        texts = pickle.load(f)

    print(f"[Vectorstore] Loaded {len(texts):,} chunk texts from {src}")
    return index, metadata, texts
