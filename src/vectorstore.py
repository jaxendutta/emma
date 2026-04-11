"""
emma.vectorstore
────────────────
Builds and queries the FAISS vector index over the 18 medical textbooks.

Two main responsibilities:
  1. BUILD  – chunk all textbooks, embed with Qwen3-Embedding-0.6B,
               persist the FAISS index + metadata + texts to disk.
  2. QUERY  – load the index and retrieve the top-k most relevant chunks
               for any natural-language query.

Typical usage
─────────────
Build once (run from notebook 01_vectorstore_build.ipynb):

    from src.vectorstore import build_index
    build_index()          # writes to models/vectorstore/

Query at inference time:

    from src.vectorstore import load_index_with_texts, load_embedding_model, search
    index, metadata, texts = load_index_with_texts()
    model = load_embedding_model()
    results = search("What is the mechanism of septic shock?", index, metadata, texts, model, k=5)
    for r in results:
        print(r["score"], r["friendly_name"], r["text"][:200])
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Repo root detection ───────────────────────────────────────────────────────

def _find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not find repo root (no pyproject.toml found).")

REPO_ROOT = _find_repo_root()

# ── Default paths ─────────────────────────────────────────────────────────────

TEXTBOOK_DIR    = REPO_ROOT / "data" / "MedQA-USMLE" / "textbooks" / "en"
VECTORSTORE_DIR = REPO_ROOT / "models" / "vectorstore"

# ── Embedding model ───────────────────────────────────────────────────────────
def _default_embedding_model() -> str:
    """Load default embedding model HF repo from config/models.json."""
    import json
    cfg_path = REPO_ROOT / "config" / "models.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        for m in cfg.get("embeddings_models", []):
            if m.get("default_embedding"):
                return m["hf_repo"]
    return "Qwen/Qwen3-Embedding-0.6B"  # fallback

BIOMEDICAL_MODEL  = _default_embedding_model()
FALLBACK_MODEL    = "all-MiniLM-L6-v2"

# Instruction prefix for queries only — documents are embedded WITHOUT prefix.
# Prepending this to queries gives a free 1-5% retrieval boost (Qwen3 recommendation).
QUERY_INSTRUCTION = (
    "Instruct: Given a medical question, retrieve relevant textbook passages "
    "that answer it\nQuery: "
)

# ── Chunking parameters ───────────────────────────────────────────────────────

CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 50    # words of overlap between consecutive chunks

# ── Friendly book names ───────────────────────────────────────────────────────

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

    Each chunk is ~chunk_size words, with `overlap` words shared
    with the next chunk to preserve context across boundaries.
    """
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
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
    texts    : list[str]       — all chunk strings in order
    metadata : list[dict]      — parallel list, one dict per chunk:
                                 {book, friendly_name, chunk_idx}
    """
    base = Path(textbook_dir) if textbook_dir else TEXTBOOK_DIR
    texts: list[str] = []
    metadata: list[dict] = []

    txt_files = sorted(base.glob("*.txt"))
    for path in tqdm(
        txt_files,
        desc="Chunking textbooks",
        unit="book",
        bar_format="{l_bar}{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}]",
    ):
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
            })

    print(f"\n[Vectorstore] Total chunks: {len(texts):,} across "
          f"{len(set(m['book'] for m in metadata))} books")
    return texts, metadata


# ── Embedding ─────────────────────────────────────────────────────────────────

def _resolve_embedding_model(model_name: str | None) -> str:
    """
    Resolve a model identifier to a HuggingFace repo path.

    Accepts either:
    - An HF repo path directly (contains '/'), e.g. 'Qwen/Qwen3-Embedding-0.6B'
    - A models.json id, e.g. 'octen-embedding-0.6b'
    - None → returns BIOMEDICAL_MODEL (the default from models.json)
    """
    if model_name is None:
        return BIOMEDICAL_MODEL
    if "/" in model_name:
        return model_name  # already an HF repo path
    # Look up by id in models.json
    cfg_path = REPO_ROOT / "config" / "models.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for m in cfg.get("embeddings_models", []):
            if m.get("id") == model_name:
                return m["hf_repo"]
    return model_name  # fall through — let sentence-transformers try it


def load_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    """
    Load an embedding model by HF repo path or models.json id.

    Accepts either a full HF repo (e.g. 'Qwen/Qwen3-Embedding-0.6B') or a
    short id from config/models.json (e.g. 'octen-embedding-0.6b').

    Device priority: CUDA → MPS (Apple Silicon) → CPU.

    On CUDA: loads in float16 with device_map='cuda' so weights stream
    directly onto GPU in fp16 — avoids the 2× peak VRAM of loading in
    fp32 then moving. Checks free VRAM before attempting to load.

    On MPS: loads with device='mps' in float16.

    On CPU: loads in float32, no special handling needed.
    """
    import torch

    target = _resolve_embedding_model(model_name)

    try:
        is_cuda = torch.cuda.is_available()
        is_mps  = (not is_cuda
                   and hasattr(torch.backends, "mps")
                   and torch.backends.mps.is_available())

        if is_cuda:
            torch.cuda.empty_cache()
            free_gb  = torch.cuda.mem_get_info()[0] / 1024**3
            total_gb = torch.cuda.mem_get_info()[1] / 1024**3
            print(f"[Vectorstore] GPU memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
            if free_gb < 1.5:
                raise RuntimeError(
                    f"Only {free_gb:.1f} GB free on GPU. "
                    "Run the GPU cleanup cell, or do Runtime → Restart session "
                    "and re-run setup cells before building."
                )
            elif free_gb < 4.0:
                print(f"[Vectorstore] ⚠️  Only {free_gb:.1f} GB free — watch for OOM")

        if is_cuda:
            model = SentenceTransformer(
                target,
                model_kwargs={
                    "torch_dtype": torch.float16,
                    "device_map": "cuda",
                },
                tokenizer_kwargs={"padding_side": "left"},
            )
            device = "cuda"
        elif is_mps:
            model = SentenceTransformer(
                target,
                model_kwargs={"torch_dtype": torch.float16},
                device="mps",
                tokenizer_kwargs={"padding_side": "left"},
            )
            device = "mps"
        else:
            model = SentenceTransformer(target)
            device = "cpu"

        dtype = next(model.parameters()).dtype
        print(f"[Vectorstore] Embedding model: {target}  (dtype={dtype}, device={device})")
        return model

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "Only" in str(e):
            raise   # re-raise OOM errors with the clear message
        print(f"[Vectorstore] Could not load {target}: {e}")
        print(f"[Vectorstore] Falling back to {FALLBACK_MODEL}")
        return SentenceTransformer(FALLBACK_MODEL)
    except Exception as e:
        print(f"[Vectorstore] Could not load {target}: {e}")
        print(f"[Vectorstore] Falling back to {FALLBACK_MODEL}")
        return SentenceTransformer(FALLBACK_MODEL)


def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed a list of text strings → float32 numpy array of shape (N, dim).

    Encodes in explicit batches with our own tqdm bar so we control
    formatting (comma-separated counts, 'chunk/s' unit).
    sentence-transformers' internal bar is suppressed.
    """
    import math

    all_embeddings = []
    pbar = tqdm(
        total=len(texts),
        desc="Embedding chunks",
        unit="chunk",
        bar_format="{l_bar}{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}]",
        disable=not show_progress,
    )

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,     # suppress internal bar — we own it
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        all_embeddings.append(emb.astype(np.float32))
        pbar.update(len(batch))

    pbar.close()
    return np.vstack(all_embeddings)


# ── FAISS index ───────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP (exact cosine) from L2-normalised embeddings.

    IndexFlatIP is exact search — no approximation.
    Fine for ~40k chunks; would need IVF or HNSW at >1M.
    """
    if embeddings.ndim < 2 or embeddings.shape[0] == 0:
        raise ValueError(
            f"embeddings array is empty (shape={embeddings.shape}). "
            "No textbook chunks found — check TEXTBOOK_DIR exists "
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
    Persist the FAISS index, metadata, and config to disk.

    Writes: index.faiss, metadata.pkl, config.json
    """
    out = Path(out_dir) if out_dir else VECTORSTORE_DIR
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out / "index.faiss"))

    with open(out / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    config = {
        "model_name":    model_name,
        "chunk_size":    CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_chunks":    len(metadata),
        "num_books":     len(set(m["book"] for m in metadata)),
        "embedding_dim": index.d,
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[Vectorstore] Saved to {out}")
    print(f"  index.faiss  ({index.ntotal:,} vectors)")
    print(f"  metadata.pkl ({len(metadata):,} entries)")
    print(f"  config.json  {config}")


def save_index_with_texts(
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    texts: list[str],
    save_dir: Path | str | None = None,
    embedding_model_id: str = "",
) -> None:
    """
    Like save_index(), but also saves chunk texts to texts.pkl
    so they don't need to be re-generated on every load.
    """
    if embedding_model_id:
        save_dir = Path(save_dir).parent / embedding_model_id
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_index(index, metadata, out_dir=save_dir, model_name=embedding_model_id)
    out = Path(save_dir) if save_dir else VECTORSTORE_DIR
    with open(out / "texts.pkl", "wb") as f:
        pickle.dump(texts, f)
    print(f"  texts.pkl    ({len(texts):,} chunks)")


def load_index(
    index_dir: Path | str | None = None,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Load the FAISS index and metadata from disk.
    """
    src = VECTORSTORE_DIR / index_dir if index_dir else VECTORSTORE_DIR
    index_path = src / "index.faiss"
    meta_path  = src / "metadata.pkl"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}.\n"
            "Run notebook 01_vectorstore_build.ipynb to generate it."
        )

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"[Vectorstore] Loaded index: {index.ntotal:,} vectors from {src}")
    return index, metadata


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
    src = VECTORSTORE_DIR / index_dir if index_dir else VECTORSTORE_DIR
    texts_path = src / "texts.pkl"

    if not texts_path.exists():
        raise FileNotFoundError(
            f"texts.pkl not found at {texts_path}.\n"
            "Re-run notebook 01 to regenerate."
        )

    with open(texts_path, "rb") as f:
        texts = pickle.load(f)

    print(f"[Vectorstore] Loaded {len(texts):,} chunk texts from {src}")
    return index, metadata, texts


# ── Search ────────────────────────────────────────────────────────────────────

# Score bands for retrieval confidence
SCORE_HIGH   = 0.70   # strong match — use freely
SCORE_MEDIUM = 0.55   # acceptable — flag to LLM as uncertain
SCORE_LOW    = 0.40   # weak — include with warning or skip


def score_band(score: float) -> str:
    """Categorise a cosine similarity score into a confidence band."""
    if score >= SCORE_HIGH:
        return "high"
    elif score >= SCORE_MEDIUM:
        return "medium"
    elif score >= SCORE_LOW:
        return "low"
    else:
        return "very_low"


def search(
    query: str,
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    texts: list[str],
    model: SentenceTransformer,
    k: int = 5,
    min_score: float = SCORE_LOW,
) -> list[dict]:
    """
    Retrieve the top-k most relevant textbook chunks for a query.

    The query is prefixed with QUERY_INSTRUCTION before embedding
    (Qwen3-Embedding recommendation for 1-5% retrieval gain).
    Documents were embedded WITHOUT any prefix.

    Parameters
    ----------
    query     : natural-language question or clinical vignette
    index     : loaded FAISS index
    metadata  : parallel metadata list
    texts     : parallel chunk strings
    model     : the same embedding model used at build time
    k         : number of candidates to retrieve before filtering
    min_score : minimum cosine similarity to include in results.
                Chunks below this threshold are silently dropped.
                Default 0.40 — set to 0.0 to return all k results.

    Returns
    -------
    List of dicts (may be shorter than k if low-scoring chunks are filtered):
        rank, score, confidence, book, friendly_name, chunk_idx, text

    Confidence bands (cosine similarity):
        high      ≥ 0.70  — strong semantic match, use freely
        medium    ≥ 0.55  — acceptable, flag as uncertain in prompts
        low       ≥ 0.40  — weak, include cautiously
        very_low  < 0.40  — filtered out by default

    Notes
    -----
    Clinical vignettes (scenario-style questions) tend to score lower than
    direct medical questions because incidental words ("58-year-old man",
    "presents with") dilute the embedding. The RAG pipeline handles this by
    running SpaCy NER to extract key entities before querying, then querying
    with the entity string rather than the raw vignette. This function is
    the low-level retriever — entity extraction happens upstream.
    """
    prefixed = QUERY_INSTRUCTION + query
    query_vec = model.encode(
        [prefixed],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(query_vec, k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:
            continue
        if float(score) < min_score:
            continue   # drop chunks below threshold
        meta = metadata[idx]
        results.append({
            "rank":          rank + 1,
            "score":         float(score),
            "confidence":    score_band(float(score)),
            "book":          meta["book"],
            "friendly_name": meta["friendly_name"],
            "chunk_idx":     meta["chunk_idx"],
            "text":          texts[idx],
        })
    return results


# ── Top-level build pipeline ──────────────────────────────────────────────────

def build_index(
    textbook_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    model_name: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    batch_size: int = 32,
) -> None:
    """
    Full pipeline: chunk → embed → index → save.

    Called from notebook 01_vectorstore_build.ipynb.
    Runtime: ~60 min on Colab T4 at batch_size=16 (Qwen3-0.6B).
    """
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    print("=" * 60)
    print(" EMMA vectorstore build")
    print("=" * 60)

    texts, metadata = chunk_all_textbooks(
        textbook_dir=textbook_dir,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    model = load_embedding_model(model_name)
    print(f"\n[Vectorstore] Embedding {len(texts):,} chunks (batch_size={batch_size})…")
    embeddings = embed_texts(texts, model, batch_size=batch_size)

    index = build_faiss_index(embeddings)

    save_index_with_texts(
        index=index,
        metadata=metadata,
        texts=texts,
        out_dir=out_dir,
        embedding_model_id=model_name or BIOMEDICAL_MODEL,
    )

    print("\n[Vectorstore] Build complete.")
