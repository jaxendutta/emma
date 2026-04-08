"""
emma.retrieval
--------------
End-to-end RAG pipeline for EMMA.

Pipeline stages
---------------
1. NER (SpaCy en_core_sci_md)
   Extract clinical entities from the raw query. On clinical vignettes,
   query the FAISS index with the extracted entity string to avoid
   embedding dilution from incidental language.

2. FAISS retrieval (src.vectorstore.search)
   Embed the (possibly rewritten) query and retrieve top-k textbook chunks
   with confidence bands.

3. Specialty classification (src.classify.predict_specialty)
   Classify the raw query into one of 19 medical specialties for prompt context.

4. Prompt construction
   Assemble a structured prompt with retrieved passages, entities, specialty,
   and confidence-aware hedging instructions for the LLM.

5. HuggingFace inference (transformers + bitsandbytes)
   Load the model in 4-bit (nf4) quantisation and generate the answer.
   Models are defined in config/models.json and loaded on demand.
   Qwen3 thinking mode is enabled via the chat template.

Backend: HuggingFace transformers + bitsandbytes.
All models run in 4-bit nf4 on Colab T4 (~2.5GB VRAM per model).
No Ollama dependency.

Usage
-----
    from src.retrieval import EMMARetriever
    retriever = EMMARetriever.load()                      # loads default model
    result    = retriever.answer("What is anaphylaxis?")
    result    = retriever.answer("...", use_rag=False)    # baseline
    rag, base = retriever.compare("...")                  # side-by-side
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


# ── Config loading ────────────────────────────────────────────────────────────

def _load_models_config() -> dict:
    """Load config/models.json from the repo root."""
    from src.data import REPO_ROOT
    config_path = REPO_ROOT / "config" / "models.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config/models.json not found at {config_path}.\n"
            "This file is the single source of truth for LLM model definitions."
        )
    return json.loads(config_path.read_text())


def get_model_config(model_id: str) -> dict:
    """Return the config dict for a specific model id from models.json."""
    cfg = _load_models_config()
    for m in cfg["models"]:
        if m["id"] == model_id:
            return m
    available = [m["id"] for m in cfg["models"]]
    raise ValueError(
        f"Model '{model_id}' not found in config/models.json.\n"
        f"Available: {available}"
    )

def get_embedding_config(embedding_id: str) -> dict:
    cfg = _load_models_config()
    for m in cfg.get("embeddings_models", []):
        if m["id"] == embedding_id:
            return m
    raise ValueError(f"Embedding '{embedding_id}' not found in config/models.json.")

def _default_embedding_id() -> str:
    cfg = _load_models_config()
    for m in cfg.get("embeddings_models", []):
        if m.get("default_embedding"):
            return m["id"]
    return "qwen3-embeddings-0.6b"

def get_default_model_id() -> str:
    """Return the default model id from models.json."""
    return _load_models_config()["default_model"]


def list_models() -> list[dict]:
    """Return all model configs from models.json."""
    return _load_models_config()["models"]


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_TOP_K   = 5
MIN_CONFIDENCE  = "medium"

ENTITY_LABELS = {
    "DISEASE", "CHEMICAL", "GENE_OR_GENE_PRODUCT",
    "ORGANISM", "CELL_TYPE", "CELL_LINE",
    "DNA", "RNA", "PROTEIN",
}

SYSTEM_PROMPT = (
    "You are EMMA, an Emergency Medicine Mentoring Agent helping medical students "
    "study for the USMLE. You answer questions accurately and concisely, grounded "
    "in the provided textbook passages when available. If you are uncertain, say so "
    "clearly rather than guessing."
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Single retrieved chunk with metadata."""
    rank:       int
    score:      float
    confidence: str
    book:       str
    text:       str
    chunk_idx:  int


@dataclass
class PipelineResult:
    """Full output of one EMMA pipeline call."""
    query:           str
    use_rag:         bool
    model_id:        str
    model_name:      str
    entities:        list[str]
    rewritten_query: str
    specialty:       str
    chunks:          list[RetrievalResult]
    prompt:          str
    answer:          str
    thinking:        str          # Qwen3 thinking trace, empty for other models
    latency_s:       float
    metadata:        dict = field(default_factory=dict)


# ── NER ───────────────────────────────────────────────────────────────────────

def load_ner_model(model_name: str = "en_core_sci_md"):
    """Load SpaCy biomedical NER model."""
    try:
        import spacy
        return spacy.load(model_name)
    except OSError:
        raise OSError(
            f"SpaCy model '{model_name}' not found.\n"
            "Install with:\n"
            "  uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
            "releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz"
        )


def extract_entities(text: str, nlp) -> list[str]:
    """
    Extract clinical entities using SpaCy en_core_sci_md.

    en_core_sci_md uses the BIONLP13CG schema. We filter to the labels
    most useful for medical retrieval. If no filtered entities are found
    (e.g. simple direct questions), falls back to all recognised entities
    rather than returning empty — this avoids silently skipping NER.
    """
    doc  = nlp(text)
    seen = set()
    ents_filtered = []
    ents_all      = []

    for ent in doc.ents:
        if ent.text.lower() not in seen:
            seen.add(ent.text.lower())
            ents_all.append(ent.text)
            if ent.label_ in ENTITY_LABELS:
                ents_filtered.append(ent.text)

    # Prefer filtered; fall back to all recognised entities
    return ents_filtered if ents_filtered else ents_all


def rewrite_query(query: str, entities: list[str]) -> str:
    """
    Replace the full query with extracted entities for FAISS search.
    Prevents clinical vignette language from diluting the embedding.
    Falls back to raw query if no entities found.
    """
    return " ".join(entities) if entities else query


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(
    query:     str,
    chunks:    list[RetrievalResult],
    entities:  list[str],
    specialty: str,
    use_rag:   bool = True,
) -> str:
    """
    Build a structured prompt for the LLM.

    With RAG: includes retrieved passages with confidence annotations and
    a hedging instruction for low-confidence chunks.
    Without RAG: bare question only (baseline condition).
    """
    if not use_rag or not chunks:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    _conf_order  = {"high": 3, "medium": 2, "low": 1, "very_low": 0}
    strong       = [c for c in chunks if _conf_order.get(c.confidence, 0) >= 2]
    weak         = [c for c in chunks if _conf_order.get(c.confidence, 0) == 1]

    ctx_lines = []
    for i, c in enumerate(strong, 1):
        ctx_lines.append(
            f"[Source {i}: {c.book} | confidence: {c.confidence} | score: {c.score:.3f}]\n"
            f"{c.text.strip()}"
        )
    if weak:
        ctx_lines.append(
            f"\n[Note: {len(weak)} additional passage(s) with low confidence -- treat cautiously.]"
        )
        for i, c in enumerate(weak, len(strong) + 1):
            ctx_lines.append(
                f"[Source {i}: {c.book} | confidence: {c.confidence} | score: {c.score:.3f}]\n"
                f"{c.text.strip()}"
            )

    entity_note   = f"Clinical entities identified: {', '.join(entities)}\n" if entities else ""
    specialty_note = f"Predicted specialty: {specialty}\n" if specialty else ""

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{entity_note}"
        f"{specialty_note}"
        f"\nRelevant textbook passages:\n\n"
        + "\n\n".join(ctx_lines)
        + f"\n\nQuestion: {query}\n\n"
        "Using the passages above where relevant, provide a clear and accurate answer. "
        "If a passage is marked low confidence, do not rely on it as the primary source.\n\n"
        "Answer:"
    )


# ── HuggingFace inference ─────────────────────────────────────────────────────

def load_hf_model(model_cfg: dict, hf_token: str | None = None):
    """
    Load a HuggingFace model in 4-bit nf4 quantisation.

    Returns (model, tokenizer).
    Requires bitsandbytes and transformers.
    All models run comfortably on Colab T4 (15GB VRAM) in 4-bit (~2.5GB each).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hf_repo = model_cfg["hf_repo"]
    gated   = model_cfg.get("gated", False)

    if gated and not hf_token:
        raise ValueError(
            f"Model '{model_cfg['id']}' ({hf_repo}) is gated and requires a HF token.\n"
            f"Note: {model_cfg.get('gated_note', '')}\n"
            "Pass hf_token=... to EMMARetriever.load() or set HF_TOKEN env var."
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {model_cfg['name']} ({hf_repo}) in 4-bit nf4...")

    # Explicitly target CUDA if available — device_map="auto" can silently
    # fall back to CPU if bitsandbytes doesn't detect the GPU correctly.
    import torch
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB VRAM)")
    else:
        print("  WARNING: No GPU detected. Inference will be very slow on CPU.")

    tokenizer = AutoTokenizer.from_pretrained(
        hf_repo, token=hf_token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo,
        quantization_config=bnb_config,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        token=hf_token,
        trust_remote_code=True,
    )
    model.eval()

    vram = _vram_used_gb()
    device = next(model.parameters()).device
    print(f"  Loaded on {device}. VRAM used: {vram:.1f} GB")
    if vram < 0.5 and torch.cuda.is_available():
        print("  WARNING: VRAM usage looks too low — model may have loaded on CPU.")
        print("  Try: Runtime -> Disconnect and delete runtime, then reconnect with T4 GPU.")
    return model, tokenizer


def _vram_used_gb() -> float:
    """Return current GPU VRAM usage in GB, or 0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
    except Exception:
        pass
    return 0.0


def _unload_model(model) -> None:
    """Unload a model from GPU memory."""
    try:
        import torch, gc
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def generate_answer(
    prompt:    str,
    model,
    tokenizer,
    model_cfg: dict,
    max_new_tokens: int = 1024,
    temperature:    float = 0.6,  # Qwen3 recommended for thinking mode
) -> tuple[str, str]:
    """
    Generate an answer from a loaded HF model.

    For Qwen3 (thinking=True), builds the chat template with enable_thinking=True
    and strips the <think>...</think> block before returning.

    Returns (answer_text, thinking_text).
    thinking_text is empty for non-thinking models.
    """
    import torch

    is_thinking = model_cfg.get("thinking", False)

    messages = [
        {"role": "user", "content": prompt}
    ]

    if is_thinking:
        # Qwen3: enable thinking mode via chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Extract thinking trace for Qwen3
    thinking = ""
    if is_thinking:
        think_match = re.search(r"<think>(.*?)</think>", generated, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            # Strip thinking block — answer is everything after </think>
            answer = generated[think_match.end():].strip()
        else:
            # Thinking tokens may have been cut off by max_new_tokens.
            # Use the full output as the answer and flag it.
            answer = generated.strip()
            thinking = "[thinking truncated — increase max_new_tokens]"
        return answer, thinking

    return generated.strip(), thinking


# ── Main retriever class ──────────────────────────────────────────────────────

class EMMARetriever:
    """
    End-to-end EMMA RAG pipeline.

    Loads the FAISS vectorstore, specialty classifier, and SpaCy NER once.
    The LLM is loaded on demand per model_id and can be swapped between
    evaluation runs without reloading the other components.

    Parameters
    ----------
    index         : FAISS index
    metadata      : list of chunk metadata dicts
    texts         : list of chunk strings
    emb_model     : sentence-transformers model for embedding queries
    clf_pipeline  : fitted sklearn classifier pipeline
    label_encoder : fitted LabelEncoder
    nlp           : SpaCy NER model
    model_id      : model id from config/models.json
    top_k         : number of chunks to retrieve
    hf_token      : HuggingFace token (required for gated models)
    """

    def __init__(
        self,
        index,
        metadata:      list[dict],
        texts:         list[str],
        emb_model,
        clf_pipeline,
        label_encoder,
        nlp,
        model_id:  str,
        top_k:     int = DEFAULT_TOP_K,
        hf_token:  str | None = None,
    ):
        self.index         = index
        self.metadata      = metadata
        self.texts         = texts
        self.emb_model     = emb_model
        self.clf_pipeline  = clf_pipeline
        self.label_encoder = label_encoder
        self.nlp           = nlp
        self.model_id      = model_id
        self.top_k         = top_k
        self.hf_token      = hf_token

        # Lazy-loaded LLM
        self._model     = None
        self._tokenizer = None
        self._loaded_id = None

    @classmethod
    def load(
        cls,
        model_id:   str | None = None,
        top_k:      int = DEFAULT_TOP_K,
        hf_token:   str | None = None,
        repo_root:  Path | None = None,
    ) -> "EMMARetriever":
        """
        Load all pre-built pipeline components from disk.

        model_id defaults to the value of 'default_model' in config/models.json.
        hf_token is required for gated models (MedGemma, Gemma, Ministral).
        Can also be set via the HF_TOKEN environment variable.
        """
        import os
        from src.data import REPO_ROOT
        from src.vectorstore import BIOMEDICAL_MODEL, embed_texts, load_embedding_model, load_index_with_texts
        import pickle

        root     = repo_root or REPO_ROOT
        token    = hf_token or os.environ.get("HF_TOKEN")
        mid      = model_id or get_default_model_id()

        # Validate model_id exists in config
        get_model_config(mid)

        print("Loading vectorstore...")
        _emb_cfg = get_embedding_config(emb_model_name or _default_embedding_id())
        _vs_path  = root / "models" / "vectorstore" / _emb_cfg["id"]
        index, metadata, texts = load_index_with_texts(_vs_path)

        print("Loading embedding model...")
        emb_model = load_embedding_model(BIOMEDICAL_MODEL)

        print("Loading specialty classifier...")
        clf_path = root / "models" / "classifier" / "tfidf_svm.pkl"
        le_path  = root / "models" / "classifier" / "label_encoder.pkl"
        if not clf_path.exists():
            raise FileNotFoundError(f"{clf_path} not found. Run notebook 02 first.")
        with open(clf_path, "rb") as f:
            clf_pipeline = pickle.load(f)
        with open(le_path, "rb") as f:
            label_encoder = pickle.load(f)

        print("Loading SpaCy NER model...")
        nlp = load_ner_model()

        print(f"EMMA retriever ready. Model: {mid}")
        return cls(
            index=index, metadata=metadata, texts=texts,
            emb_model=emb_model, clf_pipeline=clf_pipeline,
            label_encoder=label_encoder, nlp=nlp,
            model_id=mid, top_k=top_k, hf_token=token,
        )

    def _ensure_model_loaded(self) -> None:
        """Load the LLM if not already loaded, or swap if model_id changed."""
        if self._loaded_id == self.model_id and self._model is not None:
            return
        # Unload previous model
        if self._model is not None:
            print(f"Unloading {self._loaded_id}...")
            _unload_model(self._model)
            self._model     = None
            self._tokenizer = None
        # Load new model
        cfg = get_model_config(self.model_id)
        self._model, self._tokenizer = load_hf_model(cfg, self.hf_token)
        self._loaded_id = self.model_id

    def switch_model(self, model_id: str) -> None:
        """Switch to a different model (lazy — actual load deferred to next call)."""
        get_model_config(model_id)   # validate
        self.model_id = model_id

    def retrieve(
        self,
        query:          str,
        min_confidence: str = MIN_CONFIDENCE,
    ) -> tuple[list[str], str, str, list[RetrievalResult]]:
        """
        Run NER + query rewriting + FAISS retrieval.

        Returns (entities, rewritten_query, specialty, chunks).
        """
        from src.vectorstore import search

        entities        = extract_entities(query, self.nlp)
        rewritten_query = rewrite_query(query, entities)

        specialty = self.label_encoder.inverse_transform(
            self.clf_pipeline.predict([query])
        )[0]

        raw_results = search(
            rewritten_query, self.index, self.metadata,
            self.texts, self.emb_model, k=self.top_k,
        )

        _conf_order = {"high": 3, "medium": 2, "low": 1, "very_low": 0}
        min_level   = _conf_order.get(min_confidence, 1)

        chunks = [
            RetrievalResult(
                rank=r["rank"], score=r["score"], confidence=r["confidence"],
                book=r.get("friendly_name", r.get("book", "Unknown")),
                text=r["text"], chunk_idx=r.get("chunk_idx", -1),
            )
            for r in raw_results
            if _conf_order.get(r["confidence"], 0) >= min_level
        ]

        return entities, rewritten_query, specialty, chunks

    def answer(
        self,
        query:          str,
        use_rag:        bool = True,
        min_confidence: str  = MIN_CONFIDENCE,
        max_new_tokens: int  = 512,
    ) -> PipelineResult:
        """
        Full RAG pipeline: NER -> retrieve -> prompt -> LLM.

        Parameters
        ----------
        query          : raw user question or clinical vignette
        use_rag        : if False, skip retrieval (baseline condition)
        min_confidence : minimum confidence level for retrieved chunks
        max_new_tokens : maximum tokens to generate
        """
        self._ensure_model_loaded()
        model_cfg = get_model_config(self.model_id)

        if use_rag:
            entities, rewritten_query, specialty, chunks = self.retrieve(
                query, min_confidence=min_confidence
            )
        else:
            entities        = []
            rewritten_query = query
            specialty       = self.label_encoder.inverse_transform(
                self.clf_pipeline.predict([query])
            )[0]
            chunks = []

        prompt = build_prompt(
            query=query, chunks=chunks, entities=entities,
            specialty=specialty, use_rag=use_rag,
        )

        t0 = time.time()
        answer_text, thinking_text = generate_answer(
            prompt, self._model, self._tokenizer,
            model_cfg, max_new_tokens=max_new_tokens,
            temperature=0.6,  # Qwen3 recommended for thinking mode
        )
        latency = time.time() - t0

        return PipelineResult(
            query=query, use_rag=use_rag,
            model_id=self.model_id, model_name=model_cfg["name"],
            entities=entities, rewritten_query=rewritten_query,
            specialty=specialty, chunks=chunks,
            prompt=prompt, answer=answer_text, thinking=thinking_text,
            latency_s=round(latency, 2),
            metadata={
                "n_chunks_retrieved": len(chunks),
                "rewrite_applied":    rewritten_query != query,
                "top_score":          chunks[0].score if chunks else None,
                "top_confidence":     chunks[0].confidence if chunks else None,
                "thinking_tokens":    len(thinking_text.split()) if thinking_text else 0,
            },
        )

    def compare(
        self,
        query:          str,
        min_confidence: str = MIN_CONFIDENCE,
    ) -> tuple[PipelineResult, PipelineResult]:
        """
        Run the same query with and without RAG.
        Returns (rag_result, baseline_result).
        """
        rag      = self.answer(query, use_rag=True,  min_confidence=min_confidence)
        baseline = self.answer(query, use_rag=False, min_confidence=min_confidence)
        return rag, baseline