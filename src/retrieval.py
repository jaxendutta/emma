"""
emma.retrieval
--------------
End-to-end RAG pipeline for EMMA.

Pipeline stages
---------------
1. NER (SpaCy en_ner_bc5cdr_md: DISEASE + CHEMICAL entities from BC5CDR corpus)
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

5. LLM inference — Ollama-first, HuggingFace fallback
   Try Ollama first (fast, no GPU required, no model loading overhead).
   If Ollama is unavailable or the model is not pulled, fall back to
   HuggingFace transformers + bitsandbytes (4-bit nf4 on Colab T4).
   Models are defined in config/models.json.

Backend priority:
  1. Ollama  (ollama_tag field in models.json, localhost:11434)
  2. HuggingFace transformers + bitsandbytes 4-bit nf4 (hf_repo field)

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


def get_default_embedding_id() -> str:
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

NER_MODEL = "en_ner_bc5cdr_md"

# en_ner_bc5cdr_md entity labels (BC5CDR corpus: 1,500 PubMed articles,
# 4,409 chemicals, 5,818 diseases, 3,116 chemical-disease interactions).
# These are the ONLY two labels the bc5cdr model produces.
# en_core_sci_md (generic mention detector) outputs a single ENTITY label —
# it is NOT used for typed extraction; only for dependency parsing in NB04b.
ENTITY_LABELS = {"DISEASE", "CHEMICAL"}

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
    backend:         str          # "ollama" or "hf"
    metadata:        dict = field(default_factory=dict)


# ── NER ───────────────────────────────────────────────────────────────────────

def load_ner_model(model_name: str = NER_MODEL):
    """
    Load SpaCy biomedical NER model.
 
    Default: en_ner_bc5cdr_md — trained on BC5CDR corpus, labels: DISEASE, CHEMICAL.
    This is the correct model for typed clinical entity extraction and query rewriting.
 
    en_core_sci_md is a generic mention detector (single ENTITY label) and is
    unsuitable for typed NER. It is used only for dependency parsing in NB04b.
 
    Install the default model:
        pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
    """
    import spacy
    try:
        return spacy.load(model_name)
    except OSError:
        raise OSError(
            f"⚠ SpaCy model '{model_name}' not found!\n"
            f"Install with:\n"
            f"  uv add https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{model_name}-0.5.4.tar.gz"
            f"  uv sync"
        )

def extract_entities(text: str, nlp) -> list[str]:
    """Run NER and return unique entity strings for query rewriting."""
    doc = nlp(text)
    seen = set()
    entities = []
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS and ent.text.lower() not in seen:
            seen.add(ent.text.lower())
            entities.append(ent.text)
    return entities


def rewrite_query(query: str, entities: list[str]) -> str:
    """
    Replace a clinical vignette with its extracted entities for FAISS retrieval.
    If no entities were found, return the original query unchanged.
    """
    if not entities:
        return query
    return " ".join(entities)


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(
    query:     str,
    chunks:    list[RetrievalResult],
    entities:  list[str],
    specialty: str,
    use_rag:   bool = True,
) -> str:
    """Build the structured prompt sent to the LLM."""
    if not use_rag or not chunks:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    strong = [c for c in chunks if c.confidence in ("high", "medium")]
    weak   = [c for c in chunks if c.confidence not in ("high", "medium")]

    ctx_lines: list[str] = []
    for i, c in enumerate(strong, 1):
        ctx_lines.append(
            f"[Source {i}: {c.book} | confidence: {c.confidence} | score: {c.score:.3f}]\n"
            f"{c.text.strip()}"
        )
    for i, c in enumerate(weak, len(strong) + 1):
        ctx_lines.append(
            f"[Source {i}: {c.book} | confidence: {c.confidence} | score: {c.score:.3f}]\n"
            f"{c.text.strip()}"
        )

    entity_note    = f"Clinical entities identified: {', '.join(entities)}\n" if entities else ""
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


# ── Ollama inference ──────────────────────────────────────────────────────────

def _ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        import requests
        r = requests.get(f"{base_url}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _ollama_model_pulled(tag: str, base_url: str = "http://localhost:11434") -> bool:
    """Return True if the given Ollama tag is already pulled locally."""
    try:
        import requests
        r = requests.get(f"{base_url}/api/tags", timeout=2)
        if r.status_code != 200:
            return False
        models = r.json().get("models", [])
        pulled = {m["name"] for m in models}
        # Ollama names can be bare ("qwen3:4b") or include digest suffix
        return any(tag == p or p.startswith(tag + "@") for p in pulled)
    except Exception:
        return False


def warmup_ollama(tag: str, base_url: str = "http://localhost:11434") -> bool:
    """
    Send a minimal dummy request to Ollama to force the model into memory.

    Ollama loads model weights on the first inference request, which can take
    30-90 seconds for a 4B model. Calling this at server startup ensures the
    model is hot before any user request arrives, so real inference takes
    only 5-15 seconds instead of 2+ minutes.

    Returns True if the warmup succeeded, False if Ollama is unreachable
    or the request failed (caller can decide whether to fall back to HF).
    """
    import requests
    print(f"> Warming up Ollama model '{tag}' for inference...")
    try:
        r = requests.post(
            f"{base_url}/api/chat",
            json={
                "model":   tag,
                "messages": [{"role": "user", "content": "hi"}],
                "stream":  False,
                "options": {"num_predict": 1},   # generate exactly 1 token — fastest possible
            },
            timeout=120,   # allow up to 2 min for cold model load at warmup time only
        )
        r.raise_for_status()
        print(f"  Ollama warmup complete for '{tag}'.")
        return True
    except Exception as exc:
        print(f"  Ollama warmup failed ({exc}) — will fall back to HuggingFace.")
        return False


def generate_answer_ollama(
    prompt:    str,
    model_cfg: dict,
    think:     bool = False,
    base_url:  str  = "http://localhost:11434",
) -> tuple[str, str]:
    """
    Generate an answer via the Ollama HTTP API.

    Returns (answer_text, thinking_text).
    For Qwen3 thinking models, the <think>...</think> block is stripped
    from the answer and returned separately.
    """
    import requests

    tag        = model_cfg["ollama_tag"]
    is_thinking = model_cfg.get("thinking", False) and think

    # Ollama chat payload
    payload = {
        "model": tag,
        "messages": [
            {
                "role": "system",
                "content": "You are EMMA, an Emergency Medicine Mentoring Agent. You chat with medical students via text message. CRITICAL RULES: 1. Keep it brief: Never write more than 2 or 3 short sentences. 2. No Markdown: Do not use asterisks, bolding, bullet points, or headers. Use plain text only. 3. Get straight to the point: Never use introductory filler. 4. Stick to the prompt."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.6 if is_thinking else 0.7,
            "num_predict": 512,
        },
    }

    # Qwen3 thinking mode: pass think flag via options
    if is_thinking:
        payload["think"] = True

    r = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    r.raise_for_status()

    generated = r.json()["message"]["content"].strip()

    # Strip <tool_call>...<tool_call> block if present (Qwen3 with thinking=True)
    thinking = ""
    if is_thinking:
        think_match = re.search(r"<tool_call>(.*?)</tool_call>", generated, re.DOTALL)
        if think_match:
            thinking  = think_match.group(1).strip()
            generated = generated[think_match.end():].strip()

    return generated, thinking


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

    print(f"> Loading {model_cfg['name']} ({hf_repo}) in 4-bit nf4...")

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

    vram   = _vram_used_gb()
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
        import torch
        import gc
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def generate_answer_hf(
    prompt:    str,
    model,
    tokenizer,
    model_cfg: dict,
    max_new_tokens: int   = 512,
    temperature:    float = 0.6,
    think:          bool  = False,
) -> tuple[str, str]:
    """
    Generate an answer from a loaded HF model.

    For Qwen3 (thinking=True and think=True), builds the chat template with
    enable_thinking=True and strips the <think>...</think> block before returning.

    Returns (answer_text, thinking_text).
    thinking_text is empty when think=False or for non-thinking models.
    """
    import torch

    is_thinking = model_cfg.get("thinking", False) and think

    messages = [{"role": "user", "content": prompt}]

    if is_thinking:
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

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    thinking = ""
    if is_thinking:
        think_match = re.search(r"<think>(.*?)</think>", generated, re.DOTALL)
        if think_match:
            thinking  = think_match.group(1).strip()
            generated = generated[think_match.end():].strip()
        else:
            thinking  = "[thinking truncated — increase max_new_tokens]"
        return generated.strip(), thinking

    return generated.strip(), thinking


# ── Unified inference: Ollama first, HF fallback ─────────────────────────────

_inference_warned: set[str] = set()

def generate_answer(
    prompt:     str,
    model_cfg:  dict,
    hf_model=None,
    hf_tokenizer=None,
    hf_token:   str | None = None,
    think:      bool = False,
    ollama_url: str  = "http://localhost:11434",
) -> tuple[str, str, str]:
    """
    Unified inference entry point.

    Priority
    --------
    1. Ollama — tried when ollama_tag is present in model_cfg AND Ollama is
       reachable AND the model tag is already pulled locally.
       Fast, no VRAM cost, no model loading time.
    2. HuggingFace transformers — used if Ollama is unavailable, the model
       is not pulled, or inference raises an exception.
       Requires hf_model and hf_tokenizer to be pre-loaded.

    Returns (answer_text, thinking_text, backend)
    backend is "ollama" or "hf" — recorded in PipelineResult.metadata.
    """
    ollama_tag = model_cfg.get("ollama_tag")

    # ── Try Ollama ────────────────────────────────────────────────────────────
    if ollama_tag:
        if not _ollama_available(ollama_url):
            warn_key = f"unavailable:{ollama_url}"
            if warn_key not in _inference_warned:
                print(f"  [inference] Ollama not reachable at {ollama_url} -> falling back to HF")
                _inference_warned.add(warn_key)
        elif not _ollama_model_pulled(ollama_tag, ollama_url):
            warn_key = f"unpulled:{ollama_tag}"
            if warn_key not in _inference_warned:
                print(f"  [inference] Ollama model '{ollama_tag}' not pulled -> falling back to HF")
                _inference_warned.add(warn_key)
        else:
            try:
                print(f"  [inference] Using Ollama ({ollama_tag})")
                answer, thinking = generate_answer_ollama(
                    prompt, model_cfg, think=think, base_url=ollama_url
                )
                return answer, thinking, "ollama"
            except Exception as exc:
                warn_key = f"failed:{ollama_tag}"
                if warn_key not in _inference_warned:
                    print(f"  [inference] Ollama failed ({exc}) -> falling back to HF")
                    _inference_warned.add(warn_key)

    # ── HuggingFace fallback ──────────────────────────────────────────────────
    if hf_model is None or hf_tokenizer is None:
        raise RuntimeError(
            "Ollama is unavailable and no HF model is loaded. "
            "Call _ensure_model_loaded() before inference."
        )
    warn_key = f"hf:{model_cfg['hf_repo']}"
    if warn_key not in _inference_warned:
        print(f"  [inference] Using HuggingFace ({model_cfg['hf_repo']})")
        _inference_warned.add(warn_key)

    answer, thinking = generate_answer_hf(
        prompt, hf_model, hf_tokenizer, model_cfg, think=think
    )
    return answer, thinking, "hf"


# ── Main retriever class ──────────────────────────────────────────────────────

class EMMARetriever:
    """
    End-to-end EMMA RAG pipeline.

    Loads the FAISS vectorstore, specialty classifier, and SpaCy NER once.
    The LLM is resolved at inference time using Ollama-first / HF-fallback:

      - If Ollama is running and the model tag is pulled locally, Ollama is used.
        This is the fast path — no GPU loading, sub-second startup.
      - If Ollama is unavailable, the HF model is loaded lazily on first call
        and cached for subsequent calls.

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
    ollama_url    : Ollama server base URL (default: http://localhost:11434)
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
        model_id:   str,
        top_k:      int = DEFAULT_TOP_K,
        hf_token:   str | None = None,
        ollama_url: str = "http://localhost:11434",
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
        self.ollama_url    = ollama_url

        self._inference_warned: set[str] = set()

        # Lazy-loaded HF LLM (only populated when Ollama is unavailable)
        self._hf_model     = None
        self._hf_tokenizer = None
        self._hf_loaded_id = None

    @classmethod
    def load(
        cls,
        model_id:       str | None = None,
        top_k:          int = DEFAULT_TOP_K,
        hf_token:       str | None = None,
        repo_root:      Path | None = None,
        emb_model_name: str | None = None,
        ollama_url:     str = "http://localhost:11434",
    ) -> "EMMARetriever":
        """
        Load all pre-built pipeline components from disk.

        model_id defaults to the value of 'default_model' in config/models.json.
        hf_token is required for gated models (MedGemma, Gemma, Ministral).
        Can also be set via the HF_TOKEN environment variable.
        """
        import os
        import pickle
        from src.data import REPO_ROOT
        from src.vectorstore import load_embedding_model, load_index_with_texts

        root  = repo_root or REPO_ROOT
        token = hf_token or os.environ.get("HF_TOKEN")
        mid   = model_id or get_default_model_id()

        # Validate model_id exists in config
        get_model_config(mid)

        print(f"> Loading vectorstore ({mid})...")
        _emb_cfg = get_embedding_config(emb_model_name or get_default_embedding_id())
        _vs_path = root / "models" / "vectorstore" / _emb_cfg["id"]
        index, metadata, texts = load_index_with_texts(_vs_path)

        print(f"> Loading embedding model ({_emb_cfg['hf_repo']})...")
        emb_model = load_embedding_model(_emb_cfg["hf_repo"])

        print(f"> Loading specialty classifier ({NER_MODEL})...")
        clf_path = root / "models" / "classifier" / "tfidf_svm.pkl"
        le_path  = root / "models" / "classifier" / "label_encoder.pkl"
        if not clf_path.exists():
            raise FileNotFoundError(f"{clf_path} not found. Run notebook 02 first.")
        with open(clf_path, "rb") as f:
            clf_pipeline = pickle.load(f)
        with open(le_path, "rb") as f:
            label_encoder = pickle.load(f)

        print(f"> Loading SpaCy NER model ({NER_MODEL})...")
        nlp = load_ner_model()

        # ── Backend preparation ───────────────────────────────────────────────
        cfg        = get_model_config(mid)
        ollama_tag = cfg.get("ollama_tag")
        retriever  = cls(
            index=index, metadata=metadata, texts=texts,
            emb_model=emb_model, clf_pipeline=clf_pipeline,
            label_encoder=label_encoder, nlp=nlp,
            model_id=mid, top_k=top_k, hf_token=token,
            ollama_url=ollama_url,
        )

        if ollama_tag and _ollama_available(ollama_url) and _ollama_model_pulled(ollama_tag, ollama_url):
            # Ollama path: warm up so the model is hot in memory at first request.
            # This is the blocking call that takes 30-90 s — do it at startup,
            # not during a live user request.
            ollama_ok = warmup_ollama(ollama_tag, ollama_url)
            if ollama_ok:
                print(f"EMMA retriever ready. Primary: Ollama ({ollama_tag}), Fallback: HuggingFace")
            else:
                print(f"EMMA retriever ready. Ollama warmup failed — primary: HuggingFace ({cfg['hf_repo']})")
        else:
            if ollama_tag:
                reason = "not running" if not _ollama_available(ollama_url) else "model not pulled"
                print(f"EMMA retriever ready. Ollama {reason} — primary: HuggingFace ({cfg['hf_repo']})")
                print(f"  To use Ollama: ollama pull {ollama_tag}")
            else:
                print(f"EMMA retriever ready. Backend: HuggingFace ({cfg['hf_repo']})")

        # Always pre-load HF model as a genuine fallback.
        # Even when Ollama is the primary backend, it can time out mid-inference.
        # We load HF eagerly here so the fallback path never fails with
        # "no HF model is loaded". Cost: ~2.5 GB VRAM, loaded once at startup.
        print("> Pre-loading HuggingFace model as fallback...")
        retriever._ensure_hf_model_loaded()

        return retriever

    def _ensure_hf_model_loaded(self) -> None:
        """
        Load the HF model if not already loaded, or swap if model_id changed.
        Called lazily — only when Ollama is unavailable.
        """
        if self._hf_loaded_id == self.model_id and self._hf_model is not None:
            return
        if self._hf_model is not None:
            print(f"> Unloading HF model {self._hf_loaded_id}...")
            _unload_model(self._hf_model)
            self._hf_model     = None
            self._hf_tokenizer = None
        cfg = get_model_config(self.model_id)
        self._hf_model, self._hf_tokenizer = load_hf_model(cfg, self.hf_token)
        self._hf_loaded_id = self.model_id

    # Keep the old name as an alias for any existing callers (notebooks, api.py)
    def _ensure_model_loaded(self) -> None:
        """Alias for _ensure_hf_model_loaded — called by the lifespan pre-warmer."""
        self._ensure_hf_model_loaded()

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
        think:          bool = False,
    ) -> PipelineResult:
        """
        Full RAG pipeline: NER -> retrieve -> prompt -> LLM.

        Inference tries Ollama first; falls back to HF automatically.

        Parameters
        ----------
        query          : raw user question or clinical vignette
        use_rag        : if False, skip retrieval (baseline condition)
        min_confidence : minimum confidence level for retrieved chunks
        max_new_tokens : maximum tokens to generate (HF only; Ollama uses num_predict)
        think          : enable Qwen3 chain-of-thought (slow — avoid on webhook)
        """
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

        # Always ensure HF is loaded as a genuine fallback.
        # Even when Ollama is available, it can time out mid-inference.
        # Without a loaded HF model, the fallback raises RuntimeError.
        # Cost: ~2.5 GB VRAM on T4, loaded once and reused across all calls.
        self._ensure_hf_model_loaded()

        t0 = time.time()
        answer_text, thinking_text, backend = generate_answer(
            prompt      = prompt,
            model_cfg   = model_cfg,
            hf_model    = self._hf_model,
            hf_tokenizer= self._hf_tokenizer,
            hf_token    = self.hf_token,
            think       = think,
            ollama_url  = self.ollama_url,
        )
        latency = time.time() - t0

        return PipelineResult(
            query=query, use_rag=use_rag,
            model_id=self.model_id, model_name=model_cfg["name"],
            entities=entities, rewritten_query=rewritten_query,
            specialty=specialty, chunks=chunks,
            prompt=prompt, answer=answer_text, thinking=thinking_text,
            latency_s=round(latency, 2),
            backend=backend,
            metadata={
                "n_chunks_retrieved": len(chunks),
                "rewrite_applied":    rewritten_query != query,
                "top_score":          chunks[0].score if chunks else None,
                "top_confidence":     chunks[0].confidence if chunks else None,
                "thinking_tokens":    len(thinking_text.split()) if thinking_text else 0,
                "backend":            backend,
            },
        )

    def compare(
        self,
        query:          str,
        min_confidence: str  = MIN_CONFIDENCE,
        think:          bool = False,
    ) -> tuple[PipelineResult, PipelineResult]:
        """
        Run the same query with and without RAG.
        Returns (rag_result, baseline_result).
        """
        rag      = self.answer(query, use_rag=True,  min_confidence=min_confidence, think=think)
        baseline = self.answer(query, use_rag=False, min_confidence=min_confidence, think=think)
        return rag, baseline