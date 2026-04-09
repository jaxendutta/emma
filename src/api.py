"""
emma.api
--------
FastAPI webhook for Dialogflow ES + Facebook Messenger.

Architecture
------------
The Dialogflow ES webhook has a hard 5-second deadline enforced by Google.
LLM inference (Ollama or HF) takes 8-20+ seconds. These two facts are
incompatible — no timeout trick resolves it.

The solution is the TWO-TURN ASYNC PATTERN:

  Turn 1 — User asks a question
    The webhook immediately returns an acknowledgment ("Looking that up…")
    in < 200 ms. In parallel, RAG is fired as a true background task and
    the result is stored in _pending keyed by session_id.

  Turn 2 — User sends any follow-up (prompted by the acknowledgment)
    The webhook checks _pending first. If a result is ready, it delivers
    the RAG answer. If RAG is still running, it says "still working…".
    If no pending entry exists, normal intent routing continues.

This gives users real textbook-grounded RAG answers through Messenger
with zero timeouts and zero fake static responses.

Endpoint summary
----------------
  POST /webhook   Dialogflow ES — two-turn async RAG
  POST /chat      Built-in EMMA widget — full RAG, no deadline
  POST /query     Developer testing — full RAG, no deadline
  GET  /health    Service health + backend info
  GET  /conditions  Evaluation-domain condition listing

Environment variables
---------------------
  EMMA_USE_RAG=true       activate RAG pipeline
  EMMA_MODEL_ID           override default model (optional)
  EMMA_OLLAMA_URL         Ollama server URL (default: http://localhost:11434)
  EMMA_PENDING_TTL        seconds to keep pending RAG results (default: 120)
"""

from __future__ import annotations

import asyncio
import logging
import re
import random
import os
import time as _time
import json as _json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger("emma.api")

# ── Feature flags & tunables ──────────────────────────────────────────────────

RAG_ENABLED    = os.environ.get("EMMA_USE_RAG", "false").lower() == "true"
OLLAMA_URL     = os.environ.get("EMMA_OLLAMA_URL", "http://localhost:11434")
_PENDING_TTL   = float(os.environ.get("EMMA_PENDING_TTL", "120"))

# ── Thread pool ───────────────────────────────────────────────────────────────
# Both Ollama HTTP calls and HF model.generate() are synchronous/blocking.
# We run them in a thread-pool executor so the async event loop stays free
# to handle the next Dialogflow webhook call within its 5 s window.

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="emma-rag")

# ── Pending RAG result store ──────────────────────────────────────────────────
# Keyed by Dialogflow session_id. TTL-evicted on every webhook call.

@dataclass
class PendingResult:
    query:        str
    intent_key:   str
    cond_key:     str | None
    cond_display: str | None
    started_at:   float = field(default_factory=_time.time)
    answer:       str | None = None   # None means still running
    error:        bool = False        # True if RAG failed and static was used


_pending: dict[str, PendingResult] = {}


def _pending_evict() -> None:
    cutoff  = _time.time() - _PENDING_TTL
    expired = [k for k, v in _pending.items() if v.started_at < cutoff]
    for k in expired:
        del _pending[k]


# ── Lazy retriever ────────────────────────────────────────────────────────────

_retriever        = None
_RETRIEVER_FAILED = object()


def _get_retriever():
    global _retriever
    if _retriever is _RETRIEVER_FAILED:
        raise RuntimeError("RAG pipeline unavailable (failed to load at startup)")
    if _retriever is None:
        from src.retrieval import EMMARetriever
        model_id = os.environ.get("EMMA_MODEL_ID") or None
        try:
            _retriever = EMMARetriever.load(model_id=model_id, ollama_url=OLLAMA_URL)
            logger.info("EMMARetriever loaded (model=%s)", _retriever.model_id)
        except Exception:
            _retriever = _RETRIEVER_FAILED
            raise
    return _retriever


# ── Background RAG worker ─────────────────────────────────────────────────────

def _run_rag_sync(session_id: str, query: str, think: bool = False) -> None:
    """
    Blocking RAG call — runs in the thread-pool executor.
    Writes the completed answer into _pending[session_id].answer.
    On failure, substitutes the static fallback for known conditions.
    """
    entry = _pending.get(session_id)
    if entry is None:
        return   # evicted before RAG finished — discard result
    try:
        retriever = _get_retriever()
        result    = retriever.answer(query, use_rag=True, think=think)
        answer    = result.answer.strip()
        if not answer:
            raise ValueError("Empty answer from retriever")
        logger.info("RAG complete for session=%s (%.1f s elapsed)",
                    session_id, _time.time() - entry.started_at)
        entry.answer = answer
    except Exception as exc:
        logger.warning("RAG failed for session=%s: %s", session_id, exc)
        if entry.cond_key and entry.intent_key:
            entry.answer = _static_response(entry.intent_key, entry.cond_key)
        else:
            entry.answer = (
                "I ran into an issue retrieving that from the textbooks. "
                "Please try rephrasing your question."
            )
        entry.error = True


async def _fire_rag_background(
    session_id:   str,
    query:        str,
    intent_key:   str,
    cond_key:     str | None,
    cond_display: str | None,
    think:        bool = False,
) -> None:
    """Register a pending entry and fire the RAG worker. Returns immediately."""
    _pending_evict()
    _pending[session_id] = PendingResult(
        query=query, intent_key=intent_key,
        cond_key=cond_key, cond_display=cond_display,
    )
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_rag_sync, session_id, query, think)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    if RAG_ENABLED:
        logger.info("Starting up EMMA (Ollama warmup + HF model load may take 2-3 min)...")
        loop = asyncio.get_event_loop()
        try:
            # retrieval.EMMARetriever.load() handles everything:
            #   1. Ollama warmup request (warms model into Ollama memory)
            #   2. HF model load (always, as genuine fallback)
            # Blocking startup intentionally — better to wait here once than
            # to have the first live user request trigger a 2-minute cold load.
            await loop.run_in_executor(_executor, _get_retriever)
            logger.info("EMMA startup complete — ready to serve requests.")
        except Exception as exc:
            logger.warning("Startup pre-warm failed (%s) — will retry on first request", exc)
    yield
    _executor.shutdown(wait=False)


# ── Session memory ────────────────────────────────────────────────────────────

_SESSION_TTL = 600
_sessions: dict[str, dict] = {}


def _session_get(session_id: str) -> dict:
    entry = _sessions.get(session_id)
    if entry and (_time.time() - entry["ts"]) < _SESSION_TTL:
        return entry
    return {}


def _session_set(session_id: str, intent_key: str, cond_key: str | None,
                 cond_display: str | None, raw_query: str) -> None:
    _sessions[session_id] = {
        "ts": _time.time(), "intent_key": intent_key,
        "cond_key": cond_key, "cond_display": cond_display, "last_query": raw_query,
    }
    now     = _time.time()
    expired = [k for k, v in _sessions.items() if now - v["ts"] > _SESSION_TTL]
    for k in expired:
        del _sessions[k]


# ── Config loaders ────────────────────────────────────────────────────────────

def _config_path(filename: str):
    try:
        from src.data import REPO_ROOT
        return REPO_ROOT / "config" / filename
    except ImportError:
        import pathlib as _pl
        return _pl.Path(__file__).resolve().parent.parent / "config" / filename


def _load_conditions_config() -> dict:
    if not hasattr(_load_conditions_config, "_cache"):
        _load_conditions_config._cache = _json.loads(
            _config_path("conditions.json").read_text(encoding="utf-8"))
    return _load_conditions_config._cache


def _load_responses_config() -> dict:
    if not hasattr(_load_responses_config, "_cache"):
        _load_responses_config._cache = _json.loads(
            _config_path("responses.json").read_text(encoding="utf-8"))
    return _load_responses_config._cache


def _load_intents_config() -> dict:
    if not hasattr(_load_intents_config, "_cache"):
        _load_intents_config._cache = _json.loads(
            _config_path("intents.json").read_text(encoding="utf-8"))
    return _load_intents_config._cache


def _get_condition_meta() -> dict:
    return _load_conditions_config()["conditions"]

def _get_condition_aliases() -> dict:
    return {k: v for k, v in _load_conditions_config()["aliases"].items()
            if not k.startswith("_")}

def _get_static() -> dict:
    return _load_responses_config()["responses"]

def _get_intents_cfg() -> dict:
    return _load_intents_config()

def _CONDITION_META():    return _get_condition_meta()
def _CONDITION_ALIASES(): return _get_condition_aliases()
def _STATIC():            return _get_static()


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _entity_to_key(raw: str) -> str:
    return raw.lower().replace(" ", "_").replace("-", "_")

def _canonical_key(raw: str) -> str | None:
    return _CONDITION_ALIASES().get(_entity_to_key(raw))

def _display_name(raw: str) -> str | None:
    canon = _canonical_key(raw)
    return _CONDITION_META()[canon]["name"] if canon else None

def _extract_from_params(parameters: dict) -> str | None:
    for f in ("condition", "condition_name", "medical_condition", "acute_condition"):
        val = parameters.get(f, "")
        if val:
            return str(val)
    return None

def _condition_key_from_entity(parameters: dict) -> str | None:
    raw = _extract_from_params(parameters)
    return _canonical_key(raw) if raw else None

def _extract_condition(parameters: dict) -> str | None:
    raw = _extract_from_params(parameters)
    return _display_name(raw) if raw else None

def _static_response(intent_key: str, cond_key: str) -> str:
    intent_map = _STATIC().get(intent_key, {})
    if cond_key in intent_map:
        return intent_map[cond_key]
    name = _CONDITION_META().get(cond_key, {}).get("name", cond_key)
    return (
        f"I have information about {name} but no pre-written summary for "
        "that specific question type. Try asking EMMA directly in the chat."
    )

def _build_rag_query(intent_key: str, condition_name: str | None, raw_query: str) -> str:
    if not condition_name:
        return raw_query
    templates = _get_intents_cfg().get("rag_query_templates", {})
    template  = templates.get(intent_key)
    if template:
        return template.replace("{condition}", condition_name)
    return raw_query or f"Tell me about {condition_name}."


# ── Async RAG for /chat and /query (no deadline) ──────────────────────────────

def _rag_response_sync(query: str, think: bool = False) -> str:
    retriever = _get_retriever()
    result    = retriever.answer(query, use_rag=True, think=think)
    answer    = result.answer.strip()
    if not answer:
        raise ValueError("Empty answer from retriever")
    return answer


async def _rag_response(intent_key: str, query: str,
                        cond_key: str | None = None, think: bool = False) -> str:
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            _executor, lambda: _rag_response_sync(query, think=think))
    except Exception as exc:
        logger.warning("RAG async failed (intent=%s): %s", intent_key, exc)
        if cond_key:
            return _static_response(intent_key, cond_key)
        return "I encountered an issue retrieving an answer. Please try again."


# ── Free-text intent detection (for /chat) ────────────────────────────────────

_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("getdifferentiation", ["differ", "distinguish", " vs ", "versus", "compare",
                            "contrast", "tell apart", "not the same"]),
    ("geturgency",         ["urgent", "emergenc", "how serious", "how fast",
                            "time-critical", "time sensitive", "fatal", "mortalit",
                            "life-threatening", "how quickly", "how dangerous"]),
    ("getriskfactors",     ["risk factor", "risk", "predispos", "who gets",
                            "who is at risk", "susceptible", "prone to"]),
    ("getdiagnosis",       ["diagnos", "how is it found", "how do you detect",
                            "test for", "workup", "blood test", "imaging", "confirm",
                            "identify", "ct scan", "mri", "xray", "x-ray"]),
    ("gettreatment",       ["treat", "manag", "therap", "cure", "medic", "drug",
                            "antibiotic", "prescri", "intervention", "surgery",
                            "how do you fix", "how to fix"]),
    ("getsymptoms",        ["symptom", "sign of", "present", "manifest", "feel like",
                            "clinical feature", "how does it feel", "what does it feel"]),
]

def _detect_intent_from_text(text: str) -> str:
    t = text.lower()
    for intent_key, patterns in _INTENT_PATTERNS:
        if any(p in t for p in patterns):
            return intent_key
    return "general"

def _extract_condition_from_text(text: str) -> str | None:
    t       = text.lower()
    aliases = _CONDITION_ALIASES()
    for alias in sorted(aliases, key=len, reverse=True):
        term = alias.replace("_", " ")
        if len(term) <= 3:
            if re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", t):
                return aliases[alias]
        elif term in t:
            return aliases[alias]
    return None


# ── Welcome openers ───────────────────────────────────────────────────────────

_WELCOME_OPENERS: list[tuple[str, str | None, str | None]] = [
    ("Ready to prep?",                                                                           None,                   None),
    ("Quiz time — what are the classic signs of meningitis?",                                    "meningitis",           "getsymptoms"),
    ("Did you know sepsis kills more people annually than breast, bowel, and prostate cancer combined?", "sepsis",        "geturgency"),
    ("Can you name the FAST signs of a stroke?",                                                 "stroke",               "getsymptoms"),
    ("Did you know epinephrine must be given within minutes in anaphylaxis — antihistamines alone won't cut it?", "anaphylaxis", "gettreatment"),
    ("What's the first test you'd order for a suspected pulmonary embolism?",                    "pulmonary_embolism",   "getdiagnosis"),
    ("Time is brain — every minute of untreated stroke destroys ~1.9 million neurons.",          "stroke",               "geturgency"),
    ("Did you know DKA can present with a fruity breath smell?",                                 "diabetic_ketoacidosis","getsymptoms"),
    ("Can you name the Hour-1 Bundle for sepsis?",                                               "sepsis",               "gettreatment"),
    ("What's the door-to-balloon target for a STEMI?",                                          "heart_attack",         "geturgency"),
    ("Did you know appendicitis pain classically starts around the belly button before moving to the right?", "appendicitis", "getsymptoms"),
    ("Ask me anything — symptoms, diagnosis, treatment, or how to tell two conditions apart.",   None,                   None),
]

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="EMMA API",
    description="FastAPI webhook backend for EMMA — Emergency Medicine Mentoring Agent",
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    info: dict[str, Any] = {
        "status":              "ok",
        "rag_enabled":         RAG_ENABLED,
        "version":             "1.0.0",
        "ontology_conditions": len(_CONDITION_META()),
        "ollama_url":          OLLAMA_URL,
        "pending_rag_jobs":    len(_pending),
    }
    if RAG_ENABLED:
        try:
            from src.retrieval import _ollama_available, _ollama_model_pulled, get_model_config
            retriever  = _get_retriever()
            cfg        = get_model_config(retriever.model_id)
            ollama_tag = cfg.get("ollama_tag")
            ollama_up  = _ollama_available(OLLAMA_URL)
            info["inference_backend"] = (
                "ollama" if ollama_tag and ollama_up
                and _ollama_model_pulled(ollama_tag, OLLAMA_URL) else "hf"
            )
            info["ollama_available"] = ollama_up
            info["model_id"]         = retriever.model_id
        except Exception:
            info["inference_backend"] = "unknown"
    return info


# ── Response formatting ───────────────────────────────────────────────────────

def _format_bubbles(text: str) -> list[str]:
    bubbles: list[str] = []
    for section in [s.strip() for s in text.split("\n\n") if s.strip()]:
        header_lines: list[str] = []
        for line in section.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("•", "·", "-")) or (
                len(stripped) > 1 and stripped[0].isdigit() and stripped[1] in ".)"
            ):
                if header_lines:
                    bubbles.append(" ".join(header_lines))
                    header_lines = []
                bubbles.append(stripped)
            else:
                header_lines.append(stripped)
        if header_lines:
            bubbles.append(" ".join(header_lines))
    return [b for b in bubbles if b]


def _build_response(text: str) -> dict:
    bubbles  = _format_bubbles(text)
    messages = [{"text": {"text": [b]}} for b in bubbles]
    return {"fulfillmentText": text, "fulfillmentMessages": messages}


# ── Webhook ───────────────────────────────────────────────────────────────────

@app.post("/webhook")
async def dialogflow_webhook(request: Request) -> JSONResponse:
    """
    Dialogflow ES webhook — two-turn async RAG pattern.

    Turn 1 (new question):
      Returns acknowledgment in < 200 ms.
      Fires RAG in a background thread, result stored in _pending[session_id].

    Turn 2 (any follow-up message):
      Checks _pending[session_id].
      - Answer ready   -> delivers RAG answer, cleans up pending entry.
      - Still running  -> "still working, send another message soon".
      - No entry       -> routes normally via intent matching.
    """
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    query_result: dict = body.get("queryResult", {})
    intent_name: str   = query_result.get("intent", {}).get("displayName", "")
    parameters: dict   = query_result.get("parameters", {})
    raw_query: str     = query_result.get("queryText", "")
    session_id: str    = body.get("session", "unknown")

    intent_key   = intent_name.lower().replace(" ", "").replace("_", "")
    cond_key     = _condition_key_from_entity(parameters)
    cond_display = _extract_condition(parameters)
    session      = _session_get(session_id)

    logger.info("Webhook | intent=%s | cond=%s | pending=%s | query=%r",
                intent_key, cond_key, session_id in _pending, raw_query[:80])

    # ── Priority 0: deliver a pending RAG result ─────────────────────────────
    # Fires on ANY incoming message if a background job is registered for this
    # session — intent matching is irrelevant when the user is waiting for RAG.

    if session_id in _pending:
        entry = _pending[session_id]
        if entry.answer is not None:
            answer = entry.answer
            del _pending[session_id]
            _session_set(session_id, entry.intent_key, entry.cond_key,
                         entry.cond_display, entry.query)
            logger.info("Delivering pending RAG result for session=%s", session_id)
            return JSONResponse(content=_build_response(answer))
        else:
            elapsed = _time.time() - entry.started_at
            logger.info("RAG still running for session=%s (%.1f s)", session_id, elapsed)
            return JSONResponse(content=_build_response(
                "Still looking that up — just a few more seconds. "
                "Send me any message again when you're ready! ⏳"
            ))

    # ── Standard intent routing ───────────────────────────────────────────────

    HANDLED_INTENTS  = set(_get_intents_cfg().get("handled_intents", [
        "getsymptoms", "getdiagnosis", "gettreatment",
        "getriskfactors", "geturgency", "getdifferentiation",
    ]))
    WELCOME_INTENTS  = set(_get_intents_cfg().get("welcome_intents",
                           ["defaultwelcomeintent", "welcome"]))
    FALLBACK_INTENTS = set(_get_intents_cfg().get("fallback_intents",
                           ["defaultfallbackintent", "fallback"]))

    # Inherit context from session for follow-ups
    is_session_followup = (
        cond_key is None and intent_key in FALLBACK_INTENTS and session
    )
    if is_session_followup:
        cond_key     = session.get("cond_key")
        cond_display = session.get("cond_display")
        if cond_key:
            intent_key = session.get("intent_key", intent_key)

    if intent_key in HANDLED_INTENTS:

        if RAG_ENABLED:
            if cond_key is not None:
                rag_query = _build_rag_query(intent_key, cond_display, raw_query)
            elif raw_query:
                rag_query = raw_query
            else:
                return JSONResponse(content=_build_response(
                    "What condition would you like to know about?"))

            await _fire_rag_background(
                session_id=session_id, query=rag_query,
                intent_key=intent_key, cond_key=cond_key, cond_display=cond_display,
                think=False,   # CoT irrelevant for Messenger UX and adds latency
            )
            label = cond_display or "that"
            return JSONResponse(content=_build_response(
                f"Looking up {label} in the medical textbooks… "
                "Send me any message in a moment and I'll have your answer ready! 📚"
            ))

        else:
            # RAG disabled — static only
            if cond_key is not None:
                answer = _static_response(intent_key, cond_key)
            elif raw_query:
                cond_list = " · ".join(m["name"] for m in _CONDITION_META().values())
                answer = (
                    "I can give detailed answers about eight acute emergency "
                    "conditions:\n\n" + cond_list + "\n\nAsk me about any of these!"
                )
            else:
                answer = "What condition would you like to know about?"

    elif intent_key in WELCOME_INTENTS:
        opener_text, opener_cond, opener_intent = random.choice(_WELCOME_OPENERS)
        if opener_cond and opener_intent:
            opener_display = _CONDITION_META().get(opener_cond, {}).get("name")
            _session_set(session_id, opener_intent, opener_cond, opener_display, opener_text)
        return JSONResponse(content=_build_response(opener_text))

    elif intent_key in FALLBACK_INTENTS:
        query_lower = raw_query.lower().strip()
        _FOLLOWUP_PATTERNS = (
            "i don", "don't know", "no idea", "not sure", "what is it",
            "what are they", "tell me", "explain", "go on", "what's the answer",
            "show me", "what about", "and", "yes", "sure", "ok", "okay", "please",
        )
        is_vague = (
            any(query_lower.startswith(p) or p in query_lower for p in _FOLLOWUP_PATTERNS)
            and len(query_lower.split()) <= 6
        )

        if (is_session_followup or is_vague) and session:
            prev_intent  = session.get("intent_key", "getsymptoms")
            prev_cond    = session.get("cond_key")
            prev_display = session.get("cond_display")
            if prev_cond and prev_display:
                if RAG_ENABLED:
                    await _fire_rag_background(
                        session_id=session_id,
                        query=_build_rag_query(prev_intent, prev_display, ""),
                        intent_key=prev_intent, cond_key=prev_cond,
                        cond_display=prev_display,
                    )
                    return JSONResponse(content=_build_response(
                        f"Looking up {prev_display} in the medical textbooks… "
                        "Send me any message in a moment! 📚"
                    ))
                else:
                    answer = _static_response(prev_intent, prev_cond)
            else:
                answer = "What condition would you like to know about?"

        elif RAG_ENABLED and raw_query:
            await _fire_rag_background(
                session_id=session_id, query=raw_query,
                intent_key="fallback", cond_key=None, cond_display=None,
            )
            return JSONResponse(content=_build_response(
                "Looking that up in the medical textbooks… "
                "Send me any message in a moment and I'll have your answer! 📚"
            ))
        else:
            answer = (
                "I'm not sure I understood that. I can help with:\n"
                "• Symptoms of a condition\n"
                "• How a condition is diagnosed\n"
                "• Treatment and management\n"
                "• Risk factors\n"
                "• How urgent a condition is\n"
                "• Differentiating between similar conditions\n\n"
                "Try: 'What are the symptoms of sepsis?' or "
                "'How is a pulmonary embolism treated?'"
            )

    else:
        if RAG_ENABLED and raw_query:
            await _fire_rag_background(
                session_id=session_id, query=raw_query,
                intent_key="unknown", cond_key=None, cond_display=None,
            )
            return JSONResponse(content=_build_response(
                "Looking that up… send me any message in a moment! 📚"
            ))
        else:
            answer = (
                "I received your question but I'm not sure how to help. "
                "Ask me about symptoms, diagnosis, treatment, risk factors, urgency, "
                "or differentiation for any medical condition."
            )

    _session_set(session_id, intent_key, cond_key, cond_display, raw_query)
    return JSONResponse(content=_build_response(answer))


# ── Direct query endpoint ─────────────────────────────────────────────────────

@app.post("/query")
async def direct_query(request: Request) -> JSONResponse:
    """Direct query — bypasses Dialogflow, no deadline.
    Body: { "query": "...", "think": false }
    """
    try:
        body  = await request.json()
        query = body.get("query", "").strip()
        think = bool(body.get("think", False))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    if not query:
        raise HTTPException(status_code=422, detail="'query' field is required")
    if RAG_ENABLED:
        answer = await _rag_response("direct", query, think=think)
    else:
        answer = f"RAG not enabled. Set EMMA_USE_RAG=true. Your query: '{query}'"
    return JSONResponse(content={"answer": answer, "rag_used": RAG_ENABLED})


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: Request) -> JSONResponse:
    """EMMA widget endpoint — no Dialogflow, no deadline, think=True allowed.
    Body: { "message": "...", "session_id": "...", "think": false }
    """
    try:
        body       = await request.json()
        message    = body.get("message", "").strip()
        session_id = body.get("session_id", "chat-default")
        think      = bool(body.get("think", False))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    if not message:
        raise HTTPException(status_code=422, detail="'message' field is required")

    intent_key   = _detect_intent_from_text(message)
    cond_key     = _extract_condition_from_text(message)
    cond_display = _CONDITION_META().get(cond_key, {}).get("name") if cond_key else None

    session = _session_get(session_id)
    if cond_key is None and session:
        cond_key     = session.get("cond_key")
        cond_display = session.get("cond_display")
    if intent_key == "general" and session.get("intent_key"):
        intent_key = session.get("intent_key")

    logger.info("Chat | intent=%s | cond=%s | rag=%s | think=%s | query=%r",
                intent_key, cond_key, RAG_ENABLED, think, message[:80])

    if RAG_ENABLED:
        rag_query = _build_rag_query(intent_key, cond_display, message)
        answer    = await _rag_response(intent_key, rag_query, cond_key=cond_key, think=think)
    elif cond_key:
        answer = _static_response(intent_key, cond_key)
    else:
        cond_list = " · ".join(m["name"] for m in _CONDITION_META().values())
        answer = (
            "I can answer questions about eight acute emergency conditions: "
            + cond_list
            + ". Try asking about symptoms, diagnosis, treatment, risk factors, "
            "urgency, or differentiation."
        )

    _session_set(session_id, intent_key, cond_key, cond_display, message)
    return JSONResponse(content={
        "answer": answer, "intent": intent_key, "condition": cond_display
    })


# ── Conditions listing ────────────────────────────────────────────────────────

@app.get("/conditions")
async def list_conditions():
    return {
        "conditions": [{"key": k, **v} for k, v in _CONDITION_META().items()],
        "note": "The RAG pipeline answers open-domain queries beyond this list.",
    }
