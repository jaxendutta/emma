"""
emma.api
--------
FastAPI webhook for Dialogflow ES.

Architecture
------------
Two-tier response strategy:

  Tier 1 — Static knowledge layer (always available, no GPU required)
    A curated dict of medically-accurate, concise responses keyed by
    (intent, condition). Used when the RAG pipeline is unavailable or
    as a fast-path fallback for low-confidence retrievals.

  Tier 2 — RAG pipeline (activated when EMMA_USE_RAG=true and all
    model artefacts are present on disk)
    Routes the assembled query through EMMARetriever -> NER -> FAISS ->
    LLM for a textbook-grounded answer.

The feature flag RAG_ENABLED (set via env var EMMA_USE_RAG) controls
which tier is active.  The response shape is identical in both cases,
so Dialogflow can be pointed at this webhook regardless of whether
Colab artefacts have been deployed yet.

Running locally
---------------
    uv run uvicorn src.api:app --reload --port 8000

Dialogflow ES webhook URL (ngrok example):
    https://<your-ngrok>.ngrok.io/webhook

Environment variables
---------------------
    EMMA_USE_RAG=true     activate Tier 2 (requires models/ artefacts)
    EMMA_MODEL_ID         override default model (optional)
"""

from __future__ import annotations

import asyncio
import logging
import re
import random
import os
import textwrap
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger("emma.api")

# ── Feature flag ──────────────────────────────────────────────────────────────

RAG_ENABLED = os.environ.get("EMMA_USE_RAG", "false").lower() == "true"

# ── Thread pool for blocking RAG calls ────────────────────────────────────────
# Keeps the async event loop free while the LLM generates.

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="emma-rag")

# Timeout applied to webhook RAG calls (Dialogflow hard-deadline is 5 s).
# Set to 0 or "none" to disable.  /chat and /query endpoints never time out.
_RAG_TIMEOUT: float | None = (
    None if os.environ.get("EMMA_RAG_TIMEOUT", "4.5").lower() in ("0", "none", "")
    else float(os.environ.get("EMMA_RAG_TIMEOUT", "4.5"))
)

# ── Lazy retriever (loaded once on first RAG request) ─────────────────────────

_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieval import EMMARetriever
        model_id = os.environ.get("EMMA_MODEL_ID") or None
        _retriever = EMMARetriever.load(model_id=model_id)
        logger.info("EMMARetriever loaded (model=%s)", _retriever.model_id)
    return _retriever


# ── Lifespan: pre-warm retriever on startup ───────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    if RAG_ENABLED:
        logger.info("Pre-warming EMMA retriever...")
        loop = asyncio.get_event_loop()
        try:
            retriever = await loop.run_in_executor(_executor, _get_retriever)
            # Start LLM load in background — don't block startup
            loop.run_in_executor(_executor, retriever._ensure_model_loaded)
        except Exception as exc:
            logger.warning("Retriever pre-warm failed (%s) — will retry on first request", exc)
    yield
    _executor.shutdown(wait=False)


# ── Session memory ────────────────────────────────────────────────────────────
# Lightweight in-process store keyed on Dialogflow session ID.
# Remembers the last condition and intent so follow-up messages like
# "I don't know, what is it?" or "tell me more" resolve correctly.
# TTL of 10 minutes matches typical conversation length.

import time as _time

_SESSION_TTL = 600  # seconds

_sessions: dict[str, dict] = {}


def _session_get(session_id: str) -> dict:
    entry = _sessions.get(session_id)
    if entry and (_time.time() - entry["ts"]) < _SESSION_TTL:
        return entry
    return {}


def _session_set(session_id: str, intent_key: str, cond_key: str | None,
                 cond_display: str | None, raw_query: str) -> None:
    _sessions[session_id] = {
        "ts":          _time.time(),
        "intent_key":  intent_key,
        "cond_key":    cond_key,
        "cond_display": cond_display,
        "last_query":  raw_query,
    }
    # Evict expired sessions to avoid unbounded growth
    now = _time.time()
    expired = [k for k, v in _sessions.items() if now - v["ts"] > _SESSION_TTL]
    for k in expired:
        del _sessions[k]


# ── Config loaders ───────────────────────────────────────────────────────────
# All mutable data lives in config/ JSON files so teammates can edit
# responses and conditions without touching Python source.
# Follows the same pattern as config/models.json + src/retrieval.py.

import json as _json


def _config_path(filename: str):
    """Resolve a config/ filename relative to the repo root."""
    try:
        from src.data import REPO_ROOT
        return REPO_ROOT / "config" / filename
    except ImportError:
        import pathlib as _pl
        here = _pl.Path(__file__).resolve().parent.parent
        return here / "config" / filename


def _load_conditions_config() -> dict:
    """Load config/conditions.json. Cached after first call."""
    if not hasattr(_load_conditions_config, "_cache"):
        _load_conditions_config._cache = _json.loads(
            _config_path("conditions.json").read_text(encoding="utf-8")
        )
    return _load_conditions_config._cache


def _load_responses_config() -> dict:
    """Load config/responses.json. Cached after first call."""
    if not hasattr(_load_responses_config, "_cache"):
        _load_responses_config._cache = _json.loads(
            _config_path("responses.json").read_text(encoding="utf-8")
        )
    return _load_responses_config._cache


def _load_intents_config() -> dict:
    """Load config/intents.json. Cached after first call."""
    if not hasattr(_load_intents_config, "_cache"):
        _load_intents_config._cache = _json.loads(
            _config_path("intents.json").read_text(encoding="utf-8")
        )
    return _load_intents_config._cache


# ── Data accessors ────────────────────────────────────────────────────────────
# Each returns a live view of the JSON — changes to the file take effect on
# the next server restart (or cache clear).  No code changes required.

def _get_condition_meta() -> dict:
    """conditions.json -> conditions dict (canonical key -> metadata)."""
    return _load_conditions_config()["conditions"]


def _get_condition_aliases() -> dict:
    """conditions.json -> alias map (any spelling -> canonical key)."""
    return {
        k: v
        for k, v in _load_conditions_config()["aliases"].items()
        if not k.startswith("_")
    }


def _get_static() -> dict:
    """responses.json -> nested dict (intent -> condition -> response text)."""
    return _load_responses_config()["responses"]


def _get_intents_cfg() -> dict:
    """intents.json -> full intents config."""
    return _load_intents_config()


# Convenience aliases used inline below (evaluated per-call, not at import).
# Keeping these as functions rather than module-level vars means a JSON edit
# is picked up on restart without any code change.
def _CONDITION_META():   return _get_condition_meta()
def _CONDITION_ALIASES(): return _get_condition_aliases()
def _STATIC():           return _get_static()


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _entity_to_key(raw: str) -> str:
    """Normalise a raw Dialogflow entity string to a lookup key."""
    return raw.lower().replace(" ", "_").replace("-", "_")


def _canonical_key(raw: str) -> str | None:
    """
    Map any entity string (alias or canonical) -> canonical key in _CONDITION_META.
    Returns None for unrecognised strings — these are open-domain queries that
    should go to RAG rather than trigger a static catch-all prompt.
    """
    return _CONDITION_ALIASES().get(_entity_to_key(raw))


def _display_name(raw: str) -> str | None:
    """Return the human-readable display name for a condition, or None."""
    canon = _canonical_key(raw)
    return _CONDITION_META()[canon]["name"] if canon else None


def _extract_from_params(parameters: dict) -> str | None:
    """Pull the first non-empty condition value from a Dialogflow parameter dict."""
    for field in ("condition", "condition_name", "medical_condition", "acute_condition"):
        val = parameters.get(field, "")
        if val:
            return str(val)
    return None





def _condition_key_from_entity(parameters: dict) -> str | None:
    """Return the canonical condition key from Dialogflow parameters, or None."""
    raw = _extract_from_params(parameters)
    return _canonical_key(raw) if raw else None


def _extract_condition(parameters: dict) -> str | None:
    """Return the display name for the condition in parameters, or None."""
    raw = _extract_from_params(parameters)
    return _display_name(raw) if raw else None


def _static_response(intent_key: str, cond_key: str) -> str:
    """
    Retrieve a canned static response for a known canonical (intent, condition) pair.
    Only called when cond_key is a recognised canonical key from _CONDITION_META.
    """
    intent_map = _STATIC().get(intent_key, {})
    if cond_key in intent_map:
        return intent_map[cond_key]
    name = _CONDITION_META().get(cond_key, {}).get("name", cond_key)
    return (
        f"I have information about {name} but no pre-written summary for "
        "that specific question type. Try asking EMMA directly in the chat."
    )


# ── RAG response builder ──────────────────────────────────────────────────────

def _rag_response_sync(query: str, think: bool = False) -> str:
    """Blocking RAG call — runs inside the thread-pool executor."""
    try:
        retriever = _get_retriever()
        result    = retriever.answer(query, use_rag=True, think=think)
        answer    = result.answer.strip()
        if not answer:
            raise ValueError("Empty answer from retriever")
        return answer
    except Exception as exc:
        logger.warning("RAG pipeline failed (%s); using static fallback", exc)
        return "I encountered an issue retrieving an answer. Please try again."


async def _rag_response(
    intent_key: str,
    query:      str,
    cond_key:   str | None = None,
    think:      bool = False,
    timeout:    float | None = None,
) -> str:
    """
    Non-blocking RAG call.

    Runs _rag_response_sync in the thread pool so the async event loop
    stays free.  When timeout is set, falls back to a static response on
    TimeoutError (used by the Dialogflow webhook which has a 5-second limit).
    timeout=None means wait indefinitely (used by /chat and /query).
    """
    loop = asyncio.get_event_loop()
    fut  = loop.run_in_executor(_executor, lambda: _rag_response_sync(query, think=think))
    try:
        if timeout is not None:
            return await asyncio.wait_for(fut, timeout=timeout)
        return await fut
    except asyncio.TimeoutError:
        logger.warning("RAG timed out after %.1f s (intent=%s)", timeout, intent_key)
        if cond_key:
            return _static_response(intent_key, cond_key)
        return (
            "The answer took longer than expected to generate. "
            "Please try again or ask via the chat widget for a full response."
        )
    except Exception as exc:
        logger.warning("RAG async failed (%s)", exc)
        return "I encountered an issue retrieving an answer. Please try again."


# ── Free-text intent + condition detection (for /chat endpoint) ───────────────

_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("getdifferentiation", ["differ", "distinguish", " vs ", "versus", "compare", "contrast",
                            "tell apart", "not the same"]),
    ("geturgency",         ["urgent", "emergenc", "how serious", "how fast", "time-critical",
                            "time sensitive", "fatal", "mortalit", "life-threatening",
                            "how quickly", "how dangerous"]),
    ("getriskfactors",     ["risk factor", "risk", "predispos", "who gets", "who is at risk",
                            "susceptible", "prone to"]),
    ("getdiagnosis",       ["diagnos", "how is it found", "how do you detect", "test for",
                            "workup", "blood test", "imaging", "confirm", "identify",
                            "ct scan", "mri", "xray", "x-ray"]),
    ("gettreatment",       ["treat", "manag", "therap", "cure", "medic", "drug", "antibiotic",
                            "prescri", "intervention", "surgery", "how do you fix",
                            "how to fix"]),
    ("getsymptoms",        ["symptom", "sign of", "present", "manifest", "feel like",
                            "clinical feature", "how does it feel", "what does it feel"]),
]


def _detect_intent_from_text(text: str) -> str:
    """Map free-text to the closest intent key, or 'general'."""
    t = text.lower()
    for intent_key, patterns in _INTENT_PATTERNS:
        if any(p in t for p in patterns):
            return intent_key
    return "general"


def _extract_condition_from_text(text: str) -> str | None:
    """
    Scan text for any alias in conditions.json and return the canonical key,
    or None if nothing matches.
    """
    t       = text.lower()
    aliases = _CONDITION_ALIASES()
    # Longest match first so "pulmonary embolism" beats "pe"
    for alias in sorted(aliases, key=len, reverse=True):
        term = alias.replace("_", " ")
        if len(term) <= 3:
            # Short abbreviations need word-boundary protection (mi, pe, dka)
            if re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", t):
                return aliases[alias]
        elif term in t:
            return aliases[alias]
    return None


# ── Intent -> query templates ────────────────────────────────────────────────────────────

def _build_rag_query(intent_key: str, condition_name: str | None, raw_query: str) -> str:
    """Build a structured RAG query using the template from intents.json."""
    if not condition_name:
        return raw_query
    templates = _get_intents_cfg().get("rag_query_templates", {})
    template = templates.get(intent_key)
    if template:
        return template.replace("{condition}", condition_name)
    return raw_query or f"Tell me about {condition_name}."


# ── Welcome openers ───────────────────────────────────────────────────────────
# Randomly selected each time the chat widget opens. Keep them short, punchy,
# and varied — the page already explains what EMMA does.

# Each opener is (text, condition_key_or_None, intent_key_or_None).
# When a condition+intent are set, the session is seeded so a follow-up
# like "I don't know, what is it?" resolves to the right answer.
_WELCOME_OPENERS: list[tuple[str, str | None, str | None]] = [
    ("Ready to prep?",                                                                           None,                   None),
    ("Quiz time — what are the classic signs of meningitis?",                                    "meningitis",           "getsymptoms"),
    ("Did you know sepsis kills more people annually than breast, bowel, and prostate cancer combined?", "sepsis",        "geturgency"),
    ("Can you name the FAST signs of a stroke?",                                                 "stroke",               "getsymptoms"),
    ("Did you know epinephrine must be given within minutes in anaphylaxis — antihistamines alone won't cut it?", "anaphylaxis", "gettreatment"),
    ("What's the first test you'd order for a suspected pulmonary embolism?",                  "pulmonary_embolism",   "getdiagnosis"),
    ("Time is brain — every minute of untreated stroke destroys ~1.9 million neurons.",          "stroke",               "geturgency"),
    ("Did you know DKA can present with a fruity breath smell?",                                 "diabetic_ketoacidosis","getsymptoms"),
    ("Can you name the Hour-1 Bundle for sepsis?",                                               "sepsis",               "gettreatment"),
    ("What's the door-to-balloon target for a STEMI?",                                          "heart_attack",         "geturgency"),
    ("Did you know appendicitis pain classically starts around the belly button before moving to the right?", "appendicitis", "getsymptoms"),
    ("Ask me anything — symptoms, diagnosis, treatment, or how to tell two conditions apart.",   None,                   None),
]

# ── FastAPI app ─────────────────────────────────────────────────────────────────────────────

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


# ── Health check ────────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_enabled": RAG_ENABLED,
        "version": "1.0.0",
        "ontology_conditions": len(_CONDITION_META()),
    }


# ── Main Dialogflow webhook ───────────────────────────────────────────────────────

def _format_bubbles(text: str) -> list[str]:
    """
    Split a plain-text response into a list of short strings, each of
    which will become a separate df-messenger chat bubble.

    Strategy:
      1. Split on blank lines (\n\n) into sections.
      2. Within each section, split individual bullet lines (lines
         starting with • or a digit+dot) into their own bubbles.
      3. Strip empty strings.

    df-messenger renders each bubble as plain text with no markdown,
    so we keep the text as-is — no special chars needed.
    """
    bubbles: list[str] = []
    sections = [s.strip() for s in text.split("\n\n") if s.strip()]
    for section in sections:
        lines = section.splitlines()
        header_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Bullet or numbered line -> own bubble
            if stripped.startswith(("•", "·", "-")) or (
                len(stripped) > 1 and stripped[0].isdigit() and stripped[1] in ".)"
            ):
                # Flush any accumulated header lines first
                if header_lines:
                    bubbles.append(" ".join(header_lines))
                    header_lines = []
                bubbles.append(stripped)
            else:
                header_lines.append(stripped)
        if header_lines:
            bubbles.append(" ".join(header_lines))
    return [b for b in bubbles if b]


def _build_response(text: str, session_id: str | None = None) -> dict:
    """
    Convert a plain-text answer into a Dialogflow response payload.
    Each logical line becomes a separate df-messenger bubble.
    fulfillmentText is kept as a fallback for the simulator / curl.
    """
    bubbles = _format_bubbles(text)
    messages = [{"text": {"text": [b]}} for b in bubbles]
    return {
        "fulfillmentText": text,
        "fulfillmentMessages": messages,
    }


@app.post("/webhook")
async def dialogflow_webhook(request: Request) -> JSONResponse:
    """
    Dialogflow ES webhook.

    Routing logic
    -------------
    For the 6 handled intents, the decision tree is:

      1. Condition entity recognised (in _CONDITION_ALIASES)?
         a. RAG enabled  -> build structured query, call retriever.
         b. RAG disabled -> return static canned response.

      2. Condition entity present but NOT in ontology (open-domain)?
         a. RAG enabled  -> pass raw query directly to retriever.
         b. RAG disabled -> explain the 8-condition evaluation scope.

      3. No condition entity at all?
         a. RAG enabled  -> pass raw query to retriever (free-form medical Q&A).
         b. RAG disabled -> ask the user to specify a condition.

    With RAG on, EMMA answers questions about any medical topic the 18
    textbooks cover, not just the 8 ontology conditions.
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
    cond_key     = _condition_key_from_entity(parameters)   # canonical key or None
    cond_display = _extract_condition(parameters)           # display name or None
    raw_entity   = _extract_from_params(parameters)         # raw entity string or None

    # If still no condition, inherit context from the session store.
    session = _session_get(session_id)
    is_followup = (
        cond_key is None
        and intent_key in set(_get_intents_cfg().get("fallback_intents",
                              ["defaultfallbackintent", "fallback"]))
        and session
    )
    if is_followup:
        # Reconstruct intent from last turn — treat as continuation
        prev_intent   = session.get("intent_key", intent_key)
        cond_key      = session.get("cond_key")
        cond_display  = session.get("cond_display")
        raw_entity    = cond_display
        # Re-route to the previous intent so the answer makes sense
        if cond_key:
            intent_key = prev_intent

    logger.info(
        "Webhook | intent=%s | cond_key=%s | rag=%s | query=%r",
        intent_key, cond_key, RAG_ENABLED, raw_query[:80],
    )

    # ── Route by intent ─────────────────────────────────────────────────────────────

    HANDLED_INTENTS = set(_get_intents_cfg().get("handled_intents", [
        "getsymptoms", "getdiagnosis", "gettreatment",
        "getriskfactors", "geturgency", "getdifferentiation",
    ]))

    if intent_key in HANDLED_INTENTS:

        if cond_key is not None:
            # Condition is in the ontology evaluation domain.
            if RAG_ENABLED:
                rag_query = _build_rag_query(intent_key, cond_display, raw_query)
                answer = await _rag_response(intent_key, rag_query,
                                             cond_key=cond_key, timeout=_RAG_TIMEOUT)
            else:
                answer = _static_response(intent_key, cond_key)

        else:
            # No condition entity from Dialogflow parameters.
            # With RAG: send the raw query directly — SpaCy NER extracts the
            # condition and FAISS retrieves relevant passages for any condition
            # in the 18 textbooks, not just the 8 evaluation conditions.
            # Without RAG: we only have static responses for the 8 conditions,
            # so we tell the user that honestly rather than pretending otherwise.
            if RAG_ENABLED and raw_query:
                answer = await _rag_response(intent_key, raw_query, timeout=_RAG_TIMEOUT)
            elif raw_query:
                cond_list = " · ".join(
                    meta["name"] for meta in _CONDITION_META().values()
                )
                answer = (
                    "I can give detailed answers about eight acute emergency conditions:\n\n"
                    + cond_list + "\n\n"
                    "For other conditions, enable the RAG pipeline for full textbook-grounded answers."
                )
            else:
                answer = "What condition would you like to know about?"

    elif intent_key in set(_get_intents_cfg().get("welcome_intents", ["defaultwelcomeintent", "welcome"])):
        opener_text, opener_cond, opener_intent = random.choice(_WELCOME_OPENERS)
        answer = opener_text
        # Seed the session so a follow-up resolves correctly
        if opener_cond and opener_intent:
            opener_display = _CONDITION_META().get(opener_cond, {}).get("name")
            _session_set(session_id, opener_intent, opener_cond, opener_display, opener_text)

    elif intent_key in set(_get_intents_cfg().get("fallback_intents", ["defaultfallbackintent", "fallback"])):
        # Detect follow-up phrases that mean "answer the last question"
        _FOLLOWUP_PATTERNS = (
            "i don", "don't know", "no idea", "not sure", "what is it",
            "what are they", "tell me", "explain", "go on", "what's the answer",
            "show me", "what about", "and", "yes", "sure", "ok", "okay", "please",
        )
        query_lower = raw_query.lower().strip()
        is_asking_followup = any(query_lower.startswith(p) or p in query_lower
                                 for p in _FOLLOWUP_PATTERNS)

        # A genuine follow-up is short and vague (≤6 words, no new condition).
        # A real question may start with a follow-up word but contains substance.
        word_count = len(query_lower.split())
        truly_vague = is_asking_followup and word_count <= 6

        if (is_followup or truly_vague) and session:
            # Re-run the previous turn's intent+condition
            prev_intent  = session.get("intent_key", "getsymptoms")
            prev_cond    = session.get("cond_key")
            prev_display = session.get("cond_display")
            if prev_cond and prev_display:
                rag_q = _build_rag_query(prev_intent, prev_display, "")
                if RAG_ENABLED:
                    answer = await _rag_response(prev_intent, rag_q,
                                                 cond_key=prev_cond, timeout=_RAG_TIMEOUT)
                else:
                    answer = _static_response(prev_intent, prev_cond)
            else:
                answer = "What condition would you like to know about?"
        elif RAG_ENABLED and raw_query:
            answer = await _rag_response("fallback", raw_query, timeout=_RAG_TIMEOUT)
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
        # Unknown intent -- try RAG on the raw query, else graceful fallback.
        if RAG_ENABLED and raw_query:
            answer = await _rag_response("unknown", raw_query, timeout=_RAG_TIMEOUT)
        else:
            answer = (
                "I received your question but I'm not sure how to help. "
                "Ask me about symptoms, diagnosis, treatment, risk factors, urgency, "
                "or differentiation for any medical condition."
            )

    # Save context for follow-up turns
    if intent_key not in set(_get_intents_cfg().get("welcome_intents",
                             ["defaultwelcomeintent", "welcome"])):
        _session_set(session_id, intent_key, cond_key, cond_display, raw_query)

    return JSONResponse(content=_build_response(answer))


# ── Direct query endpoint (for testing without Dialogflow) ───────────────────────

@app.post("/query")
async def direct_query(request: Request) -> JSONResponse:
    """
    Direct query endpoint — bypasses Dialogflow intent routing.
    Useful for testing the RAG pipeline from curl or a test script.

    Body: { "query": "What is the treatment for anaphylaxis?", "think": false }
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
        # No timeout on /query — callers here are developers, not Dialogflow
        answer = await _rag_response("direct", query, think=think, timeout=None)
    else:
        answer = (
            "RAG pipeline is not enabled. Set EMMA_USE_RAG=true and ensure "
            "model artefacts are present in models/. "
            f"Your query: '{query}'"
        )

    return JSONResponse(content={"answer": answer, "rag_used": RAG_ENABLED})


# ── Custom chat endpoint (replaces Dialogflow df-messenger widget) ────────────

@app.post("/chat")
async def chat(request: Request) -> JSONResponse:
    """
    Conversational chat endpoint for the built-in EMMA chat widget.

    No Dialogflow dependency, no 5-second timeout constraint.
    Maintains session context the same way as the webhook.

    Body: {
        "message":    "What are the symptoms of sepsis?",
        "session_id": "browser-generated-uuid",   // optional
        "think":      false                        // optional, enables Qwen3 CoT
    }

    Response: {
        "answer":    "...",
        "intent":    "getsymptoms",
        "condition": "Sepsis"
    }
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

    # ── Detect intent and condition from free text ────────────────────────────
    intent_key  = _detect_intent_from_text(message)
    cond_key    = _extract_condition_from_text(message)
    cond_display = (
        _CONDITION_META().get(cond_key, {}).get("name") if cond_key else None
    )

    # Inherit context from the session store when nothing was detected
    session = _session_get(session_id)
    if cond_key is None and session:
        cond_key     = session.get("cond_key")
        cond_display = session.get("cond_display")
    if intent_key == "general" and session.get("intent_key"):
        intent_key = session.get("intent_key")

    logger.info(
        "Chat | intent=%s | cond_key=%s | rag=%s | query=%r",
        intent_key, cond_key, RAG_ENABLED, message[:80],
    )

    # ── Build answer ──────────────────────────────────────────────────────────
    if RAG_ENABLED:
        rag_query = _build_rag_query(intent_key, cond_display, message)
        answer    = await _rag_response(intent_key, rag_query,
                                        cond_key=cond_key, think=think, timeout=None)
    elif cond_key:
        answer = _static_response(intent_key, cond_key)
    else:
        cond_list = " · ".join(m["name"] for m in _CONDITION_META().values())
        answer = (
            "I can answer questions about eight acute emergency conditions: "
            + cond_list
            + ". Try asking about symptoms, diagnosis, treatment, risk factors, "
            "urgency, or differentiation for any of these."
        )

    # Save context for follow-up messages
    _session_set(session_id, intent_key, cond_key, cond_display, message)

    return JSONResponse(content={
        "answer":    answer,
        "intent":    intent_key,
        "condition": cond_display,
    })


# ── Conditions listing ──────────────────────────────────────────────────────────────────────
# Derived at runtime from _CONDITION_META -- the single source of truth.
# Not consumed by Dialogflow (it only POSTs to /webhook); used by the frontend
# card grid and for introspection.

@app.get("/conditions")
async def list_conditions():
    """
    Return the evaluation-domain conditions with metadata.
    Derived from _CONDITION_META -- add a new condition there and it
    appears here automatically, no endpoint edit required.
    """
    return {
        "conditions": [
            {"key": key, **meta}
            for key, meta in _CONDITION_META().items()
        ],
        "note": (
            "These are the structured evaluation conditions. "
            "The RAG pipeline answers open-domain queries beyond this list."
        ),
    }
