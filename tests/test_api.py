"""
tests/test_api.py
-----------------
Pytest tests for the EMMA FastAPI webhook.

Run with:
    uv run pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

# Import the app — RAG is off by default in tests
import os
os.environ.setdefault("EMMA_USE_RAG", "false")

from src.api import app  # noqa: E402

client = TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "rag_enabled" in data


# ── Conditions ────────────────────────────────────────────────────────────────

def test_conditions_list():
    r = client.get("/conditions")
    assert r.status_code == 200
    conds = r.json()["conditions"]
    assert len(conds) == 8
    keys = {c["key"] for c in conds}
    assert "anaphylaxis" in keys
    assert "diabetic_ketoacidosis" in keys


# ── Webhook — welcome ─────────────────────────────────────────────────────────

def _webhook(intent: str, condition: str = "", query_text: str = ""):
    return client.post("/webhook", json={
        "queryResult": {
            "intent":     {"displayName": intent},
            "parameters": {"condition": condition},
            "queryText":  query_text,
        }
    })


def test_welcome_intent():
    r = _webhook("Default Welcome Intent")
    assert r.status_code == 200
    text = r.json()["fulfillmentText"]
    assert len(text) > 10


def test_fallback_intent():
    r = _webhook("Default Fallback Intent", query_text="blah blah")
    assert r.status_code == 200
    text = r.json()["fulfillmentText"]
    assert len(text) > 20


# ── Webhook — each intent × representative conditions ────────────────────────

@pytest.mark.parametrize("intent,condition", [
    ("GetSymptoms",        "Anaphylaxis"),
    ("GetSymptoms",        "Stroke"),
    ("GetDiagnosis",       "Pulmonary Embolism"),
    ("GetDiagnosis",       "Sepsis"),
    ("GetTreatment",       "Heart Attack"),
    ("GetTreatment",       "Meningitis"),
    ("GetRiskFactors",     "Diabetic Ketoacidosis"),
    ("GetRiskFactors",     "Appendicitis"),
    ("GetUrgency",         "Heart Attack"),
    ("GetUrgency",         "Stroke"),
    ("GetDifferentiation", "Anaphylaxis"),
    ("GetDifferentiation", "DKA"),
])
def test_intent_with_condition(intent, condition):
    r = _webhook(intent, condition=condition)
    assert r.status_code == 200
    text = r.json()["fulfillmentText"]
    # Should return a substantive response, not a generic placeholder
    assert len(text) > 80, f"Response too short for {intent}/{condition}: {text!r}"


def test_intent_without_condition():
    """When no condition entity is provided, expect a prompt to specify one."""
    r = _webhook("GetSymptoms", condition="")
    assert r.status_code == 200
    text = r.json()["fulfillmentText"]
    # Should ask the user to specify a condition
    assert any(w in text.lower() for w in ["condition", "which", "asking"])


def test_unknown_intent():
    r = _webhook("SomeRandomIntent", query_text="what is the meaning of life")
    assert r.status_code == 200
    text = r.json()["fulfillmentText"]
    assert len(text) > 10


# ── Direct query endpoint ─────────────────────────────────────────────────────

def test_direct_query_no_rag():
    """Without RAG, /query should explain the feature flag and echo the query."""
    r = client.post("/query", json={"query": "What is DKA?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert data["rag_used"] is False


def test_direct_query_missing_field():
    r = client.post("/query", json={})
    assert r.status_code == 422


def test_direct_query_invalid_json():
    r = client.post("/query", content=b"not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400


# ── Normalisation edge cases ──────────────────────────────────────────────────

@pytest.mark.parametrize("condition", [
    "heart attack", "Heart Attack", "HEART ATTACK",
    "myocardial infarction", "MI", "mi",
])
def test_condition_aliases_heart_attack(condition):
    r = _webhook("GetSymptoms", condition=condition)
    assert r.status_code == 200
    text = r.json()["fulfillmentText"]
    # Static response for MI mentions crushing/pressure
    assert any(w in text.lower() for w in ["chest", "crushing", "pressure", "pain"])


@pytest.mark.parametrize("condition", [
    "diabetic ketoacidosis", "DKA", "dka",
])
def test_condition_aliases_dka(condition):
    r = _webhook("GetSymptoms", condition=condition)
    assert r.status_code == 200
    text = r.json()["fulfillmentText"]
    assert any(w in text.lower() for w in ["glucose", "ketosis", "acidosis", "polyuria"])


# ── Open-domain routing (no RAG, unrecognised condition) ──────────────────────

def test_unrecognised_condition_no_rag():
    """An unknown condition with RAG off should explain the scope, not crash."""
    r = _webhook("GetSymptoms", condition="COPD", query_text="What are the symptoms of COPD?")
    assert r.status_code == 200
    text = r.json()["fulfillmentText"]
    # Should not be an empty string or server error
    assert len(text) > 20


# ── list_conditions is data-derived ──────────────────────────────────────────

def test_conditions_endpoint_data_derived():
    """list_conditions must be derived from _CONDITION_META, not hardcoded."""
    from src.api import _CONDITION_META
    r = client.get("/conditions")
    assert r.status_code == 200
    data = r.json()
    returned_keys = {c["key"] for c in data["conditions"]}
    meta_keys = set(_CONDITION_META().keys())
    assert returned_keys == meta_keys, (
        f"Endpoint keys {returned_keys} don't match _CONDITION_META keys {meta_keys}"
    )


def test_conditions_has_note_field():
    """Response should include the open-domain note."""
    r = client.get("/conditions")
    assert "note" in r.json()


def test_conditions_metadata_structure():
    """Each condition entry must have key, name, urgency, mortality."""
    r = client.get("/conditions")
    for cond in r.json()["conditions"]:
        assert "key" in cond
        assert "name" in cond
        assert "urgency" in cond
        assert "mortality" in cond


# ── Health endpoint exposes ontology count ────────────────────────────────────

def test_health_has_ontology_count():
    r = client.get("/health")
    assert "ontology_conditions" in r.json()
    assert r.json()["ontology_conditions"] == 8
