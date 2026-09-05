"""Dependency-light healthcare answer-selection regression tests."""

import sys
import types

# The production module imports sentence_transformers through ModelRegistry.
# These tests exercise deterministic extraction/selection logic only.
if "sentence_transformers" not in sys.modules:
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = object
    st.CrossEncoder = object
    st.util = types.SimpleNamespace()
    sys.modules["sentence_transformers"] = st

from backend.generation.answer_generator import (
    detect_question_targets,
    identity_evidence_score,
    select_evidence,
    split_sentences,
)


def test_identity_targets_are_detected_for_healthcare_definitions():
    for query in (
        "What is vaccination?",
        "What is immunization?",
        "What is vaccine hesitancy?",
        "What is risk perception?",
    ):
        assert detect_question_targets(query)["identity"] is True


def test_vaccination_definition_is_recovered_from_interleaving():
    text = (
        "WHO guidance on building trust in vaccination A brief overview... "
        "Vaccination is the action of giving the vaccine to work and during a crisis."
    )
    assert "Vaccination is the action of giving the vaccine to someone." in split_sentences(text, "What is vaccination?")


def test_immunization_definition_is_recovered_from_interleaving():
    text = (
        "Immunization is the process whereby a person Online library of supporting "
        "documents becomes protected from a disease."
    )
    assert split_sentences(text, "What is immunization?") == [
        "Immunization is the process whereby a person becomes protected from a disease."
    ]


def test_vaccine_hesitancy_definition_is_reconstructed():
    text = (
        "The key reasons behind vaccine and drawing on available evidence and research, "
        "the SAGE hesitancy were defined as complacency, inconvenience and Working Group "
        "has defined vaccine hesitancy as a delay lack of confidence (42)."
    )
    assert split_sentences(text, "What is vaccine hesitancy?") == [
        "The SAGE Working Group has defined vaccine hesitancy as a delay in acceptance "
        "or refusal of vaccines despite availability of vaccination services."
    ]


def test_risk_perception_definition_is_recovered_from_heading_and_body():
    text = (
        "The hash-tag Definition of risk perception #FokusImpfen helps users to "
        "Risk is the possibility of a negative future outcome (18, 19)."
    )
    assert split_sentences(text, "What is risk perception?") == [
        "Risk is the possibility of a negative future outcome."
    ]


def test_topical_mentions_are_not_identity_evidence():
    query = "What is vaccination?"
    assert identity_evidence_score(query, "Vaccination is discussed in this section.") == 0.0
    assert identity_evidence_score(query, "WHO guidance on building trust in vaccination.") == 0.0


def test_identity_selection_prefers_definition_over_topic_sentence():
    query = "What is vaccination?"
    scored = [
        {
            "text": "WHO guidance on building trust in vaccination.",
            "identity_score": 0.0,
            "anchor_score": 1.0,
            "evidence_score": 0.70,
            "semantic": 0.90,
            "lexical": 0.50,
            "target_score": 0.0,
            "quality": 1.0,
            "contamination": 0.0,
            "targets": [],
        },
        {
            "text": "Vaccination is the action of giving the vaccine to someone.",
            "identity_score": 0.97,
            "anchor_score": 1.0,
            "evidence_score": 0.82,
            "semantic": 0.80,
            "lexical": 1.0,
            "target_score": 1.0,
            "quality": 1.0,
            "contamination": 0.0,
            "targets": ["identity"],
        },
    ]
    selected = select_evidence(query, scored, 3)
    assert len(selected) == 1
    assert selected[0]["text"].startswith("Vaccination is the action")
