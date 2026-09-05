"""Production regression tests for question-tailored evidence selection.

The tests intentionally separate:
    retrieval similarity -> evidence answering the question.

They cover the healthcare failures found in the supplied WHO PDFs and retain
cross-domain safeguards already required by the project.
"""

from backend.retrieval.query_matching import (
    assess_answerability,
    definition_subject_matches,
    phrase_present,
    reference_present,
)


def test_vaccination_definition_is_answerable():
    query = "What is vaccination?"
    evidence = "Vaccination is the action of giving the vaccine."

    assert definition_subject_matches(query, evidence)

    result = assess_answerability(
        query,
        [evidence],
        answer=evidence,
    )

    assert result["answerable"] is True
    assert result["score"] >= 0.70


def test_vaccination_guidance_is_not_a_definition():
    query = "What is vaccination?"
    evidence = (
        "WHO guidance on building trust in vaccination A brief overview of "
        "WHO guidance and recommendations on communication and confidence-"
        "building in relation to vaccines and vaccination is presented in Figs."
    )

    assert definition_subject_matches(query, evidence) is False

    result = assess_answerability(
        query,
        [evidence],
        answer=evidence,
    )

    assert result["answerable"] is False


def test_immunization_definition_is_answerable():
    query = "What is immunization?"
    evidence = (
        "Immunization is the process whereby a person becomes protected "
        "from a disease."
    )

    assert definition_subject_matches(query, evidence)

    result = assess_answerability(
        query,
        [evidence],
        answer=evidence,
    )

    assert result["answerable"] is True


def test_definition_matching_rejects_incidental_is_presented():
    query = "What is vaccination?"

    assert not definition_subject_matches(
        query,
        "Vaccination is presented in the report.",
    )
    assert not definition_subject_matches(
        query,
        "Vaccination is recommended during an outbreak.",
    )
    assert not definition_subject_matches(
        query,
        "Vaccination is discussed in this section.",
    )


def test_boundary_safe_entity_matching():
    assert phrase_present(
        "Narendra Modi was the prime minister.",
        "Narendra Modi",
    )
    assert not phrase_present(
        "MODIS satellite data were collected.",
        "modi",
    )


def test_article_reference_boundary():
    assert reference_present(
        "12. Definition of the term.",
        "Article 12",
    )
    assert not reference_present(
        "124. Additional provision.",
        "Article 12",
    )
