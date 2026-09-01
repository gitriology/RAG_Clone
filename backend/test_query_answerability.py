from backend.retrieval.query_matching import (
    assess_answerability,
    extract_focus_phrases,
    phrase_present,
    reference_present,
)


def test_modis_does_not_match_modi():
    assert phrase_present("MODIS observations over India", "modis")
    assert not phrase_present("MODIS observations over India", "modi")


def test_article_reference_matches_numbered_provision_not_124():
    text = "12. Definition.—In this Part, the State includes the Government and Parliament. 124. Establishment and constitution of Supreme Court."
    assert reference_present(text, "Article 12")
    assert not reference_present("124. Establishment and constitution of Supreme Court.", "Article 12")


def test_absent_named_entity_is_not_answerable():
    result = assess_answerability(
        "Who is Narendra Modi?",
        ["MODIS observations show an average reduction in aerosol loading."],
    )
    assert result["answerable"] is False
    assert result["score"] < 0.52


def test_vaccination_definition_is_answerable():
    result = assess_answerability(
        "What is vaccination?",
        ["Vaccination is the action of giving the vaccine."],
        answer="Vaccination is the action of giving the vaccine.",
    )
    assert result["answerable"] is True
    assert result["score"] >= 0.72


def test_compare_requires_both_named_sides():
    result = assess_answerability(
        "Compare Mars Orbiter Mission and Chandrayaan-2.",
        ["The Mars Orbiter Mission was an Indian mission to Mars."],
    )
    assert result["answerable"] is False


def test_longer_neighboring_phrase_does_not_define_shorter_focus():
    result = assess_answerability(
        "What is vaccination?",
        ["Vaccination services refers to where, when, how and by whom vaccines are given in a particular country."],
    )
    assert result["answerable"] is False
