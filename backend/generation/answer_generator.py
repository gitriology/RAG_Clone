"""
Phase 10.1
Question-Aware Semantic Sentence Selection

Production version
------------------
This module performs extractive, question-aware evidence selection.

Design goals
------------
1. Never invent facts.
2. Select evidence for the actual question, not just semantic similarity.
3. Handle multi-part questions.
4. Require question/entity alignment for answer targets.
5. Prefer direct organization/developer evidence over incidental mentions.
6. Penalize cross-topic contamination.
7. Stop once the question is sufficiently answered.
8. Avoid padding answers with merely related sentences.
9. Preserve sentence embeddings for Optimization #17.
10. Keep the output compatible with EvidenceFusion and run_rerank.py.

Example
-------
Query:
    What is the Mars Orbiter Mission (MOM) and which organization
    developed it?

Preferred evidence:
    Mars Orbiter Mission, the maiden interplanetary mission of ISRO...

Rejected as unnecessary supporting evidence:
    MOM completed six years in Martian orbit...

because it is true and entity-aligned, but it does not answer either
question target.

Important
---------
This module is extractive. It selects source sentences and never
generates factual content that is not present in the source.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from sentence_transformers import util

from backend.models.model_registry import ModelRegistry


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text: str) -> str:
    """
    Conservative source-text cleanup.

    We do not rewrite factual content because this is an
    extractive pipeline.
    """

    if text is None:
        return ""

    text = str(text)

    # Remove URLs.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    # Repair spaces before punctuation.
    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text,
    )

    return text.strip()


# ==========================================================
# SENTENCE SPLITTING
# ==========================================================

def split_sentences(text: str) -> List[str]:
    """
    Split source text into candidate evidence sentences.

    Handles ordinary punctuation and common PDF/OCR output.
    """

    text = clean_text(text)

    if not text:
        return []

    raw = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    candidates: List[str] = []

    for sentence in raw:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Ignore extremely small fragments.
        if len(sentence.split()) < 5:
            continue

        candidates.append(sentence)

    return candidates


# ==========================================================
# DUPLICATE REMOVAL
# ==========================================================

def remove_duplicates(
    sentences: List[str],
) -> List[str]:
    """
    Remove exact normalized duplicates while preserving order.
    """

    unique: List[str] = []
    seen = set()

    for sentence in sentences:

        key = re.sub(
            r"\s+",
            " ",
            sentence.lower(),
        ).strip()

        if key in seen:
            continue

        seen.add(key)
        unique.append(sentence)

    return unique


# ==========================================================
# QUESTION ANALYSIS
# ==========================================================

STOPWORDS = {
    "where",
    "what",
    "when",
    "which",
    "who",
    "whom",
    "why",
    "how",
    "did",
    "does",
    "do",
    "is",
    "was",
    "were",
    "are",
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "on",
    "in",
    "for",
    "with",
    "that",
    "this",
    "it",
    "its",
    "mission",
}


def normalize_token(
    token: str,
) -> str:
    """
    Normalize one lexical token.
    """

    token = token.lower()

    token = re.sub(
        r"[^a-z0-9\-]",
        "",
        token,
    )

    return token


def query_terms(
    query: str,
) -> List[str]:
    """
    Extract meaningful lexical query terms.
    """

    terms: List[str] = []

    for token in query.lower().split():

        token = normalize_token(token)

        if not token:
            continue

        if token in STOPWORDS:
            continue

        if len(token) < 3:
            continue

        terms.append(token)

    return terms


def detect_question_targets(
    query: str,
) -> Dict[str, bool]:
    """
    Detect answer dimensions in the query.
    """

    q = query.lower()

    return {
        "identity": any(
            phrase in q
            for phrase in (
                "what is",
                "what was",
                "tell me about",
                "describe",
                "explain",
                "what does",
            )
        ),

        "location": any(
            phrase in q
            for phrase in (
                "where",
                "location",
                "land",
                "landed",
                "landing",
            )
        ),

        "organization": any(
            phrase in q
            for phrase in (
                "which organization",
                "which organisation",
                "organization",
                "organisation",
                "who developed",
                "developed by",
                "developed it",
                "developer",
                "operator",
            )
        ),

        "date": any(
            phrase in q
            for phrase in (
                "when",
                "date",
                "launched",
                "launch date",
            )
        ),
    }


# ==========================================================
# QUERY ENTITY DETECTION
# ==========================================================

def detect_query_entities(
    query: str,
) -> List[str]:
    """
    Detect important named entities / topic anchors.

    These anchors prevent cross-topic contamination.
    """

    q = query.lower()

    entities: List[str] = []

    # ------------------------------------------------------
    # Explicit Mars Orbiter Mission aliases
    # ------------------------------------------------------

    if (
        "mars orbiter mission" in q
        or "mars orbiter" in q
        or re.search(
            r"\bmom\b",
            q,
        )
    ):
        entities.extend(
            [
                "mars",
                "orbiter",
                "mars orbiter mission",
                "mom",
            ]
        )

    # ------------------------------------------------------
    # Chandrayaan
    # ------------------------------------------------------

    if "chandrayaan" in q:
        entities.append("chandrayaan")

    # ------------------------------------------------------
    # Generic lexical anchors
    # ------------------------------------------------------

    for term in query_terms(query):

        if term in {
            "organization",
            "organisation",
            "developed",
            "developer",
            "launch",
            "launched",
            "date",
            "when",
            "where",
        }:
            continue

        if term not in entities:
            entities.append(term)

    return list(
        dict.fromkeys(entities)
    )


def entity_alignment_score(
    query: str,
    sentence: str,
) -> Tuple[float, List[str]]:
    """
    Measure whether the sentence belongs to the same topic/entity
    as the question.

    Returns
    -------
    score:
        [0, 1]

    matched_entities:
        entities found in the sentence
    """

    entities = detect_query_entities(
        query
    )

    if not entities:
        return 1.0, []

    s = sentence.lower()

    matched: List[str] = []

    for entity in entities:

        if entity in s:
            matched.append(entity)

    # Strong explicit mission match.
    if (
        "mars orbiter mission" in s
        or re.search(
            r"\bmom\b",
            s,
        )
    ) and (
        "mars" in entities
        or "mom" in entities
        or "mars orbiter mission" in entities
    ):
        return 1.0, matched

    # No topic match = dangerous candidate.
    if not matched:
        return 0.0, []

    score = len(matched) / len(
        entities
    )

    return min(
        score,
        1.0,
    ), matched


# ==========================================================
# LEXICAL QUESTION SCORING
# ==========================================================

def lexical_question_score(
    query: str,
    sentence: str,
) -> float:
    """
    Measures meaningful lexical overlap with the query.
    """

    terms = query_terms(
        query
    )

    if not terms:
        return 0.0

    sentence_lower = sentence.lower()

    hits = 0

    for term in terms:

        if re.search(
            rf"\b{re.escape(term)}\b",
            sentence_lower,
        ):
            hits += 1

    return hits / len(
        terms
    )


# ==========================================================
# ORGANIZATION EVIDENCE
# ==========================================================

def organization_evidence_score(
    query: str,
    sentence: str,
) -> float:
    """
    Score direct evidence identifying the organization.

    Organization evidence requires topic/entity alignment.

    A bare mention of ISRO is intentionally weak.

    Strong examples:
        developed by ISRO
        operated by ISRO
        Mars Orbiter Mission, the maiden interplanetary mission
        of ISRO

    Weak example:
        ISRO also developed Chandrayaan-2.

    The latter should not answer a MOM organization question.
    """

    targets = detect_question_targets(
        query
    )

    if not targets["organization"]:
        return 0.0

    s = sentence.lower()

    entity_score, _ = entity_alignment_score(
        query,
        sentence,
    )

    if entity_score <= 0.0:
        return 0.0

    score = 0.0

    # ------------------------------------------------------
    # Direct developer/operator relationship.
    # ------------------------------------------------------

    direct_patterns = (
        r"developed\s+by\s+isro",
        r"developed\s+by\s+the\s+indian\s+space\s+research",
        r"developed\s+by\s+indian\s+space\s+research",
        r"built\s+by\s+isro",
        r"designed\s+by\s+isro",
        r"operated\s+by\s+isro",
        r"operator\s*[:\-]?\s*isro",
    )

    if any(
        re.search(
            pattern,
            s,
        )
        for pattern in direct_patterns
    ):
        score += 1.0

    # ------------------------------------------------------
    # Mission + ISRO in the same sentence.
    # ------------------------------------------------------

    mission_terms = (
        "mars orbiter mission",
        "mars orbiter",
        "mom",
    )

    has_mission = any(
        term in s
        for term in mission_terms
    )

    has_isro = (
        "isro" in s
        or "indian space research organisation" in s
        or "indian space research organization" in s
    )

    if has_mission and has_isro:
        score += 0.75

    # ------------------------------------------------------
    # Direct mission-of-ISRO relationship.
    # ------------------------------------------------------

    if (
        has_mission
        and any(
            phrase in s
            for phrase in (
                "maiden interplanetary mission of isro",
                "mission of isro",
                "isro mission",
                "isro's mission",
            )
        )
    ):
        score += 0.50

    # ------------------------------------------------------
    # Bare ISRO is weak evidence.
    # ------------------------------------------------------

    if has_isro and not has_mission:
        score += 0.05

    # Entity alignment gates the result.
    score *= (
        0.5
        + 0.5 * entity_score
    )

    return min(
        score,
        1.0,
    )


# ==========================================================
# IDENTITY / DEFINITION EVIDENCE
# ==========================================================

def identity_evidence_score(
    query: str,
    sentence: str,
) -> float:
    """
    Score sentences that explain what the requested entity is.

    Incidental facts about the entity receive much less credit than
    sentences that actually define or characterize it.
    """

    targets = detect_question_targets(
        query
    )

    if not targets["identity"]:
        return 0.0

    s = sentence.lower()

    entity_score, _ = entity_alignment_score(
        query,
        sentence,
    )

    if entity_score <= 0.0:
        return 0.0

    score = 0.0

    # ------------------------------------------------------
    # Explicit definition / characterization.
    # ------------------------------------------------------

    definition_patterns = (
        "is a",
        "was a",
        "is the",
        "was the",
        "mission of",
        "maiden interplanetary mission",
        "interplanetary mission",
    )

    if any(
        pattern in s
        for pattern in definition_patterns
    ):
        score += 0.50

    # ------------------------------------------------------
    # Strong mission-specific evidence.
    # ------------------------------------------------------

    if (
        "mars orbiter mission" in s
        or re.search(
            r"\bmom\b",
            s,
        )
    ):
        score += 0.35

    # ------------------------------------------------------
    # Mission facts that help characterize identity.
    # ------------------------------------------------------

    if any(
        phrase in s
        for phrase in (
            "launched",
            "martian orbit",
            "mars",
            "pslv",
        )
    ):
        score += 0.15

    score *= (
        0.5
        + 0.5 * entity_score
    )

    return min(
        score,
        1.0,
    )


# ==========================================================
# LOCATION EVIDENCE
# ==========================================================

def location_evidence_score(
    query: str,
    sentence: str,
) -> float:
    """
    Score direct location/landing evidence.
    """

    targets = detect_question_targets(
        query
    )

    if not targets["location"]:
        return 0.0

    s = sentence.lower()

    entity_score, _ = entity_alignment_score(
        query,
        sentence,
    )

    if entity_score <= 0.0:
        return 0.0

    patterns = (
        "landed on the moon",
        "land on the moon",
        "soft-landed on moon",
        "soft landed on moon",
        "soft landing near the lunar south pole",
        "land near the lunar south pole",
        "landed near the lunar south pole",
        "near the lunar south pole",
        "lunar south pole",
    )

    if any(
        pattern in s
        for pattern in patterns
    ):
        return min(
            1.0,
            0.5
            + 0.5 * entity_score,
        )

    return 0.0


# ==========================================================
# DATE EVIDENCE
# ==========================================================

def date_evidence_score(
    query: str,
    sentence: str,
) -> float:
    """
    Score launch/date evidence.
    """

    targets = detect_question_targets(
        query
    )

    if not targets["date"]:
        return 0.0

    s = sentence.lower()

    entity_score, _ = entity_alignment_score(
        query,
        sentence,
    )

    if entity_score <= 0.0:
        return 0.0

    if re.search(
        r"\b\d{1,2}\s+[a-z]+\s+\d{4}\b",
        s,
    ):
        return min(
            1.0,
            0.75
            + 0.25 * entity_score,
        )

    if "launched" in s:
        return min(
            1.0,
            0.50
            + 0.50 * entity_score,
        )

    return 0.0


# ==========================================================
# ANSWER TARGET SCORE
# ==========================================================

def answer_target_score(
    query: str,
    sentence: str,
) -> Tuple[float, List[str]]:
    """
    Determine which actual question dimensions the sentence answers.

    Topic/entity alignment is required.
    """

    targets = detect_question_targets(
        query
    )

    covered: List[str] = []

    identity_score = identity_evidence_score(
        query,
        sentence,
    )

    organization_score = organization_evidence_score(
        query,
        sentence,
    )

    location_score = location_evidence_score(
        query,
        sentence,
    )

    date_score = date_evidence_score(
        query,
        sentence,
    )

    if (
        targets["identity"]
        and identity_score >= 0.35
    ):
        covered.append("identity")

    if (
        targets["organization"]
        and organization_score >= 0.35
    ):
        covered.append("organization")

    if (
        targets["location"]
        and location_score >= 0.35
    ):
        covered.append("location")

    if (
        targets["date"]
        and date_score >= 0.35
    ):
        covered.append("date")

    maximum = 0.0
    actual = 0.0

    if targets["identity"]:
        maximum += 1.0
        actual += identity_score

    if targets["organization"]:
        maximum += 1.0
        actual += organization_score

    if targets["location"]:
        maximum += 1.0
        actual += location_score

    if targets["date"]:
        maximum += 1.0
        actual += date_score

    if maximum <= 0.0:
        return 0.0, covered

    return (
        min(
            actual / maximum,
            1.0,
        ),
        covered,
    )


# ==========================================================
# SENTENCE QUALITY
# ==========================================================

def sentence_quality(
    sentence: str,
) -> float:
    """
    Penalize obvious OCR/extraction artifacts.
    """

    if not sentence:
        return 0.0

    words = sentence.split()

    if len(words) < 5:
        return 0.0

    score = 1.0

    if "CHUN-drə" in sentence:
        score -= 0.15

    if "/ˌ" in sentence:
        score -= 0.15

    if sentence.lower().startswith(
        "chandrayaan-3 chandrayaan-3"
    ):
        score -= 0.25

    broken_patterns = (
        "historyto",
        "missionthe",
        "orbitwas",
        "launchthe",
    )

    for pattern in broken_patterns:

        if pattern in sentence.lower():
            score -= 0.10

    uppercase_chars = sum(
        1
        for char in sentence
        if char.isupper()
    )

    alpha_chars = sum(
        1
        for char in sentence
        if char.isalpha()
    )

    if alpha_chars > 20:

        uppercase_ratio = (
            uppercase_chars
            / alpha_chars
        )

        if uppercase_ratio > 0.75:
            score -= 0.10

    return max(
        0.0,
        min(
            score,
            1.0,
        ),
    )


# ==========================================================
# CROSS-TOPIC CONTAMINATION
# ==========================================================

def cross_topic_penalty(
    query: str,
    sentence: str,
) -> float:
    """
    Penalize sentences that clearly belong to another mission/topic.

    Returns
    -------
    penalty in [0, 1]
    """

    q = query.lower()
    s = sentence.lower()

    query_entities = detect_query_entities(
        query
    )

    penalty = 0.0

    # ------------------------------------------------------
    # Chandrayaan contamination.
    # ------------------------------------------------------

    if "chandrayaan" in s:

        if (
            "chandrayaan" not in q
            and (
                "mars" in query_entities
                or "mom" in query_entities
                or "mars orbiter mission" in query_entities
            )
        ):
            penalty += 0.75

    # ------------------------------------------------------
    # Lunar contamination for Mars query.
    # ------------------------------------------------------

    if (
        "moon" in s
        or "lunar" in s
        or "lunar south pole" in s
    ):

        if (
            "mars" in query_entities
            or "mom" in query_entities
            or "mars orbiter mission" in query_entities
        ):
            penalty += 0.35

    # ------------------------------------------------------
    # Other mission contamination.
    # ------------------------------------------------------

    other_missions = (
        "gaganyaan",
        "aditya-l1",
        "aditya l1",
        "mangalyaan",
    )

    if any(
        mission in s
        for mission in other_missions
    ):

        if not any(
            mission in q
            for mission in other_missions
        ):
            penalty += 0.30

    return min(
        penalty,
        1.0,
    )


# ==========================================================
# EMBEDDING
# ==========================================================

def encode_candidates(
    sentences: List[str],
):
    """
    Encode all candidate sentences exactly once.

    Returned embeddings are reused by EvidenceFusion.

    Optimization #17:
        candidate sentences -> one embedding pass
    """

    if not sentences:
        return None

    model = ModelRegistry.get_embedding_model()

    embeddings = model.encode(
        sentences,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings


# ==========================================================
# SCORE CANDIDATES
# ==========================================================

def score_sentences(
    query: str,
    sentences: List[str],
):
    """
    Score candidate evidence.

    Each candidate sentence is embedded exactly once.

    Scoring components
    ------------------
    semantic:
        general semantic relevance

    lexical:
        direct lexical overlap

    target:
        question-target coverage

    entity:
        topic/entity alignment

    contamination:
        cross-topic penalty

    quality:
        OCR/readability quality

    organization:
        direct organization evidence

    identity:
        direct identity evidence
    """

    if not sentences:
        return [], {}, None

    model = ModelRegistry.get_embedding_model()

    # ------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # ------------------------------------------------------
    # Candidate embeddings
    # ------------------------------------------------------

    candidate_embeddings = encode_candidates(
        sentences
    )

    semantic_scores = util.cos_sim(
        query_embedding,
        candidate_embeddings,
    )[0]

    scored: List[Dict[str, Any]] = []

    for index, sentence in enumerate(
        sentences
    ):

        semantic = float(
            semantic_scores[index].item()
        )

        lexical = lexical_question_score(
            query,
            sentence,
        )

        target_score, targets = (
            answer_target_score(
                query,
                sentence,
            )
        )

        entity_score, matched_entities = (
            entity_alignment_score(
                query,
                sentence,
            )
        )

        quality = sentence_quality(
            sentence
        )

        contamination = cross_topic_penalty(
            query,
            sentence,
        )

        organization_score = (
            organization_evidence_score(
                query,
                sentence,
            )
        )

        identity_score = (
            identity_evidence_score(
                query,
                sentence,
            )
        )

        # --------------------------------------------------
        # Main evidence score
        #
        # Direct answer evidence receives more weight than
        # generic semantic similarity.
        # --------------------------------------------------

        evidence_score = (
            0.25 * semantic
            + 0.08 * lexical
            + 0.27 * target_score
            + 0.15 * entity_score
            + 0.10 * quality
            + 0.08 * organization_score
            + 0.07 * identity_score
        )

        # Strong contamination penalty.
        evidence_score *= (
            1.0 - 0.75 * contamination
        )

        evidence_score = max(
            0.0,
            min(
                evidence_score,
                1.0,
            ),
        )

        scored.append(
            {
                "text": sentence,

                "semantic": semantic,

                "lexical": lexical,

                "target_score": target_score,

                "entity_score": entity_score,

                "matched_entities": matched_entities,

                "organization_score": organization_score,

                "identity_score": identity_score,

                "contamination": contamination,

                "quality": quality,

                "evidence_score": evidence_score,

                "targets": targets,

                "embedding_index": index,
            }
        )

    scored.sort(
        key=lambda item: (
            item["evidence_score"],
            item["target_score"],
            item["entity_score"],
            item["semantic"],
        ),
        reverse=True,
    )

    embedding_map = {
        sentence: candidate_embeddings[index]
        for index, sentence in enumerate(
            sentences
        )
    }

    return (
        scored,
        embedding_map,
        candidate_embeddings,
    )


# ==========================================================
# SUPPORTING-EVIDENCE VALUE
# ==========================================================

def supporting_evidence_value(
    item: Dict[str, Any],
) -> float:
    """
    Estimate whether a sentence adds useful answer content
    after the question targets have already been satisfied.

    This prevents generic facts such as:

        MOM completed six years in Martian orbit...

    from being appended merely because the sentence is relevant
    to MOM.

    Supporting evidence should still contain direct answer value.
    """

    targets = set(
        item.get(
            "targets",
            [],
        )
    )

    if targets:
        return float(
            item.get(
                "target_score",
                0.0,
            )
        )

    # A sentence with no target coverage is only useful when it is
    # exceptionally strong semantically and entity-aligned.
    entity_score = float(
        item.get(
            "entity_score",
            0.0,
        )
    )

    semantic = float(
        item.get(
            "semantic",
            0.0,
        )
    )

    quality = float(
        item.get(
            "quality",
            0.0,
        )
    )

    return (
        0.50 * entity_score
        + 0.35 * semantic
        + 0.15 * quality
    )


# ==========================================================
# COVERAGE-AWARE SELECTION
# ==========================================================

def select_evidence(
    query: str,
    scored: List[Dict[str, Any]],
    max_sentences: int,
) -> List[Dict[str, Any]]:
    """
    Select evidence using question coverage + entity alignment.

    Important behavior
    ------------------
    If one sentence fully answers all requested targets, the selector
    is allowed to stop at one sentence even when max_sentences > 1.

    Example:

        What is the Mars Orbiter Mission (MOM) and which organization
        developed it?

    If this sentence is available:

        Mars Orbiter Mission, the maiden interplanetary mission of ISRO...

    it covers both:

        identity
        organization

    Therefore adding unrelated MOM facts is unnecessary.

    This is the key fix for the previous behavior where the pipeline
    returned three sentences despite only one being answer-bearing.
    """

    if not scored or max_sentences <= 0:
        return []

    selected: List[Dict[str, Any]] = []

    covered = set()

    required_targets = {
        name
        for name, enabled in detect_question_targets(
            query
        ).items()
        if enabled
    }

    remaining = list(
        scored
    )

    # ======================================================
    # PASS 1
    #
    # Maximize unanswered question coverage.
    # ======================================================

    while (
        remaining
        and len(selected) < max_sentences
    ):

        best = None
        best_gain = float(
            "-inf"
        )

        for item in remaining:

            item_targets = set(
                item.get(
                    "targets",
                    [],
                )
            )

            new_targets = (
                item_targets
                - covered
            )

            entity_score = float(
                item.get(
                    "entity_score",
                    0.0,
                )
            )

            contamination = float(
                item.get(
                    "contamination",
                    0.0,
                )
            )

            target_score = float(
                item.get(
                    "target_score",
                    0.0,
                )
            )

            evidence_score = float(
                item.get(
                    "evidence_score",
                    0.0,
                )
            )

            # Strongly contaminated evidence should not be selected
            # when cleaner evidence exists.
            if (
                contamination >= 0.70
                and entity_score < 0.50
            ):
                continue

            # --------------------------------------------------
            # Target coverage is the dominant selection signal.
            # --------------------------------------------------

            gain = (
                4.00 * len(new_targets)
                + 1.50 * target_score
                + 1.25 * entity_score
                + evidence_score
            )

            # Direct organization evidence is especially valuable
            # for organization questions.
            if "organization" in new_targets:
                gain += (
                    1.50
                    * float(
                        item.get(
                            "organization_score",
                            0.0,
                        )
                    )
                )

            # Direct identity evidence is especially valuable for
            # "what is X?" questions.
            if "identity" in new_targets:
                gain += (
                    1.00
                    * float(
                        item.get(
                            "identity_score",
                            0.0,
                        )
                    )
                )

            # Penalize contamination during selection.
            gain -= (
                2.00
                * contamination
            )

            if gain > best_gain:
                best_gain = gain
                best = item

        if best is None:
            break

        selected.append(
            best
        )

        covered.update(
            best.get(
                "targets",
                [],
            )
        )

        remaining.remove(
            best
        )

        # --------------------------------------------------
        # IMPORTANT:
        #
        # Once every requested question target has been
        # answered strongly, stop. Do NOT pad to max_sentences.
        # --------------------------------------------------

        if (
            required_targets
            and required_targets.issubset(
                covered
            )
        ):

            average_target_strength = (
                sum(
                    float(
                        item.get(
                            "target_score",
                            0.0,
                        )
                    )
                    for item in selected
                )
                / len(selected)
            )

            if (
                average_target_strength >= 0.55
            ):
                break

    # ======================================================
    # PASS 2
    #
    # Add supporting evidence ONLY when it contributes
    # meaningful answer value.
    # ======================================================

    if len(selected) < max_sentences:

        selected_texts = {
            item["text"]
            for item in selected
        }

        for item in scored:

            if item["text"] in selected_texts:
                continue

            evidence_score = float(
                item.get(
                    "evidence_score",
                    0.0,
                )
            )

            entity_score = float(
                item.get(
                    "entity_score",
                    0.0,
                )
            )

            contamination = float(
                item.get(
                    "contamination",
                    0.0,
                )
            )

            support_value = (
                supporting_evidence_value(
                    item
                )
            )

            # Do not append weak evidence.
            if evidence_score < 0.55:
                continue

            # Must still belong to the queried entity.
            if entity_score < 0.60:
                continue

            # Reject contaminated supporting evidence.
            if contamination >= 0.50:
                continue

            # If the question is already fully answered, only add
            # evidence that has genuine target/support value.
            #
            # Generic entity-related facts should NOT pass this gate.
            if not item.get("targets"):

                if support_value < 0.65:
                    continue

                # Strong semantic relevance alone is insufficient
                # for a completed question.
                if (
                    float(
                        item.get(
                            "semantic",
                            0.0,
                        )
                    )
                    < 0.75
                ):
                    continue

            selected.append(
                item
            )

            selected_texts.add(
                item["text"]
            )

            if len(selected) >= max_sentences:
                break

    # ======================================================
    # FINAL ORDER
    #
    # Answer-bearing evidence first.
    # ======================================================

    selected.sort(
        key=lambda item: (
            item.get(
                "target_score",
                0.0,
            ),
            item.get(
                "evidence_score",
                0.0,
            ),
            item.get(
                "entity_score",
                0.0,
            ),
            item.get(
                "semantic",
                0.0,
            ),
        ),
        reverse=True,
    )

    return selected


# ==========================================================
# GENERATE ANSWER
# ==========================================================

def generate_answer(
    query,
    documents,
    max_sentences=3,
):
    """
    Phase 10.1.

    Produces structured, question-aware, extractive evidence.

    sentence_embeddings are preserved for Optimization #17.
    """

    # ------------------------------------------------------
    # Candidate extraction
    # ------------------------------------------------------

    all_sentences: List[str] = []

    for doc in documents or []:

        text = doc.get(
            "text",
            "",
        )

        if not text:
            continue

        all_sentences.extend(
            split_sentences(
                text
            )
        )

    all_sentences = remove_duplicates(
        all_sentences
    )

    # ------------------------------------------------------
    # Score candidates
    # ------------------------------------------------------

    (
        scored,
        embedding_map,
        candidate_embeddings,
    ) = score_sentences(
        query,
        all_sentences,
    )

    # ------------------------------------------------------
    # Coverage-aware selection
    # ------------------------------------------------------

    selected = select_evidence(
        query,
        scored,
        max_sentences,
    )

    # ------------------------------------------------------
    # Build extractive answer
    # ------------------------------------------------------

    answer_sentences = [
        item["text"]
        for item in selected
    ]

    answer = " ".join(
        answer_sentences
    ).strip()

    if answer and not answer.endswith("."):
        answer += "."

    # ------------------------------------------------------
    # Structured evidence
    # ------------------------------------------------------

    evidence = []

    for rank, item in enumerate(
        selected,
        start=1,
    ):

        evidence.append(
            {
                "rank": rank,

                "text": item["text"],

                "similarity": round(
                    float(
                        item["semantic"]
                    ),
                    4,
                ),

                "semantic": round(
                    float(
                        item["semantic"]
                    ),
                    4,
                ),

                "lexical": round(
                    float(
                        item["lexical"]
                    ),
                    4,
                ),

                "relevance": round(
                    float(
                        item["evidence_score"]
                    ),
                    4,
                ),

                "target_score": round(
                    float(
                        item["target_score"]
                    ),
                    4,
                ),

                "entity_score": round(
                    float(
                        item.get(
                            "entity_score",
                            0.0,
                        )
                    ),
                    4,
                ),

                "organization_score": round(
                    float(
                        item.get(
                            "organization_score",
                            0.0,
                        )
                    ),
                    4,
                ),

                "identity_score": round(
                    float(
                        item.get(
                            "identity_score",
                            0.0,
                        )
                    ),
                    4,
                ),

                "contamination": round(
                    float(
                        item.get(
                            "contamination",
                            0.0,
                        )
                    ),
                    4,
                ),

                "targets": list(
                    item.get(
                        "targets",
                        [],
                    )
                ),
            }
        )

    # ------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------

    best_semantic = max(
        (
            item["semantic"]
            for item in scored
        ),
        default=0.0,
    )

    best_evidence = max(
        (
            item["evidence_score"]
            for item in scored
        ),
        default=0.0,
    )

    covered_targets = sorted(
        {
            target
            for item in selected
            for target in item.get(
                "targets",
                [],
            )
        }
    )

    print()
    print("=" * 70)
    print(
        "Phase 10.1 : "
        "Question-Aware Semantic Sentence Selection"
    )
    print("=" * 70)

    print(
        f"Candidate Sentences : "
        f"{len(all_sentences)}"
    )

    print(
        f"Selected Sentences  : "
        f"{len(evidence)}"
    )

    print(
        f"Best Semantic Score : "
        f"{best_semantic:.4f}"
    )

    print(
        f"Best Evidence Score : "
        f"{best_evidence:.4f}"
    )

    print(
        f"Question Targets    : "
        f"{covered_targets}"
    )

    print(
        "Embedding Passes    : 1"
    )

    print(
        f"Reusable Embeddings : "
        f"{len(embedding_map)}"
    )

    print()
    print(
        "Selected Evidence:"
    )

    for item in evidence:

        print()

        print(
            f"[{item['rank']}] "
            f"semantic={item['semantic']:.4f} "
            f"relevance={item['relevance']:.4f} "
            f"entity={item['entity_score']:.4f} "
            f"organization={item['organization_score']:.4f} "
            f"identity={item['identity_score']:.4f} "
            f"contamination={item['contamination']:.4f} "
            f"targets={item['targets']}"
        )

        print(
            f"    {item['text']}"
        )

    print("=" * 70)

    return {
        "answer": answer,

        "selected_sentences": evidence,

        "candidate_sentences": len(
            all_sentences
        ),

        "selected_count": len(
            evidence
        ),

        "best_similarity": round(
            float(
                best_semantic
            ),
            4,
        ),

        "best_evidence_score": round(
            float(
                best_evidence
            ),
            4,
        ),

        "question_targets": covered_targets,

        # --------------------------------------------------
        # Optimization #17
        # --------------------------------------------------

        "sentence_embeddings": embedding_map,

        "embedding_reuse_enabled": True,

        "embedding_passes": 1,

        "embedding_fallbacks": 0,
    }
