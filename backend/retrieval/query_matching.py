"""Query/evidence matching helpers used by retrieval and answer selection.

The RAG pipeline is extractive, so retrieval confidence is not enough to
establish that a question is actually answerable.  This module provides
small, deterministic lexical/reference checks that complement embeddings.

Design goals:
- never treat a substring as an entity match (``modi`` != ``MODIS``)
- recognize numbered references such as ``Article 12`` / ``article 12.``
- recognize common question intents without depending on one exact wording
- keep the functions dependency-light so they can be used before generation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
REFERENCE_RE = re.compile(
    r"\b(article|articles|section|sections|chapter|chapters|clause|clauses|rule|rules|paragraph|paragraphs)\s*([0-9]+[A-Za-z]?)\b",
    re.I,
)
NUMBERED_PROVISION_RE = re.compile(
    r"(?:^|\s|\()([0-9]+[A-Za-z]?)\s*[.)]\s*",
    re.I,
)

QUESTION_WORDS = {
    "what", "who", "whom", "whose", "where", "when", "why", "how",
    "which", "does", "do", "did", "is", "are", "was", "were", "can",
    "could", "would", "should", "will", "may", "might", "tell", "give",
    "explain", "describe", "discuss", "compare", "contrast", "difference",
    "differences", "between", "versus", "vs", "and", "the", "a", "an",
    "of", "to", "for", "in", "on", "with", "from", "about", "me", "please",
}

INTENT_CUES = {
    "who": ("who", "whom", "whose"),
    "what": ("what", "which", "define", "definition", "meaning", "means"),
    "where": ("where", "location", "located", "landed", "landing"),
    "when": ("when", "date", "year", "period", "launched", "launch date", "started", "ended"),
    "why": ("why", "reason", "cause", "caused", "purpose", "because", "due to"),
    "how": ("how", "process", "method", "steps", "procedure", "works", "working", "implemented"),
    "compare": ("compare", "comparison", "contrast", "difference", "differences", "between", "versus", "vs"),
    "explain": ("explain", "explanation", "describe", "discuss", "elaborate", "detail", "details", "tell me about"),
}

DEFINITION_CUES = (
    " is ", " are ", " was ", " were ", " refers to ", " means ",
    "definition", "defined",
    " defined as ", " known as ", " consists of ", " is called ",
    " is the process of ", " is the action of ", " is a ", " is an ",
)

CAUSAL_CUES = (
    "because", "due to", "therefore", "resulted in", "resulting in",
    "caused", "cause", "reason", "purpose", "so that", "led to", "leads to",
)

PROCESS_CUES = (
    "by ", "using ", "through ", "process", "method", "steps", "procedure",
    "works", "working", "implemented", "performed", "carried out", "consists of",
    "first", "then", "finally", "learns", "learn ", "predict", "generate",
    "calculat", "transform", "classif", "receive", "take ", "uses ",
)

LOCATION_CUES = (
    " in ", " at ", " on ", " near ", "from ", "located", "based", "landed",
    "landing", "region", "country", "city", "state", "site", "pole",
)

DATE_CUES = (
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "launched", "launch",
    "started", "ended", "during", "between", "year", "date",
)


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_tokens(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def phrase_present(text: str, phrase: str) -> bool:
    """Boundary-aware phrase match.

    Unlike ``phrase in text``, this does not match a query entity inside a
    larger token.  It also tolerates punctuation/whitespace differences.
    """
    text_n = normalize_text(text)
    phrase_n = normalize_text(phrase)
    if not phrase_n:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(phrase_n).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    return re.search(pattern, text_n, re.I) is not None


def token_present(text: str, token: str) -> bool:
    return phrase_present(text, token)


def definition_subject_matches(query: str, text: str) -> bool:
    """Check whether the evidence defines the requested focus, not a longer
    neighboring phrase (e.g. "vaccination services" when asked "vaccination")."""
    analysis = analyze_query(query) if False else None
    focuses = extract_focus_phrases(query)
    if not focuses:
        return False
    primary = sorted(focuses, key=lambda p: (-len(p.split()), -len(p)))[0]
    s = normalize_text(text)
    if primary in extract_references(query):
        return reference_present(text, primary)
    # A one-token focus must be the grammatical subject immediately before a
    # definition cue; otherwise a longer phrase beginning with the token is
    # not a definition of the requested term.
    if " " not in primary:
        escaped = re.escape(primary)
        return bool(re.search(
            rf"(?<![a-z0-9]){escaped}(?![a-z0-9])\s+(?:is|are|was|were|refers\s+to|means|denotes|defined|known\s+as)\b",
            s,
            re.I,
        ))
    return phrase_present(text, primary)


def _canonical_reference(kind: str, number: str) -> str:
    kind = kind.lower().rstrip("s")
    return f"{kind} {number.lower()}"


def extract_references(text: str) -> List[str]:
    """Extract explicit numbered references such as ``Article 12``."""
    refs: List[str] = []
    for match in REFERENCE_RE.finditer(text or ""):
        refs.append(_canonical_reference(match.group(1), match.group(2)))
    return list(dict.fromkeys(refs))


def reference_present(text: str, reference: str) -> bool:
    """Match a named reference against both explicit and numbered headings.

    ``Article 12`` matches ``Article 12. Definition`` and ``12. Definition``
    but does not match ``124. ...`` or ``112. ...``.
    """
    ref_n = normalize_text(reference)
    m = re.fullmatch(
        r"(article|section|chapter|clause|rule|paragraph)\s*([0-9]+[a-z]?)",
        ref_n,
        re.I,
    )
    if not m:
        return phrase_present(text, reference)

    kind = m.group(1).lower()
    number = m.group(2).lower()
    if phrase_present(text, f"{kind} {number}"):
        return True

    # Constitutional/legal PDFs commonly omit the word "Article" in the
    # provision heading and render it as "12. Definition.—...".
    if kind == "article":
        pattern = rf"(?<!\d){re.escape(number)}\s*[.)]\s*"
        return re.search(pattern, normalize_text(text), re.I) is not None

    return False


def _looks_like_proper_phrase(tokens: Sequence[str]) -> bool:
    return len(tokens) >= 2


def extract_focus_phrases(query: str) -> List[str]:
    """Extract the terms that must be grounded in the evidence.

    Priority:
    1. explicit numbered references
    2. quoted phrases
    3. capitalized multi-word names / acronyms
    4. meaningful content words
    """
    query = "" if query is None else str(query).strip()
    if not query:
        return []

    phrases: List[str] = []
    references = extract_references(query)

    for ref in references:
        phrases.append(ref)

    for quoted in re.findall(r"[\"']([^\"']{2,100})[\"']", query):
        phrases.append(normalize_text(quoted))

    # Comparison questions are best represented as two independent focus
    # phrases. This lets retrieval deliberately collect evidence for both
    # sides instead of treating "compare" as part of the entity name.
    comparison_query = re.sub(
        r"^\s*(?:what is|what are|explain|describe|tell me about)?\s*",
        "",
        query,
        flags=re.I,
    )
    comparison_query = re.sub(
        r"^\s*(?:the\s+)?(?:difference|differences|comparison|compare|contrast)\s+(?:between\s+)?",
        "",
        comparison_query,
        flags=re.I,
    )
    comparison_split = r"\s+(?:and|versus|vs\.?|v\.|with|different from)\s+"
    if re.search(comparison_split, comparison_query, re.I):
        sides = re.split(comparison_split, comparison_query, maxsplit=1, flags=re.I)
        for side in sides:
            side = re.sub(r"[?!.]+$", "", side).strip()
            side_tokens = [t for t in normalize_tokens(side) if t not in QUESTION_WORDS and t not in {
                "difference", "differences", "comparison", "compare", "contrast"
            }]
            if side_tokens:
                phrases.append(" ".join(side_tokens))

    # For short definition/explanation queries, preserve the complete topic
    # phrase when it is contiguous. This handles "machine learning" and
    # "vaccination services" instead of reducing them to isolated tokens.
    topic_query = re.sub(
        r"^\s*(?:what|who|where|when|why|how)\s+(?:(?:is|are|was|were|does|do|did)\s+)?",
        "",
        query,
        flags=re.I,
    )
    topic_query = re.sub(r"[?!.]+$", "", topic_query).strip()
    topic_tokens = normalize_tokens(topic_query)
    blocked_topic_tokens = QUESTION_WORDS | {
        "different", "difference", "differences", "compare", "contrast",
        "developed", "developer", "built", "designed", "launched", "launch",
        "started", "ended", "important", "work", "works", "working",
    }
    if 2 <= len(topic_tokens) <= 6 and not any(t in blocked_topic_tokens for t in topic_tokens):
        phrases.append(" ".join(topic_tokens))

    # Preserve multi-word proper-name runs from the original query.
    raw_tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", query)
    run: List[str] = []
    for token in raw_tokens + [""]:
        if token and token.lower() not in QUESTION_WORDS and (token[:1].isupper() or (token.isupper() and len(token) >= 2)):
            run.append(token)
            continue
        if len(run) >= 2:
            phrases.append(normalize_text(" ".join(run)))
        run = []

    # Known reference-like constructions where punctuation separates the
    # words, e.g. "Article-12".
    for match in REFERENCE_RE.finditer(query):
        phrases.append(_canonical_reference(match.group(1), match.group(2)))

    content = []
    for token in normalize_tokens(query):
        if token in QUESTION_WORDS:
            continue
        if len(token) < 2:
            continue
        if token.isdigit() and any(token == r.split()[-1] for r in references):
            continue
        content.append(token)

    # Remove generic intent terms.
    generic = {
        "information", "info", "details", "detail", "thing", "things",
        "person", "people", "organization", "organisation", "company",
        "country", "city", "state", "date", "reason", "purpose", "process",
        "method", "steps", "difference", "differences", "comparison", "compare",
        "contrast", "different", "explain", "explanation", "describe", "discuss",
        "developed", "developer", "built", "designed", "launched", "launch",
        "started", "ended", "important", "work", "works", "working",
        "used", "use", "does", "did", "make", "made",
    }
    content = [t for t in content if t not in generic]

    phrases.extend(content)

    # Prefer longer/more specific phrases and de-duplicate shorter tokens
    # that are already covered by a multi-word focus phrase.
    normalized: List[str] = []
    for phrase in phrases:
        phrase = normalize_text(phrase)
        if not phrase or phrase in normalized:
            continue
        normalized.append(phrase)

    multi = [p for p in normalized if " " in p]
    result: List[str] = []
    for phrase in normalized:
        if " " not in phrase and any(phrase in p.split() for p in multi):
            # Keep acronyms and explicit references even if part of a phrase.
            if phrase.upper() == phrase and len(phrase) >= 2:
                result.append(phrase)
            elif phrase.isdigit():
                result.append(phrase)
            continue
        result.append(phrase)

    return result[:12]


def detect_intents(query: str) -> List[str]:
    q = normalize_text(query)
    padded = f" {q} "

    # Explicit leading interrogatives are authoritative. Avoid turning words
    # such as "launched" into a second intent for a "where" question.
    leading = re.match(r"^(?:please\s+)?(what|who|whom|whose|where|when|why|how)\b", q)
    primary = leading.group(1) if leading else None

    intents: List[str] = []
    if primary:
        intents.append("what" if primary == "what" else primary)

    if any(x in q for x in ("difference between", "differences between", "different from", "compare", "contrast", " versus ", " vs ")):
        intents.insert(0, "compare")

    if re.match(r"^what\s+does\b", q) and "how" not in intents:
        intents.append("how")

    if not primary and "compare" not in intents:
        for intent, cues in INTENT_CUES.items():
            for cue in cues:
                cue_n = normalize_text(cue)
                if (f" {cue_n} " in padded) if " " in cue_n else token_present(q, cue_n):
                    intents.append(intent)
                    break

    if not intents:
        intents.append("what")

    return list(dict.fromkeys(intents))


def analyze_query(query: str) -> Dict:
    intents = detect_intents(query)
    focus = extract_focus_phrases(query)
    references = extract_references(query)
    multi_entity = (
        len([p for p in focus if " " in p or p.isupper()]) >= 2
        or ("compare" in intents and len(focus) >= 2)
    )
    return {
        "intents": intents,
        "primary_intent": intents[0],
        "focus_phrases": focus,
        "references": references,
        "multi_entity": multi_entity,
        "requires_multiple_evidence": bool({"compare", "why", "how", "explain"} & set(intents)),
    }


def lexical_match_score(query: str, text: str) -> Dict[str, object]:
    """Return deterministic query/evidence lexical support signals."""
    analysis = analyze_query(query)
    focus = analysis["focus_phrases"]
    if not focus:
        return {"score": 0.0, "matched": [], "missing": [], "exact_phrase": False, "reference_match": False}

    matched: List[str] = []
    missing: List[str] = []
    exact_phrase = False
    reference_match = False

    for phrase in focus:
        ok = False
        if phrase in analysis["references"]:
            ok = reference_present(text, phrase)
            reference_match = reference_match or ok
        else:
            ok = phrase_present(text, phrase)
        if ok:
            matched.append(phrase)
            if " " in phrase:
                exact_phrase = True
        else:
            missing.append(phrase)

    coverage = len(matched) / max(1, len(focus))
    # Longer exact phrases are more trustworthy than a bag of isolated words.
    specificity_bonus = 0.0
    if exact_phrase:
        specificity_bonus += 0.15
    if reference_match:
        specificity_bonus += 0.25
    if any(p.isupper() and len(p) >= 2 for p in focus) and any(
        p in matched for p in focus if p.isupper()
    ):
        specificity_bonus += 0.10

    return {
        "score": min(1.0, coverage + specificity_bonus),
        "matched": matched,
        "missing": missing,
        "exact_phrase": exact_phrase,
        "reference_match": reference_match,
    }


def intent_cue_score(query: str, text: str) -> float:
    q = normalize_text(query)
    s = f" {normalize_text(text)} "
    intents = set(detect_intents(query))
    scores = []
    if "what" in intents or "explain" in intents:
        scores.append(1.0 if any(cue in s for cue in DEFINITION_CUES) else 0.0)
    if "why" in intents:
        scores.append(1.0 if any(cue in s for cue in CAUSAL_CUES) else 0.0)
    if "how" in intents:
        scores.append(1.0 if any(cue in s for cue in PROCESS_CUES) else 0.0)
    if "where" in intents:
        scores.append(1.0 if any(cue in s for cue in LOCATION_CUES) else 0.0)
    if "when" in intents:
        year = bool(re.search(r"\b(?:19|20)\d{2}\b", s))
        date = bool(re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", s))
        scores.append(1.0 if year or date or any(cue in s for cue in DATE_CUES) else 0.0)
    if "who" in intents:
        scores.append(1.0 if re.search(r"\b(?:is|was|are|were|by|from|named|called|born|served|founded|developed)\b", s) else 0.0)
    if "compare" in intents:
        # Comparison questions need evidence about both sides; the actual
        # two-sided requirement is checked separately in answerability.
        scores.append(1.0 if any(c in s for c in ("whereas", "while", "however", "difference", "similar", "unlike", "compared")) else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evidence_support_score(query: str, text: str) -> Dict[str, object]:
    lexical = lexical_match_score(query, text)
    cue = intent_cue_score(query, text)
    score = min(1.0, 0.72 * float(lexical["score"]) + 0.28 * cue)
    return {
        **lexical,
        "intent_cue": cue,
        "support": score,
    }


def assess_answerability(query: str, evidence: Iterable[object], answer: str = "") -> Dict:
    """Assess whether the supplied evidence really answers the query.

    This is deliberately stricter than semantic similarity.  A semantically
    related but different entity should not make the answerable score high.
    """
    analysis = analyze_query(query)
    items: List[str] = []
    for item in evidence or []:
        if isinstance(item, dict):
            value = item.get("text", "")
        else:
            value = str(item)
        if str(value).strip():
            items.append(str(value).strip())

    if not items:
        return {
            "answerable": False,
            "score": 0.0,
            "reason": "no_evidence",
            "focus_phrases": analysis["focus_phrases"],
            "matched_focus": [],
            "missing_focus": analysis["focus_phrases"],
            "intents": analysis["intents"],
        }

    combined = " ".join(items)
    support = evidence_support_score(query, combined)
    matched = support["matched"]
    missing = support["missing"]
    focus = analysis["focus_phrases"]

    # For explicit named entities/references, exact grounding is mandatory.
    # Treat explicit references, quoted phrases and the longest proper-name
    # phrases as the hard grounding anchors. Do not require every overlapping
    # bigram (e.g. "Mars Orbiter" AND "Orbiter Mission") independently.
    explicit_focus = list(analysis["references"])
    quoted = [normalize_text(x) for x in re.findall(r"[\"']([^\"']{2,100})[\"']", query)]
    explicit_focus.extend(quoted)
    if "compare" in analysis["intents"] and len(focus) >= 2:
        # Comparison focus phrases were already split into the two sides by
        # extract_focus_phrases; keep both sides as hard anchors.
        explicit_focus.extend(focus)
    multi = sorted(
        [p for p in focus if " " in p and p not in explicit_focus],
        key=lambda p: (-len(p.split()), -len(p)),
    )
    for phrase in multi:
        if not any(phrase in existing or existing in phrase for existing in explicit_focus):
            explicit_focus.append(phrase)
    explicit_focus = list(dict.fromkeys(explicit_focus))

    if explicit_focus:
        explicit_matched = [
            p for p in explicit_focus
            if (reference_present(combined, p) if p in analysis["references"] else phrase_present(combined, p))
        ]
        explicit_coverage = len(explicit_matched) / max(1, len(explicit_focus))
    else:
        explicit_coverage = 1.0

    if "compare" in analysis["intents"] and len(explicit_focus) >= 2:
        matched_entities = [
            p for p in explicit_focus
            if (reference_present(combined, p) if p in analysis["references"] else phrase_present(combined, p))
        ]
        entity_coverage = len(matched_entities) / max(1, len(explicit_focus))
    else:
        entity_coverage = explicit_coverage

    answer_support = 0.0
    if answer.strip():
        answer_support = float(evidence_support_score(query, answer)["support"])

    score = (
        0.55 * float(support["score"])
        + 0.25 * explicit_coverage
        + 0.10 * entity_coverage
        + 0.10 * answer_support
    )

    intents = set(analysis["intents"])
    if explicit_focus and explicit_coverage < 1.0:
        score *= 0.45
    if "compare" in intents and len(explicit_focus) >= 2 and entity_coverage < 1.0:
        score *= 0.35

    # A topic mention is not automatically an answer to a question. For
    # factoid intents require the evidence to exhibit the expected answer
    # structure as well as the topic.
    cue = float(support.get("intent_cue", 0.0))
    if "what" in intents and "compare" not in intents and "explain" not in intents:
        if not definition_subject_matches(query, combined):
            cue = min(cue, 0.0)
            score *= 0.45
    if "who" in intents and cue < 0.5:
        score *= 0.55
    if "where" in intents and cue < 0.5:
        score *= 0.55
    if "when" in intents and cue < 0.5:
        score *= 0.55
    if "why" in intents and cue < 0.5:
        score *= 0.50
    if "how" in intents and cue < 0.5:
        score *= 0.55

    score = max(0.0, min(1.0, score))

    if not matched:
        reason = "query_focus_not_found"
    elif explicit_focus and explicit_coverage < 1.0:
        reason = "explicit_query_focus_not_fully_grounded"
    elif "compare" in intents and len(explicit_focus) >= 2 and entity_coverage < 1.0:
        reason = "comparison_missing_one_side"
    elif cue < 0.5 and intents & {"what", "who", "where", "when", "why", "how"}:
        reason = "topic_present_but_answer_structure_missing"
    elif score >= 0.72:
        reason = "strong_grounding"
    elif score >= 0.48:
        reason = "partial_grounding"
    else:
        reason = "weak_grounding"

    return {
        "answerable": bool(
            score >= 0.52
            and (not explicit_focus or explicit_coverage >= 1.0)
            and ("compare" not in intents or len(explicit_focus) < 2 or entity_coverage >= 1.0)
            and (not (intents & {"what", "who", "where", "when", "why", "how"}) or cue >= 0.5 or "explain" in intents or "compare" in intents)
        ),
        "score": score,
        "reason": reason,
        "focus_phrases": focus,
        "matched_focus": matched,
        "missing_focus": missing,
        "explicit_focus": explicit_focus,
        "explicit_coverage": explicit_coverage,
        "entity_coverage": entity_coverage,
        "answer_support": answer_support,
        "intent_cue": cue,
        "definition_subject_match": definition_subject_matches(query, combined),
        "intents": analysis["intents"],
        "primary_intent": analysis["primary_intent"],
    }


def to_dict(value) -> Dict:
    if isinstance(value, QueryAnalysis):
        return asdict(value)
    return dict(value)


@dataclass
class QueryAnalysis:
    intents: List[str]
    primary_intent: str
    focus_phrases: List[str]
    references: List[str]
    multi_entity: bool
    requires_multiple_evidence: bool
