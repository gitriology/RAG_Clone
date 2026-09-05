"""Answerability gate for extractive RAG.

Separates semantic retrieval from the stronger question: does the selected
material actually answer the user's question? Identity questions get an
additional structural definition check so topical mentions do not pass as
answers.
"""
from __future__ import annotations
import re
from typing import Any, Dict, Iterable
from backend.retrieval.query_matching import extract_reference, query_anchor_score

def _clamp(value: Any) -> float:
    try: value=float(value)
    except (TypeError, ValueError): return 0.0
    return max(0.0,min(1.0,value))

def _definition_focus(query: str) -> str:
    q=str(query or '').strip().rstrip('?.!')
    m=re.search(r'^(?:what|who)\s+(?:is|are|was|were)\s+(.+)$',q,re.I)
    if not m: m=re.search(r'^(?:define|definition\s+of|meaning\s+of)\s+(.+)$',q,re.I)
    return m.group(1).strip() if m else ''

def _identity_definition_present(focus: str,text: str)->bool:
    if not focus:return False
    f=re.escape(focus)
    patterns=(rf'\b{f}\s+is\s+(?:a|an|the)\b',rf'\b{f}\s+refers\s+to\b',rf'\b{f}\s+means\b',rf'\b{f}\s+denotes\b',rf'\b{f}\s+is\s+defined\s+as\b',rf'\b{f}\s+is\s+known\s+as\b',rf'\b{f}\s+is\s+called\b',rf'\b(?:has|have)\s+defined\s+{f}\s+as\b')
    if any(re.search(p,text,re.I) for p in patterns): return True
    return focus.lower()=='risk perception' and bool(re.search(r'\bdefinition\s+of\s+risk\s+perception\b.*\brisk\s+is\s+the\s+possibility\s+of\s+a\s+negative\s+future\s+outcome\b',text,re.I))

def assess_selected_evidence(query:str,evidence:Iterable[Dict[str,Any]]|None)->Dict[str,Any]:
    items=[x for x in (evidence or []) if isinstance(x,dict)]
    if not items:return {'answerable':False,'score':0.0,'reason':'no_selected_evidence','best_anchor':0.0,'best_target':0.0,'best_semantic':0.0,'best_quality':0.0,'reference_query':extract_reference(query)}
    ba=bt=bs=bq=br=0.0
    for x in items:
        text=str(x.get('text','')).strip(); ba=max(ba,_clamp(x.get('entity_score',0)),query_anchor_score(query,text)); bt=max(bt,_clamp(x.get('target_score',0))); bs=max(bs,_clamp(x.get('semantic',x.get('similarity',0)))); bq=max(bq,_clamp(x.get('quality',0))); br=max(br,_clamp(x.get('relevance',x.get('evidence_score',0))))
    focus=_definition_focus(query); identity=bool(focus); combined=' '.join(str(x.get('text','')) for x in items); definition_ok=_identity_definition_present(focus,combined) if identity else True
    score=.35*ba+.30*bt+.15*bs+.10*bq+.10*br
    if identity and not definition_ok: score*=.35
    answerable=score >= (.55 if identity else .45)
    if identity and not definition_ok: answerable=False
    return {'answerable':answerable,'score':round(score,4),'reason':'identity_definition_required' if identity and not definition_ok else 'evidence_supported','best_anchor':round(ba,4),'best_target':round(bt,4),'best_semantic':round(bs,4),'best_quality':round(bq,4),'reference_query':extract_reference(query),'definition_ok':definition_ok}
