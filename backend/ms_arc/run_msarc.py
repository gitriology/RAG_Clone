"""
Pipeline entry point for MS-ARC.
"""

from backend.ms_arc.state.retrieval_state import RetrievalState

from ms_arc.complexity.analyzer import QueryComplexityAnalyzer


def run_msarc(query: str) -> RetrievalState:

    state = RetrievalState(query=query)

    analyzer = QueryComplexityAnalyzer()

    state = analyzer.analyze(state)

    return state