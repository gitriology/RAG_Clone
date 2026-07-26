"""
Pipeline entry point for MS-ARC.

Each module updates the RetrievalState object.
"""

from ms_arc.state.retrieval_state import RetrievalState


def run_msarc(query: str) -> RetrievalState:
    """
    Main MS-ARC pipeline.

    The implementation will be completed gradually as each module
    is developed.
    """

    state = RetrievalState(query=query)

    return state