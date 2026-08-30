"""
Optimization #21
----------------

Production Evidence Graph + Evidence State integration test.

This test verifies that the research modules are not merely
working in isolation.

It verifies that the actual production pipeline:

    run_pipeline()

        ↓

    MS-ARC

        ↓

    Evidence Graph

        ↓

    Evidence State

contains the integrated research artifacts in its result.
"""

from unittest.mock import patch

from backend.reranker.scoring.run_rerank import (
    run_pipeline,
)


# ==========================================================
# TEST CONFIGURATION
# ==========================================================

TEST_QUERY = (
    "What is vaccination?"
)


# ==========================================================
# TEST
# ==========================================================

def main():

    print()
    print("=" * 70)
    print(
        "PRODUCTION EVIDENCE INTEGRATION #21"
    )
    print("=" * 70)

    # ------------------------------------------------------
    # Track production integration.
    #
    # We patch the imported functions in run_rerank.py,
    # because that is where production actually calls them.
    # ------------------------------------------------------

    with patch(
        "backend.reranker.scoring.run_rerank.build_graph"
    ) as mock_build_graph, patch(
        "backend.reranker.scoring.run_rerank.build_evidence_state"
    ) as mock_build_evidence_state:

        # --------------------------------------------------
        # Mock Evidence Graph
        # --------------------------------------------------

        class MockGraphStatistics:
            graph_density = 0.5


        class MockGraphSignalsGraph:
            graph_score = 0.5
            coherence_score = 0.5
            evidence_quality = 0.5


        class MockGraphSignalsRanking:
            graph_consensus = 0.5


        class MockGraphSignals:
            graph = (
                MockGraphSignalsGraph()
            )

            ranking = (
                MockGraphSignalsRanking()
            )


        class MockValidation:
            score = 1.0
            valid = True


        class MockGraphState:
            nodes = [
                "node-1",
                "node-2",
            ]

            edges = [
                "edge-1",
            ]

            statistics = (
                MockGraphStatistics()
            )

            signals = (
                MockGraphSignals()
            )

            validation = (
                MockValidation()
            )

            # --------------------------------------------------
            # Important:
            #
            # run_rerank.build_evidence_graph_result()
            # prefers an actual NetworkX graph when available.
            #
            # We intentionally do NOT provide one here so that
            # the helper falls back to len(nodes)/len(edges).
            # --------------------------------------------------

            graph = None

            graph_score = 0.5


        # --------------------------------------------------
        # Mock Evidence State
        # --------------------------------------------------

        class MockReasoning:

            ready = True

            readiness_score = 0.75


        class MockEvidenceState:

            features = [
                "feature-1",
                "feature-2",
                "feature-3",
            ]

            groups = [
                "retrieval",
                "graph",
            ]

            vector = [
                0.1,
                0.2,
                0.3,
            ]

            weighted_vector = [
                0.05,
                0.10,
                0.15,
            ]

            evidence_score = 0.65

            profile_score = 0.75

            health_score = 0.90

            uncertainty_score = 0.10

            reasoning = (
                MockReasoning()
            )


        # --------------------------------------------------
        # Configure mocked production builders.
        # --------------------------------------------------

        mock_build_graph.return_value = (
            MockGraphState()
        )

        mock_build_evidence_state.return_value = (
            MockEvidenceState()
        )


        # --------------------------------------------------
        # Run production pipeline.
        # --------------------------------------------------

        results = run_pipeline(
            TEST_QUERY
        )


        # --------------------------------------------------
        # Basic result validation.
        # --------------------------------------------------

        if not results:

            raise AssertionError(
                "Production pipeline returned "
                "no results."
            )


        result = results[0]


        # ==================================================
        # VERIFY GRAPH INTEGRATION
        # ==================================================

        if mock_build_graph.call_count != 1:

            raise AssertionError(
                "Evidence Graph builder was not "
                "called exactly once. "
                f"Calls={mock_build_graph.call_count}"
            )


        # ==================================================
        # VERIFY EVIDENCE STATE INTEGRATION
        # ==================================================

        if (
            mock_build_evidence_state.call_count
            != 1
        ):

            raise AssertionError(
                "Evidence State builder was not "
                "called exactly once. "
                f"Calls={mock_build_evidence_state.call_count}"
            )


        # --------------------------------------------------
        # Verify the same RetrievalState and GraphState
        # were passed into Evidence State.
        #
        # This is important for Optimization #21.
        # --------------------------------------------------

        evidence_state_call = (
            mock_build_evidence_state.call_args
        )

        if evidence_state_call is None:

            raise AssertionError(
                "Evidence State builder call "
                "could not be inspected."
            )


        evidence_state_args = (
            evidence_state_call.args
        )


        if len(evidence_state_args) != 2:

            raise AssertionError(
                "Evidence State builder must receive "
                "exactly two positional arguments: "
                "RetrievalState and EvidenceGraphState."
            )


        evidence_state_retrieval_arg = (
            evidence_state_args[0]
        )

        evidence_state_graph_arg = (
            evidence_state_args[1]
        )


        # --------------------------------------------------
        # The production graph builder receives the same
        # RetrievalState.
        # --------------------------------------------------

        graph_call = (
            mock_build_graph.call_args
        )

        if graph_call is None:

            raise AssertionError(
                "Evidence Graph builder call "
                "could not be inspected."
            )


        graph_args = (
            graph_call.args
        )


        if len(graph_args) != 1:

            raise AssertionError(
                "Evidence Graph builder must receive "
                "exactly one RetrievalState argument."
            )


        graph_retrieval_arg = (
            graph_args[0]
        )


        if (
            evidence_state_retrieval_arg
            is not graph_retrieval_arg
        ):

            raise AssertionError(
                "Optimization #21 failed: "
                "Evidence State did not reuse the "
                "same RetrievalState passed to "
                "the Evidence Graph."
            )


        # --------------------------------------------------
        # The graph object returned by build_graph()
        # must be passed directly into Evidence State.
        # --------------------------------------------------

        graph_returned = (
            mock_build_graph.return_value
        )


        if (
            evidence_state_graph_arg
            is not graph_returned
        ):

            raise AssertionError(
                "Evidence State did not receive the "
                "same EvidenceGraphState returned "
                "by build_graph()."
            )


        # ==================================================
        # VERIFY RESULT CONTAINS GRAPH
        # ==================================================

        graph = result.get(
            "evidence_graph"
        )


        if not isinstance(
            graph,
            dict,
        ):

            raise AssertionError(
                "Production result does not "
                "contain evidence_graph."
            )


        if graph.get(
            "node_count"
        ) != 2:

            raise AssertionError(
                "Unexpected graph node count: "
                f"{graph.get('node_count')}"
            )


        if graph.get(
            "edge_count"
        ) != 1:

            raise AssertionError(
                "Unexpected graph edge count: "
                f"{graph.get('edge_count')}"
            )


        # ==================================================
        # VERIFY RESULT CONTAINS EVIDENCE STATE
        # ==================================================

        evidence_state = result.get(
            "evidence_state"
        )


        if not isinstance(
            evidence_state,
            dict,
        ):

            raise AssertionError(
                "Production result does not "
                "contain evidence_state."
            )


        # --------------------------------------------------
        # Feature count
        # --------------------------------------------------

        if evidence_state.get(
            "feature_count"
        ) != 3:

            raise AssertionError(
                "Unexpected evidence feature count: "
                f"{evidence_state.get('feature_count')}"
            )


        # --------------------------------------------------
        # Group count
        # --------------------------------------------------

        if evidence_state.get(
            "group_count"
        ) != 2:

            raise AssertionError(
                "Unexpected evidence group count: "
                f"{evidence_state.get('group_count')}"
            )


        # --------------------------------------------------
        # IMPORTANT FIX
        #
        # The production result contract uses:
        #
        #     vector_dimensions
        #
        # NOT:
        #
        #     vector_length
        #
        # build_evidence_state_result() derives this
        # from len(evidence_state.vector) when the
        # explicit vector_dimensions attribute is absent.
        # --------------------------------------------------

        if evidence_state.get(
            "vector_dimensions"
        ) != 3:

            raise AssertionError(
                "Unexpected evidence vector dimensions: "
                f"{evidence_state.get('vector_dimensions')}"
            )


        # --------------------------------------------------
        # Weighted vector dimensions
        # --------------------------------------------------

        if evidence_state.get(
            "weighted_vector_dimensions"
        ) != 3:

            raise AssertionError(
                "Unexpected weighted evidence vector "
                "dimensions: "
                f"{evidence_state.get('weighted_vector_dimensions')}"
            )


        # --------------------------------------------------
        # Evidence score
        # --------------------------------------------------

        evidence_score = evidence_state.get(
            "evidence_score"
        )


        if evidence_score != 0.65:

            raise AssertionError(
                "Unexpected evidence score: "
                f"{evidence_score}"
            )


        # --------------------------------------------------
        # Diagnostic values
        # --------------------------------------------------

        if evidence_state.get(
            "profile_score"
        ) != 0.75:

            raise AssertionError(
                "Unexpected evidence profile score: "
                f"{evidence_state.get('profile_score')}"
            )


        if evidence_state.get(
            "health_score"
        ) != 0.90:

            raise AssertionError(
                "Unexpected evidence health score: "
                f"{evidence_state.get('health_score')}"
            )


        if evidence_state.get(
            "uncertainty_score"
        ) != 0.10:

            raise AssertionError(
                "Unexpected evidence uncertainty score: "
                f"{evidence_state.get('uncertainty_score')}"
            )


        # ==================================================
        # SUCCESS
        # ==================================================

        print()

        print(
            "Production RetrievalState reuse : PASS"
        )

        print(
            "Evidence Graph invocation       : PASS"
        )

        print(
            "Evidence State invocation       : PASS"
        )

        print(
            "RetrievalState propagation      : PASS"
        )

        print(
            "EvidenceGraphState propagation  : PASS"
        )

        print(
            "Graph result propagation        : PASS"
        )

        print(
            "Evidence result propagation     : PASS"
        )

        print(
            "Evidence vector dimensions     : PASS"
        )


    print()

    print("=" * 70)

    print(
        "Optimization #21 : PASS"
    )

    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()