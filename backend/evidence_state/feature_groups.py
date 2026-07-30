from backend.evidence_state.state.evidence_state import (
    FeatureGroup,
)


# ==========================================================
# BUILD FEATURE GROUPS
# ==========================================================

def build_feature_groups(
    evidence_state,
):
    """
    Organizes Evidence Features into logical groups.

    Groups

        Retrieval
        Graph
        Quality
        Ranking
        Diagnostics
    """

    groups = {

        "retrieval": FeatureGroup(
            group_name="Retrieval",
        ),

        "graph": FeatureGroup(
            group_name="Graph",
        ),

        "quality": FeatureGroup(
            group_name="Quality",
        ),

        "ranking": FeatureGroup(
            group_name="Ranking",
        ),

        "diagnostics": FeatureGroup(
            group_name="Diagnostics",
        ),

    }

    # ------------------------------------------------------
    # Assign Features
    # ------------------------------------------------------

    for feature in evidence_state.features:

        source = feature.source.lower()

        if source in groups:

            groups[source].features.append(
                feature
            )

    # ------------------------------------------------------
    # Compute Group Scores
    # ------------------------------------------------------

    for group in groups.values():

        if group.features:

            group.score = (

                sum(

                    feature.value

                    for feature in group.features

                )

                /

                len(group.features)

            )

    # ------------------------------------------------------
    # Store Groups
    # ------------------------------------------------------

    evidence_state.groups = list(
        groups.values()
    )

    # ------------------------------------------------------
    # Fast Lookup
    # ------------------------------------------------------

    evidence_state.group_lookup = {

        group.group_name: group

        for group in evidence_state.groups

    }

    return evidence_state