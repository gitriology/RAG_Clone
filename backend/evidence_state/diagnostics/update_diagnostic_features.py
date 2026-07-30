from backend.evidence_state.features.diagnostics_features import (
    extract_diagnostics_features,
)


def update_diagnostic_features(
    evidence_state,
):
    """
    Refreshes (or appends) diagnostic features after
    diagnostics have been computed.
    """

    diagnostics = extract_diagnostics_features(
        evidence_state
    )

    lookup = {

        feature.name: feature

        for feature in evidence_state.features

    }

    for feature in diagnostics:

        if feature.name in lookup:

            existing = lookup[feature.name]

            existing.value = feature.value

            existing.source = feature.source

            existing.description = feature.description

        else:

            evidence_state.features.append(
                feature
            )

            lookup[feature.name] = feature

    evidence_state.feature_lookup = lookup

    return evidence_state