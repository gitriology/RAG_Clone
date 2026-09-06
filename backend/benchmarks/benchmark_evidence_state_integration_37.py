"""Synthetic control-flow benchmark for Optimization #37.

This benchmark measures how many full EvidenceState constructions are avoided
when the cheap #36 state is allowed to stop retrieval before the full graph and
EvidenceState are constructed. It intentionally does not claim end-to-end
latency improvement.
"""

from __future__ import annotations


def run_simulation(total_queries: int = 1000, cheap_stop_rate: float = 0.70):
    if total_queries <= 0:
        raise ValueError("total_queries must be positive")
    if not 0.0 <= cheap_stop_rate <= 1.0:
        raise ValueError("cheap_stop_rate must be between 0 and 1")

    cheap_stops = int(total_queries * cheap_stop_rate)
    ambiguous = total_queries - cheap_stops

    eager_full_builds = total_queries
    demand_driven_full_builds = ambiguous
    avoided = eager_full_builds - demand_driven_full_builds
    reduction = 100.0 * avoided / eager_full_builds

    return eager_full_builds, demand_driven_full_builds, avoided, reduction


def main():
    total, full, avoided, reduction = run_simulation()
    print("Optimization #37 — Evidence State integration control-flow benchmark")
    print()
    print(f"Synthetic queries: {total}")
    print(f"Eager full EvidenceState builds: {total}")
    print(f"Demand-driven full EvidenceState builds: {full}")
    print(f"Full EvidenceState builds avoided: {avoided}")
    print(f"Measured full-build reduction: {reduction:.2f}%")
    print()
    print("Note: synthetic control-flow benchmark; not end-to-end RAG latency.")


if __name__ == "__main__":
    main()
