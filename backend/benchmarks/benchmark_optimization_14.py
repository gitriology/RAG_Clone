from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

CACHE_PATH = (
    PROJECT_ROOT
    / "backend"
    / "retrieval"
    / "index"
    / "bm25_cache.pkl"
)

RETRIEVE_MODULE = (
    "backend.ms_arc.retrieval.retrieve"
)


# ==========================================================
# CONFIGURATION
# ==========================================================

RUNS = 3

QUERY = "What is WHO?"

TOP_K = 3


# ==========================================================
# RUN RETRIEVAL PROCESS
# ==========================================================

def run_process(
    force_rebuild: bool,
):
    """
    Start a completely fresh Python process.

    This is important because importing retrieve.py
    initializes the retrieval engine at module level.

    A fresh process gives us a real startup measurement.
    """

    env = os.environ.copy()

    if force_rebuild:

        env[
            "OPT14_FORCE_BM25_REBUILD"
        ] = "1"

    else:

        env.pop(
            "OPT14_FORCE_BM25_REBUILD",
            None,
        )

    command = [

        sys.executable,

        "-m",

        RETRIEVE_MODULE,

    ]

    start = time.perf_counter()

    completed = subprocess.run(

        command,

        cwd=str(
            PROJECT_ROOT
        ),

        env=env,

        capture_output=True,

        text=True,

    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        elapsed,
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )


# ==========================================================
# PARSE STARTUP TIME
# ==========================================================

def extract_startup_time(
    output: str,
):
    """
    Extract:

        Total engine startup: Xs
    """

    marker = (
        "Total engine startup:"
    )

    for line in output.splitlines():

        if marker in line:

            try:

                value = (
                    line.split(
                        marker,
                        1,
                    )[1]
                    .strip()
                    .rstrip("s")
                )

                return float(
                    value
                )

            except (
                ValueError,
                IndexError,
            ):

                pass

    return None


# ==========================================================
# PARSE BM25 TIME
# ==========================================================

def extract_bm25_time(
    output: str,
):
    """
    Extract:

        BM25 initialization: Xs
    """

    marker = (
        "BM25 initialization:"
    )

    for line in output.splitlines():

        if marker in line:

            try:

                value = (
                    line.split(
                        marker,
                        1,
                    )[1]
                    .strip()
                    .rstrip("s")
                )

                return float(
                    value
                )

            except (
                ValueError,
                IndexError,
            ):

                pass

    return None


# ==========================================================
# RUN MULTIPLE TIMES
# ==========================================================

def benchmark_mode(
    force_rebuild: bool,
    runs: int = RUNS,
):
    """
    Run several fresh processes and collect startup
    measurements.

    The first run may include filesystem/cache effects,
    so the median is reported as the main measurement.
    """

    results = []

    print()

    if force_rebuild:

        print(
            "MODE: BEFORE "
            "(BM25 rebuilt every startup)"
        )

    else:

        print(
            "MODE: AFTER "
            "(persistent BM25 cache)"
        )

    print(
        "--------------------------------------------------"
    )

    for run_number in range(
        1,
        runs + 1,
    ):

        (
            elapsed,
            return_code,
            stdout,
            stderr,
        ) = run_process(
            force_rebuild=force_rebuild
        )

        startup_time = (
            extract_startup_time(
                stdout
            )
        )

        bm25_time = (
            extract_bm25_time(
                stdout
            )
        )

        if return_code != 0:

            print(
                f"Run {run_number}: FAILED"
            )

            print(
                stderr
            )

            raise RuntimeError(
                "Benchmark process failed."
            )

        results.append({

            "process_time":
                elapsed,

            "startup_time":
                startup_time,

            "bm25_time":
                bm25_time,

        })

        print(
            f"Run {run_number}: "
            f"process={elapsed:.4f}s "
            f"startup="
            f"{startup_time:.4f}s "
            f"bm25="
            f"{bm25_time:.4f}s"
        )

    return results


# ==========================================================
# MEDIAN
# ==========================================================

def median(
    values,
):
    values = sorted(
        values
    )

    n = len(values)

    if n == 0:

        return None

    middle = n // 2

    if n % 2 == 1:

        return values[middle]

    return (
        values[middle - 1]
        + values[middle]
    ) / 2.0


# ==========================================================
# CALCULATE IMPROVEMENT
# ==========================================================

def percentage_reduction(
    before,
    after,
):

    if before <= 0:

        return 0.0

    return (
        (before - after)
        / before
    ) * 100.0


# ==========================================================
# MAIN BENCHMARK
# ==========================================================

def main():

    print()
    print(
        "=================================================="
    )

    print(
        "OPTIMIZATION #14"
    )

    print(
        "BM25 PERSISTENCE / STARTUP BENCHMARK"
    )

    print(
        "=================================================="
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"BM25 cache:   {CACHE_PATH}"
    )

    print(
        f"Runs/mode:    {RUNS}"
    )

    # ------------------------------------------------------
    # IMPORTANT:
    #
    # We need a cache present for the AFTER benchmark.
    #
    # First run the normal engine once. This creates the
    # cache if it does not already exist.
    # ------------------------------------------------------

    print()
    print(
        "Preparing persistent BM25 cache..."
    )

    (
        prepare_elapsed,
        prepare_code,
        prepare_stdout,
        prepare_stderr,
    ) = run_process(
        force_rebuild=False
    )

    if prepare_code != 0:

        print(
            prepare_stderr
        )

        raise RuntimeError(
            "Could not prepare BM25 cache."
        )

    if not CACHE_PATH.exists():

        raise RuntimeError(
            "BM25 cache was not created."
        )

    print(
        "BM25 cache ready."
    )

    print(
        f"Preparation process: "
        f"{prepare_elapsed:.4f}s"
    )

    # ------------------------------------------------------
    # BEFORE
    # ------------------------------------------------------

    before = benchmark_mode(
        force_rebuild=True
    )

    # ------------------------------------------------------
    # AFTER
    # ------------------------------------------------------

    after = benchmark_mode(
        force_rebuild=False
    )

    # ------------------------------------------------------
    # MEDIAN VALUES
    # ------------------------------------------------------

    before_startup = median(
        [
            item["startup_time"]
            for item in before
            if item["startup_time"]
            is not None
        ]
    )

    after_startup = median(
        [
            item["startup_time"]
            for item in after
            if item["startup_time"]
            is not None
        ]
    )

    before_bm25 = median(
        [
            item["bm25_time"]
            for item in before
            if item["bm25_time"]
            is not None
        ]
    )

    after_bm25 = median(
        [
            item["bm25_time"]
            for item in after
            if item["bm25_time"]
            is not None
        ]
    )

    startup_reduction = (
        percentage_reduction(
            before_startup,
            after_startup,
        )
    )

    bm25_reduction = (
        percentage_reduction(
            before_bm25,
            after_bm25,
        )
    )

    startup_saved = (
        before_startup
        - after_startup
    )

    bm25_saved = (
        before_bm25
        - after_bm25
    )

    # ------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------

    print()
    print(
        "=================================================="
    )

    print(
        "OPTIMIZATION #14 RESULTS"
    )

    print(
        "=================================================="
    )

    print()
    print(
        "BM25 initialization"
    )

    print(
        f"Before: "
        f"{before_bm25:.4f}s"
    )

    print(
        f"After:  "
        f"{after_bm25:.4f}s"
    )

    print(
        f"Saved:  "
        f"{bm25_saved:.4f}s"
    )

    print(
        f"Reduction: "
        f"{bm25_reduction:.2f}%"
    )

    print()
    print(
        "Total MS-ARC engine startup"
    )

    print(
        f"Before: "
        f"{before_startup:.4f}s"
    )

    print(
        f"After:  "
        f"{after_startup:.4f}s"
    )

    print(
        f"Saved:  "
        f"{startup_saved:.4f}s"
    )

    print(
        f"Reduction: "
        f"{startup_reduction:.2f}%"
    )

    print()

    if after_startup < before_startup:

        print(
            "RESULT: Optimization #14 "
            "improved startup time."
        )

    elif after_startup == before_startup:

        print(
            "RESULT: No measurable startup "
            "improvement in this run."
        )

    else:

        print(
            "RESULT: Cached startup was slower "
            "in this benchmark."
        )

    print()
    print(
        "=================================================="
    )

    # ------------------------------------------------------
    # SAVE MACHINE-READABLE RESULTS
    # ------------------------------------------------------

    output = {

        "optimization":
            "Optimization #14",

        "experiment":
            "BM25 persistent cache "
            "before-vs-after startup benchmark",

        "configuration": {

            "runs_per_mode":
                RUNS,

            "cache_path":
                str(CACHE_PATH),

        },

        "before": {

            "description":
                "BM25 rebuilt on every startup",

            "runs":
                before,

            "median_bm25_initialization_seconds":
                before_bm25,

            "median_total_startup_seconds":
                before_startup,

        },

        "after": {

            "description":
                "BM25 loaded from persistent cache",

            "runs":
                after,

            "median_bm25_initialization_seconds":
                after_bm25,

            "median_total_startup_seconds":
                after_startup,

        },

        "improvement": {

            "bm25_seconds_saved":
                bm25_saved,

            "bm25_percentage_reduction":
                bm25_reduction,

            "startup_seconds_saved":
                startup_saved,

            "startup_percentage_reduction":
                startup_reduction,

        },

    }

    result_path = (
        PROJECT_ROOT
        / "optimization_14_results.json"
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    print()
    print(
        f"Results saved to:"
    )

    print(
        result_path
    )

    print()
    print(
        "Optimization #14 benchmark complete."
    )


if __name__ == "__main__":

    main()