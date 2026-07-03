"""Smoke/behaviour tests for the evaluation harness (small, fast sizes)."""
from backend.analysis.experiments import (
    consensus_ablation,
    convergence_profile,
    load_benchmark,
    role_overlap,
)


def test_consensus_ablation_monotonic_coverage():
    res = consensus_ablation(panel_sizes=(1, 2, 3), tier=1)
    distinct = [r["distinct_critiques"] for r in res["rows"]]
    assert distinct == sorted(distinct) and distinct[0] < distinct[-1]
    # Larger panels reach agreement on at least one shared critique.
    assert res["rows"][-1]["multi_reviewer"] >= 1


def test_role_overlap_shows_complementary_lenses():
    res = role_overlap(tier=1)
    # Union is smaller than the naive sum (some overlap) but the lenses are
    # mostly complementary: at least one pair shares nothing.
    assert res["union_size"] < res["sum_individual"]
    assert any(p["jaccard"] == 0.0 for p in res["pairwise"])


def test_convergence_profile_reports_signal():
    res = convergence_profile(n_runs=3, max_iterations=20)
    assert res["iterations_to_converge"]["max"] <= 20
    assert sum(res["signal_counts"].values()) == 3


def test_load_benchmark_scales():
    res = load_benchmark(concurrency_levels=(1, 4), iterations_per_run=4,
                         adapter_delay=0.002)
    rows = {r["concurrency"]: r for r in res["rows"]}
    # Concurrency should raise aggregate throughput over the serial baseline.
    assert rows[4]["throughput_iters_per_s"] > rows[1]["throughput_iters_per_s"]
