"""Regression tests: stopping rules from the frozen analysis plan can be
expressed and replayed (code-review finding tests-claims-7)."""
from __future__ import annotations

import pytest

from backend.orchestrator.convergence import (
    ALL_SIGNALS,
    ConvergenceDetector,
    StoppingRule,
    replay_stopping_rule,
)
from backend.orchestrator.dsl import parse_stop_conditions

#: "no-Critical/High AND (grade-stable OR similar)" (ANALYSIS_PLAN.md)
CONJ = StoppingRule(window=2, min_iterations=0, signals=ALL_SIGNALS,
                    required=("no_critical_high",), rule="any")


def _traj(cycles, grade, crit, high, same_output=True):
    ev, it = [], 0
    for c in range(cycles):
        it += 1
        ev.append({"iteration": it, "role": "executor",
                   "output": "same text" if same_output else f"text {c}"})
        it += 1
        ev.append({"iteration": it, "role": "consensus_merger", "grade": grade,
                   "critical": crit, "high": high})
    return ev


def test_conjunctive_rule_does_not_stop_while_high_remains():
    ev = _traj(5, "B", 0, 2)
    # approximations that DO stop with HIGH present
    k2 = StoppingRule(window=2, min_iterations=0, rule="k_of_n", k=2)
    assert replay_stopping_rule(ev, k2)["stopped"] is True
    assert replay_stopping_rule(ev, CONJ)["stopped"] is False


def test_conjunctive_rule_stops_when_clean_and_stable():
    ev = _traj(4, "A", 0, 0, same_output=False)
    out = replay_stopping_rule(ev, CONJ)
    assert out["stopped"] and "no_critical_high" in out["stop_reason"]
    assert "grade_stable" in out["stop_reason"]


def test_required_signals_validated_and_reported():
    with pytest.raises(ValueError):
        StoppingRule(signals=("grade_stable",), required=("no_critical_high",))
    r = StoppingRule.from_config({"convergence_required_signals": "no_critical_high"})
    assert r.required == ("no_critical_high",)
    assert r.to_dict()["required"] == ["no_critical_high"]
    assert "all of [no_critical_high]" in r.describe()
    stop = parse_stop_conditions(["convergence_required_signals:no_critical_high"])
    assert stop.config["convergence_required_signals"] == ["no_critical_high"]


def test_gated_conjunctive_rule_replay():
    ev = [
        {"iteration": 1, "role": "executor", "output": "x"},
        {"iteration": 2, "role": "consensus_merger", "grade": "A", "critical": 0, "high": 0},
        {"iteration": 3, "role": "executor", "output": "x"},
        {"iteration": 4, "role": "consensus_merger", "grade": "A", "critical": 0, "high": 0,
         "unresolved_critical": 1},
        {"iteration": 5, "role": "executor", "output": "x"},
    ]
    assert replay_stopping_rule(ev, CONJ)["iteration"] == 4
    assert replay_stopping_rule(ev, CONJ, gate=True)["stopped"] is False


def test_detector_only_required_signals():
    det = ConvergenceDetector(rule=StoppingRule(window=1, min_iterations=0,
                                                signals=("no_critical_high",),
                                                required=("no_critical_high",)))
    det.record_review("A", 0, 0, 1)
    assert det.evaluate().converged
