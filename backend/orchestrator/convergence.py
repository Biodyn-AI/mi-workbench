"""Convergence detection (configurable stopping rule) and the default stopping
policy of the loop engine.

Default stopping policy (calibrated)
------------------------------------
By default a run stops after a FIXED BUDGET of executor revisions,
:data:`DEFAULT_REVISION_BUDGET` = 4 (run-config key ``revision_budget``): the
initial executor submission plus at most four revisions, the last of which is
still reviewed. The adaptive rule below is off by default
(:data:`DEFAULT_CONVERGENCE_ENABLED` = False) and stays fully available
(``convergence_enabled: true``). The choice comes from the stopping
calibration (``experiments/stopping/results/stopping_results.json`` and
``stopping_rules.csv``): every candidate rule was replayed on 24 real
executor-panel trajectories (two model configurations, 12 planted-flaw items
each, horizon of five revisions). Pooled over both configurations, the
pre-specified selection rule (lowest premature-stop rate among rules that stop
before the horizon in >= 50 % of runs, ties broken by fewer revisions) selects
the fixed budget of four revisions (premature-stop rate 2/24 = 0.083). The
adaptive any-of-three rule (:data:`DEFAULT_STOPPING_RULE`, which coincided
with grade stability alone) had 3/24 = 0.125 at about three revisions; rules
requiring no Critical/High critiques or similar executor outputs never stopped
within five revisions.

Adaptive stopping rule
----------------------
Three observable signals are tracked:

``grade_stable``
    The last ``window`` review grades are identical (exact string). Every
    review step contributes one grade: a single reviewer lens step or a
    consensus step. A consensus step whose panel failed (grade
    ``INCOMPLETE``) is never recorded.
``no_critical_high``
    The last ``window`` review steps reported zero CRITICAL and zero HIGH
    critiques.
``output_similar``
    The word-level Jaccard similarity of each pair of CONSECUTIVE EXECUTOR
    outputs is at least ``similarity_threshold`` for the last ``window``
    pairs (``window + 1`` executor outputs). Reviewer, consensus feedback,
    idea-generator and other non-executor text never enters this signal.

A stopping rule combines the enabled signals (``signals``):

* ``rule="any"``: stop when at least one enabled signal holds (the original
  behaviour, and the current default);
* ``rule="all"``: stop when every enabled signal holds;
* ``rule="k_of_n"``: stop when at least ``k`` enabled signals hold.

``required`` (a subset of ``signals``) must ALL hold in addition; the
any / all / k_of_n combination then applies to the remaining enabled
signals. For example the pre-registered candidate
"no-Critical/High AND (grade-stable OR similar)" is
``signals=all three, required=("no_critical_high",), rule="any"``.

No rule can fire before ``min_iterations`` engine iterations (every executor
call and every review/consensus step counts as one iteration). With
``require_no_critical=True`` the rule additionally requires the most recent
review step to contain zero CRITICAL critiques.

The adaptive defaults live in ONE place, :data:`DEFAULT_STOPPING_RULE`. To
change them, edit that constant only; the engine, run configs and loop YAML
``stop_conditions`` all start from it. The default policy (whether the
adaptive rule runs at all, and the revision budget) is
:data:`DEFAULT_CONVERGENCE_ENABLED` and :data:`DEFAULT_REVISION_BUDGET`.

Run-config keys (see :meth:`StoppingRule.from_config`)::

    convergence_window                (int, default 3)
    convergence_min_iterations        (int, default 4)
    convergence_similarity_threshold  (float, default 0.9)
    convergence_signals               (list or comma string; default all three)
    convergence_rule                  ("any" | "all" | "k_of_n"; default "any")
    convergence_k                     (int; required for k_of_n, default 1)
    convergence_require_no_critical   (bool, default False)
    convergence_required_signals      (list or comma string; default none)

:func:`replay_stopping_rule` applies a rule offline to a recorded trajectory,
optionally with the consensus advancement gate, skipping failed panels and
(by default) partial panels exactly as the engine does. With
``revision_budget=k`` it also replays the revision budget (and with
``convergence=False`` only the budget), so the platform default can be
replayed on the same event sequences as the adaptive rules.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Optional

from backend.models import ReviewResult, SeverityLevel

logger = logging.getLogger(__name__)

# Canonical signal names, in reporting / tie-break order.
SIGNAL_GRADE_STABLE = "grade_stable"
SIGNAL_NO_CRITICAL_HIGH = "no_critical_high"
SIGNAL_OUTPUT_SIMILAR = "output_similar"
ALL_SIGNALS: tuple[str, ...] = (
    SIGNAL_GRADE_STABLE,
    SIGNAL_NO_CRITICAL_HIGH,
    SIGNAL_OUTPUT_SIMILAR,
)
RULES: tuple[str, ...] = ("any", "all", "k_of_n")

# Roles whose output is the artifact under revision (output-similarity signal).
EXECUTOR_ROLES: frozenset[str] = frozenset({"executor"})

# Grade reported by a consensus step whose panel failed; never a real grade.
INCOMPLETE_GRADE = "INCOMPLETE"

# Numeric mapping for letter grades (used by the ``grade_at_least`` stop condition).
_GRADE_SCORES: dict[str, float] = {
    "A+": 4.3, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "D-": 0.7,
    "F": 0.0,
    "PASS": 4.0, "FAIL": 0.0,
}


def grade_score(grade: Optional[str]) -> Optional[float]:
    """Numeric score of a letter grade, or None if it is not a known grade."""
    if not grade:
        return None
    return _GRADE_SCORES.get(str(grade).strip().upper())


def grade_at_least(grade: Optional[str], threshold: str) -> bool:
    """True if ``grade`` is a known grade scoring at least ``threshold``.

    Unknown / empty / ``INCOMPLETE`` grades never satisfy the condition.
    """
    g, t = grade_score(grade), grade_score(threshold)
    if g is None or t is None:
        return False
    return g >= t - 1e-9


def is_executor_role(role: str) -> bool:
    return (role or "").strip().lower() in EXECUTOR_ROLES


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _parse_signals(value: Any) -> tuple[str, ...]:
    if value is None:
        return ALL_SIGNALS
    if isinstance(value, str):
        items = [s.strip() for s in value.replace("+", ",").split(",")]
    else:
        items = [str(s).strip() for s in value]
    items = [s for s in items if s]
    if not items:
        raise ValueError("convergence_signals must name at least one signal")
    unknown = [s for s in items if s not in ALL_SIGNALS]
    if unknown:
        raise ValueError(
            f"Unknown convergence signal(s) {unknown}; valid: {list(ALL_SIGNALS)}"
        )
    # canonical order, no duplicates
    return tuple(s for s in ALL_SIGNALS if s in items)


@dataclass(frozen=True)
class StoppingRule:
    """A complete, validated convergence stopping rule."""

    window: int = 3
    min_iterations: int = 4
    similarity_threshold: float = 0.9
    signals: tuple[str, ...] = ALL_SIGNALS
    rule: str = "any"
    k: int = 1
    require_no_critical: bool = False
    required: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.window) < 1:
            raise ValueError("convergence_window must be >= 1")
        if int(self.min_iterations) < 0:
            raise ValueError("convergence_min_iterations must be >= 0")
        if not 0.0 <= float(self.similarity_threshold) <= 1.0:
            raise ValueError("convergence_similarity_threshold must be in [0, 1]")
        sig = _parse_signals(self.signals)
        object.__setattr__(self, "signals", sig)
        object.__setattr__(self, "window", int(self.window))
        object.__setattr__(self, "min_iterations", int(self.min_iterations))
        object.__setattr__(self, "similarity_threshold", float(self.similarity_threshold))
        object.__setattr__(self, "require_no_critical", _as_bool(self.require_no_critical))
        rule = str(self.rule).strip().lower().replace("-", "_")
        if rule in ("k", "kofn", "k_of"):
            rule = "k_of_n"
        if rule not in RULES:
            raise ValueError(f"convergence_rule must be one of {list(RULES)}, got {self.rule!r}")
        object.__setattr__(self, "rule", rule)
        req: tuple[str, ...] = ()
        if self.required:
            req = _parse_signals(self.required)
            missing = [r for r in req if r not in sig]
            if missing:
                raise ValueError(
                    f"convergence_required_signals {missing} must also be enabled in "
                    f"convergence_signals {list(sig)}")
        object.__setattr__(self, "required", req)
        rest = [x for x in sig if x not in req]
        k = int(self.k)
        if rule == "k_of_n" and not 1 <= k <= max(1, len(rest)):
            raise ValueError(
                f"convergence_k must be between 1 and the number of enabled, non-required "
                f"signals ({len(rest)}), got {k}"
            )
        object.__setattr__(self, "k", k)

    # Config-key mapping (run config / loop stop_conditions -> field).
    CONFIG_KEYS = {
        "convergence_window": "window",
        "convergence_min_iterations": "min_iterations",
        "convergence_similarity_threshold": "similarity_threshold",
        "convergence_signals": "signals",
        "convergence_rule": "rule",
        "convergence_k": "k",
        "convergence_require_no_critical": "require_no_critical",
        "convergence_required_signals": "required",
    }

    @classmethod
    def from_config(
        cls,
        config: Optional[Mapping[str, Any]],
        base: Optional["StoppingRule"] = None,
    ) -> "StoppingRule":
        """Build from run-config keys; absent / None keys keep ``base``
        (default :data:`DEFAULT_STOPPING_RULE`). Raises ``ValueError``."""
        base = base if base is not None else DEFAULT_STOPPING_RULE
        updates: dict[str, Any] = {}
        for key, attr in cls.CONFIG_KEYS.items():
            if config and config.get(key) is not None:
                updates[attr] = config[key]
        return replace(base, **updates) if updates else base

    def describe(self) -> str:
        if self.rule == "any":
            comb = "any of"
        elif self.rule == "all":
            comb = "all of"
        else:
            comb = f"at least {self.k} of"
        rest = [x for x in self.signals if x not in self.required]
        txt = (f"{comb} [{', '.join(rest)}] over window {self.window}, "
               f"min {self.min_iterations} iterations, similarity >= {self.similarity_threshold:g}")
        if self.required:
            txt = f"all of [{', '.join(self.required)}] and " + txt
        if self.require_no_critical:
            txt += ", and no CRITICAL critique in the latest review"
        return txt

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "min_iterations": self.min_iterations,
            "similarity_threshold": self.similarity_threshold,
            "signals": list(self.signals),
            "rule": self.rule,
            "k": self.k,
            "require_no_critical": self.require_no_critical,
            "required": list(self.required),
        }


# THE platform default adaptive stopping rule, used whenever convergence is
# enabled (``convergence_enabled: true``; off by default, see
# DEFAULT_CONVERGENCE_ENABLED): stop when ANY of the three signals holds over a
# window of 3, after at least 4 iterations. Replace this single constant to
# change the adaptive default for every run.
DEFAULT_STOPPING_RULE = StoppingRule(
    window=3,
    min_iterations=4,
    similarity_threshold=0.9,
    signals=ALL_SIGNALS,
    rule="any",
    k=1,
    require_no_critical=False,
)

# Default of run-config ``convergence_enabled``. False since the stopping
# calibration (module docstring): the adaptive rule is evaluated only when a
# run (or its loop's stop_conditions) enables it.
DEFAULT_CONVERGENCE_ENABLED = False

# Default of run-config ``revision_budget``: the maximum number of executor
# revisions after the initial executor submission (a seed_executor_output
# counts as the initial submission). The calibrated platform default (module
# docstring). ``None`` disables the budget.
DEFAULT_REVISION_BUDGET: Optional[int] = 4

# Stop reason of a run ended by the revision budget.
REVISION_BUDGET_STOP_REASON = "revision_budget"

_NO_BUDGET_WORDS = frozenset({"none", "null", "off"})


def parse_revision_budget(value: Any) -> Optional[int]:
    """Validate a ``revision_budget`` value: an integer >= 0, or ``None`` (also
    the strings ``"none"``, ``"null"``, ``"off"``) to disable the budget.

    Integer strings (``"4"``) and integral floats are accepted; booleans,
    negative numbers, fractions and other strings raise ``ValueError``."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"revision_budget must be an integer >= 0 or null, got {value!r}")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _NO_BUDGET_WORDS:
            return None
        try:
            n = int(text)
        except ValueError as exc:
            raise ValueError(
                f"revision_budget must be an integer >= 0 or null, got {value!r}") from exc
    elif isinstance(value, int):
        n = int(value)
    elif isinstance(value, float) and value.is_integer():
        n = int(value)
    else:
        raise ValueError(f"revision_budget must be an integer >= 0 or null, got {value!r}")
    if n < 0:
        raise ValueError(f"revision_budget must be an integer >= 0 or null, got {value!r}")
    return n


@dataclass
class ConvergenceDecision:
    """Outcome of one stopping-rule evaluation."""

    converged: bool
    signals: list[str] = field(default_factory=list)   # enabled signals that hold
    reason: str = ""                                    # human-readable
    blocked_by: str = ""                                # e.g. "min_iterations", "critical"

    @property
    def stop_reason(self) -> str:
        """Machine-readable stop reason, e.g. ``converged:grade_stable``."""
        if not self.converged:
            return ""
        return "converged:" + "+".join(self.signals)


class ConvergenceDetector:
    """Detects when the review loop has converged under a :class:`StoppingRule`."""

    def __init__(
        self,
        window_size: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        min_iterations: Optional[int] = None,
        *,
        rule: Optional[StoppingRule] = None,
    ):
        base = rule if rule is not None else DEFAULT_STOPPING_RULE
        updates: dict[str, Any] = {}
        if window_size is not None:
            updates["window"] = window_size
        if similarity_threshold is not None:
            updates["similarity_threshold"] = similarity_threshold
        if min_iterations is not None:
            updates["min_iterations"] = min_iterations
        self.rule: StoppingRule = replace(base, **updates) if updates else base

        # Tracked state
        self._grade_history: list[str] = []
        self._critique_counts: list[int] = []   # CRITICAL + HIGH per review step
        self._critical_counts: list[int] = []   # CRITICAL only per review step
        self._executor_outputs: list[str] = []
        self._similarity_scores: list[float] = []
        self._iteration_count = 0

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "ConvergenceDetector":
        return cls(rule=StoppingRule.from_config(config))

    # Back-compatible attribute names
    @property
    def _window_size(self) -> int:
        return self.rule.window

    @property
    def _similarity_threshold(self) -> float:
        return self.rule.similarity_threshold

    @property
    def _min_iterations(self) -> int:
        return self.rule.min_iterations

    # ── recording ────────────────────────────────────────────────────

    def note_iteration(self, iteration_num: int) -> None:
        """Advance the iteration counter without recording any signal data."""
        self._iteration_count = max(self._iteration_count, int(iteration_num))

    def record_review(
        self,
        grade: Optional[str],
        critical: int,
        high: int,
        iteration_num: Optional[int] = None,
    ) -> None:
        """Record one review step from its grade and severity counts."""
        if iteration_num is not None:
            self.note_iteration(iteration_num)
        g = (grade or "").strip().upper()
        if g == INCOMPLETE_GRADE:
            return  # a failed panel is not a review observation
        if g:
            self._grade_history.append(g)
        self._critique_counts.append(int(critical) + int(high))
        self._critical_counts.append(int(critical))

    def record_executor_output(self, output: str, iteration_num: Optional[int] = None) -> None:
        """Record one executor output (similarity vs the previous executor output)."""
        if iteration_num is not None:
            self.note_iteration(iteration_num)
        output = output or ""
        if self._executor_outputs:
            self._similarity_scores.append(_jaccard_similarity(self._executor_outputs[-1], output))
        self._executor_outputs.append(output)
        # Only the previous output is ever compared; keep memory bounded.
        if len(self._executor_outputs) > 2:
            del self._executor_outputs[:-2]

    def record_iteration(
        self,
        iteration_num: int,
        role: str,
        output: str,
        review: Optional[ReviewResult] = None,
        *,
        is_executor: Optional[bool] = None,
    ) -> None:
        """Record data from one iteration for convergence tracking.

        ``review`` (if given) is recorded as one review observation. The output
        enters the similarity signal only for executor roles
        (``is_executor`` overrides the role-based decision).
        """
        self.note_iteration(iteration_num)
        if review is not None:
            critical = sum(1 for c in review.critiques if c.severity == SeverityLevel.CRITICAL)
            high = sum(1 for c in review.critiques if c.severity == SeverityLevel.HIGH)
            self.record_review(review.overall_grade, critical, high)
        executor = is_executor_role(role) if is_executor is None else bool(is_executor)
        if executor:
            self.record_executor_output(output)

    # ── evaluation ───────────────────────────────────────────────────

    def _signal_states(self) -> dict[str, tuple[bool, str]]:
        w = self.rule.window
        states: dict[str, tuple[bool, str]] = {}
        ok = False
        msg = ""
        if len(self._grade_history) >= w:
            window = self._grade_history[-w:]
            if len(set(window)) == 1:
                ok = True
                msg = f"Grade stabilized at {window[0]} for {w} consecutive reviews"
        states[SIGNAL_GRADE_STABLE] = (ok, msg)

        ok, msg = False, ""
        if len(self._critique_counts) >= w:
            window = self._critique_counts[-w:]
            if all(c == 0 for c in window):
                ok = True
                msg = f"Zero CRITICAL/HIGH critiques for {w} consecutive reviews"
        states[SIGNAL_NO_CRITICAL_HIGH] = (ok, msg)

        ok, msg = False, ""
        thr = self.rule.similarity_threshold
        if len(self._similarity_scores) >= w:
            window = self._similarity_scores[-w:]
            if all(s >= thr for s in window):
                ok = True
                avg = sum(window) / len(window)
                msg = (f"Executor output similarity >= {thr} for {w} consecutive "
                       f"executor iterations (avg={avg:.3f})")
        states[SIGNAL_OUTPUT_SIMILAR] = (ok, msg)
        return states

    def evaluate(self) -> ConvergenceDecision:
        """Evaluate the stopping rule on the recorded history."""
        if self._iteration_count < self.rule.min_iterations:
            return ConvergenceDecision(False, blocked_by="min_iterations")
        states = self._signal_states()
        holding = [s for s in self.rule.signals if states[s][0]]
        required = self.rule.required
        rest_enabled = [s for s in self.rule.signals if s not in required]
        rest_holding = [s for s in holding if s not in required]
        if not all(states[r][0] for r in required):
            fired = False
        elif not rest_enabled:
            fired = True  # only required signals: they all hold
        elif self.rule.rule == "any":
            fired = len(rest_holding) >= 1
        elif self.rule.rule == "all":
            fired = len(rest_holding) == len(rest_enabled)
        else:
            fired = len(rest_holding) >= self.rule.k
        if not fired:
            return ConvergenceDecision(False, signals=holding)
        if self.rule.require_no_critical:
            if not self._critical_counts or self._critical_counts[-1] > 0:
                return ConvergenceDecision(False, signals=holding, blocked_by="critical")
        reason = "; ".join(states[s][1] for s in holding)
        return ConvergenceDecision(True, signals=holding, reason=reason)

    def should_stop(self) -> tuple[bool, str]:
        """Back-compatible wrapper: (converged, human-readable reason)."""
        d = self.evaluate()
        return d.converged, d.reason

    def get_metrics(self) -> dict:
        """Return current convergence metrics for API/WebSocket reporting."""
        return {
            "iteration_count": self._iteration_count,
            "grade_history": list(self._grade_history),
            "critique_counts": list(self._critique_counts),
            "critical_counts": list(self._critical_counts),
            "similarity_scores": list(self._similarity_scores),
            "window_size": self.rule.window,
            "similarity_threshold": self.rule.similarity_threshold,
            "min_iterations": self.rule.min_iterations,
            "rule": self.rule.to_dict(),
        }


def replay_stopping_rule(
    events: Iterable[Mapping[str, Any]],
    rule: Optional[StoppingRule] = None,
    *,
    gate: bool = False,
    include_partial: bool = False,
    revision_budget: Optional[int] = None,
    convergence: bool = True,
) -> dict[str, Any]:
    """Apply ``rule`` offline to a recorded trajectory, as the engine would.

    ``events`` are in iteration order; each has ``iteration`` (int) and
    ``role`` (str) and either ``output`` (executor steps) or review fields
    ``grade``, ``critical`` and ``high`` (review / consensus steps). A seeded
    run starts with the seed as an executor event at iteration 0.

    * Events with ``panel_failed=True`` advance the iteration counter only.
    * Events with ``panel_partial=True`` (or ``grade_valid=False``) do the
      same unless ``include_partial`` (the engine's default
      ``consensus_partial_policy="no_stop"``); no stop is evaluated at them.
    * ``gate=True`` mirrors the engine's ``consensus_gate``: after a review
      event with ``unresolved_critical > 0`` (default: its ``critical``
      count), no stop is evaluated until a later review event has none.
      The gate never blocks the revision budget.
    * ``revision_budget=k`` mirrors the engine's run-config key of the same
      name: once ``1 + k`` executor events (the initial submission, or the
      seed, plus ``k`` revisions) have been seen, the run stops before the
      next executor event, i.e. at the iteration of the event preceding it,
      with stop reason ``revision_budget``. If the events end after the
      budget was exhausted, the stop is reported at the last event (the
      trajectory should then end with the review of the final revision, as
      the engine's does).
    * ``convergence=False`` mirrors ``convergence_enabled: false``: ``rule`` is
      never evaluated (only the revision budget can stop the replay).

    Returns ``{"stopped": bool, "iteration": int | None, "stop_reason": str,
    "reason": str}`` for the FIRST iteration at which the run stops.
    """
    budget = parse_revision_budget(revision_budget)
    det = ConvergenceDetector(rule=rule)
    last_iter = 0
    blocked = False
    submissions = 0

    def budget_stop(at: int) -> dict[str, Any]:
        return {"stopped": True, "iteration": at, "stop_reason": REVISION_BUDGET_STOP_REASON,
                "reason": (f"Revision budget reached: initial executor submission plus "
                           f"{budget} revision(s)")}

    for ev in events:
        it = int(ev.get("iteration", last_iter + 1))
        executor_event = (not ev.get("panel_failed")
                          and not ("grade" in ev or "critical" in ev or "high" in ev)
                          and (is_executor_role(str(ev.get("role", ""))) or bool(ev.get("is_executor"))))
        if executor_event and budget is not None and submissions >= 1 + budget:
            return budget_stop(last_iter)
        if executor_event:
            submissions += 1
        last_iter = it
        det.note_iteration(it)
        if ev.get("panel_failed"):
            continue
        is_review = "grade" in ev or "critical" in ev or "high" in ev
        partial = bool(ev.get("panel_partial")) or ev.get("grade_valid") is False
        if is_review and partial and not include_partial:
            if gate:
                unresolved = ev.get("unresolved_critical", ev.get("critical", 0))
                blocked = int(unresolved or 0) > 0
            continue
        if is_review:
            det.record_review(ev.get("grade"), int(ev.get("critical", 0) or 0),
                              int(ev.get("high", 0) or 0))
            if gate:
                unresolved = ev.get("unresolved_critical", ev.get("critical", 0))
                blocked = int(unresolved or 0) > 0
        elif is_executor_role(str(ev.get("role", ""))) or ev.get("is_executor"):
            det.record_executor_output(str(ev.get("output", "")))
        if (gate and blocked) or not convergence:
            continue
        d = det.evaluate()
        if d.converged:
            return {"stopped": True, "iteration": it, "stop_reason": d.stop_reason,
                    "reason": d.reason}
    if budget is not None and submissions >= 1 + budget:
        return budget_stop(last_iter)
    return {"stopped": False, "iteration": None, "stop_reason": "", "reason": ""}


__all__ = [
    "ALL_SIGNALS",
    "ConvergenceDecision",
    "ConvergenceDetector",
    "DEFAULT_CONVERGENCE_ENABLED",
    "DEFAULT_REVISION_BUDGET",
    "DEFAULT_STOPPING_RULE",
    "EXECUTOR_ROLES",
    "REVISION_BUDGET_STOP_REASON",
    "RULES",
    "SIGNAL_GRADE_STABLE",
    "SIGNAL_NO_CRITICAL_HIGH",
    "SIGNAL_OUTPUT_SIMILAR",
    "StoppingRule",
    "grade_at_least",
    "grade_score",
    "is_executor_role",
    "parse_revision_budget",
    "replay_stopping_rule",
]
