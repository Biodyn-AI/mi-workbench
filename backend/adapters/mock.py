"""Mock adapter for testing. Returns synthetic but realistic MI responses.

Output varies by detected role and improves across iterations to exercise
convergence detection in the pipeline.

Consensus adjudication requests (role ``consensus_adjudicator``, sent by the
default ``llm`` similarity method) get a valid, deterministic partition: the
numbered critiques are grouped by exact text (case and whitespace ignored).
These calls do not advance the mock's iteration counter (so the grade
progression of the other roles is the same with or without adjudication),
draw no random numbers, never simulate failures, and are counted separately
(``adjudication_count`` / ``adjudication_history``).
"""
import asyncio
import json
import random
import re
import time
from typing import Optional

from backend.adapters.base import BaseAdapter
from backend.models import AdapterRunRequest, AdapterRunResult


# ── Grade / severity progression tables ──────────────────────────────

_GRADE_SEQUENCE = ["C", "B", "B+", "A-"]
MOCK_CLI_VERSION = "mock"  # reported as AdapterRunResult.cli_version
_TOKEN_TARGETS = {
    "executor": 2000,
    "reviewer": 1500,
    "adversarial": 1000,
    "idea_generator": 1800,
}


def _grade_for_iteration(n: int) -> str:
    """Return a grade that improves with iteration count (1-indexed)."""
    idx = min(n - 1, len(_GRADE_SEQUENCE) - 1)
    return _GRADE_SEQUENCE[max(idx, 0)]


# ── Executor output (with artifact markers) ─────────────────────────

def _executor_output(iteration: int) -> str:
    # Small variations that diminish over iterations
    auroc_base = 0.743
    corr_base = 0.703
    split_base = 0.750
    residual_pct = 76

    # Variation magnitude shrinks: +/-0.01 at iter 1, converging to 0
    mag = max(0.01 - 0.003 * (iteration - 1), 0.0)
    seed = iteration * 137  # deterministic per-iteration
    rng = random.Random(seed)
    auroc = round(auroc_base + rng.uniform(-mag, mag), 3)
    corr = round(corr_base + rng.uniform(-mag, mag), 3)
    split = round(split_base + rng.uniform(-mag, mag), 3)
    residual = residual_pct + rng.randint(-int(mag * 100), int(mag * 100))

    return f"""\
# MECH.md

## Attention-Based Gene Regulatory Network Inference

The analysis of Geneformer's attention matrices (Layer 15, V2-316M) reveals that \
attention weights correlate with gene co-expression (Pearson r=0.72, p<0.001) \
rather than causal regulatory relationships from TRRUST.

### Key Findings
- Attention AUROC for GRN recovery: {auroc} (Layer 15)
- Correlation baseline AUROC: {corr}
- Split-sample validation: AUROC={split}, p=0.017
- Confound analysis: attention loses {residual}% signal under co-expression residualization

### Statistical Tests
1. Paired bootstrap (n=1000): attention vs correlation, p=0.017
2. Permutation test: degree-preserving null, Z=2.0
3. BH-FDR correction applied across all 29 analyses

# EVAL.md

## Evaluation Metrics

| Metric | Value | CI (95%) |
|--------|-------|----------|
| AUROC (attention) | {auroc} | [0.707, 0.770] |
| AUROC (correlation) | {corr} | [0.680, 0.725] |
| Incremental value | -0.0004 | [-0.001, 0.000] |

# XP.md

## Proposed Follow-up Experiments

1. Cross-tissue validation using GTEx data
2. Temporal attention dynamics during fine-tuning
3. Comparison with scBERT attention patterns

## Analysis Code

```python
# Recompute the incremental value of attention over the correlation baseline.
auroc_attention = {auroc}
auroc_correlation = {corr}
incremental = round(auroc_attention - auroc_correlation, 4)
print("incremental_auroc", incremental)
assert 0.0 <= auroc_attention <= 1.0
```"""


# ── Reviewer lenses ─────────────────────────────────────────────────
#
# The three reviewer roles (rigour, adversarial, biological plausibility)
# emit deliberately distinct critique sets with a small, controlled overlap.
# One issue — missing effect sizes — is raised verbatim by both the rigour and
# adversarial lenses so that consensus deduplication and agreement-based
# severity escalation are exercised; every other issue is unique to one lens so
# that panel-size ablations show larger panels surfacing strictly more distinct
# issues. Severity de-escalates as the iteration index rises so the loop
# converges. The shared critique text is byte-identical across lenses on
# purpose (Jaccard-based dedup depends on it).

_SHARED_EFFECT_SIZE = (
    "Effect sizes are not reported alongside p-values for the main comparison")


def _tier(iteration: int) -> int:
    """Coarse quality tier: 1 (early, severe), 2 (mid), 3 (converged)."""
    if iteration <= 1:
        return 1
    if iteration <= 3:
        return 2
    return 3


def _reviewer_output(iteration: int) -> str:
    """Rigour/reproducibility lens."""
    grade = _grade_for_iteration(iteration)
    tier = _tier(iteration)
    if tier == 1:
        critiques = [
            "[CRITICAL] Statistical power analysis is missing for the bootstrap test",
            "[HIGH] Confidence intervals are not reported for the AUROC difference",
            f"[HIGH] {_SHARED_EFFECT_SIZE}",
        ]
    elif tier == 2:
        critiques = [
            "[HIGH] Confidence intervals are not reported for the AUROC difference",
            f"[MEDIUM] {_SHARED_EFFECT_SIZE}",
        ]
    else:
        critiques = ["[LOW] Standardize decimal places across tables"]

    return f"""\
## Review -- Iteration {iteration}

Overall Grade: {grade}

### Critiques
{chr(10).join(critiques)}

### Reproducibility Gaps
- Random seed not specified for bootstrap sampling
- GPU/CPU reproducibility not addressed"""


def _adversarial_output(iteration: int) -> str:
    """Adversarial red-team lens."""
    grade = _grade_for_iteration(iteration)
    tier = _tier(iteration)
    if tier == 1:
        critiques = [
            "[CRITICAL] The central claim is unfalsifiable as stated",
            f"[HIGH] {_SHARED_EFFECT_SIZE}",
            "[MEDIUM] The residualization method needs validation on synthetic data",
        ]
    elif tier == 2:
        critiques = [
            "[MEDIUM] The residualization method needs validation on synthetic data",
        ]
    else:
        critiques = ["[LOW] Figure resolution should be at least 300 DPI"]

    return f"""\
## Adversarial Review

Grade: {grade}

{chr(10).join(critiques)}"""


def _bio_plausibility_output(iteration: int) -> str:
    """Biological plausibility lens (issues the other two lenses do not raise)."""
    grade = _grade_for_iteration(iteration)
    tier = _tier(iteration)
    if tier == 1:
        critiques = [
            "[HIGH] Cell-type composition confounds co-expression and the tissue is unspecified",
            "[MEDIUM] Regulatory direction from transcription factor to target is not assessed",
        ]
    elif tier == 2:
        critiques = [
            "[MEDIUM] Regulatory direction from transcription factor to target is not assessed",
        ]
    else:
        critiques = ["[INFO] Pathway consistency checks pass"]

    return f"""\
## Biological Plausibility Review

Grade: {grade}

{chr(10).join(critiques)}"""


# ── Idea generator output ───────────────────────────────────────────

def _idea_generator_output(iteration: int) -> str:
    return """\
## Proposed Follow-up Studies

1. **Cross-tissue Attention Validation** -- Test whether attention-GRN patterns \
generalize across GTEx tissues. Effort: medium
2. **Temporal Fine-tuning Dynamics** -- Track how attention patterns shift during \
domain-specific fine-tuning. Effort: high
3. **Multi-model Ensemble** -- Combine attention signals from Geneformer + scGPT + \
scBERT for improved GRN inference. Effort: high"""


# ── Consensus adjudication (similarity method "llm") ─────────────────

ADJUDICATOR_ROLE = "consensus_adjudicator"
_NUMBERED_LINE = re.compile(r"^\[(\d+)\]\s?(.*)$")


def mock_adjudication_groups(user_prompt: str) -> list[list[int]]:
    """Group the ``[i] text`` lines of an adjudication prompt by exact text
    (lower-cased, whitespace collapsed); groups ordered by first index."""
    by_text: dict[str, list[int]] = {}
    for line in (user_prompt or "").splitlines():
        m = _NUMBERED_LINE.match(line.strip())
        if not m:
            continue
        key = " ".join(m.group(2).split()).lower()
        by_text.setdefault(key, []).append(int(m.group(1)))
    groups = [sorted(set(g)) for g in by_text.values()]
    groups.sort(key=lambda g: g[0])
    return groups


# ── Dispatch table ──────────────────────────────────────────────────

_OUTPUT_BUILDERS = {
    "executor": _executor_output,
    "reviewer": _reviewer_output,
    "adversarial": _adversarial_output,
    "bio_plausibility": _bio_plausibility_output,
    "idea_generator": _idea_generator_output,
}


# ── MockAdapter ─────────────────────────────────────────────────────

class MockAdapter(BaseAdapter):
    name: str = "mock"

    _iteration_count: int = 0  # class-level counter shared across instances
    # When set, every call uses this logical iteration instead of the global
    # counter. Lets experiments pin a fixed quality tier so a reviewer panel is
    # evaluated reproducibly and independently of invocation order.
    fixed_iteration: Optional[int] = None

    def __init__(self, failure_rate: float = 0.0, min_delay: float = 0.1,
                 max_delay: float = 0.5):
        self.failure_rate = failure_rate
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.invocation_count = 0
        self.history: list[dict] = []
        self.adjudication_count = 0
        self.adjudication_history: list[dict] = []

    def is_available(self) -> bool:
        return True

    async def cli_version(self) -> str:
        return MOCK_CLI_VERSION

    async def smoke_test(self) -> dict:
        return {
            "status": "ok",
            "adapter": "mock",
            "invocation_count": self.invocation_count,
            "failure_rate": self.failure_rate,
        }

    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        if self._is_adjudication(request):
            return await self._adjudicate(request)
        self.invocation_count += 1
        MockAdapter._iteration_count += 1
        iteration = (
            MockAdapter.fixed_iteration
            if MockAdapter.fixed_iteration is not None
            else MockAdapter._iteration_count
        )
        start = time.monotonic()

        delay = random.uniform(self.min_delay, self.max_delay)
        await asyncio.sleep(delay)

        # Simulate failure
        if random.random() < self.failure_rate:
            result = AdapterRunResult(
                success=False,
                error="Mock simulated failure",
                exit_code=1,
                duration_seconds=time.monotonic() - start,
                provider=self.name,
                model="mock",
                reasoning_effort=getattr(request, "reasoning_effort", ""),
                cli_version=MOCK_CLI_VERSION,
            )
            self.history.append({"request": request, "result": result})
            return result

        # Determine role from prompt content to pick appropriate mock output
        role = self._detect_role(request)
        builder = _OUTPUT_BUILDERS.get(role, _executor_output)
        output_text = builder(iteration)

        # Role-specific token usage with small random jitter
        base_tokens = _TOKEN_TARGETS.get(role, 2000)
        token_usage = base_tokens + random.randint(-100, 100)
        cost = token_usage * 0.000003  # ~$3/1M tokens
        # Deterministic 70/30 input/output split of the same total (no extra RNG
        # draws, so delays/token totals are unchanged); input + output == total.
        input_tokens = int(token_usage * 0.7)
        output_tokens = token_usage - input_tokens

        artifacts = []
        if role == "executor":
            artifacts = ["MECH.md", "EVAL.md", "XP.md"]
        elif role in ("reviewer", "adversarial", "bio_plausibility"):
            artifacts = ["EVAL.md"]
        elif role == "idea_generator":
            artifacts = ["XP.md"]

        result = AdapterRunResult(
            success=True,
            output=output_text,
            artifacts_written=artifacts,
            structured_output={
                "role": role,
                "mock": True,
                "iteration": iteration,
                "grade": _grade_for_iteration(iteration) if role != "executor" else None,
            },
            token_usage=token_usage,
            cost_estimate=cost,
            duration_seconds=time.monotonic() - start,
            exit_code=0,
            raw_log=f"[mock] role={role} iter={iteration} tokens={token_usage}",
            provider=self.name,
            model="mock",
            reasoning_effort=getattr(request, "reasoning_effort", ""),
            cli_version=MOCK_CLI_VERSION,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=0,
            raw_usage={"input_tokens": input_tokens, "output_tokens": output_tokens,
                       "synthetic": True},
        )
        self.history.append({"request": request, "result": result})
        return result

    @staticmethod
    def _is_adjudication(request: AdapterRunRequest) -> bool:
        role = str(request.prompt_bundle.variables.get("role", "") or "").strip().lower()
        return role == ADJUDICATOR_ROLE

    async def _adjudicate(self, request: AdapterRunRequest) -> AdapterRunResult:
        """Deterministic adjudicator answer (see the module docstring)."""
        self.adjudication_count += 1
        start = time.monotonic()
        # Fixed delay (no RNG draw, so the other calls' jitter is unchanged).
        await asyncio.sleep(self.min_delay)
        groups = mock_adjudication_groups(request.prompt_bundle.user_prompt)
        n = sum(len(g) for g in groups)
        output = "```json\n" + json.dumps({"groups": groups}) + "\n```"
        token_usage = 200 + 20 * n
        input_tokens = int(token_usage * 0.7)
        output_tokens = token_usage - input_tokens
        result = AdapterRunResult(
            success=True,
            output=output,
            structured_output={"role": ADJUDICATOR_ROLE, "mock": True, "groups": groups},
            token_usage=token_usage,
            cost_estimate=token_usage * 0.000003,
            duration_seconds=time.monotonic() - start,
            exit_code=0,
            raw_log=f"[mock] role={ADJUDICATOR_ROLE} n={n} groups={len(groups)}",
            provider=self.name,
            model="mock",
            reasoning_effort=getattr(request, "reasoning_effort", ""),
            cli_version=MOCK_CLI_VERSION,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=0,
            raw_usage={"input_tokens": input_tokens, "output_tokens": output_tokens,
                       "synthetic": True},
        )
        self.adjudication_history.append({"request": request, "result": result})
        return result

    # Map explicit role strings (set by the engine on the prompt bundle) to the
    # mock's output-builder keys. This is authoritative when present.
    _ROLE_ALIASES = {
        "executor": "executor",
        "reviewer": "reviewer",
        "adversarial": "adversarial",
        "adversarial_reviewer": "adversarial",
        "bio_plausibility_checker": "bio_plausibility",
        "bio_plausibility": "bio_plausibility",
        "idea_generator": "idea_generator",
    }

    def _detect_role(self, request: AdapterRunRequest) -> str:
        """Determine the role for output selection.

        Prefer the explicit ``role`` variable set by the engine; fall back to
        keyword heuristics on the prompt text (e.g. for ad-hoc callers and tests
        that do not populate variables).
        """
        role_var = request.prompt_bundle.variables.get("role", "").strip().lower()
        if role_var in self._ROLE_ALIASES:
            return self._ROLE_ALIASES[role_var]

        text = (request.prompt_bundle.system_prompt +
                request.prompt_bundle.user_prompt).lower()
        if "adversar" in text or "red.team" in text or "red team" in text:
            return "adversarial"
        if "plausib" in text or "bio_plausibility" in text or "biological plausibility" in text:
            return "bio_plausibility"
        if "idea" in text or "propose" in text:
            return "idea_generator"
        if "review" in text or "eval" in text:
            return "reviewer"
        return "executor"

    @classmethod
    def reset_iteration_count(cls) -> None:
        """Reset the class-level iteration counter (useful in tests)."""
        cls._iteration_count = 0
