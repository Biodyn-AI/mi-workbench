"""Mock adapter for testing. Returns synthetic but realistic MI responses.

Output varies by detected role and improves across iterations to exercise
convergence detection in the pipeline.
"""
import asyncio
import random
import time
from typing import Optional

from backend.adapters.base import BaseAdapter
from backend.models import AdapterRunRequest, AdapterRunResult


# ── Grade / severity progression tables ──────────────────────────────

_GRADE_SEQUENCE = ["C", "B", "B+", "A-"]
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
3. Comparison with scBERT attention patterns"""


# ── Reviewer output (severity decreases with iteration) ─────────────

def _reviewer_output(iteration: int) -> str:
    grade = _grade_for_iteration(iteration)

    # Build critique blocks based on iteration
    required_fixes = []
    suggested = []

    if iteration <= 1:
        required_fixes.append(
            "[CRITICAL] Statistical power analysis missing for the bootstrap test")
        required_fixes.append(
            "[CRITICAL] Negative control needed: attention for random gene pairs")
        required_fixes.append(
            "[HIGH] Need to report effect sizes alongside p-values")
        required_fixes.append(
            "[HIGH] Confidence intervals needed for AUROC difference")
    elif iteration == 2:
        required_fixes.append(
            "[CRITICAL] Effect size confidence interval not reported for main comparison")
        required_fixes.append(
            "[HIGH] Consider Bonferroni correction alongside BH-FDR")
    elif iteration == 3:
        required_fixes.append(
            "[HIGH] Minor gap: report Cohen's d for attention vs correlation comparison")
    # iteration >= 4: no CRITICAL or HIGH

    suggested.append(
        "[MEDIUM] Consider adding a figure showing attention vs correlation scatter")
    suggested.append(
        "[LOW] Minor: standardize decimal places in Table 1")

    required_section = "\n".join(required_fixes) if required_fixes else "(none)"
    suggested_section = "\n".join(suggested)

    return f"""\
## Review -- Iteration {iteration}

Overall Grade: {grade}

### Required Fixes
{required_section}

### Suggested Improvements
{suggested_section}

### Reproducibility Gaps
- Random seed not specified for bootstrap sampling
- GPU/CPU reproducibility not addressed

### Suspected Confounders
- Cell type composition may drive co-expression patterns
- Batch effects from multi-donor Tabula Sapiens data"""


# ── Adversarial reviewer output ─────────────────────────────────────

def _adversarial_output(iteration: int) -> str:
    grade = _grade_for_iteration(iteration)

    critiques = []
    if iteration <= 1:
        critiques.append(
            '[CRITICAL] The claim that "attention captures co-expression" '
            "is unfalsifiable as stated")
        critiques.append(
            "[CRITICAL] No negative control: what does attention look like "
            "for random gene pairs?")
        critiques.append(
            "[HIGH] The AUROC improvement over correlation (0.743 vs 0.703) "
            "may not be practically significant")
        critiques.append(
            "[MEDIUM] Confound decomposition methodology needs validation "
            "on synthetic data")
    elif iteration == 2:
        critiques.append(
            "[CRITICAL] Synthetic-data validation of residualization still missing")
        critiques.append(
            "[HIGH] Practical significance threshold should be pre-registered")
        critiques.append(
            "[MEDIUM] Consider reporting Bayes factors alongside frequentist tests")
    elif iteration == 3:
        critiques.append(
            "[HIGH] Pre-registration of significance thresholds recommended")
        critiques.append(
            "[MEDIUM] Bayes factor analysis would strengthen the null-result claim")
    else:
        critiques.append(
            "[MEDIUM] Consider expanding cross-tissue validation to more than 3 tissues")
        critiques.append(
            "[LOW] Figure resolution should be >= 300 DPI for publication")

    critique_text = "\n".join(critiques)

    return f"""\
## Adversarial Review

Grade: {grade}

{critique_text}"""


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


# ── Dispatch table ──────────────────────────────────────────────────

_OUTPUT_BUILDERS = {
    "executor": _executor_output,
    "reviewer": _reviewer_output,
    "adversarial": _adversarial_output,
    "idea_generator": _idea_generator_output,
}


# ── MockAdapter ─────────────────────────────────────────────────────

class MockAdapter(BaseAdapter):
    name: str = "mock"

    _iteration_count: int = 0  # class-level counter shared across instances

    def __init__(self, failure_rate: float = 0.0, min_delay: float = 0.1,
                 max_delay: float = 0.5):
        self.failure_rate = failure_rate
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.invocation_count = 0
        self.history: list[dict] = []

    def is_available(self) -> bool:
        return True

    async def smoke_test(self) -> dict:
        return {
            "status": "ok",
            "adapter": "mock",
            "invocation_count": self.invocation_count,
            "failure_rate": self.failure_rate,
        }

    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        self.invocation_count += 1
        MockAdapter._iteration_count += 1
        iteration = MockAdapter._iteration_count
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

        artifacts = []
        if role == "executor":
            artifacts = ["MECH.md", "EVAL.md", "XP.md"]
        elif role == "reviewer":
            artifacts = ["EVAL.md"]
        elif role == "adversarial":
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
        )
        self.history.append({"request": request, "result": result})
        return result

    def _detect_role(self, request: AdapterRunRequest) -> str:
        """Guess the role from prompt content."""
        text = (request.prompt_bundle.system_prompt +
                request.prompt_bundle.user_prompt).lower()
        if "adversar" in text or "red.team" in text:
            return "adversarial"
        if "review" in text or "eval" in text:
            return "reviewer"
        if "follow" in text or "idea" in text:
            return "idea_generator"
        return "executor"

    @classmethod
    def reset_iteration_count(cls) -> None:
        """Reset the class-level iteration counter (useful in tests)."""
        cls._iteration_count = 0
