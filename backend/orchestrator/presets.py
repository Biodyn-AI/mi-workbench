"""Loop presets for common MI research workflows.

Resolution order of :func:`get_preset` (used by the runner):

1. ``custom:<path>``: a loop YAML file (``miw run loop`` sends this). The
   path may be absolute, relative to the working directory, or relative to
   a loops directory; it must be an existing ``.yaml``/``.yml`` file.
2. A built-in Python preset (``executor_reviewer``, ``reviewer_consensus``,
   ``research_followups``).
3. ``<name>.yaml`` (or ``.yml``) in the configured loops directory
   (``MIW_LOOPS_DIR``), then in the repository's bundled ``loops/``
   directory (e.g. ``example_custom``).

YAML loops are parsed with :func:`backend.orchestrator.dsl.parse_loop_yaml`
and validated before use.

Stopping: the presets declare no stopping policy of their own; the engine's
calibrated default applies to every loop with an executor step (run config
``revision_budget``, default 4 reviewed revisions; the adaptive convergence
rule is opt-in via ``convergence_enabled``). A loop may override it with a
``revision_budget:N`` / ``revision_budget:none`` stop condition and a run
config overrides the loop.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

from backend.models import LoopDefinition, LoopNode, LoopEdge

#: The repository's bundled loop YAML directory.
BUNDLED_LOOPS_DIR = Path(__file__).resolve().parent.parent.parent / "loops"

CUSTOM_PREFIX = "custom:"
_MAX_LOOP_FILE_BYTES = 1_000_000
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def executor_reviewer_preset(adversarial_every_k: int = 3,
                              max_iterations: int = 50) -> LoopDefinition:
    """Executor -> Reviewer -> (Adversarial every K) -> repeat cycle."""
    nodes = [
        LoopNode(
            id="executor",
            role="executor",
            prompt_ref="executor/mi_executor",
            config={"description": "Run mechanistic analysis experiment"},
        ),
        LoopNode(
            id="reviewer",
            role="reviewer",
            prompt_ref="reviewer/mi_reviewer",
            config={"description": "Review executor output for rigor"},
        ),
        LoopNode(
            id="adversarial",
            role="adversarial",
            prompt_ref="adversarial_reviewer/adversarial_reviewer",
            config={"description": "Red-team the analysis for confounders"},
        ),
    ]
    edges = [
        LoopEdge(source="executor", target="reviewer", condition="always"),
        LoopEdge(source="reviewer", target="adversarial",
                 condition=f"every_k:{adversarial_every_k}"),
        LoopEdge(source="reviewer", target="executor", condition="always"),
        LoopEdge(source="adversarial", target="executor", condition="always"),
    ]
    return LoopDefinition(
        name="executor_reviewer",
        description="Executor runs experiments, reviewer checks rigor, "
                    f"adversarial red-teams in every {adversarial_every_k}th "
                    "executor-reviewer cycle.",
        nodes=nodes,
        edges=edges,
        max_iterations=max_iterations,
        config={"adversarial_every_k": adversarial_every_k},
    )


def research_followups_preset(max_iterations: int = 30) -> LoopDefinition:
    """Milestone -> Idea Generator -> auto-queue follow-ups cycle."""
    nodes = [
        LoopNode(
            id="milestone",
            role="executor",
            prompt_ref="executor/mi_executor",
            config={"description": "Execute a research milestone"},
        ),
        LoopNode(
            id="idea_generator",
            role="idea_generator",
            prompt_ref="idea_generator/follow_up_proposer",
            config={"description": "Generate follow-up experiment proposals"},
        ),
    ]
    edges = [
        LoopEdge(source="milestone", target="idea_generator", condition="always"),
        LoopEdge(source="idea_generator", target="milestone", condition="always"),
    ]
    return LoopDefinition(
        name="research_followups",
        description="Execute milestones and generate follow-up proposals in a cycle.",
        nodes=nodes,
        edges=edges,
        max_iterations=max_iterations,
    )


def reviewer_consensus_preset(max_iterations: int = 50) -> LoopDefinition:
    """Executor -> 3 parallel reviewers -> consensus merger -> repeat cycle."""
    nodes = [
        LoopNode(
            id="executor",
            role="executor",
            prompt_ref="executor/mi_executor",
            config={"description": "Run mechanistic analysis experiment"},
        ),
        LoopNode(
            id="reviewer",
            role="reviewer",
            prompt_ref="reviewer/mi_reviewer",
            config={"description": "Review executor output for rigor"},
        ),
        LoopNode(
            id="adversarial_reviewer",
            role="adversarial_reviewer",
            prompt_ref="adversarial_reviewer/adversarial_reviewer",
            config={"description": "Red-team the analysis for confounders"},
        ),
        LoopNode(
            id="bio_plausibility_checker",
            role="bio_plausibility_checker",
            prompt_ref="biological_plausibility/bio_plausibility_checker",
            config={"description": "Check biological plausibility of claims"},
        ),
        LoopNode(
            id="consensus_merger",
            role="consensus_merger",
            prompt_ref="",
            config={
                "description": "Merge critiques from all reviewers into ranked consensus",
                "merge_strategy": "deduplicate_escalate",
            },
        ),
    ]
    edges = [
        LoopEdge(source="executor", target="reviewer", condition="always"),
        LoopEdge(source="executor", target="adversarial_reviewer", condition="always"),
        LoopEdge(source="executor", target="bio_plausibility_checker", condition="always"),
        LoopEdge(source="reviewer", target="consensus_merger", condition="always"),
        LoopEdge(source="adversarial_reviewer", target="consensus_merger", condition="always"),
        LoopEdge(source="bio_plausibility_checker", target="consensus_merger", condition="always"),
        LoopEdge(source="consensus_merger", target="executor", condition="always"),
    ]
    return LoopDefinition(
        name="reviewer_consensus",
        description="Executor runs experiments, three reviewers (standard, adversarial, "
                    "bio plausibility) critique in parallel, consensus merger deduplicates "
                    "and ranks critiques before feeding back to executor.",
        nodes=nodes,
        edges=edges,
        max_iterations=max_iterations,
        config={"parallel_reviewers": True},
    )


PRESETS: dict[str, Callable[..., LoopDefinition]] = {
    "executor_reviewer": executor_reviewer_preset,
    "research_followups": research_followups_preset,
    "reviewer_consensus": reviewer_consensus_preset,
}


def loop_search_dirs() -> list[Path]:
    """Directories searched for ``<name>.yaml`` presets, in order."""
    dirs: list[Path] = []
    try:
        from backend.config import config

        if config.loops_dir:
            dirs.append(Path(config.loops_dir))
    except Exception:  # pragma: no cover - config import never fails in practice
        pass
    if BUNDLED_LOOPS_DIR not in dirs:
        dirs.append(BUNDLED_LOOPS_DIR)
    return dirs


def find_loop_yaml(name: str) -> Optional[Path]:
    """Return the YAML file for preset ``name`` (no path separators), if any."""
    if not _NAME_RE.match(name or "") or name in (".", ".."):
        return None
    for d in loop_search_dirs():
        for suffix in (".yaml", ".yml"):
            p = d / f"{name}{suffix}"
            if p.is_file():
                return p
    return None


def _resolve_custom_path(raw: str) -> Path:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("custom loop path is empty (expected 'custom:<path to .yaml>')")
    p = Path(raw).expanduser()
    candidates = [p] if p.is_absolute() else [Path.cwd() / p] + [d / p for d in loop_search_dirs()]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise ValueError(f"Loop file not found: {raw}")


def load_loop_file(path: str | Path) -> LoopDefinition:
    """Load, parse and validate a loop YAML file."""
    from backend.orchestrator.dsl import parse_loop_yaml, parse_stop_conditions, validate_loop

    p = Path(path)
    if p.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"Loop file must be a .yaml/.yml file: {p}")
    if not p.is_file():
        raise ValueError(f"Loop file not found: {p}")
    if p.stat().st_size > _MAX_LOOP_FILE_BYTES:
        raise ValueError(f"Loop file too large (> {_MAX_LOOP_FILE_BYTES} bytes): {p}")
    try:
        loop = parse_loop_yaml(p.read_text())
    except ValueError:
        raise
    except Exception as exc:  # yaml / pydantic errors
        raise ValueError(f"Invalid loop file {p}: {exc}") from exc
    errors = validate_loop(loop)
    if errors:
        raise ValueError(f"Invalid loop file {p}: {'; '.join(errors)}")
    parse_stop_conditions(loop.stop_conditions)  # raises ValueError on bad values
    loop.source = str(p)
    return loop


def list_presets() -> list[str]:
    """Names accepted by :func:`get_preset` (``custom:<path>`` aside)."""
    names = set(PRESETS)
    for d in loop_search_dirs():
        if d.is_dir():
            for f in d.iterdir():
                if (f.suffix in (".yaml", ".yml") and not f.name.startswith(".")
                        and not f.name.startswith("._") and _NAME_RE.match(f.stem)):
                    names.add(f.stem)
    return sorted(names)


def get_preset(name: str, **kwargs) -> LoopDefinition:
    """Resolve a preset name (or ``custom:<path>``) to a LoopDefinition.

    ``kwargs`` are passed to Python preset factories; for YAML loops only
    ``max_iterations`` is accepted. Raises ``ValueError`` for unknown names,
    missing files and invalid loop definitions.
    """
    name = (name or "").strip()
    if name.startswith(CUSTOM_PREFIX):
        loop = load_loop_file(_resolve_custom_path(name[len(CUSTOM_PREFIX):]))
        return _apply_yaml_kwargs(loop, kwargs)
    factory = PRESETS.get(name)
    if factory is not None:
        loop = factory(**kwargs)
        loop.source = loop.source or "python"
        return loop
    path = find_loop_yaml(name)
    if path is not None:
        return _apply_yaml_kwargs(load_loop_file(path), kwargs)
    raise ValueError(f"Unknown preset: {name}. Available: {list_presets()} or 'custom:<path>'")


def _apply_yaml_kwargs(loop: LoopDefinition, kwargs: dict) -> LoopDefinition:
    extra = set(kwargs) - {"max_iterations"}
    if extra:
        raise ValueError(f"YAML loop '{loop.name}' accepts only max_iterations, got {sorted(extra)}")
    if "max_iterations" in kwargs and kwargs["max_iterations"] is not None:
        loop.max_iterations = int(kwargs["max_iterations"])
    return loop
