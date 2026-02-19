"""Built-in loop presets for common MI research workflows."""
from backend.models import LoopDefinition, LoopNode, LoopEdge


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
            prompt_ref="reviewer/adversarial_reviewer",
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
                    f"adversarial red-teams every {adversarial_every_k} iterations.",
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
            prompt_ref="followup/idea_generator",
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


PRESETS: dict[str, callable] = {
    "executor_reviewer": executor_reviewer_preset,
    "research_followups": research_followups_preset,
    "reviewer_consensus": reviewer_consensus_preset,
}


def get_preset(name: str, **kwargs) -> LoopDefinition:
    """Get a loop definition from a preset name."""
    factory = PRESETS.get(name)
    if factory is None:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return factory(**kwargs)
