"""Loop DSL parser — parse YAML loop definitions, compile to execution plans."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from backend.models import LoopDefinition, LoopNode, LoopEdge


@dataclass
class PanelMember:
    """A reviewer that participates in a consensus panel step."""
    node_id: str
    role: str
    prompt_ref: str


@dataclass
class ExecutionStep:
    """A single step in a compiled execution plan.

    ``kind`` is ``"single"`` for an ordinary one-adapter-call step, or
    ``"consensus"`` for a step that fans out to every member of ``panel``
    concurrently and merges their critiques.
    """
    node_id: str
    role: str
    prompt_ref: str
    condition: str = "always"
    config: dict[str, Any] = field(default_factory=dict)
    kind: str = "single"
    panel: list[PanelMember] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """Ordered list of steps with conditions, compiled from a LoopDefinition.

    ``warnings`` lists edges the linear plan cannot represent (conditional
    back-edges, which are read as the loop's return to an earlier step, and
    edges leaving conditional-only nodes, which are not followed)."""
    steps: list[ExecutionStep] = field(default_factory=list)
    cycle_length: int = 0  # how many steps before the loop repeats
    warnings: list[str] = field(default_factory=list)


def _stop_value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def normalize_stop_conditions(raw: Any) -> list[str]:
    """Normalise YAML ``stop_conditions`` to the ``list[str]`` model form.

    Accepts a list of strings (``"max_iterations:40"``), a list mixing strings
    and one-key mappings (``- grade_at_least: B``), or a mapping
    (``{max_iterations: 40, grade_at_least: B}``). Mapping entries become
    ``"key:value"`` strings (lists are comma-joined).
    """
    if raw is None:
        return ["user_stop", "budget_exceeded"]
    if isinstance(raw, dict):
        return [f"{k}:{_stop_value_text(v)}" for k, v in raw.items()]
    if isinstance(raw, str):
        return [raw]
    out: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            out.extend(f"{k}:{_stop_value_text(v)}" for k, v in item.items())
        else:
            out.append(str(item))
    return out


def parse_loop_yaml(yaml_source: str) -> LoopDefinition:
    """Parse a YAML string into a LoopDefinition.

    Expected YAML format:
        name: my_loop
        description: ...
        nodes:
          - id: executor
            role: executor
            prompt_ref: executor/mi_executor
        edges:
          - source: executor
            target: reviewer
            condition: always
        stop_conditions:            # optional, see parse_stop_conditions
          - "max_iterations:20"
          - "grade_at_least:B"
        max_iterations: 50
    """
    data = yaml.safe_load(yaml_source)
    if not isinstance(data, dict):
        raise ValueError("YAML must parse to a dictionary")

    nodes = [LoopNode(**n) for n in data.get("nodes", []) or []]
    edges = [LoopEdge(**e) for e in data.get("edges", []) or []]

    return LoopDefinition(
        name=data.get("name", "unnamed"),
        description=data.get("description", "") or "",
        version=str(data.get("version", "1.0.0")),
        nodes=nodes,
        edges=edges,
        stop_conditions=normalize_stop_conditions(data.get("stop_conditions")),
        max_iterations=data.get("max_iterations", 50),
        config=data.get("config", {}) or {},
    )


# ── Stop conditions ──────────────────────────────────────────────────

# Always-active conditions that need no evaluation (listed for documentation).
INFORMATIONAL_STOP_CONDITIONS = frozenset({"user_stop", "budget_exceeded"})


def _to_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def _to_grade(value: str) -> str:
    from backend.orchestrator.convergence import grade_score

    g = str(value).strip().upper()
    if grade_score(g) is None:
        raise ValueError(f"unknown grade {value!r} (use A+..F, PASS or FAIL)")
    return g


def _to_signals(value: str) -> list[str]:
    from backend.orchestrator.convergence import _parse_signals

    return list(_parse_signals(value))


def _to_positive_int(value: str) -> int:
    n = int(str(value).strip())
    if n < 1:
        raise ValueError(f"expected an integer >= 1, got {value!r}")
    return n


def _to_nonneg_float(value: str) -> float:
    x = float(str(value).strip())
    if x < 0:
        raise ValueError(f"expected a number >= 0, got {value!r}")
    return x


def _to_revision_budget(value: str) -> Optional[int]:
    """``revision_budget:N`` (N >= 0) or ``revision_budget:none`` (disabled)."""
    from backend.orchestrator.convergence import parse_revision_budget

    return parse_revision_budget(str(value))


#: Stop-condition keys and how their values are parsed. Every key maps onto
#: the run-config key of the same name; a run-config value overrides it.
#: ``revision_budget`` may parse to ``None`` (disabled): the engine then reads
#: the key's presence, not only its value.
STOP_CONDITION_KEYS: dict[str, Any] = {
    "max_iterations": _to_positive_int,
    "budget_max_tokens": _to_positive_int,
    "budget_max_cost": _to_nonneg_float,
    "revision_budget": _to_revision_budget,
    "grade_at_least": _to_grade,
    "convergence_enabled": _to_bool,
    "convergence_window": _to_positive_int,
    "convergence_min_iterations": lambda v: int(str(v).strip()),
    "convergence_similarity_threshold": float,
    "convergence_signals": _to_signals,
    "convergence_rule": lambda v: str(v).strip().lower(),
    "convergence_k": _to_positive_int,
    "convergence_require_no_critical": _to_bool,
    "convergence_required_signals": _to_signals,
    "consensus_gate": _to_bool,
}

_STOP_KEY_ALIASES = {
    "max_tokens": "budget_max_tokens",
    "budget_tokens": "budget_max_tokens",
    "max_cost": "budget_max_cost",
    "budget_cost": "budget_max_cost",
    "min_grade": "grade_at_least",
}


@dataclass
class StopConditions:
    """Parsed ``LoopDefinition.stop_conditions``."""
    config: dict[str, Any] = field(default_factory=dict)  # run-config-style keys
    flags: list[str] = field(default_factory=list)        # on_flag:<flag> stops
    informational: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)


def parse_stop_conditions(conditions: Any) -> StopConditions:
    """Parse stop conditions into run-config keys, flags and unknown entries.

    Supported entries (``key:value`` or ``key=value``):

    - ``max_iterations:N``: stop after N iterations (an additional cap; the
      run's own ``max_iterations`` always applies too);
    - ``budget_max_tokens:N`` / ``budget_max_cost:X``: token / cost budgets;
    - ``revision_budget:N``: at most N executor revisions after the initial
      submission (the last one is still reviewed); ``revision_budget:none``
      disables the built-in default of 4;
    - ``grade_at_least:G``: stop after a review or consensus step whose grade
      is at least G (A+ > A > A- > B+ ... > F);
    - ``on_flag:<flag>``: stop when ``run.config["flags"][<flag>]`` is truthy;
    - ``convergence_enabled``, ``convergence_window``,
      ``convergence_min_iterations``, ``convergence_similarity_threshold``,
      ``convergence_signals`` (comma list), ``convergence_rule``,
      ``convergence_k``, ``convergence_require_no_critical``,
      ``consensus_gate``: convergence / gate overrides;
    - ``user_stop``, ``budget_exceeded``: always active (informational).

    Unknown entries are returned in ``unknown`` (the engine reports and
    ignores them). Malformed values raise ``ValueError``.
    """
    out = StopConditions()
    for raw in normalize_stop_conditions(conditions):
        entry = str(raw).strip()
        if not entry:
            continue
        if entry in INFORMATIONAL_STOP_CONDITIONS:
            out.informational.append(entry)
            continue
        if entry.startswith("on_flag:"):
            flag = entry.split(":", 1)[1].strip()
            if not flag:
                raise ValueError(f"stop condition {entry!r}: empty flag name")
            out.flags.append(flag)
            continue
        sep_idx = [i for i in (entry.find(":"), entry.find("=")) if i > 0]
        if not sep_idx:
            out.unknown.append(entry)
            continue
        i = min(sep_idx)
        key, value = entry[:i].strip(), entry[i + 1:].strip()
        key = _STOP_KEY_ALIASES.get(key, key)
        parser = STOP_CONDITION_KEYS.get(key)
        if parser is None:
            out.unknown.append(entry)
            continue
        try:
            out.config[key] = parser(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid stop condition {entry!r}: {exc}") from exc
    return out


#: Consensus-merger node config keys honoured as defaults for the consensus
#: step (the run config overrides them): node key -> run-config key.
MERGER_NODE_CONFIG_KEYS = {
    "similarity_method": "consensus_similarity_method",
    "similarity_threshold": "consensus_similarity_threshold",
    "consensus_threshold": "consensus_threshold",
    "role_weights": "consensus_role_weights",
    "embedding_model": "consensus_embedding_model",
}


def merger_config_defaults(node_config: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Run-config defaults declared on a ``consensus_merger`` node.

    ``similarity_method``, ``similarity_threshold``, ``consensus_threshold``,
    ``role_weights`` and ``embedding_model`` map onto the corresponding
    ``consensus_*`` keys; keys already named ``consensus_*`` pass through.
    """
    out: dict[str, Any] = {}
    for k, v in (node_config or {}).items():
        if v is None:
            continue
        if k in MERGER_NODE_CONFIG_KEYS:
            out[MERGER_NODE_CONFIG_KEYS[k]] = v
        elif k.startswith("consensus_"):
            out[k] = v
    return out


def validate_loop(loop_def: LoopDefinition) -> list[str]:
    """Validate a loop definition. Returns a list of error messages (empty = valid)."""
    errors: list[str] = []

    if not loop_def.nodes:
        errors.append("Loop must have at least one node")
        return errors

    node_ids = {n.id for n in loop_def.nodes}

    # Check for duplicate node IDs
    if len(node_ids) != len(loop_def.nodes):
        seen = set()
        for n in loop_def.nodes:
            if n.id in seen:
                errors.append(f"Duplicate node ID: {n.id}")
            seen.add(n.id)

    # Check edges reference valid nodes
    for edge in loop_def.edges:
        if edge.source not in node_ids:
            errors.append(f"Edge source '{edge.source}' not found in nodes")
        if edge.target not in node_ids:
            errors.append(f"Edge target '{edge.target}' not found in nodes")

    # Check all nodes are reachable (simple BFS from first node)
    if loop_def.edges:
        adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}
        for edge in loop_def.edges:
            if edge.source in adjacency:
                adjacency[edge.source].append(edge.target)

        visited: set[str] = set()
        queue = [loop_def.nodes[0].id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            queue.extend(adjacency.get(current, []))

        unreachable = node_ids - visited
        if unreachable:
            errors.append(f"Unreachable nodes: {unreachable}")

    # Validate condition syntax
    valid_prefixes = ("always", "every_k:", "on_flag:")
    for edge in loop_def.edges:
        if not any(edge.condition.startswith(p) for p in valid_prefixes):
            errors.append(
                f"Invalid condition '{edge.condition}' on edge "
                f"{edge.source}->{edge.target}"
            )

    return errors


CONSENSUS_ROLE = "consensus_merger"


def _find_consensus_nodes(loop_def: LoopDefinition) -> dict[str, list[PanelMember]]:
    """Map each consensus-merger node id to its reviewer panel.

    A node is a consensus merger if its role is ``consensus_merger``. Its panel
    is every node that reaches it through an ``always`` edge, in node-definition
    order. This lets a loop declare a reviewer panel of any size by fanning out
    from the executor to N reviewers that all converge on one merger node.
    """
    node_map = {n.id: n for n in loop_def.nodes}
    merger_ids = {n.id for n in loop_def.nodes if n.role == CONSENSUS_ROLE}
    panels: dict[str, list[PanelMember]] = {mid: [] for mid in merger_ids}
    if not merger_ids:
        return panels

    # Preserve node-definition order for determinism.
    order = {n.id: i for i, n in enumerate(loop_def.nodes)}
    for mid in merger_ids:
        members = [
            e.source for e in loop_def.edges
            if e.target == mid and e.condition == "always" and e.source in node_map
        ]
        members.sort(key=lambda nid: order.get(nid, 0))
        panels[mid] = [
            PanelMember(node_id=node_map[m].id, role=node_map[m].role,
                        prompt_ref=node_map[m].prompt_ref)
            for m in members
        ]
    return panels


def compile_loop(loop_def: LoopDefinition) -> ExecutionPlan:
    """Compile a LoopDefinition into an ExecutionPlan (ordered steps).

    Start from the first node and follow ``always`` edges, building a
    repeating cycle of steps. Ordinary nodes become ``single`` steps. When the
    walk reaches a ``consensus_merger`` node, its reviewer panel (all nodes
    feeding it via ``always`` edges) is absorbed into one ``consensus`` step
    that runs the panel concurrently; the panel members are not emitted as
    standalone steps.

    A conditional edge (``every_k:N`` / ``on_flag:F``) to a node that is not
    already part of the cycle becomes a gated step right after its source. A
    conditional edge back to a node already in the cycle (e.g.
    ``reviewer -> executor`` on ``on_flag:needs_revision``) is NOT a step: the
    cycle already returns there, so it is read as the loop-back edge. When a
    node has no ``always`` successor the cycle ends there and wraps to the
    first node. Edges leaving conditional-only nodes are not followed; both
    cases are reported in ``ExecutionPlan.warnings``.
    """
    errors = validate_loop(loop_def)
    if errors:
        raise ValueError(f"Invalid loop definition: {'; '.join(errors)}")

    node_map = {n.id: n for n in loop_def.nodes}

    # Build adjacency: source -> list of (target, condition)
    adjacency: dict[str, list[tuple[str, str]]] = {n.id: [] for n in loop_def.nodes}
    for edge in loop_def.edges:
        adjacency[edge.source].append((edge.target, edge.condition))

    panels = _find_consensus_nodes(loop_def)
    # Nodes absorbed into a consensus step must not be emitted on their own.
    panel_member_ids = {pm.node_id for members in panels.values() for pm in members}

    # Walk the graph from the first node, collecting steps until we revisit.
    steps: list[ExecutionStep] = []
    visited_order: list[str] = []
    warnings: list[str] = []
    gated_ids: list[str] = []
    current = loop_def.nodes[0].id

    max_walk = len(loop_def.nodes) * 2 + 1  # safety bound
    for _ in range(max_walk):
        if current in visited_order:
            break
        visited_order.append(current)
        node = node_map[current]
        outgoing = adjacency.get(current, [])

        if node.role == CONSENSUS_ROLE:
            # Emit one consensus step carrying the full reviewer panel.
            steps.append(ExecutionStep(
                node_id=node.id,
                role=node.role,
                prompt_ref=node.prompt_ref,
                condition="always",
                config=node.config,
                kind="consensus",
                panel=panels.get(node.id, []),
            ))
        else:
            steps.append(ExecutionStep(
                node_id=node.id,
                role=node.role,
                prompt_ref=node.prompt_ref,
                condition="always",
                config=node.config,
            ))

        # Classify successors, skipping panel members (absorbed into consensus).
        always_edges = [t for t, c in outgoing
                        if c == "always" and t not in panel_member_ids]
        conditional_edges = [(t, c) for t, c in outgoing
                             if c != "always" and t not in panel_member_ids]

        # If this node fans out to a panel, route to the consensus node next.
        panel_targets = [t for t, c in outgoing
                         if c == "always" and t in panel_member_ids]
        if panel_targets and not always_edges:
            merger_id = next(
                (m for m, members in panels.items()
                 if any(pm.node_id in panel_targets for pm in members)),
                None,
            )
            if merger_id is not None:
                always_edges = [merger_id]

        # Conditional steps run at the same point but gated by their condition.
        for target, condition in conditional_edges:
            if target in visited_order or (always_edges and target == always_edges[0]):
                warnings.append(
                    f"conditional edge {current}->{target} ({condition}) returns to a step "
                    f"already in the cycle; it is read as the loop-back, not as an extra step"
                )
                continue
            cond_node = node_map[target]
            steps.append(ExecutionStep(
                node_id=cond_node.id,
                role=cond_node.role,
                prompt_ref=cond_node.prompt_ref,
                condition=condition,
                config=cond_node.config,
            ))
            if target not in gated_ids:
                gated_ids.append(target)

        if always_edges:
            current = always_edges[0]
        else:
            break  # end of the cycle: wrap around to the first node

    # Edges leaving nodes that only run as gated steps are not followed.
    in_cycle = set(visited_order)
    for nid in gated_ids:
        if nid in in_cycle:
            continue
        for target, condition in adjacency.get(nid, []):
            if condition == "always" and target in in_cycle:
                continue  # returning to the cycle is what the plan does anyway
            warnings.append(
                f"edge {nid}->{target} ({condition}) leaves a conditional-only step and is "
                f"not followed; after {nid} the cycle continues with its next step"
            )
    return ExecutionPlan(steps=steps, cycle_length=len(steps), warnings=warnings)
