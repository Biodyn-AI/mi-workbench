"""Loop DSL parser — parse YAML loop definitions, compile to execution plans."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    """Ordered list of steps with conditions, compiled from a LoopDefinition."""
    steps: list[ExecutionStep] = field(default_factory=list)
    cycle_length: int = 0  # how many steps before the loop repeats


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
        max_iterations: 50
    """
    data = yaml.safe_load(yaml_source)
    if not isinstance(data, dict):
        raise ValueError("YAML must parse to a dictionary")

    nodes = [LoopNode(**n) for n in data.get("nodes", [])]
    edges = [LoopEdge(**e) for e in data.get("edges", [])]

    return LoopDefinition(
        name=data.get("name", "unnamed"),
        description=data.get("description", ""),
        version=data.get("version", "1.0.0"),
        nodes=nodes,
        edges=edges,
        stop_conditions=data.get("stop_conditions", ["user_stop", "budget_exceeded"]),
        max_iterations=data.get("max_iterations", 50),
        config=data.get("config", {}),
    )


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

    Start from the first node and follow edges, building a repeating cycle of
    steps. Ordinary nodes become ``single`` steps. When the walk reaches a
    ``consensus_merger`` node, its reviewer panel (all nodes feeding it via
    ``always`` edges) is absorbed into one ``consensus`` step that runs the
    panel concurrently; the panel members are not emitted as standalone steps.
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
            cond_node = node_map[target]
            steps.append(ExecutionStep(
                node_id=cond_node.id,
                role=cond_node.role,
                prompt_ref=cond_node.prompt_ref,
                condition=condition,
                config=cond_node.config,
            ))

        if always_edges:
            current = always_edges[0]
        elif conditional_edges:
            current = conditional_edges[0][0]
        else:
            break

    return ExecutionPlan(steps=steps, cycle_length=len(steps))
