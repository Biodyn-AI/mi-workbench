"""Loop DSL parser — parse YAML loop definitions, compile to execution plans."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from backend.models import LoopDefinition, LoopNode, LoopEdge


@dataclass
class ExecutionStep:
    """A single step in a compiled execution plan."""
    node_id: str
    role: str
    prompt_ref: str
    condition: str = "always"
    config: dict[str, Any] = field(default_factory=dict)


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


def compile_loop(loop_def: LoopDefinition) -> ExecutionPlan:
    """Compile a LoopDefinition into an ExecutionPlan (ordered steps).

    Uses topological-like ordering: start from the first node,
    follow edges in order, building a cycle of steps.
    """
    errors = validate_loop(loop_def)
    if errors:
        raise ValueError(f"Invalid loop definition: {'; '.join(errors)}")

    node_map = {n.id: n for n in loop_def.nodes}

    # Build adjacency: source -> list of (target, condition)
    adjacency: dict[str, list[tuple[str, str]]] = {n.id: [] for n in loop_def.nodes}
    for edge in loop_def.edges:
        adjacency[edge.source].append((edge.target, edge.condition))

    # Walk the graph from the first node, collecting steps until we revisit
    steps: list[ExecutionStep] = []
    visited_order: list[str] = []
    current = loop_def.nodes[0].id

    max_walk = len(loop_def.nodes) * 2 + 1  # safety bound
    for _ in range(max_walk):
        if current in visited_order:
            break
        visited_order.append(current)
        node = node_map[current]
        # Find the outgoing edge (prefer "always" edges)
        outgoing = adjacency.get(current, [])
        # Add a step for this node; attach the condition from the edge that led here
        steps.append(ExecutionStep(
            node_id=node.id,
            role=node.role,
            prompt_ref=node.prompt_ref,
            condition="always",
            config=node.config,
        ))
        # Pick next node: prefer "always" edges first
        always_edges = [t for t, c in outgoing if c == "always"]
        conditional_edges = [(t, c) for t, c in outgoing if c != "always"]

        # Add conditional steps (they run at the same point but with conditions)
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
