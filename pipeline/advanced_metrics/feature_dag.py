"""Feature DAG validation and execution ordering."""

from __future__ import annotations

from collections import defaultdict, deque

from .feature_registry import FeatureSpec


def _expand_with_dependencies(
    registry: dict[str, FeatureSpec],
    requested: list[str] | None,
) -> set[str]:
    if requested is None:
        requested_set = set(registry.keys())
    else:
        unknown = sorted(name for name in requested if name not in registry)
        if unknown:
            raise ValueError(f"Requested features not found in registry: {unknown}")
        requested_set = set(requested)

    expanded = set(requested_set)
    stack = list(requested_set)
    while stack:
        name = stack.pop()
        for dep in registry[name].dependencies:
            if dep not in registry:
                raise ValueError(f"Feature '{name}' depends on unknown feature '{dep}'")
            if dep not in expanded:
                expanded.add(dep)
                stack.append(dep)
    return expanded


def resolve_execution_order(
    registry: dict[str, FeatureSpec],
    requested: list[str] | None = None,
) -> list[str]:
    selected = _expand_with_dependencies(registry, requested)

    in_degree: dict[str, int] = {name: 0 for name in selected}
    graph: dict[str, list[str]] = defaultdict(list)
    for name in selected:
        for dep in registry[name].dependencies:
            if dep not in selected:
                continue
            graph[dep].append(name)
            in_degree[name] += 1

    queue = deque(sorted(name for name, deg in in_degree.items() if deg == 0))
    ordered: list[str] = []
    while queue:
        current = queue.popleft()
        ordered.append(current)
        for nxt in sorted(graph[current]):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    if len(ordered) != len(selected):
        cyclic = sorted(name for name, deg in in_degree.items() if deg > 0)
        raise ValueError(f"Feature dependency cycle detected: {cyclic}")

    return ordered
