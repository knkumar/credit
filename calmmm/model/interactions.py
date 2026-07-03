from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import pytensor.tensor as pt


@dataclass
class ChannelInteraction:
    """
    One directed edge in a channel interaction graph: `source`'s signal
    multiplicatively rescales `target`'s contribution.

    prior : "half_normal" (default) draws gamma ~ HalfNormal(prior_sigma) — boost-only,
            contribution can only increase. "normal" draws gamma ~ Normal(0, prior_sigma) —
            two-sided, for hypothesized suppression/cannibalization.
    """
    source: str
    target: str
    prior: Literal["half_normal", "normal"] = "half_normal"
    prior_sigma: float = 0.5


@dataclass
class InteractionGraph:
    """A directed graph of ChannelInteraction edges. Validated on construction."""
    edges: list[ChannelInteraction]

    def __post_init__(self) -> None:
        seen_pairs = set()
        for edge in self.edges:
            if edge.source == edge.target:
                raise ValueError(
                    f"self-loop not allowed: '{edge.source}' -> '{edge.target}'"
                )
            pair = (edge.source, edge.target)
            if pair in seen_pairs:
                raise ValueError(f"duplicate edge: '{edge.source}' -> '{edge.target}'")
            seen_pairs.add(pair)
        # Raises ValueError if the edges contain a cycle.
        self.topological_order()

    def _channel_names(self) -> list[str]:
        names: list[str] = []
        for edge in self.edges:
            if edge.source not in names:
                names.append(edge.source)
            if edge.target not in names:
                names.append(edge.target)
        return names

    def incoming_edges(self, channel: str) -> list[ChannelInteraction]:
        return [e for e in self.edges if e.target == channel]

    def topological_order(self) -> list[str]:
        """
        Channel names touched by this graph, base channels (no incoming edges) first.
        Raises ValueError if the edges contain a cycle.
        """
        nodes = self._channel_names()
        in_degree = {n: 0 for n in nodes}
        adjacency: dict[str, list[str]] = {n: [] for n in nodes}
        for edge in self.edges:
            adjacency[edge.source].append(edge.target)
            in_degree[edge.target] += 1

        queue = [n for n in nodes if in_degree[n] == 0]
        order: list[str] = []
        remaining_in_degree = dict(in_degree)
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in adjacency[node]:
                remaining_in_degree[neighbor] -= 1
                if remaining_in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(nodes):
            unresolved = [n for n in nodes if n not in order]
            raise ValueError(f"interaction graph contains a cycle involving: {unresolved}")
        return order

    def validate_channels(self, known_channels: list[str]) -> None:
        unknown = [n for n in self._channel_names() if n not in known_channels]
        if unknown:
            raise ValueError(
                f"interaction graph references unknown channel(s): {unknown}. "
                f"Known channels: {known_channels}"
            )
