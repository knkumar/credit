# Channel Interaction Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a general `ChannelInteraction`/`InteractionGraph` mechanism to `calmmm`'s `HierarchicalMMM` that lets one channel's signal multiplicatively boost another channel's contribution (with multi-hop chain support), then wire up the concrete `direct_mail -> search` edge.

**Architecture:** A new `calmmm/model/interactions.py` holds the graph data model (validation, topological sort) and a `build_interaction_step()` function that, given a base `channel_contrib` tensor, returns the boosted version by walking the graph in topological order and applying a per-edge multiplicative rescale inside an active PyMC model context. `_build_media_hierarchy` (`calmmm/model/components.py`) gets one new optional keyword parameter (`apply_interactions`, default `None`) so its existing behavior is unchanged unless a graph is supplied. `HierarchicalMMM` (`calmmm/model/mmm.py`) gets one new constructor parameter (`interaction_graph`, default `None`) that builds the closure and passes it through.

**Tech Stack:** Python 3.11, PyMC 5, PyTensor, pytest (project convention: `@pytest.mark.slow` for tests that run real PyMC inference; default `addopts` deselects them).

## Global Constraints

- Default `interaction_graph=None` on `HierarchicalMMM` must produce behavior identical to the current code path — this is a testable regression requirement, not just a description.
- No changes to any file under `calmmm/attribution/` or `calmmm/calibration/` — downstream compatibility is proven by tests exercising those modules unmodified against a fit built with an active interaction graph.
- All PyMC-context test code must run inside `with pm.Model(coords=...):` (or use an already-built model), matching the pattern in `tests/model/test_components.py` and `tests/model/test_mmm.py`.
- Run tests with: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest -q <path>`. Tests marked `@pytest.mark.slow` run real PyMC inference and take minutes each — run them explicitly with `-m slow`, not as part of routine fast iteration.

---

### Task 1: `ChannelInteraction` / `InteractionGraph` data model

**Files:**
- Create: `calmmm/model/interactions.py`
- Test: `tests/model/test_interactions.py`

**Interfaces:**
- Produces: `ChannelInteraction(source: str, target: str, prior: Literal["half_normal","normal"] = "half_normal", prior_sigma: float = 0.5)` — plain dataclass, no validation of its own.
- Produces: `InteractionGraph(edges: list[ChannelInteraction])` — validates on construction (raises `ValueError` on self-loop, duplicate edge, or cycle). Methods: `.topological_order() -> list[str]`, `.incoming_edges(channel: str) -> list[ChannelInteraction]`, `.validate_channels(known_channels: list[str]) -> None` (raises `ValueError` on unknown channel name).

- [ ] **Step 1: Write the failing tests**

Create `tests/model/test_interactions.py`:

```python
import pytest

from calmmm.model.interactions import ChannelInteraction, InteractionGraph


def test_channel_interaction_defaults():
    edge = ChannelInteraction(source="direct_mail", target="search")
    assert edge.prior == "half_normal"
    assert edge.prior_sigma == 0.5


def test_interaction_graph_topological_order_simple():
    graph = InteractionGraph(edges=[ChannelInteraction(source="direct_mail", target="search")])
    order = graph.topological_order()
    assert order.index("direct_mail") < order.index("search")


def test_interaction_graph_topological_order_chain():
    graph = InteractionGraph(edges=[
        ChannelInteraction(source="direct_mail", target="search"),
        ChannelInteraction(source="search", target="social"),
    ])
    order = graph.topological_order()
    assert order.index("direct_mail") < order.index("search") < order.index("social")


def test_interaction_graph_incoming_edges():
    graph = InteractionGraph(edges=[
        ChannelInteraction(source="direct_mail", target="search"),
        ChannelInteraction(source="affiliate", target="search"),
        ChannelInteraction(source="search", target="social"),
    ])
    incoming = graph.incoming_edges("search")
    assert {e.source for e in incoming} == {"direct_mail", "affiliate"}
    assert graph.incoming_edges("direct_mail") == []


def test_interaction_graph_rejects_self_loop():
    with pytest.raises(ValueError, match="self-loop"):
        InteractionGraph(edges=[ChannelInteraction(source="search", target="search")])


def test_interaction_graph_rejects_duplicate_edge():
    with pytest.raises(ValueError, match="duplicate edge"):
        InteractionGraph(edges=[
            ChannelInteraction(source="direct_mail", target="search"),
            ChannelInteraction(source="direct_mail", target="search"),
        ])


def test_interaction_graph_rejects_cycle():
    with pytest.raises(ValueError, match="cycle"):
        InteractionGraph(edges=[
            ChannelInteraction(source="search", target="social"),
            ChannelInteraction(source="social", target="search"),
        ])


def test_interaction_graph_rejects_longer_cycle():
    with pytest.raises(ValueError, match="cycle"):
        InteractionGraph(edges=[
            ChannelInteraction(source="a", target="b"),
            ChannelInteraction(source="b", target="c"),
            ChannelInteraction(source="c", target="a"),
        ])


def test_interaction_graph_validate_channels_rejects_unknown():
    graph = InteractionGraph(edges=[ChannelInteraction(source="direct_mail", target="search")])
    with pytest.raises(ValueError, match="unknown channel"):
        graph.validate_channels(["search", "social"])


def test_interaction_graph_validate_channels_accepts_known():
    graph = InteractionGraph(edges=[ChannelInteraction(source="direct_mail", target="search")])
    graph.validate_channels(["direct_mail", "search", "social", "affiliate"])  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/model/test_interactions.py -v`
Expected: FAIL / ERROR on every test with `ModuleNotFoundError: No module named 'calmmm.model.interactions'`

- [ ] **Step 3: Implement the data model**

Create `calmmm/model/interactions.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/model/test_interactions.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add calmmm/model/interactions.py tests/model/test_interactions.py
git commit -m "feat: add ChannelInteraction/InteractionGraph data model"
```

---

### Task 2: `build_interaction_step()` — the PyMC boost computation

**Files:**
- Modify: `calmmm/model/interactions.py`
- Test: `tests/model/test_interactions.py`

**Interfaces:**
- Consumes: `ChannelInteraction`, `InteractionGraph` from Task 1.
- Produces: `build_interaction_step(graph: InteractionGraph, *, channels: list[str], X_adstock: pt.TensorVariable) -> Callable[[pt.TensorVariable], pt.TensorVariable]`. `X_adstock` has shape `[T, G, C]` (channel axis ordered exactly as `channels`). The returned callable takes a `channel_contrib` tensor of shape `[T, G, K, C]` and returns a tensor of the same shape. Must be called inside an active `pm.Model()` context (creates `gamma_<source>_<target>` free RVs, one per edge).

- [ ] **Step 1: Write the failing tests**

Append to `tests/model/test_interactions.py`:

```python
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from calmmm.model.interactions import build_interaction_step


CHANNELS_3 = ["direct_mail", "search", "social"]


def _coords_3ch(T=6, G=2, K=1):
    return {"kpi": [f"kpi{i}" for i in range(K)], "geo": [f"geo{i}" for i in range(G)], "channel": CHANNELS_3}


def test_build_interaction_step_shape_unchanged():
    T, G, K, C = 6, 2, 1, 3
    rng = np.random.default_rng(0)
    contrib_val = rng.random((T, G, K, C)).astype("float64")
    adstock_val = rng.random((T, G, C)).astype("float64")

    graph = InteractionGraph(edges=[ChannelInteraction(source="direct_mail", target="search")])
    with pm.Model(coords=_coords_3ch(T, G, K)):
        apply = build_interaction_step(graph, channels=CHANNELS_3, X_adstock=pt.as_tensor_variable(adstock_val))
        boosted = apply(pt.as_tensor_variable(contrib_val))
        val = pm.draw(boosted, random_seed=0)
    assert val.shape == (T, G, K, C)


def test_build_interaction_step_boosts_target_only():
    """Only the target channel's slice changes; source and untouched channels are unchanged."""
    T, G, K, C = 4, 2, 1, 3
    contrib_val = np.ones((T, G, K, C), dtype="float64")
    adstock_val = np.zeros((T, G, C), dtype="float64")
    adstock_val[:, :, 0] = 2.0  # direct_mail's own adstocked signal is nonzero

    graph = InteractionGraph(edges=[ChannelInteraction(source="direct_mail", target="search", prior_sigma=0.1)])
    with pm.Model(coords=_coords_3ch(T, G, K)):
        apply = build_interaction_step(graph, channels=CHANNELS_3, X_adstock=pt.as_tensor_variable(adstock_val))
        boosted = apply(pt.as_tensor_variable(contrib_val))
        val = pm.draw(boosted, random_seed=0)

    dm_idx, search_idx, social_idx = 0, 1, 2
    assert np.allclose(val[:, :, :, dm_idx], 1.0)      # direct_mail's own slice untouched
    assert np.allclose(val[:, :, :, social_idx], 1.0)  # social untouched (no edge)
    # search's slice can only be boosted (gamma >= 0, signal >= 0), never reduced below 1.0
    assert np.all(val[:, :, :, search_idx] >= 1.0)


def test_build_interaction_step_registers_gamma_rv():
    T, G, K, C = 4, 2, 1, 3
    contrib_val = np.ones((T, G, K, C), dtype="float64")
    adstock_val = np.ones((T, G, C), dtype="float64")

    graph = InteractionGraph(edges=[ChannelInteraction(source="direct_mail", target="search")])
    with pm.Model(coords=_coords_3ch(T, G, K)) as model:
        apply = build_interaction_step(graph, channels=CHANNELS_3, X_adstock=pt.as_tensor_variable(adstock_val))
        apply(pt.as_tensor_variable(contrib_val))
        names = {v.name for v in model.free_RVs}
    assert "gamma_direct_mail_search" in names


def test_build_interaction_step_logp_finite():
    T, G, K, C = 5, 2, 1, 3
    rng = np.random.default_rng(1)
    contrib_val = rng.random((T, G, K, C)).astype("float64") + 0.1
    adstock_val = rng.random((T, G, C)).astype("float64")

    graph = InteractionGraph(edges=[ChannelInteraction(source="direct_mail", target="search")])
    with pm.Model(coords=_coords_3ch(T, G, K)) as model:
        apply = build_interaction_step(graph, channels=CHANNELS_3, X_adstock=pt.as_tensor_variable(adstock_val))
        apply(pt.as_tensor_variable(contrib_val))
        lp = model.compile_logp()(model.initial_point())
    assert np.isfinite(lp)


def test_build_interaction_step_normal_prior_registers_two_sided_gamma():
    T, G, K, C = 4, 2, 1, 3
    contrib_val = np.ones((T, G, K, C), dtype="float64")
    adstock_val = np.ones((T, G, C), dtype="float64")

    graph = InteractionGraph(edges=[
        ChannelInteraction(source="direct_mail", target="search", prior="normal", prior_sigma=0.5)
    ])
    with pm.Model(coords=_coords_3ch(T, G, K)) as model:
        apply = build_interaction_step(graph, channels=CHANNELS_3, X_adstock=pt.as_tensor_variable(adstock_val))
        boosted = apply(pt.as_tensor_variable(contrib_val))
        val = pm.draw(boosted, random_seed=0)
    search_idx = 1
    # exp(gamma * signal) is strictly positive regardless of gamma's sign
    assert np.all(val[:, :, :, search_idx] > 0.0)


def test_build_interaction_step_chain_uses_boosted_upstream_signal():
    """A chained edge (search->social) actually alters social's contribution,
    confirming it reads search's boosted signal rather than being a no-op."""
    T, G, K, C = 4, 2, 1, 3
    contrib_val = np.ones((T, G, K, C), dtype="float64")
    adstock_val = np.zeros((T, G, C), dtype="float64")
    adstock_val[:, :, 0] = 5.0  # direct_mail: large own signal

    graph = InteractionGraph(edges=[
        ChannelInteraction(source="direct_mail", target="search", prior_sigma=1.0),
        ChannelInteraction(source="search", target="social", prior_sigma=1.0),
    ])
    with pm.Model(coords=_coords_3ch(T, G, K)) as model:
        apply = build_interaction_step(graph, channels=CHANNELS_3, X_adstock=pt.as_tensor_variable(adstock_val))
        boosted = apply(pt.as_tensor_variable(contrib_val))
        names = {v.name for v in model.free_RVs}
        val = pm.draw(boosted, random_seed=0)

    assert "gamma_direct_mail_search" in names
    assert "gamma_search_social" in names
    social_idx = 2
    assert not np.allclose(val[:, :, :, social_idx], 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/model/test_interactions.py -v -k build_interaction_step`
Expected: FAIL / ERROR on every new test with `ImportError: cannot import name 'build_interaction_step'`

- [ ] **Step 3: Implement `build_interaction_step`**

Append to `calmmm/model/interactions.py`:

```python
import pymc as pm


def build_interaction_step(
    graph: InteractionGraph,
    *,
    channels: list[str],
    X_adstock: pt.TensorVariable,
) -> Callable[[pt.TensorVariable], pt.TensorVariable]:
    """
    Return a function that applies `graph`'s edges to a base channel_contrib
    tensor [T, G, K, C] and returns the boosted tensor of the same shape.

    Must be invoked inside an active pm.Model() context — creates one
    gamma_<source>_<target> free RV per edge.

    For a base channel (no incoming edges of its own), the source signal is
    its own adstocked spend, X_adstock[..., idx] — already scaled to a
    roughly [0, few] range because raw spend is divided by that channel's
    historical max before adstock (existing pipeline behavior), broadcast
    over the K axis. For a chained edge (source was itself boosted upstream
    in topological order), the source signal is that channel's finalized,
    boosted contribution, normalized by its own mean absolute value so
    gamma's prior scale stays comparable regardless of where in the graph
    an edge sits.
    """
    graph.validate_channels(channels)
    channel_idx = {name: i for i, name in enumerate(channels)}
    order = graph.topological_order()

    def apply(channel_contrib: pt.TensorVariable) -> pt.TensorVariable:
        slices = {name: channel_contrib[:, :, :, channel_idx[name]] for name in channels}
        chain_scales: dict[str, pt.TensorVariable] = {}

        for name in order:
            incoming = graph.incoming_edges(name)
            if not incoming:
                continue
            contrib = slices[name]  # [T, G, K]
            for edge in incoming:
                if edge.source in chain_scales:
                    source_contrib = slices[edge.source]  # [T, G, K], already boosted
                    scale = chain_scales[edge.source]
                    signal = source_contrib / scale
                else:
                    signal = X_adstock[:, :, channel_idx[edge.source]][:, :, None]  # [T, G, 1]

                if edge.prior == "half_normal":
                    gamma = pm.HalfNormal(f"gamma_{edge.source}_{edge.target}", sigma=edge.prior_sigma)
                    contrib = contrib * (1.0 + gamma * signal)
                else:  # "normal" — two-sided, exponential form guarantees strict positivity
                    gamma = pm.Normal(f"gamma_{edge.source}_{edge.target}", mu=0.0, sigma=edge.prior_sigma)
                    contrib = contrib * pt.exp(gamma * signal)

            slices[name] = contrib
            chain_scales[name] = pt.mean(pt.abs(contrib)) + 1e-8

        return pt.stack([slices[name] for name in channels], axis=-1)  # [T, G, K, C]

    return apply
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/model/test_interactions.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add calmmm/model/interactions.py tests/model/test_interactions.py
git commit -m "feat: add build_interaction_step for multiplicative channel boosts"
```

---

### Task 3: Wire `apply_interactions` into `_build_media_hierarchy`

**Files:**
- Modify: `calmmm/model/components.py`
- Test: `tests/model/test_components.py`

**Interfaces:**
- Consumes: nothing new from Task 1/2 directly (this task only adds a generic callable hook — it does not import `calmmm.model.interactions`, keeping `components.py` decoupled from the interaction graph's concrete types).
- Produces: `_build_media_hierarchy(X_sat, priors, *, apply_interactions: Callable[[pt.TensorVariable], pt.TensorVariable] | None = None) -> pt.TensorVariable` — same return type/shape as today (`[T, G, K]`), same `"channel_contrib"` Deterministic registration, but that Deterministic reflects `apply_interactions`'s output when provided.

- [ ] **Step 1: Write the failing tests**

Append to `tests/model/test_components.py`:

```python
def test_media_hierarchy_apply_interactions_default_none_unchanged():
    """Default apply_interactions=None must match calling without the kwarg at all."""
    T, G, K, C = 5, 2, 4, 2
    priors = PriorConfig()
    X_sat_val = np.random.default_rng(7).random((T, G, C)).astype("float64")

    with pm.Model(coords=_base_coords()) as model:
        _build_media_hierarchy(pt.as_tensor_variable(X_sat_val), priors)
        val_default = pm.draw(model["channel_contrib"], random_seed=42)

    with pm.Model(coords=_base_coords()) as model2:
        _build_media_hierarchy(pt.as_tensor_variable(X_sat_val), priors, apply_interactions=None)
        val_explicit_none = pm.draw(model2["channel_contrib"], random_seed=42)

    assert np.allclose(val_default, val_explicit_none)


def test_media_hierarchy_apply_interactions_invoked():
    """When apply_interactions is provided, its return value becomes channel_contrib."""
    T, G, K, C = 5, 2, 4, 2
    priors = PriorConfig()
    X_sat_val = np.random.default_rng(8).random((T, G, C)).astype("float64")

    def double_it(tensor):
        return tensor * 2.0

    with pm.Model(coords=_base_coords()) as model:
        _build_media_hierarchy(pt.as_tensor_variable(X_sat_val), priors, apply_interactions=None)
        base_val = pm.draw(model["channel_contrib"], random_seed=3)

    with pm.Model(coords=_base_coords()) as model2:
        _build_media_hierarchy(pt.as_tensor_variable(X_sat_val), priors, apply_interactions=double_it)
        boosted_val = pm.draw(model2["channel_contrib"], random_seed=3)

    assert np.allclose(boosted_val, base_val * 2.0)


def test_media_hierarchy_apply_interactions_affects_returned_sum():
    """The function's [T,G,K] return value (media_contrib) must reflect the boost too."""
    T, G, K, C = 5, 2, 4, 2
    priors = PriorConfig()
    X_sat_val = np.random.default_rng(9).random((T, G, C)).astype("float64")

    with pm.Model(coords=_base_coords()) as model:
        media_contrib = _build_media_hierarchy(pt.as_tensor_variable(X_sat_val), priors, apply_interactions=lambda t: t * 2.0)
        val = pm.draw(media_contrib, random_seed=5)
        cc_val = pm.draw(model["channel_contrib"], random_seed=5)

    assert np.allclose(val, cc_val.sum(axis=-1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/model/test_components.py -v -k apply_interactions`
Expected: FAIL with `TypeError: _build_media_hierarchy() got an unexpected keyword argument 'apply_interactions'`

- [ ] **Step 3: Implement the hook**

In `calmmm/model/components.py`, change the import line at the top from:

```python
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from calmmm.model.priors import PriorConfig
```

to:

```python
from typing import Callable, Optional

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from calmmm.model.priors import PriorConfig
```

Then change the `_build_media_hierarchy` signature and its last three lines. Find:

```python
def _build_media_hierarchy(
    X_sat: pt.TensorVariable,
    priors: PriorConfig,
) -> pt.TensorVariable:
```

Replace with:

```python
def _build_media_hierarchy(
    X_sat: pt.TensorVariable,
    priors: PriorConfig,
    *,
    apply_interactions: Optional[Callable[[pt.TensorVariable], pt.TensorVariable]] = None,
) -> pt.TensorVariable:
```

Add one sentence to the existing docstring, after the `Hierarchy:` block and before `scale_kpi and scale_geo can be negative...`:

```text
    If `apply_interactions` is provided, it is called on the raw per-channel
    contribution tensor [T, G, K, C] before it is registered as the
    "channel_contrib" Deterministic — its return value (same shape) becomes
    the model's channel_contrib. See calmmm.model.interactions.build_interaction_step.
```

Find the last three lines of the function body:

```python
    scale_geo_gkc = scale_geo.dimshuffle(2, 1, 0)  # [G, K, C]
    channel_contrib_tgkc = X_sat[:, :, None, :] * scale_geo_gkc[None, :, :, :]  # [T, G, K, C]
    pm.Deterministic("channel_contrib", channel_contrib_tgkc)
    return channel_contrib_tgkc.sum(axis=-1)  # [T, G, K]
```

Replace with:

```python
    scale_geo_gkc = scale_geo.dimshuffle(2, 1, 0)  # [G, K, C]
    channel_contrib_tgkc = X_sat[:, :, None, :] * scale_geo_gkc[None, :, :, :]  # [T, G, K, C]
    if apply_interactions is not None:
        channel_contrib_tgkc = apply_interactions(channel_contrib_tgkc)
    pm.Deterministic("channel_contrib", channel_contrib_tgkc)
    return channel_contrib_tgkc.sum(axis=-1)  # [T, G, K]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/model/test_components.py -v`
Expected: all tests in the file pass (existing tests + 3 new ones)

- [ ] **Step 5: Commit**

```bash
git add calmmm/model/components.py tests/model/test_components.py
git commit -m "feat: add apply_interactions hook to _build_media_hierarchy"
```

---

### Task 4: Wire `interaction_graph` into `HierarchicalMMM`

**Files:**
- Modify: `calmmm/model/mmm.py`
- Test: `tests/model/test_mmm.py`

**Interfaces:**
- Consumes: `InteractionGraph`, `build_interaction_step` from `calmmm.model.interactions` (Tasks 1-2); `apply_interactions` kwarg on `_build_media_hierarchy` (Task 3).
- Produces: `HierarchicalMMM(..., interaction_graph: Optional[InteractionGraph] = None)`. New public attribute `self.interaction_graph`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/model/test_mmm.py`:

```python
from calmmm.model.interactions import ChannelInteraction, InteractionGraph


def test_hierarchical_mmm_default_interaction_graph_none():
    mmm = HierarchicalMMM()
    assert mmm.interaction_graph is None


def test_build_model_with_interaction_graph_adds_gamma_rv(mmmdata):
    graph = InteractionGraph(edges=[ChannelInteraction(source="social", target="search")])
    mmm = HierarchicalMMM(interaction_graph=graph)
    model = mmm.build_model(mmmdata)
    names = {v.name for v in model.free_RVs}
    assert "gamma_social_search" in names


def test_build_model_with_interaction_graph_channel_contrib_shape_unchanged(mmmdata):
    graph = InteractionGraph(edges=[ChannelInteraction(source="social", target="search")])
    mmm = HierarchicalMMM(interaction_graph=graph)
    model = mmm.build_model(mmmdata)
    val = pm.draw(model["channel_contrib"], random_seed=0)
    T_train = int(mmm._train_mask.sum())
    assert val.shape == (T_train, len(mmmdata.geos), len(mmmdata.kpis), len(mmmdata.channels))


def test_build_model_with_interaction_graph_logp_finite(mmmdata):
    graph = InteractionGraph(edges=[ChannelInteraction(source="social", target="search")])
    mmm = HierarchicalMMM(interaction_graph=graph)
    model = mmm.build_model(mmmdata)
    with model:
        ip = model.initial_point()
        lp = model.compile_logp()(ip)
    assert np.isfinite(lp)


def test_build_model_without_interaction_graph_no_gamma_rv(mmmdata):
    """Regression: default (no interaction_graph) must not introduce any gamma_* RVs."""
    mmm = HierarchicalMMM()
    model = mmm.build_model(mmmdata)
    names = {v.name for v in model.free_RVs}
    assert not any(n.startswith("gamma_") for n in names)


def test_build_model_unknown_channel_in_interaction_graph_raises(mmmdata):
    graph = InteractionGraph(edges=[ChannelInteraction(source="tv", target="search")])
    mmm = HierarchicalMMM(interaction_graph=graph)
    with pytest.raises(ValueError, match="unknown channel"):
        mmm.build_model(mmmdata)


def test_public_import_channel_interaction():
    from calmmm import ChannelInteraction
    assert ChannelInteraction is not None


def test_public_import_interaction_graph():
    from calmmm import InteractionGraph
    assert InteractionGraph is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/model/test_mmm.py -v -k "interaction_graph or ChannelInteraction"`
Expected: FAIL — `TypeError: HierarchicalMMM.__init__() got an unexpected keyword argument 'interaction_graph'` (the two `test_public_import_*` tests will still fail after Task 4's mmm.py changes alone — they're satisfied by Task 5; leave them red for now and re-run after Task 5).

- [ ] **Step 3: Wire it into `HierarchicalMMM`**

In `calmmm/model/mmm.py`, add one import alongside the existing `calmmm.model.*` imports. Find:

```python
from calmmm.model.coords import build_coords, build_arrays, build_controls_array
from calmmm.model.priors import PriorConfig
from calmmm.model.transforms import geometric_adstock_pt, hill_saturation_pt
from calmmm.model.components import _build_baseline, _build_media_hierarchy, _add_likelihood
```

Replace with:

```python
from calmmm.model.coords import build_coords, build_arrays, build_controls_array
from calmmm.model.priors import PriorConfig
from calmmm.model.transforms import geometric_adstock_pt, hill_saturation_pt
from calmmm.model.components import _build_baseline, _build_media_hierarchy, _add_likelihood
from calmmm.model.interactions import InteractionGraph, build_interaction_step
```

Find the `__init__` signature and body:

```python
    def __init__(
        self,
        *,
        priors: Optional[PriorConfig] = None,
        n_fourier_pairs: int = 2,
        holdout_fraction: float = 0.2,
    ) -> None:
        self.priors = priors or PriorConfig()
        self.n_fourier_pairs = n_fourier_pairs
        self.holdout_fraction = holdout_fraction
```

Replace with:

```python
    def __init__(
        self,
        *,
        priors: Optional[PriorConfig] = None,
        n_fourier_pairs: int = 2,
        holdout_fraction: float = 0.2,
        interaction_graph: Optional[InteractionGraph] = None,
    ) -> None:
        self.priors = priors or PriorConfig()
        self.n_fourier_pairs = n_fourier_pairs
        self.holdout_fraction = holdout_fraction
        self.interaction_graph = interaction_graph
```

Find where `baseline` and `media_contrib` are built inside `build_model`'s `with pm.Model(coords=coords) as model:` block:

```python
            # Baseline
            baseline = _build_baseline(fourier_train, obs_mean_log, self.priors, ctrl_train)

            # Media hierarchy
            media_contrib = _build_media_hierarchy(X_sat, self.priors)
```

Replace with:

```python
            # Baseline
            baseline = _build_baseline(fourier_train, obs_mean_log, self.priors, ctrl_train)

            # Media hierarchy, with optional channel-to-channel interactions
            apply_interactions = None
            if self.interaction_graph is not None:
                apply_interactions = build_interaction_step(
                    self.interaction_graph,
                    channels=data.channels,
                    X_adstock=X_adstocked,
                )
            media_contrib = _build_media_hierarchy(X_sat, self.priors, apply_interactions=apply_interactions)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/model/test_mmm.py -v -k "interaction_graph or gamma"`
Expected: all pass except the two `test_public_import_*` tests (still red — fixed in Task 5)

- [ ] **Step 5: Commit**

```bash
git add calmmm/model/mmm.py tests/model/test_mmm.py
git commit -m "feat: wire interaction_graph into HierarchicalMMM"
```

---

### Task 5: Public export from `calmmm/__init__.py`

**Files:**
- Modify: `calmmm/__init__.py`
- Test: `tests/model/test_mmm.py` (the two tests added in Task 4, currently red)

**Interfaces:**
- Consumes: `ChannelInteraction`, `InteractionGraph` from `calmmm.model.interactions`.
- Produces: `from calmmm import ChannelInteraction, InteractionGraph` works.

- [ ] **Step 1: Confirm the tests are currently failing**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/model/test_mmm.py -v -k test_public_import_channel_interaction_or_test_public_import_interaction_graph`

(pytest `-k` doesn't support `_or_`; instead run both by name:)

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/model/test_mmm.py::test_public_import_channel_interaction tests/model/test_mmm.py::test_public_import_interaction_graph -v`
Expected: FAIL — `AttributeError: module 'calmmm' has no attribute 'ChannelInteraction'`

- [ ] **Step 2: Add the export**

In `calmmm/__init__.py`, find:

```python
__all__ = [
    "MMMData", "IncrementalityTests",
    "HierarchicalMMM", "MMMFit",
    "CalibrationTarget",
    "channel_contributions", "marginal_contributions", "compute_roi", "saturation_curve",
]
```

Replace with:

```python
__all__ = [
    "MMMData", "IncrementalityTests",
    "HierarchicalMMM", "MMMFit",
    "CalibrationTarget",
    "ChannelInteraction", "InteractionGraph",
    "channel_contributions", "marginal_contributions", "compute_roi", "saturation_curve",
]
```

Find:

```python
    if name == "CalibrationTarget":
        from calmmm.calibration.targets import CalibrationTarget
        globals()["CalibrationTarget"] = CalibrationTarget
        return CalibrationTarget
```

Replace with:

```python
    if name == "CalibrationTarget":
        from calmmm.calibration.targets import CalibrationTarget
        globals()["CalibrationTarget"] = CalibrationTarget
        return CalibrationTarget
    if name in ("ChannelInteraction", "InteractionGraph"):
        from calmmm.model.interactions import ChannelInteraction, InteractionGraph
        globals()["ChannelInteraction"] = ChannelInteraction
        globals()["InteractionGraph"] = InteractionGraph
        return globals()[name]
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/model/test_mmm.py -v -k "public_import"`
Expected: all `test_public_import_*` tests pass (including the pre-existing `test_public_import_calibration_target`)

- [ ] **Step 4: Run the full fast test suite**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest -q`
Expected: all previously-passing tests still pass, plus every new test from Tasks 1-5 (slow tests deselected as usual)

- [ ] **Step 5: Commit**

```bash
git add calmmm/__init__.py
git commit -m "feat: export ChannelInteraction/InteractionGraph from calmmm package"
```

---

### Task 6: Downstream compatibility — attribution and calibration co-exist with an active interaction graph

**Files:**
- Test: `tests/model/test_interactions_integration.py` (new file — kept separate from `tests/model/test_interactions.py` because every test here is `@pytest.mark.slow`, so a contributor running fast tests only never touches PyMC inference, and a contributor running `-m slow` doesn't have to wade through the 16 fast unit tests from Tasks 1-2 to find the 1-2 slow ones)

**Interfaces:**
- Consumes: `HierarchicalMMM`, `InteractionGraph`, `ChannelInteraction` (public API); `channel_contributions`, `compute_roi`, `saturation_curve` (public API); `mmmdata`, `lift_tests` fixtures from `tests/conftest.py`.
- Produces: nothing new — this task is pure verification that Tasks 1-5 didn't require touching `calmmm/attribution/` or `calmmm/calibration/`.

This task fits one real `mode="map"` fit with both an interaction graph and calibration experiments active, then exercises every downstream consumer named in the spec's "Downstream Compatibility" section against that single fit — one fit, many assertions, to keep total wall-clock time down (a single MAP fit on the 52-week/2-channel fixture takes a few minutes; running it once and asserting broadly is much cheaper than one fit per assertion).

- [ ] **Step 1: Write the failing test**

Create `tests/model/test_interactions_integration.py`:

```python
import numpy as np
import pytest

from calmmm import channel_contributions, compute_roi, saturation_curve
from calmmm.calibration.lift import compute_model_lift
from calmmm.model.interactions import ChannelInteraction, InteractionGraph
from calmmm.model.mmm import HierarchicalMMM


@pytest.mark.slow
def test_downstream_consumers_work_unmodified_with_active_interaction_graph(mmmdata, lift_tests):
    """
    Fit once with both an interaction graph (social -> search) and calibration
    experiments active, then confirm every downstream consumer named in
    docs/superpowers/specs/2026-07-02-channel-interaction-graph-design.md's
    "Downstream Compatibility" section runs unmodified and returns sane output.
    """
    graph = InteractionGraph(edges=[ChannelInteraction(source="social", target="search")])
    mmm = HierarchicalMMM(holdout_fraction=0.2, interaction_graph=graph)
    fit = mmm.fit(mmmdata, experiments=lift_tests, mode="map")

    # channel_contrib still [T, G, K, C], summing to exp(mu) - baseline as always.
    cc_val = np.array(fit.map_params["channel_contrib"])
    T_train = int(mmm._train_mask.sum())
    assert cc_val.shape == (T_train, len(mmmdata.geos), len(mmmdata.kpis), len(mmmdata.channels))

    # channel_contributions(): unmodified module, must still run and include every channel.
    contrib_df = channel_contributions(fit)
    assert set(contrib_df["channel"].unique()) == set(mmmdata.channels) | {"baseline"}

    # compute_roi(): unmodified module, must still run and return finite ROI values.
    roi_df = compute_roi(fit)
    assert np.isfinite(roi_df["roi"]).all()

    # saturation_curve(): unmodified module — must still work for the target channel
    # ("search"), proving its Hill curve stays a pure function of its own spend
    # even though search is the target of an active interaction edge.
    curve_df = saturation_curve(fit, "search")
    assert curve_df["saturation"].between(0.0, 1.0).all()

    # compute_model_lift() / add_calibration_likelihood(): calibration ran during
    # fit() (experiments=lift_tests was passed) without error; verify the lift
    # comparison table it produced is well-formed and finite.
    lift_df = compute_model_lift(fit, fit.calibration_targets)
    assert len(lift_df) == 1
    assert np.isfinite(lift_df["lift_model"]).all()
    assert np.isfinite(lift_df["z_score"]).all()
```

- [ ] **Step 2: Run test to verify it fails for the right reason**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/model/test_interactions_integration.py -m slow -v`
Expected: at this point in the plan Tasks 1-5 are already done, so this should actually PASS. If it fails, the failure must NOT be an import error or missing-attribute error (those would indicate Tasks 1-5 are incomplete) — investigate any other failure before proceeding.

- [ ] **Step 3: No implementation step — this task is verification only**

If Step 2 passed, there is nothing to implement. If it failed for a reason other than Tasks 1-5 being incomplete, fix the root cause (likely a shape or naming mismatch between Task 2's `build_interaction_step` and what `channel_contributions`/`compute_roi`/`saturation_curve`/`compute_model_lift` expect) and re-run.

- [ ] **Step 4: Run test to confirm it passes**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/model/test_interactions_integration.py -m slow -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add tests/model/test_interactions_integration.py
git commit -m "test: verify attribution/calibration work unmodified with an active interaction graph"
```

---

### Task 7: Wire the `direct_mail -> search` edge into the demo script

**Files:**
- Modify: `scripts/run_demo_fit.py`
- Test: `tests/test_run_demo_fit.py`

**Interfaces:**
- Consumes: `HierarchicalMMM`, `ChannelInteraction`, `InteractionGraph` (public API).
- Produces: `build_model_instance(args: argparse.Namespace) -> HierarchicalMMM` — new helper, mirrors the existing `build_data(panel) -> MMMData` / `build_experiments(lift_tests, data) -> IncrementalityTests` helpers in this file. New CLI flag `--no-direct-mail-search-interaction` (default: interaction enabled).

- [ ] **Step 1: Write the failing tests**

In `tests/test_run_demo_fit.py`, find `test_default_parser_is_demo_first`:

```python
def test_default_parser_is_demo_first():
    script = _load_script()

    args = script.parse_args([])

    assert args.mode == "map"
    assert args.weeks == 0
    assert args.maxeval == 2000
    assert args.holdout_fraction == 0.2
    assert args.adjust_lift_windows is False
    assert args.reporting_dir == Path("reporting")
    assert args.spend_multiplier == 1.10
```

Replace with (one new assertion added):

```python
def test_default_parser_is_demo_first():
    script = _load_script()

    args = script.parse_args([])

    assert args.mode == "map"
    assert args.weeks == 0
    assert args.maxeval == 2000
    assert args.holdout_fraction == 0.2
    assert args.adjust_lift_windows is False
    assert args.direct_mail_search_interaction is True
    assert args.reporting_dir == Path("reporting")
    assert args.spend_multiplier == 1.10
```

Then add two new test functions to the same file (place them near the other `parse_args`/model-construction tests):

```python
def test_build_model_instance_default_has_direct_mail_search_interaction():
    script = _load_script()
    args = script.parse_args([])
    mmm = script.build_model_instance(args)
    assert mmm.interaction_graph is not None
    assert len(mmm.interaction_graph.edges) == 1
    edge = mmm.interaction_graph.edges[0]
    assert edge.source == "direct_mail"
    assert edge.target == "search"


def test_build_model_instance_no_interaction_flag_disables():
    script = _load_script()
    args = script.parse_args(["--no-direct-mail-search-interaction"])
    mmm = script.build_model_instance(args)
    assert mmm.interaction_graph is None


def test_build_model_instance_passes_holdout_fraction():
    script = _load_script()
    args = script.parse_args(["--holdout-fraction", "0.3"])
    mmm = script.build_model_instance(args)
    assert mmm.holdout_fraction == 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/test_run_demo_fit.py -v -k "direct_mail_search or build_model_instance"`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'direct_mail_search_interaction'` and `AttributeError: module 'run_demo_fit' has no attribute 'build_model_instance'`

- [ ] **Step 3: Implement the flag and helper**

In `scripts/run_demo_fit.py`, add one import near the top. Find:

```python
from calmmm import MMMData, IncrementalityTests, HierarchicalMMM
from calmmm import channel_contributions, compute_roi, saturation_curve
from calmmm.attribution.curves import spend_response_report
from calmmm.calibration.lift import compute_model_lift
```

Replace with:

```python
from calmmm import MMMData, IncrementalityTests, HierarchicalMMM
from calmmm import channel_contributions, compute_roi, saturation_curve
from calmmm.attribution.curves import spend_response_report
from calmmm.calibration.lift import compute_model_lift
from calmmm.model.interactions import ChannelInteraction, InteractionGraph
```

In `parse_args`, find:

```python
    parser.add_argument(
        "--adjust-lift-windows",
        dest="adjust_lift_windows",
        action="store_true",
        help="Move sample lift-test windows into the training period (for use with --weeks subsets whose range excludes the tests' real dates).",
    )
    parser.add_argument(
        "--no-adjust-lift-windows",
        dest="adjust_lift_windows",
        action="store_false",
        help="Use the lift tests' real dates (default; requires the training window to cover them).",
    )
    parser.set_defaults(adjust_lift_windows=False)
    return parser.parse_args(argv)
```

Replace with:

```python
    parser.add_argument(
        "--adjust-lift-windows",
        dest="adjust_lift_windows",
        action="store_true",
        help="Move sample lift-test windows into the training period (for use with --weeks subsets whose range excludes the tests' real dates).",
    )
    parser.add_argument(
        "--no-adjust-lift-windows",
        dest="adjust_lift_windows",
        action="store_false",
        help="Use the lift tests' real dates (default; requires the training window to cover them).",
    )
    parser.set_defaults(adjust_lift_windows=False)
    parser.add_argument(
        "--no-direct-mail-search-interaction",
        dest="direct_mail_search_interaction",
        action="store_false",
        help="Disable the direct_mail -> search channel interaction edge (enabled by default).",
    )
    parser.set_defaults(direct_mail_search_interaction=True)
    return parser.parse_args(argv)
```

Add a new `build_model_instance` function. Find `def build_experiments(lift_tests: pd.DataFrame, data: MMMData) -> IncrementalityTests:` and its closing `)` before `def fit_kwargs`. Immediately after that function (before `def fit_kwargs`), insert:

```python
def build_model_instance(args: argparse.Namespace) -> HierarchicalMMM:
    interaction_graph = None
    if args.direct_mail_search_interaction:
        interaction_graph = InteractionGraph(edges=[
            ChannelInteraction(source="direct_mail", target="search"),
        ])
    return HierarchicalMMM(
        holdout_fraction=args.holdout_fraction,
        interaction_graph=interaction_graph,
    )


```

In `main()`, find:

```python
    data = build_data(panel)
    experiments = build_experiments(lift_tests, data)
    model = HierarchicalMMM(holdout_fraction=args.holdout_fraction)
    fit = model.fit(data, experiments=experiments, mode=args.mode, **fit_kwargs(args))
```

Replace with:

```python
    data = build_data(panel)
    experiments = build_experiments(lift_tests, data)
    model = build_model_instance(args)
    fit = model.fit(data, experiments=experiments, mode=args.mode, **fit_kwargs(args))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest tests/test_run_demo_fit.py -v`
Expected: all tests in the file pass

- [ ] **Step 5: Run the full fast test suite**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest -q`
Expected: all fast tests pass (slow tests deselected)

- [ ] **Step 6: Commit**

```bash
git add scripts/run_demo_fit.py tests/test_run_demo_fit.py
git commit -m "feat: add direct_mail -> search interaction edge to the demo fit script"
```

---

### Task 8: Run the real fit, validate the hypothesis, regenerate artifacts

**Files:**
- Modify (regenerated, not hand-edited): `artifacts/demo_fit/calibration_fit.csv`, `artifacts/demo_fit/channel_contributions_sample.csv`, `artifacts/demo_fit/fit_quality.csv`, `artifacts/demo_fit/fit_summary.json`, `artifacts/demo_fit/roi.csv`, `reporting/calibration_fit.svg`, `reporting/roi.svg`, `reporting/saturation_curves.csv`, `reporting/saturation_curves.svg`, `reporting/spend_response.csv`, `reporting/spend_response.svg`

This task is not a code change — Tasks 1-7 are complete and merged into this branch by this point. This task runs the real full demo fit (the `direct_mail -> search` edge is now on by default) and records what the spec's "Validation After Implementing" section asked for, honestly, before this becomes the new committed baseline.

- [ ] **Step 1: Run the full demo fit**

Run:
```bash
source .venv/bin/activate
export PYTENSOR_FLAGS='cxx='
python scripts/run_demo_fit.py
```
Expected: completes in several minutes (full 78-week panel, `maxeval=2000`, matching the runtime of prior full-panel runs in this project's history). Prints a "Fit summary" block, "Calibration fit" table, and "Fit quality" table to stdout.

- [ ] **Step 2: Record `gamma_direct_mail_search`'s MAP estimate**

Run:
```bash
python -c "
import json
summary = json.load(open('artifacts/demo_fit/fit_summary.json'))
print('map_param_count:', summary['map_param_count'])
"
python -c "
import numpy as np
import sys
sys.path.insert(0, '.')
# fit_summary.json doesn't include raw MAP params; re-derive from a fresh MAP call
# is expensive, so instead confirm the gamma parameter exists by checking model
# construction directly against the same panel/lift-tests inputs run_demo_fit.py used.
"
```

The `fit_summary.json`/CSV outputs do not currently serialize individual MAP parameter values other than `mu`/`channel_contrib` (see `calmmm/model/fit.py`'s `to_netcdf`/`fit_metrics`). To inspect `gamma_direct_mail_search` specifically without adding new serialization surface (out of scope for this plan), re-run the fit interactively:

```bash
python -c "
import os
os.environ.setdefault('PYTENSOR_FLAGS', 'cxx=')
import sys
sys.path.insert(0, 'scripts')
import run_demo_fit as script
import pandas as pd

args = script.parse_args([])
panel = pd.read_csv(args.panel)
panel = script.select_week_subset(panel, args.weeks)
lift_tests = pd.read_csv(args.lift_tests)
data = script.build_data(panel)
experiments = script.build_experiments(lift_tests, data)
model = script.build_model_instance(args)
fit = model.fit(data, experiments=experiments, mode=args.mode, **script.fit_kwargs(args))
print('gamma_direct_mail_search MAP estimate:', float(fit.map_params['gamma_direct_mail_search']))
"
```
Expected: prints a single float. Record this value.

- [ ] **Step 3: Compare calibration and fit quality against the pre-interaction baseline**

Run:
```bash
cat artifacts/demo_fit/calibration_fit.csv
cat artifacts/demo_fit/fit_quality.csv
```

Compare `direct_mail_match_q3`'s `z_score` against its pre-interaction value of **-3.02**, and `r2_applications`/`r2_funded_revenue` against their pre-interaction values of **0.9597** / **0.9728** (from `git show HEAD~8:artifacts/demo_fit/fit_quality.csv` if the working tree has already been overwritten by Step 1 — `HEAD~8` assumes Tasks 1-7 each produced exactly one commit since the branch point; adjust the ref if that assumption doesn't hold, e.g. by checking `git log --oneline` for the last commit before Task 1).

Record, in the commit message for this task, all four numbers (before/after z-score, before/after R²) and the `gamma_direct_mail_search` MAP estimate from Step 2 — report honestly per the spec: if `gamma` is near 0 with the z-score essentially unchanged, that is a valid, reportable outcome (weak identification, as anticipated), not a failure to fix before committing.

- [ ] **Step 4: Regenerate reporting outputs**

Run:
```bash
python -m calmmm.reporting.visualization
```
Expected: "Rendered reporting outputs" listing 5 files.

- [ ] **Step 5: Run the full test suite**

Run: `source .venv/bin/activate && PYTENSOR_FLAGS=cxx= python -m pytest -q`
Expected: all fast tests pass, same pass count as the end of Task 7 (this step should not add or remove any test — it's a regenerated-artifacts sanity check, not a code change)

- [ ] **Step 6: Commit the regenerated artifacts**

```bash
git add artifacts/demo_fit/ reporting/
git commit -m "chore: regenerate demo fit artifacts with direct_mail -> search interaction active

gamma_direct_mail_search MAP estimate: <value from Step 2>
direct_mail_match_q3 z-score: -3.02 -> <new value>
r2_applications: 0.9597 -> <new value>
r2_funded_revenue: 0.9728 -> <new value>"
```

(Fill in the four `<...>` placeholders with the actual recorded numbers before committing — this is the one place in this plan where a value is genuinely not knowable until the fit runs.)

---

## Self-Review

**Spec coverage:**
- "A reusable `ChannelInteraction`/`InteractionGraph` data model" → Task 1.
- "Support multiple edges, multi-hop chains, and per-edge sign constraints" → Task 1 (`incoming_edges`, `topological_order` support multiple edges/chains), Task 2 (`build_interaction_step` implements both `half_normal` and `normal` priors, and the chain-normalization path, each with a dedicated test).
- "Zero behavior change when no interaction graph is supplied" → Task 3 (`test_media_hierarchy_apply_interactions_default_none_unchanged`), Task 4 (`test_build_model_without_interaction_graph_no_gamma_rv`).
- "Zero changes required in downstream consumers" → Task 6, which touches no file under `calmmm/attribution/` or `calmmm/calibration/` and proves they work unmodified.
- "Add the `direct_mail` -> `search` edge" → Task 7 (wiring), Task 8 (running it for real and recording the result).
- Multiplicative-post-saturation mechanism (not additive-to-exposure) → Task 2's implementation applies the boost to `channel_contrib` slices, never to `X_sat` or any pre-saturation tensor.
- Cycle/self-loop/duplicate/unknown-channel validation → Task 1 (structural) + Task 2/`build_interaction_step` (`validate_channels` called against real channel names).
- "Validation After Implementing" section (gamma away from 0? z-score move? R² stay flat/improve?) → Task 8.

**Placeholder scan:** Task 8 Steps 2-3 contain a `<value from Step 2>`-style placeholder in the final commit message — this is intentional and unavoidable (a real fit result that doesn't exist until Task 8 Step 1 runs), not a spec-coverage gap; flagged explicitly in the step text rather than silently left vague.

**Type consistency:** `ChannelInteraction`/`InteractionGraph` (Task 1) → consumed identically in Tasks 2, 4, 6, 7 (`source`, `target`, `prior`, `prior_sigma` field names match everywhere). `build_interaction_step(graph, *, channels, X_adstock)` (Task 2) signature matches its one call site in Task 4. `apply_interactions` parameter name matches between Task 3's `_build_media_hierarchy` and Task 4's call site. `build_model_instance(args) -> HierarchicalMMM` (Task 7) matches its use in Task 7's `main()` edit and Task 8's Step 2 validation script.
