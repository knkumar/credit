# Channel Interaction Graph Design Spec

Date: 2026-07-02
Status: Draft for review
Audience: Data science, analytics engineering

## Summary

`HierarchicalMMM`'s channel contributions are currently purely additive: each channel's contribution is `X_sat[t,g,c] * scale_geo[c,k,g]`, computed independently of every other channel, then summed. This spec adds an optional **interaction graph** that lets one channel's adstocked signal (or, for chained edges, another channel's already-boosted contribution) multiplicatively rescale a target channel's contribution — e.g., `direct_mail` priming `search` (an offline-to-online halo effect: direct mail rarely converts on the spot, but primes a later branded search).

This is a general mechanism (a directed graph of channel interactions, with chains and cycle rejection), not a one-off hack for the `direct_mail` → `search` case. That case is the first edge added once the mechanism exists.

## Background / Motivation

Two prior findings in this project motivate this:

1. Reasoning about causality (see prior conversation, not reproduced here): direct mail plausibly drives incremental branded search demand (offline prompt → later online search to verify/compare/apply), but the reverse has no real mechanism — direct mail campaigns run off pre-built target lists with weeks of production lead time, not live triggers off search behavior.
2. The `direct_mail_match_q3` calibration test currently sits at z=-3.02 (`artifacts/demo_fit/calibration_fit.csv`) — the unconstrained model attributes more `funded_revenue` credit to `search` than the experiment says `direct_mail` should get. Because contributions are purely additive today, if a real direct-mail-primes-search effect exists, the model has no way to express it except by misattributing credit to `search`. This spec is also a test of that hypothesis.

## Goals

- A reusable `ChannelInteraction` / `InteractionGraph` data model for declaring channel-to-channel effects.
- Support multiple edges, multi-hop chains, and per-edge sign constraints (boost-only vs. two-sided).
- Zero behavior change when no interaction graph is supplied (default `None`).
- Zero changes required in downstream consumers (`channel_contributions()`, `compute_roi()`, `add_calibration_likelihood()`, `saturation_curve()`, `spend_response_report()`).
- Add the `direct_mail` → `search` edge as the first real use of the mechanism.

## Non-Goals

- Additive-to-exposure interactions (would entangle a target channel's Hill saturation curve with its source's level, breaking `saturation_curve()`'s one-channel-one-curve assumption; multiplicative-post-saturation avoids this entirely).
- Automatic interaction discovery / structure learning. Edges are declared explicitly by the caller.
- Requiring MCMC/VI for models with an interaction graph. MAP remains the default fitting mode; posterior-width caveats are a documentation concern, not an API restriction.
- Modeling interactions between more than one KPI axis differently — an edge's boost applies uniformly across all KPIs a target channel affects (direct mail plausibly boosts search *demand*, which flows through to whichever KPI search drives; there's no KPI-specific mechanism hypothesized here).

## Data Model

New file `calmmm/model/interactions.py`:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ChannelInteraction:
    source: str
    target: str
    prior: Literal["half_normal", "normal"] = "half_normal"  # half_normal = boost-only
    prior_sigma: float = 0.5

@dataclass
class InteractionGraph:
    edges: list[ChannelInteraction]

    def __post_init__(self) -> None:
        # Validate: no self-loops (source == target), no duplicate (source, target)
        # pairs, no cycles. Raise ValueError with a clear message on violation.
        ...

    def topological_order(self) -> list[str]:
        """Channel names touched by this graph, base channels (no incoming edges) first."""
        ...
```

Validation (self-loops, duplicate edges, cycles, unknown channel names against `MMMData.channels`) happens once at graph-construction / `build_model()` time via a plain Python topological sort (Kahn's algorithm) — trivial at the scale of a handful of channels, and happens before any PyTensor graph is built, so a bad graph fails fast with a normal Python exception rather than deep inside model construction.

## Wiring into `HierarchicalMMM`

```python
HierarchicalMMM(
    priors=priors,
    interaction_graph=InteractionGraph(edges=[
        ChannelInteraction(source="direct_mail", target="search", prior_sigma=0.5),
    ]),
)
```

Default `interaction_graph=None` — every existing caller and test is unaffected. This becomes a regression test in its own right: fitting with `interaction_graph=None` must produce bit-identical output to the current code path.

## Model Construction

`_build_media_hierarchy` (`calmmm/model/components.py`) is untouched — it still produces the base, unboosted `channel_contrib[t,g,k,c]` exactly as today. A new step runs after it, only when `interaction_graph` is non-empty:

1. Walk channels in topological order.
2. For each channel `c` with incoming edges, start from its base contribution slice `channel_contrib[..., c]`.
3. For each incoming edge `(source, c)`:
   - **Signal**: if `source` is a base channel (no incoming edges of its own), use its own adstocked signal, `X_adstock[..., source_idx]` — already scaled to a roughly `[0, few]` range because raw spend is divided by that channel's historical max before adstock (existing pipeline behavior). If `source` was itself boosted upstream in the topological order, use its **finalized, boosted** contribution, normalized by that channel's own training-window scale (computed once from data, the same way `media_max` is computed today) so gamma's prior scale stays comparable regardless of where in the graph an edge sits.
   - Draw `gamma_edge`: `HalfNormal(prior_sigma)` for `prior="half_normal"`, or `Normal(0, prior_sigma)` for `prior="normal"`.
   - Rescale: `half_normal` edges use `contrib[c] *= (1 + gamma_edge * signal)` (gamma and signal both non-negative — can only boost, cannot flip sign). `normal` edges use `contrib[c] *= exp(gamma_edge * signal)` (guarantees strict positivity even when gamma is negative, unlike the linear form which could cross zero and produce a nonsensical sign flip for large negative gamma × large signal).
4. `channel_contrib[..., c]` is updated to the boosted value. Because edges are processed in topological order, no channel's slice is revisited after it's finalized, so multi-hop chains (`direct_mail → search → social`) compose correctly: `social`'s edge reads `search`'s already-boosted, already-normalized contribution.

`media_contrib = channel_contrib.sum(-1)` and `mu = baseline + media_contrib` proceed exactly as today.

## Downstream Compatibility

No changes required anywhere downstream, because the shape and summation invariant (`mu = baseline + channel_contrib.sum(-1)`, `channel_contrib` shape `[T, G, K, C]`) is preserved exactly:

- `channel_contributions()`, `compute_roi()` (`calmmm/attribution/`): read `channel_contrib` generically, don't care why a coefficient is what it is.
- `add_calibration_likelihood()` / `compute_model_lift()` (`calmmm/calibration/`): slice `channel_contrib` by test-specific `(t, g, k, c)` indices — this naturally does the right thing. A geo-holdout that removes real search spend in the real world removes whatever halo-boosted portion existed too, so the model's boosted `channel_contrib[..., search]` is exactly what that experiment should be compared against. No double-counting risk.
- `saturation_curve()` (`calmmm/attribution/curves.py`): evaluates a channel's Hill curve as a pure function of that channel's own `hill_alpha`/`hill_k`, independent of the interaction step (which happens strictly after saturation). Stays valid, unmodified, for every channel regardless of graph position.

## The `direct_mail` → `search` Edge

First real use of the mechanism:

```python
ChannelInteraction(source="direct_mail", target="search", prior="half_normal", prior_sigma=0.5)
```

Boost-only (matches the one-directional halo-effect hypothesis — no plausible mechanism for search to influence direct mail's near-term effectiveness). Default `prior_sigma=0.5`, same order of magnitude as the existing `channel_scale_*_sigma` priors in `PriorConfig`.

## Testing Plan

- `InteractionGraph` validation: cycle rejection, self-loop rejection, unknown-channel rejection, duplicate-edge rejection, `topological_order()` correctness on a small multi-hop graph.
- Regression: `interaction_graph=None` produces bit-identical `MMMFit` output to the current code path (MAP params, `channel_contrib`, fit metrics).
- Sanity: with a single edge and `gamma` pinned near 0, contributions match the un-boosted case within floating-point tolerance.
- Integration: `build_model()` with the `direct_mail → search` edge active — `model["channel_contrib"]` shape unchanged, logp finite (mirrors the existing `test_add_calibration_likelihood_logp_finite` pattern).
- Downstream: `channel_contributions()`, `compute_roi()`, `add_calibration_likelihood()`, `saturation_curve()` all run unmodified against a fit built with an active interaction graph, confirming no code changes were needed there.

## Validation After Implementing

Fit the model with the `direct_mail → search` edge on the current full 78-week panel and report honestly:

- Does `gamma`'s MAP estimate land meaningfully away from 0?
- Does `direct_mail_match_q3`'s calibration z-score (currently -3.02) move toward 0, as the halo-effect hypothesis predicts?
- Do `r2_applications` / `r2_funded_revenue` stay flat or improve (not collapse)?

If `gamma` lands near 0 with wide uncertainty — plausible, since no experiment isolates the interaction itself, only the panel's spend correlations — that gets reported as a weakly-identified result, not oversold as a discovery.
