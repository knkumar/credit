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


def test_build_interaction_step_half_normal_never_flips_sign_from_negative_chain_source():
    """Regression test: a half_normal edge's multiplier must stay >= 1.0 even when the
    chained source's finalized contribution is negative. Before the pt.maximum(signal, 0.0)
    clamp, a large enough gamma draw could push (1.0 + gamma * signal) negative when signal
    (the chained source's boosted contribution) was negative, flipping the target's sign."""
    T, G, K, C = 4, 2, 1, 3
    contrib_val = np.ones((T, G, K, C), dtype="float64")
    contrib_val[:, :, :, 1] = -2.0  # search's own base contribution: forced negative
    contrib_val[:, :, :, 2] = -2.0  # social's own base contribution: known nonzero constant
    adstock_val = np.zeros((T, G, C), dtype="float64")
    adstock_val[:, :, 0] = 5.0  # direct_mail: large own signal, boosts search's magnitude

    graph = InteractionGraph(edges=[
        ChannelInteraction(source="direct_mail", target="search", prior_sigma=1.0),
        ChannelInteraction(source="search", target="social", prior_sigma=1.0),
    ])
    with pm.Model(coords=_coords_3ch(T, G, K)):
        apply = build_interaction_step(graph, channels=CHANNELS_3, X_adstock=pt.as_tensor_variable(adstock_val))
        boosted = apply(pt.as_tensor_variable(contrib_val))
        val = pm.draw(boosted, random_seed=1)  # reproduces a sign flip before the fix

    social_idx = 2
    base_social = -2.0
    multiplier = val[:, :, :, social_idx] / base_social
    assert np.all(multiplier >= 1.0 - 1e-9)
