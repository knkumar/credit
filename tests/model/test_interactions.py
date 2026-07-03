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
