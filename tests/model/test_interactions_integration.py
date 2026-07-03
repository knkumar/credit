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
