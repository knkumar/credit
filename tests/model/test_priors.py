from calmmm.model.priors import PriorConfig


def test_default_priors():
    p = PriorConfig()
    assert p.adstock_decay_alpha == 3.0
    assert p.adstock_decay_beta == 3.0
    assert p.hill_alpha_sigma == 0.5
    assert p.hill_k_sigma == 1.0
    assert p.baseline_sigma == 2.0
    assert p.seasonality_sigma == 0.5
    assert p.channel_scale_global_sigma == 1.0
    assert p.channel_scale_kpi_sigma == 0.5
    assert p.channel_scale_geo_sigma == 0.25
    assert p.sigma_sigma == 0.5
    assert p.nb_alpha_sigma == 1.0


def test_custom_priors():
    p = PriorConfig(adstock_decay_alpha=5.0, hill_k_sigma=2.0)
    assert p.adstock_decay_alpha == 5.0
    assert p.hill_k_sigma == 2.0
    # other fields keep defaults
    assert p.adstock_decay_beta == 3.0
