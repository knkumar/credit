import os

import numpy as np
import pandas as pd
import pytest

from calmmm.data.containers import MMMData, IncrementalityTests

# Disable PyTensor C compilation (clang on this host passes -ld64 which is not
# a library; pure-Python mode works fine for tests).
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")


@pytest.fixture
def synthetic_panel():
    """52-week panel: 2 geos, 4 KPIs (visits/applications/approvals/revenue), 2 channels."""
    rng = np.random.default_rng(42)
    weeks = pd.date_range("2024-01-01", periods=52, freq="W-MON")
    geos = [("DMA_1", 1_000_000), ("DMA_2", 500_000)]
    rows = []
    for geo, pop in geos:
        search_spend = rng.uniform(1_000, 5_000, 52)
        social_spend = rng.uniform(500, 2_000, 52)
        for i, week in enumerate(weeks):
            visits = int(5_000 + search_spend[i] * 2 + social_spend[i] * 0.5 + rng.normal(0, 200))
            applications = int(max(0, visits * 0.05 + rng.normal(0, 20)))
            approvals = int(max(0, applications * 0.6 + rng.normal(0, 5)))
            revenue = max(0.0, approvals * 300 + rng.normal(0, 500))
            rows.append({
                "week": week,
                "dma": geo,
                "search_spend": search_spend[i],
                "social_spend": social_spend[i],
                "search_impressions": search_spend[i] * 100,
                "social_impressions": social_spend[i] * 150,
                "visits": visits,
                "applications": applications,
                "approvals": approvals,
                "revenue": revenue,
                "price_index": rng.uniform(0.9, 1.1),
                "population": pop,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_lift_df():
    """One geo-holdout experiment: Search on visits."""
    return pd.DataFrame([{
        "test_id": "search_holdout_q1",
        "channel": "search",
        "kpi": "visits",
        "geo_scope": "DMA_1",
        "start_date": pd.Timestamp("2024-03-04"),
        "end_date": pd.Timestamp("2024-03-25"),
        "incremental_outcome": 12_000.0,
        "se": 2_500.0,
    }])


@pytest.fixture
def mmmdata(synthetic_panel):
    return MMMData.from_dataframe(
        synthetic_panel,
        time="week",
        geo="dma",
        kpis=["visits", "applications", "approvals", "revenue"],
        media=["search", "social"],
        spend=["search_spend", "social_spend"],
        exposure=["search_impressions", "social_impressions"],
        controls=["price_index"],
        population="population",
        kpi_likelihoods={
            "visits": "negative_binomial",
            "applications": "negative_binomial",
            "approvals": "negative_binomial",
            "revenue": "gaussian",
        },
        funnel_stages=["visits", "applications", "approvals", "revenue"],
    )


@pytest.fixture
def lift_tests(synthetic_lift_df, mmmdata):
    return IncrementalityTests.from_dataframe(
        synthetic_lift_df,
        channel="channel",
        kpi="kpi",
        geo_scope="geo_scope",
        start="start_date",
        end="end_date",
        lift="incremental_outcome",
        standard_error="se",
        mmmdata=mmmdata,
    )
