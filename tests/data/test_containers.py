import pytest
import numpy as np
import pandas as pd
from calmmm.data.schema import (
    ObservationRow,
    MediaRow,
    ControlRow,
    KPIMetadata,
    ExperimentRow,
    KPILikelihood,
    CalibrationLikelihood,
    Estimand,
)
from calmmm.data.containers import MMMData, IncrementalityTests


def test_observation_row_fields():
    row = ObservationRow(
        time=pd.Timestamp("2024-01-01"),
        geo="DMA_1",
        kpi="visits",
        outcome=5000.0,
        population=1_000_000.0,
    )
    assert row.kpi == "visits"
    assert row.population == 1_000_000.0


def test_kpi_metadata_defaults():
    meta = KPIMetadata(kpi="visits")
    assert meta.likelihood == KPILikelihood.NEGATIVE_BINOMIAL
    assert meta.funnel_stage is None


def test_experiment_row_requires_se_or_ci():
    with pytest.raises(ValueError, match="se or ci_lower/ci_upper"):
        ExperimentRow(
            test_id="t1",
            channel_bundle=["search"],
            kpi="visits",
            geo_scope=["DMA_1"],
            start_date=pd.Timestamp("2024-03-01"),
            end_date=pd.Timestamp("2024-03-28"),
            lift=12_000.0,
        )


def test_experiment_row_ci_converted_to_se():
    row = ExperimentRow(
        test_id="t1",
        channel_bundle=["search"],
        kpi="visits",
        geo_scope=["DMA_1"],
        start_date=pd.Timestamp("2024-03-01"),
        end_date=pd.Timestamp("2024-03-28"),
        lift=12_000.0,
        ci_lower=7_100.0,
        ci_upper=16_900.0,
    )
    # se = (16900 - 7100) / (2 * 1.96) ≈ 2500
    assert abs(row.se - 2_500.0) < 1.0


def test_experiment_row_se_provided_directly():
    row = ExperimentRow(
        test_id="t2",
        channel_bundle=["social"],
        kpi="applications",
        geo_scope=["DMA_1", "DMA_2"],
        start_date=pd.Timestamp("2024-04-01"),
        end_date=pd.Timestamp("2024-04-28"),
        lift=500.0,
        se=100.0,
    )
    assert row.se == 100.0


def test_kpi_likelihood_enum_values():
    assert KPILikelihood.GAUSSIAN == "gaussian"
    assert KPILikelihood.NEGATIVE_BINOMIAL == "negative_binomial"
    assert KPILikelihood.BINOMIAL == "binomial"
    assert KPILikelihood.LOGNORMAL == "lognormal"


def test_calibration_likelihood_enum_values():
    assert CalibrationLikelihood.NORMAL == "normal"
    assert CalibrationLikelihood.STUDENT_T == "student_t"


def test_estimand_enum_values():
    assert Estimand.TOTAL == "total"
    assert Estimand.IMMEDIATE == "immediate"
    assert Estimand.CARRYOVER == "carryover"
    assert Estimand.CUMULATIVE == "cumulative"


def test_experiment_row_inverted_ci_raises():
    with pytest.raises(ValueError, match="ci_upper.*>=.*ci_lower"):
        ExperimentRow(
            test_id="t3",
            channel_bundle=["search"],
            kpi="visits",
            geo_scope=["DMA_1"],
            start_date=pd.Timestamp("2024-03-01"),
            end_date=pd.Timestamp("2024-03-28"),
            lift=12_000.0,
            ci_lower=16_900.0,
            ci_upper=7_100.0,  # inverted: upper < lower
        )


def test_experiment_row_custom_ci_level():
    from scipy.stats import norm
    row_90 = ExperimentRow(
        test_id="t_ci90",
        channel_bundle=["search"],
        kpi="visits",
        geo_scope=["DMA_1"],
        start_date=pd.Timestamp("2024-03-01"),
        end_date=pd.Timestamp("2024-03-28"),
        lift=12_000.0,
        ci_lower=0.8,
        ci_upper=1.2,
        ci_level=0.90,
    )
    row_95 = ExperimentRow(
        test_id="t_ci95",
        channel_bundle=["search"],
        kpi="visits",
        geo_scope=["DMA_1"],
        start_date=pd.Timestamp("2024-03-01"),
        end_date=pd.Timestamp("2024-03-28"),
        lift=12_000.0,
        ci_lower=0.8,
        ci_upper=1.2,
        ci_level=0.95,
    )
    z90 = norm.ppf(0.95)
    z95 = norm.ppf(0.975)
    expected_se_90 = (1.2 - 0.8) / (2 * z90)
    expected_se_95 = (1.2 - 0.8) / (2 * z95)
    assert abs(row_90.se - expected_se_90) < 1e-10
    assert abs(row_95.se - expected_se_95) < 1e-10


def test_experiment_row_zero_se_raises():
    with pytest.raises(ValueError, match="se must be > 0"):
        ExperimentRow(
            test_id="t4",
            channel_bundle=["search"],
            kpi="visits",
            geo_scope=["DMA_1"],
            start_date=pd.Timestamp("2024-03-01"),
            end_date=pd.Timestamp("2024-03-28"),
            lift=12_000.0,
            se=0.0,
        )


def test_mmmdata_happy_path(synthetic_panel):
    dataset = MMMData.from_dataframe(
        synthetic_panel,
        time="week",
        geo="dma",
        kpis=["visits", "applications", "approvals", "revenue"],
        media=["search", "social"],
        spend=["search_spend", "social_spend"],
        exposure=["search_impressions", "social_impressions"],
        controls=["price_index"],
        population="population",
    )
    assert dataset.n_geos == 2
    assert dataset.n_kpis == 4
    assert dataset.n_channels == 2
    assert dataset.n_times == 52
    assert set(dataset.channels) == {"search", "social"}
    assert set(dataset.kpis) == {"visits", "applications", "approvals", "revenue"}


def test_mmmdata_observations_shape(synthetic_panel):
    dataset = MMMData.from_dataframe(
        synthetic_panel,
        time="week",
        geo="dma",
        kpis=["visits", "applications"],
        media=["search"],
        spend=["search_spend"],
    )
    assert len(dataset.observations) == 52 * 2 * 2


def test_mmmdata_media_shape(synthetic_panel):
    dataset = MMMData.from_dataframe(
        synthetic_panel,
        time="week",
        geo="dma",
        kpis=["visits"],
        media=["search", "social"],
        spend=["search_spend", "social_spend"],
    )
    assert len(dataset.media) == 52 * 2 * 2


def test_mmmdata_requires_matching_spend_and_media(synthetic_panel):
    with pytest.raises(ValueError, match="media and spend must have the same length"):
        MMMData.from_dataframe(
            synthetic_panel,
            time="week",
            geo="dma",
            kpis=["visits"],
            media=["search", "social"],
            spend=["search_spend"],
        )


def test_mmmdata_date_range(synthetic_panel):
    dataset = MMMData.from_dataframe(
        synthetic_panel,
        time="week",
        geo="dma",
        kpis=["visits"],
        media=["search"],
        spend=["search_spend"],
    )
    assert dataset.start_date == pd.Timestamp("2024-01-01")
    assert dataset.n_times == 52


def test_incrementality_tests_happy_path(synthetic_panel, synthetic_lift_df):
    dataset = MMMData.from_dataframe(
        synthetic_panel,
        time="week",
        geo="dma",
        kpis=["visits", "applications"],
        media=["search", "social"],
        spend=["search_spend", "social_spend"],
    )
    experiments = IncrementalityTests.from_dataframe(
        synthetic_lift_df,
        channel="channel",
        kpi="kpi",
        geo_scope="geo_scope",
        start="start_date",
        end="end_date",
        lift="incremental_outcome",
        standard_error="se",
        mmmdata=dataset,
    )
    assert len(experiments) == 1
    assert experiments[0].test_id == "search_holdout_q1"
    assert experiments[0].se == 2_500.0


def test_incrementality_tests_unknown_channel_raises(synthetic_panel, synthetic_lift_df):
    dataset = MMMData.from_dataframe(
        synthetic_panel,
        time="week",
        geo="dma",
        kpis=["visits"],
        media=["social"],
        spend=["social_spend"],
    )
    with pytest.raises(ValueError, match="unknown channel.*search"):
        IncrementalityTests.from_dataframe(
            synthetic_lift_df,
            channel="channel",
            kpi="kpi",
            geo_scope="geo_scope",
            start="start_date",
            end="end_date",
            lift="incremental_outcome",
            standard_error="se",
            mmmdata=dataset,
        )


def test_incrementality_tests_unknown_kpi_raises(synthetic_panel, synthetic_lift_df):
    dataset = MMMData.from_dataframe(
        synthetic_panel,
        time="week",
        geo="dma",
        kpis=["applications"],
        media=["search"],
        spend=["search_spend"],
    )
    with pytest.raises(ValueError, match="unknown kpi.*visits"):
        IncrementalityTests.from_dataframe(
            synthetic_lift_df,
            channel="channel",
            kpi="kpi",
            geo_scope="geo_scope",
            start="start_date",
            end="end_date",
            lift="incremental_outcome",
            standard_error="se",
            mmmdata=dataset,
        )


def test_incrementality_tests_date_outside_panel_raises(synthetic_panel):
    dataset = MMMData.from_dataframe(
        synthetic_panel,
        time="week",
        geo="dma",
        kpis=["visits"],
        media=["search"],
        spend=["search_spend"],
    )
    bad_lift = pd.DataFrame([{
        "test_id": "future_test",
        "channel": "search",
        "kpi": "visits",
        "geo_scope": "DMA_1",
        "start_date": pd.Timestamp("2025-01-01"),
        "end_date": pd.Timestamp("2025-02-01"),
        "incremental_outcome": 5_000.0,
        "se": 1_000.0,
    }])
    with pytest.raises(ValueError, match="outside the observed date range"):
        IncrementalityTests.from_dataframe(
            bad_lift,
            channel="channel",
            kpi="kpi",
            geo_scope="geo_scope",
            start="start_date",
            end="end_date",
            lift="incremental_outcome",
            standard_error="se",
            mmmdata=dataset,
        )
