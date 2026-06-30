import numpy as np
import pandas as pd
import pytest
import warnings as _warnings
from calmmm.data.containers import MMMData
from calmmm.data.validation import validate_mmmdata, ValidationResult


def _make_dataset(df, **kwargs):
    defaults = dict(
        time="week", geo="dma", kpis=["visits"],
        media=["search"], spend=["search_spend"],
    )
    defaults.update(kwargs)
    return MMMData.from_dataframe(df, **defaults)


def test_duplicate_panel_rows_error(synthetic_panel):
    duped = pd.concat([synthetic_panel, synthetic_panel.head(1)], ignore_index=True)
    dataset = _make_dataset(duped)
    result = validate_mmmdata(dataset)
    assert result.has_errors
    assert any("duplicate" in e.lower() for e in result.errors)


def test_negative_spend_error(synthetic_panel):
    bad = synthetic_panel.copy()
    bad.loc[0, "search_spend"] = -100.0
    dataset = _make_dataset(bad)
    result = validate_mmmdata(dataset)
    assert result.has_errors
    assert any("negative spend" in e.lower() for e in result.errors)


def test_missing_outcomes_error(synthetic_panel):
    bad = synthetic_panel.copy()
    bad.loc[0, "visits"] = np.nan
    dataset = _make_dataset(bad)
    result = validate_mmmdata(dataset)
    assert result.has_errors
    assert any("missing" in e.lower() for e in result.errors)


def test_binomial_kpi_without_population_warns(synthetic_panel):
    dataset = MMMData.from_dataframe(
        synthetic_panel,
        time="week", geo="dma",
        kpis=["visits"],
        media=["search"], spend=["search_spend"],
        kpi_likelihoods={"visits": "binomial"},
    )
    result = validate_mmmdata(dataset)
    assert any("binomial" in w.lower() for w in result.warnings)


def test_weak_media_variation_warns(synthetic_panel):
    flat = synthetic_panel.copy()
    flat["search_spend"] = 1000.0
    dataset = _make_dataset(flat)
    result = validate_mmmdata(dataset)
    assert any("weak" in w.lower() or "variation" in w.lower() for w in result.warnings)


def test_clean_dataset_passes(synthetic_panel):
    dataset = _make_dataset(synthetic_panel)
    result = validate_mmmdata(dataset)
    assert not result.has_errors


def test_validation_result_raise_if_errors(synthetic_panel):
    bad = synthetic_panel.copy()
    bad.loc[0, "search_spend"] = -100.0
    dataset = _make_dataset(bad)
    result = validate_mmmdata(dataset)
    with pytest.raises(ValueError, match="validation failed"):
        result.raise_if_errors()


def test_validation_result_no_raise_when_clean(synthetic_panel):
    dataset = _make_dataset(synthetic_panel)
    result = validate_mmmdata(dataset)
    result.raise_if_errors()  # should not raise


def test_raise_if_errors_emits_warnings():
    result = ValidationResult(errors=[], warnings=["low variation in channel tv"])
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result.raise_if_errors()
    assert len(caught) == 1
    assert "low variation" in str(caught[0].message)
    assert caught[0].category is UserWarning


def test_raise_if_errors_warns_then_raises():
    result = ValidationResult(errors=["bad data"], warnings=["weak signal"])
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        with pytest.raises(ValueError):
            result.raise_if_errors()
    assert len(caught) == 1
    assert "weak signal" in str(caught[0].message)
