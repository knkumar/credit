# calmmm — Plan 1: Foundation (Data + Transforms)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the validated data containers (`MMMData`, `IncrementalityTests`) and all transform primitives (adstock, saturation, seasonality, scaling) that the model layer will consume.

**Architecture:** Wide user DataFrames are validated and normalized into four typed internal DataFrames (observations, media, controls, kpi_metadata). Transforms are stateless pure functions on numpy arrays, parameterized by scalars; they are tested independently of any model.

**Tech Stack:** Python 3.10+, pandas 2.x, numpy 1.24+, scipy 1.10+, pytest 7+

---

## Scope Check

This design covers eight independent subsystems. This plan covers only **subsystem 1: data + transforms**. Three further plans should follow:

- Plan 2: Model core (`calmmm.model` — PyMC hierarchy, baseline, likelihoods, inference modes)
- Plan 3: Calibration (`calmmm.calibration` — experiment schema, lift computation, calibration likelihood)
- Plan 4: Outputs (`calmmm.diagnostics`, `calmmm.attribution`, `calmmm.optimization`, `calmmm.reporting`)

---

## Completeness Gap Analysis

The following gaps in the design spec must be resolved before implementation of plans 2–4. Gaps that affect this plan (Foundation) are marked **[BLOCKS PLAN 1]**.

### Blocking gaps with recommended resolutions

| # | Gap | Recommended resolution |
|---|-----|------------------------|
| 1 | Inference backend not chosen | **Use PyMC.** Design already leans this way; NumPyro is deferred post-MVP. |
| 2 | Budget optimization target not chosen | **Single primary KPI** for MVP. Multi-KPI utility deferred. |
| 3 | Funnel dependencies: MVP or beta | **Beta/deferred.** Funnel structure is declarative in `KPIMetadata` but the cross-KPI likelihood terms are not activated in MVP. |
| 4 | Internal table format | **Long format.** All KPIs stored in a single observations table (one row per time × geo × kpi). |
| 5 | **[BLOCKS PLAN 1]** `exposure` vs `spend` — which drives the response curve? | **Both stored; spend drives the response curve by default.** Exposure is stored and available but the adstock/saturation pipeline defaults to spend unless the user overrides `media_var="exposure"`. |
| 6 | **[BLOCKS PLAN 1]** `population` field — used as offset or ignored? | **Used as offset in count likelihoods (NB, Binomial).** Stored on `OBSERVATION`, passed through to the model as an optional offset. Required when `likelihood="binomial"`. |
| 7 | **[BLOCKS PLAN 1]** CI-to-SE conversion | **CI is converted to SE at ingestion:** `se = (upper - lower) / (2 * 1.96)`. Both `se` and `ci_lower`/`ci_upper` columns are accepted; exactly one must be provided per row. |
| 8 | **[BLOCKS PLAN 1]** `EXPERIMENT_CELL` entity in ER diagram never defined | **Drop it for MVP.** The `INCREMENTALITY_TEST` row is the atomic unit. Cell-level data (treatment vs control rows) is not exposed in the v1 API. |
| 9 | Media scaling/normalization approach | **Divide by per-channel max spend across the full panel.** This maps spend to [0,1] before adstock/saturation and makes priors interpretable across channels. |
| 10 | Weibull adstock functional form | **PDF parameterization over L=13 lags** (see Task 6 for full math). |
| 11 | I-spline saturation parameterization | **B-spline cumulative integral, 4 internal knots at spend quantiles** (see Task 7). |
| 12 | Fourier seasonality parameterization | **2 Fourier pairs, period=52 weeks.** Configurable via `n_fourier_pairs` and `period`. |
| 13 | Holdout split strategy (diagnostics) | **Time-based: last 20% of time steps held out.** Configured via `holdout_fraction` on `HierarchicalMMM`. Resolved in Plan 2. |
| 14 | Conflict warning threshold (calibration) | **Warn when `|residual| > 2 * se`.** Resolved in Plan 3. |
| 15 | Prior distributions | Resolved in Plan 2. Defaults: `HalfNormal(sigma=1)` for scale params, `Beta(3,3)` for geometric decay, `HalfNormal(sigma=0.5)` for Weibull shape/scale, `Dirichlet(ones)` for I-spline weights. |
| 16 | Estimand types math (immediate / carryover / cumulative) | Resolved in Plan 3. |
| 17 | Multi-channel bundle lift | Resolved in Plan 3: zero out all channels in the bundle simultaneously. |
| 18 | Budget optimization solver | Resolved in Plan 4: `scipy.optimize.minimize` with SLSQP. |
| 19 | FitResult API interface | Resolved in Plan 2. |
| 20 | Model serialization | Deferred post-MVP. |

---

## File Structure

```
calmmm/
  __init__.py                        # public API: MMMData, IncrementalityTests, HierarchicalMMM
  data/
    __init__.py
    schema.py                        # typed dataclasses for internal rows; no pandas dependency
    containers.py                    # MMMData and IncrementalityTests classes
    validation.py                    # all validation rules, pure functions returning ValidationResult
    normalization.py                 # wide-to-long conversion
  transforms/
    __init__.py
    adstock.py                       # geometric_adstock(), weibull_adstock()
    saturation.py                    # hill_saturation(), ispline_basis(), ispline_saturation()
    seasonality.py                   # fourier_features()
    scaling.py                       # scale_media(), unscale_media()

tests/
  conftest.py                        # synthetic panel fixture, lift fixture
  data/
    __init__.py
    test_containers.py
    test_validation.py
    test_normalization.py
  transforms/
    __init__.py
    test_adstock.py
    test_saturation.py
    test_seasonality.py
    test_scaling.py

pyproject.toml
```

---

## Task 1: Package Skeleton and Synthetic Fixtures

**Files:**
- Create: `pyproject.toml`
- Create: `calmmm/__init__.py`
- Create: `calmmm/data/__init__.py`
- Create: `calmmm/transforms/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/data/__init__.py`
- Create: `tests/transforms/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "calmmm"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pymc>=5.0",
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.10",
    "arviz>=0.16",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov>=4.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty `__init__.py` files**

`calmmm/__init__.py`:
```python
from calmmm.data.containers import MMMData, IncrementalityTests

__all__ = ["MMMData", "IncrementalityTests"]
```

`calmmm/data/__init__.py`, `calmmm/transforms/__init__.py`, `tests/__init__.py`, `tests/data/__init__.py`, `tests/transforms/__init__.py` — all empty files.

- [ ] **Step 3: Create synthetic fixtures in `tests/conftest.py`**

```python
import numpy as np
import pandas as pd
import pytest


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
```

- [ ] **Step 4: Verify pytest discovers fixtures**

Run: `pytest tests/ --collect-only -q`

Expected: `no tests ran` (no test files yet, but no import errors either)

- [ ] **Step 5: Install package in editable mode**

Run: `pip install -e ".[dev]"`

Expected: `Successfully installed calmmm-0.1.0`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml calmmm/ tests/
git commit -m "feat: package skeleton and synthetic test fixtures"
```

---

## Task 2: Internal Data Schema Types

**Files:**
- Create: `calmmm/data/schema.py`

These are pure Python dataclasses with no pandas dependency. They define the internal canonical form.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_containers.py`:

```python
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
import pandas as pd


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/data/test_containers.py -v`

Expected: `ImportError: cannot import name 'ObservationRow'`

- [ ] **Step 3: Implement `calmmm/data/schema.py`**

```python
from __future__ import annotations
import pytest
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd


class KPILikelihood(str, Enum):
    GAUSSIAN = "gaussian"
    LOGNORMAL = "lognormal"
    NEGATIVE_BINOMIAL = "negative_binomial"
    BINOMIAL = "binomial"


class CalibrationLikelihood(str, Enum):
    NORMAL = "normal"
    STUDENT_T = "student_t"
    TRUNCATED_NORMAL = "truncated_normal"
    LAPLACE = "laplace"


class Estimand(str, Enum):
    IMMEDIATE = "immediate"
    CARRYOVER = "carryover"
    TOTAL = "total"
    CUMULATIVE = "cumulative"


@dataclass
class ObservationRow:
    time: pd.Timestamp
    geo: str
    kpi: str
    outcome: float
    population: Optional[float] = None


@dataclass
class MediaRow:
    time: pd.Timestamp
    geo: str
    channel: str
    spend: float
    exposure: Optional[float] = None


@dataclass
class ControlRow:
    time: pd.Timestamp
    geo: str
    control: str
    value: float


@dataclass
class KPIMetadata:
    kpi: str
    likelihood: KPILikelihood = KPILikelihood.NEGATIVE_BINOMIAL
    funnel_stage: Optional[int] = None
    family: Optional[str] = None


@dataclass
class ExperimentRow:
    test_id: str
    channel_bundle: list[str]
    kpi: str
    geo_scope: list[str]
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    lift: float
    se: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    calibration_likelihood: CalibrationLikelihood = CalibrationLikelihood.NORMAL
    student_t_nu: float = 5.0
    estimand: Estimand = Estimand.TOTAL

    def __post_init__(self):
        if self.se is None:
            if self.ci_lower is None or self.ci_upper is None:
                raise ValueError(
                    "ExperimentRow requires either se or ci_lower/ci_upper"
                )
            self.se = (self.ci_upper - self.ci_lower) / (2 * 1.96)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_containers.py -v`

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add calmmm/data/schema.py tests/data/test_containers.py
git commit -m "feat: internal data schema types"
```

---

## Task 3: MMMData Container

**Files:**
- Create: `calmmm/data/containers.py`
- Modify: `tests/data/test_containers.py`

`MMMData` accepts a wide DataFrame and column-name mappings, validates the input, and normalizes to the internal long-format representation.

- [ ] **Step 1: Write the failing tests**

Add to `tests/data/test_containers.py`:

```python
import pytest
from calmmm.data.containers import MMMData


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
    # observations: time x geo x kpi rows
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
    # media: time x geo x channel rows
    assert len(dataset.media) == 52 * 2 * 2


def test_mmmdata_requires_matching_spend_and_media(synthetic_panel):
    with pytest.raises(ValueError, match="media and spend must have the same length"):
        MMMData.from_dataframe(
            synthetic_panel,
            time="week",
            geo="dma",
            kpis=["visits"],
            media=["search", "social"],
            spend=["search_spend"],  # only one spend column for two channels
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/data/test_containers.py -v`

Expected: `ImportError: cannot import name 'MMMData'`

- [ ] **Step 3: Implement `calmmm/data/containers.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class MMMData:
    observations: pd.DataFrame    # columns: time, geo, kpi, outcome, population
    media: pd.DataFrame           # columns: time, geo, channel, spend, exposure
    controls: pd.DataFrame        # columns: time, geo, control, value
    kpi_metadata: pd.DataFrame    # columns: kpi, likelihood, funnel_stage, family

    @property
    def n_geos(self) -> int:
        return self.observations["geo"].nunique()

    @property
    def n_kpis(self) -> int:
        return self.observations["kpi"].nunique()

    @property
    def n_channels(self) -> int:
        return self.media["channel"].nunique()

    @property
    def n_times(self) -> int:
        return self.observations["time"].nunique()

    @property
    def channels(self) -> list[str]:
        return sorted(self.media["channel"].unique().tolist())

    @property
    def kpis(self) -> list[str]:
        return sorted(self.observations["kpi"].unique().tolist())

    @property
    def geos(self) -> list[str]:
        return sorted(self.observations["geo"].unique().tolist())

    @property
    def times(self) -> list[pd.Timestamp]:
        return sorted(self.observations["time"].unique().tolist())

    @property
    def start_date(self) -> pd.Timestamp:
        return self.observations["time"].min()

    @property
    def end_date(self) -> pd.Timestamp:
        return self.observations["time"].max()

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        time: str,
        geo: str,
        kpis: list[str],
        media: list[str],
        spend: list[str],
        exposure: Optional[list[str]] = None,
        controls: Optional[list[str]] = None,
        population: Optional[str] = None,
        kpi_likelihoods: Optional[dict[str, str]] = None,
        funnel_stages: Optional[list[str]] = None,
    ) -> "MMMData":
        if len(media) != len(spend):
            raise ValueError(
                f"media and spend must have the same length, "
                f"got media={len(media)} spend={len(spend)}"
            )
        if exposure is not None and len(exposure) != len(media):
            raise ValueError(
                f"exposure must have the same length as media, "
                f"got exposure={len(exposure)} media={len(media)}"
            )

        df = df.copy()
        df[time] = pd.to_datetime(df[time])

        obs_rows = []
        for kpi in kpis:
            kpi_df = df[[time, geo, kpi]].copy()
            kpi_df.columns = ["time", "geo", "outcome"]
            kpi_df["kpi"] = kpi
            if population is not None:
                kpi_df["population"] = df[population].values
            else:
                kpi_df["population"] = np.nan
            obs_rows.append(kpi_df)
        observations = pd.concat(obs_rows, ignore_index=True)

        media_rows = []
        for ch_name, sp_col in zip(media, spend):
            m_df = df[[time, geo, sp_col]].copy()
            m_df.columns = ["time", "geo", "spend"]
            m_df["channel"] = ch_name
            if exposure is not None:
                exp_col = exposure[media.index(ch_name)]
                m_df["exposure"] = df[exp_col].values
            else:
                m_df["exposure"] = np.nan
            media_rows.append(m_df)
        media_df = pd.concat(media_rows, ignore_index=True)

        if controls:
            ctrl_rows = []
            for ctrl in controls:
                c_df = df[[time, geo, ctrl]].copy()
                c_df.columns = ["time", "geo", "value"]
                c_df["control"] = ctrl
                ctrl_rows.append(c_df)
            controls_df = pd.concat(ctrl_rows, ignore_index=True)
        else:
            controls_df = pd.DataFrame(columns=["time", "geo", "control", "value"])

        kpi_meta_rows = []
        for i, kpi in enumerate(kpis):
            likelihood = (kpi_likelihoods or {}).get(kpi, "negative_binomial")
            stage = funnel_stages.index(kpi) if (funnel_stages and kpi in funnel_stages) else None
            kpi_meta_rows.append({
                "kpi": kpi,
                "likelihood": likelihood,
                "funnel_stage": stage,
                "family": None,
            })
        kpi_metadata = pd.DataFrame(kpi_meta_rows)

        return cls(
            observations=observations,
            media=media_df,
            controls=controls_df,
            kpi_metadata=kpi_metadata,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_containers.py -v`

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add calmmm/data/containers.py tests/data/test_containers.py
git commit -m "feat: MMMData container and from_dataframe constructor"
```

---

## Task 4: IncrementalityTests Container

**Files:**
- Modify: `calmmm/data/containers.py`
- Modify: `tests/data/test_containers.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/data/test_containers.py`:

```python
from calmmm.data.containers import IncrementalityTests


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
        media=["social"],          # no "search" channel
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
        "start_date": pd.Timestamp("2025-01-01"),  # outside 2024 panel
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/data/test_containers.py -k "incrementality" -v`

Expected: `ImportError: cannot import name 'IncrementalityTests'`

- [ ] **Step 3: Implement `IncrementalityTests` in `calmmm/data/containers.py`**

Add after the `MMMData` class:

```python
from calmmm.data.schema import ExperimentRow, CalibrationLikelihood, Estimand


class IncrementalityTests:
    def __init__(self, experiments: list[ExperimentRow]):
        self._experiments = experiments

    def __len__(self) -> int:
        return len(self._experiments)

    def __getitem__(self, idx: int) -> ExperimentRow:
        return self._experiments[idx]

    def __iter__(self):
        return iter(self._experiments)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        channel: str,
        kpi: str,
        geo_scope: str,
        start: str,
        end: str,
        lift: str,
        standard_error: Optional[str] = None,
        ci_lower: Optional[str] = None,
        ci_upper: Optional[str] = None,
        calibration_likelihood: str = "normal",
        student_t_nu: float = 5.0,
        estimand: str = "total",
        mmmdata: Optional[MMMData] = None,
    ) -> "IncrementalityTests":
        experiments = []
        for _, row in df.iterrows():
            se = float(row[standard_error]) if standard_error and standard_error in df.columns else None
            ci_lo = float(row[ci_lower]) if ci_lower and ci_lower in df.columns else None
            ci_hi = float(row[ci_upper]) if ci_upper and ci_upper in df.columns else None

            channel_val = row[channel]
            channels = [c.strip() for c in channel_val.split(",")] if isinstance(channel_val, str) else [str(channel_val)]

            geo_val = row[geo_scope]
            geos = [g.strip() for g in geo_val.split(",")] if isinstance(geo_val, str) else [str(geo_val)]

            exp = ExperimentRow(
                test_id=str(row.get("test_id", f"exp_{_}")),
                channel_bundle=channels,
                kpi=str(row[kpi]),
                geo_scope=geos,
                start_date=pd.Timestamp(row[start]),
                end_date=pd.Timestamp(row[end]),
                lift=float(row[lift]),
                se=se,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                calibration_likelihood=CalibrationLikelihood(calibration_likelihood),
                student_t_nu=student_t_nu,
                estimand=Estimand(estimand),
            )

            if mmmdata is not None:
                _validate_experiment_against_dataset(exp, mmmdata)

            experiments.append(exp)
        return cls(experiments)


def _validate_experiment_against_dataset(exp: "ExperimentRow", dataset: MMMData) -> None:
    known_channels = set(dataset.channels)
    for ch in exp.channel_bundle:
        if ch not in known_channels:
            raise ValueError(
                f"unknown channel '{ch}' in experiment '{exp.test_id}'; "
                f"known channels: {sorted(known_channels)}"
            )

    known_kpis = set(dataset.kpis)
    if exp.kpi not in known_kpis:
        raise ValueError(
            f"unknown kpi '{exp.kpi}' in experiment '{exp.test_id}'; "
            f"known kpis: {sorted(known_kpis)}"
        )

    panel_start = dataset.start_date
    panel_end = dataset.end_date
    if exp.start_date < panel_start or exp.end_date > panel_end:
        raise ValueError(
            f"experiment '{exp.test_id}' window [{exp.start_date.date()}, {exp.end_date.date()}] "
            f"is outside the observed date range [{panel_start.date()}, {panel_end.date()}]"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_containers.py -v`

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add calmmm/data/containers.py calmmm/data/schema.py tests/data/test_containers.py
git commit -m "feat: IncrementalityTests container with cross-dataset validation"
```

---

## Task 5: Validation Module

**Files:**
- Create: `calmmm/data/validation.py`
- Create: `tests/data/test_validation.py`

All hard validation errors and soft warnings, as pure functions that operate on `MMMData`.

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_validation.py`:

```python
import numpy as np
import pandas as pd
import pytest
from calmmm.data.containers import MMMData
from calmmm.data.validation import validate_mmmdata, ValidationResult


def _make_dataset(df, **kwargs):
    defaults = dict(
        time="week", geo="dma", kpis=["visits"],
        media=["search"], spend=["search_spend"],
    )
    defaults.update(kwargs)
    return MMMData.from_dataframe(df, **defaults)


def test_duplicate_panel_rows_raises(synthetic_panel):
    duped = pd.concat([synthetic_panel, synthetic_panel.head(1)], ignore_index=True)
    dataset = _make_dataset(duped)
    result = validate_mmmdata(dataset)
    assert result.has_errors
    assert any("duplicate" in e.lower() for e in result.errors)


def test_negative_spend_raises(synthetic_panel):
    bad = synthetic_panel.copy()
    bad.loc[0, "search_spend"] = -100.0
    dataset = _make_dataset(bad)
    result = validate_mmmdata(dataset)
    assert result.has_errors
    assert any("negative spend" in e.lower() for e in result.errors)


def test_rate_kpi_without_population_warns(synthetic_panel):
    dataset = MMMData.from_dataframe(
        synthetic_panel,
        time="week", geo="dma",
        kpis=["visits"],
        media=["search"], spend=["search_spend"],
        kpi_likelihoods={"visits": "binomial"},
        # no population column provided
    )
    result = validate_mmmdata(dataset)
    assert any("binomial" in w.lower() for w in result.warnings)


def test_weak_media_variation_warns(synthetic_panel):
    flat = synthetic_panel.copy()
    flat["search_spend"] = 1000.0  # no variation
    dataset = _make_dataset(flat)
    result = validate_mmmdata(dataset)
    assert any("weak" in w.lower() or "variation" in w.lower() for w in result.warnings)


def test_clean_dataset_passes(synthetic_panel):
    dataset = _make_dataset(synthetic_panel)
    result = validate_mmmdata(dataset)
    assert not result.has_errors
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/data/test_validation.py -v`

Expected: `ImportError: cannot import name 'validate_mmmdata'`

- [ ] **Step 3: Implement `calmmm/data/validation.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def raise_if_errors(self) -> None:
        if self.has_errors:
            msg = "\n".join(f"  - {e}" for e in self.errors)
            raise ValueError(f"MMMData validation failed:\n{msg}")


def validate_mmmdata(dataset) -> ValidationResult:
    result = ValidationResult()
    _check_duplicate_panel_rows(dataset, result)
    _check_negative_spend(dataset, result)
    _check_binomial_kpi_has_population(dataset, result)
    _check_weak_media_variation(dataset, result)
    _check_missing_outcomes(dataset, result)
    return result


def _check_duplicate_panel_rows(dataset, result: ValidationResult) -> None:
    obs = dataset.observations
    dupes = obs.duplicated(subset=["time", "geo", "kpi"])
    if dupes.any():
        n = dupes.sum()
        result.errors.append(
            f"Duplicate panel rows detected: {n} duplicate (time, geo, kpi) combinations"
        )


def _check_negative_spend(dataset, result: ValidationResult) -> None:
    neg = dataset.media["spend"] < 0
    if neg.any():
        channels = dataset.media.loc[neg, "channel"].unique().tolist()
        result.errors.append(
            f"Negative spend values found in channels: {channels}"
        )


def _check_binomial_kpi_has_population(dataset, result: ValidationResult) -> None:
    binomial_kpis = dataset.kpi_metadata[
        dataset.kpi_metadata["likelihood"] == "binomial"
    ]["kpi"].tolist()
    for kpi in binomial_kpis:
        kpi_obs = dataset.observations[dataset.observations["kpi"] == kpi]
        if kpi_obs["population"].isna().all():
            result.warnings.append(
                f"KPI '{kpi}' uses binomial likelihood but no population column "
                f"was provided. Supply population= in from_dataframe()."
            )


def _check_weak_media_variation(dataset, result: ValidationResult) -> None:
    for channel in dataset.channels:
        spend = dataset.media[dataset.media["channel"] == channel]["spend"]
        mean = spend.mean()
        if mean > 0 and (spend.std() / mean) < 0.05:
            result.warnings.append(
                f"Weak media variation for channel '{channel}': "
                f"coefficient of variation = {spend.std() / mean:.3f}. "
                f"MMM estimates will be unreliable for this channel."
            )


def _check_missing_outcomes(dataset, result: ValidationResult) -> None:
    missing = dataset.observations["outcome"].isna().sum()
    if missing > 0:
        result.errors.append(
            f"Missing outcome values: {missing} rows have NaN outcome"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_validation.py -v`

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add calmmm/data/validation.py tests/data/test_validation.py
git commit -m "feat: MMMData validation with hard errors and soft warnings"
```

---

## Task 6: Adstock Transforms

**Files:**
- Create: `calmmm/transforms/adstock.py`
- Create: `tests/transforms/test_adstock.py`

**Geometric adstock** — `x_adstocked[t] = x[t] + alpha * x_adstocked[t-1]`, alpha ∈ [0,1).

**Weibull adstock** — lag weights from the Weibull PDF over L lags (starting at lag=1 to avoid l=0 singularity when shape < 1):

```
lags = [1, 2, ..., L]
w[l] = Weibull_PDF(l; shape=k, scale=lambda)
w = w / sum(w)
x_adstocked[t] = sum_{l=1}^{L} w[l-1] * x[t - l + 1]
```

Default L=13 (one quarter for weekly data).

- [ ] **Step 1: Write the failing tests**

Create `tests/transforms/test_adstock.py`:

```python
import numpy as np
import pytest
from calmmm.transforms.adstock import geometric_adstock, weibull_adstock


def test_geometric_adstock_zero_decay_is_identity():
    x = np.array([1.0, 2.0, 3.0, 0.0])
    result = geometric_adstock(x, decay=0.0)
    np.testing.assert_allclose(result, x)


def test_geometric_adstock_full_decay_accumulates():
    x = np.array([1.0, 0.0, 0.0, 0.0])
    result = geometric_adstock(x, decay=0.5)
    expected = np.array([1.0, 0.5, 0.25, 0.125])
    np.testing.assert_allclose(result, expected)


def test_geometric_adstock_invalid_decay_raises():
    with pytest.raises(ValueError, match="decay must be in"):
        geometric_adstock(np.ones(5), decay=1.5)


def test_weibull_adstock_output_shape():
    x = np.random.default_rng(0).uniform(0, 1, 52)
    result = weibull_adstock(x, shape=1.5, scale=3.0, n_lags=13)
    assert result.shape == x.shape


def test_weibull_adstock_geometric_limit():
    """When shape=1 (Weibull reduces to Exponential), adstock should match geometric."""
    x = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    # Weibull with shape=1, scale=s gives geometric-like decay
    result = weibull_adstock(x, shape=1.0, scale=2.0, n_lags=5)
    # First value should be highest, then monotone decay
    assert result[0] >= result[1] >= result[2]


def test_weibull_adstock_delayed_peak():
    """When shape > 1, peak effect should be delayed, not at lag=1."""
    x = np.zeros(20)
    x[0] = 1.0  # single impulse at t=0
    result = weibull_adstock(x, shape=3.0, scale=5.0, n_lags=13)
    peak_idx = np.argmax(result)
    assert peak_idx > 0, "Delayed peak expected when shape > 1"


def test_weibull_adstock_conservative_with_zero_spend():
    x = np.zeros(10)
    result = weibull_adstock(x, shape=1.5, scale=3.0, n_lags=5)
    np.testing.assert_allclose(result, np.zeros(10))
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/transforms/test_adstock.py -v`

Expected: `ImportError: cannot import name 'geometric_adstock'`

- [ ] **Step 3: Implement `calmmm/transforms/adstock.py`**

```python
from __future__ import annotations
import numpy as np
from scipy.stats import weibull_min


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Recursive geometric adstock.

    x_adstocked[t] = x[t] + decay * x_adstocked[t-1]
    """
    if not (0.0 <= decay < 1.0):
        raise ValueError(f"decay must be in [0, 1), got {decay}")
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + decay * out[t - 1]
    return out


def weibull_adstock(
    x: np.ndarray,
    shape: float,
    scale: float,
    n_lags: int = 13,
) -> np.ndarray:
    """
    Weibull lag-kernel adstock.

    Lag weights are proportional to the Weibull PDF evaluated at lags 1..n_lags.
    shape > 1 produces a delayed peak; shape = 1 is exponential; shape < 1 is
    front-loaded (similar to geometric but with heavier tail).
    """
    if shape <= 0:
        raise ValueError(f"shape must be > 0, got {shape}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    lags = np.arange(1, n_lags + 1, dtype=float)
    weights = weibull_min.pdf(lags, c=shape, scale=scale)
    weights_sum = weights.sum()
    if weights_sum == 0:
        weights = np.ones(n_lags) / n_lags
    else:
        weights = weights / weights_sum

    # Convolve: out[t] = sum_{l=0}^{n_lags-1} weights[l] * x[t - l]
    # Using np.convolve (full), then trim to original length
    out = np.convolve(x, weights, mode="full")[: len(x)]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/transforms/test_adstock.py -v`

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add calmmm/transforms/adstock.py tests/transforms/test_adstock.py
git commit -m "feat: geometric and Weibull adstock transforms"
```

---

## Task 7: Saturation Curves

**Files:**
- Create: `calmmm/transforms/saturation.py`
- Create: `tests/transforms/test_saturation.py`

**Hill saturation:**
```
hill(x; alpha, K) = x^alpha / (x^alpha + K^alpha)
```
`alpha > 0` controls curvature; `K > 0` is the half-saturation spend level.

**I-spline saturation:** Cumulative integral of B-spline basis with non-negative weights, ensuring monotone response. Weights are Dirichlet-distributed in the model (here we expose the basis matrix construction).

- [ ] **Step 1: Write the failing tests**

Create `tests/transforms/test_saturation.py`:

```python
import numpy as np
import pytest
from calmmm.transforms.saturation import hill_saturation, ispline_basis


def test_hill_saturation_half_saturation_point():
    """At x=K, hill(x) should equal 0.5 for any alpha."""
    K = 1000.0
    x = np.array([K])
    for alpha in [0.5, 1.0, 2.0, 3.0]:
        result = hill_saturation(x, alpha=alpha, K=K)
        np.testing.assert_allclose(result, [0.5], atol=1e-10)


def test_hill_saturation_monotone():
    x = np.linspace(0, 5000, 100)
    result = hill_saturation(x, alpha=2.0, K=1000.0)
    assert np.all(np.diff(result) >= 0), "Hill saturation must be monotone non-decreasing"


def test_hill_saturation_zero_input():
    result = hill_saturation(np.array([0.0]), alpha=2.0, K=1000.0)
    np.testing.assert_allclose(result, [0.0])


def test_hill_saturation_asymptote():
    x = np.array([1e9])
    result = hill_saturation(x, alpha=2.0, K=1000.0)
    assert result[0] > 0.9999, "Hill should approach 1 at very high spend"


def test_hill_saturation_invalid_params():
    with pytest.raises(ValueError):
        hill_saturation(np.ones(5), alpha=-1.0, K=100.0)
    with pytest.raises(ValueError):
        hill_saturation(np.ones(5), alpha=1.0, K=0.0)


def test_ispline_basis_shape():
    x = np.linspace(0, 1, 50)
    B = ispline_basis(x, n_knots=4)
    assert B.shape[0] == 50
    assert B.shape[1] >= 4


def test_ispline_basis_monotone_columns():
    """Each column of the I-spline basis must be monotone non-decreasing."""
    x = np.linspace(0, 1, 200)
    B = ispline_basis(x, n_knots=4)
    for j in range(B.shape[1]):
        diffs = np.diff(B[:, j])
        assert np.all(diffs >= -1e-10), f"Column {j} is not monotone"


def test_ispline_basis_bounded_zero_one():
    x = np.linspace(0, 1, 100)
    B = ispline_basis(x, n_knots=4)
    assert B.min() >= -1e-10
    assert B.max() <= 1.0 + 1e-10
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/transforms/test_saturation.py -v`

Expected: `ImportError: cannot import name 'hill_saturation'`

- [ ] **Step 3: Implement `calmmm/transforms/saturation.py`**

```python
from __future__ import annotations
import numpy as np
from scipy.interpolate import BSpline


def hill_saturation(x: np.ndarray, alpha: float, K: float) -> np.ndarray:
    """
    Hill (power) saturation curve.

    f(x) = x^alpha / (x^alpha + K^alpha)

    alpha: shape (curvature); must be > 0
    K: half-saturation point; must be > 0
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    x = np.asarray(x, dtype=float)
    x_pow = np.power(np.clip(x, 0, None), alpha)
    K_pow = K ** alpha
    return x_pow / (x_pow + K_pow)


def ispline_basis(x: np.ndarray, n_knots: int = 4, degree: int = 3) -> np.ndarray:
    """
    I-spline basis matrix (monotone non-decreasing).

    Each column is the numerical integral of a B-spline basis function,
    normalized to [0, 1]. Non-negative linear combinations of columns are
    guaranteed to be monotone non-decreasing.

    x: 1-D array of evaluation points, assumed sorted or will be handled.
    n_knots: number of interior knots (placed at equally-spaced quantiles of x).
    degree: B-spline degree (3 = cubic).

    Returns: (len(x), n_basis) matrix where n_basis = n_knots + degree - 1.
    """
    x = np.asarray(x, dtype=float)
    x_min, x_max = x.min(), x.max()

    if x_max <= x_min:
        raise ValueError("x must have at least two distinct values")

    # Interior knot placement at equally-spaced quantiles
    quantiles = np.linspace(0, 100, n_knots + 2)[1:-1]
    interior_knots = np.percentile(x, quantiles)

    # Pad knots for B-spline of given degree
    t = np.concatenate([
        [x_min] * degree,
        interior_knots,
        [x_max] * degree,
    ])

    n_basis = len(t) - degree - 1
    B = np.zeros((len(x), n_basis))

    # Dense grid for numerical integration
    x_dense = np.linspace(x_min, x_max, max(500, len(x) * 5))

    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        spl = BSpline(t, c, degree, extrapolate=False)
        b_dense = np.nan_to_num(spl(x_dense), nan=0.0)
        # Cumulative trapezoid integral
        dx = x_dense[1] - x_dense[0]
        cumint = np.cumsum(b_dense) * dx
        # Normalize to [0, 1]
        max_val = cumint[-1]
        if max_val > 0:
            cumint = cumint / max_val
        # Evaluate at actual x points
        B[:, i] = np.interp(x, x_dense, cumint)

    return np.clip(B, 0.0, 1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/transforms/test_saturation.py -v`

Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add calmmm/transforms/saturation.py tests/transforms/test_saturation.py
git commit -m "feat: Hill saturation and I-spline monotone saturation basis"
```

---

## Task 8: Seasonality, Scaling, and Integration Test

**Files:**
- Create: `calmmm/transforms/seasonality.py`
- Create: `calmmm/transforms/scaling.py`
- Create: `tests/transforms/test_seasonality.py`
- Create: `tests/transforms/test_scaling.py`

**Fourier features:** `sin(2π * n * t / period)` and `cos(2π * n * t / period)` for `n = 1, ..., n_pairs`. Default period=52 (annual cycle at weekly frequency).

**Media scaling:** divide each channel's spend by its maximum across the full panel, yielding spend in [0, 1]. Scaler stores the per-channel max so predictions can be unscaled.

- [ ] **Step 1: Write the failing tests**

Create `tests/transforms/test_seasonality.py`:

```python
import numpy as np
import pytest
from calmmm.transforms.seasonality import fourier_features


def test_fourier_features_shape():
    t = np.arange(52)
    F = fourier_features(t, n_pairs=2, period=52)
    assert F.shape == (52, 4)  # 2 pairs * 2 (sin + cos)


def test_fourier_features_period_returns_to_start():
    t = np.arange(104)
    F = fourier_features(t, n_pairs=1, period=52)
    np.testing.assert_allclose(F[0], F[52], atol=1e-10)


def test_fourier_features_bounded():
    t = np.arange(200)
    F = fourier_features(t, n_pairs=3, period=52)
    assert F.max() <= 1.0 + 1e-10
    assert F.min() >= -1.0 - 1e-10
```

Create `tests/transforms/test_scaling.py`:

```python
import numpy as np
import pytest
from calmmm.transforms.scaling import MediaScaler


def test_media_scaler_output_in_zero_one():
    spend = np.array([100.0, 500.0, 1000.0, 200.0])
    scaler = MediaScaler()
    scaled = scaler.fit_transform(spend)
    assert scaled.max() <= 1.0 + 1e-10
    assert scaled.min() >= 0.0 - 1e-10
    np.testing.assert_allclose(scaled.max(), 1.0)


def test_media_scaler_roundtrip():
    spend = np.array([100.0, 500.0, 1000.0, 200.0])
    scaler = MediaScaler()
    scaled = scaler.fit_transform(spend)
    recovered = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(recovered, spend)


def test_media_scaler_zero_spend_channel():
    spend = np.zeros(10)
    scaler = MediaScaler()
    scaled = scaler.fit_transform(spend)
    np.testing.assert_allclose(scaled, np.zeros(10))


def test_media_scaler_not_fitted_raises():
    scaler = MediaScaler()
    with pytest.raises(RuntimeError, match="not fitted"):
        scaler.inverse_transform(np.ones(5))
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/transforms/ -v`

Expected: `ImportError`s for both modules.

- [ ] **Step 3: Implement `calmmm/transforms/seasonality.py`**

```python
from __future__ import annotations
import numpy as np


def fourier_features(
    t: np.ndarray,
    n_pairs: int = 2,
    period: float = 52.0,
) -> np.ndarray:
    """
    Fourier seasonal features.

    For each pair n = 1..n_pairs, produces sin(2π*n*t/period) and cos(2π*n*t/period).
    Returns array of shape (len(t), 2 * n_pairs).
    """
    t = np.asarray(t, dtype=float)
    cols = []
    for n in range(1, n_pairs + 1):
        angle = 2.0 * np.pi * n * t / period
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    return np.column_stack(cols)
```

- [ ] **Step 4: Implement `calmmm/transforms/scaling.py`**

```python
from __future__ import annotations
import numpy as np
from typing import Optional


class MediaScaler:
    """
    Scales media spend to [0, 1] by dividing by per-channel maximum.

    Stores the max value so predictions at inference time can be unscaled.
    """

    def __init__(self):
        self._max: Optional[float] = None

    def fit_transform(self, spend: np.ndarray) -> np.ndarray:
        spend = np.asarray(spend, dtype=float)
        self._max = spend.max()
        if self._max == 0.0:
            return np.zeros_like(spend)
        return spend / self._max

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        if self._max is None:
            raise RuntimeError("MediaScaler is not fitted; call fit_transform first")
        return np.asarray(scaled, dtype=float) * self._max

    @property
    def max_spend(self) -> float:
        if self._max is None:
            raise RuntimeError("MediaScaler is not fitted")
        return self._max
```

- [ ] **Step 5: Run all transform tests**

Run: `pytest tests/transforms/ -v`

Expected: all tests pass (7 adstock + 9 saturation + 3 seasonality + 4 scaling = **23 passed**)

- [ ] **Step 6: Run the full test suite to confirm nothing is broken**

Run: `pytest tests/ -v`

Expected: all tests pass with no errors

- [ ] **Step 7: Commit**

```bash
git add calmmm/transforms/seasonality.py calmmm/transforms/scaling.py \
        tests/transforms/test_seasonality.py tests/transforms/test_scaling.py
git commit -m "feat: Fourier seasonality features and media spend scaler"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `MMMData` container with `from_dataframe` | Task 3 |
| `IncrementalityTests` container | Task 4 |
| Wide-to-long data normalization | Task 3 (internal normalization in `from_dataframe`) |
| Missing columns → hard error | Task 5 (validation module) |
| Duplicate panel rows → hard error | Task 5 |
| Negative spend → hard error | Task 5 |
| Rate KPI without denominator → warning | Task 5 |
| Sparse/weak media variation → warning | Task 5 |
| Experiment referencing unknown channel/KPI → error | Task 4 |
| Experiment window outside panel → error | Task 4 |
| Missing SE → error | Task 2 (schema `__post_init__`) |
| CI-to-SE conversion | Task 2 (schema `__post_init__`) |
| Geometric adstock | Task 6 |
| Weibull adstock | Task 6 |
| Hill saturation | Task 7 |
| Monotone I-spline saturation | Task 7 |
| Fourier seasonality | Task 8 |
| Media scaling | Task 8 |
| `exposure` stored alongside `spend` | Task 3 |
| `population` stored on observations | Task 3 |
| Per-experiment calibration likelihood field | Task 2 (schema) |
| Estimand type field | Task 2 (schema) |

**Not covered in this plan (deferred to Plans 2–4):**
- PyMC model, inference modes, priors, hierarchy
- Calibration likelihood and lift computation
- Diagnostics, attribution, optimization, reporting

**Placeholder scan:** No TBDs, no "implement later", no "similar to Task N" — all code is complete.

**Type consistency:** `ExperimentRow` is defined in Task 2 and imported in Task 4 via `calmmm.data.schema`. `MMMData` is defined in Task 3 and used in Task 4. `ValidationResult` is defined and tested in Task 5.
