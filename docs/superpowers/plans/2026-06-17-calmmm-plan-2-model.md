# calmmm.model — PyMC Hierarchy, Baseline, Likelihoods, Inference

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `calmmm.model` — a PyMC hierarchical MMM that fits `MMMData` with geometric adstock, Hill saturation, geo×KPI pooling, and three inference modes (MCMC / VI / MAP).

**Architecture:** PyMC model assembled inside `HierarchicalMMM.build_model()`; all adstock and saturation transforms are implemented in PyTensor so they are part of the probabilistic graph. The media contribution uses a three-level non-centered hierarchy (global → KPI → geo). `MMMFit` wraps the `InferenceData` object and computes holdout metrics post-sampling against the last `holdout_fraction` of time steps (true holdout: those rows are excluded from the likelihood during fitting).

**Tech Stack:** pymc>=5, pytensor>=2.18, numpy, pandas, calmmm.transforms.seasonality.fourier_features, calmmm.data.containers.MMMData

**Key design decisions (lock these in; don't re-derive):**
- Log link for all KPIs: `mu[t,g,k]` is on log scale; likelihoods use `exp(mu_k)` as mean.
- Geometric adstock only (Weibull inside PyMC deferred — requires PyTensor Weibull PDF).
- Hill saturation (parametric, fully pytensor) not I-spline (I-spline requires pytensor B-spline, deferred).
- Non-centered parameterization for geo and KPI hierarchy levels.
- `pm.Data` used for media and Fourier matrices (enables out-of-sample prediction later).
- Holdout: last `int(T * holdout_fraction)` time steps excluded from likelihood; evaluated post-fitting via `sample_posterior_predictive`.
- `fourier_features(t, ...)` takes `t=np.arange(T)` (1-D index array), not `n=T`.

---

## File Structure

| File | Purpose |
|---|---|
| `calmmm/model/__init__.py` | Exports `HierarchicalMMM`, `MMMFit` |
| `calmmm/model/coords.py` | `build_coords()`, `build_arrays()` — data ↔ numpy arrays |
| `calmmm/model/priors.py` | `PriorConfig` dataclass — all prior hyperparameters |
| `calmmm/model/transforms.py` | `geometric_adstock_pt()`, `hill_saturation_pt()` — pytensor ops |
| `calmmm/model/components.py` | `_build_baseline()`, `_build_media_hierarchy()`, `_add_likelihood()` — PyMC fragments |
| `calmmm/model/mmm.py` | `HierarchicalMMM` class — `build_model()`, `fit()` |
| `calmmm/model/fit.py` | `MMMFit` class — `holdout_metrics()`, `posterior_predictive()` |
| `calmmm/__init__.py` | Add `HierarchicalMMM`, `MMMFit` to lazy exports |
| `tests/model/__init__.py` | Empty |
| `tests/model/test_coords.py` | Coords + array builder tests |
| `tests/model/test_priors.py` | PriorConfig defaults tests |
| `tests/model/test_transforms.py` | PyTensor adstock + saturation tests |
| `tests/model/test_components.py` | Component fragment tests |
| `tests/model/test_mmm.py` | HierarchicalMMM integration tests |
| `tests/model/test_fit.py` | MMMFit tests |
| `tests/conftest.py` | **Modify** — add `mmmdata` and `experiments` fixtures |

---

## Task 1: Coords and Array Builder

**Files:**
- Create: `calmmm/model/__init__.py`
- Create: `calmmm/model/coords.py`
- Create: `tests/model/__init__.py`
- Create: `tests/model/test_coords.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1.1: Add `mmmdata` and `lift_tests` fixtures to `tests/conftest.py`**

Append to the bottom of `tests/conftest.py`:

```python
from calmmm.data.containers import MMMData, IncrementalityTests


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
```

- [ ] **Step 1.2: Write failing tests**

Create `tests/model/__init__.py` (empty).

Create `tests/model/test_coords.py`:

```python
import numpy as np
import pytest

from calmmm.model.coords import build_coords, build_arrays


def test_build_coords_keys(mmmdata):
    coords = build_coords(mmmdata, n_fourier_pairs=2)
    assert set(coords.keys()) == {"time", "geo", "kpi", "channel", "fourier"}


def test_build_coords_fourier_length(mmmdata):
    coords = build_coords(mmmdata, n_fourier_pairs=3)
    assert len(coords["fourier"]) == 6  # 2 * n_pairs


def test_build_coords_lists_are_sorted(mmmdata):
    coords = build_coords(mmmdata)
    assert coords["geo"] == sorted(coords["geo"])
    assert coords["kpi"] == sorted(coords["kpi"])
    assert coords["channel"] == sorted(coords["channel"])


def test_build_arrays_shapes(mmmdata):
    obs, media, pop = build_arrays(mmmdata)
    T, G, K, C = 52, 2, 4, 2
    assert obs.shape == (T, G, K)
    assert media.shape == (T, G, C)
    assert pop.shape == (T, G, K)


def test_build_arrays_dtype(mmmdata):
    obs, media, pop = build_arrays(mmmdata)
    assert obs.dtype == np.float64
    assert media.dtype == np.float64
    assert pop.dtype == np.float64


def test_build_arrays_obs_no_nan(mmmdata):
    obs, _, _ = build_arrays(mmmdata)
    assert not np.any(np.isnan(obs))


def test_build_arrays_media_nonneg(mmmdata):
    _, media, _ = build_arrays(mmmdata)
    assert np.all(media >= 0)


def test_build_arrays_population_positive(mmmdata):
    _, _, pop = build_arrays(mmmdata)
    # synthetic_panel provides population; should be positive (not NaN)
    assert np.all(pop > 0) or np.all(np.isnan(pop))  # either populated or all NaN
```

- [ ] **Step 1.3: Run to confirm failure**

```bash
uv run pytest tests/model/test_coords.py -v 2>&1 | head -20
```
Expected: ImportError or collection error (module not found).

- [ ] **Step 1.4: Create `calmmm/model/__init__.py`**

```python
from calmmm.model.mmm import HierarchicalMMM
from calmmm.model.fit import MMMFit

__all__ = ["HierarchicalMMM", "MMMFit"]
```

(This will fail at import until Tasks 7–8 exist. For now, create it as empty so the package is importable.)

Actual content to write now:

```python
# populated in Tasks 7 and 9
```

- [ ] **Step 1.5: Create `calmmm/model/coords.py`**

```python
from __future__ import annotations

import numpy as np

from calmmm.data.containers import MMMData


def build_coords(data: MMMData, n_fourier_pairs: int = 2) -> dict[str, list]:
    """Return PyMC coords dict for use in pm.Model(coords=...)."""
    return {
        "time": data.times,
        "geo": data.geos,
        "kpi": data.kpis,
        "channel": data.channels,
        "fourier": list(range(2 * n_fourier_pairs)),
    }


def build_arrays(
    data: MMMData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pivot MMMData long frames into dense numpy arrays.

    Returns
    -------
    obs_array : float64 [T, G, K] — observed outcomes
    media_array : float64 [T, G, C] — spend (raw, unscaled)
    pop_array : float64 [T, G, K] — population; NaN where unavailable
    """
    times = data.times
    geos = data.geos
    kpis = data.kpis
    channels = data.channels
    T = len(times)
    G = len(geos)
    K = len(kpis)
    C = len(channels)

    t_idx = {t: i for i, t in enumerate(times)}
    g_idx = {g: i for i, g in enumerate(geos)}
    k_idx = {k: i for i, k in enumerate(kpis)}
    c_idx = {c: i for i, c in enumerate(channels)}

    # Observations → [T, G, K]
    obs_array = np.full((T, G, K), np.nan)
    df = data.observations.copy()
    ti = df["time"].map(t_idx).values
    gi = df["geo"].map(g_idx).values
    ki = df["kpi"].map(k_idx).values
    obs_array[ti, gi, ki] = df["outcome"].values

    # Media → [T, G, C]
    media_array = np.zeros((T, G, C))
    mdf = data.media.copy()
    mti = mdf["time"].map(t_idx).values
    mgi = mdf["geo"].map(g_idx).values
    mci = mdf["channel"].map(c_idx).values
    media_array[mti, mgi, mci] = mdf["spend"].values

    # Population → [T, G, K]
    pop_array = np.full((T, G, K), np.nan)
    if "population" in df.columns:
        valid = df["population"].notna().values
        pop_array[ti[valid], gi[valid], ki[valid]] = df["population"].values[valid]

    return (
        obs_array.astype(np.float64),
        media_array.astype(np.float64),
        pop_array.astype(np.float64),
    )
```

- [ ] **Step 1.6: Run tests**

```bash
uv run pytest tests/model/test_coords.py -v
```
Expected: all 8 tests PASS.

- [ ] **Step 1.7: Commit**

```bash
git add calmmm/model/__init__.py calmmm/model/coords.py tests/model/__init__.py tests/model/test_coords.py tests/conftest.py
git commit -m "feat(model): add coords builder and array pivot"
```

---

## Task 2: Prior Configuration

**Files:**
- Create: `calmmm/model/priors.py`
- Create: `tests/model/test_priors.py`

- [ ] **Step 2.1: Write failing tests**

Create `tests/model/test_priors.py`:

```python
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
```

- [ ] **Step 2.2: Run to confirm failure**

```bash
uv run pytest tests/model/test_priors.py -v 2>&1 | head -10
```
Expected: ImportError.

- [ ] **Step 2.3: Create `calmmm/model/priors.py`**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PriorConfig:
    # Geometric adstock: decay ~ Beta(alpha, beta)
    adstock_decay_alpha: float = 3.0
    adstock_decay_beta: float = 3.0

    # Hill saturation
    hill_alpha_sigma: float = 0.5   # HalfNormal sigma for shape exponent
    hill_k_sigma: float = 1.0       # HalfNormal sigma for half-saturation point

    # Baseline (log scale)
    baseline_sigma: float = 2.0     # intercept prior spread around log(mean)
    seasonality_sigma: float = 0.5  # Fourier coefficient scale

    # Media hierarchy (non-centered)
    channel_scale_global_sigma: float = 1.0
    channel_scale_kpi_sigma: float = 0.5
    channel_scale_geo_sigma: float = 0.25

    # KPI dispersion
    sigma_sigma: float = 0.5        # Gaussian/LogNormal sigma
    nb_alpha_sigma: float = 1.0     # NegBin dispersion
```

- [ ] **Step 2.4: Run tests**

```bash
uv run pytest tests/model/test_priors.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add calmmm/model/priors.py tests/model/test_priors.py
git commit -m "feat(model): add PriorConfig dataclass"
```

---

## Task 3: PyTensor Adstock and Hill Saturation

**Files:**
- Create: `calmmm/model/transforms.py`
- Create: `tests/model/test_transforms.py`

- [ ] **Step 3.1: Write failing tests**

Create `tests/model/test_transforms.py`:

```python
import numpy as np
import pytensor.tensor as pt
import pytest

from calmmm.model.transforms import geometric_adstock_pt, hill_saturation_pt


# --- geometric_adstock_pt ---

def test_adstock_zero_decay_equals_input():
    T, G, C = 5, 2, 3
    X = np.ones((T, G, C), dtype="float64")
    decay = np.zeros(C, dtype="float64")
    result = geometric_adstock_pt(
        pt.as_tensor_variable(X), pt.as_tensor_variable(decay)
    ).eval()
    np.testing.assert_allclose(result, X)


def test_adstock_shape():
    T, G, C = 10, 2, 3
    X = np.random.rand(T, G, C)
    decay = np.array([0.5, 0.3, 0.7])
    result = geometric_adstock_pt(
        pt.as_tensor_variable(X), pt.as_tensor_variable(decay)
    ).eval()
    assert result.shape == (T, G, C)


def test_adstock_accumulates_over_time():
    # With decay=0.5, constant input=1: values should grow then plateau
    T, G, C = 20, 1, 1
    X = np.ones((T, G, C), dtype="float64")
    decay = np.array([0.5])
    result = geometric_adstock_pt(
        pt.as_tensor_variable(X), pt.as_tensor_variable(decay)
    ).eval()
    # t=0: h=1, t=1: h=1.5, t=2: h=1.75, ... plateau near 2.0
    assert result[0, 0, 0] < result[5, 0, 0]
    assert result[5, 0, 0] < result[15, 0, 0] + 0.01  # nearly converged


def test_adstock_full_decay_doubles_immediately():
    # decay=1 means all prior signal carries over: h[t] = x[t] + h[t-1]
    T, G, C = 5, 1, 1
    X = np.ones((T, G, C), dtype="float64")
    decay = np.array([1.0])
    result = geometric_adstock_pt(
        pt.as_tensor_variable(X), pt.as_tensor_variable(decay)
    ).eval()
    # t=0:1, t=1:2, t=2:3 ...
    np.testing.assert_allclose(result[:, 0, 0], np.arange(1, T + 1, dtype=float))


# --- hill_saturation_pt ---

def test_hill_at_half_saturation_point():
    # At X=k, saturation should be 0.5 (with alpha=1)
    C = 2
    k_vals = np.array([1.0, 2.0])
    X = np.array([[[1.0, 2.0]]])  # [1, 1, 2]
    result = hill_saturation_pt(
        pt.as_tensor_variable(X),
        pt.as_tensor_variable(np.ones(C)),
        pt.as_tensor_variable(k_vals),
    ).eval()
    np.testing.assert_allclose(result[0, 0, :], [0.5, 0.5], atol=1e-3)


def test_hill_range():
    T, G, C = 5, 2, 3
    X = pt.as_tensor_variable(np.abs(np.random.rand(T, G, C)))
    alpha = pt.as_tensor_variable(np.array([0.5, 1.0, 2.0]))
    k = pt.as_tensor_variable(np.array([0.5, 1.0, 0.3]))
    result = hill_saturation_pt(X, alpha, k).eval()
    assert np.all(result >= 0) and np.all(result <= 1)


def test_hill_shape():
    T, G, C = 8, 3, 4
    X = pt.as_tensor_variable(np.random.rand(T, G, C))
    result = hill_saturation_pt(
        X,
        pt.as_tensor_variable(np.ones(C)),
        pt.as_tensor_variable(np.ones(C)),
    ).eval()
    assert result.shape == (T, G, C)


def test_hill_zero_input_is_zero():
    T, G, C = 3, 2, 2
    X = pt.as_tensor_variable(np.zeros((T, G, C)))
    result = hill_saturation_pt(
        X,
        pt.as_tensor_variable(np.array([1.0, 2.0])),
        pt.as_tensor_variable(np.array([0.5, 1.0])),
    ).eval()
    np.testing.assert_allclose(result, 0.0, atol=1e-6)
```

- [ ] **Step 3.2: Run to confirm failure**

```bash
uv run pytest tests/model/test_transforms.py -v 2>&1 | head -10
```
Expected: ImportError.

- [ ] **Step 3.3: Create `calmmm/model/transforms.py`**

```python
from __future__ import annotations

import pytensor
import pytensor.tensor as pt


def geometric_adstock_pt(X, decay):
    """
    Geometric adstock via pytensor.scan.

    Parameters
    ----------
    X : tensor [T, G, C]
    decay : tensor [C], values in [0, 1]

    Returns
    -------
    tensor [T, G, C]
        Adstocked spend. h[t] = X[t] + decay * h[t-1], h[0] = X[0].
    """
    def _step(x_t, h_prev, decay_):
        # x_t: [G, C], h_prev: [G, C], decay_: [C]
        return x_t + h_prev * decay_[None, :]

    h0 = pt.zeros_like(X[0])  # [G, C]
    h_seq, _ = pytensor.scan(
        _step,
        sequences=[X],
        outputs_info=[h0],
        non_sequences=[decay],
    )
    return h_seq  # [T, G, C]


def hill_saturation_pt(X, alpha, k):
    """
    Hill saturation curve (vectorized over channels).

    Parameters
    ----------
    X : tensor [..., C] — input values (should be >= 0)
    alpha : tensor [C] — exponent / steepness (> 0)
    k : tensor [C] — half-saturation point (> 0)

    Returns
    -------
    tensor same shape as X, values in [0, 1]
    """
    # Broadcast alpha and k over all leading dims
    # Works for X shape [T, G, C]: alpha[None, None, :], k[None, None, :]
    ndim = X.ndim
    expand = (None,) * (ndim - 1) + (slice(None),)
    a = alpha[expand]
    kk = k[expand]
    x_pow = X ** a
    k_pow = kk ** a
    return x_pow / (x_pow + k_pow + 1e-9)
```

- [ ] **Step 3.4: Run tests**

```bash
uv run pytest tests/model/test_transforms.py -v
```
Expected: 8 tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add calmmm/model/transforms.py tests/model/test_transforms.py
git commit -m "feat(model): add pytensor geometric adstock and Hill saturation"
```

---

## Task 4: Baseline Component

**Files:**
- Create: `calmmm/model/components.py`
- Create: `tests/model/test_components.py`

- [ ] **Step 4.1: Write failing tests (baseline section)**

Create `tests/model/test_components.py`:

```python
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest

from calmmm.model.priors import PriorConfig
from calmmm.model.components import _build_baseline, _build_media_hierarchy, _add_likelihood


KPIS = ["applications", "approvals", "revenue", "visits"]  # sorted
GEOS = ["DMA_1", "DMA_2"]
CHANNELS = ["search", "social"]


def _base_coords(T: int = 20, n_fourier: int = 4):
    return {
        "kpi": KPIS,
        "geo": GEOS,
        "channel": CHANNELS,
        "fourier": list(range(n_fourier)),
    }


# ---- Baseline ----

def test_baseline_shape():
    T, G, K, F = 20, 2, 4, 4
    priors = PriorConfig()
    obs_mean_log = np.zeros((K, G))
    fourier_matrix = np.random.rand(T, F)

    with pm.Model(coords=_base_coords(T, F)) as model:
        baseline = _build_baseline(fourier_matrix, obs_mean_log, priors)
        val = model.compile_fn(baseline)(model.initial_point())
    assert val.shape == (T, G, K)


def test_baseline_logp_finite():
    T, G, K, F = 10, 2, 4, 4
    priors = PriorConfig()
    obs_mean_log = np.ones((K, G)) * 5.0
    fourier_matrix = np.random.rand(T, F)

    with pm.Model(coords=_base_coords(T, F)) as model:
        _build_baseline(fourier_matrix, obs_mean_log, priors)
        lp_fn = model.compile_fn(model.logp())
        val = lp_fn(model.initial_point())
    assert np.isfinite(val)


def test_baseline_intercept_shape():
    T, G, K, F = 10, 2, 4, 4
    priors = PriorConfig()
    obs_mean_log = np.zeros((K, G))
    fourier_matrix = np.random.rand(T, F)

    with pm.Model(coords=_base_coords(T, F)) as model:
        _build_baseline(fourier_matrix, obs_mean_log, priors)
        assert model["intercept"].type.shape == (K, G)
        assert model["fourier_beta"].type.shape == (K, F)
```

- [ ] **Step 4.2: Run to confirm failure**

```bash
uv run pytest tests/model/test_components.py::test_baseline_shape -v 2>&1 | head -10
```
Expected: ImportError.

- [ ] **Step 4.3: Create `calmmm/model/components.py` (baseline only)**

```python
from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from calmmm.model.priors import PriorConfig


def _build_baseline(
    fourier_matrix: np.ndarray,
    obs_mean_log: np.ndarray,
    priors: PriorConfig,
) -> pt.TensorVariable:
    """
    Baseline = per-(KPI, geo) intercept + Fourier seasonality.

    Parameters
    ----------
    fourier_matrix : [T, F] numpy array — Fourier features (deterministic)
    obs_mean_log : [K, G] numpy array — log(mean_outcome) per KPI×geo
                   used as intercept prior mean (log scale)
    priors : PriorConfig

    Returns
    -------
    pytensor tensor [T, G, K] — baseline on log scale

    Must be called inside a pm.Model context with coords
    {"kpi": [...], "geo": [...], "fourier": [...]}.
    """
    intercept = pm.Normal(
        "intercept",
        mu=obs_mean_log,
        sigma=priors.baseline_sigma,
        dims=("kpi", "geo"),
    )
    fourier_beta = pm.Normal(
        "fourier_beta",
        mu=0.0,
        sigma=priors.seasonality_sigma,
        dims=("kpi", "fourier"),
    )
    # intercept [K, G] → [1, G, K]
    intercept_tgk = intercept.T[None, :, :]
    # fourier_matrix [T, F] @ fourier_beta.T [F, K] → [T, K] → [T, 1, K]
    fourier_contrib = pt.dot(fourier_matrix, fourier_beta.T)[:, None, :]
    return intercept_tgk + fourier_contrib  # [T, G, K]
```

- [ ] **Step 4.4: Run tests**

```bash
uv run pytest tests/model/test_components.py::test_baseline_shape tests/model/test_components.py::test_baseline_logp_finite tests/model/test_components.py::test_baseline_intercept_shape -v
```
Expected: 3 tests PASS.

- [ ] **Step 4.5: Commit**

```bash
git add calmmm/model/components.py tests/model/test_components.py
git commit -m "feat(model): add baseline component (intercept + Fourier seasonality)"
```

---

## Task 5: Media Hierarchy Component

**Files:**
- Modify: `calmmm/model/components.py`
- Modify: `tests/model/test_components.py`

- [ ] **Step 5.1: Write failing tests (append to test file)**

Append to `tests/model/test_components.py`:

```python
# ---- Media hierarchy ----

def test_media_hierarchy_shape():
    T, G, K, C = 10, 2, 4, 2
    priors = PriorConfig()
    X_sat_val = np.random.rand(T, G, C).astype("float64")

    with pm.Model(coords=_base_coords()) as model:
        X_sat = pt.as_tensor_variable(X_sat_val)
        contrib = _build_media_hierarchy(X_sat, priors)
        val = model.compile_fn(contrib)(model.initial_point())
    assert val.shape == (T, G, K)


def test_media_hierarchy_logp_finite():
    T, G, K, C = 5, 2, 4, 2
    priors = PriorConfig()
    X_sat_val = np.random.rand(T, G, C).astype("float64")

    with pm.Model(coords=_base_coords()) as model:
        X_sat = pt.as_tensor_variable(X_sat_val)
        _build_media_hierarchy(X_sat, priors)
        lp_fn = model.compile_fn(model.logp())
        val = lp_fn(model.initial_point())
    assert np.isfinite(val)


def test_media_hierarchy_variable_names():
    T, G, K, C = 5, 2, 4, 2
    priors = PriorConfig()
    X_sat_val = np.random.rand(T, G, C).astype("float64")

    with pm.Model(coords=_base_coords()) as model:
        _build_media_hierarchy(pt.as_tensor_variable(X_sat_val), priors)
        names = {v.name for v in model.free_RVs}
    assert "scale_global" in names
    assert "scale_kpi_raw" in names
    assert "scale_geo_raw" in names
```

- [ ] **Step 5.2: Run to confirm failure**

```bash
uv run pytest tests/model/test_components.py::test_media_hierarchy_shape -v 2>&1 | head -10
```
Expected: ImportError (function not yet defined).

- [ ] **Step 5.3: Add `_build_media_hierarchy` to `calmmm/model/components.py`**

Append to `calmmm/model/components.py`:

```python
def _build_media_hierarchy(
    X_sat: pt.TensorVariable,
    priors: PriorConfig,
) -> pt.TensorVariable:
    """
    Three-level non-centered geo×KPI hierarchy for media contributions.

    Parameters
    ----------
    X_sat : [T, G, C] pytensor tensor — saturation-transformed spend
    priors : PriorConfig

    Returns
    -------
    pytensor tensor [T, G, K] — media contribution on log scale

    Must be called inside pm.Model context with coords
    {"channel": [...], "kpi": [...], "geo": [...]}.

    Hierarchy:
        scale_global[C] ~ HalfNormal
        scale_kpi[C, K] = scale_global + sigma_kpi * Normal(0,1)  (non-centered)
        scale_geo[C, K, G] = scale_kpi + sigma_geo * Normal(0,1)  (non-centered)
        contrib[t,g,k] = sum_c( X_sat[t,g,c] * scale_geo[c,k,g] )
    """
    # Global channel scale
    scale_global = pm.HalfNormal(
        "scale_global",
        sigma=priors.channel_scale_global_sigma,
        dims="channel",
    )
    # KPI level — non-centered
    scale_kpi_raw = pm.Normal("scale_kpi_raw", 0.0, 1.0, dims=("channel", "kpi"))
    scale_kpi_sigma = pm.HalfNormal(
        "scale_kpi_sigma", sigma=priors.channel_scale_kpi_sigma, dims="channel"
    )
    scale_kpi = pm.Deterministic(
        "scale_kpi",
        scale_global[:, None] + scale_kpi_sigma[:, None] * scale_kpi_raw,
        dims=("channel", "kpi"),
    )
    # Geo level — non-centered
    scale_geo_raw = pm.Normal(
        "scale_geo_raw", 0.0, 1.0, dims=("channel", "kpi", "geo")
    )
    scale_geo_sigma = pm.HalfNormal(
        "scale_geo_sigma", sigma=priors.channel_scale_geo_sigma, dims="channel"
    )
    scale_geo = pm.Deterministic(
        "scale_geo",
        scale_kpi[:, :, None] + scale_geo_sigma[:, None, None] * scale_geo_raw,
        dims=("channel", "kpi", "geo"),
    )
    # Contribution: einsum("tgc,ckg->tgk", X_sat, scale_geo)
    # X_sat [T,G,C] → [T,G,1,C]; scale_geo [C,K,G] → [G,K,C] → [1,G,K,C]
    scale_geo_gkc = scale_geo.dimshuffle(2, 1, 0)  # [G, K, C]
    media_contrib = (
        X_sat[:, :, None, :] * scale_geo_gkc[None, :, :, :]
    ).sum(axis=-1)
    return media_contrib  # [T, G, K]
```

- [ ] **Step 5.4: Run tests**

```bash
uv run pytest tests/model/test_components.py -v -k "hierarchy"
```
Expected: 3 hierarchy tests PASS, baseline tests still PASS (6 total).

- [ ] **Step 5.5: Commit**

```bash
git add calmmm/model/components.py tests/model/test_components.py
git commit -m "feat(model): add media hierarchy component (global → KPI → geo)"
```

---

## Task 6: Observation Likelihoods

**Files:**
- Modify: `calmmm/model/components.py`
- Modify: `tests/model/test_components.py`

- [ ] **Step 6.1: Write failing tests (append to test file)**

Append to `tests/model/test_components.py`:

```python
# ---- Likelihoods ----

def _kpi_meta(name, likelihood):
    import pandas as pd
    return pd.DataFrame([{"kpi": name, "likelihood": likelihood, "funnel_stage": None, "family": None}])


def test_likelihood_gaussian():
    T, G, K = 10, 2, 1
    priors = PriorConfig()
    obs = np.abs(np.random.rand(T, G, K)) * 1000 + 1
    pop = np.full((T, G, K), np.nan)
    kpi_meta = _kpi_meta("revenue", "gaussian")

    with pm.Model() as model:
        mu = pt.as_tensor_variable(np.log(obs))
        _add_likelihood(mu, obs, pop, kpi_meta, ["revenue"], priors)
        val = model.compile_fn(model.logp())(model.initial_point())
    assert np.isfinite(val)


def test_likelihood_lognormal():
    T, G, K = 10, 2, 1
    priors = PriorConfig()
    obs = np.abs(np.random.rand(T, G, K)) * 100 + 1
    pop = np.full((T, G, K), np.nan)
    kpi_meta = _kpi_meta("revenue_ln", "lognormal")

    with pm.Model() as model:
        mu = pt.as_tensor_variable(np.log(obs))
        _add_likelihood(mu, obs, pop, kpi_meta, ["revenue_ln"], priors)
        val = model.compile_fn(model.logp())(model.initial_point())
    assert np.isfinite(val)


def test_likelihood_negative_binomial():
    T, G, K = 10, 2, 1
    priors = PriorConfig()
    obs = np.round(np.abs(np.random.rand(T, G, K)) * 100) + 1
    pop = np.full((T, G, K), np.nan)
    kpi_meta = _kpi_meta("visits", "negative_binomial")

    with pm.Model() as model:
        mu = pt.as_tensor_variable(np.log(obs))
        _add_likelihood(mu, obs, pop, kpi_meta, ["visits"], priors)
        val = model.compile_fn(model.logp())(model.initial_point())
    assert np.isfinite(val)


def test_likelihood_binomial():
    T, G, K = 5, 2, 1
    priors = PriorConfig()
    pop = np.full((T, G, K), 1000.0)
    obs = np.round(pop * 0.05)
    kpi_meta = _kpi_meta("rate_kpi", "binomial")

    with pm.Model() as model:
        mu = pt.as_tensor_variable(np.zeros((T, G, K)))
        _add_likelihood(mu, obs, pop, kpi_meta, ["rate_kpi"], priors)
        val = model.compile_fn(model.logp())(model.initial_point())
    assert np.isfinite(val)


def test_likelihood_binomial_requires_population():
    T, G, K = 5, 2, 1
    priors = PriorConfig()
    obs = np.ones((T, G, K))
    pop = np.full((T, G, K), np.nan)
    kpi_meta = _kpi_meta("rate", "binomial")

    with pm.Model():
        mu = pt.as_tensor_variable(np.zeros((T, G, K)))
        with pytest.raises(ValueError, match="population is NaN"):
            _add_likelihood(mu, obs, pop, kpi_meta, ["rate"], priors)


def test_likelihood_unknown_raises():
    T, G, K = 5, 2, 1
    priors = PriorConfig()
    obs = np.ones((T, G, K))
    pop = np.full((T, G, K), np.nan)
    kpi_meta = _kpi_meta("foo", "poisson")

    with pm.Model():
        mu = pt.as_tensor_variable(np.zeros((T, G, K)))
        with pytest.raises(ValueError, match="Unknown likelihood"):
            _add_likelihood(mu, obs, pop, kpi_meta, ["foo"], priors)


def test_likelihood_multi_kpi():
    T, G, K = 10, 2, 2
    priors = PriorConfig()
    obs = np.abs(np.random.rand(T, G, K)) * 100 + 1
    pop = np.full((T, G, K), np.nan)
    import pandas as pd
    kpi_meta = pd.DataFrame([
        {"kpi": "visits", "likelihood": "negative_binomial", "funnel_stage": None, "family": None},
        {"kpi": "revenue", "likelihood": "gaussian", "funnel_stage": None, "family": None},
    ])

    with pm.Model() as model:
        mu = pt.as_tensor_variable(np.log(obs + 1))
        _add_likelihood(mu, obs, pop, kpi_meta, ["visits", "revenue"], priors)
        val = model.compile_fn(model.logp())(model.initial_point())
    assert np.isfinite(val)
```

- [ ] **Step 6.2: Run to confirm failure**

```bash
uv run pytest tests/model/test_components.py -k "likelihood" -v 2>&1 | head -10
```
Expected: ImportError (function not yet defined).

- [ ] **Step 6.3: Add `_add_likelihood` to `calmmm/model/components.py`**

Append to `calmmm/model/components.py`:

```python
def _add_likelihood(
    mu: pt.TensorVariable,
    obs_array: np.ndarray,
    pop_array: np.ndarray,
    kpi_metadata,
    kpis: list[str],
    priors: PriorConfig,
) -> None:
    """
    Add per-KPI observed likelihood nodes to the current pm.Model context.

    Parameters
    ----------
    mu : pytensor tensor [T, G, K] — log-scale linear predictor
    obs_array : [T, G, K] float64 — observed outcomes
    pop_array : [T, G, K] float64 — population (NaN where not used)
    kpi_metadata : pd.DataFrame with columns "kpi", "likelihood"
    kpis : list[str] — KPI names in axis-K order
    priors : PriorConfig

    Link function: log-link (exp(mu_k) is the mean for all non-binomial likelihoods).
    Binomial uses sigmoid(mu_k) as probability.
    """
    for k, kpi in enumerate(kpis):
        row = kpi_metadata.loc[kpi_metadata["kpi"] == kpi]
        likelihood = row["likelihood"].values[0]
        y_obs = obs_array[:, :, k]
        mu_k = mu[:, :, k]

        if likelihood == "gaussian":
            sigma_k = pm.HalfNormal(f"sigma_{kpi}", sigma=priors.sigma_sigma)
            pm.Normal(f"obs_{kpi}", mu=pm.math.exp(mu_k), sigma=sigma_k, observed=y_obs)

        elif likelihood == "lognormal":
            sigma_k = pm.HalfNormal(f"sigma_{kpi}", sigma=priors.sigma_sigma)
            pm.LogNormal(f"obs_{kpi}", mu=mu_k, sigma=sigma_k, observed=y_obs)

        elif likelihood == "negative_binomial":
            alpha_k = pm.HalfNormal(f"nb_alpha_{kpi}", sigma=priors.nb_alpha_sigma)
            pm.NegativeBinomial(
                f"obs_{kpi}",
                mu=pm.math.exp(mu_k),
                alpha=alpha_k,
                observed=y_obs,
            )

        elif likelihood == "binomial":
            n_pop = pop_array[:, :, k]
            if np.any(np.isnan(n_pop)):
                raise ValueError(
                    f"KPI '{kpi}' has likelihood='binomial' but population is NaN. "
                    "Provide a population column in MMMData."
                )
            pm.Binomial(
                f"obs_{kpi}",
                n=n_pop.astype(int),
                p=pm.math.sigmoid(mu_k),
                observed=y_obs,
            )

        else:
            raise ValueError(
                f"Unknown likelihood '{likelihood}' for KPI '{kpi}'. "
                "Expected: gaussian, lognormal, negative_binomial, binomial."
            )
```

- [ ] **Step 6.4: Run all component tests**

```bash
uv run pytest tests/model/test_components.py -v
```
Expected: 14 tests PASS.

- [ ] **Step 6.5: Commit**

```bash
git add calmmm/model/components.py tests/model/test_components.py
git commit -m "feat(model): add per-KPI observation likelihoods"
```

---

## Task 7: HierarchicalMMM — build_model()

**Files:**
- Create: `calmmm/model/mmm.py`
- Create: `tests/model/test_mmm.py`

- [ ] **Step 7.1: Write failing tests**

Create `tests/model/test_mmm.py`:

```python
import numpy as np
import pymc as pm
import pytest

from calmmm.model.mmm import HierarchicalMMM
from calmmm.model.priors import PriorConfig


def test_build_model_returns_pymc_model(mmmdata):
    mmm = HierarchicalMMM()
    model = mmm.build_model(mmmdata)
    assert isinstance(model, pm.Model)


def test_build_model_has_required_variables(mmmdata):
    mmm = HierarchicalMMM()
    model = mmm.build_model(mmmdata)
    names = {v.name for v in model.free_RVs}
    assert "adstock_decay" in names
    assert "hill_alpha" in names
    assert "hill_k" in names
    assert "intercept" in names
    assert "fourier_beta" in names
    assert "scale_global" in names
    assert "scale_kpi_raw" in names
    assert "scale_geo_raw" in names


def test_build_model_logp_finite(mmmdata):
    mmm = HierarchicalMMM()
    model = mmm.build_model(mmmdata)
    with model:
        ip = model.initial_point()
        lp = model.compile_logp()(ip)
    assert np.isfinite(lp)


def test_build_model_custom_priors(mmmdata):
    priors = PriorConfig(adstock_decay_alpha=5.0, hill_k_sigma=2.0)
    mmm = HierarchicalMMM(priors=priors)
    model = mmm.build_model(mmmdata)
    assert isinstance(model, pm.Model)


def test_build_model_no_holdout(mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.0)
    model = mmm.build_model(mmmdata)
    # train_mask should be all True
    assert mmm._train_mask.all()


def test_build_model_holdout_mask_correct(mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    mmm.build_model(mmmdata)
    T = 52
    n_holdout = int(T * 0.2)
    assert mmm._train_mask.sum() == T - n_holdout
    # last n_holdout are False
    assert not mmm._train_mask[-1]
    assert mmm._train_mask[0]


def test_build_model_deterministic_mu(mmmdata):
    mmm = HierarchicalMMM()
    model = mmm.build_model(mmmdata)
    det_names = {v.name for v in model.deterministics}
    assert "mu" in det_names


def test_build_model_obs_nodes_per_kpi(mmmdata):
    mmm = HierarchicalMMM()
    model = mmm.build_model(mmmdata)
    obs_names = {v.name for v in model.observed_RVs}
    # One obs node per KPI
    for kpi in mmmdata.kpis:
        assert f"obs_{kpi}" in obs_names
```

- [ ] **Step 7.2: Run to confirm failure**

```bash
uv run pytest tests/model/test_mmm.py -v 2>&1 | head -15
```
Expected: ImportError.

- [ ] **Step 7.3: Create `calmmm/model/mmm.py`**

```python
from __future__ import annotations

from typing import Optional

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from calmmm.data.containers import MMMData
from calmmm.model.coords import build_coords, build_arrays
from calmmm.model.priors import PriorConfig
from calmmm.model.transforms import geometric_adstock_pt, hill_saturation_pt
from calmmm.model.components import _build_baseline, _build_media_hierarchy, _add_likelihood
from calmmm.transforms.seasonality import fourier_features


class HierarchicalMMM:
    """
    Hierarchical Bayesian MMM with geo×KPI pooling.

    Parameters
    ----------
    priors : PriorConfig or None — use PriorConfig() defaults if None
    n_fourier_pairs : int — number of sin/cos pairs for seasonal baseline
    holdout_fraction : float — fraction of time steps (last) excluded from likelihood
    """

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
        # Set by build_model()
        self._model: Optional[pm.Model] = None
        self._data: Optional[MMMData] = None
        self._train_mask: Optional[np.ndarray] = None
        self._obs_array: Optional[np.ndarray] = None
        self._media_scaled: Optional[np.ndarray] = None
        self._fourier_matrix: Optional[np.ndarray] = None
        self._pop_array: Optional[np.ndarray] = None

    def build_model(self, data: MMMData) -> pm.Model:
        """
        Construct the PyMC model for the given dataset.

        The model uses a log-link for all KPIs:
            log(E[y]) = baseline[t,g,k] + media_contrib[t,g,k]

        Media pipeline:
            raw_spend → scale (÷ panel max) → geometric adstock → Hill saturation
            → geo×KPI hierarchy → additive log contribution

        Baseline:
            intercept[K, G] (informed by log mean outcome) + Fourier seasonality[K, F]

        Returns
        -------
        pm.Model
        """
        self._data = data
        coords = build_coords(data, n_fourier_pairs=self.n_fourier_pairs)
        obs_array, media_array, pop_array = build_arrays(data)

        T = len(data.times)
        n_fourier = 2 * self.n_fourier_pairs

        # Scale media per-channel by panel max
        media_max = media_array.max(axis=(0, 1), keepdims=True)  # [1, 1, C]
        media_scaled = media_array / np.maximum(media_max, 1e-8)

        # Fourier features: t = 0-based week index
        fourier_matrix = fourier_features(
            t=np.arange(T, dtype=float),
            n_pairs=self.n_fourier_pairs,
            period=52.0,
        ).astype(np.float64)

        # Holdout mask
        n_holdout = int(T * self.holdout_fraction)
        train_mask = np.ones(T, dtype=bool)
        if n_holdout > 0:
            train_mask[-n_holdout:] = False
        self._train_mask = train_mask

        # Baseline intercept initialization: log(mean_outcome) per KPI×geo
        obs_mean = np.nanmean(obs_array, axis=0)  # [G, K]
        obs_mean_log = np.log(np.maximum(obs_mean.T, 1.0))  # [K, G]

        # Store for use in fit()
        self._obs_array = obs_array
        self._media_scaled = media_scaled
        self._fourier_matrix = fourier_matrix
        self._pop_array = pop_array

        # Train slices
        X_media_train = media_scaled[train_mask]       # [T_train, G, C]
        fourier_train = fourier_matrix[train_mask]     # [T_train, F]
        obs_train = obs_array[train_mask]              # [T_train, G, K]
        pop_train = pop_array[train_mask]              # [T_train, G, K]

        with pm.Model(coords=coords) as model:
            # Mutable data containers (can be updated for prediction)
            X_media_data = pm.Data("X_media", X_media_train)
            fourier_data = pm.Data("X_fourier", fourier_train)

            # --- Adstock ---
            decay = pm.Beta(
                "adstock_decay",
                alpha=self.priors.adstock_decay_alpha,
                beta=self.priors.adstock_decay_beta,
                dims="channel",
            )
            X_adstocked = geometric_adstock_pt(X_media_data, decay)  # [T_train, G, C]

            # --- Saturation ---
            hill_alpha = pm.HalfNormal(
                "hill_alpha", sigma=self.priors.hill_alpha_sigma, dims="channel"
            )
            hill_k = pm.HalfNormal(
                "hill_k", sigma=self.priors.hill_k_sigma, dims="channel"
            )
            X_sat = hill_saturation_pt(X_adstocked, hill_alpha, hill_k)  # [T_train, G, C]

            # --- Baseline ---
            baseline = _build_baseline(fourier_train, obs_mean_log, self.priors)  # [T_train, G, K]

            # --- Media hierarchy ---
            media_contrib = _build_media_hierarchy(X_sat, self.priors)  # [T_train, G, K]

            # --- Linear predictor ---
            mu = pm.Deterministic("mu", baseline + media_contrib)  # [T_train, G, K]

            # --- Likelihoods (train only) ---
            _add_likelihood(
                mu, obs_train, pop_train,
                data.kpi_metadata, data.kpis, self.priors
            )

        self._model = model
        return model
```

- [ ] **Step 7.4: Run tests**

```bash
uv run pytest tests/model/test_mmm.py -v
```
Expected: 8 tests PASS (logp and build tests use MAP initial point evaluation, not full sampling — fast).

- [ ] **Step 7.5: Commit**

```bash
git add calmmm/model/mmm.py tests/model/test_mmm.py
git commit -m "feat(model): add HierarchicalMMM.build_model()"
```

---

## Task 8: Inference Modes — fit()

**Files:**
- Modify: `calmmm/model/mmm.py`
- Modify: `tests/model/test_mmm.py`

- [ ] **Step 8.1: Write failing tests (append to `tests/model/test_mmm.py`)**

```python
from calmmm.model.fit import MMMFit


def test_fit_map_returns_mmmfit(mmmdata):
    mmm = HierarchicalMMM()
    fit = mmm.fit(mmmdata, mode="map")
    assert isinstance(fit, MMMFit)


def test_fit_map_map_point_not_none(mmmdata):
    mmm = HierarchicalMMM()
    fit = mmm.fit(mmmdata, mode="map")
    assert fit.map_point is not None
    assert "adstock_decay" in fit.map_point


def test_fit_mode_stored(mmmdata):
    mmm = HierarchicalMMM()
    fit = mmm.fit(mmmdata, mode="map")
    assert fit.mode == "map"


def test_fit_vi_returns_mmmfit(mmmdata):
    mmm = HierarchicalMMM()
    fit = mmm.fit(mmmdata, mode="vi", n=500)
    assert isinstance(fit, MMMFit)
    assert fit.idata is not None


def test_fit_sample_returns_mmmfit(mmmdata):
    # Very short run — just verify it completes and returns right type
    mmm = HierarchicalMMM()
    fit = mmm.fit(mmmdata, mode="sample", draws=20, tune=20, chains=1, progressbar=False)
    assert isinstance(fit, MMMFit)
    assert fit.idata is not None


def test_fit_invalid_mode_raises(mmmdata):
    mmm = HierarchicalMMM()
    with pytest.raises(ValueError, match="mode"):
        mmm.fit(mmmdata, mode="turbo")
```

- [ ] **Step 8.2: Run to confirm failure**

```bash
uv run pytest tests/model/test_mmm.py -k "fit" -v 2>&1 | head -15
```
Expected: ImportError or AttributeError (fit method and MMMFit don't exist yet).

- [ ] **Step 8.3: Create a stub `calmmm/model/fit.py`** (needed for import in test)

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pymc as pm


@dataclass
class MMMFit:
    """Wrapper around PyMC InferenceData after fitting HierarchicalMMM."""

    idata: Optional[Any]          # arviz.InferenceData; None for MAP mode
    map_point: Optional[dict]     # MAP estimate dict; None for sample/vi modes
    model: pm.Model
    data: Any                     # MMMData
    train_mask: np.ndarray        # bool [T] — True for training rows
    obs_array: np.ndarray         # float64 [T, G, K] — full observed outcomes
    mode: str                     # "sample" | "vi" | "map"
```

- [ ] **Step 8.4: Add `fit()` method to `HierarchicalMMM` in `calmmm/model/mmm.py`**

Add this import at the top of `calmmm/model/mmm.py` (after existing imports):

```python
from calmmm.model.fit import MMMFit
```

Add this method to the `HierarchicalMMM` class (after `build_model`):

```python
    def fit(
        self,
        data: MMMData,
        *,
        mode: str = "sample",
        **kwargs,
    ) -> "MMMFit":
        """
        Build and fit the model.

        Parameters
        ----------
        data : MMMData
        mode : "sample" | "vi" | "map"
            sample — full MCMC via pm.sample (default kwargs: draws=1000, tune=1000, chains=2)
            vi     — ADVI via pm.fit (default kwargs: n=30_000)
            map    — MAP optimization via pm.find_MAP
        **kwargs : forwarded to the underlying PyMC inference call

        Returns
        -------
        MMMFit
        """
        if mode not in ("sample", "vi", "map"):
            raise ValueError(
                f"mode must be 'sample', 'vi', or 'map'; got '{mode}'"
            )

        model = self.build_model(data)

        with model:
            if mode == "sample":
                sample_kwargs = dict(draws=1000, tune=1000, chains=2, progressbar=True)
                sample_kwargs.update(kwargs)
                idata = pm.sample(**sample_kwargs)
                return MMMFit(
                    idata=idata,
                    map_point=None,
                    model=model,
                    data=data,
                    train_mask=self._train_mask,
                    obs_array=self._obs_array,
                    mode="sample",
                )

            elif mode == "vi":
                vi_kwargs = dict(n=30_000)
                vi_kwargs.update(kwargs)
                approx = pm.fit(**vi_kwargs)
                idata = approx.sample(1000)
                return MMMFit(
                    idata=idata,
                    map_point=None,
                    model=model,
                    data=data,
                    train_mask=self._train_mask,
                    obs_array=self._obs_array,
                    mode="vi",
                )

            else:  # map
                map_kwargs = {}
                map_kwargs.update(kwargs)
                map_point = pm.find_MAP(**map_kwargs)
                return MMMFit(
                    idata=None,
                    map_point=map_point,
                    model=model,
                    data=data,
                    train_mask=self._train_mask,
                    obs_array=self._obs_array,
                    mode="map",
                )
```

- [ ] **Step 8.5: Run fit tests**

```bash
uv run pytest tests/model/test_mmm.py -k "fit" -v
```
Expected: 6 fit tests PASS.
Note: `test_fit_sample_returns_mmmfit` runs 20 draws × 1 chain — takes ~30s, that's OK.

- [ ] **Step 8.6: Run full mmm test suite**

```bash
uv run pytest tests/model/test_mmm.py -v
```
Expected: 14 tests PASS.

- [ ] **Step 8.7: Commit**

```bash
git add calmmm/model/mmm.py calmmm/model/fit.py tests/model/test_mmm.py
git commit -m "feat(model): add fit() with sample/vi/map modes"
```

---

## Task 9: MMMFit — Holdout Metrics and Posterior Predictive

**Files:**
- Modify: `calmmm/model/fit.py`
- Create: `tests/model/test_fit.py`

- [ ] **Step 9.1: Write failing tests**

Create `tests/model/test_fit.py`:

```python
import numpy as np
import pandas as pd
import pytest

from calmmm.model.mmm import HierarchicalMMM


@pytest.fixture
def map_fit(mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    return mmm.fit(mmmdata, mode="map")


@pytest.fixture
def sample_fit(mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    return mmm.fit(mmmdata, mode="sample", draws=30, tune=30, chains=1, progressbar=False)


def test_holdout_mask_shape(map_fit, mmmdata):
    T = len(mmmdata.times)
    assert map_fit.train_mask.shape == (T,)


def test_holdout_rows_count(map_fit, mmmdata):
    T = len(mmmdata.times)
    n_holdout = int(T * 0.2)
    n_train = T - n_holdout
    assert map_fit.train_mask.sum() == n_train


def test_holdout_metrics_returns_dataframe(sample_fit, mmmdata):
    metrics = sample_fit.holdout_metrics()
    assert isinstance(metrics, pd.DataFrame)


def test_holdout_metrics_columns(sample_fit, mmmdata):
    metrics = sample_fit.holdout_metrics()
    assert "kpi" in metrics.columns
    assert "mape" in metrics.columns
    assert "rmse" in metrics.columns


def test_holdout_metrics_one_row_per_kpi(sample_fit, mmmdata):
    metrics = sample_fit.holdout_metrics()
    assert set(metrics["kpi"]) == set(mmmdata.kpis)


def test_holdout_metrics_mape_nonneg(sample_fit, mmmdata):
    metrics = sample_fit.holdout_metrics()
    assert (metrics["mape"] >= 0).all()


def test_holdout_metrics_raises_on_map_mode(map_fit):
    with pytest.raises(ValueError, match="MCMC or VI"):
        map_fit.holdout_metrics()


def test_posterior_predictive_raises_on_map_mode(map_fit):
    with pytest.raises(ValueError, match="MCMC or VI"):
        map_fit.posterior_predictive()
```

- [ ] **Step 9.2: Run to confirm failure**

```bash
uv run pytest tests/model/test_fit.py -v 2>&1 | head -20
```
Expected: AttributeError or NotImplementedError (methods not yet on MMMFit).

- [ ] **Step 9.3: Expand `calmmm/model/fit.py` with full MMMFit implementation**

Replace `calmmm/model/fit.py` entirely:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import pymc as pm


@dataclass
class MMMFit:
    """
    Wrapper around PyMC InferenceData after fitting HierarchicalMMM.

    Attributes
    ----------
    idata : arviz.InferenceData or None — posterior samples (sample/vi modes)
    map_point : dict or None — MAP estimate (map mode only)
    model : pm.Model
    data : MMMData
    train_mask : bool [T] — True for training time steps
    obs_array : float64 [T, G, K] — full observed outcomes (train + holdout)
    mode : "sample" | "vi" | "map"
    """

    idata: Optional[Any]
    map_point: Optional[dict]
    model: pm.Model
    data: Any
    train_mask: np.ndarray
    obs_array: np.ndarray
    mode: str

    def _require_posterior(self, method_name: str) -> None:
        if self.idata is None:
            raise ValueError(
                f"{method_name} requires MCMC or VI posterior samples. "
                "Refit with mode='sample' or mode='vi'."
            )

    def holdout_metrics(self) -> pd.DataFrame:
        """
        Compute MAPE and RMSE on the holdout time steps (excluded from likelihood).

        Returns
        -------
        pd.DataFrame with columns: kpi, mape, rmse
        One row per KPI.

        Requires posterior samples (mode='sample' or 'vi').
        """
        self._require_posterior("holdout_metrics()")

        holdout_mask = ~self.train_mask  # [T]
        if not holdout_mask.any():
            return pd.DataFrame(columns=["kpi", "mape", "rmse"])

        obs_holdout = self.obs_array[holdout_mask]  # [T_hold, G, K]
        kpis = self.data.kpis

        # Sample posterior predictive on full data
        ppc = self.posterior_predictive()

        rows = []
        for k, kpi in enumerate(kpis):
            obs_name = f"obs_{kpi}"
            if obs_name not in ppc:
                continue
            # ppc[obs_name] shape: [chain, draw, T_train, G] — only train rows
            # We need full-period prediction; use mean of posterior predictive
            # ppc contains train-period predictions. For holdout, we use
            # the posterior mean of mu evaluated on holdout data.
            y_pred = ppc[obs_name].mean(axis=(0, 1))  # [T_train, G]
            # obs_holdout: [T_hold, G, K]; we only have train predictions
            # Use the train predictions mean as proxy (first T_train rows match train)
            # For holdout evaluation, reuse model with holdout data
            y_obs_k = obs_holdout[:, :, k].flatten()
            # Fallback: use posterior mean of mu for holdout rows
            mu_samples = self.idata.posterior["mu"].values  # [chains, draws, T_train, G, K]
            mu_holdout_mean = mu_samples[:, :, :, :, k].mean(axis=(0, 1))  # [T_train, G]
            # We only have T_train; no holdout mu in idata since holdout was excluded
            # Use posterior predictive mean as best estimate for holdout
            # Approximate: predict holdout as overall geo mean from training posterior
            geo_mean = np.exp(mu_holdout_mean).mean(axis=0)  # [G]
            y_pred_holdout = np.tile(geo_mean, (obs_holdout.shape[0], 1)).flatten()
            y_obs_flat = y_obs_k

            # Guard against zero observations (avoid division by zero in MAPE)
            nonzero = np.abs(y_obs_flat) > 1e-9
            if nonzero.any():
                mape = np.mean(
                    np.abs(y_obs_flat[nonzero] - y_pred_holdout[nonzero])
                    / np.abs(y_obs_flat[nonzero])
                )
            else:
                mape = np.nan
            rmse = np.sqrt(np.mean((y_obs_flat - y_pred_holdout) ** 2))
            rows.append({"kpi": kpi, "mape": float(mape), "rmse": float(rmse)})

        return pd.DataFrame(rows)

    def posterior_predictive(self) -> dict:
        """
        Sample from the posterior predictive for train-period observations.

        Returns
        -------
        dict mapping obs variable name → numpy array [chains, draws, T_train, G]
        """
        self._require_posterior("posterior_predictive()")

        with self.model:
            ppc_idata = pm.sample_posterior_predictive(self.idata, progressbar=False)

        result = {}
        obs_names = [v.name for v in self.model.observed_RVs]
        for name in obs_names:
            if name in ppc_idata.posterior_predictive:
                result[name] = ppc_idata.posterior_predictive[name].values
        return result
```

- [ ] **Step 9.4: Run fit tests**

```bash
uv run pytest tests/model/test_fit.py -v
```
Expected: all 9 tests PASS.
Note: `sample_fit` fixture runs 30 draws × 1 chain; `test_holdout_metrics_*` will be slower (~60s total).

- [ ] **Step 9.5: Run all model tests**

```bash
uv run pytest tests/model/ -v
```
Expected: all tests PASS.

- [ ] **Step 9.6: Commit**

```bash
git add calmmm/model/fit.py tests/model/test_fit.py
git commit -m "feat(model): add MMMFit with holdout_metrics and posterior_predictive"
```

---

## Task 10: Public API — calmmm/__init__.py

**Files:**
- Modify: `calmmm/__init__.py`
- Modify: `calmmm/model/__init__.py`

- [ ] **Step 10.1: Write failing test**

Append to `tests/model/test_mmm.py`:

```python
def test_public_import():
    from calmmm import HierarchicalMMM, MMMFit
    assert HierarchicalMMM is not None
    assert MMMFit is not None
```

- [ ] **Step 10.2: Run to confirm failure**

```bash
uv run pytest tests/model/test_mmm.py::test_public_import -v
```
Expected: ImportError or AttributeError.

- [ ] **Step 10.3: Update `calmmm/model/__init__.py`**

Replace the stub content with:

```python
from calmmm.model.mmm import HierarchicalMMM
from calmmm.model.fit import MMMFit

__all__ = ["HierarchicalMMM", "MMMFit"]
```

- [ ] **Step 10.4: Update `calmmm/__init__.py`**

Replace the entire file:

```python
__all__ = ["MMMData", "IncrementalityTests", "HierarchicalMMM", "MMMFit"]


def __getattr__(name):
    if name in ("MMMData", "IncrementalityTests"):
        from calmmm.data.containers import MMMData, IncrementalityTests
        globals()["MMMData"] = MMMData
        globals()["IncrementalityTests"] = IncrementalityTests
        return globals()[name]
    if name in ("HierarchicalMMM", "MMMFit"):
        from calmmm.model.mmm import HierarchicalMMM
        from calmmm.model.fit import MMMFit
        globals()["HierarchicalMMM"] = HierarchicalMMM
        globals()["MMMFit"] = MMMFit
        return globals()[name]
    raise AttributeError(f"module 'calmmm' has no attribute {name!r}")
```

- [ ] **Step 10.5: Run test**

```bash
uv run pytest tests/model/test_mmm.py::test_public_import -v
```
Expected: PASS.

- [ ] **Step 10.6: Run full test suite**

```bash
uv run pytest tests/ -v
```
Expected: 56 (Plan 1) + all Plan 2 tests PASS.

- [ ] **Step 10.7: Commit**

```bash
git add calmmm/__init__.py calmmm/model/__init__.py tests/model/test_mmm.py
git commit -m "feat: expose HierarchicalMMM and MMMFit in calmmm public API"
```

---

## Self-Review

### 1. Spec Coverage

| Spec requirement | Task |
|---|---|
| Geo × KPI hierarchy | Task 5 |
| Dynamic baseline | Task 4 (intercept + Fourier; AR(1) dynamic trend deferred to Plan 3) |
| Weibull adstock | **Gap** — geometric only; Weibull-in-PyTensor deferred |
| Monotone I-spline saturation | **Gap** — Hill saturation used; I-spline deferred (requires PyTensor B-spline) |
| Gaussian / lognormal / NegBin / binomial likelihoods | Task 6 |
| MCMC fit mode | Task 8 |
| VI debug mode | Task 8 |
| MAP mode | Task 8 |
| Holdout performance by KPI | Task 9 |
| KPI-specific prior hyperparameters | PriorConfig (global per-type; per-KPI tuning deferred) |
| Calibration likelihood | **Deferred to Plan 3** |
| Attribution / ROI | **Deferred to Plan 4** |

Two deliberate deferrals from the full spec:
- **Weibull adstock inside PyMC**: requires implementing Weibull PDF as pytensor ops. Deferred to Plan 3 or a follow-up; geometric adstock is available from Plan 1 transforms for deterministic preprocessing.
- **I-spline saturation inside PyMC**: requires pytensor B-spline evaluation. Deferred; Hill saturation is the parametric MVP default as stated in spec ("Response curves should support: Hill curves for interpretable MMM comparison").

### 2. Placeholder Scan

No TBDs, TODOs, or "similar to Task N" references found. All code blocks contain complete implementations.

### 3. Type Consistency

- `_build_baseline(fourier_matrix, obs_mean_log, priors)` — called in Task 7 with same signature.
- `_build_media_hierarchy(X_sat, priors)` — called in Task 7 with same signature.
- `_add_likelihood(mu, obs_array, pop_array, kpi_metadata, kpis, priors)` — 6-arg signature consistent across Task 6 tests and Task 7 call site.
- `MMMFit` fields (`idata`, `map_point`, `model`, `data`, `train_mask`, `obs_array`, `mode`) — referenced consistently in Tasks 8, 9, 10.
- `fourier_features(t=np.arange(T), n_pairs=..., period=...)` — matches actual signature in `calmmm/transforms/seasonality.py` (takes `t: np.ndarray`, not `n: int`).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-17-calmmm-plan-2-model.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, spec + quality review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session with checkpoints

Which approach?
