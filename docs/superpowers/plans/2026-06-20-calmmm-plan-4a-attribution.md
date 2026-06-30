# calmmm.attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `calmmm.attribution` — channel contribution decomposition, ROI, and saturation curves from a fitted `MMMFit` object.

**Architecture:** Three focused modules (`contributions.py`, `roi.py`, `curves.py`) under `calmmm/attribution/`. All functions accept a `MMMFit` and return a `pd.DataFrame`. MAP and trace fits are both supported; MAP is the common path. One small addition to `HierarchicalMMM` stores `_media_max` during `build_model()` (needed by `saturation_curve`).

**Tech Stack:** NumPy, pandas, PyMC 5.28.5, PYTENSOR_FLAGS=cxx= (C compiler disabled — never change).

## Global Constraints

- Run tests with `.venv/bin/pytest` (NOT bare `pytest` — system Python lacks PyMC)
- Never change `PYTENSOR_FLAGS=cxx=` — it is set in `tests/conftest.py` via `os.environ.setdefault`
- All tests that call `pm.find_MAP` / `pm.sample` / `pm.fit` MUST be decorated `@pytest.mark.slow`
- Fast tests run by default (`addopts = "-m 'not slow'"` in pyproject.toml); slow tests run with `-m slow`
- No `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>` in any commit message
- Preserve existing code style — no docstring blocks, no trailing summaries

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `calmmm/model/mmm.py` | Add `_media_max` attribute; store during `build_model()` |
| Create | `calmmm/attribution/__init__.py` | Public API exports |
| Create | `calmmm/attribution/contributions.py` | `channel_contributions(fit) → pd.DataFrame` |
| Create | `calmmm/attribution/roi.py` | `compute_roi(fit) → pd.DataFrame` |
| Create | `calmmm/attribution/curves.py` | `saturation_curve(fit, channel, n_points=50) → pd.DataFrame` |
| Create | `tests/attribution/__init__.py` | Empty |
| Create | `tests/attribution/conftest.py` | Session-scoped `attr_map_fit` fixture (T=12, 2 geos, 1 KPI, 2 channels) |
| Create | `tests/attribution/test_contributions.py` | Tests for `channel_contributions` |
| Create | `tests/attribution/test_roi.py` | Tests for `compute_roi` |
| Create | `tests/attribution/test_curves.py` | Tests for `saturation_curve` |
| Modify | `calmmm/__init__.py` | Lazy-export `channel_contributions`, `compute_roi`, `saturation_curve` |

---

### Task 1: Store `_media_max` in `HierarchicalMMM` and implement `channel_contributions`

**Files:**
- Modify: `calmmm/model/mmm.py` (add `_media_max` attribute ~line 46 and store it ~line 76)
- Create: `calmmm/attribution/__init__.py`
- Create: `calmmm/attribution/contributions.py`
- Create: `tests/attribution/__init__.py`
- Create: `tests/attribution/conftest.py`
- Create: `tests/attribution/test_contributions.py`

**Interfaces:**
- Produces: `channel_contributions(fit: MMMFit) -> pd.DataFrame`
  - columns: `time` (pd.Timestamp), `geo` (str), `kpi` (str), `channel` (str), `contribution` (float)
  - One row per (time, geo, kpi, channel) where channel is a channel name OR `"baseline"`
  - "baseline" row = `exp(mu - channel_contrib.sum(-1))`
  - channel rows = `exp(mu) - exp(mu - cc[:,:,:,c])`
  - Only training-time steps appear (filtered by `_mmm._train_mask`)

**Context:**

`MMMFit` is defined in `calmmm/model/fit.py`:
```python
@dataclass
class MMMFit:
    trace: Optional[Any]
    map_params: Optional[dict]
    model: pm.Model
    data: MMMData
    _mmm: Optional[Any] = field(default=None, repr=False)
    calibration_targets: list = field(default_factory=list)
```

`HierarchicalMMM` is in `calmmm/model/mmm.py`. Inside `build_model()` at ~line 75:
```python
media_max = media_array.max(axis=(0, 1), keepdims=True)  # [1, 1, C]
media_scaled = media_array / np.maximum(media_max, 1e-8)
```
`media_max` is NOT currently stored as an attribute. We need to add it.

`channel_contrib` is a `pm.Deterministic` of shape `[T_train, G, K, C]` — stored in `map_params["channel_contrib"]` (pm.find_MAP includes deterministics). `mu` is shape `[T_train, G, K]`.

For trace: use `fit.trace.posterior["channel_contrib"].values.mean(axis=(0,1))` (shape `[T_train, G, K, C]`).

`data.channels` = sorted list of channel names (axis C order)
`data.kpis` = sorted list of KPI names (axis K order)
`data.geos` = sorted list of geo names (axis G order)
`data.times` = sorted list of pd.Timestamp (full T, not filtered)

Training times: `[t for t, m in zip(data.times, _mmm._train_mask) if m]`

- [ ] **Step 1: Write the failing tests**

Create `tests/attribution/__init__.py` (empty):
```python
```

Create `tests/attribution/conftest.py`:
```python
import numpy as np
import pandas as pd
import pytest
from calmmm.data.containers import MMMData
from calmmm.model.mmm import HierarchicalMMM


def _make_small_data():
    rng = np.random.default_rng(42)
    T, G, K, C = 12, 2, 1, 2
    times = pd.date_range("2024-01-01", periods=T, freq="W")
    geos = ["geo_a", "geo_b"]
    kpis = ["revenue"]
    channels = ["tv", "digital"]

    obs_rows = []
    for kpi in kpis:
        for geo in geos:
            for t in times:
                obs_rows.append({"time": t, "geo": geo, "kpi": kpi,
                                 "outcome": rng.uniform(100, 500), "population": np.nan})
    observations = pd.DataFrame(obs_rows)

    media_rows = []
    for ch in channels:
        for geo in geos:
            for t in times:
                media_rows.append({"time": t, "geo": geo, "channel": ch,
                                   "spend": rng.uniform(1, 10), "exposure": np.nan})
    media = pd.DataFrame(media_rows)

    kpi_meta = pd.DataFrame({"kpi": kpis, "likelihood": ["gaussian"],
                              "funnel_stage": ["awareness"], "family": ["normal"]})

    return MMMData(observations=observations, media=media,
                   controls=pd.DataFrame(columns=["time","geo","control","value"]),
                   kpi_metadata=kpi_meta)


@pytest.fixture(scope="session")
def attr_small_data():
    return _make_small_data()


@pytest.fixture(scope="session")
def attr_map_fit(attr_small_data):
    mmm = HierarchicalMMM(holdout_fraction=0.0)
    return mmm.fit(attr_small_data, mode="map", progressbar=False)
```

Create `tests/attribution/test_contributions.py`:
```python
import numpy as np
import pytest
from calmmm.attribution.contributions import channel_contributions


@pytest.mark.slow
def test_channel_contributions_columns(attr_map_fit):
    df = channel_contributions(attr_map_fit)
    assert set(df.columns) >= {"time", "geo", "kpi", "channel", "contribution"}


@pytest.mark.slow
def test_channel_contributions_channels_present(attr_map_fit):
    df = channel_contributions(attr_map_fit)
    fit = attr_map_fit
    expected = set(fit.data.channels) | {"baseline"}
    assert set(df["channel"].unique()) == expected


@pytest.mark.slow
def test_channel_contributions_nrows(attr_map_fit):
    df = channel_contributions(attr_map_fit)
    fit = attr_map_fit
    T_train = int(fit._mmm._train_mask.sum())
    G = len(fit.data.geos)
    K = len(fit.data.kpis)
    C = len(fit.data.channels)
    expected_rows = T_train * G * K * (C + 1)  # C channels + 1 baseline
    assert len(df) == expected_rows


@pytest.mark.slow
def test_channel_contributions_positive(attr_map_fit):
    df = channel_contributions(attr_map_fit)
    assert (df["contribution"] >= 0).all()


@pytest.mark.slow
def test_channel_contributions_baseline_plus_channels_equals_mu(attr_map_fit):
    """Sum of baseline + all channel contributions = exp(mu) for each (t,g,k)."""
    import numpy as np
    df = channel_contributions(attr_map_fit)
    total = df.groupby(["time", "geo", "kpi"])["contribution"].sum().reset_index()
    # exp(mu) over training times from map_params
    fit = attr_map_fit
    mu_val = np.array(fit.map_params["mu"])  # [T_train, G, K]
    exp_mu_flat = np.exp(mu_val).ravel()
    assert np.allclose(total["contribution"].values, exp_mu_flat, rtol=1e-4)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/pytest tests/attribution/test_contributions.py -m slow -v 2>&1 | head -30
```
Expected: ImportError or collection error (module does not exist yet).

- [ ] **Step 3: Add `_media_max` to `HierarchicalMMM`**

In `calmmm/model/mmm.py`, add to `__init__` (after `self._media_scaled: Optional[np.ndarray] = None` ~line 46):
```python
self._media_max: Optional[np.ndarray] = None
```

In `build_model()`, after the line `media_max = media_array.max(axis=(0, 1), keepdims=True)` (~line 76), add:
```python
self._media_max = media_max.squeeze()  # [C] — per-channel panel max spend
```

- [ ] **Step 4: Implement `channel_contributions`**

Create `calmmm/attribution/__init__.py`:
```python
from calmmm.attribution.contributions import channel_contributions
from calmmm.attribution.roi import compute_roi
from calmmm.attribution.curves import saturation_curve

__all__ = ["channel_contributions", "compute_roi", "saturation_curve"]
```

Create `calmmm/attribution/contributions.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd


def channel_contributions(fit) -> pd.DataFrame:
    """
    Counterfactual channel attribution from a fitted MMMFit.

    Returns a long DataFrame with columns:
        time, geo, kpi, channel, contribution

    channel is one of the model's channel names (for media) or "baseline".
    Contribution = marginal outcome attributable to that channel.
    Only training-time steps are included.
    """
    data = fit.data
    mmm = fit._mmm

    mu_val, cc_val = _eval_params(fit)
    # mu_val: [T_train, G, K]
    # cc_val: [T_train, G, K, C]

    train_mask = mmm._train_mask
    train_times = [t for t, m in zip(data.times, train_mask) if m]
    geos = data.geos
    kpis = data.kpis
    channels = data.channels

    T, G, K, C = cc_val.shape

    exp_mu = np.exp(mu_val)  # [T, G, K]
    cc_sum = cc_val.sum(axis=-1)  # [T, G, K]
    baseline = np.exp(mu_val - cc_sum)  # [T, G, K]

    rows = []

    # Baseline rows
    for ti, t in enumerate(train_times):
        for gi, g in enumerate(geos):
            for ki, k in enumerate(kpis):
                rows.append({
                    "time": t, "geo": g, "kpi": k,
                    "channel": "baseline",
                    "contribution": float(baseline[ti, gi, ki]),
                })

    # Channel rows
    for ci, ch in enumerate(channels):
        cc_c = cc_val[:, :, :, ci]  # [T, G, K]
        contrib_c = exp_mu - np.exp(mu_val - cc_c)  # [T, G, K]
        for ti, t in enumerate(train_times):
            for gi, g in enumerate(geos):
                for ki, k in enumerate(kpis):
                    rows.append({
                        "time": t, "geo": g, "kpi": k,
                        "channel": ch,
                        "contribution": float(contrib_c[ti, gi, ki]),
                    })

    return pd.DataFrame(rows, columns=["time", "geo", "kpi", "channel", "contribution"])


def _eval_params(fit):
    """Return (mu_val [T,G,K], cc_val [T,G,K,C]) as numpy arrays."""
    if fit.map_params is not None:
        return (
            np.array(fit.map_params["mu"]),
            np.array(fit.map_params["channel_contrib"]),
        )
    if fit.trace is not None:
        return (
            fit.trace.posterior["mu"].values.mean(axis=(0, 1)),
            fit.trace.posterior["channel_contrib"].values.mean(axis=(0, 1)),
        )
    raise ValueError("MMMFit has neither map_params nor trace.")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/attribution/test_contributions.py -m slow -v 2>&1 | tail -20
```
Expected: 5 tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add calmmm/model/mmm.py calmmm/attribution/__init__.py calmmm/attribution/contributions.py tests/attribution/__init__.py tests/attribution/conftest.py tests/attribution/test_contributions.py
git commit -m "feat(attribution): add channel_contributions and _media_max storage"
```

---

### Task 2: Implement `compute_roi`

**Files:**
- Create: `calmmm/attribution/roi.py`
- Create: `tests/attribution/test_roi.py`

**Interfaces:**
- Consumes: `channel_contributions(fit)` from Task 1
- Produces: `compute_roi(fit: MMMFit) -> pd.DataFrame`
  - columns: `kpi` (str), `channel` (str), `total_contribution` (float), `total_spend` (float), `roi` (float)
  - One row per (kpi, channel) pair — no "baseline" rows
  - `total_spend` = sum of `data.media["spend"]` for that channel across ALL time steps (not just training)
  - `roi = total_contribution / total_spend`

- [ ] **Step 1: Write the failing tests**

Create `tests/attribution/test_roi.py`:
```python
import numpy as np
import pytest
from calmmm.attribution.roi import compute_roi


@pytest.mark.slow
def test_compute_roi_columns(attr_map_fit):
    df = compute_roi(attr_map_fit)
    assert set(df.columns) >= {"kpi", "channel", "total_contribution", "total_spend", "roi"}


@pytest.mark.slow
def test_compute_roi_no_baseline(attr_map_fit):
    df = compute_roi(attr_map_fit)
    assert "baseline" not in df["channel"].values


@pytest.mark.slow
def test_compute_roi_nrows(attr_map_fit):
    fit = attr_map_fit
    df = compute_roi(fit)
    expected = len(fit.data.kpis) * len(fit.data.channels)
    assert len(df) == expected


@pytest.mark.slow
def test_compute_roi_total_spend_matches_data(attr_map_fit):
    fit = attr_map_fit
    df = compute_roi(fit)
    for ch in fit.data.channels:
        expected_spend = fit.data.media[fit.data.media["channel"] == ch]["spend"].sum()
        row = df[df["channel"] == ch]
        assert np.isclose(row["total_spend"].values[0], expected_spend, rtol=1e-6)


@pytest.mark.slow
def test_compute_roi_positive(attr_map_fit):
    df = compute_roi(attr_map_fit)
    assert (df["total_contribution"] > 0).all()
    assert (df["total_spend"] > 0).all()
    assert (df["roi"] > 0).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/pytest tests/attribution/test_roi.py -m slow -v 2>&1 | head -20
```
Expected: ImportError (module does not exist).

- [ ] **Step 3: Implement `compute_roi`**

Create `calmmm/attribution/roi.py`:
```python
from __future__ import annotations

import pandas as pd

from calmmm.attribution.contributions import channel_contributions


def compute_roi(fit) -> pd.DataFrame:
    """
    ROI per (KPI, channel) from a fitted MMMFit.

    Returns a DataFrame with columns:
        kpi, channel, total_contribution, total_spend, roi

    total_spend is summed over ALL data time steps (not just training).
    total_contribution is summed over training time steps.
    """
    contrib_df = channel_contributions(fit)
    contrib_df = contrib_df[contrib_df["channel"] != "baseline"]

    total_contrib = (
        contrib_df.groupby(["kpi", "channel"])["contribution"]
        .sum()
        .reset_index()
        .rename(columns={"contribution": "total_contribution"})
    )

    spend_by_channel = (
        fit.data.media.groupby("channel")["spend"]
        .sum()
        .reset_index()
        .rename(columns={"spend": "total_spend"})
    )

    merged = total_contrib.merge(spend_by_channel, on="channel", how="left")
    merged["roi"] = merged["total_contribution"] / merged["total_spend"]

    return merged[["kpi", "channel", "total_contribution", "total_spend", "roi"]].reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/attribution/test_roi.py -m slow -v 2>&1 | tail -20
```
Expected: 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add calmmm/attribution/roi.py tests/attribution/test_roi.py
git commit -m "feat(attribution): add compute_roi"
```

---

### Task 3: Implement `saturation_curve`

**Files:**
- Create: `calmmm/attribution/curves.py`
- Create: `tests/attribution/test_curves.py`

**Interfaces:**
- Consumes: `_eval_params` from `calmmm/attribution/contributions.py` (does NOT import it — reimplements param extraction inline since we need hill_alpha/hill_k not mu/cc); uses `fit._mmm._media_max` from Task 1
- Produces: `saturation_curve(fit, channel: str, n_points: int = 50) -> pd.DataFrame`
  - columns: `spend` (float), `saturation` (float), `channel` (str)
  - `spend` grid: `np.linspace(0, 2 * media_max[c], n_points)` in original spend units
  - `saturation` = Hill(spend/media_max[c], alpha[c], k[c]), values in [0, 1]
  - Hill formula (numpy): `x_pow = (x/media_max)**alpha; x_pow / (x_pow + k**alpha + 1e-9)`

**Hill parameter extraction:**
- From `fit.map_params`: `hill_alpha = np.array(fit.map_params["hill_alpha"])` (shape [C])
- From `fit.trace`: `hill_alpha = fit.trace.posterior["hill_alpha"].values.mean(axis=(0,1))` (shape [C])
- `channel_idx = fit.data.channels.index(channel)` — channels are sorted

- [ ] **Step 1: Write the failing tests**

Create `tests/attribution/test_curves.py`:
```python
import numpy as np
import pytest
from calmmm.attribution.curves import saturation_curve


@pytest.mark.slow
def test_saturation_curve_columns(attr_map_fit):
    df = saturation_curve(attr_map_fit, "tv")
    assert set(df.columns) >= {"spend", "saturation", "channel"}


@pytest.mark.slow
def test_saturation_curve_n_points(attr_map_fit):
    df = saturation_curve(attr_map_fit, "tv", n_points=20)
    assert len(df) == 20


@pytest.mark.slow
def test_saturation_curve_default_n_points(attr_map_fit):
    df = saturation_curve(attr_map_fit, "tv")
    assert len(df) == 50


@pytest.mark.slow
def test_saturation_curve_spend_range(attr_map_fit):
    """Spend grid starts at 0 and ends at 2x panel max."""
    df = saturation_curve(attr_map_fit, "tv")
    assert df["spend"].iloc[0] == pytest.approx(0.0)
    expected_max = 2 * attr_map_fit._mmm._media_max[attr_map_fit.data.channels.index("tv")]
    assert df["spend"].iloc[-1] == pytest.approx(expected_max, rel=1e-5)


@pytest.mark.slow
def test_saturation_curve_saturation_in_zero_one(attr_map_fit):
    df = saturation_curve(attr_map_fit, "tv")
    assert (df["saturation"] >= 0).all()
    assert (df["saturation"] <= 1).all()


@pytest.mark.slow
def test_saturation_curve_monotone(attr_map_fit):
    """Hill curve is monotonically non-decreasing."""
    df = saturation_curve(attr_map_fit, "digital")
    diffs = np.diff(df["saturation"].values)
    assert (diffs >= -1e-9).all()


@pytest.mark.slow
def test_saturation_curve_unknown_channel_raises(attr_map_fit):
    with pytest.raises(ValueError, match="unknown channel"):
        saturation_curve(attr_map_fit, "unknown_channel")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/pytest tests/attribution/test_curves.py -m slow -v 2>&1 | head -20
```
Expected: ImportError (module does not exist).

- [ ] **Step 3: Implement `saturation_curve`**

Create `calmmm/attribution/curves.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd


def saturation_curve(fit, channel: str, n_points: int = 50) -> pd.DataFrame:
    """
    Evaluate the Hill saturation curve for one channel.

    Parameters
    ----------
    fit : MMMFit
    channel : str — must be in fit.data.channels
    n_points : int — number of spend grid points

    Returns
    -------
    DataFrame with columns: spend, saturation, channel
        spend is in original (unscaled) spend units, grid from 0 to 2×panel_max
        saturation is Hill(spend/panel_max, alpha, k), values in [0, 1]
    """
    channels = fit.data.channels
    if channel not in channels:
        raise ValueError(f"unknown channel '{channel}'. Available: {channels}")

    c_idx = channels.index(channel)
    hill_alpha, hill_k = _eval_hill_params(fit)
    alpha_c = float(hill_alpha[c_idx])
    k_c = float(hill_k[c_idx])

    media_max = fit._mmm._media_max  # [C]
    max_spend = float(media_max[c_idx])

    x = np.linspace(0.0, 2.0 * max_spend, n_points)
    x_scaled = x / max(max_spend, 1e-8)
    x_pow = np.clip(x_scaled, 0.0, None) ** alpha_c
    k_pow = k_c ** alpha_c
    saturation = x_pow / (x_pow + k_pow + 1e-9)

    return pd.DataFrame({"spend": x, "saturation": saturation, "channel": channel})


def _eval_hill_params(fit):
    """Return (hill_alpha [C], hill_k [C]) as numpy arrays."""
    if fit.map_params is not None:
        return (
            np.array(fit.map_params["hill_alpha"]),
            np.array(fit.map_params["hill_k"]),
        )
    if fit.trace is not None:
        return (
            fit.trace.posterior["hill_alpha"].values.mean(axis=(0, 1)),
            fit.trace.posterior["hill_k"].values.mean(axis=(0, 1)),
        )
    raise ValueError("MMMFit has neither map_params nor trace.")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/attribution/test_curves.py -m slow -v 2>&1 | tail -20
```
Expected: 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add calmmm/attribution/curves.py tests/attribution/test_curves.py
git commit -m "feat(attribution): add saturation_curve"
```

---

### Task 4: Wire public API and run full suite

**Files:**
- Modify: `calmmm/__init__.py` — lazy-export attribution functions
- (No new test file — verification is running the full fast suite)

**Interfaces:**
- Produces: `from calmmm import channel_contributions, compute_roi, saturation_curve` works

- [ ] **Step 1: Write a fast import test**

Add to `tests/attribution/test_contributions.py` (at the top, before the slow tests):
```python
def test_attribution_importable():
    from calmmm import channel_contributions, compute_roi, saturation_curve
    assert callable(channel_contributions)
    assert callable(compute_roi)
    assert callable(saturation_curve)
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/pytest tests/attribution/test_contributions.py::test_attribution_importable -v 2>&1 | tail -10
```
Expected: ImportError — `calmmm` does not export these yet.

- [ ] **Step 3: Update `calmmm/__init__.py`**

The file currently is:
```python
# calmmm/__init__.py
__all__ = ["MMMData", "IncrementalityTests", "HierarchicalMMM", "MMMFit", "CalibrationTarget"]


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
    if name == "CalibrationTarget":
        from calmmm.calibration.targets import CalibrationTarget
        globals()["CalibrationTarget"] = CalibrationTarget
        return CalibrationTarget
    raise AttributeError(f"module 'calmmm' has no attribute {name!r}")
```

Replace with:
```python
# calmmm/__init__.py
__all__ = [
    "MMMData", "IncrementalityTests",
    "HierarchicalMMM", "MMMFit",
    "CalibrationTarget",
    "channel_contributions", "compute_roi", "saturation_curve",
]


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
    if name == "CalibrationTarget":
        from calmmm.calibration.targets import CalibrationTarget
        globals()["CalibrationTarget"] = CalibrationTarget
        return CalibrationTarget
    if name in ("channel_contributions", "compute_roi", "saturation_curve"):
        from calmmm.attribution.contributions import channel_contributions
        from calmmm.attribution.roi import compute_roi
        from calmmm.attribution.curves import saturation_curve
        globals()["channel_contributions"] = channel_contributions
        globals()["compute_roi"] = compute_roi
        globals()["saturation_curve"] = saturation_curve
        return globals()[name]
    raise AttributeError(f"module 'calmmm' has no attribute {name!r}")
```

- [ ] **Step 4: Run the fast suite to verify no regressions**

```bash
.venv/bin/pytest -v 2>&1 | tail -20
```
Expected: all previously passing fast tests + `test_attribution_importable` PASS.

- [ ] **Step 5: Run the slow attribution suite**

```bash
.venv/bin/pytest tests/attribution/ -m slow -v 2>&1 | tail -30
```
Expected: all 17 slow attribution tests PASS.

- [ ] **Step 6: Commit**

```bash
git add calmmm/__init__.py tests/attribution/test_contributions.py
git commit -m "feat(attribution): wire public API and lazy exports"
```

---

## Spec Coverage Check

- [x] `channel_contributions(fit)` → long DataFrame with (time, geo, kpi, channel, contribution) — Task 1
- [x] "baseline" row per (t, g, k) = `exp(mu - cc.sum(-1))` — Task 1
- [x] channel rows = counterfactual `exp(mu) - exp(mu - cc[:,:,:,c])` — Task 1
- [x] Both MAP and trace supported — Task 1 (`_eval_params`)
- [x] `compute_roi(fit)` → (kpi, channel, total_contribution, total_spend, roi) — Task 2
- [x] total_spend from full data (all time steps) — Task 2
- [x] `saturation_curve(fit, channel, n_points=50)` → (spend, saturation, channel) — Task 3
- [x] spend grid 0 → 2×panel_max in original units — Task 3
- [x] saturation = Hill(spend/panel_max, alpha, k) in [0,1] — Task 3
- [x] `_media_max` stored on `HierarchicalMMM` for saturation curve — Task 1 (mmm.py modification)
- [x] Public API: `from calmmm import channel_contributions, compute_roi, saturation_curve` — Task 4
- [x] All inference tests `@pytest.mark.slow` — all tests in attribution suite
- [x] Session-scoped fixture for speed — `attr_map_fit` in conftest.py
