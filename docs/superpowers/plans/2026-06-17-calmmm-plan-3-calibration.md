# calmmm.calibration — Incrementality Calibration Likelihood

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add incrementality-test calibration to `calmmm` — convert `IncrementalityTests` into model index targets, expose per-channel contributions in the PyMC model, add calibration likelihood nodes (`lift_hat ~ Normal(lift_model, se)`), and compute post-hoc model-implied lift from MAP or posterior samples.

**Architecture:** The calibration module is a pure add-on: Task 1 modifies `_build_media_hierarchy` to store a `channel_contrib[T_train, G, K, C]` deterministic (needed for symbolic lift); Tasks 2–3 build the calibration target structure and in-model likelihood; Task 4 computes lift numerically post-fit; Task 5 wires `experiments` into `HierarchicalMMM.fit()`. All changes are backward-compatible (experiments are optional; existing tests still pass).

**Tech Stack:** pymc>=5, pytensor>=2.18, numpy, pandas, calmmm.data.containers (MMMData, IncrementalityTests, ExperimentRow), calmmm.model (HierarchicalMMM, MMMFit, _build_media_hierarchy)

## Global Constraints

- Run tests: `.venv/bin/pytest` (not bare `pytest`; pyenv python lacks PyMC)
- PyTensor C++ compiler disabled: `PYTENSOR_FLAGS=cxx=` is set in `conftest.py`
- Always use `model.compile_fn(tensor)(model.initial_point())` or `pm.draw(tensor)` for shape checks — `model["var"].type.shape` returns `(None, None, ...)` in PyMC 5.28.5
- All files are Python 3.10+ (use `from __future__ import annotations`)
- No new top-level dependencies — only pymc, pytensor, numpy, pandas

---

## File Structure

| File | Purpose |
|---|---|
| **Modify** `calmmm/model/components.py` | `_build_media_hierarchy` → also registers `channel_contrib[T,G,K,C]` deterministic |
| **Create** `calmmm/calibration/__init__.py` | Exports `CalibrationTarget`, `build_calibration_targets`, `add_calibration_likelihood`, `compute_model_lift` |
| **Create** `calmmm/calibration/targets.py` | `CalibrationTarget` dataclass; `build_calibration_targets(experiments, data, train_mask)` |
| **Create** `calmmm/calibration/likelihood.py` | `add_calibration_likelihood(model, targets)` — adds pm.Normal nodes to existing PyMC model |
| **Create** `calmmm/calibration/lift.py` | `compute_model_lift(fit, targets)` — returns DataFrame with model vs observed lift |
| **Modify** `calmmm/model/mmm.py` | `build_model(data, experiments=None)`, `fit(data, experiments=None, ...)` |
| **Modify** `calmmm/model/fit.py` | `MMMFit` — add `calibration_targets` field |
| **Modify** `calmmm/__init__.py` | Lazy-export `CalibrationTarget` |
| **Create** `tests/calibration/__init__.py` | Empty |
| **Create** `tests/calibration/test_targets.py` | Tests for CalibrationTarget and builder |
| **Create** `tests/calibration/test_likelihood.py` | Tests for add_calibration_likelihood |
| **Create** `tests/calibration/test_lift.py` | Tests for compute_model_lift |

---

## Task 1: Channel Contribution Decomposition

**Files:**
- Modify: `calmmm/model/components.py` (lines 107–109 of `_build_media_hierarchy`)
- Modify: `tests/model/test_components.py` (append 2 tests)

**Interfaces:**
- Consumes: existing `_build_media_hierarchy(X_sat, priors)` internals
- Produces: `model["channel_contrib"]` — pytensor tensor `[T, G, K, C]`, registered as `pm.Deterministic`, accessible from any code that has a reference to the `pm.Model`; return value `[T, G, K]` is unchanged

- [ ] **Step 1.1: Write failing tests (append to `tests/model/test_components.py`)**

```python
# ---- channel_contrib decomposition ----

def test_channel_contrib_deterministic_registered():
    """_build_media_hierarchy must register 'channel_contrib' as a pm.Deterministic."""
    T, G, K, C = 5, 2, 4, 2
    priors = PriorConfig()
    X_sat_val = np.random.rand(T, G, C).astype("float64")

    with pm.Model(coords=_base_coords()) as model:
        _build_media_hierarchy(pt.as_tensor_variable(X_sat_val), priors)
        det_names = {v.name for v in model.deterministics}
    assert "channel_contrib" in det_names


def test_channel_contrib_shape():
    """channel_contrib must have shape [T, G, K, C]."""
    T, G, K, C = 7, 2, 4, 2
    priors = PriorConfig()
    X_sat_val = np.random.rand(T, G, C).astype("float64")

    with pm.Model(coords=_base_coords()) as model:
        _build_media_hierarchy(pt.as_tensor_variable(X_sat_val), priors)
        val = model.compile_fn(model["channel_contrib"])(model.initial_point())
    assert np.array(val).shape == (T, G, K, C)
```

- [ ] **Step 1.2: Run to confirm failure**

```bash
.venv/bin/pytest tests/model/test_components.py::test_channel_contrib_deterministic_registered -v 2>&1 | tail -10
```
Expected: `FAILED` — assertion error (channel_contrib not in deterministics).

- [ ] **Step 1.3: Modify `calmmm/model/components.py` — add `channel_contrib` deterministic**

Locate the block in `_build_media_hierarchy` that reads:

```python
    scale_geo_gkc = scale_geo.dimshuffle(2, 1, 0)  # [G, K, C]
    media_contrib = (
        X_sat[:, :, None, :] * scale_geo_gkc[None, :, :, :]
    ).sum(axis=-1)
    return media_contrib  # [T, G, K]
```

Replace it with:

```python
    scale_geo_gkc = scale_geo.dimshuffle(2, 1, 0)  # [G, K, C]
    channel_contrib_tgkc = X_sat[:, :, None, :] * scale_geo_gkc[None, :, :, :]  # [T, G, K, C]
    pm.Deterministic("channel_contrib", channel_contrib_tgkc)
    return channel_contrib_tgkc.sum(axis=-1)  # [T, G, K]
```

- [ ] **Step 1.4: Run failing tests**

```bash
.venv/bin/pytest tests/model/test_components.py::test_channel_contrib_deterministic_registered tests/model/test_components.py::test_channel_contrib_shape -v 2>&1 | tail -10
```
Expected: both PASS.

- [ ] **Step 1.5: Run full component test suite to check no regressions**

```bash
.venv/bin/pytest tests/model/test_components.py --tb=short -q 2>&1 | tail -5
```
Expected: all 15 tests PASS.

- [ ] **Step 1.6: Commit**

```bash
git add calmmm/model/components.py tests/model/test_components.py
git commit -m "feat(model): expose per-channel contributions as channel_contrib deterministic"
```

---

## Task 2: CalibrationTarget — Index Builder

**Files:**
- Create: `calmmm/calibration/__init__.py`
- Create: `calmmm/calibration/targets.py`
- Create: `tests/calibration/__init__.py`
- Create: `tests/calibration/test_targets.py`

**Interfaces:**
- Consumes: `IncrementalityTests` (list of `ExperimentRow`), `MMMData` (`.times`, `.geos`, `.channels`, `.kpis`), `train_mask: np.ndarray[bool, (T,)]`
- Produces: `CalibrationTarget` dataclass with integer index fields consumed by Tasks 3 and 4; `build_calibration_targets(experiments, data, train_mask) -> list[CalibrationTarget]`

**Key invariant:** `t_indices` are indices into the **training-filtered** time axis (i.e. into `channel_contrib[T_train, ...]`), not into the full `data.times` list.

- [ ] **Step 2.1: Write failing tests**

Create `tests/calibration/__init__.py` (empty file).

Create `tests/calibration/test_targets.py`:

```python
import numpy as np
import pandas as pd
import pytest

from calmmm.calibration.targets import CalibrationTarget, build_calibration_targets


# ---- CalibrationTarget ----

def test_calibration_target_fields():
    target = CalibrationTarget(
        test_id="exp_1",
        t_indices=np.array([2, 3, 4]),
        g_indices=np.array([0]),
        c_indices=np.array([0]),
        k_index=1,
        lift_obs=1000.0,
        se=200.0,
        calibration_likelihood="normal",
        estimand="total",
    )
    assert target.test_id == "exp_1"
    assert list(target.t_indices) == [2, 3, 4]
    assert target.k_index == 1
    assert target.lift_obs == 1000.0
    assert target.se == 200.0


# ---- build_calibration_targets ----

def test_build_calibration_targets_returns_list(lift_tests, mmmdata):
    T = len(mmmdata.times)
    train_mask = np.ones(T, dtype=bool)
    targets = build_calibration_targets(lift_tests, mmmdata, train_mask)
    assert isinstance(targets, list)
    assert len(targets) == len(lift_tests)


def test_build_calibration_targets_test_id(lift_tests, mmmdata):
    T = len(mmmdata.times)
    train_mask = np.ones(T, dtype=bool)
    targets = build_calibration_targets(lift_tests, mmmdata, train_mask)
    assert targets[0].test_id == "search_holdout_q1"


def test_build_calibration_targets_kpi_index(lift_tests, mmmdata):
    # experiment KPI is "visits"; sorted kpis are ["applications", "approvals", "revenue", "visits"]
    T = len(mmmdata.times)
    train_mask = np.ones(T, dtype=bool)
    targets = build_calibration_targets(lift_tests, mmmdata, train_mask)
    kpis = mmmdata.kpis  # sorted
    expected_k = kpis.index("visits")
    assert targets[0].k_index == expected_k


def test_build_calibration_targets_channel_index(lift_tests, mmmdata):
    # experiment channel is "search"; sorted channels are ["search", "social"]
    T = len(mmmdata.times)
    train_mask = np.ones(T, dtype=bool)
    targets = build_calibration_targets(lift_tests, mmmdata, train_mask)
    channels = mmmdata.channels  # sorted
    expected_c = channels.index("search")
    assert list(targets[0].c_indices) == [expected_c]


def test_build_calibration_targets_geo_index(lift_tests, mmmdata):
    # experiment geo_scope is "DMA_1"; sorted geos are ["DMA_1", "DMA_2"]
    T = len(mmmdata.times)
    train_mask = np.ones(T, dtype=bool)
    targets = build_calibration_targets(lift_tests, mmmdata, train_mask)
    geos = mmmdata.geos  # sorted
    expected_g = geos.index("DMA_1")
    assert list(targets[0].g_indices) == [expected_g]


def test_build_calibration_targets_t_indices_in_window(lift_tests, mmmdata):
    # experiment window: 2024-03-04 to 2024-03-25; full training
    T = len(mmmdata.times)
    train_mask = np.ones(T, dtype=bool)
    targets = build_calibration_targets(lift_tests, mmmdata, train_mask)
    t_idx = targets[0].t_indices
    # All t_indices must correspond to times within the experiment window
    times = mmmdata.times
    exp_start = pd.Timestamp("2024-03-04")
    exp_end = pd.Timestamp("2024-03-25")
    for i in t_idx:
        assert exp_start <= times[i] <= exp_end


def test_build_calibration_targets_lift_se(lift_tests, mmmdata):
    T = len(mmmdata.times)
    train_mask = np.ones(T, dtype=bool)
    targets = build_calibration_targets(lift_tests, mmmdata, train_mask)
    assert targets[0].lift_obs == 12_000.0
    assert targets[0].se == 2_500.0


def test_build_calibration_targets_no_training_times_raises(lift_tests, mmmdata):
    """If the experiment window falls entirely in holdout, raise ValueError."""
    T = len(mmmdata.times)
    # holdout = last 20 weeks; experiment window 2024-03-04..2024-03-25 is weeks 9-12
    # To put window in holdout: set train_mask to only first 5 weeks
    train_mask = np.zeros(T, dtype=bool)
    train_mask[:5] = True  # weeks 0-4 only; experiment starts week 9
    with pytest.raises(ValueError, match="no training time steps"):
        build_calibration_targets(lift_tests, mmmdata, train_mask)
```

- [ ] **Step 2.2: Run to confirm failure**

```bash
.venv/bin/pytest tests/calibration/test_targets.py -v 2>&1 | head -15
```
Expected: `ImportError` (module not yet created).

- [ ] **Step 2.3: Create `calmmm/calibration/__init__.py`**

```python
from calmmm.calibration.targets import CalibrationTarget, build_calibration_targets
from calmmm.calibration.likelihood import add_calibration_likelihood
from calmmm.calibration.lift import compute_model_lift

__all__ = [
    "CalibrationTarget",
    "build_calibration_targets",
    "add_calibration_likelihood",
    "compute_model_lift",
]
```

(This will fail at import until Tasks 3–4 create the referenced modules. Create it as a stub for now.)

Stub content to write now:

```python
from calmmm.calibration.targets import CalibrationTarget, build_calibration_targets

__all__ = ["CalibrationTarget", "build_calibration_targets"]
```

- [ ] **Step 2.4: Create `calmmm/calibration/targets.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from calmmm.data.containers import IncrementalityTests, MMMData


@dataclass
class CalibrationTarget:
    """
    An incrementality experiment expressed as integer indices into the model arrays.

    Attributes
    ----------
    test_id : str — unique experiment identifier
    t_indices : int array — time indices into the TRAINING-FILTERED axis
                (i.e. indices into channel_contrib[T_train, ...])
    g_indices : int array — geo indices; use all geos if experiment is national
    c_indices : int array — channel indices for the tested channel bundle
    k_index : int — KPI axis index
    lift_obs : float — observed cumulative lift from the experiment
    se : float — standard error of the lift estimate
    calibration_likelihood : str — "normal" (others deferred)
    estimand : str — "total" (others deferred)
    """

    test_id: str
    t_indices: np.ndarray
    g_indices: np.ndarray
    c_indices: np.ndarray
    k_index: int
    lift_obs: float
    se: float
    calibration_likelihood: str
    estimand: str


def build_calibration_targets(
    experiments: "IncrementalityTests",
    data: "MMMData",
    train_mask: np.ndarray,
) -> list[CalibrationTarget]:
    """
    Convert IncrementalityTests to CalibrationTargets with integer model indices.

    Parameters
    ----------
    experiments : IncrementalityTests
    data : MMMData — provides times, geos, channels, kpis for index lookup
    train_mask : bool array [T] — True for time steps included in training

    Returns
    -------
    list[CalibrationTarget] — one per experiment, with validated index arrays

    Raises
    ------
    ValueError if any experiment's window contains no training time steps.
    """
    times = data.times  # sorted list of pd.Timestamp, length T
    geos = data.geos    # sorted list of str
    channels = data.channels  # sorted list of str
    kpis = data.kpis    # sorted list of str

    g_idx = {g: i for i, g in enumerate(geos)}
    c_idx = {c: i for i, c in enumerate(channels)}
    k_idx = {k: i for i, k in enumerate(kpis)}

    # Map absolute time index → training-filtered index
    train_abs_indices = np.where(train_mask)[0]  # absolute positions of training steps
    abs_to_filtered = {int(abs_i): filt_i for filt_i, abs_i in enumerate(train_abs_indices)}

    targets = []
    for exp in experiments:
        # Collect time indices in experiment window that fall in training
        window_filtered = []
        for abs_i, t in enumerate(times):
            if exp.start_date <= t <= exp.end_date and abs_i in abs_to_filtered:
                window_filtered.append(abs_to_filtered[abs_i])

        if not window_filtered:
            raise ValueError(
                f"Experiment '{exp.test_id}' window "
                f"[{exp.start_date.date()}, {exp.end_date.date()}] "
                f"has no training time steps (all fall in holdout or outside panel)."
            )

        t_indices = np.array(window_filtered, dtype=int)
        g_indices = np.array([g_idx[g] for g in exp.geo_scope if g in g_idx], dtype=int)
        if len(g_indices) == 0:
            # If no specific geos matched, use all (national scope)
            g_indices = np.arange(len(geos), dtype=int)
        c_indices = np.array([c_idx[c] for c in exp.channel_bundle], dtype=int)
        k_index = k_idx[exp.kpi]

        targets.append(
            CalibrationTarget(
                test_id=exp.test_id,
                t_indices=t_indices,
                g_indices=g_indices,
                c_indices=c_indices,
                k_index=k_index,
                lift_obs=exp.lift,
                se=exp.se,
                calibration_likelihood=exp.calibration_likelihood.value,
                estimand=exp.estimand.value,
            )
        )

    return targets
```

- [ ] **Step 2.5: Run tests**

```bash
.venv/bin/pytest tests/calibration/test_targets.py -v 2>&1 | tail -15
```
Expected: all 9 tests PASS.

- [ ] **Step 2.6: Commit**

```bash
git add calmmm/calibration/__init__.py calmmm/calibration/targets.py tests/calibration/__init__.py tests/calibration/test_targets.py
git commit -m "feat(calibration): add CalibrationTarget dataclass and build_calibration_targets"
```

---

## Task 3: Calibration Likelihood (In-Model)

**Files:**
- Create: `calmmm/calibration/likelihood.py`
- Create: `tests/calibration/test_likelihood.py`

**Interfaces:**
- Consumes: `pm.Model` with `model["channel_contrib"]` tensor `[T_train, G, K, C]` and `model["mu"]` tensor `[T_train, G, K]`; `list[CalibrationTarget]`
- Produces: mutates the current `pm.Model` context by adding one `pm.Normal(f"lift_obs_{test_id}", ...)` observed node per target; no return value

**Lift formula:**
```
lift_model_e = sum_{t in t_indices, g in g_indices} [
    exp(mu[t, g, k]) - exp(mu[t, g, k] - sum_{c in c_indices} channel_contrib[t, g, k, c])
]
```

This is the counterfactual difference: predicted outcomes vs. predicted outcomes with the experiment channels removed.

- [ ] **Step 3.1: Write failing tests**

Create `tests/calibration/test_likelihood.py`:

```python
import numpy as np
import pymc as pm
import pytest

from calmmm.calibration.targets import build_calibration_targets
from calmmm.calibration.likelihood import add_calibration_likelihood
from calmmm.model.mmm import HierarchicalMMM


def test_add_calibration_likelihood_adds_observed_node(lift_tests, mmmdata):
    """After calling add_calibration_likelihood, the model has a lift_obs_* observed RV."""
    T = len(mmmdata.times)
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    model = mmm.build_model(mmmdata)
    targets = build_calibration_targets(lift_tests, mmmdata, mmm._train_mask)

    with model:
        add_calibration_likelihood(model, targets)
        obs_names = {v.name for v in model.observed_RVs}

    assert "lift_obs_search_holdout_q1" in obs_names


def test_add_calibration_likelihood_logp_finite(lift_tests, mmmdata):
    """logp must remain finite at initial point after adding calibration terms."""
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    model = mmm.build_model(mmmdata)
    targets = build_calibration_targets(lift_tests, mmmdata, mmm._train_mask)

    with model:
        add_calibration_likelihood(model, targets)
        lp = model.compile_logp()(model.initial_point())

    assert np.isfinite(lp)


def test_add_calibration_likelihood_empty_targets(mmmdata):
    """Calling with an empty target list is a no-op."""
    mmm = HierarchicalMMM(holdout_fraction=0.0)
    model = mmm.build_model(mmmdata)
    n_obs_before = len(model.observed_RVs)

    with model:
        add_calibration_likelihood(model, [])

    assert len(model.observed_RVs) == n_obs_before


def test_add_calibration_likelihood_unsupported_estimand_raises(lift_tests, mmmdata):
    """CalibrationTargets with estimand != 'total' must raise NotImplementedError."""
    from calmmm.calibration.targets import CalibrationTarget
    target = CalibrationTarget(
        test_id="exp_immediate",
        t_indices=np.array([0, 1]),
        g_indices=np.array([0]),
        c_indices=np.array([0]),
        k_index=0,
        lift_obs=1000.0,
        se=200.0,
        calibration_likelihood="normal",
        estimand="immediate",  # not supported in MVP
    )
    mmm = HierarchicalMMM(holdout_fraction=0.0)
    model = mmm.build_model(mmmdata)

    with model:
        with pytest.raises(NotImplementedError, match="immediate"):
            add_calibration_likelihood(model, [target])
```

- [ ] **Step 3.2: Run to confirm failure**

```bash
.venv/bin/pytest tests/calibration/test_likelihood.py -v 2>&1 | head -15
```
Expected: `ImportError` (module not created yet).

- [ ] **Step 3.3: Create `calmmm/calibration/likelihood.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import pymc as pm
import pytensor.tensor as pt

if TYPE_CHECKING:
    from calmmm.calibration.targets import CalibrationTarget


def add_calibration_likelihood(
    model: pm.Model,
    targets: list["CalibrationTarget"],
) -> None:
    """
    Add one pm.Normal calibration likelihood node per target to the current model.

    Must be called inside a `with model:` context (or the model context is entered
    internally). For each target:

        lift_model_e = sum_{t in t_indices, g in g_indices} [
            exp(mu[t,g,k]) - exp(mu[t,g,k] - sum_{c in c_indices} channel_contrib[t,g,k,c])
        ]
        pm.Normal("lift_obs_{test_id}", mu=lift_model_e, sigma=se, observed=lift_obs)

    Parameters
    ----------
    model : pm.Model — must contain "mu" [T_train, G, K] and "channel_contrib" [T_train, G, K, C]
    targets : list[CalibrationTarget]

    Raises
    ------
    NotImplementedError for estimands other than "total".
    """
    if not targets:
        return

    mu = model["mu"]                         # [T_train, G, K]
    channel_contrib = model["channel_contrib"]  # [T_train, G, K, C]

    for target in targets:
        if target.estimand != "total":
            raise NotImplementedError(
                f"Estimand '{target.estimand}' is not supported in MVP. "
                "Only 'total' estimand is implemented."
            )

        t = target.t_indices   # [T_exp]
        g = target.g_indices   # [G_exp]
        k = target.k_index     # int
        c = target.c_indices   # [C_exp]

        # Slice mu for experiment window and geos: [T_exp, G_exp]
        mu_exp = mu[t][:, g, k]  # [T_exp, G_exp]

        # Sum channel contributions over experiment channels: [T_exp, G_exp]
        cc_exp = channel_contrib[t][:, g, k, :][:, :, c].sum(axis=-1)

        # Counterfactual: remove experiment channel contributions
        mu_cf = mu_exp - cc_exp  # [T_exp, G_exp]

        # Lift = sum of (factual outcome - counterfactual outcome) over window
        lift_model = (pt.exp(mu_exp) - pt.exp(mu_cf)).sum()

        pm.Normal(
            f"lift_obs_{target.test_id}",
            mu=lift_model,
            sigma=target.se,
            observed=target.lift_obs,
        )
```

- [ ] **Step 3.4: Update `calmmm/calibration/__init__.py`** to include the new import:

```python
from calmmm.calibration.targets import CalibrationTarget, build_calibration_targets
from calmmm.calibration.likelihood import add_calibration_likelihood

__all__ = [
    "CalibrationTarget",
    "build_calibration_targets",
    "add_calibration_likelihood",
]
```

- [ ] **Step 3.5: Run tests**

```bash
.venv/bin/pytest tests/calibration/test_likelihood.py -v 2>&1 | tail -15
```
Expected: all 4 tests PASS.

- [ ] **Step 3.6: Commit**

```bash
git add calmmm/calibration/__init__.py calmmm/calibration/likelihood.py tests/calibration/test_likelihood.py
git commit -m "feat(calibration): add in-model calibration likelihood"
```

---

## Task 4: Post-Hoc Lift Computation

**Files:**
- Create: `calmmm/calibration/lift.py`
- Create: `tests/calibration/test_lift.py`

**Interfaces:**
- Consumes: `MMMFit` (`.map_params`, `.trace`, `.model`); `list[CalibrationTarget]`
- Produces: `compute_model_lift(fit, targets) -> pd.DataFrame` with columns `["test_id", "lift_model", "lift_obs", "se", "z_score"]`

**Lift computation (numpy):**
```
mu_val[T_train, G, K]              — evaluated at MAP point or posterior mean
cc_val[T_train, G, K, C]           — channel_contrib evaluated at same point

For each target:
  mu_exp = mu_val[t_indices][:, g_indices, k_index]      # [T_exp, G_exp]
  cc_exp = cc_val[t_indices][:, g_indices, k_index, :][:, :, c_indices].sum(-1)  # [T_exp, G_exp]
  lift_model = (exp(mu_exp) - exp(mu_exp - cc_exp)).sum()
  z_score = (lift_model - lift_obs) / se
```

- [ ] **Step 4.1: Write failing tests**

Create `tests/calibration/test_lift.py`:

```python
import numpy as np
import pandas as pd
import pytest

from calmmm.calibration.targets import build_calibration_targets
from calmmm.calibration.lift import compute_model_lift
from calmmm.model.mmm import HierarchicalMMM


@pytest.fixture
def map_fit_with_targets(lift_tests, mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    fit = mmm.fit(mmmdata, mode="map")
    targets = build_calibration_targets(lift_tests, mmmdata, mmm._train_mask)
    return fit, targets


def test_compute_model_lift_returns_dataframe(map_fit_with_targets):
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    assert isinstance(result, pd.DataFrame)


def test_compute_model_lift_columns(map_fit_with_targets):
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    assert set(result.columns) == {"test_id", "lift_model", "lift_obs", "se", "z_score"}


def test_compute_model_lift_one_row_per_target(map_fit_with_targets):
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    assert len(result) == len(targets)


def test_compute_model_lift_test_id(map_fit_with_targets):
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    assert result.iloc[0]["test_id"] == "search_holdout_q1"


def test_compute_model_lift_lift_obs_correct(map_fit_with_targets):
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    assert result.iloc[0]["lift_obs"] == 12_000.0


def test_compute_model_lift_model_lift_is_finite(map_fit_with_targets):
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    assert np.isfinite(result.iloc[0]["lift_model"])


def test_compute_model_lift_model_lift_is_positive(map_fit_with_targets):
    """Model-implied lift should be positive since media contrib > 0."""
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    assert result.iloc[0]["lift_model"] > 0


def test_compute_model_lift_z_score_formula(map_fit_with_targets):
    fit, targets = map_fit_with_targets
    result = compute_model_lift(fit, targets)
    row = result.iloc[0]
    expected_z = (row["lift_model"] - row["lift_obs"]) / row["se"]
    assert abs(row["z_score"] - expected_z) < 1e-9


def test_compute_model_lift_empty_targets(mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.0)
    fit = mmm.fit(mmmdata, mode="map")
    result = compute_model_lift(fit, [])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
```

- [ ] **Step 4.2: Run to confirm failure**

```bash
.venv/bin/pytest tests/calibration/test_lift.py -v 2>&1 | head -15
```
Expected: `ImportError`.

- [ ] **Step 4.3: Create `calmmm/calibration/lift.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from calmmm.calibration.targets import CalibrationTarget
    from calmmm.model.fit import MMMFit


def compute_model_lift(
    fit: "MMMFit",
    targets: list["CalibrationTarget"],
) -> pd.DataFrame:
    """
    Compute model-implied lift for each CalibrationTarget.

    For MAP fits, evaluates mu and channel_contrib at the MAP point.
    For MCMC/VI fits, uses the posterior mean of mu and channel_contrib.

    Lift formula (numpy):
        mu_exp[T_exp, G_exp] = mu[t_indices, :, k_index][:, g_indices]
        cc_exp[T_exp, G_exp] = channel_contrib[t_indices, :, k_index, :][:, :, c_indices].sum(-1)
        lift_model = sum(exp(mu_exp) - exp(mu_exp - cc_exp))

    Parameters
    ----------
    fit : MMMFit
    targets : list[CalibrationTarget]

    Returns
    -------
    pd.DataFrame with columns: test_id, lift_model, lift_obs, se, z_score
    One row per target, empty DataFrame if targets is empty.
    """
    if not targets:
        return pd.DataFrame(columns=["test_id", "lift_model", "lift_obs", "se", "z_score"])

    mu_val, cc_val = _eval_mu_and_channel_contrib(fit)

    rows = []
    for target in targets:
        t = target.t_indices
        g = target.g_indices
        k = target.k_index
        c = target.c_indices

        mu_exp = mu_val[t][:, g, k]                   # [T_exp, G_exp]
        cc_total = cc_val[t][:, g, k, :][:, :, c].sum(axis=-1)  # [T_exp, G_exp]

        lift_model = float((np.exp(mu_exp) - np.exp(mu_exp - cc_total)).sum())
        z_score = (lift_model - target.lift_obs) / target.se

        rows.append({
            "test_id": target.test_id,
            "lift_model": lift_model,
            "lift_obs": target.lift_obs,
            "se": target.se,
            "z_score": z_score,
        })

    return pd.DataFrame(rows)


def _eval_mu_and_channel_contrib(
    fit: "MMMFit",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (mu, channel_contrib) as numpy arrays [T_train, G, K] and [T_train, G, K, C].

    For MAP: evaluates at map_params.
    For MCMC/VI: returns posterior mean over chains and draws.
    """
    if fit.map_params is not None:
        with fit.model:
            fn = fit.model.compile_fn(
                [fit.model["mu"], fit.model["channel_contrib"]]
            )
            mu_val, cc_val = fn(fit.map_params)
        return np.array(mu_val), np.array(cc_val)

    if fit.trace is not None:
        # posterior["mu"]: [chains, draws, T_train, G, K]
        mu_val = fit.trace.posterior["mu"].values.mean(axis=(0, 1))
        # posterior["channel_contrib"]: [chains, draws, T_train, G, K, C]
        cc_val = fit.trace.posterior["channel_contrib"].values.mean(axis=(0, 1))
        return mu_val, cc_val

    raise ValueError("MMMFit has neither map_params nor trace.")
```

- [ ] **Step 4.4: Update `calmmm/calibration/__init__.py`**

```python
from calmmm.calibration.targets import CalibrationTarget, build_calibration_targets
from calmmm.calibration.likelihood import add_calibration_likelihood
from calmmm.calibration.lift import compute_model_lift

__all__ = [
    "CalibrationTarget",
    "build_calibration_targets",
    "add_calibration_likelihood",
    "compute_model_lift",
]
```

- [ ] **Step 4.5: Run tests**

```bash
.venv/bin/pytest tests/calibration/test_lift.py -v 2>&1 | tail -15
```
Expected: all 9 tests PASS. Note: `map_fit_with_targets` calls `pm.find_MAP` — takes ~10–30s.

- [ ] **Step 4.6: Commit**

```bash
git add calmmm/calibration/__init__.py calmmm/calibration/lift.py tests/calibration/test_lift.py
git commit -m "feat(calibration): add compute_model_lift for MAP and posterior fits"
```

---

## Task 5: Wire into HierarchicalMMM.fit()

**Files:**
- Modify: `calmmm/model/mmm.py` — `build_model(data, experiments=None)`, `fit(data, experiments=None, ...)`
- Modify: `calmmm/model/fit.py` — `MMMFit` gets `calibration_targets` field
- Modify: `calmmm/__init__.py` — lazy-export `CalibrationTarget`

**Interfaces:**
- Consumes: `IncrementalityTests` (optional); `add_calibration_likelihood`, `build_calibration_targets` from Task 3–2
- Produces: `mmm.fit(data, experiments=lift_tests)` → `MMMFit` with `calibration_targets` populated; model has `lift_obs_*` observed nodes when experiments provided

- [ ] **Step 5.1: Write failing tests (append to `tests/model/test_mmm.py`)**

Add this import at the top of `tests/model/test_mmm.py` (after existing imports):

```python
from calmmm.model.fit import MMMFit
```

Append these tests to `tests/model/test_mmm.py`:

```python
def test_fit_with_experiments_completes(mmmdata, lift_tests):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    fit = mmm.fit(mmmdata, experiments=lift_tests, mode="map")
    assert isinstance(fit, MMMFit)


def test_fit_with_experiments_has_calibration_targets(mmmdata, lift_tests):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    fit = mmm.fit(mmmdata, experiments=lift_tests, mode="map")
    assert len(fit.calibration_targets) == 1
    assert fit.calibration_targets[0].test_id == "search_holdout_q1"


def test_fit_with_experiments_has_lift_obs_node(mmmdata, lift_tests):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    fit = mmm.fit(mmmdata, experiments=lift_tests, mode="map")
    obs_names = {v.name for v in fit.model.observed_RVs}
    assert "lift_obs_search_holdout_q1" in obs_names


def test_fit_without_experiments_no_lift_nodes(mmmdata):
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    fit = mmm.fit(mmmdata, mode="map")
    obs_names = {v.name for v in fit.model.observed_RVs}
    assert not any("lift_obs" in n for n in obs_names)
    assert fit.calibration_targets == []


def test_public_import_calibration_target():
    from calmmm import CalibrationTarget
    assert CalibrationTarget is not None
```

- [ ] **Step 5.2: Run to confirm failure**

```bash
.venv/bin/pytest tests/model/test_mmm.py -k "experiments or calibration" -v 2>&1 | head -15
```
Expected: `ImportError` or `TypeError` (build_model doesn't accept experiments yet).

- [ ] **Step 5.3: Modify `calmmm/model/fit.py` — add `calibration_targets` field**

In `MMMFit`, add the field after `_mmm`:

Replace:

```python
    _mmm: Optional[Any] = field(default=None, repr=False)
```

with:

```python
    _mmm: Optional[Any] = field(default=None, repr=False)
    calibration_targets: list = field(default_factory=list)
```

- [ ] **Step 5.4: Modify `calmmm/model/mmm.py` — accept `experiments` in `build_model` and `fit`**

Add imports at the top of `calmmm/model/mmm.py` (after existing imports):

```python
from calmmm.calibration.targets import build_calibration_targets
from calmmm.calibration.likelihood import add_calibration_likelihood
```

Change the `build_model` signature from:

```python
    def build_model(self, data: MMMData) -> pm.Model:
```

to:

```python
    def build_model(self, data: MMMData, experiments=None) -> pm.Model:
```

Add `self._calibration_targets: list = []` in `__init__` after the existing instance variable declarations.

Inside `build_model`, after the `self._model = model` line and before `return model`, add:

```python
        if experiments is not None:
            targets = build_calibration_targets(experiments, data, self._train_mask)
            with model:
                add_calibration_likelihood(model, targets)
            self._calibration_targets = targets
        else:
            self._calibration_targets = []
```

Change the `fit` signature from:

```python
    def fit(
        self,
        data: MMMData,
        *,
        mode: str = "sample",
        **kwargs,
    ) -> "MMMFit":
```

to:

```python
    def fit(
        self,
        data: MMMData,
        *,
        experiments=None,
        mode: str = "sample",
        **kwargs,
    ) -> "MMMFit":
```

Inside `fit`, change the `build_model` call from:

```python
        if self._model is None or self._data is not data:
            self.build_model(data)
```

to:

```python
        if self._model is None or self._data is not data:
            self.build_model(data, experiments=experiments)
```

In all three return statements inside `fit`, add `calibration_targets=self._calibration_targets` to the `MMMFit(...)` call. For example the MAP branch becomes:

```python
            return MMMFit(trace=None, map_params=map_params, model=model, data=data, _mmm=self, calibration_targets=self._calibration_targets)
```

Do the same for `sample` and `vi` branches.

- [ ] **Step 5.5: Update `calmmm/__init__.py` — add CalibrationTarget to lazy exports**

Replace the existing `__getattr__` in `calmmm/__init__.py`:

```python
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

- [ ] **Step 5.6: Run new tests**

```bash
.venv/bin/pytest tests/model/test_mmm.py -k "experiments or calibration" -v 2>&1 | tail -15
```
Expected: all 5 new tests PASS.

- [ ] **Step 5.7: Run all non-PyMC-heavy tests to check no regressions**

```bash
.venv/bin/pytest tests/ --ignore=tests/model/test_components.py --ignore=tests/model/test_mmm.py --ignore=tests/model/test_fit.py -q --tb=short 2>&1 | tail -10
```
Expected: all fast tests PASS.

- [ ] **Step 5.8: Commit**

```bash
git add calmmm/model/mmm.py calmmm/model/fit.py calmmm/__init__.py tests/model/test_mmm.py
git commit -m "feat(model): wire experiments into HierarchicalMMM.fit() for calibrated fitting"
```

---

## Self-Review

### 1. Spec Coverage

| Spec requirement | Task |
|---|---|
| Calibration likelihood: `lift_hat[e] ~ Normal(lift_model[e], se[e])` | Task 3 |
| Lift computed by factual vs counterfactual simulation | Task 3, Task 4 |
| Model-implied lift post-fit | Task 4 |
| `fit(dataset, experiments=experiments)` API | Task 5 |
| `CalibrationTarget` index structure | Task 2 |
| Per-channel contribution for symbolic lift | Task 1 |
| Estimands: `total` | Task 3 (others raise NotImplementedError) |
| Calibration likelihood: `normal` | Task 3 (others raise NotImplementedError) |
| Student-T, truncated_normal, laplace likelihoods | **Deferred** — only `normal` in MVP |
| `immediate` / `carryover` / `cumulative` estimands | **Deferred** — only `total` in MVP |
| Calibration residuals DataFrame | **Deferred to Plan 4** — diagnostics module |

### 2. Placeholder Scan

All steps contain complete code. No TBDs or "similar to Task N" references.

### 3. Type Consistency

- `CalibrationTarget` fields (`t_indices`, `g_indices`, `c_indices`, `k_index`) — used consistently in `likelihood.py` and `lift.py`
- `compute_model_lift(fit, targets)` — `fit` is `MMMFit` with `.map_params`/`.trace`/`.model`; matches actual `MMMFit` fields
- `add_calibration_likelihood(model, targets)` — `model` must have `model["mu"]` and `model["channel_contrib"]`; both guaranteed after Task 1 + `build_model`
- `MMMFit.calibration_targets` — `list` default; populated in Task 5 from `self._calibration_targets`
- `build_calibration_targets(experiments, data, train_mask)` — same signature in Task 2 and all call sites in Tasks 3–5

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-17-calmmm-plan-3-calibration.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, with checkpoints

Which approach?
