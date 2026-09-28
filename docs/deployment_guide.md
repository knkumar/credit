# calmmm Deployment Guide

This guide covers running `calmmm` in a production environment — containerised batch jobs, dependency management, environment-specific configuration, and observability.

---

## 1. Environment requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| pymc | ≥ 5.0, < 6 |
| pytensor | ≥ 2.18, < 3 |
| numpy | ≥ 1.24 |
| pandas | ≥ 2.0 |
| scipy | ≥ 1.10 |
| arviz | ≥ 0.16 |
| numba | < 0.61 (llvmlite build constraint) |

PyMC can compile PyTensor graphs using a C compiler at runtime. The repository image sets `PYTENSOR_FLAGS=cxx=` so the minimal image does not need a compiler. Extend the image with `gcc` and `g++`, then override `PYTENSOR_FLAGS`, when compiled PyTensor operations are required for production throughput.

When C compilation is enabled, PyTensor caches compiled operations under `~/.pytensor` or the `compiledir` set in `PYTENSOR_FLAGS`. Mount that directory on a persistent volume to avoid compiling the same graph for each batch run.

---

## 2. Containerisation

The checked-in `Dockerfile` has two targets:

- `test` installs the `dev` dependency group from `pyproject.toml` and runs the fast pytest suite.
- `runtime` installs production dependencies and uses `scripts/run_demo_fit.py` as its entry point.

Build and test in a clean environment:

```bash
docker build --target test -t calmmm-test .
docker run --rm calmmm-test
```

Run the bundled demo and retain its outputs on the host:

```bash
docker build --target runtime -t calmmm:local .
docker run --rm \
  -v "$PWD/artifacts:/app/artifacts" \
  -v "$PWD/reporting:/app/reporting" \
  calmmm:local --mode map
```

The entry point accepts the same arguments as `scripts/run_demo_fit.py`. The valid inference modes are `map`, `vi`, and `sample`.

---

## 3. Dependency management

Use `uv` with the lockfile to guarantee reproducible installs:

```bash
# Install exact locked versions
uv sync --frozen

# Upgrade all deps and regenerate lockfile
uv lock --upgrade
uv sync --frozen
```

Do not install project dependencies in production without the lockfile. The `numba<0.61` pin is intentional because `llvmlite>=0.44` fails to build from source on many Linux hosts; do not remove it without testing.

---

## 4. Batch job design

A typical production MMM job runs on a weekly schedule:

```
fetch_data → fit_model → compute_attribution → write_outputs → alert_on_divergence
```

### Production wrapper example

```python
# This application-owned wrapper is not part of the calmmm package.
import os
import pandas as pd
from calmmm import MMMData, HierarchicalMMM, IncrementalityTests
from calmmm.attribution.roi import compute_roi
from calmmm.calibration.lift import compute_model_lift

def run():
    mode = os.environ.get("CALMMM_MODE", "sample")
    data_path = os.environ["CALMMM_DATA_PATH"]
    output_path = os.environ["CALMMM_OUTPUT_PATH"]
    experiments_path = os.environ.get("CALMMM_EXPERIMENTS_PATH")

    # 1. Load data
    df = pd.read_csv(data_path)
    data = MMMData.from_dataframe(df, ...)

    # 2. Load experiments (optional)
    exps = None
    if experiments_path:
        exps_df = pd.read_csv(experiments_path)
        exps = IncrementalityTests.from_dataframe(exps_df, ..., mmmdata=data)

    # 3. Fit
    mmm = HierarchicalMMM(holdout_fraction=0.2)
    if mode == "sample":
        inference_kwargs = {
            "draws": 2000,
            "tune": 1000,
            "target_accept": 0.9,
            "chains": 4,
        }
    elif mode == "vi":
        inference_kwargs = {"n": 10_000, "draws": 1000}
    elif mode == "map":
        inference_kwargs = {"maxeval": 2000}
    else:
        raise ValueError(f"Unsupported CALMMM_MODE: {mode}")
    fit = mmm.fit(
        data,
        experiments=exps,
        mode=mode,
        **inference_kwargs,
    )

    # 4. Save trace
    if fit.trace is not None:
        fit.trace.to_netcdf(f"{output_path}/trace.nc")

    # 5. Attribution
    roi = compute_roi(fit)
    roi.to_csv(f"{output_path}/roi.csv", index=False)

    # 6. Holdout metrics
    metrics = fit.holdout_metrics()
    pd.Series(metrics).to_json(f"{output_path}/holdout_metrics.json")

    # 7. Calibration check
    if fit.calibration_targets:
        lift_df = compute_model_lift(fit, fit.calibration_targets)
        lift_df.to_csv(f"{output_path}/calibration_check.csv", index=False)
        _alert_on_calibration_failure(lift_df)

def _alert_on_calibration_failure(lift_df, z_threshold=2.5):
    bad = lift_df[lift_df["z_score"].abs() > z_threshold]
    if not bad.empty:
        # Replace with your alerting mechanism (PagerDuty, Slack, etc.)
        raise RuntimeError(
            f"Calibration z-score exceeded {z_threshold} for experiments: "
            f"{bad['test_id'].tolist()}"
        )

if __name__ == "__main__":
    run()
```

---

## 5. MCMC-specific configuration

For full posterior sampling in production, tune these parameters:

| Parameter | Recommended | Notes |
|---|---|---|
| `draws` | 1000–2000 | Total post-warmup samples per chain |
| `tune` | 1000 | Warmup steps for NUTS adaptation |
| `chains` | 4 | Parallelise with `cores=4` |
| `target_accept` | 0.9 | Increase to 0.95 if divergences > 0.5 % |
| `progressbar` | `False` | Suppress in batch jobs |

#### Divergence check

```python
divergences = fit.trace.sample_stats["diverging"].sum().item()
if divergences > 0:
    # Log a warning — do not silently ignore
    print(f"WARNING: {divergences} divergent transitions")
```

Divergences indicate geometry problems. Common causes: too-wide priors on `hill_k`, near-zero spend channels, or multimodality in the posterior. Address by tightening priors or removing near-zero channels.

#### Convergence check

```python
import arviz as az

summary = az.summary(fit.trace, var_names=["adstock_decay", "hill_alpha", "hill_k"])
bad_rhat = summary[summary["r_hat"] > 1.05]
if not bad_rhat.empty:
    raise RuntimeError(f"Poor convergence: {bad_rhat.index.tolist()}")
```

---

## 6. Storage and artifact management

| Artifact | Format | Notes |
|---|---|---|
| Posterior trace | `.nc` (NetCDF via ArviZ) | `fit.trace.to_netcdf(path)` / `az.from_netcdf(path)` |
| MAP parameters | `.npz` or `.pkl` | `np.savez(path, **fit.map_params)` |
| ROI table | `.csv` | One row per KPI and channel |
| Holdout metrics | `.json` | Lightweight; easy to ingest into a metrics store |
| Calibration check | `.csv` | One row per experiment per run |

Store artifacts in versioned paths:

```
s3://your-bucket/mmm/
  2026-06-21/
    trace.nc
    roi.csv
    holdout_metrics.json
    calibration_check.csv
  latest -> 2026-06-21/   # symlink or redirect
```

---

## 7. Memory and compute

Full MCMC on a typical marketing dataset (104 weeks × 10 geos × 3 channels × 2 KPIs) with 4 chains × 2000 draws:

- **RAM**: 4–8 GB. PyTensor compiles the full graph per chain; each chain holds its own trace in memory during sampling.
- **CPU**: one core per chain. A 4-chain run benefits from 4 physical cores.
- **Wall time**: 20–90 min depending on geometry complexity and step-size adaptation.

For MAP fitting (quick iteration, calibration checks): < 5 min, < 2 GB RAM.

Recommended instance class: `c6i.2xlarge` (AWS) or equivalent — 8 vCPU / 16 GB RAM.

---

## 8. Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PYTENSOR_FLAGS` | — | Override compile dir: `compiledir=/mnt/pytensor_cache` |
| `CALMMM_MODE` | `sample` | Inference mode passed to `fit()` |
| `CALMMM_DATA_PATH` | — | Path to input CSV |
| `CALMMM_EXPERIMENTS_PATH` | — | Path to experiments CSV (optional) |
| `CALMMM_OUTPUT_PATH` | — | Directory for output artifacts |
| `PYTHONFAULTHANDLER` | `1` | Enable fault handler for segfault debugging |

Set `PYTENSOR_FLAGS="cxx="` to disable C compilation entirely (CPU-only, slower) — useful in environments without a C toolchain (CI, restricted containers).

---

## 9. CI/CD

### Recommended pipeline stages

The checked-in GitHub Actions workflow builds the `test` target, runs the fast suite, builds the `runtime` target, and runs an eight-week MAP inference smoke job. Use `pytest -m slow` as a scheduled or pre-release extension when the full inference suite is needed.

### Synthetic smoke-test data

```python
import numpy as np, pandas as pd
from calmmm import MMMData

T, G = 52, 3
rng = np.random.default_rng(42)
df = pd.DataFrame({
    "week": pd.date_range("2024-01-01", periods=T, freq="W") \
              .repeat(G),
    "region": [f"g{i}" for i in range(G)] * T,
    "revenue": rng.poisson(10000, T * G).astype(float),
    "tv_spend": rng.exponential(1000, T * G),
    "search_spend": rng.exponential(500, T * G),
})
data = MMMData.from_dataframe(
    df, time="week", geo="region",
    kpis=["revenue"], media=["tv", "search"],
    spend=["tv_spend", "search_spend"],
    kpi_likelihoods={"revenue": "gaussian"},
)
from calmmm import HierarchicalMMM
fit = HierarchicalMMM(holdout_fraction=0.0).fit(data, mode="map")
assert fit.map_params is not None
```

---

## 10. Observability checklist

- [ ] Log `holdout_metrics` values per run to a time-series store (DataDog, Prometheus, CloudWatch).
- [ ] Alert if `rmse_{kpi}` increases by > 20 % week-over-week.
- [ ] Log divergences and r-hat violations for MCMC runs.
- [ ] Alert if calibration `z_score` > 2.5 for any experiment.
- [ ] Store trace artifacts for at least 90 days to enable retrospective debugging.
- [ ] Record git SHA and dataset date range in artifact metadata.
