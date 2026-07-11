# calmmm End-to-End Workflow

This document shows how the demo and production package components connect from raw weekly marketing data through model fitting, calibration, attribution, and reporting visuals.

## Component Flow

```mermaid
flowchart TD
    panel["Weekly panel data<br/>outputs/calmmm_sample_weekly_panel.csv"]
    lift["Incrementality tests<br/>outputs/calmmm_sample_lift_tests.csv"]
    script["Demo runner<br/>scripts/run_demo_fit.py"]

    data["calmmm.data<br/>MMMData"]
    tests["calmmm.data<br/>IncrementalityTests"]
    validation["calmmm.data.validation<br/>schema checks"]

    model["calmmm.model<br/>HierarchicalMMM"]
    transforms["calmmm.model.transforms<br/>geometric adstock + Hill saturation"]
    priors["calmmm.model.priors<br/>PriorConfig"]
    calibration["calmmm.calibration<br/>CalibrationTarget + likelihood"]
    fit["calmmm.model.fit<br/>MMMFit"]

    attribution["calmmm.attribution<br/>contributions + ROI + curves"]
    reporting_csv["Reporting CSVs<br/>reporting/*.csv<br/>artifacts/demo_fit/*.csv"]
    reporting_viz["calmmm.reporting<br/>visualization module"]
    visuals["Visual outputs<br/>summary table + SVG charts"]

    panel --> script
    lift --> script
    script --> data
    script --> tests
    data --> validation
    validation --> model
    tests --> calibration
    calibration --> model
    priors --> model
    model --> transforms
    transforms --> model
    model --> fit
    fit --> attribution
    attribution --> reporting_csv
    reporting_csv --> reporting_viz
    reporting_viz --> visuals
```

## Runtime Sequence

```mermaid
sequenceDiagram
    participant User
    participant Demo as scripts/run_demo_fit.py
    participant Data as MMMData
    participant Experiments as IncrementalityTests
    participant Model as HierarchicalMMM
    participant Calibration as calibration module
    participant Fit as MMMFit
    participant Attribution as attribution module
    participant Reporting as calmmm.reporting.visualization
    participant Files as artifacts/ + reporting/

    User->>Demo: Run demo fit command
    Demo->>Files: Read sample panel CSV
    Demo->>Files: Read sample lift-test CSV
    Demo->>Data: Build time x geo x KPI x channel container
    Demo->>Experiments: Build calibrated experiment container
    Model->>Data: Validate schema and build model arrays
    Model->>Model: Scale media, create Fourier seasonality, split holdout
    Model->>Model: Apply adstock and saturation transforms
    Model->>Calibration: Build calibration targets from lift tests
    Calibration->>Model: Add calibration likelihood
    Demo->>Model: Fit with MAP, VI, or MCMC
    Model->>Fit: Return fitted object
    Demo->>Attribution: Compute ROI and contribution sample
    Demo->>Attribution: Compute saturation and spend-response reports
    Attribution->>Files: Write fit and reporting CSVs
    User->>Reporting: Run visualization module
    Reporting->>Files: Read reporting CSVs and fit summary tables
    Reporting->>Files: Write summary table and SVG visuals
```

## Component Responsibilities

| Component | Primary responsibility | Inputs | Outputs |
|---|---|---|---|
| `scripts/run_demo_fit.py` | Orchestrates the demo fit from sample data to CSV outputs. | Sample panel, sample lift tests, fit arguments. | `artifacts/demo_fit/*`, `reporting/*.csv`. |
| `calmmm.data.MMMData` | Normalizes wide or long marketing data into validated model containers. | Time, geo, KPI, media, spend, exposure, control, population columns. | Model-ready observations, media, controls, and KPI metadata. |
| `calmmm.data.IncrementalityTests` | Normalizes lift experiment rows and validates them against the MMM data. | Channel, KPI, geo scope, date window, lift, standard error. | Experiment container used for calibration. |
| `calmmm.model.HierarchicalMMM` | Builds and fits the Bayesian MMM. | `MMMData`, optional `IncrementalityTests`, priors, inference settings. | `MMMFit` with trace or MAP parameters. |
| `calmmm.model.transforms` | Applies differentiable media transformations inside the PyMC graph. | Scaled media spend and channel parameters. | Adstocked and saturated media tensors. |
| `calmmm.calibration` | Converts lift tests into model targets and adds calibration likelihood terms. | Experiment container, training mask, model contribution tensors. | Calibration likelihood and model-vs-observed lift table. |
| `calmmm.attribution` | Converts a fitted model into business-facing measurement outputs. | `MMMFit`. | Channel contributions, marginal contributions, ROI, saturation curves, spend response. |
| `calmmm.reporting.visualization` | Renders report tables and curves independently from the fit script. | Existing CSV outputs in `reporting/` and `artifacts/demo_fit/`. | `summary_table.csv` and SVG report charts. |

## Demo Commands

Run the fit and write raw report tables:

```bash
PYTENSOR_FLAGS='cxx=' uv run python scripts/run_demo_fit.py
```

Render the visuals from the generated report tables:

```bash
PYTENSOR_FLAGS='cxx=' uv run python -m calmmm.reporting.visualization
```

Expected output files:

```text
artifacts/demo_fit/
  calibration_fit.csv
  channel_contributions_sample.csv
  fit_quality.csv
  fit_summary.json
  mcmc_diagnostics.csv
  roi.csv

reporting/
  calibration_fit.svg
  roi.svg
  saturation_curves.csv
  saturation_curves.svg
  spend_response.csv
  spend_response.svg
  summary_table.csv
```

## Production Adaptation

The demo runner is intentionally thin: it reads local CSVs, constructs package objects, fits once, and writes local artifacts. A production workflow should keep the same component boundaries but replace the outer orchestration with a scheduled job:

```mermaid
flowchart LR
    source["Warehouse / feature store"] --> build["Build MMMData + IncrementalityTests"]
    build --> fit["Fit HierarchicalMMM"]
    fit --> checks["Holdout + calibration checks"]
    fit --> measure["Attribution + ROI + response curves"]
    checks --> store["Versioned artifacts"]
    measure --> store
    store --> report["Dashboards / reporting exports"]
```

Keep fitting, attribution, and visualization as separate steps so model outputs can be audited before business-facing reports are published.
