import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from calmmm.attribution.curves import spend_response_report


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_demo_fit.py"
    spec = importlib.util.spec_from_file_location("run_demo_fit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_parser_is_demo_first():
    script = _load_script()

    args = script.parse_args([])

    assert args.mode == "map"
    assert args.weeks == 16
    assert args.maxeval == 300
    assert args.holdout_fraction == 0.2
    assert args.adjust_lift_windows is True
    assert args.reporting_dir == Path("reporting")
    assert args.spend_multiplier == 1.10


def test_select_week_subset_keeps_complete_weeks():
    script = _load_script()
    panel = pd.DataFrame(
        {
            "week": ["2024-01-08", "2024-01-01", "2024-01-08", "2024-01-15"],
            "geo": ["A", "A", "B", "A"],
            "value": [1, 2, 3, 4],
        }
    )

    selected = script.select_week_subset(panel, weeks=2)

    assert selected["week"].tolist() == ["2024-01-01", "2024-01-08", "2024-01-08"]


def test_adjust_lift_windows_moves_experiments_inside_training_window():
    script = _load_script()
    panel = pd.DataFrame(
        {
            "week": pd.date_range("2024-01-01", periods=16, freq="W-MON").repeat(2),
            "geo": ["A", "B"] * 16,
        }
    )
    lift_tests = pd.DataFrame(
        [
            {
                "test_id": "exp_1",
                "start_date": "2025-01-01",
                "end_date": "2025-01-29",
            },
            {
                "test_id": "exp_2",
                "start_date": "2025-02-01",
                "end_date": "2025-02-28",
            },
        ]
    )

    adjusted = script.adjust_lift_windows(
        lift_tests,
        panel,
        holdout_fraction=0.2,
        window_weeks=4,
    )

    assert adjusted["start_date"].tolist() == ["2024-02-05", "2024-03-04"]
    assert adjusted["end_date"].tolist() == ["2024-02-26", "2024-03-25"]


def test_script_uses_attribution_spend_response_report():
    script = _load_script()

    assert script.spend_response_report is spend_response_report


def test_demo_treats_applications_as_gaussian():
    script = _load_script()

    panel = pd.read_csv(script.DEFAULT_PANEL)
    panel = script.select_week_subset(panel, weeks=2)
    data = script.build_data(panel)

    likelihoods = data.kpi_metadata.set_index("kpi")["likelihood"].to_dict()
    assert likelihoods["applications"] == "gaussian"
    assert likelihoods["funded_revenue"] == "gaussian"


def test_write_outputs_includes_fit_quality_and_mcmc_diagnostics(tmp_path, monkeypatch):
    script = _load_script()
    output_dir = tmp_path / "artifacts"
    reporting_dir = tmp_path / "reporting"
    args = SimpleNamespace(
        mode="map",
        output_dir=output_dir,
        reporting_dir=reporting_dir,
        spend_multiplier=1.10,
    )
    panel = pd.DataFrame(
        {
            "week": ["2024-01-01", "2024-01-08"],
            "geo": ["A", "A"],
            "search_spend": [100.0, 110.0],
            "social_spend": [50.0, 55.0],
            "direct_mail_spend": [10.0, 11.0],
            "affiliate_spend": [5.0, 6.0],
        }
    )
    lift_tests = pd.DataFrame({"test_id": ["exp_1"]})
    fit = SimpleNamespace(
        data=SimpleNamespace(channels=["search"]),
        calibration_targets=[],
        map_params={"mu": 1},
        fit_metrics=lambda: {"rmse_applications": 12.0, "r2_applications": 0.82},
        mcmc_diagnostics=lambda: pd.DataFrame(
            {"parameter": ["adstock_decay[search]"], "r_hat": [1.01], "ess_bulk": [250.0], "ess_tail": [200.0]}
        ),
    )

    monkeypatch.setattr(script, "compute_roi", lambda _fit: pd.DataFrame({"roi": [1.2]}))
    monkeypatch.setattr(
        script,
        "compute_model_lift",
        lambda _fit, _targets: pd.DataFrame({"test_id": ["exp_1"], "z_score": [0.2]}),
    )
    monkeypatch.setattr(
        script,
        "channel_contributions",
        lambda _fit: pd.DataFrame({"channel": ["search"], "contribution": [10.0]}),
    )
    monkeypatch.setattr(
        script,
        "saturation_curve",
        lambda _fit, channel, n_points: pd.DataFrame(
            {"spend": [0.0], "saturation": [0.0], "channel": [channel]}
        ),
    )
    monkeypatch.setattr(
        script,
        "spend_response_report",
        lambda *_args, **_kwargs: pd.DataFrame({"channel": ["search"], "response_lift": [0.1]}),
    )

    script.write_outputs(args=args, panel=panel, lift_tests=lift_tests, fit=fit)

    fit_quality = pd.read_csv(output_dir / "fit_quality.csv")
    diagnostics = pd.read_csv(output_dir / "mcmc_diagnostics.csv")
    assert set(fit_quality.columns) == {"metric", "kpi", "window", "value"}
    assert {"rmse_applications", "r2_applications"} == set(fit_quality["metric"])
    assert set(fit_quality["window"]) == {"train"}
    assert diagnostics.loc[0, "parameter"] == "adstock_decay[search]"
    assert (output_dir / "fit_summary.json").read_text().find("fit_quality") != -1
    assert (output_dir / "fit_summary.json").read_text().find("mcmc_diagnostics") != -1
