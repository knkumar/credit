from pathlib import Path

import pandas as pd

from calmmm.reporting.visualization import build_summary_table, render_reporting_outputs


def _write_inputs(reporting_dir: Path, artifacts_dir: Path) -> None:
    reporting_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "channel": ["search", "social", "direct_mail"],
            "spend_multiplier": [1.1, 1.1, 1.1],
            "current_spend": [100.0, 50.0, 500.0],
            "increased_spend": [110.0, 55.0, 550.0],
            "current_response": [0.40, 0.25, 0.55],
            "increased_response": [0.44, 0.27, 0.56],
            "response_lift": [0.04, 0.02, 0.01],
            "response_lift_pct": [0.10, 0.08, 0.018],
        }
    ).to_csv(reporting_dir / "spend_response.csv", index=False)
    pd.DataFrame(
        {
            "spend": [0.0, 100.0, 0.0, 50.0],
            "saturation": [0.0, 0.8, 0.0, 0.5],
            "channel": ["search", "search", "social", "social"],
        }
    ).to_csv(reporting_dir / "saturation_curves.csv", index=False)
    pd.DataFrame(
        {
            "kpi": ["applications", "funded_revenue"],
            "channel": ["search", "affiliate"],
            "total_contribution": [1000.0, 500000.0],
            "total_spend": [600.0, 40000.0],
            "roi": [1.67, 12.5],
        }
    ).to_csv(artifacts_dir / "roi.csv", index=False)
    pd.DataFrame(
        {
            "test_id": ["exp_1", "direct_mail_match_q3"],
            "lift_model": [120.0, 278200.0],
            "lift_obs": [100.0, 780000.0],
            "se": [20.0, 175000.0],
            "z_score": [1.0, -2.87],
        }
    ).to_csv(artifacts_dir / "calibration_fit.csv", index=False)


def test_build_summary_table_describes_reports(tmp_path):
    reporting_dir = tmp_path / "reporting"
    artifacts_dir = tmp_path / "artifacts"
    _write_inputs(reporting_dir, artifacts_dir)

    summary = build_summary_table(reporting_dir=reporting_dir, artifacts_dir=artifacts_dir)

    assert set(summary["report"]) == {
        "spend_response",
        "saturation_curves",
        "roi",
        "calibration_fit",
    }
    assert summary.set_index("report").loc["spend_response", "rows"] == 3
    assert "response_lift_pct" in summary.set_index("report").loc["spend_response", "metric_columns"]


def test_render_reporting_outputs_writes_visuals_and_summary(tmp_path):
    reporting_dir = tmp_path / "reporting"
    artifacts_dir = tmp_path / "artifacts"
    _write_inputs(reporting_dir, artifacts_dir)

    outputs = render_reporting_outputs(reporting_dir=reporting_dir, artifacts_dir=artifacts_dir)

    expected = {
        reporting_dir / "summary_table.csv",
        reporting_dir / "spend_response.svg",
        reporting_dir / "saturation_curves.svg",
        reporting_dir / "roi.svg",
        reporting_dir / "calibration_fit.svg",
    }
    assert set(outputs) == expected
    for path in expected:
        assert path.exists()
        assert path.stat().st_size > 0


def test_render_reporting_outputs_labels_plot_units(tmp_path):
    reporting_dir = tmp_path / "reporting"
    artifacts_dir = tmp_path / "artifacts"
    _write_inputs(reporting_dir, artifacts_dir)

    render_reporting_outputs(reporting_dir=reporting_dir, artifacts_dir=artifacts_dir)

    assert "10% spend increase" in (reporting_dir / "spend_response.svg").read_text()
    assert "Response change (percentage points)" in (reporting_dir / "spend_response.svg").read_text()
    assert "Spend ($)" in (reporting_dir / "saturation_curves.svg").read_text()
    assert "Saturation (%)" in (reporting_dir / "saturation_curves.svg").read_text()
    assert "ROI (KPI units per $1 spend)" in (reporting_dir / "roi.svg").read_text()
    assert "Lift (KPI units)" in (reporting_dir / "calibration_fit.svg").read_text()


def test_render_reporting_outputs_keeps_labels_and_ticks_readable(tmp_path):
    reporting_dir = tmp_path / "reporting"
    artifacts_dir = tmp_path / "artifacts"
    _write_inputs(reporting_dir, artifacts_dir)

    render_reporting_outputs(reporting_dir=reporting_dir, artifacts_dir=artifacts_dir)

    roi_svg = (reporting_dir / "roi.svg").read_text()
    calibration_svg = (reporting_dir / "calibration_fit.svg").read_text()
    saturation_svg = (reporting_dir / "saturation_curves.svg").read_text()

    assert "funded_revenue / affiliate" in roi_svg
    assert "funded_revenue / affiliat..." not in roi_svg
    assert "observed: 780,000" in calibration_svg
    assert "direct_mail_match_q3" in calibration_svg
    assert "$0" in saturation_svg
    assert "50%" in saturation_svg
    assert "0% = no response; 100% = fully saturated" in saturation_svg
