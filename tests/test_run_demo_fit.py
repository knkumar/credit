import importlib.util
from pathlib import Path

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
    assert args.maxeval == 50
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
