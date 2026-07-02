#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from calmmm import MMMData, IncrementalityTests, HierarchicalMMM
from calmmm import channel_contributions, compute_roi, saturation_curve
from calmmm.attribution.curves import spend_response_report
from calmmm.calibration.lift import compute_model_lift


DEFAULT_PANEL = Path("outputs/calmmm_sample_weekly_panel.csv")
DEFAULT_LIFT_TESTS = Path("outputs/calmmm_sample_lift_tests.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/demo_fit")
DEFAULT_REPORTING_DIR = Path("reporting")
SPEND_COLUMNS = {
    "search": "search_spend",
    "social": "social_spend",
    "direct_mail": "direct_mail_spend",
    "affiliate": "affiliate_spend",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fast calmmm demo fit and write fit outputs.",
    )
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--lift-tests", type=Path, default=DEFAULT_LIFT_TESTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reporting-dir", type=Path, default=DEFAULT_REPORTING_DIR)
    parser.add_argument(
        "--spend-multiplier",
        type=float,
        default=1.10,
        help="Spend scenario multiplier for reporting/spend_response.csv.",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=0,
        help="Number of earliest complete weeks to fit. Use 0 for the full panel.",
    )
    parser.add_argument("--mode", choices=["map", "vi", "sample"], default="map")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--maxeval", type=int, default=2000, help="MAP maxeval.")
    parser.add_argument("--draws", type=int, default=200, help="MCMC draws.")
    parser.add_argument("--tune", type=int, default=200, help="MCMC tuning steps.")
    parser.add_argument("--chains", type=int, default=1, help="MCMC chains.")
    parser.add_argument("--vi-iterations", type=int, default=5_000)
    parser.add_argument(
        "--adjust-lift-windows",
        dest="adjust_lift_windows",
        action="store_true",
        help="Move sample lift-test windows into the training period (for use with --weeks subsets whose range excludes the tests' real dates).",
    )
    parser.add_argument(
        "--no-adjust-lift-windows",
        dest="adjust_lift_windows",
        action="store_false",
        help="Use the lift tests' real dates (default; requires the training window to cover them).",
    )
    parser.set_defaults(adjust_lift_windows=False)
    return parser.parse_args(argv)


def select_week_subset(panel: pd.DataFrame, weeks: int) -> pd.DataFrame:
    if weeks <= 0:
        return panel.copy()

    selected_weeks = sorted(pd.to_datetime(panel["week"]).unique())[:weeks]
    selected = panel[pd.to_datetime(panel["week"]).isin(selected_weeks)].copy()
    return selected.sort_values(["week", "geo"]).reset_index(drop=True)


def adjust_lift_windows(
    lift_tests: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    holdout_fraction: float,
    window_weeks: int = 4,
) -> pd.DataFrame:
    adjusted = lift_tests.copy()
    weeks = sorted(pd.to_datetime(panel["week"]).unique())
    if not weeks:
        raise ValueError("panel has no weeks")

    n_holdout = int(len(weeks) * holdout_fraction)
    train_weeks = weeks[: len(weeks) - n_holdout] if n_holdout else weeks
    if len(train_weeks) < window_weeks:
        raise ValueError(
            f"need at least {window_weeks} training weeks after holdout; "
            f"got {len(train_weeks)}"
        )

    starts = [min(5 + i * window_weeks, len(train_weeks) - window_weeks) for i in range(len(adjusted))]
    for row_idx, start_idx in enumerate(starts):
        start = pd.Timestamp(train_weeks[start_idx])
        end = pd.Timestamp(train_weeks[start_idx + window_weeks - 1])
        adjusted.loc[adjusted.index[row_idx], "start_date"] = start.strftime("%Y-%m-%d")
        adjusted.loc[adjusted.index[row_idx], "end_date"] = end.strftime("%Y-%m-%d")
    return adjusted


def build_data(panel: pd.DataFrame) -> MMMData:
    return MMMData.from_dataframe(
        panel,
        time="week",
        geo="geo",
        kpis=["applications", "funded_revenue"],
        media=["search", "social", "direct_mail", "affiliate"],
        spend=[
            "search_spend",
            "social_spend",
            "direct_mail_spend",
            "affiliate_spend",
        ],
        exposure=[
            "search_clicks",
            "social_impressions",
            "direct_mail_pieces",
            "affiliate_clicks",
        ],
        controls=["price_index", "approval_rate_proxy"],
        population="population",
        kpi_likelihoods={
            "applications": "gaussian",
            "funded_revenue": "gaussian",
        },
    )


def build_experiments(lift_tests: pd.DataFrame, data: MMMData) -> IncrementalityTests:
    return IncrementalityTests.from_dataframe(
        lift_tests,
        channel="channel",
        kpi="kpi",
        geo_scope="geo_scope",
        start="start_date",
        end="end_date",
        lift="incremental_outcome",
        standard_error="se",
        mmmdata=data,
    )


def fit_kwargs(args: argparse.Namespace) -> dict:
    if args.mode == "map":
        return {"progressbar": False, "maxeval": args.maxeval}
    if args.mode == "vi":
        return {"progressbar": False, "n": args.vi_iterations}
    return {
        "progressbar": False,
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
    }


def fit_quality_table(metrics: dict[str, float], *, window: str = "train") -> pd.DataFrame:
    rows = []
    for name, value in metrics.items():
        metric, kpi = name.split("_", 1)
        rows.append({"metric": name, "kpi": kpi, "window": window, "value": value})
    return pd.DataFrame(rows)


def write_outputs(
    *,
    args: argparse.Namespace,
    panel: pd.DataFrame,
    lift_tests: pd.DataFrame,
    fit,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reporting_dir.mkdir(parents=True, exist_ok=True)

    roi = compute_roi(fit)
    calibration = compute_model_lift(fit, fit.calibration_targets)
    fit_quality = fit_quality_table(fit.fit_metrics(), window="train")
    diagnostics = fit.mcmc_diagnostics()
    contrib_sample = channel_contributions(fit).head(250)
    curves = pd.concat(
        [saturation_curve(fit, channel=channel, n_points=100) for channel in fit.data.channels],
        ignore_index=True,
    )
    response_report = spend_response_report(
        fit,
        panel,
        spend_columns=SPEND_COLUMNS,
        spend_multiplier=args.spend_multiplier,
        n_points=100,
    )

    roi.to_csv(args.output_dir / "roi.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_fit.csv", index=False)
    fit_quality.to_csv(args.output_dir / "fit_quality.csv", index=False)
    diagnostics.to_csv(args.output_dir / "mcmc_diagnostics.csv", index=False)
    contrib_sample.to_csv(args.output_dir / "channel_contributions_sample.csv", index=False)
    curves.to_csv(args.reporting_dir / "saturation_curves.csv", index=False)
    response_report.to_csv(args.reporting_dir / "spend_response.csv", index=False)

    summary = {
        "mode": args.mode,
        "weeks": int(panel["week"].nunique()),
        "rows": int(len(panel)),
        "geos": sorted(panel["geo"].unique().tolist()),
        "lift_tests": int(len(lift_tests)),
        "map_param_count": len(fit.map_params) if fit.map_params is not None else None,
        "outputs": {
            "roi": str(args.output_dir / "roi.csv"),
            "calibration": str(args.output_dir / "calibration_fit.csv"),
            "fit_quality": str(args.output_dir / "fit_quality.csv"),
            "mcmc_diagnostics": str(args.output_dir / "mcmc_diagnostics.csv"),
            "channel_contributions_sample": str(
                args.output_dir / "channel_contributions_sample.csv"
            ),
            "saturation_curves": str(args.reporting_dir / "saturation_curves.csv"),
            "spend_response": str(args.reporting_dir / "spend_response.csv"),
        },
    }
    (args.output_dir / "fit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFit summary")
    print(json.dumps(summary, indent=2))
    print("\nCalibration fit")
    print(calibration.to_string(index=False))
    print("\nFit quality")
    print(fit_quality.round(4).to_string(index=False))
    print("\nMCMC diagnostics")
    if diagnostics.empty:
        print("No posterior diagnostics available for MAP fits.")
    else:
        print(diagnostics.round(3).to_string(index=False))
    print("\nROI")
    print(roi.round(3).to_string(index=False))
    print(f"\nSpend response for {args.spend_multiplier:.0%} spend scenario")
    print(response_report.round(4).to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir}")
    print(f"Wrote reporting to {args.reporting_dir}")


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
    args = parse_args(argv)

    panel = pd.read_csv(args.panel)
    panel = select_week_subset(panel, args.weeks)

    lift_tests = pd.read_csv(args.lift_tests)
    if args.adjust_lift_windows and args.weeks > 0:
        lift_tests = adjust_lift_windows(
            lift_tests,
            panel,
            holdout_fraction=args.holdout_fraction,
        )

    data = build_data(panel)
    experiments = build_experiments(lift_tests, data)
    model = HierarchicalMMM(holdout_fraction=args.holdout_fraction)
    fit = model.fit(data, experiments=experiments, mode=args.mode, **fit_kwargs(args))
    write_outputs(args=args, panel=panel, lift_tests=lift_tests, fit=fit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
