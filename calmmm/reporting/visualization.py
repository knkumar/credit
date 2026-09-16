from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPORT_SPECS = {
    "spend_response": {
        "path": ("reporting", "spend_response.csv"),
        "metrics": ["response_lift", "response_lift_pct"],
    },
    "saturation_curves": {
        "path": ("reporting", "saturation_curves.csv"),
        "metrics": ["saturation"],
    },
    "roi": {
        "path": ("artifacts", "roi.csv"),
        "metrics": ["roi", "total_contribution", "total_spend"],
    },
    "calibration_fit": {
        "path": ("artifacts", "calibration_fit.csv"),
        "metrics": ["lift_model", "lift_obs", "z_score"],
    },
    "fit_quality": {
        "path": ("artifacts", "fit_quality.csv"),
        "metrics": ["value"],
    },
    "mcmc_diagnostics": {
        "path": ("artifacts", "mcmc_diagnostics.csv"),
        "metrics": ["r_hat", "ess_bulk", "ess_tail"],
    },
}


def build_summary_table(*, reporting_dir: Path, artifacts_dir: Path) -> pd.DataFrame:
    rows = []
    for report_name, spec in REPORT_SPECS.items():
        path = _resolve_report_path(spec["path"], reporting_dir, artifacts_dir)
        if not path.exists():
            continue

        df = pd.read_csv(path)
        metric_columns = [col for col in spec["metrics"] if col in df.columns]
        rows.append(
            {
                "report": report_name,
                "path": str(path),
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "metric_columns": ",".join(metric_columns),
            }
        )

    return pd.DataFrame(rows)


def render_reporting_outputs(*, reporting_dir: Path, artifacts_dir: Path) -> list[Path]:
    reporting_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    summary = build_summary_table(reporting_dir=reporting_dir, artifacts_dir=artifacts_dir)
    summary_path = reporting_dir / "summary_table.csv"
    summary.to_csv(summary_path, index=False)
    outputs.append(summary_path)

    spend_response = reporting_dir / "spend_response.csv"
    if spend_response.exists():
        out = reporting_dir / "spend_response.svg"
        _render_spend_response(pd.read_csv(spend_response), out)
        outputs.append(out)

    saturation_curves = reporting_dir / "saturation_curves.csv"
    if saturation_curves.exists():
        out = reporting_dir / "saturation_curves.svg"
        _render_saturation_curves(pd.read_csv(saturation_curves), out)
        outputs.append(out)

    roi = artifacts_dir / "roi.csv"
    if roi.exists():
        out = reporting_dir / "roi.svg"
        _render_roi(pd.read_csv(roi), out)
        outputs.append(out)

    calibration = artifacts_dir / "calibration_fit.csv"
    if calibration.exists():
        out = reporting_dir / "calibration_fit.svg"
        _render_calibration(pd.read_csv(calibration), out)
        outputs.append(out)

    fit_quality = artifacts_dir / "fit_quality.csv"
    if fit_quality.exists():
        fit_quality_df = pd.read_csv(fit_quality)
        if not fit_quality_df.empty:
            out_r2 = reporting_dir / "fit_quality_r2.svg"
            out_rmse = reporting_dir / "fit_quality_rmse.svg"
            _render_fit_quality(fit_quality_df, out_r2, out_rmse)
            outputs.extend([out_r2, out_rmse])

    mcmc_diagnostics = artifacts_dir / "mcmc_diagnostics.csv"
    if mcmc_diagnostics.exists():
        mcmc_diagnostics_df = pd.read_csv(mcmc_diagnostics)
        if not mcmc_diagnostics_df.empty:
            outputs.extend(
                _render_mcmc_diagnostics(
                    mcmc_diagnostics_df,
                    reporting_dir / "mcmc_rhat.svg",
                    reporting_dir / "mcmc_ess.svg",
                )
            )

    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render calmmm report CSVs as SVG visualizations.",
    )
    parser.add_argument("--reporting-dir", type=Path, default=Path("reporting"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/demo_fit"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = render_reporting_outputs(
        reporting_dir=args.reporting_dir,
        artifacts_dir=args.artifacts_dir,
    )
    print("Rendered reporting outputs:")
    for output in outputs:
        print(f"- {output}")
    return 0


def _resolve_report_path(parts: tuple[str, str], reporting_dir: Path, artifacts_dir: Path) -> Path:
    root, filename = parts
    if root == "reporting":
        return reporting_dir / filename
    return artifacts_dir / filename


def _render_spend_response(df: pd.DataFrame, out: Path) -> None:
    labels = df["channel"].astype(str).tolist()
    values = (df["response_lift"].astype(float) * 100.0).tolist()
    spend_multiplier = float(df["spend_multiplier"].iloc[0]) if "spend_multiplier" in df else 1.10
    spend_increase_pct = (spend_multiplier - 1.0) * 100.0
    _write_bar_svg(
        out,
        title="Spend scenario response by channel",
        subtitle=f"Modeled saturation response change from a {spend_increase_pct:.0f}% spend increase.",
        labels=labels,
        values=values,
        value_label="Response change (percentage points)",
        value_suffix=" pp",
    )


def _render_saturation_curves(df: pd.DataFrame, out: Path) -> None:
    series = []
    for channel, group in df.groupby("channel"):
        ordered = group.sort_values("spend")
        series.append(
            {
                "label": str(channel),
                "x": ordered["spend"].astype(float).tolist(),
                "y": (ordered["saturation"].astype(float) * 100.0).tolist(),
            }
        )
    _write_line_svg(
        out,
        title="Fitted saturation curves",
        series=series,
        x_label="Spend ($)",
        y_label="Saturation (%)",
    )


def _render_roi(df: pd.DataFrame, out: Path) -> None:
    grouped = (
        df.assign(label=df["kpi"].astype(str) + " / " + df["channel"].astype(str))
        .sort_values("roi", ascending=False)
        .head(12)
    )
    _write_bar_svg(
        out,
        title="ROI by KPI and channel",
        subtitle="Marginal contribution per $1 of spend; higher values indicate stronger modeled return.",
        labels=grouped["label"].tolist(),
        values=grouped["roi"].astype(float).tolist(),
        value_label="ROI (KPI units per $1 spend)",
        value_suffix="",
    )


def _render_calibration(df: pd.DataFrame, out: Path) -> None:
    labels = df["test_id"].astype(str).tolist()
    model = df["lift_model"].astype(float).tolist()
    observed = df["lift_obs"].astype(float).tolist()
    _write_grouped_bar_svg(
        out,
        title="Calibration: modeled vs observed lift",
        subtitle="Experiment lift comparison in the KPI's original outcome units.",
        labels=labels,
        series=[("modeled", model), ("observed", observed)],
        value_label="Lift (KPI units)",
    )


def _render_fit_quality(df: pd.DataFrame, out_r2: Path, out_rmse: Path) -> None:
    metric_type = df["metric"].str.split("_", n=1).str[0]
    r2 = df[metric_type == "r2"].sort_values("kpi")
    rmse = df[metric_type == "rmse"].sort_values("kpi")
    _write_diverging_bar_svg(
        out_r2,
        title="Fit quality: R2 by KPI",
        subtitle="Training-window R2; negative means the fitted mean is worse than the KPI mean.",
        labels=r2["kpi"].astype(str).tolist(),
        values=r2["value"].astype(float).tolist(),
        value_label="R2",
    )
    _write_bar_svg(
        out_rmse,
        title="Fit quality: RMSE by KPI",
        subtitle="Training-window RMSE in each KPI's original outcome units.",
        labels=rmse["kpi"].astype(str).tolist(),
        values=rmse["value"].astype(float).tolist(),
        value_label="RMSE (KPI units)",
        value_suffix="",
    )


def _render_mcmc_diagnostics(df: pd.DataFrame, out_rhat: Path, out_ess: Path) -> list[Path]:
    written = [out_ess]
    ess = df.sort_values("ess_bulk").head(20)
    _write_grouped_bar_svg(
        out_ess,
        title="MCMC diagnostics: effective sample size",
        subtitle="Lowest-ESS parameters first; low ESS means the posterior is not well explored yet.",
        labels=ess["parameter"].astype(str).tolist(),
        series=[
            ("ess_bulk", ess["ess_bulk"].astype(float).tolist()),
            ("ess_tail", ess["ess_tail"].astype(float).tolist()),
        ],
        value_label="Effective sample size",
    )

    rhat = df.dropna(subset=["r_hat"])
    if rhat.empty:
        # R-hat needs >=2 chains; single-chain VI/sample runs report NaN for every parameter.
        return written

    rhat = rhat.sort_values("r_hat", ascending=False).head(20)
    _write_bar_svg(
        out_rhat,
        title="MCMC diagnostics: R-hat by parameter",
        subtitle="Highest R-hat first; values above 1.01 indicate the chains have not converged.",
        labels=rhat["parameter"].astype(str).tolist(),
        values=rhat["r_hat"].astype(float).tolist(),
        value_label="R-hat",
        value_suffix="",
        reference_line=1.01,
    )
    written.append(out_rhat)
    return written


def _write_bar_svg(
    out: Path,
    *,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    value_label: str,
    value_suffix: str,
    reference_line: float | None = None,
) -> None:
    width, height = 1200, 600
    margin_left, margin_top, margin_bottom, margin_right = 280, 112, 88, 180
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_value = max([abs(v) for v in values] + [1.0]) * 1.15
    bar_gap = 14
    bar_height = max(18, (chart_height - bar_gap * max(len(values) - 1, 0)) / max(len(values), 1))

    elements = [_svg_header(width, height, title)]
    elements.append(_text(24, 64, subtitle, size=13))
    elements.append(_text(margin_left, margin_top - 18, value_label, size=13, weight="700"))
    if reference_line is not None and reference_line <= max_value:
        ref_x = margin_left + chart_width * reference_line / max_value
        elements.append(
            f'<line x1="{ref_x:.1f}" y1="{margin_top:.1f}" x2="{ref_x:.1f}" y2="{margin_top + chart_height:.1f}" '
            'stroke="#111111" stroke-width="1" stroke-dasharray="4 3" />'
        )
        elements.append(_text(ref_x + 4, margin_top + 12, f"{reference_line:g} threshold", size=11))
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + idx * (bar_height + bar_gap)
        bar_width = chart_width * max(value, 0.0) / max_value
        elements.append(_text(24, y + bar_height * 0.65, _truncate(label, 38), size=14))
        elements.append(
            f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="#ff6b35" />'
        )
        elements.append(
            _text(
                margin_left + bar_width + 8,
                y + bar_height * 0.65,
                f"{_format_number(value)}{value_suffix}",
                size=13,
            )
        )
    out.write_text("\n".join(elements + [_svg_footer()]), encoding="utf-8")


def _write_diverging_bar_svg(
    out: Path,
    *,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    value_label: str,
) -> None:
    width, height = 1200, 600
    margin_left, margin_top, margin_bottom, margin_right = 280, 112, 88, 180
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    min_value = min(values + [0.0])
    max_value = max(values + [0.0])
    value_range = max(max_value - min_value, 1e-9)
    zero_x = margin_left + chart_width * (0.0 - min_value) / value_range
    bar_gap = 14
    bar_height = max(18, (chart_height - bar_gap * max(len(values) - 1, 0)) / max(len(values), 1))

    elements = [_svg_header(width, height, title)]
    elements.append(_text(24, 64, subtitle, size=13))
    elements.append(_text(margin_left, margin_top - 18, value_label, size=13, weight="700"))
    elements.append(
        f'<line x1="{zero_x:.1f}" y1="{margin_top:.1f}" x2="{zero_x:.1f}" y2="{margin_top + chart_height:.1f}" stroke="#b8bcc4" />'
    )
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + idx * (bar_height + bar_gap)
        bar_width = chart_width * abs(value) / value_range
        bar_x = zero_x - bar_width if value < 0 else zero_x
        color = "#ff6b35" if value >= 0 else "#b91c1c"
        elements.append(_text(24, y + bar_height * 0.65, _truncate(label, 38), size=14))
        elements.append(
            f'<rect x="{bar_x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="{color}" />'
        )
        text_x = bar_x + bar_width + 8 if value >= 0 else bar_x - 8
        anchor = "start" if value >= 0 else "end"
        elements.append(
            _text(text_x, y + bar_height * 0.65, _format_number(value), size=13, anchor=anchor)
        )
    out.write_text("\n".join(elements + [_svg_footer()]), encoding="utf-8")


def _write_grouped_bar_svg(
    out: Path,
    *,
    title: str,
    subtitle: str,
    labels: list[str],
    series: list[tuple[str, list[float]]],
    value_label: str,
) -> None:
    width, height = 1200, 620
    margin_left, margin_top, margin_bottom, margin_right = 280, 124, 88, 220
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_value = max([abs(v) for _, values in series for v in values] + [1.0]) * 1.15
    group_gap = 24
    group_height = max(42, (chart_height - group_gap * max(len(labels) - 1, 0)) / max(len(labels), 1))
    bar_height = group_height / max(len(series), 1) - 4
    colors = ["#ff6b35", "#111111", "#b8bcc4"]

    elements = [_svg_header(width, height, title)]
    elements.append(_text(24, 64, subtitle, size=13))
    elements.append(_text(margin_left, margin_top - 20, value_label, size=13, weight="700"))
    for idx, label in enumerate(labels):
        group_y = margin_top + idx * (group_height + group_gap)
        elements.append(_text(24, group_y + group_height * 0.58, _truncate(label, 36), size=14))
        for s_idx, (name, values) in enumerate(series):
            value = values[idx]
            y = group_y + s_idx * (bar_height + 4)
            bar_width = chart_width * max(value, 0.0) / max_value
            elements.append(
                f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="{colors[s_idx % len(colors)]}" />'
            )
            elements.append(
                _text(
                    margin_left + bar_width + 8,
                    y + bar_height * 0.72,
                    f"{name}: {_format_number(value, decimals=0)}",
                    size=12,
                )
            )
    out.write_text("\n".join(elements + [_svg_footer()]), encoding="utf-8")


def _write_line_svg(
    out: Path,
    *,
    title: str,
    series: list[dict],
    x_label: str,
    y_label: str,
) -> None:
    width, height = 1100, 620
    left, top, right, bottom = 92, 112, 56, 112
    chart_width = width - left - right
    chart_height = height - top - bottom
    max_x = max([max(item["x"]) for item in series if item["x"]] + [1.0])
    max_y = max(100.0, max([max(item["y"]) for item in series if item["y"]] + [1.0]))
    colors = ["#ff6b35", "#111111", "#6b7280", "#2563eb", "#287d3c"]

    elements = [_svg_header(width, height, title)]
    elements.append(_text(24, 64, "Media response index by spend level. 0% = no response; 100% = fully saturated.", size=13))
    elements.append(_text(left + chart_width * 0.42, height - 44, x_label, size=13, weight="700"))
    elements.append(_text(16, top - 16, y_label, size=13, weight="700"))
    elements.append(f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#b8bcc4" />')
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#b8bcc4" />')
    for tick in _ticks(max_x, count=5):
        x = left + chart_width * tick / max_x
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_height}" stroke="#edf0f2" />')
        elements.append(f'<line x1="{x:.1f}" y1="{top + chart_height}" x2="{x:.1f}" y2="{top + chart_height + 6}" stroke="#6b7280" />')
        elements.append(_text(x - 20, top + chart_height + 24, _format_dollars(tick), size=11))
    for tick in _ticks(max_y, count=5):
        y = top + chart_height - chart_height * tick / max_y
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#edf0f2" />')
        elements.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#6b7280" />')
        elements.append(_text(34, y + 4, f"{tick:.0f}%", size=11))
    for idx, item in enumerate(series):
        points = []
        for x_val, y_val in zip(item["x"], item["y"]):
            x = left + chart_width * x_val / max_x
            y = top + chart_height - chart_height * y_val / max_y
            points.append(f"{x:.1f},{y:.1f}")
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{colors[idx % len(colors)]}" stroke-width="3" />'
        )
        legend_x = left + idx * 150
        legend_y = height - 28
        elements.append(f'<rect x="{legend_x}" y="{legend_y - 12}" width="14" height="14" fill="{colors[idx % len(colors)]}" />')
        elements.append(_text(legend_x + 20, legend_y, str(item["label"]), size=13))
    out.write_text("\n".join(elements + [_svg_footer()]), encoding="utf-8")


def _svg_header(width: int, height: int, title: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            _text(24, 38, title, size=22, weight="700"),
        ]
    )


def _svg_footer() -> str:
    return "</svg>"


def _text(x: float, y: float, text: str, *, size: int, weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="#111111" text-anchor="{anchor}">{_escape(text)}</text>'
    )


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def _ticks(max_value: float, *, count: int) -> list[float]:
    if count <= 1:
        return [0.0]
    step = max_value / (count - 1)
    return [idx * step for idx in range(count)]


def _format_number(value: float, *, decimals: int = 2) -> str:
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def _format_dollars(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}k"
    return f"${value:.0f}"


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
