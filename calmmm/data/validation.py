from __future__ import annotations

import warnings
from dataclasses import dataclass, field
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from calmmm.data.containers import MMMData


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def raise_if_errors(self) -> "ValidationResult":
        for w in self.warnings:
            warnings.warn(w, UserWarning, stacklevel=2)
        if self.has_errors:
            msg = "\n".join(f"  - {e}" for e in self.errors)
            raise ValueError(f"MMMData validation failed:\n{msg}")
        return self


def validate_mmmdata(dataset: MMMData) -> ValidationResult:
    result = ValidationResult()
    _check_kpi_metadata_completeness(dataset, result)
    _check_unknown_likelihoods(dataset, result)
    _check_duplicate_panel_rows(dataset, result)
    _check_negative_spend(dataset, result)
    _check_all_zero_spend(dataset, result)
    _check_missing_outcomes(dataset, result)
    _check_missing_features(dataset, result)
    _check_count_kpi_integrity(dataset, result)
    _check_lognormal_kpi_integrity(dataset, result)
    _check_binomial_kpi_has_population(dataset, result)
    _check_binomial_not_exceeds_population(dataset, result)
    _check_weak_media_variation(dataset, result)
    _check_temporal_continuity(dataset, result)
    return result


def _check_kpi_metadata_completeness(dataset: MMMData, result: ValidationResult) -> None:
    metadata_kpis = set(dataset.kpi_metadata["kpi"])
    missing_kpis = [kpi for kpi in dataset.kpis if kpi not in metadata_kpis]
    if missing_kpis:
        result.errors.append(f"KPIs present in observations but missing from metadata: {missing_kpis}")


def _check_all_zero_spend(dataset: MMMData, result: ValidationResult) -> None:
    if not dataset.media.empty and "spend" in dataset.media.columns:
        spend_sums = dataset.media.groupby("channel")["spend"].sum()
        zero_channels = spend_sums[spend_sums == 0.0].index.tolist()
        if zero_channels:
            result.errors.append(f"The following channels have all-zero spend across the panel: {zero_channels}. They provide no signal.")


def _check_temporal_continuity(dataset: MMMData, result: ValidationResult) -> None:
    diffs = pd.Series(dataset.times).diff().dropna()
    if not diffs.empty:
        if (diffs.dt.days.max() - diffs.dt.days.min()) > 3:
            result.errors.append(
                "Dataset has missing dates or irregular gaps, which breaks sequential adstock and seasonality assumptions."
            )


def _check_unknown_likelihoods(dataset: MMMData, result: ValidationResult) -> None:
    valid_likelihoods = {"gaussian", "lognormal", "negative_binomial", "binomial"}
    for _, row in dataset.kpi_metadata.iterrows():
        if row["likelihood"] not in valid_likelihoods:
            result.errors.append(
                f"KPI '{row['kpi']}' has unknown likelihood='{row['likelihood']}'. "
                f"Valid likelihoods are: {valid_likelihoods}"
            )


def _check_duplicate_panel_rows(dataset: MMMData, result: ValidationResult) -> None:
    obs = dataset.observations
    dupes = obs.duplicated(subset=["time", "geo", "kpi"])
    if dupes.any():
        n = int(dupes.sum())
        result.errors.append(
            f"Duplicate panel rows detected: {n} duplicate (time, geo, kpi) combinations"
        )

    media = dataset.media
    if not media.empty:
        media_dupes = media.duplicated(subset=["time", "geo", "channel"])
        if media_dupes.any():
            n = int(media_dupes.sum())
            result.errors.append(
                f"Duplicate panel rows detected: {n} duplicate (time, geo, channel) combinations in media data"
            )

    controls = dataset.controls
    if not controls.empty:
        controls_dupes = controls.duplicated(subset=["time", "geo", "control"])
        if controls_dupes.any():
            n = int(controls_dupes.sum())
            result.errors.append(
                f"Duplicate panel rows detected: {n} duplicate (time, geo, control) combinations in controls data"
            )


def _check_negative_spend(dataset: MMMData, result: ValidationResult) -> None:
    neg = dataset.media["spend"] < 0
    if neg.any():
        channels = dataset.media.loc[neg, "channel"].unique().tolist()
        result.errors.append(
            f"Negative spend values found in channels: {channels}"
        )


def _check_missing_outcomes(dataset: MMMData, result: ValidationResult) -> None:
    for kpi in dataset.kpis:
        kpi_obs = dataset.observations[dataset.observations["kpi"] == kpi]
        missing = int(kpi_obs["outcome"].isna().sum())
        if missing > 0:
            result.errors.append(
                f"Missing outcome values for KPI '{kpi}': {missing} missing values found. "
                "PyMC imputation is no longer supported; please impute before modeling."
            )

    expected_obs_len = dataset.n_times * dataset.n_geos * dataset.n_kpis
    if len(dataset.observations) != expected_obs_len:
        result.errors.append(
            f"Incomplete observations panel: expected {expected_obs_len} rows "
            f"but found {len(dataset.observations)}. Please provide balanced panel data."
        )

    expected_media_len = dataset.n_times * dataset.n_geos * dataset.n_channels
    if len(dataset.media) != expected_media_len:
        result.errors.append(
            f"Incomplete media panel: expected {expected_media_len} rows "
            f"but found {len(dataset.media)}. Please provide balanced panel data."
        )


def _check_missing_features(dataset: MMMData, result: ValidationResult) -> None:
    if dataset.media is not None and "spend" in dataset.media.columns:
        if dataset.media["spend"].isna().any():
            missing = int(dataset.media["spend"].isna().sum())
            result.errors.append(f"Missing spend values: {missing} rows have NaN spend in media data")
    
    if dataset.controls is not None and "value" in dataset.controls.columns:
        if dataset.controls["value"].isna().any():
            missing = int(dataset.controls["value"].isna().sum())
            result.errors.append(f"Missing control values: {missing} rows have NaN value in controls data")


def _check_count_kpi_integrity(dataset: MMMData, result: ValidationResult) -> None:
    count_likelihoods = {"negative_binomial", "binomial"}
    for _, row in dataset.kpi_metadata.iterrows():
        if row["likelihood"] in count_likelihoods:
            kpi = row["kpi"]
            obs = dataset.observations.loc[
                dataset.observations["kpi"] == kpi, "outcome"
            ].dropna()
            non_int_mask = obs % 1 != 0
            if non_int_mask.any():
                result.errors.append(
                    f"KPI '{kpi}' has likelihood='{row['likelihood']}' but "
                    f"{int(non_int_mask.sum())} non-integer outcome value(s) found. "
                    "Count likelihoods require whole numbers."
                )
            neg_mask = obs < 0
            if neg_mask.any():
                result.errors.append(
                    f"KPI '{kpi}' has likelihood='{row['likelihood']}' but "
                    f"{int(neg_mask.sum())} negative outcome value(s) found. "
                    "Count likelihoods require non-negative values."
                )


def _check_lognormal_kpi_integrity(dataset: MMMData, result: ValidationResult) -> None:
    for _, row in dataset.kpi_metadata.iterrows():
        if row["likelihood"] == "lognormal":
            kpi = row["kpi"]
            obs = dataset.observations.loc[
                dataset.observations["kpi"] == kpi, "outcome"
            ].dropna()
            obs_numeric = pd.to_numeric(obs, errors='coerce')
            non_pos_mask = (obs_numeric <= 0) | obs_numeric.isna()
            if non_pos_mask.any():
                result.errors.append(
                    f"KPI '{kpi}' has likelihood='lognormal' but "
                    f"{int(non_pos_mask.sum())} zero or negative outcome value(s) found. "
                    "Lognormal likelihood requires strictly positive values."
                )


def _check_binomial_kpi_has_population(dataset: MMMData, result: ValidationResult) -> None:
    binomial_kpis = dataset.kpi_metadata.loc[
        dataset.kpi_metadata["likelihood"] == "binomial", "kpi"
    ].tolist()
    for kpi in binomial_kpis:
        kpi_obs = dataset.observations[dataset.observations["kpi"] == kpi]
        if kpi_obs["population"].isna().any():
            result.errors.append(
                f"KPI '{kpi}' uses binomial likelihood but missing population data "
                f"for some rows. Supply population= in from_dataframe()."
            )


def _check_binomial_not_exceeds_population(dataset: MMMData, result: ValidationResult) -> None:
    binomial_kpis = dataset.kpi_metadata.loc[
        dataset.kpi_metadata["likelihood"] == "binomial", "kpi"
    ].tolist()
    for kpi in binomial_kpis:
        kpi_obs = dataset.observations[dataset.observations["kpi"] == kpi]
        valid = kpi_obs[kpi_obs["population"].notna() & kpi_obs["outcome"].notna()]
        bad = valid[valid["outcome"] > valid["population"]]
        if not bad.empty:
            result.errors.append(
                f"KPI '{kpi}' (binomial): {len(bad)} row(s) where "
                f"outcome > population."
            )


def _check_weak_media_variation(dataset: MMMData, result: ValidationResult) -> None:
    for channel in dataset.channels:
        spend = dataset.media.loc[dataset.media["channel"] == channel, "spend"]
        mean = spend.mean()
        if mean > 0 and (spend.std() / mean) < 0.05:
            result.warnings.append(
                f"Weak media variation for channel '{channel}': "
                f"coefficient of variation = {spend.std() / mean:.3f}. "
                f"MMM estimates will be unreliable for this channel."
            )
