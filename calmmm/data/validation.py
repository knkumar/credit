from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def raise_if_errors(self) -> None:
        if self.has_errors:
            msg = "\n".join(f"  - {e}" for e in self.errors)
            raise ValueError(f"MMMData validation failed:\n{msg}")


def validate_mmmdata(dataset) -> ValidationResult:
    result = ValidationResult()
    _check_duplicate_panel_rows(dataset, result)
    _check_negative_spend(dataset, result)
    _check_missing_outcomes(dataset, result)
    _check_count_kpi_integrity(dataset, result)
    _check_binomial_kpi_has_population(dataset, result)
    _check_binomial_not_exceeds_population(dataset, result)
    _check_weak_media_variation(dataset, result)
    return result


def _check_duplicate_panel_rows(dataset, result: ValidationResult) -> None:
    obs = dataset.observations
    dupes = obs.duplicated(subset=["time", "geo", "kpi"])
    if dupes.any():
        n = int(dupes.sum())
        result.errors.append(
            f"Duplicate panel rows detected: {n} duplicate (time, geo, kpi) combinations"
        )


def _check_negative_spend(dataset, result: ValidationResult) -> None:
    neg = dataset.media["spend"] < 0
    if neg.any():
        channels = dataset.media.loc[neg, "channel"].unique().tolist()
        result.errors.append(
            f"Negative spend values found in channels: {channels}"
        )


def _check_missing_outcomes(dataset, result: ValidationResult) -> None:
    missing = int(dataset.observations["outcome"].isna().sum())
    if missing > 0:
        result.errors.append(
            f"Missing outcome values: {missing} rows have NaN outcome"
        )


def _check_count_kpi_integrity(dataset, result: ValidationResult) -> None:
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


def _check_binomial_kpi_has_population(dataset, result: ValidationResult) -> None:
    binomial_kpis = dataset.kpi_metadata.loc[
        dataset.kpi_metadata["likelihood"] == "binomial", "kpi"
    ].tolist()
    for kpi in binomial_kpis:
        kpi_obs = dataset.observations[dataset.observations["kpi"] == kpi]
        if kpi_obs["population"].isna().all():
            result.warnings.append(
                f"KPI '{kpi}' uses binomial likelihood but no population column "
                f"was provided. Supply population= in from_dataframe()."
            )


def _check_binomial_not_exceeds_population(dataset, result: ValidationResult) -> None:
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


def _check_weak_media_variation(dataset, result: ValidationResult) -> None:
    for channel in dataset.channels:
        spend = dataset.media.loc[dataset.media["channel"] == channel, "spend"]
        mean = spend.mean()
        if mean > 0 and (spend.std() / mean) < 0.05:
            result.warnings.append(
                f"Weak media variation for channel '{channel}': "
                f"coefficient of variation = {spend.std() / mean:.3f}. "
                f"MMM estimates will be unreliable for this channel."
            )
