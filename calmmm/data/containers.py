from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from calmmm.data.schema import ExperimentRow, CalibrationLikelihood, Estimand


@dataclass
class MMMData:
    observations: pd.DataFrame  # columns: time, geo, kpi, outcome, population
    media: pd.DataFrame         # columns: time, geo, channel, spend, exposure
    controls: pd.DataFrame      # columns: time, geo, control, value
    kpi_metadata: pd.DataFrame  # columns: kpi, likelihood, funnel_stage, family

    @property
    def n_geos(self) -> int:
        return self.observations["geo"].nunique()

    @property
    def n_kpis(self) -> int:
        return self.observations["kpi"].nunique()

    @property
    def n_channels(self) -> int:
        return self.media["channel"].nunique()

    @property
    def n_times(self) -> int:
        return self.observations["time"].nunique()

    @functools.cached_property
    def channels(self) -> list[str]:
        return sorted(self.media["channel"].unique().tolist())

    @functools.cached_property
    def kpis(self) -> list[str]:
        return sorted(self.observations["kpi"].unique().tolist())

    @functools.cached_property
    def geos(self) -> list[str]:
        return sorted(self.observations["geo"].unique().tolist())

    @functools.cached_property
    def times(self) -> list[pd.Timestamp]:
        return sorted(self.observations["time"].unique().tolist())

    @property
    def start_date(self) -> pd.Timestamp:
        return self.observations["time"].min()

    @property
    def end_date(self) -> pd.Timestamp:
        return self.observations["time"].max()

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        time: str,
        geo: str,
        kpis: list[str],
        media: list[str],
        spend: list[str],
        exposure: Optional[list[str]] = None,
        controls: Optional[list[str]] = None,
        population: Optional[str] = None,
        kpi_likelihoods: Optional[dict[str, str]] = None,
        funnel_stages: Optional[list[str]] = None,
    ) -> "MMMData":
        if len(media) != len(spend):
            raise ValueError(
                f"media and spend must have the same length, "
                f"got media={len(media)} spend={len(spend)}"
            )
        if exposure is not None and len(exposure) != len(media):
            raise ValueError(
                f"exposure must have the same length as media, "
                f"got exposure={len(exposure)} media={len(media)}"
            )

        df = df.copy()
        df[time] = pd.to_datetime(df[time])

        # Build observations: long format (one row per time x geo x kpi)
        # column order: time, geo, kpi, outcome, population
        obs_rows = []
        for kpi in kpis:
            kpi_df = df[[time, geo]].copy()
            kpi_df.columns = ["time", "geo"]
            kpi_df["kpi"] = kpi
            kpi_df["outcome"] = df[kpi].values
            kpi_df["population"] = df[population].values if population is not None else np.nan
            obs_rows.append(kpi_df)
        observations = pd.concat(obs_rows, ignore_index=True)

        # Build media: long format (one row per time x geo x channel)
        # column order: time, geo, channel, spend, exposure
        media_rows = []
        for idx, (ch_name, sp_col) in enumerate(zip(media, spend)):
            m_df = df[[time, geo]].copy()
            m_df.columns = ["time", "geo"]
            m_df["channel"] = ch_name
            m_df["spend"] = df[sp_col].values
            exp_col = exposure[idx] if exposure is not None else None
            m_df["exposure"] = df[exp_col].values if exp_col is not None else np.nan
            media_rows.append(m_df)
        media_df = pd.concat(media_rows, ignore_index=True)

        # Build controls: long format (one row per time x geo x control)
        # column order: time, geo, control, value
        if controls:
            ctrl_rows = []
            for ctrl in controls:
                c_df = df[[time, geo]].copy()
                c_df.columns = ["time", "geo"]
                c_df["control"] = ctrl
                c_df["value"] = df[ctrl].values
                ctrl_rows.append(c_df)
            controls_df = pd.concat(ctrl_rows, ignore_index=True)
        else:
            controls_df = pd.DataFrame(columns=["time", "geo", "control", "value"])

        # Build kpi_metadata
        kpi_meta_rows = []
        for kpi in kpis:
            likelihood = (kpi_likelihoods or {}).get(kpi, "negative_binomial")
            stage = (
                funnel_stages.index(kpi)
                if funnel_stages and kpi in funnel_stages
                else None
            )
            kpi_meta_rows.append({
                "kpi": kpi,
                "likelihood": likelihood,
                "funnel_stage": stage,
                "family": None,
            })
        kpi_metadata = pd.DataFrame(kpi_meta_rows)

        return cls(
            observations=observations,
            media=media_df,
            controls=controls_df,
            kpi_metadata=kpi_metadata,
        )


class IncrementalityTests:
    def __init__(self, experiments: list[ExperimentRow]) -> None:
        self._experiments = experiments

    def __len__(self) -> int:
        return len(self._experiments)

    def __getitem__(self, idx: int) -> ExperimentRow:
        return self._experiments[idx]

    def __iter__(self):
        return iter(self._experiments)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        channel: str,
        kpi: str,
        geo_scope: str,
        start: str,
        end: str,
        lift: str,
        standard_error: Optional[str] = None,
        ci_lower: Optional[str] = None,
        ci_upper: Optional[str] = None,
        calibration_likelihood: str = "normal",
        student_t_nu: float = 5.0,
        estimand: str = "total",
        mmmdata: Optional["MMMData"] = None,
    ) -> "IncrementalityTests":
        experiments = []
        for i, row in df.iterrows():
            se = float(row[standard_error]) if standard_error and standard_error in df.columns else None
            ci_lo = float(row[ci_lower]) if ci_lower and ci_lower in df.columns else None
            ci_hi = float(row[ci_upper]) if ci_upper and ci_upper in df.columns else None

            channel_val = row[channel]
            channels = (
                [c.strip() for c in channel_val.split(",")]
                if isinstance(channel_val, str)
                else [str(channel_val)]
            )

            geo_val = row[geo_scope]
            geos = (
                [g.strip() for g in geo_val.split(",")]
                if isinstance(geo_val, str)
                else [str(geo_val)]
            )

            test_id = str(row["test_id"]) if "test_id" in df.columns else f"exp_{i}"

            exp = ExperimentRow(
                test_id=test_id,
                channel_bundle=channels,
                kpi=str(row[kpi]),
                geo_scope=geos,
                start_date=pd.Timestamp(row[start]),
                end_date=pd.Timestamp(row[end]),
                lift=float(row[lift]),
                se=se,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                calibration_likelihood=CalibrationLikelihood(calibration_likelihood),
                student_t_nu=student_t_nu,
                estimand=Estimand(estimand),
            )

            if mmmdata is not None:
                _validate_experiment_against_dataset(exp, mmmdata)

            experiments.append(exp)

        return cls(experiments)


def _validate_experiment_against_dataset(
    exp: ExperimentRow, dataset: "MMMData"
) -> None:
    known_channels = set(dataset.channels)
    for ch in exp.channel_bundle:
        if ch not in known_channels:
            raise ValueError(
                f"unknown channel '{ch}' in experiment '{exp.test_id}'; "
                f"known channels: {sorted(known_channels)}"
            )

    known_kpis = set(dataset.kpis)
    if exp.kpi not in known_kpis:
        raise ValueError(
            f"unknown kpi '{exp.kpi}' in experiment '{exp.test_id}'; "
            f"known kpis: {sorted(known_kpis)}"
        )

    panel_start = dataset.start_date
    panel_end = dataset.end_date
    if exp.start_date < panel_start or exp.end_date > panel_end:
        raise ValueError(
            f"experiment '{exp.test_id}' window "
            f"[{exp.start_date.date()}, {exp.end_date.date()}] "
            f"is outside the observed date range "
            f"[{panel_start.date()}, {panel_end.date()}]"
        )
