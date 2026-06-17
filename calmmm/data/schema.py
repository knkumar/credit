from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class KPILikelihood(str, Enum):
    GAUSSIAN = "gaussian"
    LOGNORMAL = "lognormal"
    NEGATIVE_BINOMIAL = "negative_binomial"
    BINOMIAL = "binomial"


class CalibrationLikelihood(str, Enum):
    NORMAL = "normal"
    STUDENT_T = "student_t"
    TRUNCATED_NORMAL = "truncated_normal"
    LAPLACE = "laplace"


class Estimand(str, Enum):
    IMMEDIATE = "immediate"
    CARRYOVER = "carryover"
    TOTAL = "total"
    CUMULATIVE = "cumulative"


@dataclass
class ObservationRow:
    time: pd.Timestamp
    geo: str
    kpi: str
    outcome: float
    population: Optional[float] = None


@dataclass
class MediaRow:
    time: pd.Timestamp
    geo: str
    channel: str
    spend: float
    exposure: Optional[float] = None


@dataclass
class ControlRow:
    time: pd.Timestamp
    geo: str
    control: str
    value: float


@dataclass
class KPIMetadata:
    kpi: str
    likelihood: KPILikelihood = KPILikelihood.NEGATIVE_BINOMIAL
    funnel_stage: Optional[int] = None
    family: Optional[str] = None


@dataclass
class ExperimentRow:
    test_id: str
    channel_bundle: list[str]
    kpi: str
    geo_scope: list[str]
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    lift: float
    se: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    calibration_likelihood: CalibrationLikelihood = CalibrationLikelihood.NORMAL
    student_t_nu: float = 5.0
    estimand: Estimand = Estimand.TOTAL

    def __post_init__(self) -> None:
        if self.se is None:
            if self.ci_lower is None or self.ci_upper is None:
                raise ValueError(
                    "ExperimentRow requires either se or ci_lower/ci_upper"
                )
            self.se = (self.ci_upper - self.ci_lower) / (2 * 1.96)
