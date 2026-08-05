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
    ci_level: float = 0.95  # confidence level for the reported CI; 0.95 → z≈1.96

    def __post_init__(self) -> None:
        import math
        if math.isnan(self.lift) or math.isinf(self.lift):
            raise ValueError("lift must be a finite number")

        if self.se is None:
            if self.ci_lower is None or self.ci_upper is None:
                raise ValueError(
                    "ExperimentRow requires either se or ci_lower/ci_upper"
                )
            if self.ci_upper < self.ci_lower:
                raise ValueError(
                    f"ci_upper ({self.ci_upper}) must be >= ci_lower ({self.ci_lower})"
                )
            if self.calibration_likelihood == CalibrationLikelihood.STUDENT_T:
                from scipy.stats import t
                z = t.ppf((1 + self.ci_level) / 2, df=self.student_t_nu)
            else:
                from scipy.stats import norm
                z = norm.ppf((1 + self.ci_level) / 2)
            self.se = (self.ci_upper - self.ci_lower) / (2 * z)
        
        if self.se <= 0 or math.isnan(self.se) or math.isinf(self.se):
            raise ValueError(f"se must be > 0 and finite, got {self.se}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExperimentRow):
            return NotImplemented
        return (
            self.test_id == other.test_id
            and self.channel_bundle == other.channel_bundle
            and self.kpi == other.kpi
            and self.geo_scope == other.geo_scope
            and self.start_date == other.start_date
            and self.end_date == other.end_date
            and self.lift == other.lift
            and self.se == other.se
            and self.ci_lower == other.ci_lower
            and self.ci_upper == other.ci_upper
            and self.calibration_likelihood == other.calibration_likelihood
            and self.student_t_nu == other.student_t_nu
            and self.estimand == other.estimand
            and self.ci_level == other.ci_level
        )
