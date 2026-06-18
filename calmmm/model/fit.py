from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pymc as pm

from calmmm.data.containers import MMMData


@dataclass
class MMMFit:
    """
    Result of HierarchicalMMM.fit().

    Attributes
    ----------
    trace : arviz InferenceData (MCMC/VI) or None (MAP)
    map_params : dict of param_name → value (MAP) or None
    model : the underlying PyMC model
    data : the MMMData used to build the model
    """
    trace: Optional[Any]
    map_params: Optional[dict]
    model: pm.Model
    data: MMMData
