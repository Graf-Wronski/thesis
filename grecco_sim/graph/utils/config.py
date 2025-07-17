from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from grecco_sim.graph.utils.format import Format


@dataclass
class SamplerConfiguration:
    p_heatpump: float = 0.2
    p_pv: float = 0.5
    transformer_type: str = "0.25 MVA 20/0.4 kV"
    s_nom: Optional[float] = .05
    seed: Optional[int] = None

@dataclass
class PushRelabelConfiguration:
    time_index: pd.DatetimeIndex
    trafo_sign: pd.Series

    seed: int = 65537
    verbose: bool = False
    slack_as_source: bool = True
    conversion_order: int = 2
    dt_h: float = 0.25
    max_runtime: float = 1500 # The maximal runtime in seconds.

    @property
    def charging_request_path(self) -> Path:
        year = self.time_index[0].year
        return Format().data_root / "ev" / f"charging_sessions_{year}.csv"

    def __post_init__(self):
        if not np.all(self.trafo_sign.index == self.time_index):
            raise ValueError("trafo_sign index does not match time_index.")

@dataclass
class BuilderConfig:
    push_relabel_config: PushRelabelConfiguration
