from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from datetime import datetime
import pandas as pd



@dataclass
class DataConfiguration:
    topology: Path
    ts_data_base: Literal["SimBench", "Opfingen"]

    pv_quota: float
    hp_quota: float
    bss_quota: float

    sample_name: str = ""

    day: int = 24
    month: int = 6

    forecast: Literal["Perfect", "Gaussian"] = "Perfect"
    seed: int = 65537

    def __post_init__(self):
        for quota in [self.pv_quota, self.hp_quota, self.bss_quota]:
            if not (0 <= quota <= 1):
                raise ValueError("Quotas must be between 0 and 1.")

        if self.pv_quota < self.bss_quota:
            msg = ("BSS quota must me smaller than pv quota since every"
                   "household with bss also has PV.")
            raise ValueError(msg)

        if self.sample_name == "":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.sample_name = f"{self.ts_data_base}_{timestamp}"

    @property
    def year(self) -> int:
        if self.ts_data_base == "Opfingen":
            return 2023
        elif self.ts_data_base == "SimBench":
            return 2016

    @property
    def date_range(self) -> pd.DatetimeIndex:
        start = pd.Timestamp(self.year, self.month, self.day, 0, 0)
        end = pd.Timestamp(self.year, self.month, self.day, 23, 45)

        return pd.date_range(start=start, end=end, freq='15min')
