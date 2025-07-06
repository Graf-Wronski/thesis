from typing import Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import random
from datetime import datetime

from thesis.data.data_configuration import DataConfiguration
from thesis.graph.utils.format import Format


class TimeSeriesSampler:
    def __init__(self, config: DataConfiguration, buses: list):
        self.config = config
        self.buses = buses
        self.pv_buses = random.sample(buses, int(len(buses) * config.pv_quota))
        self.bss_buses = random.sample(self.pv_buses, int(len(buses) * config.bss_quota))
        self.hp_buses = random.sample(buses, int(len(buses) * config.hp_quota))
        self.ev_buses = random.sample(buses, int(len(buses) * config.ev_quota))

        np.random.seed(config.seed)
        random.seed(config.seed)

        if config.ts_data_base == "SimBench":
            # Simbench has data for 2016.
            self.date = datetime(year=2016, month=config.month, day=config.day)
            self.db = SimbenchDB(self.date)
        elif config.ts_data_base == "Opfingen":
            # Opfingen has data for 2023.
            self.date = datetime(year=2023, month=config.month, day=config.day)
            self.db = OpfingenDB(self.date)

    def sample_baseload(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Baseload is defined by bus and carrier. TS is defined by p_set. """
        buses = self.buses
        baseload_df = pd.DataFrame({"bus": buses, "carrier": "baseload"})
        ts_data = random.choices(self.db.baseload_ts, k=len(buses))
        baseload_ts_df = pd.DataFrame(ts_data, index=self.buses).T
        return baseload_df, baseload_ts_df

    def sample_pv(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        buses = self.pv_buses
        pv_df = pd.DataFrame(
            {"bus": buses, "carrier": "solar", "control": "PQ"})
        ts_data = random.choices(self.db.pv_ts, k=len(buses))
        baseload_ts_df = pd.DataFrame(ts_data, index=buses).T
        return pv_df, baseload_ts_df

    def sample_bss(self) -> pd.DataFrame:
        buses = self.bss_buses
        p = random.choices(self.db.bss_p_nom, k=len(buses))
        bss_df = pd.DataFrame({"bus": buses, "type": "h0_battery", "p_nom": p})
        return bss_df

    def sample_hp(self) -> pd.DataFrame:
        buses = self.bss_buses
        p_set = random.choices(self.db.hp_p_set, k=len(buses))
        bss_df = pd.DataFrame({
            "bus": buses, "carrier": "heat_pump", "p_set": p_set})
        return bss_df

    def sample_ev(self) -> pd.DataFrame:
        buses = self.ev_buses
        profile = random.choices(self.db.ev_chargers, k=len(buses))
        ev_df = pd.DataFrame({"type": profile, "bus": buses, "p_nom": 0.011})

        return ev_df

    @property
    def irrandiance(self) -> pd.Series:
        return self.db.weather["Solar Irradiance"]

    @property
    def temperature(self) -> pd.Series:
        return self.db.weather["Outside Temperature"]

class TimeSeriesDB:
    def __init__(self, date: datetime):
        self.date = date
        p_weather = Path("/home/carl-wanninger/data/weather")
        weather = pd.read_csv(p_weather / f"{date.year}.csv")
        weather.index = pd.to_datetime(weather.index)
        self.weather = weather[weather.index.date == date]

    @property
    def hp_p_set(self):
        raise NotImplementedError

    @property
    def bss_p_nom(self):
        raise NotImplementedError

    @property
    def baseload_ts(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def pv_ts(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def ev_chargers(self) -> list[str]:
        """ Charger types as provided by Rebecca and Álvaro. """
        return [f"charger_{i}" for i in [1, 2, 3, 4]]


class OpfingenDB(TimeSeriesDB):
    def __init__(self, date: datetime):
        super().__init__(date)
        self.data_dir = Path("/home/carl-wanninger/data/opfingen")

    def filter_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Convenience for Opfingne index. """

        ordinal_day = self.date.timetuple().tm_yday
        return df[(ordinal_day - 1) * 96: ordinal_day * 96]

    @property
    def hp_p_set(self) -> list:
        loads = pd.read_csv(self.data_dir / "loads.csv")
        loads = loads.query("carrier == 'heat_pump'")
        return loads["p_set"].to_list()

    @property
    def bss_p_nom(self) -> list:
        loads = pd.read_csv(self.data_dir / "storage_units.csv")
        loads = loads.query("type == 'h0_battery'")
        return loads["p_nom"].to_list()

    @property
    def baseload_ts(self) -> list[pd.Series]:
        p_load_ts = pd.read_csv(self.data_dir / "loads-p.csv")
        p_load_ts = self.filter_date(p_load_ts)

        loads = pd.read_csv(self.data_dir / "loads.csv")
        loads = loads.query("carrier != 'heat_pump'")
        p_load_ts = p_load_ts.loc[:, [str(x) for x in loads.name]]

        return [p_load_ts[col] for col in p_load_ts.columns]

    @property
    def pv_ts(self) ->  list[pd.Series]:
        p_pv_ts = pd.read_csv(self.data_dir / "generators-p.csv")
        p_pv_ts = self.filter_date(p_pv_ts)

        buses = pd.read_csv(self.data_dir / "generators.csv")
        buses = buses.query("carrier == 'solar'")
        p_pv_ts = p_pv_ts.loc[:, [str(x) for x in buses.name]]
        return [p_pv_ts[col] for col in p_pv_ts.columns]


class SimbenchDB(TimeSeriesDB):
    def __init__(self, date: datetime):
        super().__init__(date)
        self.data_dir = Path("/home/carl-wanninger/data/simbench/units")

        def load_csv(csv_name: str) -> pd.DataFrame:
            csv_path = self.data_dir / csv_name
            df = pd.read_csv(csv_path, sep=";", on_bad_lines="warn")

            # For time-series: Remove duplicates by averaging.

            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], format="%d.%m.%Y "
                                                               "%H:%M")
                df = df[df["time"] >= self.date]
                df = df[df["time"] < self.date + pd.Timedelta(days=1)]
                old_len = len(df)
                df = df.groupby(["time"]).mean()
                print(f"{old_len - len(df)} duplicates in {csv_name} "
                      f"were averaged. ")

            return df

        self.gen_t = load_csv("RESProfile.csv")
        self.load_t = load_csv("LoadProfile.csv")

        time_format = Format().date
        for time_varying_data in [self.gen_t, self.load_t]:
            time_varying_data.index = pd.DatetimeIndex(
                time_varying_data.index).strftime(time_format)

        self.load = load_csv("Load.csv").query('id.str.contains("LV")')
        self.gen = load_csv("RES.csv").query('id.str.contains("LV")')
        self.storage = load_csv("Storage.csv").query('id.str.contains("LV")')

    @property
    def hp_p_set(self) -> list:
        hps = self.load[self.load['profile'].str.contains('Air|Soil')]
        return hps["pLoad"].to_list()

    @property
    def bss_p_nom(self) -> list:
        return self.storage["eStore"].to_list()

    @property
    def baseload_ts(self) -> list[pd.Series]:
        # ToDo: Maybe only households?
        baseload_ts = []
        baseloads = self.load[self.load['profile'].str.contains('H0')]

        profiles = baseloads["profile"]
        ploads = baseloads["pLoad"]

        for profile, pload in zip(profiles, ploads):
            ts = self.load_t[f"{profile}_pload"] * pload
            baseload_ts.append(ts)

        return baseload_ts

    @property
    def pv_ts(self) -> list[pd.Series]:
        gen_ts = []
        pvs = self.gen[self.gen['profile'].str.contains('PV')]

        profiles = pvs["profile"]
        pgenerations = pvs["pRES"]

        for profile, pload in zip(profiles, pgenerations):
            ts = self.gen_t[f"{profile}"] * pload
            gen_ts.append(ts)

        return gen_ts
