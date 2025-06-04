from typing import Dict
from pathlib import Path
import pytz
import json

import numpy as np
import pandas as pd

from grecco_sim.util import type_defs


UNIX_TIME = "unixtimestamp"


def write_ts(ts: pd.DataFrame, output_filename: Path):
    """Write a time series using unixtime to decode time info."""

    assert isinstance(ts.index, pd.DatetimeIndex), "Only usable for time series data"

    ts.loc[:, UNIX_TIME] = ts.index.astype(np.int64).values / 1e9

    # Sanity check
    unixtime = ts.loc[:, UNIX_TIME].values
    if (unixtime[1] - unixtime[0]) * (len(unixtime) - 1) != unixtime[-1] - unixtime[0]:
        print(f"{unixtime[-1]}, {unixtime[0]}, {unixtime[1]}, {len(unixtime)}")
        raise ValueError("Data has non-constant time step. Result is incorrect")

    ts.to_csv(output_filename, sep=";", index_label="timestamp", float_format="%f")

    ts.drop(columns=UNIX_TIME, inplace=True)


def read_ts(
    input_filename: Path,
    ambiguous: str = "raise",
    tz_info=pytz.timezone("Europe/Berlin"),
) -> pd.DataFrame:
    """Read a time series file in the format used in GreCCo sim output."""

    data_raw = pd.read_csv(input_filename, sep=";", index_col="timestamp")

    ind_start = pd.to_datetime(
        data_raw.loc[data_raw.index[0:2], UNIX_TIME].apply(pd.to_numeric).values, unit="s"
    )
    data_raw.index = pd.date_range(
        start=ind_start[0],
        freq=f"{int((ind_start[1] - ind_start[0]).total_seconds())}s",
        periods=len(data_raw.index),
    )

    data_tz = data_raw.tz_localize("UTC", ambiguous=ambiguous).tz_convert(tz_info)
    data_tz = data_tz.drop(columns=UNIX_TIME)

    return data_tz


def dump_parameterization(out_file_name: Path, parameters):
    """Write a json file of parameter classes"""
    with open(out_file_name, "w") as out_file:
        json.dump(parameters, out_file, cls=type_defs.EnhancedJSONEncoder)

def load_sizing(in_file_name: Path) -> Dict[str, type_defs.SysPars]:

    with open(in_file_name) as sizing_file:
        raw_sizing = json.load(sizing_file)

    SIZING_CLASS = {
        "heatpump": type_defs.SysParsHeatPump,
        "load": type_defs.SysParsLoad,
        "pv_bat": type_defs.SysParsPVBat,
    }
    return {
        sys_id: SIZING_CLASS[raw_dict["system"]](**raw_dict)
        for sys_id, raw_dict in raw_sizing.items()
    }
