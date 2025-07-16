from pathlib import Path
from typing import Tuple, Any, Optional

import pandas as pd
import pypsa


from grecco_sim.util import data_io, build, configs


def pypsa_df_to_grecco_df(df: pd.DataFrame) -> pd.DataFrame:

    # Avoid side effects.
    df = df.copy(deep=True)

    def mw_to_kw(x: Any) -> Any:
        return x * 1000

    if type(df.index) == pd.DatetimeIndex:
        df = data_io.set_tz_index_to_utc(df)

    df = mw_to_kw(df)

    return df


def check_unique(data: pd.Series, unit_type: Optional[str] = None):
    if not data.is_unique:
        duplicates = data[data.duplicated()]
        msg = f"Duplicate units in data: {duplicates}."
        msg = msg + f"\n Type: {unit_type}" if unit_type else ""
        raise NotImplementedError(msg)


def get_system_buses(network: pypsa.Network) -> list[str]:
    """ Systems correspond to (aggregated) unit control (= optimization).
        System buses are identified by baseloads, which are assume to be
        unique per bus. """
    baseload_params, _ = get_baseload(network)
    return baseload_params["bus"].to_list()


def get_baseload(network: pypsa.Network) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Baseloads are all loads except heat pumps. """
    # ToDo: Would be better to expect something like carrier == "baseload"
    baseload_params = network.loads.query("carrier != 'heat_pump'")
    check_unique(baseload_params["bus"], unit_type='baseload')
    baseload_p_set = network.loads_t["p_set"].loc[:, baseload_params.index]
    baseload_p_set = pypsa_df_to_grecco_df(baseload_p_set)
    
    return baseload_params, baseload_p_set

def get_pv(network: pypsa.Network) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pv_params = network.generators.query("carrier == 'solar'")
    check_unique(pv_params["bus"], unit_type='pv')
    pv_p_set = network.generators_t["p_set"].loc[:, pv_params.index]
    pv_p_set = pypsa_df_to_grecco_df(pv_p_set)
    
    return pv_params, pv_p_set

def get_bss(network: pypsa.Network) -> pd.DataFrame:
    # ToDo: Focusing on h0_battery seems over specific.
    bat_params = network.storage_units.query("type == 'h0_battery'")
    check_unique(bat_params["bus"], unit_type='bat')
    bat_params.loc[:, "p_nom"] = pypsa_df_to_grecco_df(bat_params["p_nom"])
    return bat_params


def get_hp(network: pypsa.Network) -> pd.DataFrame:
    """ Baseloads are all loads except heat pumps. """
    # ToDo: Would be better to expect something like carrier == "baseload"
    hp_params = network.loads.query("carrier == 'heat_pump'")
    check_unique(hp_params["bus"], unit_type='heat pump')
    hp_params.loc[:, "p_set"] = pypsa_df_to_grecco_df(hp_params["p_set"])

    return hp_params


def get_ev(network: pypsa.Network, charging_process_path: Path) -> (
        Tuple[pd.DataFrame, pd.DataFrame]):

    # Unidirectional EVs should be declared as loads, not as storages.

    charging_processes = pd.read_csv(charging_process_path)
    query = network.storage_units["type"].str.contains("charger")
    charger_params = network.storage_units[query].copy()
    check_unique(charger_params["bus"], unit_type='ev_charger')

    # ToDo: Capacity could be added by extra file.
    charger_params.loc[:, "p_nom"] = pypsa_df_to_grecco_df(charger_params["p_nom"])
    charger_params.loc[:, "charger_id"] = charger_params["type"]
    charger_params.loc[:, "ev_id"] = charger_params["charger_id"] + "_ev"
    charger_params.loc[:, "capacity"] = 60.0
    charger_params.index = build.sys_id(charger_params["bus"]) + "_ev"

    return charger_params, charging_processes


def preprocess_charging_requests(
        request_ts: pd.DataFrame,
        time_index: pd.DatetimeIndex) -> pd.DataFrame:

    # Filter charging processes.
    for key in ["StartOfProcess", "EndOfProcess"]:
        request_ts.loc[:, key] = pd.to_datetime(request_ts[key])
    start = time_index.min().tz_localize(None)
    end = time_index.max().tz_localize(None)
    q1 = "(@start <= EndOfProcess) and (@end >= StartOfProcess)"
    request_ts = request_ts.query(q1).copy()
    q2 = "StartSoc < TargetSoc"
    request_ts = request_ts.query(q2).copy()

    charging_time = request_ts["EndOfProcess"] - request_ts["StartOfProcess"]
    request_ts.loc[:, "total_time"] = charging_time

    # ChargingProcesses are not always fully in SimulationTime
    # We use a fictive_soc_start to adapt charge debt.

    # Clip start and end time at simulation time borders.
    relative_start = request_ts["StartOfProcess"].clip(lower=start)
    relative_end = request_ts["EndOfProcess"].clip(upper=end)
    request_ts.loc[:, "relative_start"] = relative_start
    request_ts.loc[:, "relative_end"] = relative_end
    request_ts.loc[:, "relative_time"] = relative_end - relative_start

    # Reduce (charge) "debt" by assuming that some charging took place.
    factor = request_ts["relative_time"] / request_ts["total_time"]
    debt = request_ts.loc[:, "TargetSoc"] - request_ts.loc[:, "StartSoc"]
    request_ts.loc[:, "fictive_soc_start"] = request_ts["StartSoc"]
    request_ts.loc[:, "fictive_soc_start"] += (1 - factor) * debt

    return request_ts


def determine_feeders(n: pypsa.Network) -> dict[str, int]:
    """ A feeeder is defined as subtree rooted in main bus bar (root bus).

    Returns:
        dict[str, int]: Map bus or line to feeder index. """

    feeder_map = {}

    # Copy network to avoid side effects.
    n = n.copy()

    # Remove slack and main bus.
    if len(n.transformers) != 1:
        raise ValueError("Multiple transformers not supported.")

    slack = n.transformers.iloc[0]["bus0"]
    root_bus = n.transformers.iloc[0]["bus1"]

    n.remove("Bus", slack)
    n.remove("Bus", root_bus)
    n.remove("Transformer", n.transformers.index[0])

    # Root segments have to be removed for topology determination.
    # Their feeder is determined later by the second bus.
    root_segments = []

    for line_idx, line in n.lines.iterrows():
        if line["bus0"] == root_bus:
            root_segments.append((line_idx, line["bus1"]))
            n.remove("Line", line_idx)
        if line["bus1"] == root_bus:
            root_segments.append((line_idx, line["bus0"]))
            n.remove("Line", line_idx)

    n.determine_network_topology()

    for bus_idx, bus_data in n.buses.iterrows():
        feeder_map[bus_idx] = int(bus_data["sub_network"])

    for line_idx, line_data in n.lines.iterrows():
        feeder_map[line_idx] = int(line_data["sub_network"])

    for line_idx, bus in root_segments:
        feeder_map[line_idx] = feeder_map[bus]

    return feeder_map