import difflib

from typing import Tuple, Any

import pandas as pd
import pypsa

from grecco_sim.util import configs, data_io


def pypsa_df_to_grecco_df(df: pd.DataFrame) -> pd.DataFrame:

    # Avoid side effects.
    df = df.copy(deep=True)

    def mw_to_kw(x: Any) -> Any:
        return x * 1000

    if type(df.index) == pd.DatetimeIndex:
        df = data_io.set_tz_index_to_utc(df)

    df = mw_to_kw(df)

    return df


def check_unique(data: pd.DataFrame, unit_type: str):
    if not data.is_unique:
        duplicates = data[data.duplicated()]
        msg = f"Multiple units of type {unit_type} in data: {duplicates}."
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

def get_bat(network: pypsa.Network) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # ToDo: Focusing on h0_battery seems over specific.
    bat_params = network.storage_units.query("type == 'h0_battery'")
    check_unique(bat_params["bus"], unit_type='bat')
    bat_params.loc[:, "p_nom"] = pypsa_df_to_grecco_df(bat_params["p_nom"])
    # bat_p_ts = network.storage_units_t["p_set"].loc[:, bat_params.index]
    # bat_p_ts = pypsa_ts_to_grecco_ts(bat_p_ts)
    # bat_soc_ts = network.storage_units_t["state_of_charge"].loc[:,
    #             bat_params.index]
    # bat_soc_ts = pypsa_ts_to_grecco_ts(bat_soc_ts)
    
    return bat_params, pd.DataFrame(), pd.DataFrame()


def get_hp(network: pypsa.Network) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Baseloads are all loads except heat pumps. """
    # ToDo: Would be better to expect something like carrier == "baseload"
    hp_params = network.loads.query("carrier == 'heat_pump'")
    check_unique(hp_params["bus"], unit_type='heat pump')
    hp_params.loc[:, "p_set"] = pypsa_df_to_grecco_df(hp_params["p_set"])
    # hp_p_set = network.loads_t["p_set"].loc[:, hp_params.index]
    # hp_p_set = pypsa_ts_to_grecco_ts(hp_p_set)

    return hp_params, pd.DataFrame()


def get_ev(
        network: pypsa.Network,
        sim_config: configs.SimulationConfiguration) -> (
        Tuple)[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    ev_params = network.storage_units.query("type != 'h0_battery'")
    check_unique(ev_params["bus"], unit_type='ev')
    check_unique(ev_params["Name"], unit_type='ev')

    ev_params.loc[:]["capacity"] = 60.  # Default value

    # EV capacity data can be imported from an extra file.
    if sim_config.ev_capacity_data_path:
        p = sim_config.ev_capacity_data_path
        date_format = "%Y-%m-%d %H:%M:%S"
        ev_capacity_data = pd.read_csv(p, index_col=0, date_format=date_format)

        for ev in ev_params.index:
            try:
                capacity = ev_capacity_data.loc[ev, "capacity"]
            # Fallback with close matches.
            except KeyError:
                match = difflib.get_close_matches(
                    word=ev,
                    possibilities=ev_capacity_data.index,
                    n=1)

                capacity = ev_capacity_data.loc[match, "capacity_kWh"]

            ev_params.loc[ev, "capacity"] = capacity.values

    ev_bat_ts = network.storage_units_t.loc[:, ev_params["Name"]]

    plugged_in = network.storage_units_t.plugged_in[ev_params["Name"]]
    plugged_in = plugged_in.fillna(0)

    soc_departure_max_percent = network.storage_units_t.soc_departure_max_percent
    soc_departure_min_percent = network.soc_departure_min_percent

    ev_data_in = pd.concat({'cp': plugged_in,
                            'soc_max_percent': soc_departure_max_percent,
                            'soc_min_percent': soc_departure_min_percent},
                            axis=1)

    # Swap MultiIndex levels
    ev_data_in.columns = ev_data_in.columns.swaplevel(0, 1)
    # Sort index to group 'A' and 'B' together
    ev_charging_ts = ev_data_in.sort_index(axis=1)
    if ev_charging_ts.index.tz is None:
        ev_charging_ts = ev_charging_ts.tz_localize("UTC")

    # interpolate soc data blockiwse while plugged in and get parking duration
    # Compute and add "soc" and "until_departure" for each sys_id
    # Store computed charging data in a list
    charging_data_list = []

    # Iterate over each system ID (level 0 column)
    for sys_id in ev_charging_ts.columns.levels[0]:
        ev_charging_ts = data_io.get_charging_data(
            ev_charging_ts[sys_id],
            sim_config.dt_h)[["initial_soc", "target_soc", "until_departure"]]
        ev_charging_ts.columns = pd.MultiIndex.from_product(
            [[sys_id], ev_charging_ts.columns])  # Ensure correct MultiIndex
        charging_data_list.append(ev_charging_ts)

    ev_charging_ts = pd.concat([ev_charging_ts] + charging_data_list, axis=1)

    ev_charging_ts = pd.MultiIndex.from_tuples(
        [(left, f"{left}_{right}") for left, right in ev_charging_ts.columns])


    raise NotImplementedError("Please check implementation.")

    return ev_params, ev_bat_ts, ev_charging_ts

