from typing import Tuple, Any, Optional

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

def get_bat(network: pypsa.Network) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # ToDo: Focusing on h0_battery seems over specific.
    bat_params = network.storage_units.query("type == 'h0_battery'")
    check_unique(bat_params["bus"], unit_type='bat')
    bat_params.loc[:, "p_nom"] = pypsa_df_to_grecco_df(bat_params["p_nom"])
    return bat_params, pd.DataFrame(), pd.DataFrame()


def get_hp(network: pypsa.Network) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Baseloads are all loads except heat pumps. """
    # ToDo: Would be better to expect something like carrier == "baseload"
    hp_params = network.loads.query("carrier == 'heat_pump'")
    check_unique(hp_params["bus"], unit_type='heat pump')
    hp_params.loc[:, "p_set"] = pypsa_df_to_grecco_df(hp_params["p_set"])

    return hp_params, pd.DataFrame()


def get_ev(
        network: pypsa.Network,
        sim_config: configs.SimulationConfiguration) -> (
        Tuple[pd.DataFrame, pd.DataFrame]):

    # ToDO: In storages, we find charger p_nom. In storages we should find EV
    #   data, actually.

    charging_processes = pd.read_csv(sim_config.charging_process_path)

    q = "type == 'EMHOMESINGLE' or type == 'EMHOMEMULTI'"
    charger_params = network.storage_units.query(q)
    check_unique(charger_params["bus"], unit_type='ev_charger')

    # Add EV params.
    # p = sim_config.ev_capacity_data_path
    # ev_capacity_data = pd.read_csv(p, index_col=0, date_format=Format().date)
    # ToDo: Capacity could be added by extra file.
    charger_params.loc[:, "charger_id"] = charger_params.index
    charger_params.loc[:, "ev_id"] = charger_params["charger_id"] + "_ev"
    charger_params.loc[:, "capacity"] = 60.0
    charger_params.index = "sys_at_bus_" + charger_params["bus"] + "_ev"

    return charger_params, charging_processes

