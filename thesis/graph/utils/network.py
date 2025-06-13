from typing import Dict

import numpy as np
import pandas as pd
import pypsa
from pypsa import Network

from thesis.graph.utils.format import Format


def get_p_capacity_w(network: Network) -> Dict[str, float]:
    p_capacity_kw = get_p_capacity_kw(network)
    return {key: 1000 * val for key,val in p_capacity_kw.items()}


def get_p_capacity_mw(network: Network) -> Dict[str, float]:
    """ Capacities (kW) for each line and the (single!) transformer.

    S: Apparent power.
    S_nom: Nominal apparent power (only given for transformers).
    P: Active power.
    P_max: Maximal active power (capacity).
    I: Current.
    I_max: Rated current (maximal current).
    U: Voltage.
    U_nom: Nominal Voltage (typically 0.4 kV).
    lam: Powerfactor (~ 0.95).

    For lines, we have:
    I. S = U * I                                 | Assumption: U_nom ~ U
         = U_nom * I.

    II.            I < I_max                     | * U_nom
       <=> I * U_nom < I_max * U_nom             | Using I.
       <=>         S < I_max * U_nom.

    III.    lam = P / S                          | * S / lam
       <=>    S = P / lam.

    IV.          I < I_max                       | Combining II. & III.
       <=> P / lam < I_max * U_nom               | * lam
       <=>       P < I_max * U_nom * lam.
    --------------------------------------------------------------------------
    Ergo: P_max := I_max * U_nom * lam.


    For transformer, it holds that:
    I.              S < S_max
       <=>    P / lam < S_max                    | III. from above.
       <=>          P < S_max * lam.             | * lam.
    --------------------------------------------------------------------------
    Ergo: P_max := S_max * lam.

    """

    # ToDo: Check if powerflow was conducted beforehand.

    if len(network.transformers) != 1:
        raise ValueError("Networks must have exactly one transformer.")

    # Powerfactor by assumption.
    lam = 0.95
    capacities = {}

    # 1. Line capacities.
    # Iteration over rows is okay, since we expect no more than #buses lines.
    for line in network.lines.itertuples():
        idx_bus0 = network.buses.index.tolist().index(line.bus0)
        idx_bus1 = network.buses.index.tolist().index(line.bus1)

        # Use nominal voltage to estimate actual voltage during grid operation.
        nominal_voltage_bus0 = network.buses.loc[line.bus0]['v_nom']
        nominal_voltage_bus1 = network.buses.loc[line.bus1]['v_nom']

        if nominal_voltage_bus0 != nominal_voltage_bus1:
            raise ValueError(f"Voltage difference between directly connected "
                             f"buses {idx_bus0} and {idx_bus1}.")
        else:
            # Voltage in kV (see https://pypsa.readthedocs.io).
            voltage_kv = nominal_voltage_bus0

        # ToDo: Ensure that line type is specified.
        # Max current in kA (see https://pypsa.readthedocs.io).
        max_current_ka = network.line_types.loc[line.type]["i_nom"]

        # Calculate capacity (in mW = kA * kV).
        capacity_mw = lam * max_current_ka * voltage_kv

        # Store capacity in both directions.
        capacities[line.Index] = capacity_mw

    # 2. Transformer capacity (in mW).
    nominal_apparent_power = network.transformers.iloc[0]["s_nom"]
    transformer_capacity = lam * nominal_apparent_power
    capacities[network.transformers.index[0]] = transformer_capacity

    return capacities

def get_loading_mw(network: Network) -> pd.DataFrame:
    """ Loadings (MegaWatt) for all snapshots."""

    # ToDo: Check if powerflow was conducted beforehand.

    # Get element-wise maximum
    line_loadings = np.maximum(network.lines_t["p0"], network.lines_t["p1"])

    # Convert back to DataFrame (optional)
    line_loadings = pd.DataFrame(
        line_loadings,
        index=network.lines_t["p0"].index,
        columns=network.lines_t["p0"].columns)

    # To compute transformer loading we compare LV and MV load and take the
    # maximal absolute value.
    transformer_p0 = network.transformers_t["p0"].values
    transformer_p1 = network.transformers_t["p1"].values
    transformer_p = np.max(np.abs([transformer_p0, transformer_p1]), axis=0)

    loadings = line_loadings
    loadings[network.transformers.index[0]] = transformer_p

    return loadings

def utilization_ratio(network: Network) -> pd.DataFrame:
    """ Determines the capacity utilization in %."""

    loads_mw = get_loading_mw(network)
    capacity_w = pd.Series(get_p_capacity_w(network))
    capacity_mw = capacity_w.apply(Format().w_to_mw)
    utilization = loads_mw.div(capacity_mw)
    utilization = np.ceil(utilization * 100).astype(int)  # Float -> Percent

    return utilization

def validate(network: Network):
    """ Define a valid network in the sense of this thesis.

    Args:
        network:

    Returns:

    Raises:

    """
    # Assert that every line has line types defined.

    pass

def get_inflexible_loads(network: pypsa.Network) -> pd.DataFrame:
    """ Extract inflexible loads from network. """
    return network.loads[network.loads["carrier"] == "inflex"]

def get_heatpumps(network: Network) -> pd.DataFrame:
    """ Extract heat pumps from network. """
    return network.loads[network.loads["carrier"] == "heat_pump"]
