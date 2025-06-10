import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
from termcolor import colored

import logging

from thesis.optimization.optimal_flow.graph_builder import \
    GraphBuilder
from thesis.optimization.optimal_flow.push_relabel import \
    PushRelabel
from thesis.complex_network.utils.config import PushRelabelConfiguration
from thesis.complex_network.utils.network import get_p_capacity, get_loading_mw, \
    utilization_ratio
from thesis.complex_network.utils.utils import now

# Mute PyPSA Info:
logging.getLogger("pypsa").setLevel(logging.WARNING)

from grecco_sim.simulator import simulation_setup
from grecco_sim.util import type_defs


def main():
    # Define configuration.
    configuration = PushRelabelConfiguration()

    # Import data, check congestion status.
    data_root = Path("/home/carl-wanninger/data")
    grid_dir = (data_root / "pypsa" / "trafo_violation" / "12-02-2025" /
                "10_grid_19")
    weather_data = data_root / "weather" / "test" / "pvgis_2016_00.csv"
    output_dir = data_root / "push_relabel" / now()

    network = pypsa.Network()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        network.import_from_csv_folder(grid_dir)

    # Build graph.
    graph_builder = GraphBuilder(configuration)
    graph = graph_builder.build(network)

    # Optimize graph.
    push_relabel = PushRelabel(configuration)
    max_flow = push_relabel.calculate_maximal_flow(graph)

    # Check congestion status.

    # Return results.

    # import warnings; warnings.simplefilter("error")  # Convert warnings to
    # exceptions



    test_00 = {
        "name": "test_00",
        "n_agents": 4,
        "grid_data_path": grid_dir,
        "weather_data_path": weather_data,
        "hp": True
    }



    test_grid.transformer_types.at["0.25 MVA 20/0.4 kV", "s_nom"] = .025

    loads = utilization_ratio(test_grid)
    capacity = get_p_capacity(test_grid)
    utilization = utilization_ratio(test_grid)

    if (utilization >= 100).any(axis=None):
        if (utilization["MV/LV Transformer"] >= 100).any(axis=None):
            print(colored("Transformer limit violation.", "red"))
            print(f"Max. {utilization['MV/LV Transformer'].max()} % "
                  f"Min. {utilization['MV/LV Transformer'].min()} %")
        else:
            print(colored("Line limit violation.", "red"))
    else:
        print(colored("No capacity violation.", "green"))

    coord_types = ["central_optimization", "second_order", "admm",
                   "plain_grid_fee", "local_self_suff"]

    # start_date = datetime(2023, 12, 1, 0, 0, 0, tzinfo=pytz.utc)
    start_date = test_grid.snapshots[0].tz_localize("utc")

    run_parameters = simulation_setup.RunParameters(
        sim_horizon=60,
        start_time=start_date,
        max_market_iterations=4,
        coordination_mechanism=coord_types[0],
        scenario=test_00,
        sim_tag=coord_types[0],
        # inspection=[36],
        use_prev_signals=False,
        plot=True,
        show=True,
        profile_run=True,
        output_file_dir=output_dir
    )

    opt_pars = type_defs.OptParameters(
        rho=100.0,
        mu=5000.0,
        horizon=20,
        alpha=0.05,
        solver_name="gurobi",
        fc_type="perfect",
    )

    grid_pars = type_defs.GridDescription(
        p_lim=capacity["MV/LV Transformer"],  # Algorithm parameters
    )

    cap_0 = get_p_capacity(test_grid)
    load_0 = get_loading_mw(test_grid)

    print("Maximal utilization")
    print(utilization_ratio(test_grid)['MV/LV Transformer'].max())
    print("Total utilization")
    print(utilization_ratio(test_grid)['MV/LV Transformer'].sum())

    simulator = simulation_setup.SimulationSetup(run_parameters)
    _, result = simulator.run_sim(opt_pars, grid_pars)

    # Set loads according to optimization.
    optimized_loads_p_t = {}
    optimized_loads_q_t = {}
    for key in result.agents_ts.keys():
        bus = key.split("_")[1]
        for col in test_grid.loads_t["p"].columns:
            if bus in col and "heatpump" in col.lower():
                if "hp_p_in" in result.agents_ts[key].columns:
                    optimized_loads_p_t[col] = result.agents_ts[key][
                        "hp_p_in"] / 1000
                    optimized_loads_q_t[col] = result.agents_ts[key][
                        "hp_q_hp"] / 1000
                else:
                    print(f"Heatpump found in {col}, but hp_p_in not found "
                          f"in {result.agents_ts[key].keys()}")
            if bus in col and "Inflex" in col:
                optimized_loads_p_t[col] = result.agents_ts[key]["grid"] / 1000
                optimized_loads_q_t[col] = test_grid.loads_t["q_set"][col]

    # Remove tz awareness.
    for timeseries in optimized_loads_p_t.values():
        timeseries.index = timeseries.index.tz_localize(None)

    for timeseries in optimized_loads_q_t.values():
        timeseries.index = timeseries.index.tz_localize(None)


    heatpumps = [hp for hp in test_grid.loads_t["p_set"].columns if "heatpump"
                 in hp.lower()]
    for hp in heatpumps:
        print(test_grid.loads_t["p_set"].sum(axis=0).loc[hp],
              pd.DataFrame(optimized_loads_p_t).sum(axis=0).loc[hp])

    print("Total kV in heatpumps: before, after, before/after")
    print(test_grid.loads_t["p_set"].to_numpy().sum(), pd.DataFrame(
            optimized_loads_p_t).to_numpy().sum())

    # ToDo: Find out which results from GrECCo are important.

    print(np.mean(test_grid.loads_t["p_set"].to_numpy()),
          np.min(test_grid.loads_t["p_set"].to_numpy()),
          np.max(test_grid.loads_t["p_set"].to_numpy()),
          test_grid.loads_t["p_set"].to_numpy().sum(),
          test_grid.transformers_t["p1"].to_numpy().sum())

    test_grid.loads_t["p_set"] = pd.DataFrame(optimized_loads_p_t)
    test_grid.loads_t["q_set"] = pd.DataFrame(optimized_loads_q_t)

    # Conduct Powerflow.
    test_grid.lpf()
    test_grid.pf(use_seed=True)

    print(np.mean(test_grid.loads_t["p_set"].to_numpy()),
          np.min(test_grid.loads_t["p_set"].to_numpy()),
          np.max(test_grid.loads_t["p_set"].to_numpy()),
          test_grid.loads_t["p_set"].to_numpy().sum(),
          test_grid.transformers_t["p1"].to_numpy().sum())

    cap_1 = get_p_capacity(test_grid)
    load_1 = get_loading_mw(test_grid)

    # Check congestion.
    print("Maximal utilization")
    print(utilization_ratio(test_grid)['MV/LV Transformer'].max())
    print("Total utilization")
    print(utilization_ratio(test_grid)['MV/LV Transformer'].sum())

    for unit in cap_0.keys():
        print()
        print(unit)
        print(cap_0[unit], cap_1[unit])
        print(load_0[unit].sum(), load_1[unit].sum())

if __name__ == "__main__":
    main()
