import pickle
import warnings

import numpy as np
import pypsa

import sys
from copy import copy


import yaml

import timeit

from grecco_sim.graph import complex_network_analysis
from grecco_sim.graph.utils.config import PushRelabelConfiguration
from grecco_sim.graph.utils.format import Format
from grecco_sim.util import configs
from grecco_sim.simulator import simulation
from sklearn.model_selection import ParameterGrid

import pandas as pd
from warnings import simplefilter

from grecco_sim.util.network_io import determine_feeders

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

data_root = Format().data_root
output_root = Format().output_root / "experiment_5"

scenario = "Opfingen"
load_distribution = "shed_2050"

param_grid = {
    "seed": [3, 5, 17, 257, 65537],
    "date": ["02_27", "08_11", "08_30", "10_11"],
    "kw_per_prosumer": [1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    "feeder_trafo_ratio": [1.0, 2.0, 3.0, 4.0]}

param_grid = list(ParameterGrid(param_grid))


def main(param_index : int, solver: str, topology: str, control: str):

    params = copy(param_grid[param_index])
    meta = copy(params)
    meta.update(**locals())

    dir_name = f"run_{param_index}_{solver}_{topology}"

    for val in params.values():
        dir_name += f"_{val}".replace('.', '')

    result_path = output_root / dir_name / control

    
    if (result_path / f"bottleneck_{seed}.pkl").exists():
        print("Run already done.")
        exit()

    if not result_path.exists():
        result_path.mkdir(parents=True)

    seed = params["seed"]
    feeder_trafo_ratio = params["feeder_trafo_ratio"]
    kw_per_prosumer = params["kw_per_prosumer"]

    month, day = int(params["date"][:2]), int(params["date"][3:])
    meta["month"], meta["day"] = month, day

    grid = f"{topology}_{scenario}_{month}_{day}_{load_distribution}_{seed}"
    grid_path = data_root / "samples" / scenario / load_distribution / grid

    if control == "uncoordinated":
        coordination_mechanism = "uncoordinated"
        temporal_resolution = "None"
    elif control == "central":
        coordination_mechanism = "central"
        temporal_resolution = "None"
    elif control == "feeder":
        coordination_mechanism = "feeder_fee"
        temporal_resolution = "cubic_restricted"
    elif control == "transformer":
        coordination_mechanism = "transformer_fee"
        temporal_resolution = "cubic_restricted"
    else:
        raise NotImplementedError

    optimizer_config = configs.OptimizerConfiguration(solver, horizon=12)
    market_config = configs.MarketConfiguration(max_market_iterations=2)

    dates = pd.read_csv(grid_path / "snapshots.csv", index_col=0)["snapshot"]
    start, end = dates.iloc[0], dates.iloc[-1]
    time_index = pd.date_range(start=start, end=end, freq="15min")
    weather_data_path = data_root / "weather" / f"{time_index[0].year}_dwd.csv"

    meta["start"], meta["end"] = start, end

    network_sneak = pypsa.Network()
    with warnings.catch_warnings(action="ignore"):
        network_sneak.import_from_csv_folder(grid_path, skip_time=True)

    n_prosumer = len(network_sneak.loads.query("carrier == 'baseload'"))
    transformer_lim = (kw_per_prosumer * n_prosumer) / 1000
    feeder_map = determine_feeders(network_sneak)
    n_feeders = len({x for x in feeder_map.values()})
    feeder_lim = (feeder_trafo_ratio * transformer_lim) / n_feeders

    with open(result_path / 'feeder_map.yaml', 'w') as f:
        yaml.dump(feeder_map, f)

    meta["feeder_lim"], meta["transformer_lim"] = feeder_lim, transformer_lim

    simulation_config = configs.SimulationConfiguration(
        feeder_lim=feeder_lim,
        transformer_lim=transformer_lim,
        time_index=time_index,
        coordinator_name=coordination_mechanism,
        temporal_resolution=temporal_resolution,
        sim_tag=f"{dir_name}",
        optimizer_config=optimizer_config,
        grid_data_path=grid_path,
        weather_data_path=weather_data_path,
        market_config=market_config,
        heat_pump_model="continous")

    sim = simulation.Simulation(simulation_config)
    start_time = timeit.default_timer()
    sim.run()
    end_time = timeit.default_timer()
    meta["runtime"] = end_time - start_time

    sim.write(result_path)
    with open(result_path / 'meta.yaml', 'w+') as ff:
        yaml.dump(meta, ff)


    # Deactivated for now.
    if control == "uncoordinated" and False:

        n = sim.grid.n

        trafo_sign = np.sign(n.transformers_t["p0"])

        for seed in [17]:
            pr_config = PushRelabelConfiguration(
                n.snapshots,
                seed=seed,
                max_runtime=25 * 60,
                trafo_sign=trafo_sign)

            cna = complex_network_analysis.ComplexNetworkAnalysis(n, pr_config)
            graph_congestion_table, bottleneck = cna.run()
            graph_congestion_table.to_csv(result_path / f"min_cut_{seed}.csv")

        with open(result_path / f"bottleneck_{seed}.pkl", "wb") as f:
            pickle.dump(bottleneck, f)

if __name__ == "__main__":
    if len (sys.argv) != 4:
        print("Usage: python grecco_sim/scripts/experiments/experiment_4 "
              "<index> <solver> <topology>")
        exit()

    for control in ["feeder", "transformer", "central", "uncoordinated"]:

        index = int(sys.argv[1])
        solver = str(sys.argv[2])
        topology = str(sys.argv[3])

        main(index, solver, topology, control)

