import sys
from copy import copy


import yaml

import timeit

from fontTools.misc.cython import returns

from grecco_sim.graph.utils.format import Format
from grecco_sim.util import configs
from grecco_sim.simulator import simulation
from sklearn.model_selection import ParameterGrid

import pandas as pd
from warnings import simplefilter


simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

data_root = Format().data_root
output_root = Format().output_root / "experiment_3"

scenario = "Opfingen"
load_distribution = "shed_2050"

param_grid = {
    "seed": [3, 5, 17, 257, 65537],
    "control": ["cubic", "cubic_restricted", "gaussian", "step"],
    "date": ["02_27", "08_11", "08_30", "10_11"]}

param_grid = list(ParameterGrid(param_grid))


def main(param_index : int, solver: str, topology: str):
    params = copy(param_grid[param_index])
    meta = copy(params)

    dir_name = f"run_{param_index}_{solver}_{topology}"

    for val in params.values():
        dir_name += f"_{val}"

    result_path = output_root / dir_name

    seed = params["seed"]

    month, day = int(params["date"][:2]), int(params["date"][3:])
    meta["month"], meta["day"] = month, day

    grid = f"{topology}_{scenario}_{month}_{day}_{load_distribution}_{seed}"
    grid_path = data_root / "samples" / scenario / load_distribution / grid

    control = params["control"]

    if control == "uncoordinated":
        coordination_mechanism = "uncoordinated"
        temporal_resolution = "None"
    elif control == "central":
        coordination_mechanism = "central"
        temporal_resolution = "None"
    else:
        coordination_mechanism = "feeder_fee"
        temporal_resolution = control

    # Horizon = 12 founded on experiment 1.
    optimizer_config = configs.OptimizerConfiguration(solver, horizon=12)

    market_config = configs.MarketConfiguration(max_market_iterations=2)

    dates = pd.read_csv(grid_path / "snapshots.csv", index_col=0)["snapshot"]
    start, end = dates.iloc[0], dates.iloc[-1]
    time_index = pd.date_range(start=start, end=end, freq="15min")
    weather_data_path = data_root / "weather" / f"{time_index[0].year}_dwd.csv"

    meta["start"], meta["end"] = start, end
    feeder_lim, transformer_lim = None, None
    if topology == "simbench-LV-rural1--2":
        feeder_lim, transformer_lim = 0.017, 0.025
    if topology == "simbench-LV-rural2--2":
        feeder_lim, transformer_lim = 0.162, 0.212
    if topology == "simbench-LV-semiurb5--2":
        feeder_lim, transformer_lim = 0.170, 0.223
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

if __name__ == "__main__":
    if len (sys.argv) != 4:
        print("Usage: python grecco_sim/scripts/experiments/experiment_3 "
              "<index> <solver> <topology>")
        exit()

    index = int(sys.argv[1])
    solver = str(sys.argv[2])
    topology = str(sys.argv[3])
    main(index, solver, topology)
